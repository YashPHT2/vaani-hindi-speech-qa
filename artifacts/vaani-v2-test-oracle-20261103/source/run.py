"""Offline, single-utterance ASR -> text LLM research runner. See README.md."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor

ASR = ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d")
LLM = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SYSTEM = "Answer the user's spoken question briefly in the same language. If a passage is supplied, answer only from that passage. If the supplied information is insufficient, say UNKNOWN. Give only the answer."


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_wave(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) != (16000, 2, 1):
            raise ValueError(f"Expected mono 16 kHz PCM16 WAV: {path}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if not len(samples):
        raise ValueError("Empty audio")
    return samples


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("manifest", type=Path)
    cli.add_argument("output", type=Path, help="New directory, one run only")
    cli.add_argument("--split", choices=["dev", "test"], default="dev")
    cli.add_argument("--mode", choices=["auto", "hint", "oracle"], default="auto")
    cli.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    cli.add_argument("--limit", type=int)
    cfg = cli.parse_args()
    if cfg.limit is not None and cfg.limit < 1:
        cli.error("limit must be positive")
    rows = [json.loads(s) for s in cfg.manifest.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    rows = [r for r in rows if r["split"] == cfg.split][:cfg.limit]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty split or duplicate row IDs")
    cfg.output.mkdir(parents=True, exist_ok=False)
    clock = time.perf_counter
    torch.manual_seed(17)
    cuda = cfg.device == "cuda"
    if cuda:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 12 * 2**30 / total_memory))
    dtype = torch.float16 if cuda else torch.float32
    def sync():
        if cuda:
            torch.cuda.synchronize()
    meta = {"asr": ASR, "llm": LLM, "mode": cfg.mode, "split": cfg.split, "device": cfg.device,
            "dtype": str(dtype), "seed": 17, "system_prompt": SYSTEM, "llm_token_cap": 64,
            "manifest_sha256": digest(cfg.manifest), "runner_sha256": digest(Path(__file__)),
            "python": platform.python_version(), "packages": {n: importlib.metadata.version(n) for n in ["torch","transformers","numpy"]},
            "device_name": torch.cuda.get_device_name() if cuda else platform.processor(), "cuda": torch.version.cuda}
    meta_path = cfg.output / "config.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    start = clock()
    if cfg.mode != "oracle":
        proc = WhisperProcessor.from_pretrained(ASR[0], revision=ASR[1])
        asr = WhisperForConditionalGeneration.from_pretrained(ASR[0], revision=ASR[1], torch_dtype=dtype).to(cfg.device).eval()
    tok = AutoTokenizer.from_pretrained(LLM[0], revision=LLM[1])
    llm = AutoModelForCausalLM.from_pretrained(LLM[0], revision=LLM[1], torch_dtype=dtype).to(cfg.device).eval()
    sync()
    meta["load_seconds"] = clock() - start
    def infer(row):
        audio = (cfg.manifest.parent / row["audio"]).resolve()
        samples = read_wave(audio)
        sync()
        t0 = clock()
        chunks = np.array_split(samples, (len(samples) + 479999) // 480000)
        if cfg.mode == "oracle":
            hyp = row["transcript"]
        else:
            parts = []
            for chunk in chunks:
                features = proc(chunk, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                out = asr.generate(features.input_features.to(cfg.device, dtype=dtype),
                                   attention_mask=features.attention_mask.to(cfg.device), task="transcribe",
                                   language=row["language"] if cfg.mode == "hint" else None, do_sample=False, num_beams=1)
                parts.append(proc.batch_decode(out, skip_special_tokens=True)[0].strip())
            hyp = " ".join(parts).strip()
        sync()
        t1 = clock()
        question = hyp
        if row.get("question") and row.get("context"):
            raise ValueError("Choose either spoken evidence + written question or written evidence + spoken question")
        if row.get("question"):
            question = "Spoken passage:\n" + hyp + "\n\nQuestion:\n" + row["question"]
        elif row.get("context"):
            question = "Passage:\n" + row["context"] + "\n\nSpoken question:\n" + hyp
        system = ("Answer the written question in its language, using only the spoken passage. "
                  "Use the specified abstention word for missing information. Give only the answer."
                  if row.get("question") else SYSTEM)
        system += ("\n" + row["task_instruction"] if row.get("task_instruction") else "")
        messages = [{"role":"system","content":system},{"role":"user","content":question}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tok(prompt, add_special_tokens=False, return_tensors="pt").to(cfg.device)
        result = llm.generate(**tokens, max_new_tokens=64, do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id)
        generated = result[0, tokens.input_ids.shape[1]:].tolist()
        answer = tok.decode(generated, skip_special_tokens=True).strip()
        sync()
        t2 = clock()
        return {"id":row["id"], "mode":cfg.mode, "audio_sha256":digest(audio), "condition":row["condition"],
                "hypothesis":hyp, "asr_chunks":len(chunks) if cfg.mode!="oracle" else 0,
                "messages":messages, "answer":answer, "tokens":len(generated),
                "finish":"length" if len(generated)==64 and generated[-1]!=tok.eos_token_id else "stop",
                "seconds_audio":len(samples)/16000,"asr_s":t1-t0 if cfg.mode!="oracle" else None,"llm_s":t2-t1,"total_s":t2-t0}
    with torch.inference_mode(), (cfg.output/"predictions.jsonl").open("x",encoding="utf-8", buffering=1) as f:
        # One repeated unscored forward pass warms kernels. No answers are seen.
        infer(rows[0])
        for index,row in enumerate(rows,1):
            print(json.dumps(infer(row),ensure_ascii=False),file=f,flush=True)
            if index==1 or index%10==0 or index==len(rows):
                print(json.dumps({'progress':index,'total':len(rows),'mode':cfg.mode,'split':cfg.split}),flush=True)
    meta["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if cuda else None
    meta["completed_rows"] = len(rows)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

