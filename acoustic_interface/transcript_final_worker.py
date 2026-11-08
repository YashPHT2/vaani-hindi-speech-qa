"""Frozen standalone Whisper-to-Qwen evaluation with one attempt per view."""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import unicodedata
import wave

RUN_ID = "transcript-only-final100-v1"
ASR = ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d")
LLM = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
SHARD_SHA = "8a0be26c5584c1478233a6f47fb90275c713f72565327ea1da76c9f1cd5dd38b"
VIEW_SPECS_SHA = "af2319c2ac65ca6aea4f05bbbce5015e473126a77701bf5311d29b468376982e"
FROZEN_HASHES = {
    "historical_manifest": "4872a984b9e19d317930263a2f1c1ce2c5c4d3c416e3284cfce99e0ec4d8db52",
    "selection": "890ad93143c644a8dde28d3c30c5c77319d370a4ec21b602c44a544c1a5f70e2",
    "qa": "8c3ea7780a8ea5a594dd09b8b8f6b94133a41a6497afe240ee024f87a6fd4937",
}
CONDITIONS = ("clean", "noise_10db")
RESOURCE_LIMITS = {"gpu": "A10", "gpu_count": 1, "cpu": 2, "memory_gib": 16,
    "function_s": 7200, "startup_s": 300, "queue_s": 120, "idle_s": 2, "retries": 0}
SYSTEM_PROMPT = "केवल दिए गए साक्ष्य से छोटा उत्तर दें। जानकारी न हो तो अज्ञात लिखें।"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def normalized_transcript(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def validate_config(config):
    assert config["run_id"] == RUN_ID
    assert tuple(config["asr"]) == ASR and tuple(config["llm"]) == LLM
    assert config["seed"] == 17 and config["memory_slots"] == 0
    assert config["asr_max_new_tokens"] == 224 and config["qa_max_new_tokens"] == 48
    assert config["resource_limits"] == RESOURCE_LIMITS
    assert config["work_timeout_s"] == 7000 and config["reservation_usd"] == 4
    assert math.isclose(config["worst_case_cost_usd"], 7502 * .00036772 + .75, abs_tol=1e-9)
    assert config["input_sha256"] == FROZEN_HASHES
    assert config["publisher_shard_sha256"] == SHARD_SHA
    assert config["system_prompt"] == SYSTEM_PROMPT
    assert config["original_view_specs_sha256"] == VIEW_SPECS_SHA


def load_annotations(historical_bytes, selection_bytes, qa_bytes):
    for name, payload in (("historical_manifest", historical_bytes), ("selection", selection_bytes), ("qa", qa_bytes)):
        assert sha(payload) == FROZEN_HASHES[name], "Frozen annotation bytes changed"
    historical = [json.loads(line) for line in historical_bytes.decode("utf-8").splitlines()]
    selection = json.loads(selection_bytes)["records"]
    questions = [json.loads(line) for line in qa_bytes.decode("utf-8").splitlines()]
    assert len(historical) == 224 and len({r["source_id"] for r in historical}) == 112
    assert len(selection) == len(questions) == 100
    source_rows = {int(r["source_id"]): r for r in selection}
    question_rows = {int(r["source_id"]): r for r in questions}
    assert len(source_rows) == len(question_rows) == 100 and set(source_rows) == set(question_rows)
    old_ids = {int(r["source_id"].rsplit("-", 1)[-1]) for r in historical}
    assert not old_ids & set(source_rows)
    historical_text = {normalized_transcript(r["transcript"]) for r in historical}
    selected_text = [normalized_transcript(r["transcript"]) for r in selection]
    assert len(set(selected_text)) == 100 and not historical_text & set(selected_text)
    canonical_hashes = [r["canonical_source_audio_sha256"] for r in selection]
    assert len(set(canonical_hashes)) == 100
    for source, annotation in question_rows.items():
        assert annotation["reference_transcript"] == source_rows[source]["transcript"]
        assert annotation["answerable"] == source_rows[source]["answerable"]
        assert annotation["question"].strip() and annotation["answers"]
        assert all(alias.strip() for alias in annotation["answers"])
    assert sum(r["answerable"] for r in questions) == 80
    return historical, selection, question_rows


def transformed_wave(shard_row, condition, expected_canonical):
    """Exact shared deterministic v2 clean/noise transform, without model work."""
    import numpy as np
    from scipy.io import wavfile

    assert condition in CONDITIONS
    rate, samples = wavfile.read(io.BytesIO(shard_row["audio"]["bytes"]))
    assert rate == 16000 and samples.ndim == 1 and 0 < len(samples) <= 480000
    canonical = sha(str(rate).encode() + b"\0" + str(samples.dtype).encode() + b"\0" + samples.tobytes(order="C"))
    assert canonical == expected_canonical and np.isfinite(samples).all()
    if np.issubdtype(samples.dtype, np.integer):
        clean = samples.astype(np.float64) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    else:
        clean = samples.astype(np.float64)
    seed = int(sha(("noise:" + str(shard_row["id"])).encode())[:16], 16)
    noise = np.random.default_rng(seed).standard_normal(len(clean))
    energy = np.mean(clean ** 2)
    assert energy > 0
    noise *= np.sqrt(energy / (10 * np.mean(noise ** 2)))
    noisy = clean + noise
    gain = min(1., .98 / max(np.max(np.abs(clean)), np.max(np.abs(noisy))))
    signal = clean if condition == "clean" else noisy
    pcm = np.rint(np.clip(signal * gain, -1, 1) * 32767).astype("<i2")
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as target:
        target.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        target.writeframes(pcm.tobytes())
    return encoded.getvalue(), {"audio_sha256": sha(encoded.getvalue()), "seconds_audio": len(pcm) / 16000,
        "shared_gain": gain, "noise_seed": seed, "sample_count": len(pcm), "canonical_source_audio_sha256": canonical}


def validate_view_specs(spec_bytes, views):
    assert sha(spec_bytes) == VIEW_SPECS_SHA
    specs = json.loads(spec_bytes)
    assert len(specs) == len(views) == 200
    indexed = {(int(row["source_id"]), row["condition"]): row for row in specs}
    assert len(indexed) == 200
    assert set(indexed) == {(row["source_id"], row["condition"]) for row in views}
    for view in views:
        original = indexed[(view["source_id"], view["condition"])]
        for field in ("canonical_source_audio_sha256", "original_audio_path", "question_sha256"):
            assert original[field] == view[field]
        assert original["waveform_transform"] == "shared v2 gain; 16kHz int16; noise seed SHA256 noise:decimal-publisher-id first16hex; 10dB white-noise; clean and noisy share gain"
        assert original["canonical_hash_definition"] == "original publisher PCM decoded by scipy.wavfile; rate+NUL+dtype+NUL+PCM C bytes"


def build_view_manifest(table, selection, question_rows):
    """Validate all source bytes and materialize the fixed 200 waveforms."""
    publishers = {}
    for index, row in enumerate(table):
        publishers.setdefault((int(row["id"]), Path(row["audio"]["path"]).name), index)
    views, waveforms, references = [], {}, []
    for record in selection:
        source = int(record["source_id"])
        index = publishers[(source, record["original_audio_path"])]
        row = table[index]
        annotation = question_rows[source]
        for condition in CONDITIONS:
            identity = f"fresh-hi-{source}-{condition}"
            payload, info = transformed_wave(row, condition, record["canonical_source_audio_sha256"])
            public = {"id": identity, "source_id": source, "publisher_id": source,
                "condition": condition, "question": annotation["question"],
                "question_sha256": sha(annotation["question"].encode("utf-8")),
                "source_row_index": index, "original_audio_path": record["original_audio_path"], **info}
            views.append(public)
            waveforms[identity] = payload
            references.append({"id": identity, "source_id": public["source_id"], "condition": condition,
                "split": "test", "transcript": annotation["reference_transcript"], "question": annotation["question"],
                "answers": annotation["answers"], "answerable": annotation["answerable"], **info})
    assert len(views) == len(waveforms) == len(references) == 200
    assert len({view["audio_sha256"] for view in views}) == 200
    return views, waveforms, references


def pack_k0(lm, tokenizer, transcript, question):
    """The established K0 prompt, with zero acoustic positions and no answer."""
    import torch
    from connector import pack_prompt
    device = lm.get_input_embeddings().weight.device
    empty = torch.empty(0, lm.config.hidden_size, device=device)
    prompt = pack_prompt(lm, tokenizer, transcript, question, empty, answer=None)
    assert prompt.pop("labels") is None
    prompt.pop("position_ids")
    return prompt


def run(historical_bytes, selection_bytes, qa_bytes, spec_bytes, config, commit):
    import platform
    import numpy as np
    import pyarrow
    import pyarrow.parquet as pq
    import scipy
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor

    assert torch.cuda.is_available()
    validate_config(config)
    started = time.monotonic()
    deadline = started + config["work_timeout_s"]
    out = Path("/cache/runs") / RUN_ID
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_bytes(json_bytes(config))
    historical, selection, questions = load_annotations(historical_bytes, selection_bytes, qa_bytes)
    shard = Path("/cache/fleurs-hi-test.parquet")
    with shard.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == SHARD_SHA
    table = pq.read_table(shard).to_pylist()
    views, waveforms, references = build_view_manifest(table, selection, questions)
    validate_view_specs(spec_bytes, views)
    assert sha(json_bytes(views)) == config["view_manifest_sha256"]
    assert sha(json_bytes(references)) == config["reference_manifest_sha256"]
    assert max(row["seconds_audio"] for row in views) == config["max_audio_duration_s"]
    assert config["sources_over_30_seconds"] == 0
    historical_audio = {row["audio_sha256"] for row in historical}
    assert not historical_audio & {view["audio_sha256"] for view in views}
    (out / "view-manifest.json").write_bytes(json_bytes(views))
    (out / "reference-manifest.json").write_bytes(json_bytes(references))
    commit()
    del table, questions, selection, historical

    torch.set_num_threads(2)
    torch.manual_seed(17)
    load_start = time.monotonic()
    processor = WhisperProcessor.from_pretrained(ASR[0], revision=ASR[1], local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language="hi", task="transcribe", predict_timestamps=False)
    asr = WhisperForConditionalGeneration.from_pretrained(ASR[0], revision=ASR[1], torch_dtype=torch.float16,
        attn_implementation="eager", use_safetensors=True, local_files_only=True).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(LLM[0], revision=LLM[1], local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(LLM[0], revision=LLM[1], torch_dtype=torch.float16,
        attn_implementation="sdpa", use_safetensors=True, local_files_only=True).cuda().eval().requires_grad_(False)
    torch.cuda.synchronize()
    loading_s = time.monotonic() - load_start
    assert asr.config.d_model == 768 and lm.config.hidden_size == 1536
    versions = [[parameter._version for parameter in model.parameters()] for model in (asr, lm)]
    rows, fatal_error = [], None
    predictions_path = out / "predictions.jsonl"
    for view in views:
        base = {**view, "mode": "baseline", "hypothesis": "", "answer": "", "finish": "failed",
            "failed": False, "asr_failed": True, "error": None, "asr_s": 0., "llm_s": 0., "total_s": 0.,
            "waveform_decode_s": 0., "asr_generated_tokens": 0, "generated_tokens": 0,
            "asr_capped": False, "capped": False, "asr_finish": "failed", "empty": True,
            "asr_token_ids": [], "generated_token_ids": [], "failure_stage": None,
            "prompt_sha256": None, "prompt_hash_definition": "SHA256 of contiguous fp16 inputs_embeds bytes, copied after synchronized online interval",
            "peak_cuda_bytes": torch.cuda.memory_allocated()}
        if fatal_error or time.monotonic() >= deadline:
            base.update(failed=True, error=fatal_error or "internal_deadline_before_view", failure_stage="not_executed")
        else:
            stage = "waveform_decode"
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = None
            decode_tick = time.monotonic()
            asr_tick = llm_tick = None
            prompt = None
            try:
                with torch.inference_mode():
                    raw = waveforms[view["id"]]
                    assert sha(raw) == view["audio_sha256"]
                    with wave.open(io.BytesIO(raw), "rb") as source:
                        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
                        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
                    base["waveform_decode_s"] = time.monotonic() - decode_tick
                    stage = "asr"
                    torch.cuda.synchronize()
                    tick = asr_tick = time.monotonic()
                    batch = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                    features = batch.input_features.cuda().half()
                    attention = batch.attention_mask.cuda()
                    asr_ids = asr.generate(input_features=features, attention_mask=attention, language="hi", task="transcribe",
                        do_sample=False, num_beams=1, max_new_tokens=224, return_timestamps=False,
                        return_token_timestamps=False, output_scores=False, output_logits=False, use_cache=True)[0]
                    tokens = asr_ids.tolist()
                    hypothesis = processor.tokenizer.decode(asr_ids, skip_special_tokens=True).strip()
                    prefix_length = len(processor.tokenizer.prefix_tokens)
                    asr_new = max(0, len(tokens) - prefix_length)
                    asr_eos = bool(tokens) and tokens[-1] == processor.tokenizer.eos_token_id
                    torch.cuda.synchronize()
                    base.update(hypothesis=hypothesis, asr_failed=False, asr_s=time.monotonic() - asr_tick, asr_token_ids=tokens,
                        asr_generated_tokens=asr_new, asr_eos=asr_eos, asr_capped=asr_new >= 224 and not asr_eos,
                        asr_finish="length" if asr_new >= 224 and not asr_eos else "stop")
                    stage = "qa"
                    torch.cuda.synchronize()
                    llm_tick = time.monotonic()
                    prompt = pack_k0(lm, tokenizer, hypothesis, view["question"])
                    prompt_tokens = int(prompt["inputs_embeds"].shape[1])
                    answer_ids = lm.generate(**prompt, max_new_tokens=48, do_sample=False, num_beams=1, use_cache=True,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)[0]
                    answer_tokens = answer_ids.tolist()
                    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                    eos = bool(answer_tokens) and answer_tokens[-1] == tokenizer.eos_token_id
                    torch.cuda.synchronize()
                    base.update(answer=answer, llm_s=time.monotonic() - llm_tick, prompt_tokens=prompt_tokens,
                        generated_tokens=len(answer_tokens), generated_token_ids=answer_tokens, eos=eos,
                        capped=len(answer_tokens) == 48 and not eos, empty=not bool(answer),
                        finish="length" if len(answer_tokens) == 48 and not eos else "stop")
            except Exception as error:
                now = time.monotonic()
                base.update(failed=True, error=type(error).__name__ + ": " + str(error), failure_stage=stage, answer="", empty=True)
                if stage == "asr" and asr_tick is not None:
                    base["asr_s"] = now - asr_tick
                if stage == "qa" and llm_tick is not None:
                    base["llm_s"] = now - llm_tick
                if isinstance(error, torch.cuda.OutOfMemoryError) or "CUDA" in str(error):
                    fatal_error = "terminal_gpu_error_after_" + view["id"]
            finally:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    fatal_error = fatal_error or "terminal_gpu_synchronization_error"
                base["total_s"] = time.monotonic() - tick if tick is not None else 0.
                base["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
            if prompt is not None:
                try:
                    prompt_bytes = prompt["inputs_embeds"].detach().cpu().contiguous().numpy().tobytes()
                    base["prompt_sha256"] = sha(prompt_bytes)
                except Exception as error:
                    base["prompt_hash_error"] = type(error).__name__ + ": " + str(error)
                    fatal_error = fatal_error or "terminal_prompt_hash_error"
                    base.update(failed=True, error=base["error"] or base["prompt_hash_error"],
                        failure_stage=base["failure_stage"] or "post_timing_prompt_hash", answer="", empty=True, finish="failed")
        assert all(math.isfinite(base[key]) and base[key] >= 0 for key in ("asr_s", "llm_s", "total_s", "waveform_decode_s"))
        rows.append(base)
        with predictions_path.open("ab") as stream:
            stream.write(json.dumps(base, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
        (out / "progress.json").write_bytes(json_bytes({"completed_view_records": len(rows),
            "successful_views": sum(not r["failed"] for r in rows), "last_id": view["id"]}))
        commit()
        print(json.dumps({"phase": "transcript_final", "views_recorded": len(rows), "id": view["id"], "failed": base["failed"]}), flush=True)
    assert len(rows) == 200 and len({r["id"] for r in rows}) == 200
    for model, initial in zip((asr, lm), versions):
        assert not model.training and [p._version for p in model.parameters()] == initial
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    failed = sum(row["failed"] for row in rows)
    result = {"status": "transcript_final_complete" if not failed else "transcript_final_partial",
        "config": config, "eval_rows": rows, "references": references, "sources": 100, "views": 200,
        "failed_views": failed, "successful_views": 200 - failed, "model_loading_s": loading_s,
        "input_validation_s": load_start - started, "elapsed_s": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(), "peak_cuda_bytes": max(r["peak_cuda_bytes"] for r in rows),
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "pyarrow": pyarrow.__version__},
        "weight_invariance": {"frozen_eval_models": True, "parameter_versions_unchanged": True, "gradients_none": True},
        "timing_scope": "Prepared decoded waveform availability through ASR preprocessing, recognition, text decoding, QA packing, generation and answer decoding; synchronized GPU boundaries; WAV decode, input transforms, model loading and artifact writes excluded",
        "warmup_views": 0, "first_view_includes_online_kernel_initialization": True}
    result_path = out / "result.json"
    result_path.write_bytes(json_bytes(result))
    references_path = out / "references.jsonl"
    references_path.write_bytes(b"".join(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n" for row in references))
    (out / "artifact-manifest.json").write_bytes(json_bytes({"run_id": RUN_ID,
        "config_sha256": sha(json_bytes(config)), "sha256": {"result.json": sha(result_path.read_bytes()),
            "predictions.jsonl": sha(predictions_path.read_bytes()), "references.jsonl": sha(references_path.read_bytes())}}))
    commit()
    return result
