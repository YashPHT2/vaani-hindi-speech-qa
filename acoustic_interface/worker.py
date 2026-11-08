"""GPU-only real-model smoke and supervised adapter runner."""
import hashlib
import json
import os
from pathlib import Path
import time

ASR = ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d")
LLM = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")


def models():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
    assert torch.cuda.is_available(), "CPU execution is not authorized"
    torch.set_num_threads(2)
    torch.manual_seed(17)
    proc = WhisperProcessor.from_pretrained(ASR[0], revision=ASR[1])
    asr = WhisperForConditionalGeneration.from_pretrained(ASR[0], revision=ASR[1],
        torch_dtype=torch.float16, attn_implementation="eager", use_safetensors=True).cuda().eval()
    tok = AutoTokenizer.from_pretrained(LLM[0], revision=LLM[1])
    lm = AutoModelForCausalLM.from_pretrained(LLM[0], revision=LLM[1],
        torch_dtype=torch.float16, attn_implementation="sdpa", use_safetensors=True).cuda().eval()
    for model in (asr, lm):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    assert asr.config.d_model == 768 and lm.config.hidden_size == 1536
    return proc, asr, tok, lm


def smoke(audio_path, row):
    import soundfile as sf
    import torch
    from connector import AcousticMemory, pack_prompt
    from features import extract
    started = time.monotonic()
    proc, asr, tok, lm = models()
    loaded = time.monotonic()
    wav, rate = sf.read(audio_path, dtype="float32")
    evidence = extract(asr, proc, wav, rate)
    extracted = time.monotonic()
    connector = AcousticMemory().cuda()
    inputs = {k: v.cuda() for k, v in evidence.items() if isinstance(v, torch.Tensor)}
    qids = tok(row["question"], return_tensors="pt").input_ids.cuda()
    inputs.update(question=lm.get_input_embeddings()(qids).float(), question_mask=torch.ones_like(qids).bool())
    optimizer = torch.optim.AdamW(connector.parameters(), lr=.001, weight_decay=.01)
    before = {n: p.detach().clone() for n, p in connector.named_parameters()}
    memory, indices, gates = connector(**inputs)
    packed = pack_prompt(lm, tok, evidence["transcript"], row["question"], memory[0], row["answers"][0])
    output = lm(**packed, use_cache=False)
    output.loss.backward()
    gradients = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in connector.named_parameters()}
    assert gradients["score.2.weight"] > 0, "Task gradient must reach gate"
    assert gradients["project.weight"] > 0, "Task gradient must reach acoustic projection"
    assert all(p.grad is None for p in lm.parameters())
    torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.)
    optimizer.step()
    updates = {n: float((p.detach() - before[n]).norm()) for n, p in connector.named_parameters()}
    assert updates["project.weight"] > 0
    end = time.monotonic()
    result = {"status": "real_gpu_smoke_pass", "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "asr": ASR, "llm": LLM, "source_id": row["source_id"], "id": row["id"],
        "audio_sha256": hashlib.sha256(Path(audio_path).read_bytes()).hexdigest(),
        "transcript": evidence["transcript"], "span_audit": evidence["span_audit"],
        "selected": indices[0].tolist(), "gates": gates[0].detach().tolist(),
        "loss": float(output.loss), "gradient_norms": gradients, "update_norms": updates,
        "parameter_count": sum(p.numel() for p in connector.parameters()),
        "model_load_s": loaded-started, "extract_s": extracted-loaded,
        "forward_backward_update_s": end-extracted, "elapsed_s": end-started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "alignment": evidence["alignment"], "posterior_definition": evidence["posterior_definition"],
        "scientific_claim": "Engineering smoke only; no held-out performance claim"}
    # Tiny real learned artifact returned to the local caller, never model weights.
    import io
    checkpoint = io.BytesIO()
    torch.save(connector.cpu().state_dict(), checkpoint)
    payload = checkpoint.getvalue()
    result["checkpoint_sha256"] = hashlib.sha256(payload).hexdigest()
    return result, payload
