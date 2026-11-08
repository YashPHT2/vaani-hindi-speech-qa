"""Historical-development ASR decoding-policy experiment (fallback vs greedy).

Question: does repetition-aware decoding reduce the degeneration loops observed
in the frozen evaluation, without regressing ordinary views?

Design, frozen before execution:

  Arm A  frozen baseline: greedy, scalar temperature 0, no fallback (identical
         call to the recorded final worker).
  Arm B  repetition-aware: temperature tuple (0.0,0.2,0.4,0.6,0.8,1.0) with
         compression_ratio_threshold=2.4 and logprob_threshold=-1.0. HF retries
         the SAME decode at the next temperature only when a check trips.

Both arms run in one job over the same waveforms, so the comparison is paired at
the view level. Data is the historical development split ONLY (12 sources, 24
views); the fresh-100 evaluation is never touched.

Validity checks, in order of strength:
  1. Every reconstructed waveform must match the dev manifest audio hash, so the
     shard lookup and noise transform are proven correct before decoding.
  2. Arm A within this job is the decoder baseline for arm B. The historical
     dev-auto transcripts are context only: they were produced with transformers
     4.57.2 and this job pins 4.51.3, so exact transcript reproduction is NOT
     claimed and must not be reported as such.

Run locally only to inspect the plan; execution happens on Modal:
    python -X utf8 research-team/runner_fallback.py --plan
    python -X utf8 research-team/runner_fallback.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
import wave
from pathlib import Path

import modal

RUN_ID = "dev-fallback-policy-v1"
ASR = ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d")
SHARD_SHA = "8a0be26c5584c1478233a6f47fb90275c713f72565327ea1da76c9f1cd5dd38b"
CACHE_VOLUME = "vaani-acoustic-interface-cache-v1"
OUT_DIR = f"/cache/runs/{RUN_ID}"
CONDITIONS = ("clean", "noise_10db")
MAX_NEW_TOKENS = 224

# Frozen decoding policy for arm B.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
COMPRESSION_RATIO_THRESHOLD = 2.4
LOGPROB_THRESHOLD = -1.0

RESOURCE_LIMITS = {
    "gpu": "A10", "gpu_count": 1, "cpu": 2, "memory_gib": 16,
    "function_s": 3600, "startup_s": 300, "retries": 0,
}

app = modal.App("vaani-dev-fallback-policy")
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4",
        "scipy==1.15.2", "soundfile==0.13.1", "accelerate==1.6.0",
        "safetensors==0.5.3", "pyarrow==19.0.1",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def whisper_compression_ratio(text: str) -> float:
    """Exact OpenAI whisper.utils.compression_ratio (higher = more repetitive)."""
    import zlib
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw))


def transformed_wave(shard_row, condition):
    """Bit-exact reproduction of the frozen v2 clean/noise transform."""
    import numpy as np
    from scipy.io import wavfile

    rate, samples = wavfile.read(io.BytesIO(shard_row["audio"]["bytes"]))
    assert rate == 16000 and samples.ndim == 1 and 0 < len(samples) <= 480000
    if np.issubdtype(samples.dtype, np.integer):
        clean = samples.astype(np.float64) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    else:
        clean = samples.astype(np.float64)
    seed = int(sha(("noise:" + str(shard_row["id"])).encode())[:16], 16)
    noise = np.random.default_rng(seed).standard_normal(len(clean))
    energy = np.mean(clean ** 2)
    noise *= np.sqrt(energy / (10 * np.mean(noise ** 2)))
    noisy = clean + noise
    gain = min(1., .98 / max(np.max(np.abs(clean)), np.max(np.abs(noisy))))
    signal = clean if condition == "clean" else noisy
    pcm = np.rint(np.clip(signal * gain, -1, 1) * 32767).astype("<i2")
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as target:
        target.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        target.writeframes(pcm.tobytes())
    return encoded.getvalue(), {"audio_sha256": sha(encoded.getvalue()), "seconds_audio": len(pcm) / 16000}


class FallbackCounter:
    """Count real fallback decisions by wrapping the library's own gate.

    The Whisper mixin calls _need_fallback once per candidate sequence per
    temperature rung. Counting invocations and how many returned True is direct
    evidence that retries happened, instead of inferring it from output length.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.tripped = 0
        self.ratios: list[float] = []
        self._original = None

    def __enter__(self):
        from transformers.models.whisper import generation_whisper as gw

        counter = self
        original = gw.WhisperGenerationMixin._need_fallback

        def wrapped(self_, seek_sequence, seek_outputs, index, logits_processor,
                    generation_config, vocab_size, temperature):
            result = original(self_, seek_sequence, seek_outputs, index, logits_processor,
                             generation_config, vocab_size, temperature)
            counter.calls += 1
            if result[0]:
                counter.tripped += 1
            try:
                counter.ratios.append(
                    self_._retrieve_compression_ratio(seek_sequence, vocab_size)
                )
            except Exception:
                pass
            return result

        self._original = original
        gw.WhisperGenerationMixin._need_fallback = wrapped
        return self

    def __exit__(self, *exc):
        from transformers.models.whisper import generation_whisper as gw

        if self._original is not None:
            gw.WhisperGenerationMixin._need_fallback = self._original
        return False


def decode_arm_a(asr, processor, features, attention):
    """Frozen baseline call, byte-identical policy to the recorded worker."""
    import torch
    with torch.inference_mode():
        ids = asr.generate(
            input_features=features, attention_mask=attention,
            language="hi", task="transcribe", do_sample=False, num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS, return_timestamps=False,
            return_token_timestamps=False, output_scores=False, output_logits=False,
            use_cache=True,
        )[0]
    return ids


def decode_arm_b(asr, processor, features, attention):
    """Repetition-aware call: HF retries only when a registered check trips.

    The Whisper mixin owns the per-rung sampling switch: it sets
    generation_config.do_sample = (temperature > 0) and generation_config.temperature
    itself, and deletes any user-supplied do_sample/temperature/num_beams from the
    inner kwargs. Passing do_sample here would therefore be ignored, so it is
    omitted deliberately rather than silently negated. Rung 0 is greedy; later
    rungs sample at higher temperature only if a check tripped.
    """
    import torch
    with torch.inference_mode():
        ids = asr.generate(
            input_features=features, attention_mask=attention,
            language="hi", task="transcribe", num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS, return_timestamps=False,
            return_token_timestamps=False, output_scores=False, output_logits=False,
            use_cache=True,
            temperature=FALLBACK_TEMPERATURES,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
        )[0]
    return ids


@app.function(
    image=image, gpu=RESOURCE_LIMITS["gpu"], cpu=(2, 2), memory=(16384, 16384),
    timeout=RESOURCE_LIMITS["function_s"], startup_timeout=RESOURCE_LIMITS["startup_s"],
    scaledown_window=2, max_containers=1, min_containers=0, buffer_containers=0,
    retries=0, volumes={"/cache": cache},
)
def run(payload: dict):
    """Execute both arms over the historical development views."""
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    started = time.monotonic()
    shard_path = Path("/cache/fleurs-hi-test.parquet")
    assert sha(shard_path.read_bytes()) == SHARD_SHA, "cached shard digest mismatch"
    table = pq.read_table(shard_path).to_pylist()

    publishers = {}
    for index, row in enumerate(table):
        publishers.setdefault((int(row["id"]), Path(row["audio"]["path"]).name), index)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the frozen plan before any decoding happens.
    plan = {
        "run_id": RUN_ID,
        "created_unix": time.time(),
        "arm_a": {"policy": "greedy_scalar_temperature0_no_fallback",
                  "do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS},
        "arm_b": {"policy": "temperature_fallback_with_compression_guard",
                  "temperatures": list(FALLBACK_TEMPERATURES),
                  "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
                  "logprob_threshold": LOGPROB_THRESHOLD,
                  "max_new_tokens": MAX_NEW_TOKENS},
        "split": "dev (historical development only; fresh-100 untouched)",
        "sources": len({r["source_id"] for r in payload["views"]}),
        "views": len(payload["views"]),
        "resources": RESOURCE_LIMITS,
    }
    (out_dir / "plan.json").write_bytes(json.dumps(plan, indent=2).encode("utf-8"))
    cache.commit()

    asr = WhisperForConditionalGeneration.from_pretrained(ASR[0], revision=ASR[1]).to("cuda").half().eval()
    processor = WhisperProcessor.from_pretrained(ASR[0], revision=ASR[1])

    rows = []
    for view in payload["views"]:
        index = publishers[(int(view["publisher_id"]), view["original_audio_path"])]
        shard_row = table[index]
        raw, info = transformed_wave(shard_row, view["condition"])
        assert info["audio_sha256"] == view["audio_sha256"], f"waveform mismatch {view['id']}"

        with wave.open(io.BytesIO(raw), "rb") as source:
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768
        batch = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        features = batch.input_features.cuda().half()
        attention = batch.attention_mask.cuda()

        row = {"id": view["id"], "source_id": view["source_id"], "condition": view["condition"],
               "transcript": view["transcript"], "answerable": view["answerable"]}
        for arm, fn in (("a", decode_arm_a), ("b", decode_arm_b)):
            torch.cuda.synchronize()
            tick = time.monotonic()
            if arm == "b":
                # Instrument the library gate so retries are observed, not assumed.
                with FallbackCounter() as counter:
                    ids = fn(asr, processor, features, attention)
                row["b_fallback_calls"] = counter.calls
                row["b_fallback_tripped"] = counter.tripped
                row["b_gate_ratios"] = [round(x, 4) for x in counter.ratios]
            else:
                ids = fn(asr, processor, features, attention)
            torch.cuda.synchronize()
            tokens = ids.tolist()
            text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            row[f"{arm}_seconds"] = time.monotonic() - tick
            row[f"{arm}_tokens"] = tokens
            row[f"{arm}_generated"] = len(tokens)
            row[f"{arm}_hypothesis"] = text
            row[f"{arm}_capped"] = len(tokens) >= MAX_NEW_TOKENS
            row[f"{arm}_eos"] = bool(tokens) and tokens[-1] == processor.tokenizer.eos_token_id
            row[f"{arm}_compression_ratio"] = whisper_compression_ratio(text)
        rows.append(row)

    (out_dir / "rows.json").write_bytes(json.dumps(rows, ensure_ascii=False, indent=1).encode("utf-8"))
    meta = {
        "run_id": RUN_ID, "rows": len(rows),
        "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__, "numpy": np.__version__},
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.monotonic() - started,
    }
    (out_dir / "run.json").write_bytes(json.dumps(meta, indent=2).encode("utf-8"))
    cache.commit()
    return meta


def build_payload() -> dict:
    root = Path(__file__).resolve().parents[1]
    manifest = [json.loads(s) for s in
                (root / "data/manifest.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    dev = [r for r in manifest if r["split"] == "dev"]
    # Deterministic order: condition then source id.
    dev.sort(key=lambda r: (r["condition"], r["source_id"]))
    views = []
    for r in dev:
        views.append({
            "id": r["id"], "source_id": r["source_id"], "publisher_id": int(r["source_id"].split("-")[-1]),
            "condition": r["condition"], "original_audio_path": r["original_audio_path"],
            "audio_sha256": r["audio_sha256"],
            "transcript": r["transcript"], "answerable": r["answerable"],
        })
    return {"views": views}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="Validate locally and print the frozen plan")
    args = ap.parse_args()
    payload = build_payload()
    if args.plan:
        print(json.dumps({"views": len(payload["views"]),
                          "sources": len({v["source_id"] for v in payload["views"]}),
                          "arm_a": "greedy scalar temperature 0, no fallback",
                          "arm_b": {"temperatures": list(FALLBACK_TEMPERATURES),
                                    "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
                                    "logprob_threshold": LOGPROB_THRESHOLD}}, indent=2))
        return 0
    with app.run():
        result = run.remote(payload)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
