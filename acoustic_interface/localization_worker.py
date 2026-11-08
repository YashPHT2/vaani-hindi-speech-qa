"""Frozen Whisper sensitivity to human-bounded target/placebo waveform masks.

This module does not infer boundaries. The caller must bind verified boundary
provenance before a remote call. Importing it performs no model or GPU work.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import time
import wave

REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SAMPLE_RATE = 16000
CONTROLS = ("original", "target_masked", "placebo_masked", "mismatch")
PREFERENCE_METRICS = (
    "lexical_sum_logp", "lexical_mean_logp", "sequence_sum_logp",
    "sequence_mean_logp", "differing_span_sum_logp", "first_divergent_token_logp",
)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def token_regions(plus, minus):
    common = 0
    while common < min(len(plus), len(minus)) and plus[common] == minus[common]:
        common += 1
    suffix = 0
    while suffix < min(len(plus) - common, len(minus) - common) and plus[-suffix - 1] == minus[-suffix - 1]:
        suffix += 1
    return {
        "common_prefix_lexical_tokens": common,
        "common_suffix_lexical_tokens": suffix,
        "plus_differing_span": [common, len(plus) - suffix],
        "minus_differing_span": [common, len(minus) - suffix],
    }


def mask_bounds(config, sample_count):
    """Validate supplied intervals, then round their endpoints once to samples."""
    assert sample_count > 0
    assert config["fade_s"] == .005, "Only the declared five-millisecond inside fade is allowed"
    bounds = {}
    for name in ("target", "placebo"):
        interval = config[name + "_mask_interval_s"]
        assert isinstance(interval, (list, tuple)) and len(interval) == 2
        start, end = map(float, interval)
        assert math.isfinite(start) and math.isfinite(end)
        assert 0 <= start < end <= sample_count / SAMPLE_RATE
        lo, hi = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        assert 0 <= lo < hi <= sample_count
        bounds[name] = (lo, hi)
    a, b = bounds["target"], bounds["placebo"]
    assert a[1] - a[0] == b[1] - b[0], "Masks must have exactly equal rounded sample lengths"
    assert a[1] <= b[0] or b[1] <= a[0], "Target and placebo windows overlap"
    fade_samples = round(config["fade_s"] * SAMPLE_RATE)
    assert fade_samples >= 2 and a[1] - a[0] >= 2 * fade_samples + 1, "Mask must contain a zero plateau"
    return bounds, fade_samples


def apply_mask(samples, bounds, fade_samples):
    """Inside-only raised-cosine gain, one at window edges and zero centrally."""
    import numpy as np

    assert samples.dtype == np.float32 and samples.ndim == 1
    lo, hi = bounds
    assert 0 <= lo < hi <= len(samples) and hi - lo >= 2 * fade_samples + 1
    gain = np.zeros(hi - lo, dtype=np.float32)
    ramp = (.5 * (1 + np.cos(np.linspace(0., np.pi, fade_samples, dtype=np.float64)))).astype(np.float32)
    gain[:fade_samples] = ramp
    gain[-fade_samples:] = ramp[::-1]
    masked = samples.copy()
    masked[lo:hi] *= gain
    # Equality is by bytes, so signed zeros outside the window are preserved too.
    outside_equal = masked[:lo].tobytes() == samples[:lo].tobytes() and masked[hi:].tobytes() == samples[hi:].tobytes()
    assert outside_equal and len(masked) == len(samples) and masked.dtype == samples.dtype
    before = samples[lo:hi].astype(np.float64)
    after = masked[lo:hi].astype(np.float64)
    audit = {
        "sample_bounds_half_open": [lo, hi], "sample_count": hi - lo,
        "rounded_interval_s": [lo / SAMPLE_RATE, hi / SAMPLE_RATE],
        "fade_samples_each_edge": fade_samples,
        "fade_definition": "inside-window raised cosine gain 1 to 0; mirrored at end; zero plateau; outside unchanged",
        "before_window_rms": float(np.sqrt(np.mean(before ** 2))),
        "after_window_rms": float(np.sqrt(np.mean(after ** 2))),
        "removed_energy_sample_sum": float(np.sum(before ** 2 - after ** 2)),
        "removed_signal_energy_sample_sum": float(np.sum((before - after) ** 2)),
        "outside_mask_bitwise_unchanged": outside_equal,
        "masked_float32_sha256": sha256(masked.astype("<f4", copy=False).tobytes(order="C")),
    }
    return masked, audit


def decode_wave(raw, expected_sha256):
    import numpy as np

    assert sha256(raw) == expected_sha256
    with wave.open(io.BytesIO(raw), "rb") as source:
        assert source.getnchannels() == 1 and source.getsampwidth() == 2 and source.getframerate() == SAMPLE_RATE
        count = source.getnframes()
        payload = source.readframes(count)
    assert len(payload) == count * 2
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768
    assert samples.dtype == np.float32 and 0 < len(samples) <= 480000 and np.isfinite(samples).all()
    return samples


def validate_inputs(inputs, config):
    """Cheap local or remote invariants; no tokenizer/model/network calls."""
    assert config["run_id"] == "localized-omission-sensitivity-v1"
    assert config["model"] == "openai/whisper-small" and config["model_revision"] == REVISION
    assert config["boundary_provenance"]["validation_status"] in ("human_validated", "human_reported_bounds")
    assert len(inputs) == 2 and {item["condition"] for item in inputs} == {"clean", "noise_10db"}
    assert len({item["id"] for item in inputs}) == 2
    decoded = []
    for item in inputs:
        assert item["source_id"] == "fleurs-hi-1806"
        assert item["id"] == "fleurs-hi-1806-" + item["condition"]
        assert item["donor_id"] == "fleurs-hi-1868-" + item["condition"]
        assert item["duration_s"] == item["donor_duration_s"] == 13.68
        actual = decode_wave(item["audio_bytes"], item["audio_sha256"])
        donor = decode_wave(item["donor_bytes"], item["donor_audio_sha256"])
        assert len(actual) == len(donor) == round(13.68 * SAMPLE_RATE)
        bounds, fade_samples = mask_bounds(config, len(actual))
        decoded.append((item, actual, donor, bounds, fade_samples))
    return decoded


def run(inputs, config, commit):
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.modeling_outputs import BaseModelOutput

    started = time.monotonic()
    assert torch.cuda.is_available(), "GPU-only diagnostic; no CPU model fallback"
    torch.set_num_threads(2)
    torch.manual_seed(17)
    decoded = validate_inputs(inputs, config)
    out = Path("/cache/runs") / config["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def check():
        if time.monotonic() - started > 540:
            raise TimeoutError("Internal localized diagnostic deadline; no automatic retry")

    counts = {"acoustic_encoder_forwards": 0, "full_candidate_decoder_forwards": 0, "shared_prefix_decoder_forwards": 0}
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", revision=REVISION)
    tokenizer = processor.tokenizer
    tokenizer.set_prefix_tokens(language="hi", task="transcribe", predict_timestamps=False)
    prefix = list(tokenizer.prefix_tokens)
    candidates = {name: tokenizer.encode(config["pair"][name], add_special_tokens=False) for name in ("plus", "minus")}
    regions = token_regions(candidates["plus"], candidates["minus"])
    common = regions["common_prefix_lexical_tokens"]
    assert common < min(map(len, candidates.values()))
    assert candidates["plus"][:common] == candidates["minus"][:common]
    assert candidates["plus"][common] != candidates["minus"][common]
    common_decoder = prefix + candidates["plus"][:common]
    tokenization = {
        "prefix_tokens": prefix, "eos_token_id": tokenizer.eos_token_id,
        "candidates": candidates, "regions": regions,
        "shared_prefix_decoder_input_ids": common_decoder,
        "shared_prefix_first_difference_token_ids": {name: candidates[name][common] for name in candidates},
        "scoring": "Forced-prefix targets excluded; every lexical token included; EOS separate; no truncation",
        "first_difference_primary": "ONE shared-prefix decoder forward; both token probabilities from the SAME final logit distribution",
        "full_candidate_first_difference": "Secondary teacher-forced per-candidate diagnostic; not primary log odds",
        "differing_span": "Token LCP/LCS region, independent of human waveform windows",
        "expected_forward_counts": {"acoustic_encoder_forwards": 8, "full_candidate_decoder_forwards": 16,
            "shared_prefix_decoder_forwards": 8, "total_decoder_forwards": 24},
    }
    token_bytes = json.dumps(tokenization, ensure_ascii=False, indent=2).encode("utf-8")
    (out / "tokenization.json").write_bytes(token_bytes)
    # Persist the exact scoring mask/token boundaries before any acoustic encode.
    commit()
    asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", revision=REVISION,
        torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    for parameter in asr.parameters():
        parameter.requires_grad_(False)
    assert prefix[0] == asr.config.decoder_start_token_id

    def encode(samples):
        batch = processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        assert batch.input_features.shape[-1] == 3000
        counts["acoustic_encoder_forwards"] += 1
        return asr.model.encoder(batch.input_features.to("cuda", dtype=torch.float16), return_dict=True).last_hidden_state

    def full_score(hidden, name):
        lexical = candidates[name]
        sequence = prefix + lexical + [tokenizer.eos_token_id]
        ids = torch.tensor([sequence], device="cuda")
        counts["full_candidate_decoder_forwards"] += 1
        logits = asr(encoder_outputs=BaseModelOutput(last_hidden_state=hidden), decoder_input_ids=ids[:, :-1],
            use_cache=False, return_dict=True).logits.float()
        logp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)[0]
        lex = logp[len(prefix) - 1:len(prefix) - 1 + len(lexical)]
        eos = logp[-1]
        assert len(lex) == len(lexical) and torch.isfinite(logp).all()
        lo, hi = regions[name + "_differing_span"]
        differing = lex[lo:hi]
        return {
            "candidate": name, "lexical_token_ids": lexical, "lexical_token_logp": lex.tolist(),
            "lexical_token_count": len(lexical), "lexical_sum_logp": float(lex.sum()), "lexical_mean_logp": float(lex.mean()),
            "eos_logp": float(eos), "sequence_sum_logp": float(lex.sum() + eos),
            "sequence_mean_logp": float((lex.sum() + eos) / (len(lexical) + 1)),
            "differing_span_sum_logp": float(differing.sum()),
            "differing_span_mean_logp": float(differing.mean()) if len(differing) else None,
            "differing_span_token_count": len(differing), "first_divergent_token_id": lexical[common],
            "first_divergent_token_logp": float(lex[common]),
        }

    def shared_prefix_score(hidden):
        ids = torch.tensor([common_decoder], device="cuda")
        counts["shared_prefix_decoder_forwards"] += 1
        # Exactly one forward and one final-position distribution for BOTH IDs.
        distribution = asr(encoder_outputs=BaseModelOutput(last_hidden_state=hidden), decoder_input_ids=ids,
            use_cache=False, return_dict=True).logits[0, -1].float().log_softmax(-1)
        assert torch.isfinite(distribution).all()
        plus_id, minus_id = candidates["plus"][common], candidates["minus"][common]
        plus_logp, minus_logp = float(distribution[plus_id]), float(distribution[minus_id])
        return {"decoder_input_ids": common_decoder, "plus_token_id": plus_id, "minus_token_id": minus_id,
            "plus_logp": plus_logp, "minus_logp": minus_logp,
            "log_odds": plus_logp - minus_logp, "decoder_forwards": 1,
            "same_final_distribution": True}

    rows = []
    with torch.inference_mode():
        for item, actual, donor, bounds, fade_samples in decoded:
            target, target_audit = apply_mask(actual, bounds["target"], fade_samples)
            placebo, placebo_audit = apply_mask(actual, bounds["placebo"], fade_samples)
            assert target_audit["sample_count"] == placebo_audit["sample_count"]
            for control, samples, mask_audit in (
                ("original", actual, None), ("target_masked", target, target_audit),
                ("placebo_masked", placebo, placebo_audit), ("mismatch", donor, None),
            ):
                check()
                hidden = encode(samples)
                shared = shared_prefix_score(hidden)
                scores = {name: full_score(hidden, name) for name in ("plus", "minus")}
                row = {"id": item["id"], "source_id": item["source_id"], "condition": item["condition"], "control": control,
                    "input_duration_s": len(samples) / SAMPLE_RATE, "sample_count": len(samples),
                    "actual_audio_sha256": item["audio_sha256"],
                    "scored_float32_sha256": sha256(samples.astype("<f4", copy=False).tobytes(order="C")),
                    "donor_id": item["donor_id"] if control == "mismatch" else None,
                    "donor_audio_sha256": item["donor_audio_sha256"] if control == "mismatch" else None,
                    "mask_audit": mask_audit,
                    "before_whole_waveform_rms": float(np.sqrt(np.mean(actual.astype(np.float64) ** 2))),
                    "scores": scores,
                    "shared_prefix": shared,
                    "primary_shared_prefix_log_odds": shared["log_odds"],
                    "preference": {metric: scores["plus"][metric] - scores["minus"][metric]
                        for metric in PREFERENCE_METRICS},
                    "decoder_forwards_this_control": 3}
                rows.append(row)
                with (out / "scores.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                (out / "progress.json").write_text(json.dumps({"completed_control_rows": len(rows), "counts": counts}), encoding="utf-8")
                commit()
                print(json.dumps({"phase": "localized_scoring", "id": item["id"], "control": control}), flush=True)
    assert counts == {"acoustic_encoder_forwards": 8, "full_candidate_decoder_forwards": 16, "shared_prefix_decoder_forwards": 8}
    assert len(rows) == 8 and all(p.grad is None and not p.requires_grad for p in asr.parameters())
    reported_window = config["boundary_provenance"]["validation_status"] == "human_reported_bounds"
    interpretation = ("Single-source user-reported uncertain waveform-window sensitivity; full phrase extent not established"
        if reported_window else "Single-source frozen-decoder synthetic-omission sensitivity to verified waveform windows")
    result = {"status": "localized_omission_sensitivity_complete", "config": config, "tokenization": tokenization,
        "tokenization_sha256": sha256(token_bytes), "rows": rows,
        "forward_counts": {**counts, "total_decoder_forwards": 24, "total_model_forwards_including_encoder": 32},
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "elapsed_s": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "training_updates": 0, "fresh_test_loaded": False, "independent_sources": 1,
        "boundary_validation_status": config["boundary_provenance"]["validation_status"],
        "full_phrase_extent_verified": not reported_window,
        "interpretation": interpretation + "; masking energy and synthetic candidate lengths remain confounded; no encoder absence/recovery or QA claim"}
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "artifact-manifest.json").write_text(json.dumps({"result_sha256": sha256(result_path.read_bytes()),
        "tokenization_sha256": sha256(token_bytes), "scores_sha256": sha256((out / "scores.jsonl").read_bytes())}, indent=2), encoding="utf-8")
    commit()
    return result
