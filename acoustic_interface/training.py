"""Bounded historical-only paired pilot; fresh test data are never loaded here."""
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from connector import AcousticMemory, pack_prompt
from features import extract
from worker import models


def waveform(shard_row, row):
    import wave
    from scipy.io import wavfile
    rate, samples = wavfile.read(io.BytesIO(shard_row["audio"]["bytes"]))
    assert rate == 16000
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    else:
        samples = samples.astype(np.float64)
    seed = int(hashlib.sha256(("noise:" + str(shard_row["id"])).encode()).hexdigest()[:16], 16)
    noise = np.random.default_rng(seed).standard_normal(len(samples))
    noise *= np.sqrt(np.mean(samples ** 2) / (10 * np.mean(noise ** 2)))
    noisy = samples + noise
    gain = min(1., .98 / max(np.max(np.abs(samples)), np.max(np.abs(noisy))))
    signal = samples if row["condition"] == "clean" else noisy
    pcm = np.rint(np.clip(signal * gain, -1, 1) * 32767).astype("<i2")
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as f:
        f.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        f.writeframes(pcm.tobytes())
    assert hashlib.sha256(encoded.getvalue()).hexdigest() == row["audio_sha256"]
    return pcm.astype(np.float32) / 32768


def run(rows, config, commit):
    import pyarrow.parquet as pq
    import urllib.request
    from torch.nn import functional as F
    start = time.monotonic()
    deadline = start + config["work_timeout_s"]
    out = Path("/cache/runs") / config["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    def event(payload):
        payload["elapsed_s"] = time.monotonic() - start
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    def check():
        if time.monotonic() > deadline:
            raise TimeoutError("Internal budget deadline; cloud checkpoints preserved")
    historical = sorted({r["source_id"] for r in rows if r["split"] == "test"},
                        key=lambda s: hashlib.sha256(("acoustic-interface-pilot-v1:" + s.rsplit("-", 1)[-1]).encode()).hexdigest())[:config["train_sources"]]
    chosen = [r for r in rows if r["source_id"] in historical or r["split"] == "dev"]
    assert len({r["source_id"] for r in chosen if r["split"] == "dev"}) == 12
    assert not ({r["source_id"] for r in chosen if r["split"] == "dev"} & set(historical))
    (out / "config.json").write_text(json.dumps({**config, "train_source_ids": historical,
        "view_ids": [r["id"] for r in chosen]}, indent=2), encoding="utf-8")
    shard = Path("/cache/fleurs-hi-test.parquet")
    if not shard.exists():
        urllib.request.urlretrieve("https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/hi_in/test/0000.parquet", shard)
    assert hashlib.file_digest(shard.open("rb"), "sha256").hexdigest() == "8a0be26c5584c1478233a6f47fb90275c713f72565327ea1da76c9f1cd5dd38b"
    table = pq.read_table(shard).to_pylist()
    from gpu_preflight import run as preflight
    contract = preflight()
    event({"phase": "gpu_contract", **contract})
    proc, asr, tok, lm = models()
    cache = Path("/cache/features-v1")
    cache.mkdir(exist_ok=True)
    evidence = {}
    for row in chosen:
        check()
        path = cache / (row["id"] + ".pt")
        if path.exists():
            feat = torch.load(path, weights_only=False)
        else:
            feat = extract(asr, proc, waveform(table[row["source_row_index"]], row))
            torch.save(feat, path)
        evidence[row["id"]] = feat
        event({"phase": "features", "id": row["id"], "spans": int(feat["span_mask"].sum())})
    del asr, proc, table
    torch.cuda.empty_cache()
    commit()
    train = [r for r in chosen if r["split"] == "test"]
    dev = [r for r in chosen if r["split"] == "dev"]
    def inputs(row, donor=None):
        feat = evidence[row["id"]]
        data = {k: v.cuda() for k, v in feat.items() if isinstance(v, torch.Tensor)}
        if donor is not None:
            # Recipient masks/span counts remain fixed; donor entire valid frame
            # stream is resampled into those exact slots, no true audio survives.
            other = evidence[donor["id"]]
            flat = other["audio"][other["frame_mask"]].T[None].cuda()
            assert flat.shape[-1] > 0 and data["frame_mask"].any()
            count = int(data["frame_mask"].sum())
            replacement = F.interpolate(flat, size=count, mode="linear", align_corners=False)[0].T
            data["audio"] = torch.zeros_like(data["audio"])
            data["audio"][data["frame_mask"]] = replacement
        ids = tok(row["question"], return_tensors="pt").input_ids.cuda()
        data.update(question=lm.get_input_embeddings()(ids).float(), question_mask=torch.ones_like(ids).bool())
        return data
    def packed(adapter, row, condition, donor=None, target=True):
        mem, selected, gates = adapter(**inputs(row, donor), condition=condition)
        p = pack_prompt(lm, tok, evidence[row["id"]]["transcript"], row["question"], mem[0], row["answers"][0] if target else None)
        return p, selected[0].tolist(), gates[0].detach().tolist()
    # Both learned arms share all parameter tensors at initialization.
    torch.manual_seed(17)
    base = AcousticMemory().cuda()
    raw_u = torch.cat([evidence[r["id"]]["uncertainty"][evidence[r["id"]]["span_mask"] & evidence[r["id"]]["uncertainty_available"] & evidence[r["id"]]["frame_mask"].any(-1)] for r in train])
    base.uncertainty_mean.copy_(raw_u.mean(0).cuda())
    base.uncertainty_std.copy_(raw_u.std(0).clamp_min(1e-5).cuda())
    initial = copy.deepcopy(base.state_dict())
    all_metrics, blobs = [], {}
    qa_gradient_checks = {}
    for arm, condition in [("full", "full"), ("no_uncertainty", "uncertainty_off")]:
        adapter = AcousticMemory().cuda()
        adapter.load_state_dict(initial)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001, weight_decay=.01)
        with torch.no_grad():
            adapter.eval()
            initial_selection = {r["id"]: packed(adapter, r, condition)[1] for r in dev}
        best, best_epoch = float("inf"), None
        best_state = None
        for epoch in range(config["epochs"]):
            order = list(train)
            random.Random(17 + epoch).shuffle(order)
            optimizer.zero_grad()
            losses = []
            adapter.train()
            update_rows = 0
            for i, row in enumerate(order):
                check()
                captured = []
                hook = None
                fixture_valid = evidence[row["id"]]["span_mask"] & evidence[row["id"]]["frame_mask"].any(-1)
                if arm not in qa_gradient_checks and int(fixture_valid.sum()) > adapter.slots:
                    def capture(module, inputs, output):
                        output.retain_grad()
                        captured.append(output)
                    hook = adapter.score.register_forward_hook(capture)
                p, selected, _ = packed(adapter, row, condition)
                loss = lm(**p, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                group_start = (i // 8) * 8
                group_size = min(8, len(order) - group_start)
                if int(fixture_valid.sum()) > 0 and loss.requires_grad:
                    (loss / group_size).backward()
                    update_rows += 1
                else:
                    event({"phase": "text_only_no_adapter_update", "arm": arm, "id": row["id"], "epoch": epoch})
                if hook is not None:
                    hook.remove()
                    valid = evidence[row["id"]]["span_mask"][0] & evidence[row["id"]]["frame_mask"].any(-1)[0]
                    omitted = valid.clone()
                    omitted[selected] = False
                    logit_grad = captured[0].grad[0, :, 0].detach().cpu()
                    assert torch.isfinite(logit_grad).all()
                    assert omitted.any(), "Actual QA gradient fixture needs S>K"
                    assert (logit_grad[~valid] == 0).all()
                    assert all(p.grad is None for p in lm.parameters())
                    absolute = logit_grad[omitted].abs()
                    signal = {"id": row["id"], "omitted_count": int(omitted.sum()),
                        "nonzero_count": int((absolute > 0).sum()),
                        "nonzero_fraction": float((absolute > 0).float().mean()),
                        "omitted_gradient_norm": float(absolute.norm()),
                        "min_abs_omitted_gradient": float(absolute.min()),
                        "max_abs_omitted_gradient": float(absolute.max()), "frozen_lm": True}
                    if signal["nonzero_count"] > 0:
                        qa_gradient_checks[arm] = signal
                    event({"phase": "actual_qa_gradient", "arm": arm, **signal})
                losses.append(float(loss.detach()))
                if (i + 1) % 8 == 0 or i + 1 == len(order):
                    grad = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.))
                    if not np.isfinite(grad):
                        raise RuntimeError("Nonfinite adapter gradient")
                    if update_rows:
                        optimizer.step()
                    optimizer.zero_grad()
                    event({"phase": "train", "arm": arm, "epoch": epoch, "examples": i+1, "loss": float(np.mean(losses[-8:])), "gradient_norm": grad, "effective_update_rows": update_rows})
                    update_rows = 0
            adapter.eval()
            with torch.no_grad():
                vals = [float(lm(**packed(adapter, r, condition)[0], use_cache=False).loss) for r in dev]
            score = float(np.mean(vals))
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite development loss")
            if score < best:
                best, best_epoch = score, epoch
                best_state = copy.deepcopy(adapter.state_dict())
            torch.save({"state_dict": adapter.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "arm": arm, "config": config, "train_ids": historical}, out / (arm + "-last.pt"))
            event({"phase": "epoch", "arm": arm, "epoch": epoch, "dev_nll": score, "train_nll": float(np.mean(losses))})
            commit()
        if arm not in qa_gradient_checks:
            raise RuntimeError("No S>K historical QA fixture available for routing gradient check")
        adapter.load_state_dict(best_state)
        checkpoint = io.BytesIO()
        torch.save({"state_dict": adapter.cpu().state_dict(), "best_epoch": best_epoch, "config": config}, checkpoint)
        blobs[arm] = checkpoint.getvalue()
        (out / (arm + "-best.pt")).write_bytes(blobs[arm])
        event({"phase": "checkpoint", "arm": arm, "best_epoch": best_epoch,
            "parameter_update_norms": {n: float((p.detach().cpu() - initial[n].cpu()).norm())
                for n, p in adapter.named_parameters()},
            "sha256": hashlib.sha256(blobs[arm]).hexdigest()})
        commit()
        adapter.cuda().eval()
        # Pair only genuine aligned rows. Odd tail uses a three-cycle.
        donors = {r["id"]: None for r in dev}
        for acoustic_condition in ("clean", "noise_10db"):
            ordered = sorted([r for r in dev if r["condition"] == acoustic_condition
                and evidence[r["id"]]["frame_mask"].any()], key=lambda r: (r["duration_s"], r["id"]))
            n = len(ordered)
            paired = n if n % 2 == 0 else max(0, n - 3)
            for i in range(paired):
                donors[ordered[i]["id"]] = ordered[i ^ 1]
            if n % 2 and n >= 3:
                for i in range(n - 3, n):
                    donors[ordered[i]["id"]] = ordered[n - 3 + (i - (n - 3) + 1) % 3]
        for row in dev:
            for control in (["correct", "wrong_audio", "text"] if arm == "full" else ["correct"]):
                check()
                with torch.no_grad():
                    saved_slots = adapter.slots
                    if control == "text":
                        adapter.slots = 0
                    donor = donors[row["id"]] if control == "wrong_audio" else None
                    p, ids, gates = packed(adapter, row, condition, donor)
                    nll = float(lm(**p, use_cache=False).loss)
                    gp, _, _ = packed(adapter, row, condition, donor, target=False)
                    gp.pop("labels")
                    gp.pop("position_ids")
                    generation_start = time.monotonic()
                    answer_ids = lm.generate(**gp, max_new_tokens=48, do_sample=False,
                                             eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)[0]
                    answer = tok.decode(answer_ids, skip_special_tokens=True).strip()
                    adapter.slots = saved_slots
                record = {"arm": arm, "control": control, "id": row["id"], "source_id": row["source_id"],
                    "condition": row["condition"], "nll": nll, "answer": answer,
                    "generated_tokens": len(answer_ids), "generation_s": time.monotonic()-generation_start,
                    "answer_target_tokens_including_end": int((p["labels"] != -100).sum()),
                    "answers": row["answers"], "exact_match": answer in row["answers"], "selected": ids,
                    "gates": gates, "alignment_available": bool(evidence[row["id"]]["frame_mask"].any()),
                    "counterfactual_available": not bool(evidence[row["id"]]["frame_mask"].any()) or donors[row["id"]] is not None,
                    "initial_selected": initial_selection[row["id"]],
                    "valid_spans": int(evidence[row["id"]]["span_mask"].sum()),
                    "donor": donor["id"] if donor else None,
                    "donor_duration_ratio": donor["duration_s"] / row["duration_s"] if donor else None,
                    "transcript": evidence[row["id"]]["transcript"], "span_audit": evidence[row["id"]]["span_audit"]}
                all_metrics.append(record)
                event({"phase": "evaluation", **record})
        commit()
    result = {"status": "pilot_complete", "config": config, "rows": all_metrics,
        "elapsed_s": time.monotonic()-start, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "training_source_ids": historical,
        "gpu_contract": contract, "actual_qa_gradient_checks": qa_gradient_checks,
        "alignment_counts": {c: {"total": sum(r["condition"] == c for r in chosen),
            "aligned": sum(r["condition"] == c and bool(evidence[r["id"]]["frame_mask"].any()) for r in chosen),
            "more_than_k": sum(r["condition"] == c and int(evidence[r["id"]]["span_mask"].sum()) > 4 for r in chosen)} for c in ("clean", "noise_10db")},
        "checkpoint_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}}
    (out / "artifact-manifest.json").write_text(json.dumps({"run_id": config["run_id"],
        "checkpoint_sha256": result["checkpoint_sha256"], "config": config}, indent=2))
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    commit()
    return result, blobs
