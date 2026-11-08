"""Frozen stage-3 protocol: historical fitting, then sealed fresh evaluation."""
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import time
import wave

import numpy as np
import torch
from torch.nn import functional as F

from connector import AcousticMemory, pack_prompt
from features import extract
from training import waveform
from worker import ASR, LLM, models


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fresh_wave(shard_row, condition, expected_canonical):
    from scipy.io import wavfile
    rate, samples = wavfile.read(io.BytesIO(shard_row["audio"]["bytes"]))
    assert rate == 16000 and samples.ndim == 1
    canonical = digest(str(rate).encode() + b"\0" + str(samples.dtype).encode() + b"\0" + samples.tobytes(order="C"))
    assert canonical == expected_canonical, "Fresh publisher recording byte binding failed"
    if np.issubdtype(samples.dtype, np.integer):
        clean = samples.astype(np.float64) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
    else:
        clean = samples.astype(np.float64)
    seed = int(digest(("noise:" + str(shard_row["id"])).encode())[:16], 16)
    noise = np.random.default_rng(seed).standard_normal(len(clean))
    noise *= np.sqrt(np.mean(clean ** 2) / (10 * np.mean(noise ** 2)))
    noisy = clean + noise
    gain = min(1., .98 / max(np.max(np.abs(clean)), np.max(np.abs(noisy))))
    signal = clean if condition == "clean" else noisy
    pcm = np.rint(np.clip(signal * gain, -1, 1) * 32767).astype("<i2")
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as target:
        target.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        target.writeframes(pcm.tobytes())
    return pcm.astype(np.float32) / 32768, digest(encoded.getvalue()), gain


def derange(rows, evidence, key):
    """Declared same-condition adjacent pairs, with an odd final three-cycle."""
    mapping = {r["id"]: None for r in rows}
    for condition in ("clean", "noise_10db"):
        ordered = sorted([r for r in rows if r["condition"] == condition
            and bool(evidence[r["id"]]["frame_mask"].any())], key=key)
        n = len(ordered)
        paired = n if n % 2 == 0 else max(0, n - 3)
        for i in range(paired):
            mapping[ordered[i]["id"]] = ordered[i ^ 1]
        if n % 2 and n >= 3:
            for i in range(n - 3, n):
                mapping[ordered[i]["id"]] = ordered[n - 3 + (i - (n - 3) + 1) % 3]
    return mapping


def run(historical_bytes, fresh_qa_bytes, selection_bytes, fresh_spec_bytes, config, commit):
    import pyarrow.parquet as pq
    from gpu_preflight import run as gpu_preflight

    start = time.monotonic()
    deadline = start + config["work_timeout_s"]
    out = Path("/cache/runs") / config["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    for name, payload in (("historical_manifest", historical_bytes), ("fresh_qa", fresh_qa_bytes), ("fresh_selection", selection_bytes), ("fresh_view_specs", fresh_spec_bytes)):
        assert digest(payload) == config["input_sha256"][name], name
        (out / (name + ".input")).write_bytes(payload)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def check():
        if time.monotonic() > deadline:
            raise TimeoutError("Final work deadline; preserved partial artifacts, no automatic retry")

    def event(record, save=False):
        record["elapsed_s"] = time.monotonic() - start
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (out / "phase.json").write_text(json.dumps({k: v for k, v in record.items()
            if k in ("phase", "arm", "epoch", "id", "elapsed_s", "completed_rows")}), encoding="utf-8")
        if save:
            commit()

    contract = gpu_preflight()
    event({"phase": "gpu_contract", **contract}, save=True)
    # Fresh gold and reference strings remain serialized until all fitting ends.
    historical = [json.loads(line) for line in historical_bytes.decode("utf-8").splitlines()]
    train = [r for r in historical if r["split"] == "test"]
    dev = [r for r in historical if r["split"] == "dev"]
    train_ids = sorted({r["source_id"] for r in train})
    assert len(train_ids) == 100 and len(train) == 200 and len(dev) == 24
    assert not (set(train_ids) & {r["source_id"] for r in dev})
    assert train_ids == config["training_source_ids"]
    shard = Path("/cache/fleurs-hi-test.parquet")
    with shard.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == config["publisher_shard_sha256"]
    table = pq.read_table(shard).to_pylist()
    model_start = time.monotonic()
    processor, asr, tokenizer, lm = models()
    model_load_s = time.monotonic() - model_start
    evidence = {}
    cache = Path("/cache/features-v1")
    cache.mkdir(exist_ok=True)
    for index, row in enumerate(train + dev):
        check()
        path = cache / (row["id"] + ".pt")
        if path.exists():
            feat = torch.load(path, weights_only=False)
        else:
            feat = extract(asr, processor, waveform(table[row["source_row_index"]], row))
            torch.save(feat, path)
        evidence[row["id"]] = feat
        event({"phase": "historical_features", "id": row["id"], "valid_spans": int(feat["span_mask"].sum())}, save=(index + 1) % 32 == 0)

    def inputs(row, condition, donor=None, query_donor=None, shuffle=False):
        feat = evidence[row["id"]]
        data = {k: v.cuda() for k, v in feat.items() if isinstance(v, torch.Tensor)}
        if donor is not None:
            other = evidence[donor["id"]]
            flat = other["audio"][other["frame_mask"]].T[None].cuda()
            assert flat.shape[-1] > 0 and data["frame_mask"].any()
            count = int(data["frame_mask"].sum())
            replacement = F.interpolate(flat, size=count, mode="linear", align_corners=False)[0].T
            data["audio"] = torch.zeros_like(data["audio"])
            data["audio"][data["frame_mask"]] = replacement
        moved, changed = 0, 0
        if shuffle:
            valid = data["span_mask"][0] & data["frame_mask"].any(-1)[0] & data["uncertainty_available"][0]
            indices = valid.nonzero(as_tuple=True)[0]
            if len(indices) > 1:
                seed = int(digest(("acoustic-interface-shuffle-v1:" + row["source_id"] + ":" + row["condition"]).encode())[:16], 16)
                shift = 1 + seed % (len(indices) - 1)
                original = data["uncertainty"][0, indices].clone()
                replacement = original.roll(int(shift), 0)
                data["uncertainty"][0, indices] = replacement
                moved = len(indices)
                changed = int((original != replacement).any(-1).sum())
        question = query_donor["question"] if query_donor is not None else row["question"]
        ids = tokenizer(question, return_tensors="pt").input_ids.cuda()
        data.update(question=lm.get_input_embeddings()(ids).float(), question_mask=torch.ones_like(ids).bool())
        return data, {"posterior_moved": moved, "posterior_changed": changed}

    def memory(adapter, row, condition, **controls):
        data, audit = inputs(row, condition, **controls)
        vectors, selected, gates = adapter(**data, condition=condition)
        return vectors[0], selected[0].tolist(), gates[0].detach().tolist(), audit

    torch.manual_seed(17)
    base = AcousticMemory().cuda()
    eligible_u = [evidence[r["id"]]["uncertainty"][evidence[r["id"]]["span_mask"]
        & evidence[r["id"]]["frame_mask"].any(-1) & evidence[r["id"]]["uncertainty_available"]] for r in train]
    raw_u = torch.cat(eligible_u)
    assert len(raw_u) > 0 and torch.isfinite(raw_u).all()
    base.uncertainty_mean.copy_(raw_u.mean(0).cuda())
    base.uncertainty_std.copy_(raw_u.std(0).clamp_min(1e-5).cuda())
    initial = copy.deepcopy(base.cpu().state_dict())
    del base
    adapters, blobs, qa_gradient_checks, training_audits = {}, {}, {}, {}
    for arm, condition in (("full", "full"), ("no_uncertainty", "uncertainty_off"), ("uniform", "uniform")):
        adapter = AcousticMemory().cuda()
        adapter.load_state_dict(initial)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001, weight_decay=.01)
        best, best_epoch, best_state = float("inf"), None, None
        update_examples, update_steps, no_audio_views = 0, 0, 0
        with torch.no_grad():
            adapter.eval()
            initial_selection = {r["id"]: memory(adapter, r, condition)[1] for r in dev}
        for epoch in range(3):
            adapter.train()
            order = list(train)
            random.Random(17 + epoch).shuffle(order)
            optimizer.zero_grad()
            losses, contributing = [], 0
            for index, row in enumerate(order):
                check()
                valid = evidence[row["id"]]["span_mask"][0] & evidence[row["id"]]["frame_mask"].any(-1)[0]
                captured, hook = [], None
                if arm != "uniform" and arm not in qa_gradient_checks and int(valid.sum()) > 4:
                    def capture(module, args, output):
                        output.retain_grad()
                        captured.append(output)
                    hook = adapter.score.register_forward_hook(capture)
                vector, selected, _, _ = memory(adapter, row, condition)
                packed = pack_prompt(lm, tokenizer, evidence[row["id"]]["transcript"], row["question"], vector, row["answers"][0])
                loss = lm(**packed, use_cache=False).loss
                assert torch.isfinite(loss)
                size = min(8, len(order) - (index // 8) * 8)
                if valid.any():
                    (loss / size).backward()
                    contributing += 1
                    update_examples += 1
                else:
                    no_audio_views += 1
                    event({"phase": "text_only_no_adapter_update", "arm": arm, "epoch": epoch, "id": row["id"]})
                if hook is not None:
                    hook.remove()
                    gradient = captured[0].grad[0, :, 0].detach().cpu()
                    omitted = valid.clone()
                    omitted[selected] = False
                    assert torch.isfinite(gradient).all() and (gradient[~valid] == 0).all()
                    absolute = gradient[omitted].abs()
                    signal = {"id": row["id"], "omitted_count": int(omitted.sum()), "nonzero_count": int((absolute > 0).sum()),
                        "nonzero_fraction": float((absolute > 0).float().mean()), "omitted_gradient_norm": float(absolute.norm()),
                        "min_abs_omitted_gradient": float(absolute.min()), "max_abs_omitted_gradient": float(absolute.max()), "frozen_lm": True}
                    if signal["nonzero_count"]:
                        qa_gradient_checks[arm] = signal
                    event({"phase": "actual_qa_gradient", "arm": arm, **signal})
                assert all(p.grad is None for p in lm.parameters())
                losses.append(float(loss.detach()))
                if (index + 1) % 8 == 0 or index + 1 == len(order):
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.))
                    assert np.isfinite(gradient_norm)
                    if contributing:
                        optimizer.step()
                        update_steps += 1
                    optimizer.zero_grad()
                    event({"phase": "train", "arm": arm, "epoch": epoch, "examples": index + 1,
                        "loss": float(np.mean(losses[-8:])), "gradient_norm": gradient_norm, "effective_update_rows": contributing})
                    contributing = 0
            adapter.eval()
            dev_rows = []
            with torch.no_grad():
                for row in dev:
                    check()
                    vector, selected, _, _ = memory(adapter, row, condition)
                    packed = pack_prompt(lm, tokenizer, evidence[row["id"]]["transcript"], row["question"], vector, row["answers"][0])
                    dev_rows.append({"id": row["id"], "nll": float(lm(**packed, use_cache=False).loss), "selected": selected,
                        "initial_selected": initial_selection[row["id"]]})
            score = float(np.mean([r["nll"] for r in dev_rows]))
            assert np.isfinite(score)
            if score < best:
                best, best_epoch = score, epoch
                best_state = copy.deepcopy(adapter.cpu().state_dict())
                adapter.cuda()
            torch.save({"state_dict": adapter.cpu().state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "config": config, "training_source_ids": train_ids}, out / (arm + "-last.pt"))
            adapter.cuda()
            (out / (arm + f"-dev-epoch{epoch}.json")).write_text(json.dumps(dev_rows, indent=2), encoding="utf-8")
            event({"phase": "epoch", "arm": arm, "epoch": epoch, "dev_nll": score, "train_nll": float(np.mean(losses))}, save=True)
        assert arm == "uniform" or arm in qa_gradient_checks
        adapter.load_state_dict(best_state)
        adapter.cpu()
        checkpoint = io.BytesIO()
        torch.save({"state_dict": adapter.state_dict(), "best_epoch": best_epoch, "config": config}, checkpoint)
        blobs[arm] = checkpoint.getvalue()
        (out / (arm + "-best.pt")).write_bytes(blobs[arm])
        training_audits[arm] = {"best_epoch": best_epoch, "best_dev_nll": best, "effective_update_examples": update_examples,
            "optimizer_steps": update_steps, "text_only_view_passes": no_audio_views,
            "parameter_update_norms": {n: float((p.detach() - initial[n]).norm()) for n, p in adapter.named_parameters()}}
        event({"phase": "checkpoint", "arm": arm, "sha256": digest(blobs[arm]), **training_audits[arm]}, save=True)
        adapters[arm] = adapter.eval()
        del optimizer
    frozen_checkpoint_sha = {arm: digest(blob) for arm, blob in blobs.items()}
    (out / "training-frozen.json").write_text(json.dumps({"checkpoint_sha256": frozen_checkpoint_sha,
        "fresh_qa_parsed_before_frozen": False, "training_audits": training_audits}, indent=2), encoding="utf-8")
    event({"phase": "training_frozen", "checkpoint_sha256": frozen_checkpoint_sha}, save=True)

    # Unlock fresh content only after final checkpoint selection and freeze.
    qa = [json.loads(line) for line in fresh_qa_bytes.decode("utf-8").splitlines()]
    selection = json.loads(selection_bytes)["records"]
    assert len(qa) == len(selection) == 100
    specs = json.loads(fresh_spec_bytes)
    assert len(specs) == 200
    view_specs = {(int(s["source_id"]), s["condition"]): s for s in specs}
    assert len(view_specs) == 200
    selected = {int(r["source_id"]): r for r in selection}
    assert len(selected) == 100 and {int(r["source_id"]) for r in qa} == set(selected)
    assert not (set(selected) & {int(r["source_id"].rsplit("-", 1)[-1]) for r in historical})
    publishers = {}
    for index, publisher in enumerate(table):
        key = (int(publisher["id"]), Path(publisher["audio"]["path"]).name)
        publishers.setdefault(key, index)
    fresh, targets = [], {}
    for annotation in qa:
        source = int(annotation["source_id"])
        sel = selected[source]
        assert annotation["reference_transcript"] == sel["transcript"]
        index = publishers[(source, sel["original_audio_path"])]
        publisher = table[index]
        assert bool(annotation["answerable"]) == bool(sel["answerable"])
        for condition in ("clean", "noise_10db"):
            check()
            spec = view_specs[(source, condition)]
            assert spec["canonical_source_audio_sha256"] == sel["canonical_source_audio_sha256"]
            assert spec["question_sha256"] == digest(annotation["question"].encode())
            assert spec["original_audio_path"] == sel["original_audio_path"]
            source_id = "fleurs-hi-" + str(source)
            identity = source_id + "-" + condition
            samples, audio_sha, gain = fresh_wave(publisher, condition, sel["canonical_source_audio_sha256"])
            row = {"id": identity, "source_id": source_id, "condition": condition, "question": annotation["question"],
                "duration_s": len(samples) / 16000, "audio_sha256": audio_sha, "shared_gain": gain, "source_row_index": index,
                "input_spec_sha256": digest(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()),
                "answerable": annotation["answerable"], "question_tokens": len(tokenizer(annotation["question"], add_special_tokens=False).input_ids)}
            targets[identity] = {"answers": annotation["answers"], "reference_transcript": annotation["reference_transcript"]}
            stamp = time.monotonic()
            feat = extract(asr, processor, samples)
            torch.cuda.synchronize()
            feat["shared_asr_extract_s"] = time.monotonic() - stamp
            feat["audio_sha256"] = audio_sha
            evidence[identity] = feat
            torch.save(feat, out / (identity + "-features.pt"))
            fresh.append(row)
            event({"phase": "fresh_features", "id": identity, "valid_spans": int(feat["span_mask"].sum()),
                "shared_asr_extract_s": feat["shared_asr_extract_s"]}, save=len(fresh) % 16 == 0)
    assert len(fresh) == 200
    (out / "fresh-input-manifest.json").write_text(json.dumps(fresh, ensure_ascii=False, indent=2), encoding="utf-8")
    donors = derange(fresh, evidence, key=lambda r: (r["duration_s"], r["id"]))
    query_donors = {}
    for condition in ("clean", "noise_10db"):
        ordered = sorted([r for r in fresh if r["condition"] == condition], key=lambda r: (r["question_tokens"], r["source_id"]))
        for index, row in enumerate(ordered):
            query_donors[row["id"]] = ordered[index ^ 1]
    commit()
    del asr, processor, table
    torch.cuda.empty_cache()

    records = []
    arm_specs = (("full", "full", "full"), ("no_uncertainty", "no_uncertainty", "uncertainty_off"),
        ("uniform", "uniform", "uniform"), ("text", None, "full"), ("wrong_audio", "full", "full"),
        ("shuffled_uncertainty", "full", "full"), ("wrong_adapter_question", "full", "full"),
        ("reference_text", None, "full"), ("question_only", None, "full"))
    for label, learned_arm, condition in arm_specs:
        adapter = adapters[learned_arm].cuda() if learned_arm is not None else None
        for row in fresh:
            check()
            donor = donors[row["id"]] if label == "wrong_audio" else None
            query_donor = query_donors[row["id"]] if label == "wrong_adapter_question" else None
            target = targets[row["id"]]
            scaffold = evidence[row["id"]]["transcript"]
            if label == "reference_text":
                scaffold = target["reference_transcript"]
            elif label == "question_only":
                scaffold = ""
            torch.cuda.synchronize()
            path_start = time.monotonic()
            with torch.no_grad():
                if adapter is None:
                    vector = torch.empty(0, lm.config.hidden_size, device="cuda")
                    selected_indices, gates, audit = [], [], {"posterior_moved": 0, "posterior_changed": 0}
                else:
                    vector, selected_indices, gates, audit = memory(adapter, row, condition, donor=donor,
                        query_donor=query_donor, shuffle=label == "shuffled_uncertainty")
                torch.cuda.synchronize()
                adapter_s = time.monotonic() - path_start
                # Generation has no gold-answer argument. NLL is a separate call.
                packing_start = time.monotonic()
                generation = pack_prompt(lm, tokenizer, scaffold, row["question"], vector, answer=None)
                generation.pop("labels")
                generation.pop("position_ids")
                prompt_tokens = generation["inputs_embeds"].shape[1]
                torch.cuda.synchronize()
                packing_s = time.monotonic() - packing_start
                generation_start = time.monotonic()
                output = lm.generate(**generation, max_new_tokens=48, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)[0]
                torch.cuda.synchronize()
                generation_s = time.monotonic() - generation_start
                generated = tokenizer.decode(output, skip_special_tokens=True).strip()
                nll_start = time.monotonic()
                nll_packed = pack_prompt(lm, tokenizer, scaffold, row["question"], vector, target["answers"][0])
                nll = float(lm(**nll_packed, use_cache=False).loss)
                torch.cuda.synchronize()
                nll_s = time.monotonic() - nll_start
                assert np.isfinite(nll)
            has_audio = bool(evidence[row["id"]]["frame_mask"].any())
            shared_s = evidence[row["id"]]["shared_asr_extract_s"] if label not in ("reference_text", "question_only") else 0.
            eos = int(output[-1]) == tokenizer.eos_token_id if len(output) else False
            record = {"arm": label, **row, "answer": generated, "answers": target["answers"],
                "exact_match": generated in target["answers"], "nll": nll, "target_tokens_including_end": int((nll_packed["labels"] != -100).sum()),
                "transcript": scaffold, "shared_asr_extract_s": shared_s, "adapter_s": adapter_s,
                "generation_s": generation_s, "packing_s": packing_s, "nll_s": nll_s, "generation_pipeline_s": shared_s + adapter_s + packing_s + generation_s,
                "packing_included_in_pipeline_s": True, "feature_timings_s": evidence[row["id"]]["feature_timings_s"], "prompt_tokens": prompt_tokens, "generated_tokens": len(output),
                "eos": eos, "capped": len(output) == 48 and not eos, "empty": not bool(generated), "failure": None,
                "selected": selected_indices, "gates": gates, "alignment_available": has_audio,
                "valid_spans": int(evidence[row["id"]]["span_mask"].sum()), "counterfactual_available": not has_audio or donors[row["id"]] is not None,
                "donor": donor["id"] if donor is not None else None,
                "donor_duration_ratio": donor["duration_s"] / row["duration_s"] if donor is not None else None,
                "query_donor": query_donor["id"] if query_donor is not None else None,
                "query_token_difference": query_donor["question_tokens"] - row["question_tokens"] if query_donor is not None else None,
                "query_control_changes": "pooling_and_scalar_gate_jointly" if query_donor is not None else None, **audit}
            records.append(record)
            with (out / "raw-predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            event({"phase": "evaluation", "arm": label, "id": row["id"], "completed_rows": len(records)}, save=len(records) % 25 == 0)
        if adapter is not None:
            adapter.cpu()
        commit()
    assert len(records) == 1800
    result = {"status": "final_complete", "config": config, "asr": ASR, "llm": LLM,
        "checkpoint_sha256": frozen_checkpoint_sha, "gpu_contract": contract, "actual_qa_gradient_checks": qa_gradient_checks,
        "training_audits": training_audits, "training_source_ids": train_ids, "rows": records,
        "elapsed_s": time.monotonic() - start, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "model_load_s": model_load_s, "fresh_input_manifest_sha256": digest((out / "fresh-input-manifest.json").read_bytes()),
        "alignment_counts": {condition: {"total": 100,
            "aligned": sum(r["condition"] == condition and bool(evidence[r["id"]]["frame_mask"].any()) for r in fresh),
            "more_than_k": sum(r["condition"] == condition and int(evidence[r["id"]]["span_mask"].sum()) > 4 for r in fresh)}
            for condition in ("clean", "noise_10db")}}
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "artifact-manifest.json").write_text(json.dumps({"config": config, "checkpoint_sha256": frozen_checkpoint_sha,
        "result_sha256": digest((out / "result.json").read_bytes()), "raw_sha256": digest((out / "raw-predictions.jsonl").read_bytes())}, indent=2), encoding="utf-8")
    event({"phase": "complete", "completed_rows": len(records)}, save=True)
    return result, blobs
