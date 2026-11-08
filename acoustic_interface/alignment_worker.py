"""Historical matched connector training with an optional teacher KL term."""
import io
import json
import math
from pathlib import Path
import random
import time
import wave

from alignment_objective import answer_distribution, matched_loss
from scale_worker import ASR, CHECKPOINT_SHA, KEYS, LLM, canonical, sha

RUN_ID = "alignment-ce-kl-v1"
SCHEDULE = {"optimizer": "AdamW", "lr": .001, "weight_decay": .01, "gradient_clip": 1.,
    "epochs": 4, "batch": 8, "updates_per_arm": 32, "seed": 17, "selection": "final_update"}
DISTILLATION = {"temperature": 2, "lambda": .5, "direction": "teacher_to_student",
    "target": "identical_first_alias_tokens_including_assistant_eos",
    "reduction": "mean_target_tokens_per_view_then_mean_all_eight_views"}
DONOR_ALIGNMENT = "linear_interpolation_of_complete_valid_encoder_sequence_align_corners_false"
EVAL_ARMS = ("text", "ce_real", "ce_wrong", "kl_real", "kl_wrong")
RESOURCE_LIMITS = {"gpu": "A10", "gpu_count": 1, "cpu": 2, "memory_gib": 16,
    "function_s": 1800, "startup_s": 300, "queue_s": 120, "retries": 0}


def validate_config(config):
    assert config["run_id"] == RUN_ID and config["pilot_checkpoint_sha256"] == CHECKPOINT_SHA
    assert config["seed"] == 17 and config["max_new_tokens"] == 48
    assert config["fit"] == {"optimizer": "AdamW", "lr": .001, "weight_decay": .01,
        "clip_norm": 1., "batch": 8, "epochs": 4, "updates": 32}
    assert config["temperature"] == 2 and config["kl_lambda"] == .5
    assert config["minimum_answerable_teacher_sources"] == 4 and config["memory_scale"] == 1
    assert config["arms"] == list(EVAL_ARMS) and config["fresh_test_loaded"] is False
    assert config["resource_limits"] == RESOURCE_LIMITS and config["reservation_usd"] == 2
    assert math.isclose(config["worst_case_cost_usd"], 2102 * .00036772 + .5, abs_tol=1e-9)


def split_rows(historical_bytes, config):
    """Validate the exact historical views and the predetermined donor mapping."""
    assert sha(historical_bytes) == config["historical_manifest_sha256"]
    rows = [json.loads(line) for line in historical_bytes.decode("utf-8").splitlines()]
    assert len(rows) == 224 and len({r["id"] for r in rows}) == 224
    train_ids = set(config["train_source_ids"])
    dev_ids = set(config["primary_source_ids"])
    assert len(train_ids) == 32 and len(dev_ids) == 11 and not train_ids & dev_ids
    assert "fleurs-hi-1806" not in train_ids | dev_ids
    train = [r for r in rows if r["source_id"] in train_ids]
    dev = [r for r in rows if r["source_id"] in dev_ids]
    assert len(train) == 64 and len(dev) == 22
    assert all(r["split"] == "test" for r in train) and all(r["split"] == "dev" for r in dev)
    for group, ids in ((train, train_ids), (dev, dev_ids)):
        for source in ids:
            pair = [r for r in group if r["source_id"] == source]
            assert len(pair) == 2 and {r["condition"] for r in pair} == {"clean", "noise_10db"}
            for key in ("transcript", "question", "answers", "answerable"):
                assert pair[0][key] == pair[1][key], f"Source views disagree on {key}"
    donors = config["donor_by_view"]
    assert config["donor_alignment"] == DONOR_ALIGNMENT and set(donors) == {r["id"] for r in dev}
    for row in dev:
        eligible = [d for d in train if d["condition"] == row["condition"] and d["transcript"] != row["transcript"]]
        donor = min(eligible, key=lambda d: (abs(d["duration_s"] - row["duration_s"]), int(d["source_id"].rsplit("-", 1)[-1])))
        assert donor["id"] == donors[row["id"]], "Frozen donor is not the prescribed duration match"
    return train, dev


def run(inputs, historical_bytes, config, commit):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
    from connector import AcousticMemory, pack_prompt
    from readout_candidates import cached_windows, grid_windows, pack_windows, replace_geometry_audio
    from score import norm

    assert torch.cuda.is_available()
    validate_config(config)
    train, dev = split_rows(historical_bytes, config)
    all_views = {r["id"]: r for r in train + dev}
    waves = {item["id"]: item for item in inputs}
    assert len(inputs) == len(waves) == 86 and set(waves) == set(all_views)
    assert set(config["cache_sha256_by_view"]) == set(all_views)
    for identity, item in waves.items():
        assert sha(item["audio_bytes"]) == item["audio_sha256"] == all_views[identity]["audio_sha256"]

    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.monotonic()
    out = Path("/cache/runs") / RUN_ID
    out.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        (out / name).write_bytes(payload)
        return sha(payload)

    def append(name, row):
        with (out / name).open("ab") as stream:
            stream.write(canonical(row) + b"\n")

    def check():
        if time.monotonic() - started > 1740:
            raise TimeoutError("Alignment work deadline. Partial artifacts retained.")

    write("config.json", config)
    checkpoint = Path("/cache/runs/paired-pilot-st-seed17-v1-r1/full-best.pt")
    checkpoint_bytes = checkpoint.read_bytes()
    assert sha(checkpoint_bytes) == CHECKPOINT_SHA == config["pilot_checkpoint_sha256"]
    parent_state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)["state_dict"]
    tokenizer = AutoTokenizer.from_pretrained(LLM[0], revision=LLM[1], local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(LLM[0], revision=LLM[1], torch_dtype=torch.float16,
        attn_implementation="sdpa", use_safetensors=True, local_files_only=True).cuda().eval()
    processor = WhisperProcessor.from_pretrained(ASR[0], revision=ASR[1], local_files_only=True)
    asr = WhisperForConditionalGeneration.from_pretrained(ASR[0], revision=ASR[1], torch_dtype=torch.float16,
        attn_implementation="eager", use_safetensors=True, local_files_only=True).cuda().eval()
    for model in (lm, asr):
        model.requires_grad_(False)
    frozen_versions = [[p._version for p in model.parameters()] for model in (lm, asr)]
    empty = torch.empty(0, 1536, device="cuda")

    def check_backbones():
        for model, versions in zip((lm, asr), frozen_versions):
            assert not model.training
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            assert [p._version for p in model.parameters()] == versions

    def tensor_hash(value):
        return sha(value.detach().cpu().contiguous().numpy().tobytes())

    def decode(identity):
        with wave.open(io.BytesIO(waves[identity]["audio_bytes"]), "rb") as source:
            assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
            count = source.getnframes()
            pcm = source.readframes(count)
        assert len(pcm) == 2 * count and 0 < count <= 480000
        assert count == round(all_views[identity]["duration_s"] * 16000)
        return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768

    def encode(identity):
        samples = decode(identity)
        features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        hidden = asr.model.encoder(features, return_dict=True).last_hidden_state[0].float()
        count = math.ceil(all_views[identity]["duration_s"] / .02)
        assert 0 < count <= len(hidden)
        return hidden[:count]

    def generate(scaffold, question, memory):
        prompt = pack_prompt(lm, tokenizer, scaffold, question, memory, answer=None)
        assert prompt.pop("labels") is None
        prompt.pop("position_ids")
        torch.cuda.synchronize()
        tick = time.monotonic()
        ids = lm.generate(**prompt, max_new_tokens=48, do_sample=False, use_cache=True,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)[0]
        torch.cuda.synchronize()
        tokens = ids.tolist()
        eos = bool(tokens) and tokens[-1] == tokenizer.eos_token_id
        return {"answer": tokenizer.decode(ids, skip_special_tokens=True).strip(), "generated_token_ids": tokens,
            "generated_tokens": len(tokens), "generation_s": time.monotonic() - tick,
            "eos": eos, "finish": "length" if len(tokens) == 48 and not eos else "stop"}

    def target_logits(scaffold, row, memory):
        prompt = pack_prompt(lm, tokenizer, scaffold, row["question"], memory, answer=row["answers"][0])
        labels = prompt.pop("labels")
        logits = lm(**prompt, use_cache=False).logits
        selected, ids = answer_distribution(logits, labels)
        assert ids[-1].item() == tokenizer.eos_token_id and int(ids.eq(tokenizer.eos_token_id).sum()) == 1
        return selected, ids

    teacher_rows = []
    teacher_cache = {}
    representatives = [r for r in train if r["condition"] == "clean"]
    assert len(representatives) == 32
    with torch.no_grad():
        for row in representatives:
            check()
            generated = generate(row["transcript"], row["question"], empty)
            answer = norm(generated["answer"])
            aliases = [alias for alias in row["answers"] if norm(alias) == answer]
            eligible = bool(answer) and bool(aliases)
            target_ids = tokenizer(row["answers"][0] + "<|im_end|>", add_special_tokens=False).input_ids
            record = {"source_id": row["source_id"], "id": row["id"], "question": row["question"],
                "answers": row["answers"], "answerable": row["answerable"], "eligible": eligible,
                "normalized_answer": answer, "matched_aliases": aliases, "target_token_ids": target_ids, **generated}
            if eligible:
                logits, ids = target_logits(row["transcript"], row, empty)
                assert torch.isfinite(logits).all()
                cached_logits = logits.detach().cpu().contiguous()
                cached_ids = ids.detach().cpu()
                assert cached_ids.tolist() == target_ids
                teacher_cache[row["source_id"]] = (cached_logits, cached_ids)
                cache_path = out / (row["source_id"] + "-teacher-logits.pt")
                torch.save({"logits": cached_logits, "target_ids": cached_ids}, cache_path)
                record["teacher_logits_sha256"] = sha(cache_path.read_bytes())
                record["teacher_logits_tensor_sha256"] = tensor_hash(cached_logits)
            teacher_rows.append(record)
            append("teacher-rows.jsonl", record)
        check_backbones()
    eligible_answerable = sum(record["eligible"] and record["answerable"] for record in teacher_rows)
    eligible_unanswerable = len(teacher_cache) - eligible_answerable
    eligible_tokens = sum(len(record["target_token_ids"]) for record in teacher_rows if record["eligible"])
    teacher_eligibility = {"eligible_sources": len(teacher_cache), "total_sources": 32,
        "eligible_views": 2 * len(teacher_cache), "total_views": 64,
        "eligible_answerable_sources": eligible_answerable, "eligible_unanswerable_sources": eligible_unanswerable,
        "eligible_unanswerable_abstention_sources": sum(record["eligible"] and not record["answerable"]
            and record["normalized_answer"] in {norm("अज्ञात"), norm("unknown")} for record in teacher_rows),
        "answerable_sources": sum(record["answerable"] for record in teacher_rows),
        "minimum_eligible_answerable_sources": 4, "passed": eligible_answerable >= 4,
        "eligible_supervised_tokens": 2 * eligible_tokens,
        "total_supervised_tokens": 2 * sum(len(record["target_token_ids"]) for record in teacher_rows),
        "eligible_views_by_condition": {"clean": len(teacher_cache), "noise_10db": len(teacher_cache)},
        "eligible_supervised_tokens_by_condition": {"clean": eligible_tokens, "noise_10db": eligible_tokens},
        "rule": "frozen score.py norm, nonempty exact match against any alias"}
    write("teacher-eligibility.json", teacher_eligibility)
    commit()
    if not teacher_eligibility["passed"]:
        check_backbones()
        assert sha(checkpoint.read_bytes()) == CHECKPOINT_SHA
        result = {"status": "alignment_teacher_gate_stopped", "config": config,
            "teacher_rows": teacher_rows, "teacher_eligibility": teacher_eligibility, "fit_trace": {}, "eval_rows": [],
            "schedule": SCHEDULE, "distillation": DISTILLATION, "checkpoint_sha256": {}, "fresh_test_loaded": False,
            "weight_invariance": {"initial_checkpoint_sha256": CHECKPOINT_SHA, "parent_checkpoint_unchanged": True,
                "whisper_qwen_parameter_versions_unchanged": True, "whisper_qwen_gradients_none": True},
            "elapsed_s": time.monotonic() - started}
        result_sha = write("result.json", result)
        write("artifact-manifest.json", {"result_sha256": result_sha,
            "teacher_rows_sha256": sha((out / "teacher-rows.jsonl").read_bytes())})
        commit()
        return result

    features_by_view = {}
    donor_features = {}
    questions = {}
    memory_metadata = {}
    donor_encoder_cache = {}
    with torch.no_grad():
        for row in train + dev:
            check()
            identity = row["id"]
            cache_path = Path("/cache/features-v1") / (identity + ".pt")
            payload = cache_path.read_bytes()
            cache_hash = sha(payload)
            assert cache_hash == config["cache_sha256_by_view"][identity]
            cached = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            count = math.ceil(row["duration_s"] / .02)
            has_dtw = bool((cached["span_mask"] & cached["frame_mask"].any(-1)).any())
            if has_dtw:
                windows, _ = cached_windows(cached, count)
                actual = {key: cached[key] for key in KEYS}
            else:
                windows = grid_windows(count)
                packed = pack_windows(encode(identity), count, windows)
                actual = {key: packed[key].cpu() for key in KEYS}
            features_by_view[identity] = actual
            qids = tokenizer(row["question"], return_tensors="pt").input_ids.cuda()
            questions[identity] = (lm.get_input_embeddings()(qids).float().detach(), torch.ones_like(qids, dtype=torch.bool))
            meta = {"asr_transcript": cached["transcript"], "feature_cache_sha256": cache_hash,
                "fallback_used": not has_dtw, "candidate_windows_frames": windows, "valid_frames": count,
                "actual_audio_sha256": row["audio_sha256"], "feature_audio_sha256": tensor_hash(actual["audio"]),
                "question_embedding_sha256": tensor_hash(questions[identity][0]),
                "recipient_geometry_sha256": sha(canonical({"windows": windows,
                    **{key: tensor_hash(actual[key]) for key in KEYS if key != "audio"}}))}
            if identity in config["donor_by_view"]:
                donor_id = config["donor_by_view"][identity]
                if donor_id not in donor_encoder_cache:
                    donor_encoder_cache[donor_id] = encode(donor_id).cpu()
                donor = donor_encoder_cache[donor_id]
                if len(donor) != count:
                    resized = F.interpolate(donor.T[None], size=count, mode="linear", align_corners=False)[0].T
                else:
                    resized = donor
                if has_dtw:
                    wrong = replace_geometry_audio(cached, resized, count)
                else:
                    packed = pack_windows(resized, count, windows)
                    wrong = {key: packed[key] for key in KEYS}
                assert all(torch.equal(actual[key], wrong[key]) for key in KEYS if key != "audio")
                donor_features[identity] = wrong
                meta.update(donor_id=donor_id, donor_audio_sha256=all_views[donor_id]["audio_sha256"],
                    donor_valid_frames=len(donor), donor_resampling_ratio=count / len(donor),
                    donor_duration_ratio=all_views[donor_id]["duration_s"] / row["duration_s"],
                    donor_geometry_identical=True, donor_feature_audio_sha256=tensor_hash(wrong["audio"]))
            memory_metadata[identity] = meta
    write("memory-metadata.json", memory_metadata)
    commit()

    def route(adapter, identity, inputs):
        question, qmask = questions[identity]
        vectors, selected, gates = adapter(**{key: inputs[key].cuda() for key in KEYS},
            question=question, question_mask=qmask, condition="full")
        assert 1 <= len(vectors[0]) <= 4 and torch.isfinite(vectors[0]).all()
        assert torch.isfinite(gates[0]).all()
        return vectors[0], selected[0], gates[0]

    orders = []
    for epoch in range(SCHEDULE["epochs"]):
        order = list(train)
        random.Random(17 + epoch).shuffle(order)
        orders.append(order)
    orders_sha = write("training-orders.json", [[r["id"] for r in order] for order in orders])
    fit_trace, adapters, checkpoints, arm_proofs = {}, {}, {}, {}
    for arm in ("ce", "kl"):
        torch.manual_seed(17)
        adapter = AcousticMemory().cuda()
        adapter.load_state_dict(parent_state, strict=True)
        adapter.train()
        assert sum(p.numel() for p in adapter.parameters()) == 305281
        assert all(torch.equal(value.cpu(), parent_state[name]) for name, value in adapter.state_dict().items())
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001, weight_decay=.01)
        assert not optimizer.state
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in adapter.parameters()}
        trace = []
        first_kl_gradient = None
        for epoch, order in enumerate(orders):
            for start in range(0, len(order), 8):
                check()
                optimizer.zero_grad(set_to_none=True)
                batch = order[start:start + 8]
                assert len(batch) == 8
                ce_values, kl_values, total_values = [], [], []
                eligible_views = 0
                for row in batch:
                    identity = row["id"]
                    memory, _, _ = route(adapter, identity, features_by_view[identity])
                    student, ids = target_logits(memory_metadata[identity]["asr_transcript"], row, memory)
                    teacher = teacher_cache.get(row["source_id"]) if arm == "kl" else None
                    if teacher is not None:
                        teacher_logits, teacher_ids = (value.cuda() for value in teacher)
                        eligible_views += 1
                    else:
                        teacher_logits = teacher_ids = None
                    loss, ce, kl = matched_loss(student, ids, teacher_logits, teacher_ids)
                    assert torch.isfinite(loss) and torch.isfinite(ce)
                    if teacher is not None and first_kl_gradient is None:
                        parts = torch.autograd.grad(kl, tuple(adapter.parameters()), retain_graph=True, allow_unused=True)
                        norms = [float(part.detach().float().norm()) for part in parts if part is not None]
                        assert norms and all(math.isfinite(value) for value in norms)
                        kl_gradient_norm = math.sqrt(sum(n * n for n in norms))
                        assert kl_gradient_norm > 0, "KL-only signal did not reach the connector on the first eligible view"
                        first_kl_gradient = {"id": identity, "connector_gradient_norm": kl_gradient_norm}
                    (loss / 8).backward()
                    ce_values.append(float(ce.detach()))
                    kl_values.append(float(kl.detach()))
                    total_values.append(float(loss.detach()))
                gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                check_backbones()
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in adapter.parameters())
                update = {"arm": arm, "epoch": epoch, "update": len(trace) + 1,
                    "view_ids": [r["id"] for r in batch], "kl_eligible_views_used": eligible_views,
                    "ce": sum(ce_values) / 8, "kl_unweighted": sum(kl_values) / 8,
                    "kl_weighted_temperature_squared": 2 * sum(kl_values) / 8, "loss": sum(total_values) / 8,
                    "gradient_norm_before_clip": float(gradient), "gradient_clipped": float(gradient) > 1.}
                trace.append(update)
                append("fit-trace.jsonl", update)
            torch.save({"state_dict": {n: v.detach().cpu() for n, v in adapter.state_dict().items()},
                "optimizer": optimizer.state_dict(), "epoch": epoch, "updates": len(trace),
                "training_orders_sha256": orders_sha}, out / (arm + "-progress.pt"))
            commit()
        assert len(trace) == 32 and any(row["gradient_norm_before_clip"] > 0 for row in trace)
        assert (first_kl_gradient is not None) == (arm == "kl")
        final_state = {n: v.detach().cpu() for n, v in adapter.state_dict().items()}
        optimizer_steps = sorted({int(state["step"].item()) for state in optimizer.state.values()})
        assert optimizer_steps == [32]
        assert any(not torch.equal(final_state[name], parent_state[name]) for name, _ in adapter.named_parameters())
        for name, _ in adapter.named_buffers():
            assert torch.equal(final_state[name], parent_state[name]), "Frozen uncertainty buffers changed"
        path = out / (arm + "-final.pt")
        torch.save({"state_dict": final_state, "arm": arm, "updates": 32, "initial_checkpoint_sha256": CHECKPOINT_SHA,
            "training_orders_sha256": orders_sha, "seed": 17}, path)
        checkpoints[arm] = sha(path.read_bytes())
        fit_trace[arm] = trace
        arm_proofs[arm] = {"fresh_optimizer": True, "connector_only_optimizer": True,
            "optimizer_step_counts": optimizer_steps, "training_orders_sha256": orders_sha,
            "first_kl_only_connector_gradient": first_kl_gradient,
            "parameter_update_norms": {n: float((final_state[n] - parent_state[n]).norm()) for n, _ in adapter.named_parameters()}}
        optimizer.zero_grad(set_to_none=True)
        adapters[arm] = adapter.eval().requires_grad_(False)
        commit()

    evaluation = []
    with torch.no_grad():
        for row in dev:
            identity = row["id"]
            scaffold = memory_metadata[identity]["asr_transcript"]
            for label in EVAL_ARMS:
                check()
                if label == "text":
                    memory, selected, gates = empty, [], []
                else:
                    arm, condition = label.split("_")
                    payload = features_by_view[identity] if condition == "real" else donor_features[identity]
                    memory, indices, amplitudes = route(adapters[arm], identity, payload)
                    selected, gates = indices.tolist(), amplitudes.tolist()
                generated = generate(scaffold, row["question"], memory)
                logits, ids = target_logits(scaffold, row, memory)
                _, nll, _ = matched_loss(logits, ids)
                assert torch.isfinite(nll)
                record = {"id": identity, "source_id": row["source_id"], "condition": row["condition"], "arm": label,
                    "question": row["question"], "answers": row["answers"], "transcript_used": scaffold,
                    "nll": float(nll), "target_token_count_including_eos": len(ids),
                    "selected_indices": selected, "selected_windows_frames": [memory_metadata[identity]["candidate_windows_frames"][i] for i in selected],
                    "gates": gates, "memory_slots": len(memory), "memory_sha256": tensor_hash(memory),
                    "donor_id": config["donor_by_view"][identity] if label.endswith("_wrong") else None,
                    **generated}
                evaluation.append(record)
                append("predictions.jsonl", record)
            commit()
    assert len(evaluation) == 110
    check_backbones()
    assert sha(checkpoint.read_bytes()) == CHECKPOINT_SHA
    for arm, adapter in adapters.items():
        path = out / (arm + "-final.pt")
        saved_bytes = path.read_bytes()
        assert sha(saved_bytes) == checkpoints[arm]
        saved = torch.load(io.BytesIO(saved_bytes), map_location="cpu", weights_only=True)["state_dict"]
        assert not adapter.training and all(p.grad is None and not p.requires_grad for p in adapter.parameters())
        assert all(torch.equal(value.cpu(), saved[name]) for name, value in adapter.state_dict().items())
    for record in teacher_rows:
        if record["eligible"]:
            logits, ids = teacher_cache[record["source_id"]]
            assert not logits.requires_grad and logits.grad_fn is None
            assert ids.tolist() == record["target_token_ids"]
            assert tensor_hash(logits) == record["teacher_logits_tensor_sha256"]
    result = {"status": "alignment_complete", "config": config, "teacher_rows": teacher_rows,
        "teacher_eligibility": teacher_eligibility, "fit_trace": fit_trace, "eval_rows": evaluation,
        "schedule": SCHEDULE, "distillation": DISTILLATION, "checkpoint_sha256": checkpoints,
        "memory_metadata": memory_metadata, "weight_invariance": {"initial_checkpoint_sha256": CHECKPOINT_SHA,
            "parent_checkpoint_unchanged": True, "whisper_qwen_parameter_versions_unchanged": True,
            "whisper_qwen_gradients_none": True, "teacher_cache_unchanged": True,
            "trained_connectors_unchanged_during_evaluation": True, "arm_proofs": arm_proofs},
        "fresh_test_loaded": False, "gpu": torch.cuda.get_device_name(),
        "elapsed_s": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    result_sha = write("result.json", result)
    write("artifact-manifest.json", {"result_sha256": result_sha, "checkpoint_sha256": checkpoints,
        "training_orders_sha256": orders_sha, "predictions_sha256": sha((out / "predictions.jsonl").read_bytes()),
        "teacher_rows_sha256": sha((out / "teacher-rows.jsonl").read_bytes()),
        "fit_trace_sha256": sha((out / "fit-trace.jsonl").read_bytes())})
    commit()
    return result
