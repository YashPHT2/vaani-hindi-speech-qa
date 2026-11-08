"""Inference-only existing-checkpoint readout and allowed historical cache audit."""
import hashlib
import io
import json
import math
from pathlib import Path
import time
import wave

RUN_ID = "readout-fallback-dev2-v1"
CHECKPOINT_SHA = "8474698e8123d52e4ea96e310691062e49ce76f62d9e924d1976102ffbad2345"
ASR = ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d")
LLM = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
KEYS = ("audio", "frame_mask", "span_mask", "uncertainty", "uncertainty_available")
ARMS = ("A_text", "B_existing_DTW", "C_reported_window", "D_hybrid_fallback", "E_oracle_mismatch", "E_fallback_mismatch")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def run(inputs, historical_bytes, config, commit):
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperProcessor, WhisperForConditionalGeneration
    from connector import AcousticMemory, pack_prompt
    from readout_candidates import grid_windows, reported_windows, pack_windows, cached_windows, replace_geometry_audio, coverage

    assert torch.cuda.is_available() and config["run_id"] == RUN_ID
    assert config["reported_window_frame_bounds"] == [145, 205] and config["full_phrase_extent_verified"] is False
    assert sha(historical_bytes) == config["historical_manifest_sha256"]
    rows = [json.loads(line) for line in historical_bytes.decode("utf-8").splitlines()]
    assert len(rows) == 224 and len({row["id"] for row in rows}) == 224
    assert sum(row["split"] == "test" for row in rows) == 200 and sum(row["split"] == "dev" for row in rows) == 24
    by_id = {row["id"]: row for row in rows}
    assert {item["id"] for item in inputs} == {"fleurs-hi-1806-clean", "fleurs-hi-1806-noise_10db"}
    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.monotonic()
    out = Path("/cache/runs") / RUN_ID
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    def check():
        if time.monotonic()-started > 540:
            raise TimeoutError("Readout work deadline; no automatic retry")
    old_result_path = Path("/cache/runs/matched-headroom-dev24-v1/result.json")
    old_bytes = old_result_path.read_bytes()
    assert sha(old_bytes) == config["headroom_remote_lf_expected_sha256"]
    old_object = json.loads(old_bytes)
    semantic_bytes = json.dumps(old_object, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    assert sha(semantic_bytes) == config["headroom_semantic_sha256"]
    old_rows = old_object["rows"]
    baseline = {(row["id"], row["arm"]): row for row in old_rows}
    inventory = []
    # Only manifest-listed historical files are inspected. Missing remains a row.
    for row in rows:
        check()
        path = Path("/cache/features-v1") / (row["id"] + ".pt")
        entry = {"id": row["id"], "split": row["split"], "condition": row["condition"],
            "duration_s": row["duration_s"], "status": "missing"}
        if path.exists():
            blob = path.read_bytes()
            feat = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
            valid_frames = math.ceil(row["duration_s"]/.02)
            windows, stats = cached_windows(feat, valid_frames)
            valid_count = int((feat["span_mask"] & feat["frame_mask"].any(-1)).sum())
            entry.update(status="available", availability_kind="present_nonempty" if valid_count else "present_empty",
                expected_selected_count_under_K4_policy=min(4,valid_count), expected_selected_count_is_observed=False,
                cache_sha256=sha(blob), transcript=feat["transcript"], span_audit=feat["span_audit"],
                candidate_windows=windows, span_mask=feat["span_mask"].tolist(),
                valid_span_count=int((feat["span_mask"] & feat["frame_mask"].any(-1)).sum()),
                frame_counts=feat["frame_mask"].sum(-1).tolist(), uncertainty_available=feat["uncertainty_available"].tolist(), **stats)
            prior = baseline.get((row["id"], "D_existing_adapter"))
            entry["old_selected_indices"] = prior["selected_spans"] if prior else None
            del feat, blob
        inventory.append(entry)
    inventory_path = out / "historical-cache-inventory.json"
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    commit()
    checkpoint = Path("/cache/runs/paired-pilot-st-seed17-v1-r1/full-best.pt")
    checkpoint_bytes = checkpoint.read_bytes()
    assert sha(checkpoint_bytes) == config["pilot_checkpoint_sha256"] == CHECKPOINT_SHA
    processor = WhisperProcessor.from_pretrained(ASR[0], revision=ASR[1])
    asr = WhisperForConditionalGeneration.from_pretrained(ASR[0], revision=ASR[1], torch_dtype=torch.float16,
        attn_implementation="eager", use_safetensors=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(LLM[0], revision=LLM[1])
    lm = AutoModelForCausalLM.from_pretrained(LLM[0], revision=LLM[1], torch_dtype=torch.float16,
        attn_implementation="sdpa", use_safetensors=True).cuda().eval()
    adapter = AcousticMemory().cuda().eval()
    adapter.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)["state_dict"], strict=True)
    for model in (asr, lm, adapter):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    initial_state = {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}
    torch.cuda.synchronize()
    load_seconds = time.monotonic()-started
    assert sum(p.numel() for p in adapter.parameters()) == 305281
    def encode(raw, expected):
        assert sha(raw) == expected
        with wave.open(io.BytesIO(raw), "rb") as source:
            assert source.getnchannels() == 1 and source.getsampwidth() == 2 and source.getframerate() == 16000
            count = source.getnframes()
            samples = np.frombuffer(source.readframes(count), dtype="<i2").astype(np.float32)/32768
        assert count == round(13.68*16000) and np.isfinite(samples).all()
        features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        stamp = time.monotonic()
        hidden = asr.model.encoder(features, return_dict=True).last_hidden_state[0].float()
        torch.cuda.synchronize()
        return hidden, math.ceil(count/16000/.02), time.monotonic()-stamp
    predictions, guards, integrity_guards = [], [], []
    with torch.inference_mode():
        # Match the previous headroom unscored one-token text-only warm-up.
        first_dev = next(row for row in rows if row["split"] == "dev")
        warm = pack_prompt(lm, tokenizer, "", first_dev["question"], torch.empty(0,1536,device="cuda"))
        warm.pop("labels"); warm.pop("position_ids")
        lm.generate(**warm, max_new_tokens=1, do_sample=False, use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        for item in inputs:
            check()
            row = by_id[item["id"]]
            assert row["source_id"] == "fleurs-hi-1806" and row["split"] == "dev"
            assert row["audio_sha256"] == item["audio_sha256"]
            assert item["donor_id"] == "fleurs-hi-1868-"+item["condition"]
            assert by_id[item["donor_id"]]["audio_sha256"] == item["donor_audio_sha256"]
            path = Path("/cache/features-v1")/(row["id"]+".pt")
            cache_bytes = path.read_bytes()
            cached = torch.load(io.BytesIO(cache_bytes), map_location="cpu", weights_only=False)
            actual, valid_frames, actual_encode_s = encode(item["audio_bytes"], item["audio_sha256"])
            donor, donor_valid, donor_encode_s = encode(item["donor_bytes"], item["donor_audio_sha256"])
            assert valid_frames == donor_valid == 684
            old_windows, old_stats = cached_windows(cached, valid_frames)
            dtw_valid = bool((cached["span_mask"] & cached["frame_mask"].any(-1)).any())
            oracle = pack_windows(actual, valid_frames, reported_windows(valid_frames))
            dtw = {key: cached[key].cuda() for key in KEYS}
            hybrid = dtw if dtw_valid else pack_windows(actual, valid_frames, grid_windows(valid_frames))
            oracle_donor = pack_windows(donor, valid_frames, reported_windows(valid_frames))
            hybrid_donor = replace_geometry_audio(cached, donor.cpu(), valid_frames) if dtw_valid else pack_windows(donor, valid_frames, grid_windows(valid_frames))
            payloads = {"B_existing_DTW": dtw, "C_reported_window": oracle, "D_hybrid_fallback": hybrid,
                "E_oracle_mismatch": oracle_donor, "E_fallback_mismatch": hybrid_donor}
            geometries = {"A_text": [], "B_existing_DTW": old_windows, "C_reported_window": oracle["candidate_windows"],
                "D_hybrid_fallback": old_windows if dtw_valid else hybrid["candidate_windows"],
                "E_oracle_mismatch": oracle["candidate_windows"], "E_fallback_mismatch": old_windows if dtw_valid else hybrid["candidate_windows"]}
            question_ids = tokenizer(row["question"], return_tensors="pt").input_ids.cuda()
            question = lm.get_input_embeddings()(question_ids).float()
            question_mask = torch.ones_like(question_ids,dtype=torch.bool)
            per_arm = {}
            for arm in ARMS:
                check()
                stamp = time.monotonic()
                selected, gates = [], []
                memory = torch.empty(0,1536,device="cuda")
                if arm != "A_text":
                    data = {key: payloads[arm][key].cuda() for key in KEYS}
                    memories, indices, amplitude = adapter(**data, question=question, question_mask=question_mask, condition="full")
                    memory, selected, gates = memories[0], indices[0].tolist(), amplitude[0].tolist()
                prompt = pack_prompt(lm, tokenizer, cached["transcript"], row["question"], memory, answer=None)
                assert prompt.pop("labels") is None
                prompt.pop("position_ids")
                torch.cuda.synchronize()
                packing_adapter_s = time.monotonic()-stamp
                prompt_sha = sha(prompt["inputs_embeds"].detach().cpu().contiguous().numpy().tobytes())
                stamp=time.monotonic()
                generated=lm.generate(**prompt,max_new_tokens=48,do_sample=False,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)[0]
                torch.cuda.synchronize()
                generation_s=time.monotonic()-stamp
                answer=tokenizer.decode(generated,skip_special_tokens=True).strip()
                stamp=time.monotonic()
                target=pack_prompt(lm,tokenizer,cached["transcript"],row["question"],memory,answer=row["answers"][0])
                nll=float(lm(**target,use_cache=False).loss)
                torch.cuda.synchronize()
                nll_s=time.monotonic()-stamp
                assert math.isfinite(nll)
                embed_norm=lm.get_input_embeddings()(question_ids).float().norm(dim=-1)
                norm=memory.float().norm(dim=-1)
                windows=geometries[arm]
                avail=payloads[arm]["uncertainty_available"].cpu().tolist() if arm!="A_text" else [[]]
                record={"id":row["id"],"source_id":row["source_id"],"condition":row["condition"],"arm":arm,
                    "question":row["question"],"transcript_used":cached["transcript"],"answer":answer,"answers":row["answers"],
                    "audio_sha256":row["audio_sha256"],"feature_cache_sha256":sha(cache_bytes),"donor_id":item["donor_id"] if arm.startswith("E_") else None,
                    "candidate_windows_frames":windows,"coverage":old_stats if arm in ("B_existing_DTW","D_hybrid_fallback","E_fallback_mismatch") and dtw_valid else coverage(windows,valid_frames),
                    "selected_indices":selected,"selected_windows_frames":[windows[i] for i in selected],"gates":gates,"K":len(memory),
                    "uncertainty_available":avail,"fallback_used":not dtw_valid and arm in ("D_hybrid_fallback","E_fallback_mismatch"),
                    "reported_window_privileged":arm in ("C_reported_window","E_oracle_mismatch"),"full_phrase_extent_verified":False,
                    "memory_token_norms":norm.tolist(),"question_embedding_norm_mean":float(embed_norm.mean()),
                    "memory_to_question_embedding_norm_ratio":float(norm.mean()/embed_norm.mean()) if len(norm) else None,
                    "prompt_embedding_sha256":prompt_sha,"prompt_tokens":prompt["inputs_embeds"].shape[1],"nll":nll,
                    "target_tokens_including_end":int((target["labels"]!=-100).sum()),"generated_token_ids":generated.tolist(),"generated_tokens":len(generated),
                    "finish":"length" if len(generated)==48 and int(generated[-1])!=tokenizer.eos_token_id else "stop",
                    "packing_adapter_s":packing_adapter_s,"generation_s":generation_s,"nll_s":nll_s,"actual_encoder_s":actual_encode_s,"donor_encoder_s":donor_encode_s}
                predictions.append(record);per_arm[arm]=record
                with (out/"predictions.jsonl").open("a",encoding="utf-8") as stream:stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                commit()
            if dtw_valid:
                b,d=per_arm["B_existing_DTW"],per_arm["D_hybrid_fallback"]
                assert all(torch.equal(dtw[key],hybrid[key]) for key in KEYS)
                integrity_guards.append({"id":row["id"],"guard":"valid_DTW_D_equals_B",
                    "passed":b["prompt_embedding_sha256"]==d["prompt_embedding_sha256"] and b["generated_token_ids"]==d["generated_token_ids"]
                        and b["selected_indices"]==d["selected_indices"]})
            else:
                a,b=per_arm["A_text"],per_arm["B_existing_DTW"]
                integrity_guards.append({"id":row["id"],"guard":"empty_DTW_B_equals_A",
                    "passed":a["K"]==b["K"]==0 and a["prompt_embedding_sha256"]==b["prompt_embedding_sha256"] and a["generated_token_ids"]==b["generated_token_ids"]})
            integrity_guards.append({"id":row["id"],"guard":"expected_clean_nonempty_noisy_empty_DTW",
                "passed":dtw_valid==(row["condition"]=="clean")})
            for label,old_arm in (("A_text","B_asr_text"),("B_existing_DTW","D_existing_adapter")):
                prior=baseline[(row["id"],old_arm)];new=per_arm[label]
                guards.append({"id":row["id"],"arm":label,"prior_arm":old_arm,"answer_match":new["answer"]==prior["answer"],
                    "transcript_match":new["transcript_used"]==prior["transcript_used"],"selected_match":new["selected_indices"]==prior["selected_spans"],
                    "memory_count_match":new["K"]==prior["memory_tokens"],"nll_delta":new["nll"]-prior["nll"]})
            assert sha(path.read_bytes())==sha(cache_bytes),"Cached baseline modified"
    assert len(predictions)==12 and sha(checkpoint.read_bytes())==CHECKPOINT_SHA
    assert all(torch.equal(value.detach().cpu(),initial_state[name]) for name,value in adapter.state_dict().items())
    result={"status":"readout_fallback_complete","config":config,"rows":predictions,"baseline_guards":guards,
        "integrity_guards":integrity_guards,"integrity_guard_mismatch":any(not g["passed"] for g in integrity_guards),
        "baseline_guard_mismatch":any(not all(g[k] for k in ("answer_match","transcript_match","selected_match","memory_count_match")) for g in guards),
        "cache_inventory":inventory,"cache_inventory_sha256":sha(inventory_path.read_bytes()),
        "cache_available":sum(item["status"]=="available" for item in inventory),"cache_missing":sum(item["status"]=="missing" for item in inventory),
        "checkpoint_before_sha256":sha(checkpoint_bytes),"checkpoint_after_sha256":sha(checkpoint.read_bytes()),"training_updates":0,"fresh_test_loaded":False,
        "load_seconds_including_inventory":load_seconds,"gpu":torch.cuda.get_device_name(),"peak_cuda_bytes":torch.cuda.max_memory_allocated(),"elapsed_s":time.monotonic()-started,
        "interpretation":"Single historical source, two acoustic views; fixed existing checkpoint; reported window is privileged and uncertain; no general QA gain or training claim"}
    result_path=out/"result.json";result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"artifact-manifest.json").write_text(json.dumps({"result_sha256":sha(result_path.read_bytes()),"inventory_sha256":sha(inventory_path.read_bytes()),"predictions_sha256":sha((out/"predictions.jsonl").read_bytes())},indent=2),encoding="utf-8")
    commit();return result
