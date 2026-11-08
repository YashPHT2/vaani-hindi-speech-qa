"""Launch-once guarded runner. No authorization or human boundaries are invented."""
import hashlib
import json
from pathlib import Path
import time

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("vaani-localized-omission-sensitivity")
volume = modal.Volume.from_name("vaani-acoustic-interface-cache-v1", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4",
        "scipy==1.15.2", "soundfile==0.13.1", "accelerate==1.6.0",
        "safetensors==0.5.3", "pyarrow==19.0.1")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/app", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "localization_worker.py", "/app/localization_worker.py"))


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def checked_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_boundary_capture(raw, boundary, provenance, config, bound_files):
    """Keep complete-phrase validation distinct from the explicit chat capture."""
    status = provenance["validation_status"]
    assert boundary["validation_status"] == status
    assert raw["source_id"] == boundary["source_id"] == "fleurs-hi-1806"
    assert boundary["human_review"] is True
    assert raw["placebo_checked_unrelated"] is True
    if status == "human_validated":
        assert raw["schema"] == "vaani-localization-boundaries-v1"
        assert raw["complete_phrase_checked_both_versions"] is True
        return "complete_phrase_checked_both_versions"
    assert status == "human_reported_bounds", "Unknown boundary provenance category"
    assert raw["schema"] == "vaani-localization-boundaries-chat-v1"
    assert raw["phrase_bounds_reported_by_user"] is True
    assert raw["complete_phrase_checked_both_versions"] is None, "An unseen checkbox cannot become a positive review"
    assert boundary["full_phrase_extent_verified"] is False
    assert raw["prior_target_audible_both_versions"] is True
    capture = raw["capture_provenance"]
    assert capture["method"] == "user_screenshots_and_chat"
    assert capture["user_reply"] == "0.1"
    assert raw["boundary_uncertainty_s"] == .1
    assert raw["phrase_interval_s"] == [3, 4]

    def verify_evidence(name, expected_sha):
        path = Path(name)
        if not path.is_absolute():
            path = HERE / path
        assert bound_files[path.resolve()] == expected_sha, "Capture evidence missing from frozen source hashes"
        payload = path.read_bytes()
        assert sha256(payload) == expected_sha, "Capture evidence bytes changed"
        return payload

    screenshots = capture["screenshots"]
    assert isinstance(screenshots, list) and screenshots
    assert len({str(Path(row["path"])) for row in screenshots}) == len(screenshots)
    for screenshot in screenshots:
        verify_evidence(screenshot["path"], screenshot["sha256"])
    prior = json.loads(verify_evidence(capture["prior_listening_file"], capture["prior_listening_sha256"]))
    annotations = [row for row in prior["annotations"] if row["source_id"] == "fleurs-hi-1806"]
    assert len(annotations) == 2 and {row["condition"] for row in annotations} == {"clean", "noise_10db"}
    for item in config["inputs"]:
        matches = [row for row in annotations if row["id"] == item["id"]]
        assert len(matches) == 1
        heard = matches[0]
        assert heard["audio_sha256"] == item["audio_sha256"]
        assert heard["duration_s"] == item["duration_s"]
        assert heard["effective_target_spoken_verified"] is True
    return "user_reported_uncertain_window_not_complete_phrase"


def load_validated_inputs():
    """No remote calls: require future root freeze, old bindings, and human review."""
    from localization_worker import validate_inputs

    artifacts = HERE / "artifacts"
    freeze_path = artifacts / "localization-freeze-v1.json"
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    config = freeze["config"]
    assert freeze["review_status"] == "static_review_passed"
    assert config["run_id"] == "localized-omission-sensitivity-v1"
    expected_resources = {"gpu": "A10", "gpu_count": 1, "cpu": 2, "memory_gib": 16,
        "function_s": 600, "startup_s": 300, "queue_s": 120, "retries": 0}
    assert config["resource_limits"] == expected_resources
    assert config.get("training_updates") == 0 and config.get("fresh_test_loaded") is False
    assert config.get("independent_sources") == 1
    sources = config["source_sha256"]
    assert {"localization_worker.py", "localization_runner.py"} <= set(sources)
    for name, expected in sources.items():
        path = Path(name) if Path(name).is_absolute() else HERE / name
        assert sha256(path.read_bytes()) == expected, "Source/provenance hash changed: " + name
    old_path = artifacts / "omission-freeze-v1.json"
    old_freeze_sha = sources["artifacts/omission-freeze-v1.json"]
    assert old_freeze_sha == sha256(old_path.read_bytes())
    if "omission_freeze_sha256" in config:
        assert config["omission_freeze_sha256"] == old_freeze_sha
    old = checked_json(old_path)["config"]
    assert config["pair"] == old["pair"]
    assert config["model"] == old["model"] and config["model_revision"] == old["model_revision"]
    assert config["inputs"] == old["inputs"], "Both original and frozen mismatched recordings must remain identical"
    provenance = config["boundary_provenance"]
    assert provenance["validation_status"] in ("human_validated", "human_reported_bounds")
    boundary_path = Path(provenance["file"])
    if not boundary_path.is_absolute():
        boundary_path = HERE / boundary_path
    boundary_bytes = boundary_path.read_bytes()
    assert sha256(boundary_bytes) == provenance["sha256"]
    boundary = json.loads(boundary_bytes)
    assert boundary["schema"] == "vaani-localization-boundaries-validated-v1"
    assert boundary["validation_status"] == provenance["validation_status"] and boundary["human_review"] is True
    assert boundary["source_id"] == "fleurs-hi-1806"
    raw_path = Path(boundary["raw_annotation_file"])
    if not raw_path.is_absolute():
        raw_path = HERE / raw_path
    raw_bytes = raw_path.read_bytes()
    assert sha256(raw_bytes) == boundary["raw_annotation_sha256"]
    raw = json.loads(raw_bytes)
    assert raw["source_id"] == "fleurs-hi-1806"
    assert raw["created_at"] and raw["method"]
    assert len(raw["clips"]) == 2
    raw_clips = {row["condition"]: row for row in raw["clips"]}
    assert set(raw_clips) == {"clean", "noise_10db"}
    for item in config["inputs"]:
        clip = raw_clips[item["condition"]]
        assert clip["id"] == item["id"] and clip["audio_sha256"] == item["audio_sha256"]
        assert clip["duration_s"] == item["duration_s"]
    assert boundary["phrase_interval_s"] == raw["phrase_interval_s"]
    assert boundary["boundary_uncertainty_s"] == raw["boundary_uncertainty_s"]
    assert raw["target_phrase"] == "प्रकाश प्रदूषण"
    from localization_worker import mask_bounds
    checked_bounds, _ = mask_bounds(config, round(13.68 * 16000))
    assert boundary["target_sample_bounds"] == list(checked_bounds["target"])
    assert boundary["placebo_sample_bounds"] == list(checked_bounds["placebo"])
    assert boundary["equal_sample_count"] == checked_bounds["target"][1] - checked_bounds["target"][0]
    assert "rounding" in boundary or "rounding_details" in boundary
    # Both the unchanged raw annotation and the derived validation are code-freeze inputs.
    bound_files = {(Path(name) if Path(name).is_absolute() else HERE / name).resolve(): expected
        for name, expected in sources.items()}
    assert bound_files[boundary_path.resolve()] == provenance["sha256"]
    assert bound_files[raw_path.resolve()] == boundary["raw_annotation_sha256"]
    validate_boundary_capture(raw, boundary, provenance, config, bound_files)
    for name in ("target_mask_interval_s", "placebo_mask_interval_s"):
        assert boundary[name] == config[name], "Reviewed boundary differs from frozen job config"
    ledger = checked_json(artifacts / "modal-cost-ledger.json")
    allocation = ledger["localization_diagnostic"]
    assert allocation["status"] == "authorized_prelaunch"
    assert allocation["all_reserved_plus_protected_usd"] <= ledger["global_ceiling"] == 30
    assert allocation["reservation_usd"] == config["reservation_usd"]
    assert 0 < config["worst_case_cost_usd"] <= config["reservation_usd"]
    assert config["worst_case_cost_usd"] >= 902 * .00036772, "Declared bound must cover maximum A10+CPU+RAM time"
    inputs = []
    for meta in config["inputs"]:
        item = dict(meta)
        for field, binary in (("audio_file", "audio_bytes"), ("donor_file", "donor_bytes")):
            item[binary] = Path(item[field]).read_bytes()
        inputs.append(item)
    validate_inputs(inputs, config)
    return inputs, config, sha256(freeze_bytes), provenance["sha256"]


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(16384, 16384), timeout=600,
    startup_timeout=300, max_containers=1, min_containers=0, buffer_containers=0,
    scaledown_window=2, retries=0, volumes={"/cache": volume})
def diagnose(inputs, config):
    import sys
    sys.path.insert(0, "/app")
    from localization_worker import run
    try:
        return run(inputs, config, volume.commit)
    except BaseException as exc:
        path = Path("/cache/runs") / config["run_id"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "error": str(exc)}), encoding="utf-8")
        raise
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    artifacts = HERE / "artifacts"
    path = artifacts / "localization-job.json"
    if path.exists() or (artifacts / "localization-result.json").exists() or (artifacts / "localization-config.json").exists():
        raise RuntimeError("Existing localization job/config/result: inspect or recover; never duplicate")
    inputs, config, freeze_sha, boundary_sha = load_validated_inputs()
    config_bytes = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Exclusive creation claims the run before spawning, including spawn failures.
    job = {"phase": "validated_pre_spawn", "started_unix": time.time(), "config": config,
        "freeze_sha256": freeze_sha, "boundary_provenance_sha256": boundary_sha,
        "config_sha256": sha256(config_bytes), "reservation_usd": config["reservation_usd"],
        "cloud_run_dir": "/runs/" + config["run_id"], "automatic_retries": 0}
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(job, ensure_ascii=False, indent=2) + "\n")
    call = None
    try:
        (artifacts / "localization-config.json").write_bytes(config_bytes)
        call = diagnose.spawn(inputs, config)
        job.update(phase="queued", app_id=app.app_id, call_id=call.object_id)
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in job.items() if k != "config"}), flush=True)
        queued = time.monotonic()
        while diagnose.get_current_stats().num_total_runners == 0:
            if time.monotonic() - queued >= 120:
                raise TimeoutError("A10 queue exceeded 120-second cap")
            time.sleep(5)
        job["phase"] = "running"
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = call.get(timeout=920)
        assert result["status"] == "localized_omission_sensitivity_complete" and result["config"] == config
        assert len(result["rows"]) == 8 and result["forward_counts"]["total_decoder_forwards"] == 24
        expected_keys = {(item["condition"], control) for item in inputs
            for control in ("original", "target_masked", "placebo_masked", "mismatch")}
        assert {(row["condition"], row["control"]) for row in result["rows"]} == expected_keys
        assert all(row["shared_prefix"]["same_final_distribution"] and row["shared_prefix"]["decoder_forwards"] == 1 for row in result["rows"])
        result_path = artifacts / "localization-result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        job.update(phase="complete", result_sha256=sha256(result_path.read_bytes()), elapsed_s=result["elapsed_s"])
        print(json.dumps({k: v for k, v in job.items() if k != "config"}), flush=True)
    except BaseException as exc:
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
                job["cancellation"] = "requested_terminate_containers"
            except BaseException as cancel_exc:
                job["cancellation"] = "failed: " + type(cancel_exc).__name__ + ": " + str(cancel_exc)
        job.update(phase="failed_cancelled" if call is not None else "failed_before_spawn",
            error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
