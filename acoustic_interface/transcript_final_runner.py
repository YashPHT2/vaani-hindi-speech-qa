"""One bounded final cascade, guarded by an independently reviewed local freeze."""
import json
import os
from pathlib import Path
import time

import modal

from transcript_final_worker import CONDITIONS, FROZEN_HASHES, RUN_ID, json_bytes, sha, validate_config

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULT_DIR = ARTIFACTS / "transcript-final"
app = modal.App("vaani-transcript-only-final100")
volume = modal.Volume.from_name("vaani-acoustic-interface-cache-v1", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4", "scipy==1.15.2",
        "soundfile==0.13.1", "accelerate==1.6.0", "safetensors==0.5.3", "pyarrow==19.0.1")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/app", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .add_local_file(HERE / "connector.py", "/app/connector.py")
    .add_local_file(HERE / "transcript_final_worker.py", "/app/transcript_final_worker.py"))


def read_prepared_inputs():
    config_bytes = (ARTIFACTS / "transcript-final-config.json").read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    assert config_bytes == json_bytes(config)
    config_sha = sha(config_bytes)
    archive_sha = sha((ARTIFACTS / "transcript-final-source.zip").read_bytes())
    review = json.loads((ARTIFACTS / "transcript-final-freeze-v1.json").read_bytes())
    assert review["review_status"] == "static_review_passed" and review["run_id"] == RUN_ID
    assert review["config_sha256"] == config_sha and review["source_archive_sha256"] == archive_sha
    for name, expected in config["source_sha256"].items():
        path = Path(name) if Path(name).is_absolute() else HERE / name
        assert sha(path.read_bytes()) == expected, "Frozen implementation changed: " + name
    protocol = HERE / "transcript_final_protocol.md"
    decision = ARTIFACTS / "transcript-final-decision.json"
    assert sha(protocol.read_bytes()) == config["protocol_sha256"]
    assert sha(decision.read_bytes()) == config["scientific_decision_sha256"]
    assert sha((HERE / "data/final-experiment-freeze-v1.json").read_bytes()) == config["original_source_freeze_sha256"]
    paths = {"historical_manifest": HERE.parent / "data/manifest.jsonl",
        "selection": HERE / "data/selection-v1.json", "qa": HERE / "data/qa-v1.jsonl"}
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    for name, payload in payloads.items():
        assert sha(payload) == FROZEN_HASHES[name]
    payloads["view_specs"] = (ARTIFACTS / "final-fresh200-input-specs.json").read_bytes()
    assert sha(payloads["view_specs"]) == config["original_view_specs_sha256"]
    from transcript_final_worker import validate_view_specs
    validate_view_specs(payloads["view_specs"], json.loads((ARTIFACTS / "transcript-final-view-manifest.json").read_bytes()))
    assert sha((ARTIFACTS / "transcript-final-view-manifest.json").read_bytes()) == config["view_manifest_sha256"]
    assert sha((ARTIFACTS / "transcript-final-reference-manifest.json").read_bytes()) == config["reference_manifest_sha256"]
    ledger = json.loads((ARTIFACTS / "modal-cost-ledger.json").read_bytes())
    reservation = ledger["transcript_final_diagnostic"]
    assert reservation["status"] == "authorized_prelaunch"
    assert reservation["run_id"] == RUN_ID and reservation["config_sha256"] == config_sha
    assert reservation["reservation_usd"] == config["reservation_usd"] == 4
    assert reservation["all_reserved_plus_protected_usd"] <= ledger["global_ceiling"] == 30
    return payloads, config, config_sha, archive_sha


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(16384, 16384), timeout=7200, startup_timeout=300,
    max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2, retries=0, volumes={"/cache": volume})
def evaluate(historical, selection, qa, specs, config):
    import sys
    sys.path.insert(0, "/app")
    from transcript_final_worker import run
    try:
        return run(historical, selection, qa, specs, config, volume.commit)
    finally:
        volume.commit()


def validate_returned(result, config):
    assert result["config"] == config
    assert result["status"] in ("transcript_final_complete", "transcript_final_partial")
    expected = set(config["view_ids"])
    assert len(result["eval_rows"]) == len(result["references"]) == len(expected) == 200
    predictions = {row["id"]: row for row in result["eval_rows"]}
    references = {row["id"]: row for row in result["references"]}
    assert set(predictions) == set(references) == expected
    assert {row["condition"] for row in predictions.values()} == set(CONDITIONS)
    for identity, row in predictions.items():
        reference = references[identity]
        assert row["audio_sha256"] == reference["audio_sha256"] and row["question"] == reference["question"]
        assert row["mode"] == "baseline" and type(row["failed"]) is bool and type(row["asr_failed"]) is bool
        if row["asr_failed"]:
            assert not row["hypothesis"]
        if row["failed"]:
            assert not row["answer"] and row["error"]
    assert sum(row["failed"] for row in predictions.values()) == result["failed_views"]
    assert (result["status"] == "transcript_final_complete") == (result["failed_views"] == 0)


def save_result(result, config_sha):
    RESULT_DIR.mkdir(exist_ok=True)
    result_bytes = json_bytes(result)
    predictions = b"".join(json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n" for row in result["eval_rows"])
    references = b"".join(json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n" for row in result["references"])
    files = {"result.json": result_bytes, "predictions.jsonl": predictions, "references.jsonl": references}
    manifest = {"run_id": RUN_ID, "config_sha256": config_sha,
        "sha256": {name: sha(payload) for name, payload in files.items()}}
    files["artifact-manifest.json"] = json_bytes(manifest)
    # Each rename is atomic. The manifest is written last and certifies the complete set.
    for name, payload in files.items():
        destination = RESULT_DIR / name
        if destination.exists():
            assert destination.read_bytes() == payload, "Existing downloaded final artifact differs"
            continue
        temporary = RESULT_DIR / (name + ".partial")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    return manifest


@app.local_entrypoint()
def main(recover: bool = False):
    job_path = ARTIFACTS / "transcript-final-job.json"
    call = None
    if recover:
        job = json.loads(job_path.read_bytes())
        assert job["run_id"] == RUN_ID and job.get("call_id")
        config = job["config"]
        call = modal.FunctionCall.from_id(job["call_id"])
    else:
        if job_path.exists() or any((RESULT_DIR / name).exists() for name in ("result.json", "predictions.jsonl", "artifact-manifest.json")):
            raise RuntimeError("Final evaluation already has execution evidence. Recover the recorded call explicitly.")
        payloads, config, config_sha, archive_sha = read_prepared_inputs()
        job = {"phase": "validated_pre_spawn", "run_id": RUN_ID, "config": config, "config_sha256": config_sha,
            "source_archive_sha256": archive_sha, "started_unix": time.time(), "automatic_retries": 0,
            "cloud_run_dir": "/runs/" + RUN_ID}
        with job_path.open("xb") as stream:
            stream.write(json_bytes(job))
    try:
        if not recover:
            call = evaluate.spawn(payloads["historical_manifest"], payloads["selection"], payloads["qa"], payloads["view_specs"], config)
            job.update(phase="queued", app_id=app.app_id, call_id=call.object_id)
            job_path.write_bytes(json_bytes(job))
            print(json.dumps({k: v for k, v in job.items() if k != "config"}), flush=True)
            queue_start = time.monotonic()
            while evaluate.get_current_stats().num_total_runners == 0:
                if time.monotonic() - queue_start >= 120:
                    raise TimeoutError("A10 queue exceeded 120 seconds")
                time.sleep(5)
        job["phase"] = "recovering" if recover else "running"
        job_path.write_bytes(json_bytes(job))
        result = call.get(timeout=7520)
        validate_returned(result, config)
        manifest = save_result(result, job["config_sha256"])
        job.update(phase="complete", result_status=result["status"], result_sha256=manifest["sha256"]["result.json"],
            failed_views=result["failed_views"], elapsed_s=result["elapsed_s"])
        print(json.dumps({k: v for k, v in job.items() if k != "config"}), flush=True)
    except BaseException as error:
        if call is not None and not recover:
            try:
                call.cancel(terminate_containers=True)
                job["cancellation"] = "requested_terminate_containers"
            except Exception as cancellation_error:
                job["cancellation"] = "failed: " + str(cancellation_error)
        job.update(phase="recovery_failed" if recover else "failed_cancelled",
            error_type=type(error).__name__, error=str(error))
        raise
    finally:
        job_path.write_bytes(json_bytes(job))
