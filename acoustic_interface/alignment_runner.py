"""One bounded alignment call after explicit, separately recorded user approval."""
import json
from pathlib import Path
import sys
import time

import modal

from alignment_worker import EVAL_ARMS, RUN_ID, split_rows, validate_config
from scale_worker import canonical, sha

HERE = Path(__file__).resolve().parent
app = modal.App("vaani-alignment-ce-kl")
volume = modal.Volume.from_name("vaani-acoustic-interface-cache-v1", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4", "scipy==1.15.2",
        "soundfile==0.13.1", "accelerate==1.6.0", "safetensors==0.5.3", "pyarrow==19.0.1")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/app", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .add_local_file(HERE / "connector.py", "/app/connector.py")
    .add_local_file(HERE / "readout_candidates.py", "/app/readout_candidates.py")
    .add_local_file(HERE / "scale_worker.py", "/app/scale_worker.py")
    .add_local_file(HERE.parent / "score.py", "/app/score.py")
    .add_local_file(HERE / "alignment_objective.py", "/app/alignment_objective.py")
    .add_local_file(HERE / "alignment_worker.py", "/app/alignment_worker.py"))


def load_inputs():
    artifacts = HERE / "artifacts"
    freeze_bytes = (artifacts / "alignment-freeze-v1.json").read_bytes()
    freeze = json.loads(freeze_bytes)
    assert freeze["review_status"] == "static_review_passed"
    config = freeze["config"]
    validate_config(config)
    freeze_sha = sha(freeze_bytes)

    # The science freeze and approval are separate artifacts. A draft never authorizes compute.
    auth_bytes = (artifacts / "alignment-compute-authorization.json").read_bytes()
    authorization = json.loads(auth_bytes)
    assert authorization["status"] == "explicit_user_authorized"
    assert authorization["run_id"] == RUN_ID and authorization["freeze_sha256"] == freeze_sha
    assert authorization["reservation_usd"] == config["reservation_usd"]
    assert isinstance(authorization["user_reply"], str) and authorization["user_reply"].strip()
    assert authorization["authorized_at"]
    auth_sha = sha(auth_bytes)
    ledger = json.loads((artifacts / "modal-cost-ledger.json").read_bytes())
    allocation = ledger["alignment_diagnostic"]
    assert allocation["status"] == "authorized_prelaunch"
    assert allocation["run_id"] == RUN_ID and allocation["freeze_sha256"] == freeze_sha
    assert allocation["authorization_sha256"] == auth_sha
    assert allocation["reservation_usd"] == config["reservation_usd"]
    assert allocation["all_reserved_plus_protected_usd"] <= ledger["global_ceiling"] == 30

    names = config["source_sha256"]
    required_sources = {"alignment_runner.py", "alignment_worker.py", "alignment_objective.py",
        "connector.py", "readout_candidates.py", "scale_worker.py", "../score.py"}
    assert required_sources <= set(names)
    for name, digest in names.items():
        path = Path(name) if Path(name).is_absolute() else HERE / name
        assert sha(path.read_bytes()) == digest, "Frozen source or evidence changed: " + name
    historical = (HERE.parent / "data/manifest.jsonl").read_bytes()
    train, dev = split_rows(historical, config)
    allowed = {row["id"]: row for row in train + dev}
    assert len(config["inputs"]) == 86
    assert {item["id"] for item in config["inputs"]} == set(allowed) == set(config["cache_sha256_by_view"])
    inputs = []
    for meta in config["inputs"]:
        row = allowed[meta["id"]]
        audio = Path(meta["audio_file"]).read_bytes()
        assert sha(audio) == meta["audio_sha256"] == row["audio_sha256"]
        inputs.append({"id": row["id"], "audio_sha256": row["audio_sha256"], "audio_bytes": audio})
    return inputs, historical, config, freeze_sha, auth_sha


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(16384, 16384), timeout=1800, startup_timeout=300,
    max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2, retries=0, volumes={"/cache": volume})
def diagnose(inputs, historical, config):
    sys.path.insert(0, "/app")
    from alignment_worker import run

    try:
        return run(inputs, historical, config, volume.commit)
    finally:
        volume.commit()


def validate_result(result, config, historical):
    assert result["config"] == config and result["fresh_test_loaded"] is False
    train, dev = split_rows(historical, config)
    teachers = result["teacher_rows"]
    assert len(teachers) == 32 and {r["source_id"] for r in teachers} == {r["source_id"] for r in train}
    passed = sum(r["eligible"] and r["answerable"] for r in teachers) >= 4
    assert result["teacher_eligibility"]["passed"] == passed
    if not passed:
        assert result["status"] == "alignment_teacher_gate_stopped"
        assert not result["fit_trace"] and not result["eval_rows"] and not result["checkpoint_sha256"]
        return
    assert result["status"] == "alignment_complete"
    assert set(result["fit_trace"]) == set(result["checkpoint_sha256"]) == {"ce", "kl"}
    assert all(len(trace) == 32 for trace in result["fit_trace"].values())
    assert len(result["eval_rows"]) == 110
    assert {(row["id"], row["arm"]) for row in result["eval_rows"]} == {
        (row["id"], arm) for row in dev for arm in EVAL_ARMS}


@app.local_entrypoint()
def main():
    artifacts = HERE / "artifacts"
    path = artifacts / "alignment-job.json"
    if any((artifacts / name).exists() for name in ("alignment-job.json", "alignment-config.json", "alignment-result.json")):
        raise RuntimeError("An alignment job, config or result already exists. Inspect or recover it before any new decision.")
    inputs, historical, config, freeze_sha, auth_sha = load_inputs()
    config_bytes = json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8")
    job = {"phase": "validated_pre_spawn", "run_id": RUN_ID, "config": config, "freeze_sha256": freeze_sha,
        "authorization_sha256": auth_sha, "config_sha256": sha(config_bytes), "started_unix": time.time(),
        "cloud_run_dir": "/runs/" + RUN_ID, "automatic_retries": 0}
    with path.open("xb") as stream:
        stream.write(canonical(job))
    call = None
    try:
        (artifacts / "alignment-config.json").write_bytes(config_bytes)
        call = diagnose.spawn(inputs, historical, config)
        job.update(phase="queued", app_id=app.app_id, call_id=call.object_id)
        path.write_bytes(canonical(job))
        print(json.dumps({k: v for k, v in job.items() if k != "config"}), flush=True)
        queued = time.monotonic()
        while diagnose.get_current_stats().num_total_runners == 0:
            if time.monotonic() - queued >= 120:
                raise TimeoutError("A10 queue exceeded 120 seconds")
            time.sleep(5)
        job["phase"] = "running"
        path.write_bytes(canonical(job))
        result = call.get(timeout=2120)
        validate_result(result, config, historical)
        result_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
        (artifacts / "alignment-result.json").write_bytes(result_bytes)
        job.update(phase="complete", result_status=result["status"], result_sha256=sha(result_bytes),
            elapsed_s=result["elapsed_s"])
        print(json.dumps({k: v for k, v in job.items() if k != "config"}), flush=True)
    except BaseException as error:
        if call is not None:
            try:
                call.cancel(terminate_containers=True)
                job["cancellation"] = "requested_terminate_containers"
            except Exception as cancellation_error:
                job["cancellation"] = "failed: " + str(cancellation_error)
        job.update(phase="failed_cancelled" if call is not None else "failed_before_spawn",
            error_type=type(error).__name__, error=str(error))
        raise
    finally:
        path.write_bytes(canonical(job))
