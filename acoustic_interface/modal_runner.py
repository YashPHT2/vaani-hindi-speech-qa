"""Ephemeral single-A10 runner. No deployment, schedules, retries or warm containers.

Run only after parent budget authorization: modal run modal_runner.py --stage smoke
"""
import hashlib
import json
import time
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
app = modal.App("vaani-acoustic-interface-bounded")
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy==1.26.4",
                 "scipy==1.15.2", "soundfile==0.13.1", "accelerate==1.6.0",
                 "safetensors==0.5.3", "pyarrow==19.0.1")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/app", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "connector.py", "/app/connector.py")
    .add_local_file(HERE / "features.py", "/app/features.py")
    .add_local_file(HERE / "worker.py", "/app/worker.py")
    .add_local_file(HERE / "training.py", "/app/training.py")
    .add_local_file(HERE / "gpu_preflight.py", "/app/gpu_preflight.py")
    .add_local_file(HERE / "final_training.py", "/app/final_training.py"))

# Cache is cloud-only, bounded by the two pinned checkpoints and one 306MB shard.
volume = modal.Volume.from_name("vaani-acoustic-interface-cache-v1", create_if_missing=True)


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(16384, 16384),
              timeout=300, startup_timeout=300, max_containers=1,
              min_containers=0, buffer_containers=0, scaledown_window=2, retries=0,
              volumes={"/cache": volume})
def gpu_smoke(audio_bytes: bytes, row: dict):
    import sys
    import tempfile
    sys.path.insert(0, "/app")
    from worker import smoke
    print("VAANI_GPU_FUNCTION_STARTED", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
        audio.write(audio_bytes)
        audio.flush()
        try:
            return smoke(audio.name, row)
        finally:
            volume.commit()


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(16384, 16384),
              timeout=3300, startup_timeout=300, max_containers=1,
              min_containers=0, buffer_containers=0, scaledown_window=2, retries=0,
              volumes={"/cache": volume})
def gpu_pilot(rows: list, config: dict):
    import sys
    sys.path.insert(0, "/app")
    from training import run
    print("VAANI_PILOT_GPU_STARTED", flush=True)
    try:
        return run(rows, config, volume.commit)
    except BaseException as exc:
        error = {"status": "pilot_failed", "type": type(exc).__name__, "error": str(exc)}
        out = Path("/cache/runs") / config["run_id"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "failure.json").write_text(json.dumps(error), encoding="utf-8")
        print(json.dumps(error), flush=True)
        raise
    finally:
        volume.commit()


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(16384, 16384),
              timeout=7200, startup_timeout=300, max_containers=1,
              min_containers=0, buffer_containers=0, scaledown_window=2, retries=0,
              volumes={"/cache": volume})
def gpu_final(historical_bytes, fresh_qa_bytes, selection_bytes, fresh_spec_bytes, config):
    import sys
    sys.path.insert(0, "/app")
    from final_training import run
    print("VAANI_FINAL_GPU_STARTED", flush=True)
    try:
        return run(historical_bytes, fresh_qa_bytes, selection_bytes, fresh_spec_bytes, config, volume.commit)
    except BaseException as exc:
        failure = {"status": "final_failed", "type": type(exc).__name__, "error": str(exc)}
        path = Path("/cache/runs") / config["run_id"]
        path.mkdir(parents=True, exist_ok=True)
        (path / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
        print(json.dumps(failure), flush=True)
        raise
    finally:
        volume.commit()


def campaign(update):
    path = HERE / "artifacts" / "modal-campaign.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state.update(update)
    state["updated_unix"] = time.time()
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


@app.local_entrypoint()
def main(stage: str = "smoke"):
    rows = [json.loads(line) for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    out = HERE / "artifacts"
    out.mkdir(exist_ok=True)
    if stage == "final":
        config = json.loads((out / "final-run-config.json").read_text(encoding="utf-8"))
        for name, expected in config["code_sha256"].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected
        previous = out / "modal-campaign.json"
        if previous.exists() and not (out / "pilot-completed-campaign.json").exists():
            (out / "pilot-completed-campaign.json").write_bytes(previous.read_bytes())
        previous.write_text(json.dumps({"phase": "spawning", "stage": "final", "run_id": config["run_id"],
            "config": config, "reservation_usd": 4, "code_sha256": config["code_sha256"],
            "cloud_volume": "vaani-acoustic-interface-cache-v1", "cloud_run_dir": "/runs/"+config["run_id"]}), encoding="utf-8")
        function = gpu_final
        call = function.spawn((DATA / "manifest.jsonl").read_bytes(), (HERE / "data/qa-v1.jsonl").read_bytes(),
            (HERE / "data/selection-v1.json").read_bytes(), (out / "final-fresh200-input-specs.json").read_bytes(), config)
    elif stage == "pilot":
        from validate_artifact import validate_smoke
        validated = validate_smoke(out)
        (out / "smoke-validation.json").write_text(json.dumps({**validated,
            "artifact_role": "exposed_historical_training_only", "pilot_initialized_from_smoke": False}, indent=2))
        config = {"run_id": "paired-pilot-st-seed17-v1-r1", "train_sources": 32,
            "epochs": 3, "work_timeout_s": 3200, "seed": 17, "slots": 4,
            "reservation_usd": 2, "selection": "hard_forward_localized_st_lambda2_valid_rank"}
        campaign({"phase": "spawning", "config": config,
            "code_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ("connector.py", "features.py", "training.py", "worker.py", "gpu_preflight.py", "modal_runner.py")}})
        function = gpu_pilot
        call = function.spawn(rows, config)
    elif stage == "smoke":
        row = next(r for r in rows if r["split"] == "test" and r["condition"] == "clean")
        wav = (DATA / row["audio"]).read_bytes()
        assert hashlib.sha256(wav).hexdigest() == row["audio_sha256"]
        public_row = {k: row[k] for k in ("id", "source_id", "question", "answers")}
        function = gpu_smoke
        call = function.spawn(wav, public_row)
    else:
        raise ValueError(stage)
    print(f"VAANI_CALL_ID={call.object_id}", flush=True)
    campaign({"phase": "queued", "stage": stage, "call_id": call.object_id,
        "app_id": app.app_id, "started_unix": time.time()})
    try:
        queue_start = time.monotonic()
        while function.get_current_stats().num_total_runners == 0:
            if time.monotonic() - queue_start >= 120:
                raise TimeoutError("A10 queue exceeded authorized 120-second cap")
            time.sleep(5)
        campaign({"phase": "running"})
        result, checkpoints = call.get(timeout={"final": 7510, "pilot": 3610, "smoke": 610}[stage])
        if stage == "smoke":
            checkpoints = {"smoke": checkpoints}
        for arm, blob in checkpoints.items():
            expected = result["checkpoint_sha256"][arm] if stage != "smoke" else result["checkpoint_sha256"]
            assert hashlib.sha256(blob).hexdigest() == expected
            (out / (f"{stage}-{arm}-adapter.pt" if stage != "smoke" else "smoke-adapter.pt")).write_bytes(blob)
        result_path = out / (stage + "-result.json")
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        campaign({"phase": "complete", "result_path": str(result_path),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "elapsed_s": result["elapsed_s"], "checkpoint_sha256": result["checkpoint_sha256"]})
        print(json.dumps({"status": result["status"], "elapsed_s": result["elapsed_s"]}), flush=True)
    except BaseException as exc:
        call.cancel(terminate_containers=True)
        campaign({"phase": "failed_cancelled", "error_type": type(exc).__name__, "error": str(exc)})
        raise
