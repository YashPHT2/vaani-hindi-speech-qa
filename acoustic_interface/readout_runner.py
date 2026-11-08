"""Guarded one-call readout diagnostic; root owns authorization and freeze."""
import hashlib
import json
from pathlib import Path
import time

import modal

HERE=Path(__file__).resolve().parent
app=modal.App("vaani-readout-fallback-dev2")
volume=modal.Volume.from_name("vaani-acoustic-interface-cache-v1",create_if_missing=False)
image=(modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0","transformers==4.51.3","numpy==1.26.4","scipy==1.15.2",
        "soundfile==0.13.1","accelerate==1.6.0","safetensors==0.5.3","pyarrow==19.0.1")
    .env({"HF_HOME":"/cache/huggingface","PYTHONPATH":"/app","TOKENIZERS_PARALLELISM":"false"})
    .add_local_file(HERE/"connector.py","/app/connector.py")
    .add_local_file(HERE/"readout_candidates.py","/app/readout_candidates.py")
    .add_local_file(HERE/"readout_worker.py","/app/readout_worker.py"))


def sha(payload):return hashlib.sha256(payload).hexdigest()


def load_inputs():
    from readout_worker import CHECKPOINT_SHA,RUN_ID
    art=HERE/"artifacts"
    freeze_bytes=(art/"readout-freeze-v1.json").read_bytes()
    freeze=json.loads(freeze_bytes)
    assert freeze["review_status"]=="static_review_passed"
    config=freeze["config"]
    assert config["run_id"]==RUN_ID and config["pilot_checkpoint_sha256"]==CHECKPOINT_SHA
    assert config["reported_window_frame_bounds"]==[145,205] and config["full_phrase_extent_verified"] is False
    assert config["seed"]==17 and config["max_new_tokens"]==48
    assert config["training_updates"]==0 and config["fresh_test_loaded"] is False
    expected={"gpu":"A10","gpu_count":1,"cpu":2,"memory_gib":16,"function_s":600,"startup_s":300,"queue_s":120,"retries":0}
    assert config["resource_limits"]==expected
    names=config["source_sha256"]
    assert {"readout_candidates.py","readout_worker.py","readout_runner.py","connector.py"}<=set(names)
    for name,expected_sha in names.items():
        path=Path(name) if Path(name).is_absolute() else HERE/name
        assert sha(path.read_bytes())==expected_sha,"Frozen source/evidence changed: "+name
    old_path=art/"omission-freeze-v1.json"
    assert names["artifacts/omission-freeze-v1.json"]==sha(old_path.read_bytes())
    assert config["inputs"]==json.loads(old_path.read_bytes())["config"]["inputs"]
    headroom=art/"headroom-result.json"
    local_headroom_bytes=headroom.read_bytes()
    assert sha(local_headroom_bytes)==config["headroom_result_sha256"]
    headroom_object=json.loads(local_headroom_bytes)
    semantic_bytes=json.dumps(headroom_object,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    assert sha(semantic_bytes)==config["headroom_semantic_sha256"]
    remote_lf_bytes=json.dumps(headroom_object,indent=2,ensure_ascii=False).encode("utf-8")
    assert sha(remote_lf_bytes)==config["headroom_remote_lf_expected_sha256"]
    historical=(HERE.parent/"data/manifest.jsonl").read_bytes()
    assert sha(historical)==config["historical_manifest_sha256"]
    manifest=[json.loads(line) for line in historical.decode("utf-8").splitlines()]
    assert len(manifest)==224 and len({row["id"] for row in manifest})==224
    ledger=json.loads((art/"modal-cost-ledger.json").read_text(encoding="utf-8"))
    allocation=ledger["readout_diagnostic"]
    assert allocation["status"]=="authorized_prelaunch"
    assert allocation["all_reserved_plus_protected_usd"]<=ledger["global_ceiling"]==30
    assert config["reservation_usd"]==allocation["reservation_usd"]
    assert 902*.00036772<=config["worst_case_cost_usd"]<=config["reservation_usd"]
    inputs=[]
    for meta in config["inputs"]:
        item=dict(meta)
        item["audio_bytes"]=Path(item["audio_file"]).read_bytes()
        item["donor_bytes"]=Path(item["donor_file"]).read_bytes()
        assert sha(item["audio_bytes"])==item["audio_sha256"]
        assert sha(item["donor_bytes"])==item["donor_audio_sha256"]
        assert item["duration_s"]==item["donor_duration_s"]==13.68
        inputs.append(item)
    assert {item["id"] for item in inputs}=={"fleurs-hi-1806-clean","fleurs-hi-1806-noise_10db"} and len(inputs)==2
    return inputs,historical,config,sha(freeze_bytes)


@app.function(image=image,gpu="A10",cpu=(2,2),memory=(16384,16384),timeout=600,startup_timeout=300,
    max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,volumes={"/cache":volume})
def diagnose(inputs,historical_bytes,config):
    import sys
    sys.path.insert(0,"/app")
    from readout_worker import run
    try:return run(inputs,historical_bytes,config,volume.commit)
    finally:volume.commit()


@app.local_entrypoint()
def main():
    art=HERE/"artifacts"
    path=art/"readout-job.json"
    if any((art/name).exists() for name in ("readout-job.json","readout-result.json","readout-config.json")):
        raise RuntimeError("Existing readout job/result/config: recover, never duplicate")
    inputs,historical,config,freeze_sha=load_inputs()
    config_bytes=json.dumps(config,ensure_ascii=False,indent=2).encode("utf-8")
    job={"phase":"validated_pre_spawn","started_unix":time.time(),"config":config,"freeze_sha256":freeze_sha,
        "config_sha256":sha(config_bytes),"cloud_run_dir":"/runs/"+config["run_id"],"automatic_retries":0}
    with path.open("x",encoding="utf-8") as stream:stream.write(json.dumps(job,ensure_ascii=False,indent=2))
    call=None
    try:
        (art/"readout-config.json").write_bytes(config_bytes)
        call=diagnose.spawn(inputs,historical,config)
        job.update(phase="queued",app_id=app.app_id,call_id=call.object_id)
        path.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({key:value for key,value in job.items() if key!="config"}),flush=True)
        queued=time.monotonic()
        while diagnose.get_current_stats().num_total_runners==0:
            if time.monotonic()-queued>=120:raise TimeoutError("A10 queue exceeded120seconds")
            time.sleep(5)
        job["phase"]="running"
        path.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding="utf-8")
        result=call.get(timeout=920)
        assert result["status"]=="readout_fallback_complete" and result["config"]==config and len(result["rows"])==12
        assert len(result["cache_inventory"])==224 and result["cache_available"]+result["cache_missing"]==224
        assert result["checkpoint_before_sha256"]==result["checkpoint_after_sha256"]==config["pilot_checkpoint_sha256"]
        result_path=art/"readout-result.json"
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        inventory_path=art/"readout-cache-inventory.json"
        inventory_path.write_text(json.dumps(result["cache_inventory"],ensure_ascii=False,indent=2),encoding="utf-8")
        assert sha(inventory_path.read_bytes())==result["cache_inventory_sha256"]
        job.update(phase="complete",elapsed_s=result["elapsed_s"],result_sha256=sha(result_path.read_bytes()),
            cache_inventory_sha256=result["cache_inventory_sha256"],baseline_guard_mismatch=result["baseline_guard_mismatch"])
        print(json.dumps({key:value for key,value in job.items() if key!="config"}),flush=True)
    except BaseException as exc:
        if call is not None:
            try:call.cancel(terminate_containers=True);job["cancellation"]="requested_terminate_containers"
            except BaseException as cancel_exc:job["cancellation"]="failed: "+str(cancel_exc)
        job.update(phase="failed_cancelled" if call is not None else "failed_before_spawn",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:path.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding="utf-8")
