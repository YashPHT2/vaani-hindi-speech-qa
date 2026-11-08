"""Exclusive bounded scalar diagnostic runner; root freezes inputs and budget."""
import hashlib
import json
from pathlib import Path
import time
import modal

HERE=Path(__file__).resolve().parent
app=modal.App("vaani-scalar-calibration")
volume=modal.Volume.from_name("vaani-acoustic-interface-cache-v1",create_if_missing=False)
image=(modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0","transformers==4.51.3","numpy==1.26.4","scipy==1.15.2","soundfile==0.13.1","accelerate==1.6.0","safetensors==0.5.3","pyarrow==19.0.1")
    .env({"HF_HOME":"/cache/huggingface","PYTHONPATH":"/app","TOKENIZERS_PARALLELISM":"false"})
    .add_local_file(HERE/"connector.py","/app/connector.py")
    .add_local_file(HERE/"readout_candidates.py","/app/readout_candidates.py")
    .add_local_file(HERE/"scale_worker.py","/app/scale_worker.py"))
def sha(payload):return hashlib.sha256(payload).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")


def load_inputs():
    from scale_worker import CHECKPOINT_SHA,RUN_ID,GRID
    art=HERE/"artifacts";payload=(art/"scale-freeze-v1.json").read_bytes();freeze=json.loads(payload)
    assert freeze["review_status"]=="static_review_passed"
    config=freeze["config"]
    assert config["run_id"]==RUN_ID and config["pilot_checkpoint_sha256"]==CHECKPOINT_SHA
    assert config["seed"]==17 and config["max_new_tokens"]==48 and config["alphas"]==list(GRID)
    assert config["resource_limits"]=={"gpu":"A10","gpu_count":1,"cpu":2,"memory_gib":16,"function_s":1800,"startup_s":300,"queue_s":120,"retries":0}
    assert config["optimized_parameter_count"]==1 and config["fresh_test_loaded"] is False
    names=config["source_sha256"]
    assert {"scale_worker.py","scale_runner.py","connector.py","readout_candidates.py"}<=set(names)
    for name,digest in names.items():
        path=Path(name) if Path(name).is_absolute() else HERE/name
        assert sha(path.read_bytes())==digest,"Frozen source/evidence changed: "+name
    historical=(HERE.parent/"data/manifest.jsonl").read_bytes();assert sha(historical)==config["historical_manifest_sha256"]
    rows=[json.loads(line) for line in historical.decode("utf-8").splitlines()];by_id={r["id"]:r for r in rows};assert len(by_id)==len(rows)==224
    allowed=set(config["train_source_ids"])|set(config["primary_source_ids"])|{config["illustrative_source_id"]}
    assert len(allowed)==44 and len(config["inputs"])==88
    inputs=[]
    for meta in config["inputs"]:
        row=by_id[meta["id"]];assert row["source_id"] in allowed
        item=dict(meta);item["audio_bytes"]=Path(item["audio_file"]).read_bytes()
        assert sha(item["audio_bytes"])==item["audio_sha256"]==row["audio_sha256"]
        inputs.append(item)
    assert len({item["id"] for item in inputs})==88
    assert set(config["cache_sha256_by_view"])=={item["id"] for item in inputs}
    assert config["donor_alignment"]=="linear_interpolation_of_complete_valid_encoder_sequence_align_corners_false"
    assert len(config["baseline_rows"])==48
    assert config["fit"]=={"parameter":"log_alpha only","initial":0,"optimizer":"Adam","lr":.1,"batch":8,"epochs":4,"updates":32,"alpha_bounds":[1/32,2],"selection":"final update only"}
    assert sha(canonical(config["baseline_rows"]))==config["baseline_rows_semantic_sha256"]
    ledger=json.loads((art/"modal-cost-ledger.json").read_bytes());allocation=ledger["scale_diagnostic"]
    assert allocation["status"]=="authorized_prelaunch" and allocation["reservation_usd"]==config["reservation_usd"]==2
    assert allocation["all_reserved_plus_protected_usd"]<=ledger["global_ceiling"]==30
    assert 2102*.00036772<=config["worst_case_cost_usd"]<2
    return inputs,historical,config,sha(payload)


@app.function(image=image,gpu="A10",cpu=(2,2),memory=(16384,16384),timeout=1800,startup_timeout=300,max_containers=1,
    min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,volumes={"/cache":volume})
def diagnose(inputs,historical,config):
    import sys
    sys.path.insert(0,"/app")
    from scale_worker import run
    try:return run(inputs,historical,config,volume.commit)
    finally:volume.commit()


@app.local_entrypoint()
def main():
    art=HERE/"artifacts";path=art/"scale-job.json"
    if any((art/name).exists() for name in ("scale-job.json","scale-result.json","scale-config.json")):raise RuntimeError("Existing scalar job/result/config: recover, never repeat")
    inputs,historical,config,freeze_sha=load_inputs();config_bytes=json.dumps(config,ensure_ascii=False,indent=2).encode("utf-8")
    job={"phase":"validated_pre_spawn","config":config,"freeze_sha256":freeze_sha,"config_sha256":sha(config_bytes),"started_unix":time.time(),"automatic_retries":0}
    with path.open("xb") as stream:stream.write(canonical(job))
    call=None
    try:
        (art/"scale-config.json").write_bytes(config_bytes);call=diagnose.spawn(inputs,historical,config)
        job.update(phase="queued",app_id=app.app_id,call_id=call.object_id);path.write_bytes(canonical(job))
        print(json.dumps({k:v for k,v in job.items() if k!="config"}),flush=True);queued=time.monotonic()
        while diagnose.get_current_stats().num_total_runners==0:
            if time.monotonic()-queued>=120:raise TimeoutError("A10 queue exceeded120seconds")
            time.sleep(5)
        job["phase"]="running";path.write_bytes(canonical(job));result=call.get(timeout=2120)
        assert result["config"]==config and result["status"] in ("scale_preflight_stopped","scale_calibration_complete")
        assert len(result["preflight_rows"]) in (0,40) and result["weight_invariance"]["checkpoint_after_sha256"]==config["pilot_checkpoint_sha256"]
        assert (result["status"]=="scale_preflight_stopped" and result["scalar_updates"]==0 and not result["eval_rows"]) or (result["status"]=="scale_calibration_complete" and result["scalar_updates"]==32 and len(result["eval_rows"])==120)
        result_bytes=json.dumps(result,ensure_ascii=False,indent=2).encode("utf-8");(art/"scale-result.json").write_bytes(result_bytes)
        job.update(phase="complete",result_status=result["status"],result_sha256=sha(result_bytes),elapsed_s=result["elapsed_s"])
        print(json.dumps({k:v for k,v in job.items() if k!="config"}),flush=True)
    except BaseException as exc:
        if call is not None:
            try:call.cancel(terminate_containers=True);job["cancellation"]="requested_terminate_containers"
            except BaseException as error:job["cancellation"]="failed: "+str(error)
        job.update(phase="failed_cancelled" if call is not None else "failed_before_spawn",error_type=type(exc).__name__,error=str(exc));raise
    finally:path.write_bytes(canonical(job))
