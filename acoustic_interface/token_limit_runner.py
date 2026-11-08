"""One bounded, non-retrying C-only Modal length diagnostic under existing cap."""
import hashlib
import json
from pathlib import Path
import time
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('vaani-gold-token-limit-dev24')
volume=modal.Volume.from_name('vaani-acoustic-interface-cache-v1',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0','transformers==4.51.3','numpy==1.26.4',
        'scipy==1.15.2','soundfile==0.13.1','accelerate==1.6.0',
        'safetensors==0.5.3','pyarrow==19.0.1')
    .env({'HF_HOME':'/cache/huggingface','PYTHONPATH':'/app','TOKENIZERS_PARALLELISM':'false'})
    .add_local_file(HERE/'connector.py','/app/connector.py')
    .add_local_file(HERE/'token_limit_worker.py','/app/token_limit_worker.py'))


@app.function(image=image,gpu='A10',cpu=(2,2),memory=(16384,16384),timeout=600,
    startup_timeout=300,max_containers=1,min_containers=0,buffer_containers=0,
    scaledown_window=2,retries=0,volumes={'/cache':volume})
def diagnose(payload,config):
    import sys
    sys.path.insert(0,'/app')
    from token_limit_worker import run
    try:
        return run(payload,config,volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    out=HERE/'artifacts'
    if any((out/name).exists() for name in ('token-limit-job.json','token-limit-result.json')):
        raise RuntimeError('Existing job/result: inspect or recover, never repeat silently')
    freeze=json.loads((out/'token-limit-freeze-v1.json').read_text(encoding='utf-8'))
    assert freeze['authorization']=='user_explicit_correct_transcript_only_token_limit_diagnostic'
    config=freeze['config']
    for name,digest in config['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    payload=(out/'token-limit-input.json').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==config['input_sha256']
    ledger=json.loads((out/'modal-cost-ledger.json').read_text(encoding='utf-8'))
    assert ledger['token_limit_diagnostic']['status']=='authorized_prelaunch'
    assert ledger['global_ceiling']==30 and ledger['protected_reserve']==3
    assert ledger['token_limit_diagnostic']['all_reserved_plus_protected_usd']<=30
    assert config['worst_case_cost_usd']<config['reservation_usd']==1
    call=diagnose.spawn(payload,config)
    job=dict(phase='queued',app_id=app.app_id,call_id=call.object_id,
        started_unix=time.time(),config=config)
    path=out/'token-limit-job.json'
    path.write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(job),flush=True)
    try:
        queued=time.monotonic()
        while diagnose.get_current_stats().num_total_runners==0:
            if time.monotonic()-queued>120:
                raise TimeoutError('A10 queue cap120s')
            time.sleep(5)
        job['phase']='running'
        path.write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
        result=call.get(timeout=920)
        assert result['status']=='token_limit_complete' and len(result['rows'])==24
        assert result['config']==config
        result_path=out/'token-limit-result.json'
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        job.update(phase='complete',result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
            elapsed_s=result['elapsed_s'],prefix_mismatches=result['baseline_prefix_mismatches'])
        print(json.dumps({k:v for k,v in job.items() if k!='config'}),flush=True)
    except BaseException as exc:
        call.cancel(terminate_containers=True)
        job.update(phase='failed_cancelled',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
