"""One bounded diagnostic; no training, retries, adapters or fresh data."""
import hashlib
import json
from pathlib import Path
import time
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('vaani-verified-omission-sensitivity')
volume=modal.Volume.from_name('vaani-acoustic-interface-cache-v1',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0','transformers==4.51.3','numpy==1.26.4',
        'scipy==1.15.2','soundfile==0.13.1','accelerate==1.6.0',
        'safetensors==0.5.3','pyarrow==19.0.1')
    .env({'HF_HOME':'/cache/huggingface','PYTHONPATH':'/app','TOKENIZERS_PARALLELISM':'false'})
    .add_local_file(HERE/'omission_worker.py','/app/omission_worker.py'))


@app.function(image=image,gpu='A10',cpu=(2,2),memory=(16384,16384),timeout=600,
    startup_timeout=300,max_containers=1,min_containers=0,buffer_containers=0,
    scaledown_window=2,retries=0,volumes={'/cache':volume})
def diagnose(inputs,config):
    import sys
    sys.path.insert(0,'/app')
    from omission_worker import run
    try:
        return run(inputs,config,volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    art=HERE/'artifacts'
    if any((art/n).exists() for n in ['omission-job.json','omission-result.json']):
        raise RuntimeError('Existing job/result: inspect or recover, never duplicate')
    freeze=json.loads((art/'omission-freeze-v1.json').read_text(encoding='utf-8'))
    config=freeze['config']
    assert freeze['review_status']=='static_review_passed'
    for name,digest in config['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    ledger=json.loads((art/'modal-cost-ledger.json').read_text())
    assert ledger['omission_diagnostic']['status']=='authorized_prelaunch'
    assert ledger['omission_diagnostic']['all_reserved_plus_protected_usd']<=ledger['global_ceiling']==30
    inputs=[]
    for meta in config['inputs']:
        item=dict(meta)
        for field,binary in [('audio_file','audio_bytes'),('donor_file','donor_bytes')]:
            item[binary]=Path(item[field]).read_bytes()
        assert hashlib.sha256(item['audio_bytes']).hexdigest()==item['audio_sha256']
        assert hashlib.sha256(item['donor_bytes']).hexdigest()==item['donor_audio_sha256']
        inputs.append(item)
    call=diagnose.spawn(inputs,config)
    job=dict(phase='queued',app_id=app.app_id,call_id=call.object_id,started_unix=time.time(),config=config)
    path=art/'omission-job.json'
    path.write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(job,ensure_ascii=False),flush=True)
    try:
        queued=time.monotonic()
        while diagnose.get_current_stats().num_total_runners==0:
            if time.monotonic()-queued>120:
                raise TimeoutError('A10 queue exceeded120s')
            time.sleep(5)
        job['phase']='running'
        path.write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
        result=call.get(timeout=920)
        assert result['status']=='omission_sensitivity_complete' and len(result['rows'])==6 and result['config']==config
        rp=art/'omission-result.json'
        rp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        job.update(phase='complete',result_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),elapsed_s=result['elapsed_s'])
        print(json.dumps({k:v for k,v in job.items() if k!='config'}),flush=True)
    except BaseException as exc:
        call.cancel(terminate_containers=True)
        job.update(phase='failed_cancelled',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(job,indent=2)+'\n',encoding='utf-8')
