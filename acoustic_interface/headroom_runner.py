"""Bounded $1-reserved inference-only Modal diagnostic. No training or fresh test."""
import hashlib
import json
from pathlib import Path
import time
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('vaani-matched-headroom-dev24')
volume = modal.Volume.from_name('vaani-acoustic-interface-cache-v1', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0', 'transformers==4.51.3', 'numpy==1.26.4',
                 'scipy==1.15.2', 'soundfile==0.13.1', 'accelerate==1.6.0',
                 'safetensors==0.5.3', 'pyarrow==19.0.1')
    .env({'HF_HOME': '/cache/huggingface', 'PYTHONPATH': '/app', 'TOKENIZERS_PARALLELISM': 'false'})
    .add_local_file(HERE / 'connector.py', '/app/connector.py')
    .add_local_file(HERE / 'headroom_worker.py', '/app/headroom_worker.py'))


@app.function(image=image, gpu='A10', cpu=(2, 2), memory=(16384, 16384),
    timeout=600, startup_timeout=300, max_containers=1, min_containers=0,
    buffer_containers=0, scaledown_window=2, retries=0, volumes={'/cache': volume})
def diagnose(manifest_bytes, config):
    import sys
    sys.path.insert(0, '/app')
    from headroom_worker import run
    print('HEADROOM_GPU_STARTED', flush=True)
    try:
        return run(manifest_bytes, config, volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    out = HERE / 'artifacts'
    if (out / 'headroom-result.json').exists() or (out / 'headroom-job.json').exists():
        raise RuntimeError('Existing headroom result/job: recover or inspect, never silently repeat')
    manifest_bytes = (HERE.parent / 'data/manifest.jsonl').read_bytes()
    config = {'run_id': 'matched-headroom-dev24-v1', 'seed': 17, 'max_new_tokens': 48,
        'historical_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'code_sha256': {n: hashlib.sha256((HERE/n).read_bytes()).hexdigest()
                       for n in ('connector.py', 'headroom_worker.py', 'headroom_runner.py')},
        'arms': ['A_question_only', 'B_asr_text', 'C_correct_text', 'D_existing_adapter'],
        'training_updates': 0, 'fresh_test_loaded': False, 'reservation_usd': 1,
        'resource_limits': {'gpu': 'A10', 'gpu_count': 1, 'cpu': 2, 'memory_gib': 16,
                            'timeout_s': 600, 'startup_timeout_s': 300, 'retries': 0},
        'worst_case_cost_usd': 902 * .00036772 + .30,
        'qualification': 'Public-rate estimate with overhead allowance, not a provider hard billing cap'}
    assert config['worst_case_cost_usd'] < 1
    call = diagnose.spawn(manifest_bytes, config)
    job = {'phase': 'queued', 'app_id': app.app_id, 'call_id': call.object_id,
           'started_unix': time.time(), 'config': config}
    (out / 'headroom-job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
    print(json.dumps(job), flush=True)
    try:
        queued = time.monotonic()
        while diagnose.get_current_stats().num_total_runners == 0:
            if time.monotonic() - queued > 120:
                raise TimeoutError('A10 capacity wait exceeded120s')
            time.sleep(5)
        job['phase'] = 'running'
        (out / 'headroom-job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
        result = call.get(timeout=920)
        assert result['status'] == 'matched_headroom_complete' and len(result['rows']) == 96
        assert result['config'] == config
        path = out / 'headroom-result.json'
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        job.update(phase='complete', result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   elapsed_s=result['elapsed_s'])
        print(json.dumps({k: v for k, v in job.items() if k != 'config'}), flush=True)
    except BaseException as exc:
        call.cancel(terminate_containers=True)
        job.update(phase='failed_cancelled', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (out / 'headroom-job.json').write_text(json.dumps(job, indent=2), encoding='utf-8')
