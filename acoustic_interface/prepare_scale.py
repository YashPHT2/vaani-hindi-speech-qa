"""Build historical-only split/donor/wave inputs; never launch or reserve compute."""
import hashlib
import json
from pathlib import Path
import time
import wave

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    art=HERE/'artifacts'
    assert not any((art/name).exists() for name in ['scale-job.json','scale-result.json','scale-config.json'])
    decision=json.loads((art/'scale-scientific-decision.json').read_text(encoding='utf-8'))
    mp=HERE.parent/'data/manifest.jsonl'
    manifest=[json.loads(s) for s in mp.read_text(encoding='utf-8').splitlines()]
    by={r['id']:r for r in manifest}
    source={r['source_id']:r for r in manifest if r['condition']=='clean'}
    train=decision['training_sources']
    primary=decision['primary_dev_sources']
    illustrative=decision['illustrative_source']
    assert len(set(train))==32 and len(set(primary))==11
    assert not set(train)&set(primary) and illustrative not in train+primary
    assert all(source[s]['split']=='test' for s in train) and all(source[s]['split']=='dev' for s in primary+[illustrative])
    preflight=sorted(train,key=lambda s:hashlib.sha256(('scale-preflight-v1:'+s.rsplit('-',1)[-1]).encode()).hexdigest())[:4]
    assert preflight==decision['preflight_sources']
    eval_ids=[r['id'] for r in manifest if r['source_id'] in primary+[illustrative]]
    donor={}
    for identifier in eval_ids:
        row=by[identifier]
        candidates=[r for r in manifest if r['source_id'] in train and r['condition']==row['condition']
                    and r['source_id']!=row['source_id'] and r['transcript']!=row['transcript'] and r['audio_sha256']!=row['audio_sha256']]
        chosen=min(candidates,key=lambda r:(abs(r['duration_s']-row['duration_s']),int(r['source_id'].rsplit('-',1)[-1])))
        donor[identifier]=chosen['id']
    allowed=[r for r in manifest if r['source_id'] in train+primary+[illustrative]]
    assert len(allowed)==88
    inventory=json.loads((art/'readout-result.json').read_text(encoding='utf-8'))['cache_inventory']
    caches={r['id']:r for r in inventory if r['status']=='available'}
    assert {r['id'] for r in allowed}<=caches.keys()
    wave_inputs=[]
    for row in allowed:
        audio=HERE.parent/'data'/row['audio']
        assert sha(audio)==row['audio_sha256']
        with wave.open(str(audio),'rb') as f:
            assert f.getnchannels()==1 and f.getsampwidth()==2 and f.getframerate()==16000
            assert abs(f.getnframes()/16000-row['duration_s'])<1e-6
        wave_inputs.append(dict(id=row['id'],audio_file=str(audio.resolve()),audio_sha256=row['audio_sha256']))
    head=json.loads((art/'headroom-result.json').read_text(encoding='utf-8'))['rows']
    readout=json.loads((art/'readout-result.json').read_text(encoding='utf-8'))['rows']
    baseline=[]
    for row in head:
        if row['id'] in eval_ids and row['arm']=='B_asr_text':
            baseline.append(dict(id=row['id'],arm='text',answer=row['answer'],transcript=row['transcript_used']))
        if row['id'] in eval_ids and row['source_id']!=illustrative and row['arm']=='D_existing_adapter':
            baseline.append(dict(id=row['id'],arm='real',answer=row['answer'],transcript=row['transcript_used']))
    baseline += [dict(id=r['id'],arm='real',answer=r['answer'],transcript=r['transcript_used'])
                 for r in readout if r['arm']=='D_hybrid_fallback']
    assert len(baseline)==48
    source_files=['connector.py','readout_candidates.py','scale_worker.py','scale_runner.py','prepare_scale.py',
        'scale_protocol.md','artifacts/scale-scientific-decision.json','artifacts/pilot-result.json',
        'artifacts/headroom-result.json','artifacts/readout-result.json']
    for extra in ['scale_analysis.py','reconcile_scale_labels.py']:
        if (HERE/extra).exists():source_files.append(extra)
    missing=[p for p in source_files if not (HERE/p).exists()]
    config=dict(run_id='scale-calibration-v1',seed=17,max_new_tokens=48,
        pilot_checkpoint_sha256='8474698e8123d52e4ea96e310691062e49ce76f62d9e924d1976102ffbad2345',
        historical_manifest_sha256=sha(mp),train_source_ids=train,preflight_source_ids=preflight,
        primary_source_ids=primary,illustrative_source_id=illustrative,donor_by_view=donor,
        donor_alignment='linear_interpolation_of_complete_valid_encoder_sequence_align_corners_false',
        inputs=wave_inputs,cache_sha256_by_view={r['id']:caches[r['id']]['cache_sha256'] for r in allowed},
        baseline_rows=baseline,baseline_rows_semantic_sha256=hashlib.sha256(json.dumps(baseline,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        alphas=decision['alphas'],preflight_logprob_threshold=.1,preflight_min_sources=2,
        fit=decision['fit'],resource_limits=dict(gpu='A10',gpu_count=1,cpu=2,memory_gib=16,function_s=1800,startup_s=300,queue_s=120,retries=0),
        reservation_usd=2,worst_case_cost_usd=decision['bound_usd'],fresh_test_loaded=False,
        optimized_parameter_count=1,primary_eval_views=22,illustrative_eval_views=2,
        source_sha256={n:sha(HERE/n) for n in source_files if (HERE/n).exists()})
    draft=dict(state='prospective_implementation_review_pending',created_unix=time.time(),config=config,
        missing_implementation_files=missing,scientific_decision=decision,
        source_claim='Historical source-disjoint calibration/evaluation; development has prior exposure, not fresh generalization',
        annotation_policy='Stable previous case judgments reused; new cases independently arm-masked model-rated; disputes preserved as ambiguous; strict sensitivity rejects accepted ambiguous answers.',
        stopping='Sensitivity fail/incomplete stops before fit/eval; otherwise fixed32updates+120evalresponses then review. No score-driven rerun.')
    (art/'scale-config-draft.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='draft_no_GPU',training_sources=32,primary_sources=11,illustrative_sources=1,waves=len(wave_inputs),
        preflight_sources=preflight,donor_views=len(donor),missing_implementation_files=missing),indent=2))


if __name__=='__main__':main()
