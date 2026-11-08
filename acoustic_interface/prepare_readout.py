"""Prepare the lead-approved historical inference draft; no compute or reservation."""
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    art=HERE/'artifacts'
    assert not any((art/n).exists() for n in ['readout-job.json','readout-config.json','readout-result.json'])
    decision=json.loads((art/'readout-scientific-decision.json').read_text(encoding='utf-8'))
    assert decision['expected_rows']==12 and decision['training_updates']==0 and decision['fresh_test']=='sealed'
    old=json.loads((art/'omission-freeze-v1.json').read_text(encoding='utf-8'))['config']
    bounds=json.loads((art/'localization-boundaries-validated.json').read_text(encoding='utf-8'))
    assert bounds['target_mask_interval_s']==[2.9,4.1] and bounds['full_phrase_extent_verified'] is False
    assert [round(t/.02) for t in bounds['target_mask_interval_s']]==[145,205]
    manifest=HERE.parent/'data/manifest.jsonl'
    rows=[json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==224 and len({r['source_id'] for r in rows})==112
    names=['connector.py','readout_candidates.py','readout_worker.py','readout_runner.py','prepare_readout.py',
        'readout_protocol_draft.md','readout_scientific_decision.md','artifacts/readout-scientific-decision.json',
        'artifacts/localization-boundaries-validated.json','artifacts/omission-freeze-v1.json']
    if (HERE/'readout_analysis.py').exists():
        names.append('readout_analysis.py')
    headroom=json.loads((art/'headroom-result.json').read_text(encoding='utf-8'))
    canonical=json.dumps(headroom,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    linux_bytes=json.dumps(headroom,ensure_ascii=False,indent=2).encode('utf-8')
    config=dict(run_id='readout-fallback-dev2-v1',seed=17,max_new_tokens=48,
        pilot_checkpoint_sha256=decision['checkpoint_sha256'],inputs=old['inputs'],
        historical_manifest_sha256=sha(manifest),headroom_result_sha256=sha(art/'headroom-result.json'),
        headroom_semantic_sha256=hashlib.sha256(canonical).hexdigest(),
        headroom_remote_lf_expected_sha256=hashlib.sha256(linux_bytes).hexdigest(),
        reported_window_frame_bounds=[145,205],reported_bin_edges=[145,160,175,190,205],
        full_phrase_extent_verified=False,fallback_window_frames=20,fallback_only_when_empty=True,
        arms=['A_text','B_existing_DTW','C_reported_window','D_hybrid_fallback','E_oracle_mismatch','E_fallback_mismatch'],
        whisper=dict(model='openai/whisper-small',revision=old['model_revision'],dtype='float16',attention='eager (as original feature extraction)'),
        llm=dict(model='Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',dtype='float16',attention='sdpa'),
        uncertainty_for_new_windows='zeros_and_availability_false',generation_use_cache=True,
        resource_limits=old['resource_limits'],reservation_usd=1,worst_case_cost_usd=.63168344,
        independent_sources=1,expected_rows=12,training_updates=0,fresh_test_loaded=False,
        source_sha256={n:sha(HERE/n) for n in names})
    draft=dict(state='prospective_technical_review_pending',created_unix=time.time(),config=config,
        scientific_decision=decision,primary_availability=decision['primary_availability'],
        primary_utility=decision['primary_utility'],stopping=decision['stopping'],
        scoring_policy='Generate first without answer; separate gold-answer CE afterward, no teacher target cache reused',
        label_policy='Independent model semantic review; preserve ambiguities and old case hashes; no human-QA claim',
        audit_scope='224 historical IDs: missing remains missing, no fresh or unallowlisted cache inspection, no re-ASR or training')
    (art/'readout-config-draft.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='draft_prepared_no_GPU',responses=12,primary_source='fleurs-hi-1806',fresh_loaded=False)))


if __name__=='__main__':
    main()
