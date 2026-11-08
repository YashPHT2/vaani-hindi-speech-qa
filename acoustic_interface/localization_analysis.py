"""Offline verification and paired target/placebo contrasts; no new model calls."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def finite(value):
    if isinstance(value,float):
        return math.isfinite(value)
    if isinstance(value,dict):
        return all(finite(x) for x in value.values())
    if isinstance(value,list):
        return all(finite(x) for x in value)
    return True


def contrasts(values):
    td=values['original']-values['target_masked']
    pd=values['original']-values['placebo_masked']
    return dict(preference=values,target_drop=td,placebo_drop=pd,
                target_minus_placebo_drop=td-pd,original_minus_mismatch=values['original']-values['mismatch'])


def main():
    art=HERE/'artifacts'
    rp=art/'localization-result.json'
    result=json.loads(rp.read_text(encoding='utf-8'))
    job=json.loads((art/'localization-job.json').read_text(encoding='utf-8'))
    freeze=json.loads((art/'localization-freeze-v1.json').read_text(encoding='utf-8'))
    assert job['phase']=='complete' and hashlib.sha256(rp.read_bytes()).hexdigest()==job['result_sha256']
    assert result['status']=='localized_omission_sensitivity_complete' and result['config']==freeze['config']
    assert result['training_updates']==0 and result['fresh_test_loaded'] is False and finite(result)
    assert result['forward_counts']['acoustic_encoder_forwards']==8
    assert result['forward_counts']['full_candidate_decoder_forwards']==16
    assert result['forward_counts']['shared_prefix_decoder_forwards']==8
    assert result['forward_counts']['total_decoder_forwards']==24
    for name,digest in freeze['config']['source_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else HERE/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    rows=result['rows']
    controls=['original','target_masked','placebo_masked','mismatch']
    assert len(rows)==8 and len({r['source_id'] for r in rows})==1
    assert {(r['condition'],r['control']) for r in rows}=={(c,k) for c in ['clean','noise_10db'] for k in controls}
    tokens=result['tokenization']
    for row in rows:
        shared=row['shared_prefix']
        assert shared['same_final_distribution'] is True and shared['decoder_forwards']==1
        assert shared['decoder_input_ids']==tokens['shared_prefix_decoder_input_ids']
        assert abs(shared['plus_logp']-shared['minus_logp']-shared['log_odds'])<1e-8
        assert row['primary_shared_prefix_log_odds']==shared['log_odds']
        for name,score in row['scores'].items():
            assert score['lexical_token_ids']==tokens['candidates'][name]
            assert abs(sum(score['lexical_token_logp'])-score['lexical_sum_logp'])<1e-3
            assert abs(score['lexical_sum_logp']+score['eos_logp']-score['sequence_sum_logp'])<1e-3
        for metric,value in row['preference'].items():
            assert abs(value-(row['scores']['plus'][metric]-row['scores']['minus'][metric]))<1e-8
        if row['control'] in ['target_masked','placebo_masked']:
            mask=row['mask_audit']
            assert mask['outside_mask_bitwise_unchanged'] is True and mask['fade_samples_each_edge']==80
            assert mask['masked_float32_sha256']==row['scored_float32_sha256']
            assert mask['sample_count']==mask['sample_bounds_half_open'][1]-mask['sample_bounds_half_open'][0]
        else:
            assert row['mask_audit'] is None
    old=json.loads((art/'omission-result.json').read_text(encoding='utf-8'))
    assert tokens['candidates']==old['tokenization']['candidates'] and tokens['prefix_tokens']==old['tokenization']['prefix_tokens']
    analyses=[]
    for condition in ['clean','noise_10db']:
        by={r['control']:r for r in rows if r['condition']==condition}
        assert by['target_masked']['mask_audit']['sample_count']==by['placebo_masked']['mask_audit']['sample_count']
        metrics={m:contrasts({k:by[k]['preference'][m] for k in controls}) for m in by['original']['preference']}
        shared=contrasts({k:by[k]['primary_shared_prefix_log_odds'] for k in controls})
        prior={r['control']:r for r in old['rows'] if r['condition']==condition}
        replication={k:{m:by[k]['preference'][m]-prior['actual' if k=='original' else k]['preference'][m]
            for m in by[k]['preference']} for k in ['original','mismatch']}
        analyses.append(dict(condition=condition,full_candidate_metrics=metrics,
            single_shared_prefix_first_token=shared,old_original_donor_replication_differences=replication,
            mask_audits={k:by[k]['mask_audit'] for k in ['target_masked','placebo_masked']}))
    summary=dict(status='verified',result_sha256=job['result_sha256'],independent_sources=1,acoustic_views=2,
        target_mask_interval_s=freeze['config']['target_mask_interval_s'],
        placebo_mask_interval_s=freeze['config']['placebo_mask_interval_s'],comparisons=analyses,
        forward_counts=result['forward_counts'],elapsed_s=result['elapsed_s'],peak_cuda_bytes=result['peak_cuda_bytes'],
        interpretation='Signed sensitivity of declared waveform windows; no isolated word representation or QA improvement claim',
        limitations=['One historical source, two correlated acoustic views; no population significance.',
            'Unequal synthetic-deletion candidate lengths remain; summed and mean scores are separate.',
            'Masking changes contextual acoustics, energy and alignment; target/placebo speech content differs.',
            'Shared-prefix odds use one forward distribution; old per-candidate firsttoken metrics remain secondary.',
            'Boundary provenance and any incomplete-phrase caveat in the prospective freeze remain controlling.'])
    (art/'localization-analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=summary['status'],target=summary['target_mask_interval_s'],placebo=summary['placebo_mask_interval_s'],
        result_sha256=summary['result_sha256'],comparisons=[dict(condition=r['condition'],
            summed_sequence=r['full_candidate_metrics']['sequence_sum_logp'],
            mean_sequence=r['full_candidate_metrics']['sequence_mean_logp'],
            shared_prefix=r['single_shared_prefix_first_token']) for r in analyses]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
