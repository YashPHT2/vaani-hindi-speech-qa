"""Offline verification and acoustic-control preference contrasts; no training."""
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


def main():
    art=HERE/'artifacts'
    path=art/'omission-result.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    job=json.loads((art/'omission-job.json').read_text(encoding='utf-8'))
    freeze=json.loads((art/'omission-freeze-v1.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==job['result_sha256']
    assert result['config']==freeze['config'] and result['training_updates']==0 and not result['fresh_test_loaded']
    assert all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h for n,h in freeze['config']['source_sha256'].items())
    assert finite(result)
    rows=result['rows']
    assert len(rows)==6 and len({r['source_id'] for r in rows})==1
    assert {(r['condition'],r['control']) for r in rows}=={(c,k) for c in ['clean','noise_10db'] for k in ['actual','silence','mismatch']}
    token=result['tokenization']
    for row in rows:
        for name,score in row['scores'].items():
            assert score['lexical_token_ids']==token['candidates'][name]
            lp=score['lexical_token_logp']
            assert len(lp)==score['lexical_token_count']
            assert abs(sum(lp)-score['lexical_sum_logp'])<1e-3
            assert abs(score['lexical_sum_logp']+score['eos_logp']-score['sequence_sum_logp'])<1e-3
        for metric,pref in row['preference'].items():
            assert abs(pref-(row['scores']['plus'][metric]-row['scores']['minus'][metric]))<1e-8
    paired=[]
    for condition in ['clean','noise_10db']:
        by={r['control']:r for r in rows if r['condition']==condition}
        actual=by['actual']['preference']
        paired.append(dict(condition=condition,actual_preference=actual,
            silence_preference=by['silence']['preference'],mismatch_preference=by['mismatch']['preference'],
            actual_minus_silence={k:actual[k]-by['silence']['preference'][k] for k in actual},
            actual_minus_mismatch={k:actual[k]-by['mismatch']['preference'][k] for k in actual}))
    summary=dict(status='verified',result_sha256=job['result_sha256'],independent_sources=1,conditions=2,
        input_audio_records=6,candidate_forward_scores=12,training_updates=0,fresh_predictions=0,
        lexical_token_counts={k:len(v) for k,v in token['candidates'].items()},token_regions=token['regions'],
        comparisons=paired,elapsed_s=result['elapsed_s'],peak_cuda_bytes=result['peak_cuda_bytes'],
        interpretation='Exploratory frozen-decoder synthetic-deletion sensitivity only',
        limitations=['Negative is synthetic omission, not literal ASRalternative; token lengths differ.',
            'Full-recording controls alter all acoustic content; no target-localization proof.',
            'User confirmed target phrase only; publisher context not fully transcribed by human.',
            'No encoder-absence, learned-interface, QA-improvement or generalization conclusion from one source.'])
    (art/'omission-analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
