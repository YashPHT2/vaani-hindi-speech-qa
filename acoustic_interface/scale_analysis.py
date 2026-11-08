"""Offline source-level scale diagnostics and stable blinded annotation packets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from score import norm
from scale_worker import normalized_answer


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def key(r):
    return hashlib.sha256(json.dumps([r['source_id'],r['question'],r['answer']],ensure_ascii=False).encode()).hexdigest()[:20]


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def source_gate(sources,eval_rows,labels,strict):
    changes=[]
    def correct(r):
        label=labels[key(r)]
        return bool(label['correct'] and not label['hallucinated'] and not(strict and label.get('ambiguous',False)))
    for source in sources:
        subset=[r for r in eval_rows if r['source_id']==source]
        by={(r['condition'],r['arm']):r for r in subset}
        assert len(by)==10
        actual=donor=0
        detail=[]
        for condition in ['clean','noise_10db']:
            values={arm:correct(by[condition,arm]) for arm in ['text','real_one','real_fitted','wrong_one','wrong_fitted']}
            ad=int(values['real_fitted'])-int(values['real_one'])
            wd=int(values['wrong_fitted'])-int(values['wrong_one'])
            actual+=ad/2;donor+=wd/2
            detail.append(dict(condition=condition,correct=values,actual_delta=ad,donor_delta=wd))
        changes.append(dict(source_id=source,actual_change=actual,donor_change=donor,actual_minus_donor_change=actual-donor,views=detail))
    positive_specific=sum(r['actual_change']>0 and r['actual_change']>r['donor_change'] for r in changes)
    negative=sum(r['actual_change']<0 for r in changes)
    total=sum(r['actual_change'] for r in changes)
    difference=sum(r['actual_minus_donor_change'] for r in changes)
    return dict(strict_ambiguity=strict,source_changes=changes,positive_specific_sources=positive_specific,
        regressed_sources=negative,sum_actual_source_equivalents=total,sum_actual_minus_donor_source_equivalents=difference,
        passed=positive_specific>=2 and negative<=1 and total>=1 and difference>=1,
        full_primary_view_denominator=22)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',type=Path,help='Independently reviewed/reconciled new case labels; preserve prior judgments')
    args=parser.parse_args()
    art=HERE/'artifacts'
    result=json.loads((art/'scale-result.json').read_text(encoding='utf-8'))
    job=json.loads((art/'scale-job.json').read_text(encoding='utf-8'))
    assert sha(art/'scale-result.json')==job['result_sha256']
    config=result['config']
    assert config['fresh_test_loaded'] is False
    proof=result['weight_invariance']
    assert proof['checkpoint_before_sha256']==proof['checkpoint_after_sha256']==config['pilot_checkpoint_sha256']
    assert proof['adapter_state_unchanged'] and proof['lm_asr_adapter_parameter_versions_unchanged'] and proof['backbone_gradients_none']
    assert proof['detached_memory_bytes_unchanged']
    assert result['backbone_training_updates']==0
    for name,digest in config['source_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else HERE/name
        assert sha(path)==digest
    pre=result['preflight_rows']
    if result['preflight_gate'].get('incomplete'):
        assert not pre and not result.get('fit_trace') and not result.get('eval_rows')
        output=dict(status=result['status'],result_sha256=job['result_sha256'],preflight_gate_passed=False,
            preflight_gate_incomplete=True,details=result['preflight_gate'],decision='Stop; no replacements, fitting or dev generation.',fresh_predictions=0)
        (art/'scale-analysis.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(output,indent=2));return
    assert len(pre)==40
    alphas=config['alphas']
    preflight=[]
    for source in config['preflight_source_ids']:
        source_rows=[r for r in pre if r['source_id']==source]
        assert len(source_rows)==10
        shifts=[];changed=False
        for condition in ['clean','noise_10db']:
            by={float(r['alpha']):r for r in source_rows if r['condition']==condition}
            assert set(by)==set(map(float,alphas))
            assert all(math.isfinite(r['first_token_logprob']) for r in by.values())
            shifts.append(max(abs(r['first_token_logprob']-by[1.]['first_token_logprob']) for r in by.values()))
            changed|=any(normalized_answer(r['answer'])!=normalized_answer(by[1.]['answer']) for r in by.values())
        preflight.append(dict(source_id=source,mean_max_logprob_change=sum(shifts)/2,normalized_generation_changed=changed))
    passed=sum(r['mean_max_logprob_change']>=.1 for r in preflight)>=2 or sum(r['normalized_generation_changed'] for r in preflight)>=2
    assert passed==result['preflight_gate']['passed']
    output=dict(status=result['status'],preflight_gate_passed=passed,preflight_sources=preflight,
        result_sha256=job['result_sha256'],fresh_predictions=0,
        interpretation='Sensitivity alone is not answer utility; calibration/evaluation conditional on the prespecified gate.')
    if not passed:
        assert not result.get('fit_trace') and not result.get('eval_rows')
        output['decision']='Stop scalar branch at sensitivity gate; no fitting or dev generation.'
    else:
        trace=result['fit_trace'];ev=result['eval_rows']
        assert len(trace)==32 and len(ev)==120
        assert all(math.isfinite(x['mean_loss']) and math.isfinite(x['theta_gradient']) and .03125-1e-6<=x['alpha_after']<=2+1e-6 for x in trace)
        guards_ok=not result['baseline_guard_mismatch']
        output['baseline_guards_passed']=guards_ok
        output.update(fitted_alpha=result['fitted_alpha'],optimizer_updates=32,primary_views=22,illustrative_views=2)
        previous={}
        # Preserve earlier correctness/hallucination conflicts as ambiguity.
        primary_old={r['case']:r for r in rows(art/'headroom-review/model-labels.jsonl')}
        for name in ['headroom-review/independent-labels.jsonl','token-limit-review/independent-labels.jsonl','readout-review/independent-labels.jsonl']:
            for label in rows(art/name):
                old=primary_old.get(label['case'])
                combined=dict(label)
                if old and any(old[k]!=label[k] for k in ['correct','hallucinated']):
                    combined['ambiguous']=True
                    combined['correct']=False
                    combined['hallucinated']=old['hallucinated'] or label['hallucinated']
                    combined['prior_reviewer_disagreement']=True
                    combined['prior_primary_original']=old
                    combined['prior_independent_original']=label
                previous[label['case']]=combined
        manifest={r['id']:r for r in rows(HERE.parent/'data/manifest.jsonl')}
        packet=art/'scale-review';packet.mkdir(exist_ok=True)
        cases={}
        for row in ev:
            k=key(row)
            if k not in previous:
                ref=manifest[row['id']]
                cases[k]=dict(case=k,source_id=row['source_id'],question=row['question'],answer=row['answer'],reference=ref['answers'],evidence=ref['transcript'])
        (packet/'new-blinded-cases.jsonl').write_text(''.join(json.dumps(cases[k],ensure_ascii=False)+'\n' for k in sorted(cases)),encoding='utf-8')
        (packet/'private-mapping.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],arm=r['arm'],case=key(r)))+'\n' for r in ev),encoding='utf-8')
        output['new_distinct_review_cases']=len(cases)
        output['normalized_em']={arm:sum(norm(r['answer']) in {norm(a) for a in manifest[r['id']]['answers']} for r in ev
            if r['source_id'] in config['primary_source_ids'] and r['arm']==arm) for arm in ['text','real_one','real_fitted','wrong_one','wrong_fitted']}
        if args.labels:
            labels=rows(args.labels);new={r['case']:r for r in labels}
            assert len(labels)==len(new) and new.keys()==cases.keys()
            assert all(type(x['correct']) is bool and type(x['hallucinated']) is bool and x['reason'] for x in labels)
            merged={**previous,**new}
            primary=[r for r in ev if r['source_id'] in config['primary_source_ids']]
            reconciled=source_gate(config['primary_source_ids'],primary,merged,False)
            strict=source_gate(config['primary_source_ids'],primary,merged,True)
            output.update(reconciled=reconciled,strict=strict,utility_gate_passed=guards_ok and reconciled['passed'] and strict['passed'],
                labels_sha256=sha(args.labels))
    (art/'scale-analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
