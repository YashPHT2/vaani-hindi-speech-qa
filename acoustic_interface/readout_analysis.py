"""Offline readout, stable annotation packet, and historical cache availability audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from score import norm,edits


def case_key(row):
    return hashlib.sha256(json.dumps([row['source_id'],row['question'],row['answer']],ensure_ascii=False).encode()).hexdigest()[:20]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',type=Path)
    args=parser.parse_args()
    art=HERE/'artifacts'
    path=art/'readout-result.json'
    r=json.loads(path.read_text(encoding='utf-8'))
    job=json.loads((art/'readout-job.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==job['result_sha256']
    assert r['training_updates']==0 and r['fresh_test_loaded'] is False
    assert len(r['rows'])==12 and len(r['cache_inventory'])==224
    assert r['checkpoint_before_sha256']==r['checkpoint_after_sha256']==r['config']['pilot_checkpoint_sha256']
    guards_ok=not r['baseline_guard_mismatch'] and not r['integrity_guard_mismatch']
    manifest={x['id']:x for x in map(json.loads,(HERE.parent/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    labels={}
    for p in [art/'headroom-review/independent-labels.jsonl',art/'token-limit-review/independent-labels.jsonl']:
        for x in map(json.loads,p.read_text(encoding='utf-8').splitlines()):
            assert x['case'] not in labels or labels[x['case']]['correct']==x['correct']
            labels[x['case']]=x
    cases={}
    for row in r['rows']:
        k=case_key(row)
        if k not in labels:
            cases[k]=dict(case=k,source_id=row['source_id'],question=row['question'],answer=row['answer'],
                evidence=manifest[row['id']]['transcript'],reference=row['answers'])
    packet=art/'readout-review'
    packet.mkdir(exist_ok=True)
    (packet/'new-blinded-cases.jsonl').write_text(''.join(json.dumps(cases[k],ensure_ascii=False)+'\n' for k in sorted(cases)),encoding='utf-8')
    (packet/'private-mapping.jsonl').write_text(''.join(json.dumps(dict(id=row['id'],arm=row['arm'],case=case_key(row)))+'\n' for row in r['rows']),encoding='utf-8')
    if args.labels:
        new=[json.loads(s) for s in args.labels.read_text(encoding='utf-8').splitlines() if s.strip()]
        by={x['case']:x for x in new}
        assert len(new)==len(by) and by.keys()==cases.keys()
        assert all(type(x['correct']) is bool and type(x['hallucinated']) is bool and x['reason'] for x in new)
        labels.update(by)
    by_condition={c:{x['arm']:x for x in r['rows'] if x['condition']==c} for c in ['clean','noise_10db']}
    summaries=[]
    for condition,arms in by_condition.items():
        a,b,c,d,eo,ef=[arms[n] for n in r['config']['arms']]
        out=dict(condition=condition,arms=[dict(arm=x['arm'],answer=x['answer'],K=x['K'],
            candidate_count=x['coverage']['candidate_count'],selected_frames=x['selected_windows_frames'],
            nll=x['nll'],nll_improvement_over_text=a['nll']-x['nll'],
            exact_match=norm(x['answer']) in {norm(z) for z in x['answers']},
            memory_norm_ratio=x['memory_to_question_embedding_norm_ratio']) for x in arms.values()],
            availability_gate=(d['coverage']['candidate_count']==35 and d['coverage']['physical_union_frames']==684 and d['K']==4) if condition=='noise_10db' else None)
        if args.labels:
            def judged(x):return labels[case_key(x)]
            def valid(x):return judged(x)['correct'] and not judged(x)['hallucinated'] and not judged(x).get('ambiguous',False)
            def wrong(x):return not judged(x)['correct'] and not judged(x).get('ambiguous',False)
            out['semantic']={x['arm']:judged(x) for x in arms.values()}
            out['automatic_specificity_ambiguous']=any(judged(x).get('ambiguous',False) for x in [a,b,d,ef])
            out['privileged_specificity_ambiguous']=any(judged(x).get('ambiguous',False) for x in [a,b,c,eo])
            out['automatic_specific_rescue']=guards_ok and valid(d) and wrong(a) and wrong(b) and wrong(ef)
            out['privileged_specific_rescue']=guards_ok and valid(c) and wrong(a) and wrong(b) and wrong(eo)
            out['automatic_regression_vs_text']=judged(a)['correct'] and not judged(d)['correct']
            out['privileged_regression_vs_text']=judged(a)['correct'] and not judged(c)['correct']
        summaries.append(out)
    audit=[]
    for item in r['cache_inventory']:
        ref=manifest[item['id']]
        out=dict(item,source_id=ref['source_id'])
        if item['status']=='available':
            truth,hyp=norm(ref['transcript']),norm(item['transcript'])
            out['wer']=edits(truth.split(),hyp.split())/len(truth.split())
            out['cer']=edits(truth.replace(' ',''),hyp.replace(' ',''))/len(truth.replace(' ',''))
            out['window_durations_s']=[(hi-lo)*.02 for lo,hi in item['candidate_windows']]
            out['selected_proportion_expected']=min(4,item['valid_span_count'])/item['valid_span_count'] if item['valid_span_count'] else None
        audit.append(out)
    groups=[]
    for split in ['dev','test']:
        for condition in ['clean','noise_10db']:
            views=[x for x in audit if x['split']==split and x['condition']==condition]
            available=[x for x in views if x['status']=='available']
            groups.append(dict(historical_split=split,condition=condition,allowed_views=len(views),
                available=len(available),missing=len(views)-len(available),
                empty_ids=[x['id'] for x in available if x['valid_span_count']==0],
                empty_distinct_source_ids=sorted({x['source_id'] for x in available if x['valid_span_count']==0}),
                median_wer_empty=statistics.median([x['wer'] for x in available if x['valid_span_count']==0]) if any(x['valid_span_count']==0 for x in available) else None,
                median_wer_nonempty=statistics.median([x['wer'] for x in available if x['valid_span_count']>0]) if any(x['valid_span_count']>0 for x in available) else None,
                median_coverage=statistics.median([x['coverage_fraction'] for x in available]) if available else None))
    (art/'readout-cache-audit.json').write_text(json.dumps(dict(views=audit,groups=groups),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary=dict(status='verified_guards_pass' if guards_ok else 'integrity_guard_mismatch_interpretation_held',
        result_sha256=job['result_sha256'],sources=1,views=2,responses=12,training_updates=0,fresh_predictions=0,
        new_distinct_review_cases=len(cases),comparisons=summaries,cache_audit_groups=groups,
        unique_cached_sources=len({x['source_id'] for x in audit if x['status']=='available'}),
        unique_empty_sources=len({x['source_id'] for x in audit if x['status']=='available' and x['valid_span_count']==0}),
        limitation='Existing checkpoint, new candidate distribution, one source; NLL/coverage do not establish generated-answer utility.')
    (art/'readout-analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
