"""Offline paired length analysis, stable case reuse and arm-masked review packet."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from score import norm


def key(r,answer):
    return hashlib.sha256(json.dumps([r['source_id'],r['question'],answer],ensure_ascii=False).encode()).hexdigest()[:20]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def paired(rows,old,new):
    rescued=[r['id'] for r in rows if not old[r['id']] and new[r['id']]]
    regressed=[r['id'] for r in rows if old[r['id']] and not new[r['id']]]
    source_delta={}
    for source in sorted({r['source_id'] for r in rows}):
        views=[r for r in rows if r['source_id']==source]
        source_delta[source]=sum(int(new[r['id']])-int(old[r['id']]) for r in views)/len(views)
    rng=random.Random(17)
    values=list(source_delta.values())
    boot=sorted(100*sum(rng.choices(values,k=len(values)))/len(values) for _ in range(10000))
    return dict(old_correct=sum(old.values()),new_correct=sum(new.values()),views=len(rows),
        rescued=rescued,regressed=regressed,rescued_sources=sorted({r['source_id'] for r in rows if r['id'] in rescued}),
        net_accuracy_delta_pp=100*(len(rescued)-len(regressed))/len(rows),
        source_clustered_bootstrap_95ci_delta_pp=[boot[249],boot[9749]],
        bootstrap=dict(seed=17,replicates=10000,unit='12 source recordings, paired views retained'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',type=Path)
    args=parser.parse_args()
    art=HERE/'artifacts'
    result_path=art/'token-limit-result.json'
    result=load(result_path)
    job=load(art/'token-limit-job.json')
    assert hashlib.sha256(result_path.read_bytes()).hexdigest()==job['result_sha256']
    assert result['interpretation_allowed'] and not result['baseline_prefix_mismatches']
    rows=result['rows']
    assert len(rows)==24 and len({r['id'] for r in rows})==24
    assert all(r['baseline_prefix_match'] and r['memory_tokens']==0 for r in rows)
    packet=art/'token-limit-review'
    packet.mkdir(exist_ok=True)
    old_labels={r['case']:r for r in lines(art/'headroom-review/independent-labels.jsonl')}
    cases={}
    mapping=[]
    for r in rows:
        for cap,answer in ((48,r['old_answer']),(128,r['answer'])):
            case=key(r,answer)
            cases[case]=dict(case=case,source_id=r['source_id'],evidence=r['publisher_evidence'],
                question=r['question'],reference=r['answers'],answer=answer)
            mapping.append(dict(id=r['id'],cap=cap,case=case))
    new_cases={k:v for k,v in cases.items() if k not in old_labels}
    (packet/'new-blinded-cases.jsonl').write_text(''.join(json.dumps(new_cases[k],ensure_ascii=False)+'\n' for k in sorted(new_cases)),encoding='utf-8')
    (packet/'private-mapping.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mapping),encoding='utf-8')
    old_em={r['id']:norm(r['old_answer']) in {norm(a) for a in r['answers']} for r in rows}
    new_em={r['id']:norm(r['answer']) in {norm(a) for a in r['answers']} for r in rows}
    summary=dict(result_sha256=job['result_sha256'],views=24,sources=12,
        old_cap_hit_ids=[r['id'] for r in rows if r['old_cap_hit']],
        old_cap_hit_sources=sorted({r['source_id'] for r in rows if r['old_cap_hit']}),
        new_cap_hit_ids=[r['id'] for r in rows if r['finish']=='length'],
        changed_responses=[r['id'] for r in rows if r['answer']!=r['old_answer']],
        new_distinct_review_cases=len(new_cases),all_first48_prefixes_match=True,
        normalized_em=paired(rows,old_em,new_em),
        limitations=['Historical dev only, no fresh test','Only C128 run; other arms remain C48-matched, not C128-matched',
                    'Any semantic labels are model judgments, not human/audio validation',
                    'Intervals with12sources are exploratory; two acoustic views are not independent'])
    if args.labels:
        new_labels=lines(args.labels)
        by_case={r['case']:r for r in new_labels}
        assert len(by_case)==len(new_labels) and by_case.keys()==new_cases.keys()
        assert all(type(r['correct']) is bool and type(r['hallucinated']) is bool and r['reason'] for r in new_labels)
        merged={**old_labels,**by_case}
        old={r['id']:merged[key(r,r['old_answer'])]['correct'] for r in rows}
        new={r['id']:merged[key(r,r['answer'])]['correct'] for r in rows}
        semantic=paired(rows,old,new)
        semantic['old_hallucinated']=sum(merged[key(r,r['old_answer'])]['hallucinated'] for r in rows)
        semantic['new_hallucinated']=sum(merged[key(r,r['answer'])]['hallucinated'] for r in rows)
        rescued_capped_sources={r['source_id'] for r in rows if r['old_cap_hit'] and not old[r['id']] and new[r['id']]}
        semantic['rescued_previously_capped_sources']=sorted(rescued_capped_sources)
        semantic['prospective_material_truncation_gate_pass']=(len(rescued_capped_sources)>=2
            and semantic['new_correct']>semantic['old_correct']
            and semantic['new_hallucinated']<=semantic['old_hallucinated'])
        semantic['answerable']={str(cap):dict(correct=sum(values[r['id']] for r in rows if r['answers']!=['अज्ञात']),
            views=sum(r['answers']!=['अज्ञात'] for r in rows)) for cap,values in ((48,old),(128,new))}
        semantic['labels_sha256']=hashlib.sha256(args.labels.read_bytes()).hexdigest()
        summary['semantic']=semantic
    (art/'token-limit-analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
