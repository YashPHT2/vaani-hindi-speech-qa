"""Conservative, provenance-preserving model-label reconciliation; no model call."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--independent',type=Path,required=True)
    args=parser.parse_args()
    a={r['case']:r for r in read(args.primary)}
    b={r['case']:r for r in read(args.independent)}
    assert a.keys()==b.keys()
    expected={r['case'] for r in read(HERE/'artifacts/scale-review/new-blinded-cases.jsonl')}
    assert a.keys()==expected
    output=[];disagreements=[]
    for k in sorted(a):
        x,y=a[k],b[k]
        conflict=any(x[field]!=y[field] for field in ['correct','hallucinated'])
        row=dict(case=k,correct=x['correct'] and y['correct'] and not conflict,
            hallucinated=x['hallucinated'] or y['hallucinated'],
            ambiguous=conflict or x.get('ambiguous',False) or y.get('ambiguous',False),
            reason=('Reviewer disagreement; conservative incorrect/ambiguous. ' if conflict else 'Reviewers agree. ')+
                   'Primary: '+x['reason']+' Independent: '+y['reason'],
            reviewer_type='Conservative reconciliation of two model reviews; no human QA validation',
            primary_original=x,independent_original=y)
        if conflict:disagreements.append(k)
        output.append(row)
    target=HERE/'artifacts/scale-review/reconciled-labels.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    provenance=dict(new_cases=len(output),disagreement_case_ids=disagreements,
        primary_sha256=hashlib.sha256(args.primary.read_bytes()).hexdigest(),
        independent_sha256=hashlib.sha256(args.independent.read_bytes()).hexdigest(),
        reconciled_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        rule='Correctness/hallucination disagreement -> incorrect and ambiguous; accepted ambiguity rejected by strict analysis; no cases dropped.')
    (target.parent/'reconciliation.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
