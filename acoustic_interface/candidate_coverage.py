"""Audit physical candidate/selected frame coverage; no training or inference."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def covered(intervals,start,end):
    clipped=sorted((max(start,a),min(end,b)) for a,b in intervals if max(start,a)<min(end,b))
    merged=[]
    for a,b in clipped:
        if merged and a<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],b)
        else:
            merged.append([a,b])
    return dict(seconds=sum(b-a for a,b in merged),fraction=sum(b-a for a,b in merged)/(end-start),intervals=merged)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=float,default=2.0)
    parser.add_argument('--end',type=float,default=3.0)
    parser.add_argument('--validated',action='store_true',help='Use only after a separately preserved user boundary confirmation')
    args=parser.parse_args()
    assert 0<=args.start<args.end<=13.68
    path=HERE/'artifacts/pilot-result.json'
    pilot=json.loads(path.read_text(encoding='utf-8'))
    rows=[r for r in pilot['rows'] if r['source_id']=='fleurs-hi-1806' and r['arm']=='full' and r['control']=='correct']
    assert len(rows)==2
    reports=[]
    for row in rows:
        spans=[s for s in row['span_audit'] if 'frame_start' in s and 'discarded' not in s]
        assert len(spans)==row['valid_spans']
        assert all(0<=s['frame_start']<s['frame_end']<=684 for s in spans)
        assert len(row['selected'])<=4 and all(0<=i<len(spans) for i in row['selected'])
        intervals=[(s['frame_start']*.02,s['frame_end']*.02) for s in spans]
        selected=[intervals[i] for i in row['selected']]
        report=dict(id=row['id'],candidate_count=len(spans),discarded_spans=len(row['span_audit'])-len(spans),
            selected_indices=row['selected'],memory_vectors=len(row['selected']),
            nominal_target=[args.start,args.end],boundaries_listening_validated=args.validated,
            candidate_coverage=covered(intervals,args.start,args.end),selected_coverage=covered(selected,args.start,args.end),
            all_candidate_union=covered(intervals,0,13.68),
            candidates_overlapping_nominal_target=[dict(index=i,start=a,end=b,text=spans[i]['text'],selected=i in row['selected'])
                for i,(a,b) in enumerate(intervals) if max(a,args.start)<min(b,args.end)],
            selected_intervals=[dict(index=i,start=intervals[i][0],end=intervals[i][1],text=spans[i]['text']) for i in row['selected']],
            causal_scope='Physical availability of stored feature frames only; not semantic content or successful readout')
        reports.append(report)
    summary=dict(status='stored_candidate_audit_complete',pilot_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_id='fleurs-hi-1806',independent_sources=1,interval_status='validated_user_boundaries' if args.validated else 'approximate_unvalidated_2to3second_annotation',
        frame_period_s=.02,indexing='Filterdiscardedspans beforeapplying storedselectedindices; halfopen frame intervals',
        rows=reports,limitations=['Predicted DTW timestamp labels do not prove omitted-word localization.',
            'Zero candidates entails zero access for every nonempty interval, independent of uncertain boundaries.',
            'Positive frame coverage does not prove target information survives pooling or Qwen readout.'])
    out=HERE/'artifacts/candidate-coverage.json'
    out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
