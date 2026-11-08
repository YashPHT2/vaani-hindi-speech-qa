"""Paired B-minus-A intervals by source for complete A/B prediction files."""
import argparse
import json
import math
import random
from collections import defaultdict
from score import edits, load, norm, unique


def counts(ref,p):
    truth=norm(ref["transcript"]).split()
    return (edits(truth,norm(p["hypothesis"]).split()),len(truth),
            int(norm(p["answer"]) in {norm(a) for a in ref["answers"]}),1)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("manifest")
    cli.add_argument("baseline")
    cli.add_argument("intervention")
    cli.add_argument("--split",required=True,choices=["dev","test"])
    args=cli.parse_args()
    refs=unique([r for r in load(args.manifest) if r["split"]==args.split])
    a,b=unique(load(args.baseline)),unique(load(args.intervention))
    if not refs or set(refs)!=set(a) or set(a)!=set(b):
        raise ValueError("Complete matched splits required")
    if any(a[k]["audio_sha256"]!=b[k]["audio_sha256"] or a[k]["audio_sha256"]!=r["audio_sha256"] for k,r in refs.items()):
        raise ValueError("Audio pairs disagree")
    tags=["all",*sorted({r["condition"] for r in refs.values()})]
    reports={}
    for tag in tags:
        groups=defaultdict(list)
        for key,r in refs.items():
            if tag=="all" or tag==r["condition"]:
                groups[r["source_id"]].append((counts(r,a[key]),counts(r,b[key])))
        ids=sorted(groups)
        def delta(sample):
            pairs=[pair for s in sample for pair in groups[s]]
            def metric(side,i):
                nums=[p[side] for p in pairs]
                return sum(p[i] for p in nums)/sum(p[i+1] for p in nums)
            return (metric(1,0)-metric(0,0),metric(1,2)-metric(0,2))
        rng=random.Random(17)
        trials=[delta(rng.choices(ids,k=len(ids))) for _ in range(2000)]
        report={"unique_sources":len(ids)}
        for i,key in enumerate(("wer_delta","answer_em_delta")):
            samples=sorted(d[i] for d in trials)
            report[key]={"estimate":delta(ids)[i],"ci95":[samples[math.floor(.025*(len(samples)-1))],
                                                         samples[math.ceil(.975*(len(samples)-1))]]}
        reports[tag]=report
    print(json.dumps(reports,indent=2))


if __name__=="__main__":
    main()

