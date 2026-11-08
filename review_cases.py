"""Blind, deduplicated claim-review packets; propagate decisions to identical outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from score import load, unique


def dump(path,rows):
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("manifest")
    cli.add_argument("output",type=Path)
    cli.add_argument("predictions",nargs="+",type=Path)
    cli.add_argument("--labels",type=Path,help="Reviewer JSONL with case, correct, hallucinated, reason; applies labels to exact raw outputs")
    a=cli.parse_args()
    refs=unique(load(a.manifest))
    a.output.mkdir(parents=True,exist_ok=True)
    cases={}
    mappings=[]
    for path in a.predictions:
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        pred=unique(load(path))
        if not pred or len({r["mode"] for r in pred.values()})!=1:
            raise ValueError("Expected non-empty single-mode prediction files")
        for key,row in pred.items():
            ref=refs[key]
            if row["audio_sha256"]!=ref["audio_sha256"]:
                raise ValueError("Mismatched audio")
            case=hashlib.sha256(json.dumps([ref["source_id"],row["answer"]],ensure_ascii=False).encode()).hexdigest()[:16]
            entry={"case":case,"evidence":ref["transcript"],"question":ref["question"],"reference":ref["answers"],"answer":row["answer"]}
            if case in cases and cases[case]!=entry:
                raise ValueError("Inconsistent source or hash collision")
            cases[case]=entry
            mappings.append({"id":key,"case":case,"mode":row["mode"],"prediction_sha256":sha})
    if not a.labels:
        dump(a.output/"blinded-cases.jsonl",[cases[k] for k in sorted(cases)])
        dump(a.output/"director-mapping.jsonl",mappings)
        return
    rows=load(a.labels)
    labels={r["case"]:r for r in rows}
    if len(labels)!=len(rows) or set(labels)!=set(cases):
        raise ValueError("Reviewed case keys must exactly match")
    for r in rows:
        if not r.get("reason") or type(r["correct"])!=bool or type(r["hallucinated"])!=bool:
            raise ValueError("Expected boolean labels and a rationale")
    for m in sorted({r["mode"] for r in mappings}):
        out=[dict(row,**{k:labels[row["case"]][k] for k in ("correct","hallucinated","reason")}) for row in mappings if row["mode"]==m]
        if len({r["prediction_sha256"] for r in out})!=1:
            raise ValueError("One prediction file per mode required")
        dump(a.output/(m+"-ratings.jsonl"),out)


if __name__=="__main__":
    main()

