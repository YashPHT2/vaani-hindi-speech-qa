"""Join frozen references to run outputs. No inference or network access."""
import argparse
import json
import math
import statistics
import unicodedata
from pathlib import Path


def norm(text):
    chars = []
    for c in unicodedata.normalize("NFC", text.casefold()):
        cat = unicodedata.category(c)
        if cat[0] == "P":
            c = " "
        elif cat == "Nd":
            c = str(unicodedata.decimal(c))
        chars.append(c)
    return " ".join("".join(chars).split())


def edits(a,b):
    last = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row = [i]
        for j,y in enumerate(b,1):
            row.append(min(last[j]+1,row[-1]+1,last[j-1]+(x!=y)))
        last = row
    return last[-1]


def load(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def unique(rows):
    result = {r["id"]:r for r in rows}
    if len(result)!=len(rows):
        raise ValueError("Duplicate IDs")
    return result


def summarize(pairs, ratings):
    n = len(pairs)
    if not n:
        raise ValueError("Empty set")
    word_err = char_err = words = chars = exact = timed_asr = abstained = empty = truncated = 0
    latencies = []
    llm_times = []
    asr_times = []
    duration = 0.0
    for ref,p in pairs:
        truth,hyp = norm(ref["transcript"]),norm(p["hypothesis"])
        if p["mode"] != "oracle":
            if not truth:
                raise ValueError("Empty reference transcript")
            word_err += edits(truth.split(),hyp.split())
            words += len(truth.split())
            t,h = truth.replace(" ",""), hyp.replace(" ","")
            char_err += edits(t,h)
            chars += len(t)
            timed_asr += 1
            asr_times.append(p["asr_s"])
        if not ref["answers"] or not all(norm(a) for a in ref["answers"]):
            raise ValueError("Accepted answers must be nonempty")
        exact += norm(p["answer"]) in {norm(a) for a in ref["answers"]}
        abstained += norm(p["answer"]) in {"अज्ञात","unknown"}
        empty += not norm(p["answer"])
        truncated += p.get("finish") == "length"
        latencies.append(p["total_s"])
        llm_times.append(p["llm_s"])
        duration += p["seconds_audio"]
    output = {"n":n, "exact_matches":exact, "answer_em":exact/n,"wer":word_err/words if words else None,
              "cer":char_err/chars if chars else None, "abstained":abstained,"empty":empty,"truncated":truncated,
              "word_edits":word_err, "reference_words":words,
              "character_edits":char_err,"reference_characters":chars,"asr_scored":timed_asr,
              "p50_completed_text_s":statistics.median(latencies),"p95_completed_text_s":sorted(latencies)[math.ceil(.95*n)-1],
              "median_asr_s":statistics.median(asr_times) if asr_times else None,"median_llm_s":statistics.median(llm_times),
              "pipeline_rtf":sum(latencies)/duration if timed_asr==n else None}
    if ratings is not None:
        for key in ("correct","hallucinated"):
            vals = [ratings[r["id"]][key] for r,_ in pairs]
            if any(type(v) is not bool for v in vals):
                raise ValueError("Adjudicated labels must be true/false")
            output[key+"_count"] = sum(vals)
            output[key+"_rate"] = sum(vals)/n
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("predictions")
    p.add_argument("--split",choices=("dev","test"),required=True)
    p.add_argument("--ratings",help="JSONL adjudicated for exactly this prediction file; id, correct, hallucinated, prediction_sha256")
    a=p.parse_args()
    refs = unique([r for r in load(a.manifest) if r["split"]==a.split])
    preds = unique(load(a.predictions))
    if set(refs)!=set(preds):
        raise ValueError(f"Require complete split: {len(refs.keys()-preds.keys())} missing and {len(preds.keys()-refs.keys())} unexpected")
    if len({v["mode"] for v in preds.values()})!=1:
        raise ValueError("One mode per prediction file")
    if any("audio_sha256" in ref and ref["audio_sha256"]!=preds[i]["audio_sha256"] for i,ref in refs.items()):
        raise ValueError("Prediction audio differs from the frozen reference")
    ratings=unique(load(a.ratings)) if a.ratings else None
    if ratings is not None:
        import hashlib
        sha = hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest()
        if set(refs)!=set(ratings) or any(v["prediction_sha256"]!=sha for v in ratings.values()):
            raise ValueError("Ratings must match the exact prediction file and all items")
    pairs=[(r,preds[i]) for i,r in refs.items()]
    subgroups={str(v):summarize([(r,p) for r,p in pairs if r.get("answerable")==v],ratings)
               for v in {r["answerable"] for r,_ in pairs if "answerable" in r}}
    print(json.dumps({"all":summarize(pairs,ratings),"answerable":subgroups,"conditions":{
        tag:summarize([(r,p) for r,p in pairs if r["condition"]==tag],ratings)
        for tag in sorted({r["condition"] for r in refs.values()})}},indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()

