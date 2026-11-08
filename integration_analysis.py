"""Paired, source-resampled semantic effects and oracle transitions from adjudicated outputs."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from score import load, unique


def rated_arm(mode, prediction_path, ratings_path, references):
    predictions = unique(load(prediction_path))
    ratings = unique(load(ratings_path))
    if predictions.keys() != references.keys() or ratings.keys() != references.keys():
        raise ValueError(f"{mode}: predictions and ratings must cover the entire split")
    digest = hashlib.sha256(Path(prediction_path).read_bytes()).hexdigest()
    for key, prediction in predictions.items():
        rating = ratings[key]
        if prediction["mode"] != mode or prediction["audio_sha256"] != references[key]["audio_sha256"]:
            raise ValueError(f"{mode}: prediction or audio mismatch at {key}")
        if rating["prediction_sha256"] != digest or rating.get("mode", mode) != mode:
            raise ValueError(f"{mode}: ratings do not correspond to this prediction file")
        if any(type(rating[field]) is not bool for field in ("correct", "hallucinated")):
            raise ValueError(f"{mode}: non-boolean judgement at {key}")
    return ratings


def summarize(ids, references, arms, baseline, intervention, oracle):
    pooled = {}
    rescues = []
    reversals = []
    for key in ids:
        a, b, c = (arms[mode][key] for mode in (baseline, intervention, oracle))
        source = references[key]["source_id"]
        delta = (b["correct"] - a["correct"], b["hallucinated"] - a["hallucinated"])
        pooled.setdefault(source, []).append(delta)
        if a["correct"] != c["correct"]:
            (rescues if c["correct"] else reversals).append(key)

    unit_sizes = {len(rows) for rows in pooled.values()}
    if len(unit_sizes) != 1:
        raise ValueError("Each source in a reported slice must contribute the same number of views")
    units = [tuple(map(statistics.mean, zip(*pooled[k]))) for k in sorted(pooled)]
    rng = random.Random(17)
    draws = []
    for _ in range(2000):
        sample = rng.choices(units, k=len(units))
        draws.append(tuple(sum(column) / len(sample) for column in zip(*sample)))
    result = {"views": len(ids), "sources": len(units)}
    for i, field in enumerate(("correct", "hallucinated")):
        values = sorted(trial[i] for trial in draws)
        result[field] = {
            "counts": {label: sum(ratings[key][field] for key in ids) for label, ratings in arms.items()},
            "delta_intervention_minus_baseline": statistics.mean(row[i] for row in units),
            "ci95": [values[math.floor(.025 * (len(values) - 1))],
                     values[math.ceil(.975 * (len(values) - 1))]],
        }
    result["baseline_to_oracle"] = {
        "rescued_views": len(rescues), "reversed_views": len(reversals),
        "rescued_ids": sorted(rescues), "reversed_ids": sorted(reversals),
    }
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("manifest")
    cli.add_argument("--split", required=True, choices=("dev", "test"))
    for arm in ("baseline", "intervention", "oracle"):
        cli.add_argument("--" + arm, required=True, nargs=3, metavar=("MODE", "PREDICTIONS", "RATINGS"))
    args = cli.parse_args()
    refs = unique([r for r in load(args.manifest) if r["split"] == args.split])
    if not refs:
        raise ValueError("Empty split")
    settings = (args.baseline, args.intervention, args.oracle)
    modes = [setting[0] for setting in settings]
    if len(set(modes)) != 3 or modes[-1] != "oracle":
        raise ValueError("Expected three distinct modes with a transcript oracle")
    arms = {mode: rated_arm(mode, pred, labels, refs) for mode, pred, labels in settings}
    conditions = sorted({r["condition"] for r in refs.values()})
    source_conditions = {}
    for r in refs.values():
        source_conditions.setdefault(r["source_id"], []).append(r["condition"])
    if any(sorted(tags) != conditions for tags in source_conditions.values()):
        raise ValueError("Every source must have one observation per acoustic condition")
    groups = {"all": list(refs)}
    for condition in conditions:
        view = [key for key, row in refs.items() if row["condition"] == condition]
        groups[condition] = view
        for answerable, name in ((True, "answerable"), (False, "unanswerable")):
            keys = [key for key in view if refs[key]["answerable"] is answerable]
            if keys:
                groups[f"{condition}/{name}"] = keys
    output = {name: summarize(keys, refs, arms, *modes) for name, keys in groups.items()}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
