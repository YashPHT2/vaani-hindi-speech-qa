"""Score the frozen transcript cascade and prepare blinded model-review packets."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from score import edits, load, norm, summarize, unique


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def case_key(reference, prediction):
    fields = [reference["source_id"], reference["question"], prediction["answer"]]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()[:20]


def save_packet(path, rows):
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"Existing review packet differs: {path}")
    path.write_text(content, encoding="utf-8")


def rated(path, expected):
    rows = load(path)
    labels = {row["case"]: row for row in rows}
    if len(labels) != len(rows) or labels.keys() != expected:
        raise ValueError(f"Rating coverage differs from packet: {path}")
    for label in labels.values():
        if label.get("reviewer_type") != "model":
            raise ValueError("Model annotation provenance is required")
        if not str(label.get("reason", "")).strip():
            raise ValueError("Each judgment requires a rationale")
        for field in ("correct", "hallucinated", "ambiguous"):
            if type(label[field]) is not bool:
                raise ValueError(f"Expected Boolean {field}")
    return labels


def reconcile(primary, secondary, strict=False):
    labels = {}
    conflicts = []
    for key, first in primary.items():
        second = secondary.get(key)
        conflict = second is not None and any(first[k] != second[k] for k in ("correct", "hallucinated"))
        if conflict:
            conflicts.append(key)
        hallucinated = first["hallucinated"] or (second is not None and second["hallucinated"])
        ambiguous = first["ambiguous"] or conflict or (second is not None and second["ambiguous"])
        correct = first["correct"] and (second is None or second["correct"])
        labels[key] = dict(correct=bool(correct and not hallucinated and not conflict and not (strict and ambiguous)),
                           hallucinated=bool(hallucinated), ambiguous=bool(ambiguous))
    return labels, conflicts


def completed_timing(pred):
    return not pred.get("failed") and all(
        isinstance(pred.get(key), (int, float)) and math.isfinite(pred[key]) and pred[key] >= 0
        for key in ("total_s", "asr_s", "llm_s"))


def group_summary(pairs, labels):
    # The shared scorer supplies error counts and rates. Timing below uses only
    # completed, measured rows and their matching audio durations.
    safe_pairs = [(ref, dict(pred, **{key: pred[key] if completed_timing(pred) else 0.
                                    for key in ("total_s", "asr_s", "llm_s")})) for ref, pred in pairs]
    output = summarize(safe_pairs, labels)
    timed = [pred for _, pred in pairs if completed_timing(pred)]
    latency = sorted(pred["total_s"] for pred in timed)
    duration = sum(pred["seconds_audio"] for pred in timed)
    failed_times = [pred["total_s"] for _, pred in pairs if pred.get("failed")
                    and isinstance(pred.get("total_s"), (int, float)) and math.isfinite(pred["total_s"])]
    output.update(p50_completed_text_s=statistics.median(latency) if timed else None,
                  p95_completed_text_s=latency[math.ceil(.95 * len(timed)) - 1] if timed else None,
                  median_asr_s=statistics.median(pred["asr_s"] for pred in timed) if timed else None,
                  median_llm_s=statistics.median(pred["llm_s"] for pred in timed) if timed else None,
                  pipeline_rtf=sum(latency) / duration if duration else None,
                  asr_rtf=sum(pred["asr_s"] for pred in timed) / duration if duration else None,
                  timing_rows=len(timed), timing_audio_seconds=duration,
                  failed=sum(bool(pred.get("failed")) for _, pred in pairs),
                  asr_failed=sum(bool(pred.get("asr_failed")) for _, pred in pairs),
                  asr_capped=sum(bool(pred.get("asr_capped")) for _, pred in pairs),
                  time_to_failure_s=failed_times)
    return output


def bootstrap_metrics(values, semantic):
    """Values have shape (replicate, views, field)."""
    totals = np.nansum(values, axis=1)
    latency = values[:, :, 6]
    counts = np.isfinite(latency).sum(axis=1)
    median = np.full(len(values), np.nan)
    median[counts > 0] = np.nanmedian(latency[counts > 0], axis=1)
    rank = np.maximum(np.ceil(.95 * counts).astype(int) - 1, 0)
    p95 = np.sort(latency, axis=1)[np.arange(len(values)), rank]
    timed_duration = totals[:, 11]
    rtf = np.divide(totals[:, 6], timed_duration, out=np.full(len(values), np.nan), where=timed_duration > 0)
    asr_rtf = np.divide(totals[:, 8], timed_duration, out=np.full(len(values), np.nan), where=timed_duration > 0)
    output = {
        "wer": totals[:, 0] / totals[:, 1],
        "cer": totals[:, 2] / totals[:, 3],
        "answer_em": values[:, :, 4].mean(axis=1),
        "abstention_rate": values[:, :, 5].mean(axis=1),
        "p50_completed_text_s": median,
        "p95_completed_text_s": p95,
        "pipeline_rtf": rtf,
        "asr_rtf": asr_rtf,
    }
    if semantic:
        output.update(correct_rate=values[:, :, 9].mean(axis=1),
                      hallucinated_rate=values[:, :, 10].mean(axis=1))
    return output


def interval(estimate, samples):
    finite = samples[np.isfinite(samples)]
    return dict(estimate=float(estimate) if math.isfinite(estimate) else None,
                lower=float(np.quantile(finite, .025)) if len(finite) else None,
                upper=float(np.quantile(finite, .975)) if len(finite) else None,
                defined_replicates=int(len(finite)), undefined_replicates=int(len(samples) - len(finite)))


def uncertainty(pairs, labels):
    source_rows = {}
    for ref, pred in pairs:
        truth, hypothesis = norm(ref["transcript"]), norm(pred["hypothesis"])
        label = labels[ref["id"]] if labels else None
        timed = completed_timing(pred)
        fields = [edits(truth.split(), hypothesis.split()), len(truth.split()),
                  edits(truth.replace(" ", ""), hypothesis.replace(" ", "")), len(truth.replace(" ", "")),
                  norm(pred["answer"]) in {norm(x) for x in ref["answers"]},
                  norm(pred["answer"]) in {"अज्ञात", "unknown"}, pred["total_s"] if timed else np.nan,
                  pred["seconds_audio"], pred["asr_s"] if timed else np.nan,
                  label["correct"] if label else 0, label["hallucinated"] if label else 0,
                  pred["seconds_audio"] if timed else 0.]
        source_rows.setdefault(str(ref["source_id"]), {})[ref["condition"]] = fields
    sources = sorted(source_rows, key=int)
    values = np.asarray([[source_rows[source][c] for c in ("clean", "noise_10db")] for source in sources], dtype=float)
    assert values.shape == (100, 2, 12)
    draws = np.random.default_rng(17).integers(0, 100, size=(2000, 100))
    sampled = values[draws]
    groups = {"clean": sampled[:, :, 0, :], "noise_10db": sampled[:, :, 1, :],
              "all": sampled.reshape(2000, 200, 12)}
    observed_groups = {"clean": values[None, :, 0, :], "noise_10db": values[None, :, 1, :],
                       "all": values.reshape(1, 200, 12)}
    samples = {name: bootstrap_metrics(group, labels is not None) for name, group in groups.items()}
    observed = {name: bootstrap_metrics(group, labels is not None) for name, group in observed_groups.items()}
    output = {}
    for name, metrics in samples.items():
        output[name] = {metric: interval(observed[name][metric][0], values)
                        for metric, values in metrics.items()}
    output["noise_minus_clean"] = {
        metric: interval(observed["noise_10db"][metric][0] - observed["clean"][metric][0],
                         samples["noise_10db"][metric] - samples["clean"][metric])
        for metric in samples["clean"]}
    return dict(unit="source with both correlated acoustic views", sources=100, replicates=2000,
                seed=17, generator="numpy.default_rng", percentile=.95, intervals=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE / "artifacts/transcript-final")
    parser.add_argument("--primary-labels", type=Path)
    parser.add_argument("--secondary-labels", type=Path)
    args = parser.parse_args()
    folder = args.directory
    manifest = json.loads((folder / "artifact-manifest.json").read_bytes())
    if not {"predictions.jsonl", "references.jsonl", "result.json"}.issubset(manifest["sha256"]):
        raise ValueError("Manifest must bind predictions, references and the raw result")
    for name, expected in manifest["sha256"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    refs, predictions = unique(load(folder / "references.jsonl")), unique(load(folder / "predictions.jsonl"))
    if len(refs) != 200 or refs.keys() != predictions.keys():
        raise ValueError("Expected exactly 200 complete matched reference and prediction rows")
    freeze = json.loads((HERE / "data/final-experiment-freeze-v1.json").read_bytes())
    secondary_sources = {str(x) for x in freeze["rating"]["secondary_sources"]}
    pairs = [(ref, predictions[key].copy()) for key, ref in sorted(refs.items())]
    for _, pred in pairs:
        if pred.get("asr_failed"):
            pred["hypothesis"] = ""
        if pred.get("failed"):
            pred["answer"] = ""
    cases, mapping = {}, []
    secondary_keys = set()
    for ref, pred in pairs:
        assert ref["split"] == "test" and ref["condition"] in ("clean", "noise_10db")
        assert ref["audio_sha256"] == pred["audio_sha256"]
        assert ref["source_id"] == pred["source_id"] and ref["condition"] == pred["condition"]
        assert ref["question"] == pred["question"] and norm(ref["transcript"])
        assert pred["mode"] == "baseline"
        assert pred["seconds_audio"] > 0
        raw_prediction = predictions[ref["id"]]
        key = case_key(ref, raw_prediction)
        cases[key] = dict(case=key, evidence=ref["transcript"], question=ref["question"],
                          reference=ref["answers"], answer=raw_prediction["answer"])
        mapping.append(dict(id=ref["id"], source_id=ref["source_id"], condition=ref["condition"], case=key))
        if str(ref["source_id"]) in secondary_sources:
            secondary_keys.add(key)
    assert len({str(ref["source_id"]) for ref, _ in pairs}) == 100
    for source in {ref["source_id"] for ref, _ in pairs}:
        assert sorted(ref["condition"] for ref, _ in pairs if ref["source_id"] == source) == ["clean", "noise_10db"]
    for condition in ("clean", "noise_10db"):
        assert sum(ref["answerable"] is True for ref, _ in pairs if ref["condition"] == condition) == 80
        assert sum(ref["answerable"] is False for ref, _ in pairs if ref["condition"] == condition) == 20
    assert len({str(row["source_id"]) for row in mapping if row["case"] in secondary_keys}) == 20
    packet = folder / "review"
    packet.mkdir(exist_ok=True)
    save_packet(packet / "cases.jsonl", [cases[key] for key in sorted(cases)])
    save_packet(packet / "secondary-cases.jsonl", [cases[key] for key in sorted(secondary_keys)])
    save_packet(packet / "private-mapping.jsonl", mapping)
    primary = rated(args.primary_labels, cases.keys()) if args.primary_labels else None
    secondary = rated(args.secondary_labels, secondary_keys) if args.secondary_labels else {}
    if secondary and primary is None:
        raise ValueError("Secondary labels require complete primary labels")
    case_labels, conflicts = reconcile(primary, secondary) if primary else ({}, [])
    labels = {row["id"]: case_labels[row["case"]].copy() for row in mapping} if primary else None
    if labels:
        for ref, pred in pairs:
            if pred.get("failed") or not norm(pred["answer"]):
                labels[ref["id"]].update(correct=False, hallucinated=False)
    groups = {"all": pairs, "clean": [(r, p) for r, p in pairs if r["condition"] == "clean"],
              "noise_10db": [(r, p) for r, p in pairs if r["condition"] == "noise_10db"]}
    output = dict(sources=100, views=200, input_sha256=manifest["sha256"], unique_review_cases=len(cases),
                  secondary_review_cases=len(secondary_keys), secondary_review_sources=20,
                  semantic_status="model_primary_plus_independent_subset" if secondary else "model_primary_only" if primary else "unrated",
                  conflicts=conflicts, annotations=dict(primary_sha256=digest(args.primary_labels) if primary else None,
                  secondary_sha256=digest(args.secondary_labels) if secondary else None), groups={})
    for name, group in groups.items():
        output["groups"][name] = group_summary(group, labels)
    if primary:
        first, _ = reconcile(primary, {})
        strict, _ = reconcile(primary, secondary, strict=True)
        first_by_id = {row["id"]: first[row["case"]].copy() for row in mapping}
        strict_by_id = {row["id"]: strict[row["case"]].copy() for row in mapping}
        for ref, pred in pairs:
            if pred.get("failed") or not norm(pred["answer"]):
                first_by_id[ref["id"]].update(correct=False, hallucinated=False)
                strict_by_id[ref["id"]].update(correct=False, hallucinated=False)
        output["primary_only"] = {name: group_summary(group, first_by_id) for name, group in groups.items()}
        output["strict"] = {name: group_summary(group, strict_by_id) for name, group in groups.items()}
        output["answerability"] = {name: {str(value): group_summary([(r, p) for r, p in group if r["answerable"] == value], labels)
                                          for value in (True, False)} for name, group in groups.items()}
        effective = [dict(id=key, **label) for key, label in sorted(labels.items())]
        (packet / "effective-labels.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in effective), encoding="utf-8")
    output["bootstrap"] = uncertainty(pairs, labels)
    save_json(folder / "analysis.json", output)
    print(json.dumps({key: value for key, value in output.items() if key not in ("bootstrap", "input_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
