"""Replay Tables 2 and 3: the M1 detector and the head-confidence comparison.

Recomputes from records/m1-scored-slots.jsonl (the 348 scored slots, each with
source id, corrected gold label, emitted value, encoder margin and decoder
per-token confidence):

    AUROC of the encoder margin against numeral correctness
    source-clustered bootstrap CI for that AUROC
    AUROC of the decoder's own confidence, and the paired difference
    selective prediction at 50-100 % coverage, both signals
    the frozen-threshold confusion counts quoted in the text
    the matched-budget comparison (flag as many slots as there are errors)

CPU only. No audio, no model, no network, no Modal volume.

    python numeral-evidence/replay/table2_m1_detector.py
"""

from __future__ import annotations

import sys

from common import (Checker, auroc, clustered_bootstrap, expected, load_json,
                    load_jsonl, paired_bootstrap_diff)


def selective_precision(labels, scores, coverage):
    """Precision (share correct) among the `coverage` fraction of most-confident slots.

    A slot is *kept* when its score says "not an error". For the encoder the
    score is the margin (higher = more reliable); for head confidence it is the
    mean token probability (higher = more reliable). Both are oriented the same
    way here, which is why the call sites do not negate either one.
    """
    import numpy as np

    y = np.asarray(labels, bool)
    s = np.asarray(scores, float)
    order = np.argsort(-s)          # most reliable first
    k = max(1, int(round(coverage * len(order))))
    kept = order[:k]
    return float(np.mean(y[kept] == 0)), int(k)


def main() -> int:
    exp = expected()["m1"]
    slots = load_jsonl("m1-scored-slots.jsonl")
    c = Checker("Table 2 - M1 detector vs decoder confidence")

    c.check("n scored slots", len(slots), exp["n_scored"])
    n_sources = len({r["source_id"] for r in slots})
    c.check("n sources", n_sources, exp["n_sources"])

    labels = [0 if r["correct"] else 1 for r in slots]
    margins = [float(r["margin"]) for r in slots]
    head = [float(r["head_conf"]) for r in slots]
    sources = [r["source_id"] for r in slots]

    n_wrong = sum(labels)
    c.check("n wrong", n_wrong, exp["n_wrong"])

    # Encoder margin is oriented so that LOWER margin means more likely wrong.
    enc_auroc = auroc(labels, [-m for m in margins])
    c.check("AUROC encoder margin", round(enc_auroc, 6), round(exp["auroc_margin"], 6), tol=1e-6)

    ci = clustered_bootstrap(labels, [-m for m in margins], sources, n_boot=2000, seed=0)
    c.check_ci("CI encoder margin", [round(v, 4) for v in ci],
               [round(v, 4) for v in exp["auroc_margin_ci"]], tol=2e-4)

    head_auroc = auroc(labels, [-h for h in head])
    c.check("AUROC decoder confidence", round(head_auroc, 6), round(exp["head_auroc"], 6), tol=1e-6)

    diff = paired_bootstrap_diff(labels, [-m for m in margins], [-h for h in head],
                                 sources, n_boot=2000, seed=2)
    c.check("paired difference mean", round(diff["mean"], 4),
            round(exp["paired_encoder_minus_head"]["mean"], 4), tol=2e-4)
    c.check_ci("paired difference CI", [round(v, 4) for v in diff["ci"]],
               [round(v, 4) for v in exp["paired_encoder_minus_head"]["ci"]], tol=2e-3)

    # Selective prediction: precision among the most reliable slots.
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        enc_p, k = selective_precision(labels, margins, cov)
        head_p, _ = selective_precision(labels, head, cov)
        want = next((p for p in exp["selective_encoder"]
                     if abs(p["coverage"] - round(k / len(labels), 3)) < 1e-9), None)
        if want is not None:
            c.check(f"encoder precision @ {cov:.0%}", round(enc_p, 6), round(want["precision"], 6),
                    tol=1e-6)

    # Threshold behaviour quoted in the report text.
    thr = float(load_json("m1-report-values.json")["m1v5_score"]["threshold"])
    flagged = [i for i, m in enumerate(margins) if m < thr]
    tp = sum(1 for i in flagged if labels[i] == 1)
    kept = [i for i, m in enumerate(margins) if m >= thr]
    kept_wrong = sum(1 for i in kept if labels[i] == 1)
    at = exp["at_threshold"]
    c.check("flagged at frozen threshold", len(flagged), at["flagged"])
    c.check("true positives at threshold", tp, at["true_positives"])
    c.check("recall of errors", round(tp / n_wrong, 6), round(at["recall_of_errors"], 6), tol=1e-6)
    c.check("flagged precision", round(tp / len(flagged), 6),
            round(at["precision_when_flagged"], 6), tol=1e-6)
    c.check("kept precision", round(1 - kept_wrong / len(kept), 6),
            round(at["kept_precision"], 6), tol=1e-6)

    # Matched budget: flag exactly as many slots as there are errors.
    order = sorted(range(len(margins)), key=lambda i: margins[i])
    top = order[:n_wrong]
    hits = sum(labels[i] for i in top)
    c.check("matched-budget hits", hits, exp["matched_coverage"]["hits"])

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
