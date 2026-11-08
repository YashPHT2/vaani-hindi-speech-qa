"""Paired uncertainty for the development decoding-policy comparison.

24 views from 12 sources are correlated (clean/noisy pairs share a recording),
so the unit of resampling is the source, not the view. This reports a
source-cluster bootstrap for the corpus-WER difference and the exact paired sign
test over changed views, so the size of the evidence is explicit.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "research-team/fallback-analysis.json"
ROWS = ROOT / "research-team/modal-retrieval/rows.json"


def main() -> None:
    a = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    rows = json.loads(ROWS.read_text(encoding="utf-8"))

    # Rebuild per-view edits keyed by source for cluster resampling.
    import unicodedata

    def norm(t):
        ch = []
        for c in unicodedata.normalize("NFC", t.casefold()):
            cat = unicodedata.category(c)
            if cat[0] == "P":
                c = " "
            elif cat == "Nd":
                c = str(unicodedata.decimal(c))
            ch.append(c)
        return " ".join("".join(ch).split())

    def edits(x, y):
        last = list(range(len(y) + 1))
        for i, cx in enumerate(x, 1):
            cur = [i]
            for j, cy in enumerate(y, 1):
                cur.append(min(last[j] + 1, cur[-1] + 1, last[j - 1] + (cx != cy)))
            last = cur
        return last[-1]

    by_source: dict[str, list[dict]] = {}
    for r in rows:
        ref = norm(r["transcript"]).split()
        rec = {
            "ref": len(ref),
            "a": edits(ref, norm(r["a_hypothesis"]).split()),
            "b": edits(ref, norm(r["b_hypothesis"]).split()),
        }
        by_source.setdefault(r["source_id"], []).append(rec)

    ids = sorted(by_source)
    rng = random.Random(17)
    deltas = []
    for _ in range(20000):
        sample = [rng.choice(ids) for _ in ids]
        ae = sum(v["a"] for s in sample for v in by_source[s])
        be = sum(v["b"] for s in sample for v in by_source[s])
        ref = sum(v["ref"] for s in sample for v in by_source[s])
        deltas.append((be - ae) / ref)
    deltas.sort()

    # Exact sign test on changed views: 3 improved, 1 regressed.
    improved = a["views_improved"]
    regressed = a["views_regressed"]
    n = improved + regressed
    from math import comb

    # two-sided exact binomial under p=0.5, count the smaller tail
    if n:
        tail = sum(comb(n, k) for k in range(0, min(improved, regressed) + 1)) / (2 ** n)
        p_two_sided = min(1.0, 2 * tail)
    else:
        p_two_sided = None

    summary = {
        "unit": "source (12 clusters, 24 correlated views)",
        "corpus_wer_delta": a["corpus_wer_delta"],
        "cluster_bootstrap_95": {
            "lower": deltas[int(0.025 * len(deltas))],
            "upper": deltas[int(0.975 * len(deltas)) - 1],
            "replicates": len(deltas),
            "seed": 17,
        },
        "changed_views": n,
        "improved": improved,
        "regressed": regressed,
        "exact_sign_test_p": p_two_sided,
        "word_edits_removed": a["arm_a_word_edits"] - a["arm_b_word_edits"],
        "reference_words": a["reference_words"],
        "note": (
            "12 source clusters cannot support a precise interval. The bootstrap "
            "is reported to bound optimism, not to claim a population effect."
        ),
    }
    out = ROOT / "research-team/fallback-uncertainty.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    b = summary["cluster_bootstrap_95"]
    print(f"corpus WER delta {100 * a['corpus_wer_delta']:+.2f} pp")
    print(f"source-cluster bootstrap 95%: [{100 * b['lower']:+.2f}, {100 * b['upper']:+.2f}] pp")
    print(f"changed views {n}: improved {improved}, regressed {regressed}")
    print(f"exact sign test p = {p_two_sided:.4f}" if p_two_sided is not None else "n/a")
    print(f"word edits removed {summary['word_edits_removed']} of {summary['reference_words']} reference words")
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
