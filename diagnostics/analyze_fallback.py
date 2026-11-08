"""Analyze the completed development decoding-policy experiment.

Compares arm A (frozen greedy baseline) with arm B (compression-guarded
temperature fallback) on the 24 historical development views, using the shipped
score.py normalization and corpus WER definition.

Read-only over retrieved results. Writes research-team/fallback-analysis.json.
"""
from __future__ import annotations

import json
import unicodedata
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "research-team/modal-retrieval/rows.json"
THRESHOLD = 2.4


def norm(text: str) -> str:
    chars = []
    for c in unicodedata.normalize("NFC", text.casefold()):
        cat = unicodedata.category(c)
        if cat[0] == "P":
            c = " "
        elif cat == "Nd":
            c = str(unicodedata.decimal(c))
        chars.append(c)
    return " ".join("".join(chars).split())


def edits(a: list[str], b: list[str]) -> int:
    last = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(last[j] + 1, row[-1] + 1, last[j - 1] + (x != y)))
        last = row
    return last[-1]


def cr(text: str) -> float:
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))

    totals = {"a": {"err": 0, "ref": 0}, "b": {"err": 0, "ref": 0}}
    per_view = []
    for r in rows:
        ref = norm(r["transcript"]).split()
        rec = {"id": r["id"], "condition": r["condition"], "ref_words": len(ref)}
        for arm in ("a", "b"):
            hyp = norm(r[f"{arm}_hypothesis"]).split()
            err = edits(ref, hyp)
            totals[arm]["err"] += err
            totals[arm]["ref"] += len(ref)
            rec[f"{arm}_wer"] = err / len(ref) if ref else 0.0
            rec[f"{arm}_ratio"] = cr(r[f"{arm}_hypothesis"])
            rec[f"{arm}_capped"] = bool(r[f"{arm}_capped"])
            rec[f"{arm}_seconds"] = r[f"{arm}_seconds"]
            rec[f"{arm}_tokens"] = r[f"{arm}_generated"]
        rec["identical_output"] = r["a_hypothesis"] == r["b_hypothesis"]
        rec["fallback_tripped"] = r["b_fallback_tripped"]
        rec["changed"] = not rec["identical_output"]
        rec["delta_wer"] = rec["b_wer"] - rec["a_wer"]
        per_view.append(rec)

    summary = {
        "views": len(rows),
        "sources": len({r["source_id"] for r in rows}),
        "arm_a_corpus_wer": totals["a"]["err"] / totals["a"]["ref"],
        "arm_b_corpus_wer": totals["b"]["err"] / totals["b"]["ref"],
        "corpus_wer_delta": (totals["b"]["err"] - totals["a"]["err"]) / totals["a"]["ref"],
        "arm_a_word_edits": totals["a"]["err"],
        "arm_b_word_edits": totals["b"]["err"],
        "reference_words": totals["a"]["ref"],
        "arm_a_capped_views": sum(1 for r in per_view if r["a_capped"]),
        "arm_b_capped_views": sum(1 for r in per_view if r["b_capped"]),
        "arm_a_views_over_threshold": sum(1 for r in per_view if r["a_ratio"] > THRESHOLD),
        "arm_b_views_over_threshold": sum(1 for r in per_view if r["b_ratio"] > THRESHOLD),
        "views_changed": sum(1 for r in per_view if r["changed"]),
        "views_identical": sum(1 for r in per_view if r["identical_output"]),
        "views_improved": sum(1 for r in per_view if r["changed"] and r["delta_wer"] < 0),
        "views_regressed": sum(1 for r in per_view if r["changed"] and r["delta_wer"] > 0),
        "views_changed_same_wer": sum(1 for r in per_view if r["changed"] and r["delta_wer"] == 0),
        "total_fallback_trips": sum(r["fallback_tripped"] for r in per_view),
        "arm_a_total_seconds": sum(r["a_seconds"] for r in per_view),
        "arm_b_total_seconds": sum(r["b_seconds"] for r in per_view),
    }
    summary["asr_time_ratio_b_over_a"] = (
        summary["arm_b_total_seconds"] / summary["arm_a_total_seconds"]
    )
    summary["changed_views"] = [
        {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in r.items()
            if k in {"id", "a_ratio", "b_ratio", "a_wer", "b_wer", "delta_wer",
                     "a_tokens", "b_tokens", "fallback_tripped", "a_capped", "b_capped"}
        }
        for r in per_view
        if r["changed"]
    ]

    out = ROOT / "research-team/fallback-analysis.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"views {summary['views']} ({summary['sources']} sources)")
    print(f"corpus WER  A={100 * summary['arm_a_corpus_wer']:.2f}%  "
          f"B={100 * summary['arm_b_corpus_wer']:.2f}%  "
          f"delta={100 * summary['corpus_wer_delta']:+.2f} pp")
    print(f"capped views A={summary['arm_a_capped_views']} -> B={summary['arm_b_capped_views']}")
    print(f"views over ratio 2.4 A={summary['arm_a_views_over_threshold']} -> "
          f"B={summary['arm_b_views_over_threshold']}")
    print(f"changed {summary['views_changed']}, identical {summary['views_identical']}, "
          f"improved {summary['views_improved']}, regressed {summary['views_regressed']}")
    print(f"fallback trips {summary['total_fallback_trips']}, "
          f"ASR time ratio {summary['asr_time_ratio_b_over_a']:.2f}x")
    print("\nchanged views:")
    for c in summary["changed_views"]:
        print(f"  {c['id']:<28} ratio {c['a_ratio']:.2f}->{c['b_ratio']:.2f}  "
              f"tok {c['a_tokens']}->{c['b_tokens']}  "
              f"wer {c['a_wer']:.2f}->{c['b_wer']:.2f}  trips={c['fallback_tripped']}")
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
