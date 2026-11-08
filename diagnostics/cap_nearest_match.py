"""Nearest-neighbour duration matching for the ASR token-cap comparison.

A coarse duration window is uninformative here because the capped views span
5.8-24.5 s, which contains almost every uncapped view. This script matches each
capped view to the uncapped view of the SAME acoustic condition with the closest
clip duration (1:1, without replacement) and reports the paired difference.

Also reports corpus WER (sum of edits / sum of reference words, the definition
used in the report) separately from the unweighted mean of per-view WER, because
the two differ materially for these short clips.

Read-only, offline. Writes research-team/cap-nearest-match.json.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "acoustic_interface/artifacts/transcript-final"
MAX_ASR = 224


def load(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


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


def main() -> None:
    preds = {p["id"]: p for p in load(RESULT / "predictions.jsonl")}
    refs = {r["id"]: r for r in load(RESULT / "references.jsonl")}
    labels = {r["id"]: r for r in load(RESULT / "review/effective-labels.jsonl")}

    rows = []
    for vid in sorted(preds):
        p, r = preds[vid], refs[vid]
        ref_w = norm(r["transcript"]).split()
        hyp_w = norm(p["hypothesis"]).split()
        err = edits(ref_w, hyp_w)
        rows.append(
            {
                "id": vid,
                "condition": p["condition"],
                "seconds": r["seconds_audio"],
                "ref_words": len(ref_w),
                "err": err,
                "wer_view": err / len(ref_w) if ref_w else 0.0,
                "hyp_words": len(hyp_w),
                "capped": len(p["asr_token_ids"]) >= MAX_ASR,
                "correct": bool(labels[vid]["correct"]),
            }
        )

    pairs = []
    for cond in ("clean", "noise_10db"):
        cap = [r for r in rows if r["capped"] and r["condition"] == cond]
        pool = sorted(
            [r for r in rows if not r["capped"] and r["condition"] == cond],
            key=lambda r: r["seconds"],
        )
        used: set[str] = set()
        for c in sorted(cap, key=lambda r: r["seconds"]):
            best = min(
                (u for u in pool if u["id"] not in used),
                key=lambda u: abs(u["seconds"] - c["seconds"]),
            )
            used.add(best["id"])
            pairs.append({"capped": c, "control": best})

    def corpus(rs):
        refw = sum(r["ref_words"] for r in rs)
        return sum(r["err"] for r in rs) / refw if refw else None

    cap_rows = [p["capped"] for p in pairs]
    ctl_rows = [p["control"] for p in pairs]
    deltas = sorted(p["capped"]["wer_view"] - p["control"]["wer_view"] for p in pairs)

    summary = {
        "pairs": len(pairs),
        "median_abs_duration_gap_s": sorted(
            abs(p["capped"]["seconds"] - p["control"]["seconds"]) for p in pairs
        )[len(pairs) // 2],
        "capped_corpus_wer": corpus(cap_rows),
        "matched_control_corpus_wer": corpus(ctl_rows),
        "capped_mean_view_wer": sum(r["wer_view"] for r in cap_rows) / len(cap_rows),
        "matched_control_mean_view_wer": sum(r["wer_view"] for r in ctl_rows) / len(ctl_rows),
        "paired_mean_view_wer_delta": sum(deltas) / len(deltas),
        "paired_median_view_wer_delta": deltas[len(deltas) // 2],
        "pairs_with_worse_wer_when_capped": sum(1 for d in deltas if d > 0),
        "capped_correct_rate": sum(r["correct"] for r in cap_rows) / len(cap_rows),
        "matched_control_correct_rate": sum(r["correct"] for r in ctl_rows) / len(ctl_rows),
        "capped_hyp_longer_than_control": sum(
            1 for p in pairs if p["capped"]["hyp_words"] > p["control"]["hyp_words"]
        ),
    }

    out = ROOT / "research-team/cap-nearest-match.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"1:1 nearest-duration pairs: {summary['pairs']}")
    print(f"  median |duration gap|: {summary['median_abs_duration_gap_s']:.2f}s")
    print(
        f"  corpus WER   capped {100 * summary['capped_corpus_wer']:.2f}%   "
        f"matched control {100 * summary['matched_control_corpus_wer']:.2f}%"
    )
    print(
        f"  mean view WER capped {100 * summary['capped_mean_view_wer']:.2f}%   "
        f"control {100 * summary['matched_control_mean_view_wer']:.2f}%"
    )
    print(
        f"  paired delta mean {100 * summary['paired_mean_view_wer_delta']:.2f} pp, "
        f"median {100 * summary['paired_median_view_wer_delta']:.2f} pp"
    )
    print(
        f"  pairs where capped is worse: "
        f"{summary['pairs_with_worse_wer_when_capped']}/{summary['pairs']}"
    )
    print(
        f"  correct rate capped {100 * summary['capped_correct_rate']:.1f}% vs "
        f"control {100 * summary['matched_control_correct_rate']:.1f}%"
    )
    print(
        f"  capped hypothesis longer than matched control: "
        f"{summary['capped_hyp_longer_than_control']}/{summary['pairs']}"
    )
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
