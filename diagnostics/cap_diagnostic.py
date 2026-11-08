"""Offline diagnostic of the 30 ASR token-cap views in the frozen final evaluation.

Read-only. Uses no audio, models, network or GPU. Nothing here changes a
prediction, a label or a saved metric; all numbers are derived from the frozen
predictions/references and the packaged review labels.

Question addressed: the frozen worker recorded asr_capped=False for every view,
but returned Whisper token arrays of length 224 (the configured ceiling) for 30
views. Were those outputs actually truncated, and does cap incidence track
recognition error and answer outcome?

Run:  python -X utf8 research-team/cap_diagnostic.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "acoustic_interface/artifacts/transcript-final"
MAX_ASR = 224


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def norm(text: str) -> str:
    """Same normalization as the shipped score.py."""
    import unicodedata

    chars = []
    for c in unicodedata.normalize("NFC", text.casefold()):
        cat = unicodedata.category(c)
        if cat[0] == "P":
            c = " "
        elif cat == "Nd":
            c = str(unicodedata.decimal(c))
        chars.append(c)
    return " ".join("".join(chars).split())


def edits(a: list, b: list) -> int:
    last = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(last[j] + 1, row[-1] + 1, last[j - 1] + (x != y)))
        last = row
    return last[-1]


def main() -> None:
    preds = {p["id"]: p for p in load_jsonl(RESULT / "predictions.jsonl")}
    refs = {r["id"]: r for r in load_jsonl(RESULT / "references.jsonl")}
    assert set(preds) == set(refs), "prediction/reference id mismatch"

    labels = {r["id"]: r for r in load_jsonl(RESULT / "review/effective-labels.jsonl")}

    rows = []
    for vid in sorted(preds):
        p, r = preds[vid], refs[vid]
        toks = p["asr_token_ids"]
        ref_n = norm(r["transcript"])
        hyp_n = norm(p["hypothesis"])
        ref_w, hyp_w = ref_n.split(), hyp_n.split()
        lab = labels.get(vid, {})
        rows.append(
            {
                "id": vid,
                "source_id": p["source_id"],
                "condition": p["condition"],
                "answerable": r["answerable"],
                "tokens": len(toks),
                "capped_derived": len(toks) >= MAX_ASR,
                "capped_flag": bool(p["asr_capped"]),
                "asr_eos": p.get("asr_eos"),
                "ref_words": len(ref_w),
                "hyp_words": len(hyp_w),
                "seconds": r["seconds_audio"],
                "wer": edits(ref_w, hyp_w) / len(ref_w) if ref_w else None,
                "correct": lab.get("correct"),
                "hallucinated": lab.get("hallucinated"),
                "answer_empty": not norm(p["answer"]),
                "abstained": norm(p["answer"]) in {"अज्ञात", "unknown"},
                "hyp_chars": len(hyp_n),
                "ref_chars": len(ref_n),
            }
        )

    cap = [r for r in rows if r["capped_derived"]]
    unc = [r for r in rows if not r["capped_derived"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def rate(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")

    summary = {
        "views": len(rows),
        "sources": len({r["source_id"] for r in rows}),
        "derived_capped_views": len(cap),
        "frozen_flag_true": sum(1 for r in rows if r["capped_flag"]),
        "derived_by_condition": {
            c: sum(1 for r in cap if r["condition"] == c) for c in ("clean", "noise_10db")
        },
        "capped_distinct_sources": len({r["source_id"] for r in cap}),
        "capped_both_conditions": sum(
            1
            for s in {r["source_id"] for r in cap}
            if sum(1 for r in cap if r["source_id"] == s) == 2
        ),
        "wer_capped": mean([r["wer"] for r in cap]),
        "wer_uncapped": mean([r["wer"] for r in unc]),
        "median_audio_s_capped": sorted(r["seconds"] for r in cap)[len(cap) // 2],
        "median_audio_s_uncapped": sorted(r["seconds"] for r in unc)[len(unc) // 2],
        "correct_rate_capped": rate(cap, "correct"),
        "correct_rate_uncapped": rate(unc, "correct"),
        "hallucination_rate_capped": rate(cap, "hallucinated"),
        "hallucination_rate_uncapped": rate(unc, "hallucinated"),
        "abstention_rate_capped": rate(cap, "abstained"),
        "abstention_rate_uncapped": rate(unc, "abstained"),
    }

    # Length evidence for truncation: a capped hypothesis that is far shorter
    # than its reference, or that stops mid-utterance, is materially truncated.
    hyp_vs_ref = [r["hyp_chars"] / r["ref_chars"] for r in cap if r["ref_chars"]]
    summary["capped_hyp_over_ref_chars_median"] = sorted(hyp_vs_ref)[len(hyp_vs_ref) // 2]
    summary["capped_hyp_shorter_than_ref"] = sum(1 for x in hyp_vs_ref if x < 1.0)
    summary["capped_views_with_eos"] = sum(1 for r in cap if r["asr_eos"])

    # Within-condition, paired comparison for sources that are capped in BOTH
    # conditions is not meaningful (both capped); instead compare capped views
    # against uncapped views of the same condition, and report both.
    per_condition = {}
    for c in ("clean", "noise_10db"):
        cs = [r for r in rows if r["condition"] == c and r["capped_derived"]]
        us = [r for r in rows if r["condition"] == c and not r["capped_derived"]]
        per_condition[c] = {
            "capped_n": len(cs),
            "capped_wer": mean([r["wer"] for r in cs]),
            "uncapped_wer": mean([r["wer"] for r in us]),
            "capped_correct": rate(cs, "correct"),
            "uncapped_correct": rate(us, "correct"),
            "capped_median_ref_words": sorted(r["ref_words"] for r in cs)[len(cs) // 2],
            "uncapped_median_ref_words": sorted(r["ref_words"] for r in us)[len(us) // 2],
        }
    summary["per_condition"] = per_condition

    # Source-cluster bootstrap for the capped-minus-uncapped WER difference.
    by_source: dict[str, dict] = {}
    for r in rows:
        by_source.setdefault(r["source_id"], []).append(r)
    source_ids = sorted(by_source)
    rng = random.Random(17)
    deltas = []
    for _ in range(2000):
        sample = [rng.choice(source_ids) for _ in range(len(source_ids))]
        cs = [r for s in sample for r in by_source[s] if r["capped_derived"]]
        us = [r for s in sample for r in by_source[s] if not r["capped_derived"]]
        if len(cs) < 3 or len(us) < 3:
            continue
        deltas.append(mean([r["wer"] for r in cs]) - mean([r["wer"] for r in us]))
    deltas.sort()
    if deltas:
        summary["wer_delta_capped_minus_uncapped"] = {
            "estimate": mean([r["wer"] for r in cap]) - mean([r["wer"] for r in unc]),
            "lower": deltas[int(0.025 * len(deltas))],
            "upper": deltas[int(0.975 * len(deltas)) - 1],
            "replicates": len(deltas),
        }

    # How much of the cap burden lands on answerable vs unanswerable views.
    summary["capped_answerable"] = sum(1 for r in cap if r["answerable"])
    summary["capped_unanswerable"] = sum(1 for r in cap if not r["answerable"])

    # Sanity: the frozen flag and the derived flag must disagree on exactly the
    # cap views; that is the documented metadata defect.
    summary["flag_defect_confirmed"] = (
        summary["frozen_flag_true"] == 0 and summary["derived_capped_views"] > 0
    )

    out = ROOT / "research-team/cap-diagnostic.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
