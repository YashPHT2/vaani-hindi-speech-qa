"""Characterize WHY the 30 capped views score badly: truncation vs degeneration.

Read-only, offline. A capped view can fail in two very different ways:
  (a) truncation  - the hypothesis is a faithful prefix that stops early because
                    the decoder hit the ceiling without emitting EOS;
  (b) degeneration - the decoder loops/repeats and exhausts tokens on
                    duplicated text, so raising the ceiling would not recover
                    the missing content.

Discriminating these decides whether raising max_new_tokens can help at all.

Run:  python -X utf8 research-team/cap_mechanism.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "acoustic_interface/artifacts/transcript-final"
MAX_ASR = 224


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def norm(text: str) -> str:
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


def ngram_repetition(words: list[str], n: int = 4) -> float:
    """Fraction of n-grams that are duplicates of an earlier n-gram."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def compress_ratio(hyp: str) -> float:
    """zlib ratio: low value indicates highly repetitive text."""
    import zlib

    raw = hyp.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 9)) / len(raw)


def _coverage(hyp_words: list[str], ref_words: list[str]) -> float:
    """Fraction of reference words recoverable as a longest common subsequence."""
    if not ref_words:
        return 0.0
    prev = [0] * (len(hyp_words) + 1)
    for rw in ref_words:
        cur = [0]
        for j, hw in enumerate(hyp_words, 1):
            cur.append(prev[j - 1] + 1 if rw == hw else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1] / len(ref_words)


def main() -> None:
    preds = {p["id"]: p for p in load_jsonl(RESULT / "predictions.jsonl")}
    refs = {r["id"]: r for r in load_jsonl(RESULT / "references.jsonl")}
    rows = []
    for vid in sorted(preds):
        p, r = preds[vid], refs[vid]
        capped = len(p["asr_token_ids"]) >= MAX_ASR
        hyp_n, ref_n = norm(p["hypothesis"]), norm(r["transcript"])
        hw, rw = hyp_n.split(), ref_n.split()
        lcp = 0
        for a, b in zip(hw, rw):
            if a != b:
                break
            lcp += 1
        rows.append(
            {
                "id": vid,
                "condition": p["condition"],
                "capped": capped,
                "hyp_words": len(hw),
                "ref_words": len(rw),
                "wer": edits(rw, hw) / len(rw) if rw else None,
                "rep4": ngram_repetition(hw),
                "zlib": compress_ratio(p["hypothesis"]),
                "lcp_frac": lcp / len(rw) if rw else 0.0,
                "ref_coverage": _coverage(hw, rw),
                "asr_eos": bool(p.get("asr_eos")),
            }
        )

    cap = [r for r in rows if r["capped"]]
    unc = [r for r in rows if not r["capped"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    # Degeneration vs truncation rule, fixed before looking at outcomes:
    # degenerate when content repeats (>=15% duplicate 4-grams or zlib <= 0.45).
    for r in cap:
        r["mechanism"] = (
            "degenerate" if (r["rep4"] >= 0.15 or r["zlib"] <= 0.45) else "truncation_candidate"
        )

    deg = [r for r in cap if r["mechanism"] == "degenerate"]
    trc = [r for r in cap if r["mechanism"] == "truncation_candidate"]

    summary = {
        "capped": len(cap),
        "degenerate": len(deg),
        "truncation_candidate": len(trc),
        "degenerate_wer": mean([r["wer"] for r in deg]),
        "truncation_candidate_wer": mean([r["wer"] for r in trc]),
        "uncapped_wer": mean([r["wer"] for r in unc]),
        "uncapped_rep4": mean([r["rep4"] for r in unc]),
        "capped_rep4": mean([r["rep4"] for r in cap]),
        "uncapped_zlib": mean([r["zlib"] for r in unc]),
        "capped_zlib": mean([r["zlib"] for r in cap]),
        "capped_hyp_longer_than_ref": sum(1 for r in cap if r["hyp_words"] > r["ref_words"]),
        "capped_mean_ref_coverage": mean([r["ref_coverage"] for r in cap]),
        "uncapped_mean_ref_coverage": mean([r["ref_coverage"] for r in unc]),
        "capped_zero_coverage": sum(1 for r in cap if r["ref_coverage"] == 0),
    }
    summary["cases"] = [
        {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in r.items()
            if k in {"id", "condition", "mechanism", "hyp_words", "ref_words", "wer", "rep4", "zlib", "ref_coverage"}
        }
        for r in sorted(cap, key=lambda r: -(r["wer"] or 0))
    ]

    out = ROOT / "research-team/cap-mechanism.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2, ensure_ascii=False))
    print("\ntop degenerate examples:")
    for c in summary["cases"][:6]:
        print(" ", c)
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
