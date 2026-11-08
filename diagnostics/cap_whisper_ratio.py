"""Recompute repetition metrics in Whisper's own compression-ratio convention.

The earlier diagnostic used zlib(compressed)/len(raw), which is the INVERSE of
Whisper's definition. OpenAI whisper/utils.py defines:

    compression_ratio(text) = len(text.encode("utf-8")) / len(zlib.compress(...))

with a fallback threshold of 2.4 (higher = more repetitive). This script
recomputes every view under that exact convention, over the same token text the
reference implementation would score (decoded token sequence, not the
whitespace-normalized string), so the number is directly comparable to 2.4.

Read-only, offline. Writes research-team/cap-whisper-ratio.json.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "acoustic_interface/artifacts/transcript-final"
MAX_ASR = 224
WHISPER_THRESHOLD = 2.4


def load(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def whisper_compression_ratio(text: str) -> float:
    """Exact OpenAI whisper.utils.compression_ratio."""
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw)
    return len(raw) / len(compressed)


def main() -> None:
    preds = {p["id"]: p for p in load(RESULT / "predictions.jsonl")}

    rows = []
    for vid in sorted(preds):
        p = preds[vid]
        capped = len(p["asr_token_ids"]) >= MAX_ASR
        hyp = p["hypothesis"]
        rows.append(
            {
                "id": vid,
                "condition": p["condition"],
                "capped": capped,
                "ratio": whisper_compression_ratio(hyp),
                "chars": len(hyp),
            }
        )

    cap = [r for r in rows if r["capped"]]
    unc = [r for r in rows if not r["capped"]]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else float("nan")

    summary = {
        "definition": "len(utf8 bytes) / len(zlib.compress(bytes)), OpenAI whisper convention",
        "whisper_default_threshold": WHISPER_THRESHOLD,
        "capped_ratio_mean": mean([r["ratio"] for r in cap]),
        "capped_ratio_median": median([r["ratio"] for r in cap]),
        "capped_ratio_min": min(r["ratio"] for r in cap),
        "capped_ratio_max": max(r["ratio"] for r in cap),
        "uncapped_ratio_mean": mean([r["ratio"] for r in unc]),
        "uncapped_ratio_median": median([r["ratio"] for r in unc]),
        "uncapped_ratio_max": max(r["ratio"] for r in unc),
        "capped_above_threshold": sum(1 for r in cap if r["ratio"] > WHISPER_THRESHOLD),
        "uncapped_above_threshold": sum(1 for r in unc if r["ratio"] > WHISPER_THRESHOLD),
        "separation": {
            "threshold": WHISPER_THRESHOLD,
            "capped_detected": sum(1 for r in cap if r["ratio"] > WHISPER_THRESHOLD),
            "capped_total": len(cap),
            "uncapped_flagged_false_positive": sum(
                1 for r in unc if r["ratio"] > WHISPER_THRESHOLD
            ),
            "uncapped_total": len(unc),
        },
        "cases": [
            {"id": r["id"], "capped": r["capped"], "ratio": round(r["ratio"], 3), "chars": r["chars"]}
            for r in sorted(rows, key=lambda r: -r["ratio"])
        ],
    }

    out = ROOT / "research-team/cap-whisper-ratio.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Whisper-convention compression ratio (higher = more repetitive, threshold 2.4)")
    print(f"  capped   n={len(cap)}  mean={summary['capped_ratio_mean']:.2f}  "
          f"median={summary['capped_ratio_median']:.2f}  "
          f"range=[{summary['capped_ratio_min']:.2f}, {summary['capped_ratio_max']:.2f}]")
    print(f"  uncapped n={len(unc)}  mean={summary['uncapped_ratio_mean']:.2f}  "
          f"median={summary['uncapped_ratio_median']:.2f}  "
          f"max={summary['uncapped_ratio_max']:.2f}")
    print()
    print(f"  capped above 2.4 : {summary['capped_above_threshold']}/{len(cap)}")
    print(f"  uncapped above 2.4: {summary['uncapped_above_threshold']}/{len(unc)}")
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
