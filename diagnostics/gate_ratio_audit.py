"""Recompute the fallback gate's OWN criterion over the recorded token ids.

Important distinction, confirmed against the pinned source:

  HF  _retrieve_compression_ratio(tokens, vocab_size)
      = len(token_id_bytes) / len(zlib.compress(token_id_bytes))
      with token_id_bytes = each id encoded little-endian in
      int(log2(vocab_size)/8)+1 bytes.

  OpenAI whisper.utils.compression_ratio(text)
      = len(text_bytes) / len(zlib.compress(text_bytes))

These are different quantities. Only the first is what the 2.4 threshold in the
HF call actually tests. Reporting the text ratio as if it were the gate ratio
would overstate the separation, so both are reported separately here.

Read-only. Writes research-team/gate-ratio-audit.json.
"""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "research-team/modal-retrieval/rows.json"
VOCAB_SIZE = 51865  # whisper-small multilingual vocab
THRESHOLD = 2.4


def gate_ratio(tokens: list[int], vocab_size: int = VOCAB_SIZE) -> float:
    width = int(math.log2(vocab_size) / 8) + 1
    token_bytes = b"".join(t.to_bytes(width, "little") for t in tokens)
    if not token_bytes:
        return 0.0
    return len(token_bytes) / len(zlib.compress(token_bytes))


def text_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


def main() -> None:
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    records = []
    for r in rows:
        records.append(
            {
                "id": r["id"],
                "capped": bool(r["a_capped"]),
                "gate_ratio_arm_a": gate_ratio(r["a_tokens"]),
                "text_ratio_arm_a": text_ratio(r["a_hypothesis"]),
                "observed_gate_ratio": r["b_gate_ratios"][0] if r["b_gate_ratios"] else None,
                "fallback_tripped": r["b_fallback_tripped"],
            }
        )

    # Validate our reconstruction against the ratios the library itself logged.
    errors = [
        abs(x["gate_ratio_arm_a"] - x["observed_gate_ratio"])
        for x in records
        if x["observed_gate_ratio"] is not None
    ]

    capped = [x for x in records if x["capped"]]
    uncapped = [x for x in records if not x["capped"]]

    summary = {
        "vocab_size_assumed": VOCAB_SIZE,
        "threshold": THRESHOLD,
        "reconstruction_max_abs_error_vs_library": max(errors) if errors else None,
        "reconstruction_mean_abs_error_vs_library": sum(errors) / len(errors) if errors else None,
        "capped_gate_ratio": sorted(round(x["gate_ratio_arm_a"], 3) for x in capped),
        "uncapped_gate_ratio_max": max((x["gate_ratio_arm_a"] for x in uncapped), default=None),
        "capped_above_threshold": sum(1 for x in capped if x["gate_ratio_arm_a"] > THRESHOLD),
        "uncapped_above_threshold": sum(1 for x in uncapped if x["gate_ratio_arm_a"] > THRESHOLD),
        "capped_total": len(capped),
        "uncapped_total": len(uncapped),
        "text_ratio_capped_max": max((x["text_ratio_arm_a"] for x in capped), default=None),
        "records": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}
            for x in records
        ],
    }
    out = ROOT / "research-team/gate-ratio-audit.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"reconstruction vs library-logged ratio: max abs error "
          f"{summary['reconstruction_max_abs_error_vs_library']:.6f}")
    print(f"threshold {THRESHOLD} (applied to token-byte ratio)")
    print(f"capped views gate ratio: {summary['capped_gate_ratio']}")
    print(f"capped above threshold: {summary['capped_above_threshold']}/{len(capped)}")
    print(f"uncapped above threshold: {summary['uncapped_above_threshold']}/{len(uncapped)}")
    print(f"uncapped max gate ratio: {summary['uncapped_gate_ratio_max']:.3f}")
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
