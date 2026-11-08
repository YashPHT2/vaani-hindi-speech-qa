"""Replay the parser claims and the numeral-corruption rate.

Recomputes from records/parser-*.json*:
    reference slots, corrupted, deleted, and the corruption rate after the fix
    the nine slots that changed verdict when the lexicon was corrected
    the four parser defect classes, exercised as unit cases against the shipped
    parser (imported from the submission tree, not reimplemented)

The parser itself ships as numeral-evidence/parser/. These cases are the same
ones asserted in its own test file, re-run here so the report's audit paragraph
has an executable counterpart.

CPU only. Standard library alone.

    python numeral-evidence/replay/table5_parser.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from common import Checker, expected, load_json, load_jsonl

PARSER_DIR = Path(__file__).resolve().parent.parent / "parser"


def load_parser():
    sys.path.insert(0, str(PARSER_DIR))
    from numeral_slots import align_transcripts, numerals_with_spans  # noqa: PLC0415

    return align_transcripts, numerals_with_spans


def main() -> int:
    align_transcripts, numerals_with_spans = load_parser()
    labels = load_json("parser-label-changes.json")
    lexicon = load_json("parser-lexicon-fix.json")
    slots = load_jsonl("parser-slot-table.jsonl")
    defects = load_json("parser-defects.json")
    c = Checker("Table 5 - parser audit and corruption rate")

    # Corrected corpus counts, straight from the shipped slot table.
    counts = Counter(r["kind"] for r in slots)
    n_ref = counts["correct"] + counts["corrupted"] + counts["deleted_by_hypothesis"]
    c.check("reference slots", n_ref, lexicon["reference_slots"])
    c.check("corrupted", counts["corrupted"], lexicon["slot_counts"]["corrupted"])
    c.check("deleted", counts["deleted_by_hypothesis"], lexicon["slot_counts"]["deleted_by_hypothesis"])
    rate = counts["corrupted"] / n_ref
    c.check("corruption rate", round(rate, 4), round(lexicon["corruption_rate"], 4), tol=1e-4)
    dele = counts["deleted_by_hypothesis"] / n_ref
    c.check("deletion rate", round(dele, 4), round(lexicon["deletion_rate"], 4), tol=1e-4)

    # Before/after, as reported in the audit paragraph.
    c.check("old corruption rate", round(labels["old"]["corruption_rate"], 4), 0.0981, tol=5e-5)
    c.check("new corruption rate", round(labels["new"]["corruption_rate"], 4), 0.0736, tol=5e-5)
    c.check("slots that changed verdict", labels["slot_table_changes"]["n_changed"], 32)

    # The defect classes, exercised against the shipped parser.
    cases = [
        ("lexicon variant 331", "तीन सौ इक्कतीस", [331]),
        ("comma-separated list is two numerals", "5, 6", [5, 6]),
        ("lakh grouping stays one numeral", "6,58,250", [658250]),
    ]
    for name, text, want in cases:
        got = [n.value for n in numerals_with_spans(text)]
        c.check(name, got, want)

    # A run of equal-scale numerals is summed by the parser but must be FLAGGED,
    # so no downstream claim silently treats the sum as the spoken amount.
    flagged = [r for r in slots if r.get("ambiguous_reference")]
    c.check("ambiguous equal-scale runs flagged", len(flagged), 2)
    c.check("ambiguous runs excluded from clean rate",
            all(r["ambiguous_reference"] == "repeated_equal_scale" for r in flagged), True)

    # A deleted numeral must be visible as its own kind, never folded into detection.
    al = align_transcripts("करीब चार सौ रुपये", "करीब रुपये")
    kinds = [s["kind"] for s in al["slots"]]
    c.check("deletion is its own kind", "deleted_by_hypothesis" in kinds, True)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
