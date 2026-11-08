"""Replay Table 1: the four-system numeral bench.

Recomputes, from records/bench-*.json:
    WER (as published), numerals correct out of the frozen slot count,
    the failure-mode split (corrupted vs deleted), the corruption share,
    McNemar for Koyal vs SraVaani, and both Fisher tests.

CPU only. No audio, no model, no network.

    python numeral-evidence/replay/table1_bench.py
"""

from __future__ import annotations

import sys

from common import Checker, expected, fisher_exact_2x2, load_json, mcnemar_exact


def main() -> int:
    want = expected()["bench"]
    summary = load_json("bench-summary.json")
    outcomes = load_json("bench-system-outcomes.json")

    systems = ("sravaani", "koyal", "qwen", "whisper")
    c = Checker("Table 1 - four-system bench")

    # The accuracy table counts every reference slot; the failure-mode table
    # omits one slot that is correct in all four systems, so error counts and
    # corruption shares are unaffected. Both denominators are checked.
    n_failure = summary["n_slots_failure_table"]
    for system in systems:
        rows = outcomes[system]
        n = len(rows)
        correct = sum(1 for r in rows if r["outcome"] == "correct")
        corrupted = sum(1 for r in rows if r["outcome"] == "corrupted")
        deleted = sum(1 for r in rows if r["outcome"] == "deleted")
        errors = corrupted + deleted
        c.check(f"{system} slots (failure table)", n, n_failure)
        c.check(f"{system} correct (failure table)", correct, want["numerals_correct"][system])
        c.check(f"{system} WER as published", summary["systems"][system]["wer_published"],
                want["wer"][system])
        share = corrupted / errors if errors else None
        key = f"corruption_share_{system}"
        if key in want:
            c.check(f"{system} corruption share", round(share, 6), round(want[key], 6), tol=1e-6)

    # McNemar, paired on the same slots
    koyal = {f"{r['source_id']}|{r['slot']}": r["outcome"] for r in outcomes["koyal"]}
    srav = {f"{r['source_id']}|{r['slot']}": r["outcome"] for r in outcomes["sravaani"]}
    keys = [k for k in koyal if k in srav]
    b = sum(1 for k in keys if koyal[k] != "correct" and srav[k] == "correct")
    cc = sum(1 for k in keys if koyal[k] == "correct" and srav[k] != "correct")
    c.check("McNemar discordant Koyal-only-wrong", b, want["mcnemar_discordant"][0])
    c.check("McNemar discordant SraVaani-only-wrong", cc, want["mcnemar_discordant"][1])
    c.check("McNemar exact p", round(mcnemar_exact(b, cc), 6),
            round(want["mcnemar_koyal_vs_sravaani_p"], 6), tol=1e-6)

    # Fisher: corrupted vs deleted, Koyal against each other system
    def split(system):
        rows = outcomes[system]
        return (sum(1 for r in rows if r["outcome"] == "corrupted"),
                sum(1 for r in rows if r["outcome"] == "deleted"))

    kc, kd = split("koyal")
    for other, label in (("sravaani", "koyal_vs_sravaani"), ("whisper", "koyal_vs_whisper")):
        oc, od = split(other)
        got = fisher_exact_2x2([[kc, kd], [oc, od]])
        exp = want[f"fisher_{label}"]
        c.check(f"Fisher {label} OR", round(got["odds_ratio"], 4),
                round(exp["odds_ratio"], 4), tol=1e-4)
        c.check(f"Fisher {label} p", round(got["p"], 6), round(exp["p"], 6), tol=1e-6)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
