"""Replay the propagation test: a corrupted numeral precedes a wrong answer.

Recomputes from records/propagation.json (11 NUMBER questions x 3 ASR arms =
33 arm-items on development data):

    P(wrong answer | numeral preserved)   and its denominator
    P(wrong answer | numeral corrupted)   and its denominator
    the Fisher exact test on that 2x2

This is the mechanism evidence, not an effect size: the report says so and the
denominators here are the reason.

CPU only.

    python numeral-evidence/replay/table6_propagation.py
"""

from __future__ import annotations

import sys

from common import Checker, expected, fisher_exact_2x2, load_json


def main() -> int:
    exp = expected()["propagation"]
    prop = load_json("propagation.json")
    c = Checker("Table 6 - numeral corruption propagates to wrong answers")

    detail = prop["detail"]
    c.check("arm-items", len(detail), 33)

    present = [r for r in detail if r["support"] == "PRESENT"]
    mutated = [r for r in detail if r["support"] == "MUTATED"]

    # The claim is about confidently WRONG answers, not about every failure to be
    # correct. An abstention is a separate outcome: a deleted numeral leaves a gap
    # the reader can notice, which is why absent items abstain rather than err.
    p_wrong = sum(1 for r in present if r["outcome"] == "wrong")
    m_wrong = sum(1 for r in mutated if r["outcome"] == "wrong")

    c.check("present items", len(present), exp["n_present"])
    c.check("corrupted items", len(mutated), exp["n_corrupted"])
    c.check("wrong | preserved", round(p_wrong / len(present), 6),
            round(exp["p_wrong_given_present"], 6), tol=1e-6)
    c.check("wrong | corrupted", round(m_wrong / len(mutated), 6),
            round(exp["p_wrong_given_corrupted"], 6), tol=1e-6)

    got = fisher_exact_2x2([[m_wrong, len(mutated) - m_wrong],
                            [p_wrong, len(present) - p_wrong]])
    c.check("Fisher p", round(got["p"], 6), round(exp["fisher_p"], 6), tol=1e-6)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
