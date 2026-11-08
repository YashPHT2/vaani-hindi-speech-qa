"""Replay Table 3: the M5 answer-level routing experiment.

Recomputes from records/m5-*.json*:
    per policy: correct / wrong / abstain / answered out of 26 questions,
    repairs and newly introduced errors against CASCADE on the same question,
    net, second listens, and whether answer coverage is matched.

This is the scorer the report's table comes from, rewritten to read shipped
records instead of a scratch directory. The counting rules are unchanged:

    abstention ("अज्ञात" and variants) on an unanswerable question is correct
    an alias match or a numeric equivalence is correct
    accuracy among answered items excludes abstentions from the denominator
    repairs / new errors are per question, both directions, against CASCADE

CPU only. No audio, no model, no network.

    python numeral-evidence/replay/table3_m5_routing.py
"""

from __future__ import annotations

import sys
from collections import defaultdict

from common import Checker, expected, is_abstain, load_json, load_jsonl, norm, numeric_of


def gold_ok(question: dict, answer: str) -> bool:
    if not question.get("answerable", True):
        return is_abstain(answer)
    if is_abstain(answer):
        return False
    a = norm(answer)
    for alias in question.get("answers", []):
        al = norm(alias)
        if al and (al == a or al in a):
            return True
    g = numeric_of(question["answers"][-1] if question.get("answers") else "")
    n = numeric_of(answer)
    return bool(g is not None and n is not None and g == n)


def main() -> int:
    exp = expected()["m5"]
    rows = load_jsonl("m5-answer-rows.jsonl")
    questions = {q["id"]: q for q in load_json("m5-questions.json")}
    audit = load_json("m5-routing-audit.json")
    c = Checker("Table 3 - M5 answer-level routing")

    c.check("answer rows", len(rows), 260)
    c.check("questions", len(questions), exp["n_questions"])

    by_pol = defaultdict(dict)
    for r in rows:
        by_pol[r["policy"]][r["qid"]] = r["answer"]

    base = by_pol["CASCADE"]
    qids = sorted(base)
    c.check("questions scored", len(qids), exp["n_questions"])

    for pol in exp["policies"]:
        answers = by_pol[pol]
        correct = wrong = abst = 0
        correct_answered = 0
        repairs = introduced = 0
        for qid in qids:
            a = answers.get(qid, "")
            ok = gold_ok(questions[qid], a)
            b_ok = gold_ok(questions[qid], base.get(qid, ""))
            if is_abstain(a):
                abst += 1
                correct += 1 if ok else 0
                wrong += 0 if ok else 1
            elif ok:
                correct += 1
                correct_answered += 1
            else:
                wrong += 1
            if not b_ok and ok:
                repairs += 1
            if b_ok and not ok:
                introduced += 1

        listens = sum(len(v) for v in audit.get(pol, {}).values())
        want = exp["policies"][pol]
        c.check(f"{pol} correct", correct, want["correct"])
        c.check(f"{pol} wrong", wrong, want["wrong"])
        c.check(f"{pol} abstain", abst, want["abstain"])
        c.check(f"{pol} repairs", repairs, want["repairs"])
        c.check(f"{pol} new errors", introduced, want["new_errors"])
        c.check(f"{pol} net", repairs - introduced, want["net"])
        c.check(f"{pol} second listens", listens, want["second_listens"])

        answered = len(qids) - abst
        c.check(f"{pol} answered", answered, want["answered"])
        acc_ans = round(correct_answered / answered, 4) if answered else None
        c.check(f"{pol} accuracy among answered", acc_ans, want["accuracy_answered"], tol=1e-4)

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
