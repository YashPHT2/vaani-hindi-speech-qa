"""Offline error analysis of the frozen final evaluation.

Answers four questions the report currently asserts but does not quantify:
  1. How do word errors decompose into substitutions/insertions/deletions?
  2. Does recognition error actually track answer correctness?
  3. How does noise change answerable-question behaviour specifically?
  4. What share of total recognition error comes from the 30 capped views?

Read-only. No audio, models, network or GPU. Writes research-team/error-analysis.json
and prints a compact table.
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


def align_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein backtrace -> (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j = n, m
    sub = dele = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    preds = {p["id"]: p for p in load(RESULT / "predictions.jsonl")}
    refs = {r["id"]: r for r in load(RESULT / "references.jsonl")}
    labels = {r["id"]: r for r in load(RESULT / "review/effective-labels.jsonl")}

    rows = []
    for vid in sorted(preds):
        p, r, lab = preds[vid], refs[vid], labels[vid]
        ref_w = norm(r["transcript"]).split()
        hyp_w = norm(p["hypothesis"]).split()
        sub, dele, ins = align_counts(ref_w, hyp_w)
        rows.append(
            {
                "id": vid,
                "source_id": p["source_id"],
                "condition": p["condition"],
                "answerable": r["answerable"],
                "ref_words": len(ref_w),
                "sub": sub,
                "del": dele,
                "ins": ins,
                "err": sub + dele + ins,
                "capped": len(p["asr_token_ids"]) >= MAX_ASR,
                "correct": bool(lab["correct"]),
                "hallucinated": bool(lab["hallucinated"]),
                "abstained": norm(p["answer"]) in {"अज्ञात", "unknown"},
            }
        )

    def agg(rs, key):
        return sum(r[key] for r in rs)

    def block(rs):
        refw = agg(rs, "ref_words")
        err = agg(rs, "err")
        return {
            "views": len(rs),
            "reference_words": refw,
            "sub": agg(rs, "sub"),
            "del": agg(rs, "del"),
            "ins": agg(rs, "ins"),
            "wer": err / refw if refw else None,
            "correct_rate": agg(rs, "correct") / len(rs) if rs else None,
            "hallucination_rate": agg(rs, "hallucinated") / len(rs) if rs else None,
            "abstention_rate": agg(rs, "abstained") / len(rs) if rs else None,
        }

    summary = {"overall": block(rows)}
    for c in ("clean", "noise_10db"):
        summary[c] = block([r for r in rows if r["condition"] == c])

    for key in ("overall", "clean", "noise_10db"):
        b = summary[key]
        tot = b["sub"] + b["del"] + b["ins"]
        b["sub_share"] = b["sub"] / tot if tot else None
        b["del_share"] = b["del"] / tot if tot else None
        b["ins_share"] = b["ins"] / tot if tot else None

    ordered = sorted(rows, key=lambda r: r["err"] / r["ref_words"] if r["ref_words"] else 0)
    q = len(ordered) // 4
    buckets = []
    for i in range(4):
        chunk = ordered[i * q : (i + 1) * q] if i < 3 else ordered[3 * q :]
        wer = sum(r["err"] for r in chunk) / sum(r["ref_words"] for r in chunk)
        buckets.append(
            {
                "quartile": i + 1,
                "views": len(chunk),
                "wer": wer,
                "correct_rate": sum(r["correct"] for r in chunk) / len(chunk),
                "abstention_rate": sum(r["abstained"] for r in chunk) / len(chunk),
                "hallucination_rate": sum(r["hallucinated"] for r in chunk) / len(chunk),
            }
        )
    summary["by_wer_quartile"] = buckets

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    summary["answerable_views"] = {
        c: block([r for r in ans if r["condition"] == c]) for c in ("clean", "noise_10db")
    }
    summary["unanswerable_views"] = {
        c: block([r for r in una if r["condition"] == c]) for c in ("clean", "noise_10db")
    }

    cap = [r for r in rows if r["capped"]]
    uncap = [r for r in rows if not r["capped"]]
    total_err = agg(rows, "err")
    summary["cap_concentration"] = {
        "capped_views": len(cap),
        "capped_share_of_views": len(cap) / len(rows),
        "capped_share_of_word_errors": agg(cap, "err") / total_err if total_err else None,
        "capped_share_of_reference_words": agg(cap, "ref_words") / agg(rows, "ref_words"),
        "capped_wer": block(cap)["wer"],
        "uncapped_wer": block(uncap)["wer"],
        "capped_correct_rate": block(cap)["correct_rate"],
        "uncapped_correct_rate": block(uncap)["correct_rate"],
        "capped_insertion_share": agg(cap, "ins") / agg(cap, "err") if agg(cap, "err") else None,
        "uncapped_insertion_share": agg(uncap, "ins") / agg(uncap, "err") if agg(uncap, "err") else None,
    }

    summary["abstention"] = {
        "overall": block(rows)["abstention_rate"],
        "answerable": block(ans)["abstention_rate"],
        "unanswerable": block(una)["abstention_rate"],
        "of_abstentions_that_were_correct": sum(
            1 for r in rows if r["abstained"] and r["correct"]
        ),
        "of_abstentions_that_were_incorrect": sum(
            1 for r in rows if r["abstained"] and not r["correct"]
        ),
    }

    out = ROOT / "research-team/error-analysis.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Condition     WER      S/D/I          Correct  Abstain  Halluc")
    for key in ("clean", "noise_10db", "overall"):
        b = summary[key]
        print(
            f"{key:<13} {pct(b['wer']):<8} "
            f"{b['sub_share']:.2f}/{b['del_share']:.2f}/{b['ins_share']:.2f}    "
            f"{pct(b['correct_rate']):<8} {pct(b['abstention_rate']):<8} {pct(b['hallucination_rate'])}"
        )
    print("\nWER quartile  views  WER      Correct  Abstain")
    for b in buckets:
        print(
            f"Q{b['quartile']}            {b['views']:>5}  {pct(b['wer']):<8} "
            f"{pct(b['correct_rate']):<8} {pct(b['abstention_rate'])}"
        )
    cc = summary["cap_concentration"]
    print(
        f"\nCap concentration: {cc['capped_views']} views "
        f"({pct(cc['capped_share_of_views'])}) carry "
        f"{pct(cc['capped_share_of_word_errors'])} of word errors "
        f"while holding {pct(cc['capped_share_of_reference_words'])} of reference words."
    )
    print(
        f"  insertion share of errors: capped {pct(cc['capped_insertion_share'])}, "
        f"uncapped {pct(cc['uncapped_insertion_share'])}"
    )
    print(
        f"  correct rate: capped {pct(cc['capped_correct_rate'])}, "
        f"uncapped {pct(cc['uncapped_correct_rate'])}"
    )
    ab = summary["abstention"]
    print(
        f"\nAbstention: overall {pct(ab['overall'])}, answerable {pct(ab['answerable'])}, "
        f"unanswerable {pct(ab['unanswerable'])}"
    )
    print(
        f"  abstentions correct {ab['of_abstentions_that_were_correct']}, "
        f"incorrect {ab['of_abstentions_that_were_incorrect']}"
    )
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
