"""Slot-aware numeral alignment between a reference and an ASR transcript.

THE ERROR THIS FIXES
--------------------
`build_repair_items.py` paired a gold value with the ASR numeral closest in digit length. On
gramvaani-1012 the reference is "चार सौ अठारह से चार सौ बाईस" (418 to 422) and Koyal emitted
"618 से 422". The real corruption is 418->618 at the FIRST slot. But 422 was emitted correctly
at the SECOND slot, and it is equidistant in length from 418, so the pairing grabbed 422. Every
subsequent measurement then compared 418 against a value that was never a candidate for the
same position, which invalidates the window.

THE FIX
-------
Numerals occur in ORDER in both transcripts. Align the two numeral sequences monotonically
(Needleman-Wunsch style) so slot i of the reference pairs with slot j of the transcript. A
slot is CORRUPTED only when a reference numeral is paired with a transcript numeral of
different value. Unpaired reference numerals (ASR deleted the slot) are ABSENT and out of
scope for a window-based repair.

Alignment cost:
    match same value          0
    substitute different      1
    gap (deletion/insertion)  1
A tie prefers substitution over two gaps, so a corrupted slot is not silently read as a
deletion plus an insertion.

This module is pure text logic, no audio.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "factual"))

from hindi_numbers import (  # noqa: E402
    DEVANAGARI_DIGITS,
    MULTIPLIERS,
    UNITS,
    _ADDITIVE_FRACTION_LOOKUP,
    _FRACTION_LOOKUP,
    _MULTIPLIER_LOOKUP,
    _UNIT_LOOKUP,
    fold,
    parse_number_expression,
)

STOPWORDS = {
    "और", "से", "तक", "के", "का", "की", "रूपए", "रूपया", "रुपया", "रुपये", "रूपये",
    "बजे", "बज", "मिनट", "साल", "सालों", "वर्ष", "घंटे", "घंटा", "दिन", "महीने",
    "माह", "प्रतिशत", "अंक", "लोग", "शेयर", "बार", "हफ्ता", "हफ़्ते", "हफ्ते",
    "किलो", "मीटर", "बीच", "बजकर", "पर",
}
MULTIPLIER_MARKERS = {"गुणा", "गुना", "गुन", "गुणे"}


@dataclass
class Numeral:
    value: int
    surface: str
    char_start: int
    char_end: int


def _is_numeral_word(word: str) -> bool:
    """Is this token part of a numeral?

    MUST fold before lookup. The lookup tables are built with `_with_folded`, so their keys are
    the FOLDED spellings, but the token stream comes from the ORIGINAL text. `fold()` removes
    the nukta (U+093C), so a transcript that writes "हज़ार" (with nukta) tested false against
    the folded key "हजार", the numeral run ended there, and "दस हज़ार" was truncated to "दस" =
    10. That silently changed reference values on real corpus rows (gramvaani-0011: 10,000
    became 10, inventing a corruption that did not exist).
    """
    w = fold(word)
    return (w in _UNIT_LOOKUP or w in _MULTIPLIER_LOOKUP or w in _FRACTION_LOOKUP
            or w.translate(DEVANAGARI_DIGITS).replace(",", "").isdigit())


def numerals_with_spans(text: str) -> list[Numeral]:
    """Ordered numerals with character spans, using the SAME parser as everything else.

    INDEX BUG FIXED 2026-11-08
    --------------------------
    The first version built a folded, whitespace-collapsed copy (`norm`) and returned offsets
    into THAT string. Callers then sliced the ORIGINAL string with them. Folding removes the
    nukta (U+093C) and collapsing replaces runs of whitespace with one space, so every such
    removal shifts all later offsets left. On a transcript with two nukta letters before a
    numeral the returned span pointed at the wrong characters, and `gate_d_representation.py`
    deleted the wrong token run -- silently corrupting the window it measured.

    Spans are now computed on the original string; folding happens only inside the parser
    lookup. `test_numeral_spans.py` asserts `text[start:end]` really contains the numeral.
    """
    if not text:
        return []
    # DIGIT TOKENISATION (fixed 2026-11-08)
    # --------------------------------------
    # The old pattern `[0-9०-९][0-9०-९,]*` swallowed ANY comma that followed a digit, so the
    # list "5, 6" became one token "5," -> int("56") after comma removal. Measured on the
    # corpus: 16 numeral surfaces contained a comma, and the listener pack exposed the effect
    # directly -- item004 recorded a "reference" of 56 where the transcript says "एक" and the
    # ASR says "1 अक्टूबर ... 6 अक्टूबर", i.e. two separate numerals.
    #
    # A comma is part of a number only when what follows it is a digit group, which is the
    # `(?:,[0-9]+)*` part; "5, 6" has a space after the comma and therefore stops at "5".
    tokens = [(m.group(0), m.start(), m.end())
              for m in re.finditer(r"[0-9०-९]+(?:,[0-9०-९]+)*|[^\s]+", text)]

    out: list[Numeral] = []
    run: list[tuple[str, int, int]] = []

    def flush() -> None:
        nonlocal run
        if not run:
            return
        surface = text[run[0][1]:run[-1][2]]
        value = parse_number_expression(" ".join(t for t, _, _ in run))
        if value is not None:
            out.append(Numeral(value, surface, run[0][1], run[-1][2]))
        run = []

    for tok, s, e in tokens:
        if tok in MULTIPLIER_MARKERS:
            run = []
            continue
        if tok in STOPWORDS:
            flush()
            continue
        if _is_numeral_word(tok):
            run.append((tok, s, e))
        else:
            flush()
    flush()
    return out


def align(a: list[int], b: list[int]) -> list[tuple[int | None, int | None]]:
    """Monotonic alignment of two value sequences. Returns (index_in_a, index_in_b) pairs."""
    n, m = len(a), len(b)
    INF = float("inf")
    # cost[i][j], 0-based prefixes
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if i < n and j < m:
                sub = 0 if a[i] == b[j] else 1
                # substitution preferred on ties: check strictly-less only for gaps
                if c + sub < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1] = c + sub
                    back[i + 1][j + 1] = ("sub", i, j)
            if i < n and c + 1 < cost[i + 1][j]:
                cost[i + 1][j] = c + 1
                back[i + 1][j] = ("del", i, j)
            if j < m and c + 1 < cost[i][j + 1]:
                cost[i][j + 1] = c + 1
                back[i][j + 1] = ("ins", i, j)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move, pi, pj = back[i][j]
        if move == "sub":
            pairs.append((pi, pj))
        elif move == "del":
            pairs.append((pi, None))
        else:
            pairs.append((None, pj))
        i, j = pi, pj
    pairs.reverse()
    return pairs


def align_transcripts(reference: str, hypothesis: str) -> dict:
    """Slot-level comparison of the numerals in two transcripts."""
    ref = numerals_with_spans(reference)
    hyp = numerals_with_spans(hypothesis)
    pairs = align([r.value for r in ref], [h.value for h in hyp])

    slots = []
    for i, j in pairs:
        if i is None:
            slots.append({"kind": "extra_in_hypothesis", "hyp_value": hyp[j].value,
                          "hyp_surface": hyp[j].surface})
        elif j is None:
            slots.append({"kind": "deleted_by_hypothesis", "ref_value": ref[i].value,
                          "ref_surface": ref[i].surface})
        elif ref[i].value == hyp[j].value:
            slots.append({"kind": "correct", "ref_value": ref[i].value,
                          "hyp_value": hyp[j].value,
                          "ref_surface": ref[i].surface, "hyp_surface": hyp[j].surface})
        else:
            slots.append({"kind": "corrupted", "ref_value": ref[i].value,
                          "hyp_value": hyp[j].value,
                          "ref_surface": ref[i].surface, "hyp_surface": hyp[j].surface})
    return {"reference_numerals": [r.value for r in ref],
            "hypothesis_numerals": [h.value for h in hyp], "slots": slots}


def main() -> None:
    # Regression cases from the real corpus, including the one that exposed the bug.
    cases = [
        ("ये खुल गया प्राइस बैंड चार सौ अठारह से चार सौ बाईस रूपए और मिनिमम लोट साइज़ इज़ पैंतीस शेयर रखा गया है आई पी ओ दस अगस्त तक खुला रहेगा",
         "खुल गया प्राइस बैंड 618 से 422 रुपये और मिनिमम लॉर्ड साइज 35 शेयर रखा गया है। आईपीओ 10 अगस्त तक खुला रहेगा।"),
        ("आँगनबाड़ी सहायिकाओं को पन्द्रह सौ रूपए के स्थान पर बाईस सौ सौ पचास रूपए मिलेंगे",
         "आंगनबाड़ी सहायिकाओं को 1500 रुपये के स्थान पर 2250 रुपये मिलेंगे"),
        ("राज्य के उनचालीस हजार सात सौ दो प्राथमिक", "राज्य के 39702 प्राथमिक"),
    ]
    for ref, hyp in cases:
        out = align_transcripts(ref, hyp)
        print(f"\nref numerals {out['reference_numerals']}  hyp {out['hypothesis_numerals']}")
        for s in out["slots"]:
            print("   ", s)


if __name__ == "__main__":
    main()
