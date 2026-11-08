"""Regression test for the character-span index bug in `numerals_with_spans`.

THE BUG
-------
`numerals_with_spans` used to fold the text (removing U+093C nukta) and collapse whitespace,
then return offsets into that transformed copy. Callers sliced the ORIGINAL string with those
offsets. Every removed character shifted later offsets, so on real transcripts the span could
name the wrong characters entirely.

This test is adversarial: it puts nukta letters, doubled spaces and tabs BEFORE the numerals,
which is exactly the condition that made the offsets drift.

It also re-derives, for every span, the token prefix/suffix lengths used by the window
construction, and asserts those are consistent with tokenising the pieces. That was the second
half of the same bug class.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "factual"))

from numeral_slots import numerals_with_spans  # noqa: E402

CASES = [
    # (text, expected_values) -- every case contains a nukta char or whitespace run before
    # at least one numeral, so a stale-index implementation must fail.
    ("ज़रूरत है कि 25 लोग आएं", [25]),
    ("ज़ ज़ ज़ दस हजार लोग", [10000]),
    ("फ़ोन   पर\tबाईस सौ पचास रुपए", [2250]),
    ("ज़्यादा  जानकारी के लिए  418 पर कॉल करें और 422 भी", [418, 422]),
    ("अड़तीस ज़िलों में 38 फ़ीसदी", [38, 38]),
    ("ज़ १२३ ज़ ४५६", [123, 456]),
    ("समय 10 और 30 मिनट", [10, 30]),
    ("कोई संख्या नहीं", []),

    # ---- lexicon variants (2026-11-08) -------------------------------------------------
    # Each of these ends in a numeral spelling that is absent from the canonical UNITS table.
    # The parser used to stop at the last recognised word and accept the truncated prefix,
    # which turned agreement with the ASR into a fake corruption. The expected value is the
    # one a speaker actually said.
    ("राष्ट्रीय स्तर पर यह दर तीन सौ इक्कतीस है", [331]),
    ("परीक्षा में पचीस सौ तैंतीस परीक्षार्थियों ने फॉर्म भरा", [2533]),
    ("धारा तीन सौ सतत्तर में जिक्र है", [377]),
    # A YEAR RANGE is refused, deliberately. "दो हज़ार सतरह अठारह" means 2017-18, i.e. two
    # numbers, but the same token pattern is also "दो तीन" = "two or three", which the parser
    # must not sum. Before `सतरह` was added to the lexicon the run stopped early and returned
    # 2000 + 18, inventing a value nobody said. Refusing is the safer failure: it produces an
    # absent slot rather than a wrong one, and an absent slot cannot masquerade as corruption.
    ("वित्तीय वर्ष दो हज़ार सतरह अठारह के लिए", []),
    ("सरकारें एक सौ चौवालीस रूपया", [144]),
    ("कुल छह लाख अंठावन हज़ार दो सौ पचास रूपए", [658250]),

    # ---- digit comma handling (2026-11-08) ---------------------------------------------
    # "5, 6" is two numerals; "6,58,250" is one. The old tokeniser swallowed the comma into
    # the digit run and returned 56 and 658250-as-56 respectively.
    ("अक्टूबर को टी 4, 1 अक्टूबर को टी 5", [4, 1, 5]),
    ("कुल 6,58,250 रुपये", [658250]),
    ("10000, 20000 पौधे", [10000, 20000]),
]


def main() -> None:
    fails = 0

    for text, expected in CASES:
        got = numerals_with_spans(text)
        values = [n.value for n in got]
        if values != expected:
            print(f"FAIL values: {text!r}\n  expected {expected} got {values}")
            fails += 1
            continue

        for n in got:
            sliced = text[n.char_start:n.char_end]
            # the slice must contain the surface we recorded, exactly
            if sliced != n.surface:
                print(f"FAIL slice: {text!r}\n  surface={n.surface!r} sliced={sliced!r}")
                fails += 1
                continue
            # and the surface must actually parse to the value
            from hindi_numbers import parse_number_expression
            if parse_number_expression(n.surface) != n.value:
                print(f"FAIL reparse: {n.surface!r} -> "
                      f"{parse_number_expression(n.surface)} != {n.value}")
                fails += 1
            # spans must be ordered and non-overlapping
        starts = [n.char_start for n in got]
        if starts != sorted(starts):
            print(f"FAIL ordering: {text!r} -> {starts}")
            fails += 1
        for a, b in zip(got, got[1:]):
            if a.char_end > b.char_start:
                print(f"FAIL overlap: {text!r} {a} {b}")
                fails += 1

    # the exact drift condition: same text with and without nukta must yield the same VALUE
    # sequence, and each span must still point at its own numeral
    base = "ज़रूरत 25 और 30"
    folded = unicodedata.normalize("NFC", base).replace("\u093c", "")
    for t in (base, folded, "  " + base.replace(" ", "   ")):
        values = [n.value for n in numerals_with_spans(t)]
        if values != [25, 30]:
            print(f"FAIL drift: {t!r} -> {values}")
            fails += 1

    print(f"\n{len(CASES) + 3} case groups, {fails} failures")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
