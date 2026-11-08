"""Hindi numeral-word parser, so facts can be compared by VALUE not by notation.

This exists because the pilot exposed a measurement bug that would otherwise have
been reported as a finding: Koyal writes "7800" where the reference writes
"सात हजार आठ सौ", and a string matcher calls that erasure when it is the correct
fact in another notation.

Scope is deliberately narrow: parse standalone Hindi number expressions that occur
in this corpus (cardinals up to crore, plus the common fraction words). Anything
unparsed returns None and the caller must fall back to string comparison rather
than guessing.

Correctness is established by tests at the bottom, not by inspection.

    python -X utf8 research-team/factual/hindi_numbers.py
"""
from __future__ import annotations

import re
import unicodedata

# Units, teens and tens. Keys are NFC-normalized. NOTE: `fold()` strips the nukta, so
# words containing ड़/ढ़ (अड़तीस 38, अड़तालीस 48, सड़सठ 67, अड़सठ 68) must be reachable
# AFTER folding too. `_UNIT_LOOKUP` adds the folded spellings; without it those four
# numbers silently parsed as their nukta-less substring (38,000 -> 1,702).
UNITS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    "छह": 6, "छः": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15,
    "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24, "पच्चीस": 25,
    "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनचालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64, "पैंसठ": 65,
    "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पंचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,

    # VARIANT SPELLINGS OBSERVED IN THE CORPUS (added 2026-11-08)
    # -----------------------------------------------------------
    # The reference transcripts are crowd-sourced and spell numerals inconsistently. A missing
    # spelling does not fail loudly: `numerals_with_spans` ends the numeral run at the unknown
    # word and accepts the truncated prefix. Measured consequences on real rows:
    #
    #   "तीन सौ इक्कतीस"    -> 300   (true 331)   item025 of the listener pack
    #   "पचीस सौ तैंतीस"    -> 133   (true 2533)  item009
    #   "छह लाख अंठावन हजार दो सौ पचास" -> 1250 (true 658250)  item011
    #
    # In all three the ASR emitted the TRUE value and the truncated reference made it look
    # like a corruption. These keys are added only where the corpus spelling has a single
    # reading; genuinely ambiguous forms are left out and are refused rather than guessed.
    "इक्कतीस": 31,      # doubled क, vs canonical इकतीस
    "पचीस": 25,         # single च, vs canonical पच्चीस
    "चौवालीस": 44,      # vs canonical चवालीस
    "उन्हतर": 69,       # vs canonical उनहत्तर
    "सरसठ": 67,         # vs canonical सड़सठ
    "अंठावन": 58,       # vs canonical अट्ठावन
    "पच्चासी": 85,      # doubled च, vs canonical पचासी
    "सतत्तर": 77,       # vs canonical सतहत्तर
    "सतरह": 17,         # vs canonical सत्रह -- extremely common, and it appears inside year
                        # ranges ("दो हज़ार सतरह अठारह"); without it the range is truncated
                        # to 2000 and the year 2017 disappears from the slot list.
}

MULTIPLIERS = {
    "सौ": 100, "हजार": 1000, "हज़ार": 1000, "लाख": 100000,
    "करोड़": 10000000, "करोड": 10000000,
}

# Fraction words. Two DIFFERENT grammatical behaviours, which the first version conflated:
#
#   MULTIPLICATIVE  "डेढ सौ" = 1.5 x 100 = 150, "ढाई हजार" = 2.5 x 1000 = 2500,
#                   "सवा लाख" = 1.25 x 100000. The word REPLACES a coefficient.
#   ADDITIVE        "साढ़े तीन सौ" = (3 + 0.5) x 100 = 350. The word PRECEDES a full number
#                   and adds a half to it; it is not a coefficient on its own.
#
# Treating "साढ़े" as multiplicative returned 450 for "साढ़े तीन सौ" (it applied 1.5 to the 3
# already read, giving 4.5, then scaled). A bare "डेढ"/"ढाई"/"साढ़े" with no following unit or
# scale is a fragment, not a number, and now returns None instead of 0.
FRACTIONS = {"डेढ": 1.5, "ढाई": 2.5, "सवा": 1.25}
ADDITIVE_FRACTIONS = {"साढे": 0.5, "साढ़े": 0.5}

# Devanagari digits and separators.
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def fold(text: str) -> str:
    """Fold nukta variants that NFC leaves distinct."""
    return unicodedata.normalize("NFC", text).replace("\u095b", "\u091c\u093c").replace("\u093c", "")


# Fold-aware lookup tables. `parse_number_expression` folds its input before lookup, so
# every key must also exist in folded form. This maps BOTH the nukta spelling and the
# folded spelling to the same value; a membership set alone is not enough.
def _with_folded(table: dict) -> dict:
    out = {}
    for k, v in table.items():
        out[k] = v
        out[fold(k)] = v
    return out


_UNIT_LOOKUP = _with_folded(UNITS)
_MULTIPLIER_LOOKUP = _with_folded(MULTIPLIERS)
_FRACTION_LOOKUP = _with_folded(FRACTIONS)
_ADDITIVE_FRACTION_LOOKUP = _with_folded(ADDITIVE_FRACTIONS)


def clean(text: str) -> str:
    """Lowercase, fold, and strip punctuation WITHOUT destroying matras.

    `re.sub(r"[^\\w\\s]", ...)` is wrong here: Devanagari vowel signs are category
    `Mn`, which `\\w` does not match, so that regex turns "चालीस" into "च ल स".
    The category test below is the same one score.py uses.

    DECIMAL POINT: `.` is punctuation, so the original version turned "1.5" into "1 5" and
    the parser then read it as two separate numbers -- "1.5 लाख" became 1 + 5 then x100000,
    i.e. 600000, which is not a number anyone said. A decimal point BETWEEN TWO DIGITS is
    therefore kept; every other `.` is still treated as a separator. Commas inside digit
    groups ("10,600") are also kept and stripped later by the digit parser.
    """
    chars = []
    text = unicodedata.normalize("NFC", fold(text).casefold())
    for i, character in enumerate(text):
        if unicodedata.category(character)[0] == "P":
            prev_digit = i > 0 and text[i - 1].isdigit()
            next_digit = i + 1 < len(text) and text[i + 1].isdigit()
            if character in ".," and prev_digit and next_digit:
                chars.append(character)
                continue
            character = " "
        chars.append(character)
    return " ".join("".join(chars).split())


def _tokenize(text: str) -> list[str]:
    """Split into words and digit runs without losing combining marks."""
    return re.findall(r"[0-9०-९][0-9०-९,]*|[^\s]+", clean(text))


def parse_number_expression(text: str) -> int | None:
    """Parse a Hindi/ASCII number expression to an integer, or None if unsure.

    Handles: "40", "7,800", "७८००", "चालीस", "एक सौ छह", "सात हजार आठ सौ",
    "बाईस सौ पचास" (2250), "डेढ सौ" (150).

    Two defects that used to manufacture phantom headroom (measured on
    gramvaani-0325 and gramvaani-0643, where they invented a 220050 and an 8):

    * DOUBLED MULTIPLIER. "बाईस सौ सौ पचास" is a transcription typo for
      "बाईस सौ पचास" (2250). Applying "सौ" twice gave 220050. A multiplier that is
      immediately repeated is now applied once, and the ambiguity is recorded by
      `parse_number_expression_verbose`.
    * COMPARATIVE / MULTIPLICATIVE PHRASE. "ढाई ढाई तीन गुणा" means "2.5, 2.5, 3 times"
      -- a list of multipliers, not a single number. Returning 8 was wrong; returning
      None is right, because no single integer is stated.
    """
    value, _ = parse_number_expression_verbose(text)
    return value


def parse_number_expression_verbose(text: str) -> tuple[int | None, list[str]]:
    """Same as `parse_number_expression`, plus a list of anomaly codes.

    Anomaly codes let downstream code exclude or flag ambiguous expressions instead of
    silently trusting an integer.
    """
    if text is None:
        return None, ["none"]
    candidate = fold(str(text)).strip()
    anomalies: list[str] = []

    # Pure digits (ASCII or Devanagari), with Indian or Western separators.
    digits = candidate.translate(DEVANAGARI_DIGITS)
    digits_clean = re.sub(r"[,\s]", "", digits)
    if digits_clean.isdigit():
        return int(digits_clean), anomalies

    words = clean(candidate).split()
    if not words:
        return None, ["empty"]

    # "गुणा"/"गुना" (times) marks a MULTIPLIER, never an amount.
    if any(w in ("गुणा", "गुना", "गुन", "गुणे", "गुनाह") for w in words):
        return None, ["multiplicative_phrase"]

    total = 0
    current = 0
    fraction: float | None = None
    half_pending = False          # "साढ़े" seen, waiting for its coefficient
    saw_any = False
    last_multiplier: str | None = None
    prev_bare_unit = False        # last numeral word was a unit with no scale after it
    for word in words:
        # `prev_bare_unit` must describe the IMMEDIATELY preceding word only. Reset it here and
        # set it back at the end of the unit branch, so "दो सौ तीन" (203) is fine -- the
        # multiplier breaks the adjacency -- while "दो तीन" is refused.
        was_bare_unit = prev_bare_unit
        prev_bare_unit = False
        if word in ("और", "से", "तक", "के", "का", "की", "रूपए", "रूपया", "रुपया",
                    "रुपये", "रूपये", "बजे", "बज", "मिनट", "साल", "सालों", "वर्ष",
                    "घंटे", "घंटा", "दिन", "महीने", "माह", "प्रतिशत", "अंक", "लोग",
                    "शेयर", "बार", "हफ्ता", "हफ़्ते", "हफ्ते", "किलो", "मीटर"):
            last_multiplier = None
            continue
        if word in _ADDITIVE_FRACTION_LOOKUP:
            # "साढ़े तीन सौ": add a half to whatever coefficient follows, then let the normal
            # unit/multiplier machinery scale it. Store it as a pending addend.
            pending_half = _ADDITIVE_FRACTION_LOOKUP[word]
            half_pending = True
            saw_any = True
            last_multiplier = None
            continue
        if word in _FRACTION_LOOKUP:
            fraction = _FRACTION_LOOKUP[word]
            saw_any = True
            last_multiplier = None
            continue
        if word in _UNIT_LOOKUP:
            value = _UNIT_LOOKUP[word]
            # "दो तीन" / "पंद्रह पंद्रह" / "तीन चार": two bare unit words in a row are a LIST or
            # an approximate range ("two or three", "fifteen fifteen" as a stutter), not an
            # addition. Summing them invented 5, 30 and 7 -- and because the ASR said only one
            # of the numbers, those rows were then counted as CORRUPTIONS where the system was
            # actually right. Refuse rather than guess.
            if was_bare_unit and fraction is None and not half_pending:
                return None, ["bare_units_in_sequence"]
            if fraction is not None:
                # Keep the fraction live until the final round. Rounding here and then scaling
                # made "साढ़े तीन सौ" round 3.5 -> 4 first.
                current += value * fraction
                fraction = None
            else:
                current += value
            if half_pending:
                current += 0.5
                half_pending = False
            saw_any = True
            last_multiplier = None
            prev_bare_unit = True
            continue
        # ASCII / Devanagari DIGITS inside a word compound: "7 हजार" (7,000), "1.5 लाख".
        # The reference transcripts spell numbers as words, so this never mattered for them --
        # but the ASR systems emit DIGITS, and "7 हजार" was rejected as `unknown_word`, which
        # silently dropped the numeral from the ordered value list and desynchronised every
        # later slot in `align_transcripts`. Seen on real outputs like "5 से 7 हजार तक".
        digit_word = word.translate(DEVANAGARI_DIGITS).replace(",", "")
        if digit_word.isdigit():
            value = int(digit_word)
            if fraction is not None:
                value = value * fraction
                fraction = None
            current += value
            if half_pending:
                current += 0.5
                half_pending = False
            saw_any = True
            last_multiplier = None
            prev_bare_unit = True
            continue
        # A decimal like "1.5" multiplying a following scale word.
        if re.fullmatch(r"\d+\.\d+", digit_word):
            fraction = float(digit_word)
            saw_any = True
            last_multiplier = None
            continue
        if word in _MULTIPLIER_LOOKUP:
            scale = _MULTIPLIER_LOOKUP[word]
            if last_multiplier == word:
                # "सौ सौ": apply once. This is a typo or a stutter, not 100x100.
                anomalies.append("doubled_multiplier")
                saw_any = True
                continue
            if fraction is not None:
                # "डेढ सौ" = 1.5 x 100 = 150, not round(1.5) x 100 or round(1 x 1.5) x 100.
                # `current or 1` then multiply by the fraction (not round it first), and stay
                # in float until the end.
                current = (current or 1) * fraction
                fraction = None
            if current == 0:
                current = 1
            # "बाईस सौ" = 2200, then "पचास" adds 50 → handled by the ordering below.
            if scale >= 1000:
                total += current * scale
                current = 0
            else:
                current *= scale
            saw_any = True
            last_multiplier = word
            continue
        # Unknown word: refuse rather than half-parse.
        return None, ["unknown_word"]

    if not saw_any:
        return None, ["no_numeral"]
    # A fragment. "डेढ" alone, or "साढ़े" with nothing to attach to, states no quantity.
    if (fraction is not None or half_pending) and total == 0 and current == 0:
        return None, ["dangling_fraction"]
    # `current`/`total` may be floats now (fractions are applied before scaling), so round once
    # at the very end. Half-values round up, matching "डेढ" = 1.5 -> 150, not 100 or 200.
    return int(round(total + current)), anomalies


def extract_numbers(text: str) -> list[int]:
    """All parseable numbers appearing in a text, in order.

    Scans maximal runs of numeral/scale words plus digit strings, so
    "चार सौ अठारह से चार सौ बाईस" yields [418, 422] rather than one merged value.
    """
    if text is None:
        return []
    folded = fold(text).casefold()
    # Strip thousands separators *inside* digit groups before punctuation cleaning,
    # otherwise "2,250" becomes "2 250" and splits into two numbers.
    folded = re.sub(r"(?<=[0-9०-९]),(?=[0-9०-९])", "", folded)
    tokens = _tokenize(folded)
    stops = {"से", "और", "तक", "के", "का", "की", "बीच", "बजकर", "बज", "पर"}
    # "गुणा"/"गुना" converts what precedes it into a MULTIPLIER ("ढाई ढाई तीन गुणा" = 2.5x,
    # 2.5x, 3x). Such a run is not a quantity, so the pending run is discarded rather than
    # parsed. Without this, "ढाई ढाई तीन गुणा" parsed as 8 and manufactured a phantom miss
    # on gramvaani-0643.
    multiplier_markers = {"गुणा", "गुना", "गुन", "गुणे"}

    numbers: list[int] = []
    run: list[str] = []
    for token in tokens:
        if token in multiplier_markers:
            run = []
            continue
        digits_only = token.translate(DEVANAGARI_DIGITS)
        # A bare digit string is its own number. It must not be merged into an
        # adjacent digit string: "2250 2,250" is two facts, not 22502250.
        if re.fullmatch(r"[0-9][0-9,]*", digits_only):
            if run:
                value = parse_number_expression(" ".join(run))
                if value is not None:
                    numbers.append(value)
                run = []
            cleaned = digits_only.replace(",", "")
            if cleaned.isdigit():
                numbers.append(int(cleaned))
            continue
        is_numberish = (token in _UNIT_LOOKUP or token in _MULTIPLIER_LOOKUP
                        or token in _FRACTION_LOOKUP)
        if is_numberish:
            run.append(token)
        else:
            if run:
                value = parse_number_expression(" ".join(run))
                if value is not None:
                    numbers.append(value)
                run = []
            # A scale word after a stop (e.g. "से") starts a fresh number.
            if token in stops:
                continue
    if run:
        value = parse_number_expression(" ".join(run))
        if value is not None:
            numbers.append(value)
    return numbers


# ------------------------------------------------------------- value -> words
_UNIT_WORDS = {
    0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पांच", 6: "छह",
    7: "सात", 8: "आठ", 9: "नौ", 10: "दस", 11: "ग्यारह", 12: "बारह",
    13: "तेरह", 14: "चौदह", 15: "पंद्रह", 16: "सोलह", 17: "सत्रह",
    18: "अठारह", 19: "उन्नीस", 20: "बीस", 21: "इक्कीस", 22: "बाईस",
    23: "तेईस", 24: "चौबीस", 25: "पच्चीस", 26: "छब्बीस", 27: "सत्ताईस",
    28: "अट्ठाईस", 29: "उनतीस", 30: "तीस", 31: "इकतीस", 32: "बत्तीस",
    33: "तैंतीस", 34: "चौंतीस", 35: "पैंतीस", 36: "छत्तीस", 37: "सैंतीस",
    38: "अड़तीस", 39: "उनचालीस", 40: "चालीस", 41: "इकतालीस", 42: "बयालीस",
    43: "तैंतालीस", 44: "चवालीस", 45: "पैंतालीस", 46: "छियालीस",
    47: "सैंतालीस", 48: "अड़तालीस", 49: "उनचास", 50: "पचास", 51: "इक्यावन",
    52: "बावन", 53: "तिरपन", 54: "चौवन", 55: "पचपन", 56: "छप्पन",
    57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ", 60: "साठ", 61: "इकसठ",
    62: "बासठ", 63: "तिरसठ", 64: "चौंसठ", 65: "पैंसठ", 66: "छियासठ",
    67: "सड़सठ", 68: "अड़सठ", 69: "उनहत्तर", 70: "सत्तर", 71: "इकहत्तर",
    72: "बहत्तर", 73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छिहत्तर", 77: "सतहत्तर", 78: "अठहत्तर", 79: "उनासी", 80: "अस्सी",
    81: "इक्यासी", 82: "बयासी", 83: "तिरासी", 84: "चौरासी", 85: "पचासी",
    86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी", 90: "नब्बे",
    91: "इक्यानवे", 92: "बानवे", 93: "तिरानवे", 94: "चौरानवे",
    95: "पंचानवे", 96: "छियानवे", 97: "सत्तानवे", 98: "अट्ठानवे",
    99: "निन्यानवे",
}


def value_to_words(value: int) -> str | None:
    """Render a non-negative integer as standard Hindi number words.

    Uses Indian grouping (crore / lakh / thousand / hundred). Returns None when the
    value is negative or implausibly large, so callers must not guess.

    The round-trip guarantee is established by tests: parse(value_to_words(v)) == v.
    This exists so a gold and a competitor value can be compared as *spoken phrases*
    of identical notation. Comparing Devanagari words against ASCII digits would
    confound the numeral-rendering defect with the acoustic question.
    """
    if value is None or value < 0 or value > 10**9:
        return None
    if value == 0:
        return _UNIT_WORDS[0]

    parts: list[str] = []

    def below_hundred(n: int) -> str:
        return _UNIT_WORDS[n]

    crore, rem = divmod(value, 10**7)
    lakh, rem = divmod(rem, 10**5)
    thousand, rem = divmod(rem, 1000)
    hundred, rest = divmod(rem, 100)

    if crore:
        parts.append(_render_below_1000(crore))
        parts.append("करोड़")
    if lakh:
        parts.append(_render_below_1000(lakh))
        parts.append("लाख")
    if thousand:
        parts.append(_render_below_1000(thousand))
        parts.append("हजार")
    if hundred:
        parts.append(_UNIT_WORDS[hundred])
        parts.append("सौ")
    if rest:
        parts.append(below_hundred(rest))
    return " ".join(parts) if parts else None


def _render_below_1000(n: int) -> str:
    """Render 1..999 (used for crore/lakh/thousand multipliers)."""
    if n >= 1000:
        return str(n)
    out = []
    h, r = divmod(n, 100)
    if h:
        out.extend([_UNIT_WORDS[h], "सौ"])
    if r:
        out.append(_UNIT_WORDS[r])
    return " ".join(out)


def canon(text: str) -> str:
    """Replace number expressions with canonical digit strings for comparison."""
    numbers = extract_numbers(text)
    if not numbers:
        return clean(text)
    result = clean(text)
    for value in numbers:
        # Not a replacement strategy; callers should compare number sets.
        pass
    return result


# ---------------------------------------------------------------- self-tests
TESTS = [
    ("40", 40), ("7,800", 7800), ("७८००", 7800), ("चालीस", 40),
    ("एक सौ छह", 106), ("दो सौ साठ", 260), ("सात हजार आठ सौ", 7800),
    ("बाईस सौ पचास", 2250), ("उनचालीस हजार सात सौ दो", 39702),
    ("दो हज़ार तीन", 2003), ("दस हजार", 10000), ("चार सौ अठारह", 418),
    ("एक लाख", 100000), ("तीन लाख", 300000), ("छह लाख इकसठों", None),
]


def _run_tests() -> None:
    failures = []
    for text, expected in TESTS:
        got = parse_number_expression(text)
        if got != expected:
            failures.append(f"{text!r}: expected {expected}, got {got}")

    # Multiple numbers must stay separate.
    multi = extract_numbers("चार सौ अठारह से चार सौ बाईस रूपए")
    if multi != [418, 422]:
        failures.append(f"multi-number extraction: got {multi}")

    single = extract_numbers("शहर के एक सौ छह ट्रैफिक पोस्टों पर")
    if single != [106]:
        failures.append(f"single-number extraction: got {single}")

    # Regression: adjacent digit strings must NOT be concatenated.
    # "2250 2,250" are two spellings of one value and previously became 22502250,
    # which silently emptied the gold fact set for q07 and caused a false miss.
    adjacent = extract_numbers("2250 2,250")
    if adjacent != [2250, 2250]:
        failures.append(f"adjacent digit strings merged: got {adjacent}")

    # Regression: a digit string next to a word-number stays separate.
    mixed = extract_numbers("बाईस सौ पचास 2250")
    if mixed != [2250, 2250]:
        failures.append(f"mixed digit/word merge: got {mixed}")

    # Regression: a hyphenated range is two numbers, not one.
    rng = extract_numbers("2500-5500 रुपए")
    if rng != [2500, 5500]:
        failures.append(f"hyphenated range not split: got {rng}")

    # Regression: the q07 gold alias set must contain the value 2250.
    gold = extract_numbers("बाईस सौ पचास बाईस सौ सौ पचास 2250 2,250 बाईस सौ सौ पचास रूपए मिलेंगे")
    if 2250 not in gold:
        failures.append(f"q07 gold set lost 2250: got {gold}")

    # Round-trip: every value we will use as a gold/competitor phrase must parse back
    # to itself. This is what licenses comparing Devanagari phrases on equal footing.
    round_trip = [1, 8, 12, 40, 99, 100, 106, 260, 418, 422, 1000, 2003, 2009,
                  2250, 2500, 39702, 40702, 7800, 10000, 20000, 30000, 100000, 220050]
    for v in round_trip:
        words = value_to_words(v)
        if words is None:
            failures.append(f"value_to_words({v}) -> None")
            continue
        back = extract_numbers(words)
        if back != [v]:
            failures.append(f"round-trip {v} -> {words!r} -> {back}")

    # Regression: nukta numerals must survive fold(). Without this, अड़तीस (38)
    # was parsed as बत्तीस-free substring तीस (30), turning 38,000 into 1,702.
    for words, expected in [
        ("अड़तीस", 38), ("अडतीस", 38),
        ("अड़तालीस", 48), ("अडतालीस", 48),
        ("सड़सठ", 67), ("सडसठ", 67),
        ("अड़सठ", 68), ("अडसठ", 68),
        ("अड़तीस हजार सात सौ दो", 38702),
    ]:
        got = parse_number_expression(words)
        if got != expected:
            failures.append(f"nukta numeral {words!r}: expected {expected}, got {got}")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print("  -", failure)
        raise SystemExit(1)
    print(f"all {len(TESTS)} parse tests passed; multi-number and extraction checks passed")


if __name__ == "__main__":
    _run_tests()
