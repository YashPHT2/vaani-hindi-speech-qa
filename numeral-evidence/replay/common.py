"""Shared helpers for the shipped numeral replay. CPU only, standard library plus NumPy.

Every replay script in this directory reads records that ship with the
submission. Nothing here downloads a model, reads an audio file or touches a
Modal volume; those steps are documented separately in NUMERAL_EVIDENCE.md.

Directory layout assumed by the replay scripts (all relative to the tree root):

    numeral-evidence/
        records/      shipped per-example records (JSON/JSONL/NPZ)
        replay/       these scripts
        EXPECTED.json the numbers the report quotes
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RECORDS = PKG / "records"
EXPECTED = PKG / "EXPECTED.json"

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
ABSTAIN = {"अज्ञात", "अज्ञात।", "ज्ञात नहीं", "पता नहीं", "उपलब्ध नहीं"}


def load_json(name: str):
    return json.loads((RECORDS / name).read_text(encoding="utf-8"))


def load_jsonl(name: str) -> list[dict]:
    text = (RECORDS / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def expected() -> dict:
    return json.loads(EXPECTED.read_text(encoding="utf-8"))


def norm(s: str) -> str:
    """Answer normalisation, identical to m5_score.py."""
    s = unicodedata.normalize("NFC", s or "").casefold()
    s = s.translate(DEVANAGARI_DIGITS)
    s = re.sub(r"[^\w\u0900-\u097F]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_abstain(answer: str) -> bool:
    a = norm(answer)
    if not a:
        return True
    return any(a == norm(x) or a.startswith(norm(x)) for x in ABSTAIN)


def numeric_of(s: str):
    m = re.search(r"\d+", unicodedata.normalize("NFC", s or "").translate(DEVANAGARI_DIGITS))
    return int(m.group(0)) if m else None


def auroc(labels, scores):
    """Rank-based AUROC with mid-ranks for ties.

    Implemented here rather than imported so the replay has no dependency on
    any research script. `labels` are 1 for the positive class (an error).
    """
    import numpy as np

    y = np.asarray(labels, bool)
    s = np.asarray(scores, float)
    if y.all() or not y.any():
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def clustered_bootstrap(labels, scores, clusters, n_boot=2000, seed=0):
    """Percentile CI for AUROC, resampling whole clusters (sources) with replacement."""
    import numpy as np

    y = np.asarray(labels, bool)
    s = np.asarray(scores, float)
    by = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in pick for i in by[keys[k]]]
        yy = y[idx]
        if len(set(yy.tolist())) > 1:
            vals.append(auroc(yy, s[idx]))
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_bootstrap_diff(labels, score_a, score_b, clusters, n_boot=2000, seed=2):
    """AUROC(a) - AUROC(b) over the same resampled clusters, plus P(diff <= 0)."""
    import numpy as np

    y = np.asarray(labels, bool)
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    by = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in pick for i in by[keys[k]]]
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        va, vb = auroc(yy, a[idx]), auroc(yy, b[idx])
        if va is not None and vb is not None:
            diffs.append(va - vb)
    if not diffs:
        return None
    arr = np.asarray(diffs)
    return {
        "mean": float(arr.mean()),
        "ci": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_le_0": float((arr <= 0).mean()),
    }


def fisher_exact_2x2(table):
    """Two-sided Fisher exact p and the sample odds ratio. No SciPy."""
    (a, b), (c, d) = table
    n = a + b + c + d

    def log_comb(nn, kk):
        if kk < 0 or kk > nn:
            return -math.inf
        return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)

    def prob(aa):
        return math.exp(log_comb(a + b, aa) + log_comb(c + d, a + c - aa)
                        - log_comb(n, a + c))

    row1 = a + b
    col1 = a + c
    lo = max(0, col1 - (c + d))
    hi = min(row1, col1)
    observed = prob(a)
    p_two = sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-9))
    p_two = min(1.0, p_two)
    odds = (a * d / (b * c)) if b and c else math.inf
    return {"odds_ratio": odds, "p": p_two, "table": [list(table[0]), list(table[1])]}


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    def log_comb(nn, kk):
        return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)

    tail = sum(math.exp(log_comb(n, i)) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


class Checker:
    """Collects pass/fail lines so every replay ends with one honest verdict."""

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, got, want, tol: float = 0.0):
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            ok = abs(float(got) - float(want)) <= tol
        else:
            ok = got == want
        self.rows.append((name, ok, f"{got!r} vs expected {want!r}"))
        return ok

    def check_ci(self, name: str, got, want, tol: float = 1e-6):
        ok = (got is not None and len(got) == len(want)
              and all(abs(float(g) - float(w)) <= tol for g, w in zip(got, want)))
        self.rows.append((name, ok, f"{got} vs expected {want}"))
        return ok

    def report(self) -> int:
        width = max(len(r[0]) for r in self.rows) if self.rows else 10
        print(f"\n{self.title}")
        print("-" * (width + 30))
        for name, ok, detail in self.rows:
            print(f"{name:<{width}}  {'PASS' if ok else 'FAIL':<4}  {detail}")
        failed = sum(1 for _, ok, _ in self.rows if not ok)
        print("-" * (width + 30))
        print(f"{len(self.rows) - failed}/{len(self.rows)} checks passed")
        return 0 if failed == 0 else 1
