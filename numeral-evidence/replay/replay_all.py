"""Run every numeral replay and print one verdict.

    python numeral-evidence/replay/replay_all.py

Exits non-zero if any table fails to reproduce. This is the command quoted in
the submission README; the per-table scripts can also be run individually.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    ("Table 1  four-system bench", "table1_bench.py"),
    ("Table 2  M1 detector vs confidence", "table2_m1_detector.py"),
    ("Table 3  M5 answer-level routing", "table3_m5_routing.py"),
    ("Table 4  probe: encoder carries value", "table4_probe.py"),
    ("Table 5  parser audit and corruption rate", "table5_parser.py"),
    ("Table 6  corruption propagates", "table6_propagation.py"),
]


def main() -> int:
    results = []
    for label, script in SCRIPTS:
        proc = subprocess.run([sys.executable, str(HERE / script)],
                              capture_output=True, text=True, encoding="utf-8")
        tail = [line for line in (proc.stdout or "").strip().splitlines() if line]
        verdict = tail[-1] if tail else "(no output)"
        results.append((label, proc.returncode == 0, verdict))
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)

    width = max(len(label) for label, _, _ in results)
    print("\nnumeral evidence replay")
    print("=" * (width + 46))
    for label, ok, verdict in results:
        print(f"{label:<{width}}  {'PASS' if ok else 'FAIL'}  {verdict}")
    print("=" * (width + 46))
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{len(results) - failed}/{len(results)} tables reproduced")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
