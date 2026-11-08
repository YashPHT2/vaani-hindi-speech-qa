"""Select the preregistered secondary sample using only a complete blinded packet."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from score import load, norm


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("packet", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    cases = load(args.packet)
    if not cases or len({row["case"] for row in cases}) != len(cases):
        raise ValueError("Expected a nonempty packet with unique case identifiers")
    groups = defaultdict(list)
    for row in cases:
        allowed = {"case", "evidence", "question", "reference", "answer"}
        if set(row) != allowed:
            raise ValueError("Packet must contain only blinded case fields")
        references = {norm(answer) for answer in row["reference"]}
        answer = norm(row["answer"])
        answerable = references != {norm("अज्ञात")}
        if answer in {"", "unknown", norm("अज्ञात")}:
            category = "abstention_or_empty"
        elif answer in references:
            category = "exact_non_abstention"
        else:
            category = "other_nonempty"
        groups[(answerable, category)].append(row)
    selected = []
    strata = []
    for (answerable, category), rows in sorted(groups.items()):
        sample = sorted(rows, key=lambda row: row["case"])[:4]
        selected.extend(sample)
        strata.append({"answerable": answerable, "output_category": category,
                       "population": len(rows), "selected": [row["case"] for row in sample]})
    args.output.mkdir(parents=True, exist_ok=False)
    packet = "".join(json.dumps(row, ensure_ascii=False) + "\n"
                     for row in sorted(selected, key=lambda row: row["case"]))
    (args.output / "secondary-cases.jsonl").write_text(packet, encoding="utf-8")
    metadata = {"input_sha256": hashlib.sha256(args.packet.read_bytes()).hexdigest(),
                "case_count": len(cases), "selected_count": len(selected), "strata": strata}
    (args.output / "selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "selected": len(selected)}))


if __name__ == "__main__":
    main()
