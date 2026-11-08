"""Validate tiny returned historical pilot artifacts and compute frozen gates."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

def validate(directory):
    out = Path(directory)
    result = json.loads((out / "pilot-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "pilot_complete"
    config = result["config"]
    assert config["epochs"] == 3 and config["seed"] == 17 and config["slots"] == 4
    assert config["train_sources"] == 32
    assert result["gpu_contract"]["status"] == "pass"
    assert result["gpu_contract"]["zero_candidate_spans_empty"]
    for arm in ("full", "no_uncertainty"):
        check = result["actual_qa_gradient_checks"][arm]
        assert check["nonzero_count"] > 0 and check["frozen_lm"]
        assert math.isfinite(check["omitted_gradient_norm"]) and check["omitted_gradient_norm"] > 0
        payload = (out / ("pilot-" + arm + "-adapter.pt")).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == result["checkpoint_sha256"][arm]
    manifest_path = HERE.parent / "data" / "manifest.jsonl"
    historical = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    expected_train = sorted({r["source_id"] for r in historical if r["split"] == "test"},
        key=lambda s: hashlib.sha256(("acoustic-interface-pilot-v1:" + s.rsplit("-", 1)[-1]).encode()).hexdigest())[:32]
    assert result["training_source_ids"] == expected_train
    expected_dev = {r["id"] for r in historical if r["split"] == "dev"}
    assert len(expected_dev) == 24
    rows = result["rows"]
    assert len(rows) == 96
    groups = defaultdict(dict)
    for row in rows:
        assert row["id"] in expected_dev and row["source_id"] not in expected_train
        assert math.isfinite(row["nll"]) and row["nll"] >= 0
        assert math.isfinite(row["generation_s"]) and row["generation_s"] >= 0
        assert row["answer_target_tokens_including_end"] > 0
        assert len(row["selected"]) <= 4
        assert len(row["selected"]) == len(row["gates"])
        assert all(math.isfinite(v) for v in row["gates"])
        group = groups[(row["arm"], row["control"])]
        assert row["id"] not in group
        group[row["id"]] = row
    assert set(groups) == {("full", "correct"), ("full", "wrong_audio"), ("full", "text"), ("no_uncertainty", "correct")}
    assert all(set(g) == expected_dev for g in groups.values())
    correct, wrong = groups[("full", "correct")], groups[("full", "wrong_audio")]
    absent = groups[("no_uncertainty", "correct")]
    source_deltas = defaultdict(list)
    available = True
    for key in expected_dev:
        a, w = correct[key], wrong[key]
        assert a["answer_target_tokens_including_end"] == w["answer_target_tokens_including_end"]
        assert a["transcript"] == w["transcript"]
        if not a["alignment_available"]:
            assert a["selected"] == w["selected"] == []
            assert abs(a["nll"] - w["nll"]) < 1e-6
        else:
            available = available and w["counterfactual_available"]
            if w["counterfactual_available"]:
                assert w["donor"] is not None
                assert w["donor"].rsplit("-", 1)[0] != key.rsplit("-", 1)[0]
        source_deltas[a["source_id"]].append(w["nll"] - a["nll"])
    assert len(source_deltas) == 12 and all(len(v) == 2 for v in source_deltas.values())
    means = {s: sum(v)/2 for s, v in source_deltas.items()}
    delta = sum(means.values()) / 12
    positive = sum(v > 0 for v in means.values())
    full_nll = sum(r["nll"] for r in correct.values()) / 24
    no_u_nll = sum(r["nll"] for r in absent.values()) / 24
    validated = {"status": "validated", "result_sha256": hashlib.sha256((out / "pilot-result.json").read_bytes()).hexdigest(),
        "historical_manifest_current_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_qualification": "Current unchanged historical manifest checked against exact returned training IDs; remote config did not independently record its byte hash",
        "config": config, "dev_sources": 12, "dev_views": 24, "raw_control_rows": len(rows),
        "mean_source_wrong_minus_correct_nll": delta, "positive_sources": positive,
        "full_correct_nll": full_nll, "no_uncertainty_correct_nll": no_u_nll,
        "all_aligned_counterfactuals_available": available,
        "source_deltas": means, "gate_pass": available and delta >= .02 and positive >= 8 and full_nll <= no_u_nll + .05,
        "elapsed_s": result["elapsed_s"], "peak_cuda_bytes": result["peak_cuda_bytes"],
        "alignment_counts": result["alignment_counts"], "scientific_scope": "Exploratory historical-dev engineering gate, no fresh accuracy claim"}
    (out / "pilot-validation.json").write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    return validated

if __name__ == "__main__":
    print(json.dumps(validate(HERE / "artifacts"), ensure_ascii=False, indent=2))
