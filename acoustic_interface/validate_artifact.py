"""Read-only validation of returned GPU engineering artifacts; no model work."""
import hashlib
import json
import math
from pathlib import Path


def validate_smoke(directory):
    directory = Path(directory)
    record = json.loads((directory / "smoke-result.json").read_text(encoding="utf-8"))
    assert record["status"] == "real_gpu_smoke_pass"
    assert "A10" in record["device"] and record["cuda"]
    assert math.isfinite(record["loss"]) and record["loss"] > 0
    for group in ("gradient_norms", "update_norms"):
        assert all(v is None or math.isfinite(v) for v in record[group].values())
    for name in ("score.2.weight", "project.weight"):
        assert record["gradient_norms"][name] > 0
        assert record["update_norms"][name] > 0
    assert len(record["selected"]) <= 4
    assert len(record["selected"]) == len(record["gates"])
    spans = [s for s in record["span_audit"] if "discarded" not in s]
    assert all(s["frame_end"] > s["frame_start"] >= 0 and s["end"] > s["start"] >= 0 for s in spans)
    assert all(i < len(spans) for i in record["selected"])
    digest = hashlib.sha256((directory / "smoke-adapter.pt").read_bytes()).hexdigest()
    assert digest == record["checkpoint_sha256"]
    return {"status": "validated", "loss": record["loss"], "spans": len(spans),
            "selected": record["selected"], "checkpoint_sha256": digest,
            "duration_s": record["elapsed_s"], "extract_s": record["extract_s"],
            "train_step_s": record["forward_backward_update_s"]}


if __name__ == "__main__":
    print(json.dumps(validate_smoke(Path(__file__).resolve().parent / "artifacts"), indent=2))
