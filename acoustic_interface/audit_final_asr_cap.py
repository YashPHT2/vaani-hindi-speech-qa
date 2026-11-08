"""Reconstruct ASR token-ceiling counts from immutable final predictions."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "artifacts/transcript-final"


def audit(folder):
    manifest = json.loads((folder / "artifact-manifest.json").read_bytes())
    predictions = (folder / "predictions.jsonl").read_bytes()
    digest = hashlib.sha256(predictions).hexdigest()
    if digest != manifest["sha256"]["predictions.jsonl"]:
        raise ValueError("Prediction bytes differ from frozen manifest")
    rows = [json.loads(line) for line in predictions.splitlines()]
    result_bytes = (folder / "result.json").read_bytes()
    if hashlib.sha256(result_bytes).hexdigest() != manifest["sha256"]["result.json"]:
        raise ValueError("Result bytes differ from frozen manifest")
    result = json.loads(result_bytes)
    assert rows == result["eval_rows"] and len(rows) == 200
    assert result["software"]["transformers"] == "4.51.3"
    cap = result["config"]["asr_max_new_tokens"]
    assert all(len(row["asr_token_ids"]) <= cap for row in rows)
    # This frozen short-form tensor-return path strips decoder initialization
    # and EOS. The returned sequence length is the usable ceiling diagnostic.
    capped = [row for row in rows if not row["asr_failed"] and len(row["asr_token_ids"]) == cap]
    return {"prediction_sha256": digest, "max_new_tokens": cap,
            "method": "Count returned Whisper token arrays of length 224 after library prefix/EOS stripping. Preserve frozen raw predictions.",
            "source": "https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/whisper/generation_whisper.py#L921-L973",
            "raw_asr_capped": sum(row["asr_capped"] for row in rows),
            "capped_views": len(capped), "by_condition": {c: sum(row["condition"] == c for row in capped)
                                                          for c in ("clean", "noise_10db")},
            "view_ids": [row["id"] for row in capped]}


if __name__ == "__main__":
    outcome = audit(DEFAULT)
    (DEFAULT / "asr-cap-audit.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in outcome.items() if key != "view_ids"}, indent=2))
