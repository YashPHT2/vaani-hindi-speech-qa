"""Prepare an immutable final cascade config/archive using existing local data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from transcript_final_worker import ASR, FROZEN_HASHES, LLM, RESOURCE_LIMITS, RUN_ID, SHARD_SHA
from transcript_final_worker import SYSTEM_PROMPT, VIEW_SPECS_SHA, build_view_manifest, json_bytes, load_annotations, sha, validate_config, validate_view_specs

HERE = Path(__file__).resolve().parent
SOURCE_NAMES = ("transcript_final_worker.py", "transcript_final_runner.py", "prepare_transcript_final.py")
ANALYSIS_NAMES = ("transcript_final_analysis.py", "../score.py", "connector.py")


def prepare():
    import pyarrow.parquet as pq

    artifacts = HERE / "artifacts"
    result_dir = artifacts / "transcript-final"
    for path in (artifacts / "transcript-final-job.json", result_dir / "result.json", result_dir / "predictions.jsonl"):
        if path.exists():
            raise RuntimeError("Final evaluation execution evidence already exists. Inspect it before any new decision.")
    inputs = {"historical_manifest": HERE.parent / "data/manifest.jsonl",
        "selection": HERE / "data/selection-v1.json", "qa": HERE / "data/qa-v1.jsonl"}
    payloads = {name: path.read_bytes() for name, path in inputs.items()}
    scientific_source = HERE / "data/final-experiment-freeze-v1.json"
    original = json.loads(scientific_source.read_bytes())
    assert original["hashes"] == {"historical_manifest": FROZEN_HASHES["historical_manifest"],
        "source_selection": FROZEN_HASHES["selection"], "qa": FROZEN_HASHES["qa"]}
    historical, selection, questions = load_annotations(payloads["historical_manifest"], payloads["selection"], payloads["qa"])
    shard = HERE.parent / "data/fleurs-hi-test.parquet"
    with shard.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == SHARD_SHA
    table = pq.read_table(shard).to_pylist()
    views, _, references = build_view_manifest(table, selection, questions)
    spec_path = artifacts / "final-fresh200-input-specs.json"
    spec_bytes = spec_path.read_bytes()
    validate_view_specs(spec_bytes, views)
    maximum_duration = max(row["seconds_audio"] for row in views)
    over_30 = sum(row["seconds_audio"] > 30 for row in views) // 2
    assert over_30 == 0, "The frozen ordinary single-window pipeline requires recordings at most 30 seconds"
    assert not {row["audio_sha256"] for row in historical} & {row["audio_sha256"] for row in views}
    protocol_path = HERE / "transcript_final_protocol.md"
    decision_path = artifacts / "transcript-final-decision.json"
    assert protocol_path.is_file() and decision_path.is_file(), "A prospective final protocol and decision must be saved first"
    assert all((HERE / name).is_file() for name in ANALYSIS_NAMES), "Freeze the final scoring implementation before preparation"
    protocol_sha = sha(protocol_path.read_bytes())
    decision = json.loads(decision_path.read_bytes())
    if "protocol_sha256" in decision:
        assert decision["protocol_sha256"] == protocol_sha
    config = {"run_id": RUN_ID, "asr": list(ASR), "llm": list(LLM), "seed": 17, "memory_slots": 0,
        "system_prompt": SYSTEM_PROMPT, "asr_max_new_tokens": 224, "qa_max_new_tokens": 48,
        "asr_language": "hi", "asr_task": "transcribe", "do_sample": False, "num_beams": 1,
        "resource_limits": RESOURCE_LIMITS, "work_timeout_s": 7000, "reservation_usd": 4,
        "worst_case_cost_usd": 7502 * .00036772 + .75, "publisher_shard_sha256": SHARD_SHA,
        "input_sha256": FROZEN_HASHES, "original_view_specs_sha256": VIEW_SPECS_SHA,
        "max_audio_duration_s": maximum_duration, "sources_over_30_seconds": over_30,
        "original_source_freeze_sha256": sha(scientific_source.read_bytes()),
        "view_manifest_sha256": sha(json_bytes(views)), "reference_manifest_sha256": sha(json_bytes(references)),
        "source_sha256": {name: sha((HERE / name).read_bytes()) for name in SOURCE_NAMES + ANALYSIS_NAMES},
        "protocol_sha256": protocol_sha, "scientific_decision_sha256": sha(decision_path.read_bytes()),
        "source_ids": [record["source_id"] for record in selection], "view_ids": [row["id"] for row in views],
        "sources": 100, "views": 200, "answerable_sources": 80, "unanswerable_sources": 20,
        "waveform_policy": "Shared v2 deterministic SHA noise:publisherID seed, 10 dB white noise, shared clean/noisy gain, rounded int16 WAV",
        "timing_scope": "Synchronized prepared waveform availability through completed answer. WAV decoding, models and input validation excluded.",
        "attempt_policy": "One attempt per view. Partial rows persist. Existing cloud run and local job/result block automatic execution."}
    validate_config(config)
    artifacts.mkdir(exist_ok=True)
    config_path = artifacts / "transcript-final-config.json"
    views_path = artifacts / "transcript-final-view-manifest.json"
    refs_path = artifacts / "transcript-final-reference-manifest.json"
    for path, payload in ((config_path, json_bytes(config)), (views_path, json_bytes(views)), (refs_path, json_bytes(references))):
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError("Prepared final artifact already differs. Preserve and review it before preparing a new version: " + path.name)
        if not path.exists():
            path.write_bytes(payload)
    archive_path = artifacts / "transcript-final-source.zip"
    # Existing preparation may be verified repeatedly, but the frozen ZIP is never overwritten.
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCE_NAMES:
                archive.write(HERE / name, name)
            for name in ANALYSIS_NAMES:
                archive.write(HERE / name, "analysis/" + Path(name).name)
            for path in (config_path, views_path, refs_path, protocol_path, decision_path, scientific_source, spec_path):
                archive.write(path, path.name)
            for name, path in inputs.items():
                archive.write(path, name + path.suffix)
    else:
        with zipfile.ZipFile(archive_path) as archive:
            for name in SOURCE_NAMES:
                assert sha(archive.read(name)) == config["source_sha256"][name]
            for name in ANALYSIS_NAMES:
                assert sha(archive.read("analysis/" + Path(name).name)) == config["source_sha256"][name]
            assert archive.read(config_path.name) == json_bytes(config)
    return {"status": "prepared_for_review", "config_path": str(config_path),
        "config_sha256": sha(config_path.read_bytes()), "source_archive": str(archive_path),
        "source_archive_sha256": sha(archive_path.read_bytes()), "view_manifest_sha256": config["view_manifest_sha256"],
        "sources": 100, "views": 200, "max_audio_duration_s": maximum_duration, "sources_over_30_seconds": over_30, "worst_case_cost_usd": config["worst_case_cost_usd"],
        "launch_command": "python -m modal run --detach transcript_final_runner.py"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", "--freeze-only", action="store_true", dest="prepare",
        help="Validate existing local data and write reviewable artifacts")
    arguments = parser.parse_args()
    if not arguments.prepare:
        parser.error("Pass --prepare or --freeze-only")
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
