Vaani Hindi speech QA

Reproduce the results
Run the following commands from the repository root in a Python environment with NumPy:

    python verify_submission.py
    python numeral-evidence/replay/replay_all.py

The six numeral tables are recomputed from the packaged per-example records.
To reproduce the final clean/noisy evaluation with the saved judgments, run:

    python -X utf8 acoustic_interface/transcript_final_analysis.py --directory acoustic_interface/artifacts/transcript-final --primary-labels acoustic_interface/artifacts/transcript-final/review/primary-labels.jsonl --secondary-labels acoustic_interface/artifacts/transcript-final/review/secondary-labels.jsonl
    python acoustic_interface/audit_final_asr_cap.py

Original final analysis environment: Python 3.12, NumPy 2.5.1.
The 100 Hindi recordings each have a clean and a seeded 10 dB noise condition.
Answer judgments for this evaluation are model-based.

Reproduce model inference

Inference consumes GPU resources and is a separate action from offline analysis. The supplied run record prevents accidental duplicate execution. To intentionally reproduce it, create a separate copy of the code, frozen input files, configuration, source archive and reviewed freeze. Omit acoustic_interface/artifacts/transcript-final-job.json and the downloaded acoustic_interface/artifacts/transcript-final/ output directory from that copy. Use an isolated Modal workspace with an empty volume named vaani-acoustic-interface-cache-v1, because the cloud worker also protects its fixed run directory against reuse.

The runner image pins Python 3.11, PyTorch 2.6.0, Transformers 4.51.3, NumPy 1.26.4, SciPy 1.15.2, SoundFile 0.13.1, Accelerate 1.6.0, SafeTensors 0.5.3 and PyArrow 19.0.1. The local orchestration used Modal 1.5.3. The image definition and actual returned software versions are included. Keep the inference image separate from the NumPy 2.5.1 analysis environment.

Download the official Hindi FLEURS Parquet shard to data/fleurs-hi-test.parquet using the URL in data/prepare_fleurs.py. Its required SHA256 is 8a0be26c5584c1478233a6f47fb90275c713f72565327ea1da76c9f1cd5dd38b. Preserve the shipped historical manifest and fresh selection/QA files. See LICENSE-NOTICE.txt and the frozen input records in acoustic_interface/data/. The approximately 306 MB shard, audio and pretrained model weights are excluded from this package.

Populate the isolated volume with that exact shard at /fleurs-hi-test.parquet and a standard Hugging Face cache at /huggingface. Download both pinned snapshots, including processor/tokenizer files, using huggingface_hub.snapshot_download with these revisions before uploading the cache. The cloud image runs offline and does not download missing models during measurement.

from huggingface_hub import snapshot_download

snapshot_download("openai/whisper-small", revision="973afd24965f72e36ca33b3055d56a652f456b4d", cache_dir="cache-upload/huggingface/hub")
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306", cache_dir="cache-upload/huggingface/hub")

From the isolated workspace, create the volume and upload those files using Modal's volume commands. Authentication belongs in the Modal login flow.

python -m pip install modal==1.5.3
python -m modal setup
python -m modal volume create vaani-acoustic-interface-cache-v1
python -m modal volume put vaani-acoustic-interface-cache-v1 data/fleurs-hi-test.parquet /fleurs-hi-test.parquet
python -m modal volume put vaani-acoustic-interface-cache-v1 cache-upload/huggingface /huggingface

From acoustic_interface/ in that isolated copy, record the operator's authorized USD 4 reproduction reservation. The original completed ledger is a historical record. The following snippet creates the required local prelaunch record for the intentional new run and retains the original USD 30 ceiling with a USD 3 reserve.

import hashlib
import json
from pathlib import Path

config_path = Path("artifacts/transcript-final-config.json")
config = json.loads(config_path.read_bytes())
ledger = {"global_ceiling": 30, "transcript_final_diagnostic": {
    "status": "authorized_prelaunch", "run_id": config["run_id"],
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "reservation_usd": 4, "all_reserved_plus_protected_usd": 7}}
Path("artifacts/modal-cost-ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

Then use the following commands. Preparation checks the sealed selection, all source identities, every derived waveform and the frozen code. The runner additionally requires the supplied independent freeze and reservation record. It performs one A10 run with zero retries and stores partial per-view evidence in the volume. Recovery retrieves the same recorded call.

python -X utf8 prepare_transcript_final.py --prepare
python -m modal run --detach transcript_final_runner.py
# Only to recover the existing call after a local disconnection
python -m modal run --detach transcript_final_runner.py --recover

The completed evaluation record carries the scientific code and input hashes. Different hardware/software can change numerical outputs and timing. Preserve any newly generated output and its own review labels. The supplied labels apply to the packaged prediction bytes.
