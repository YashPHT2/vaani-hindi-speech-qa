"""Verify allowlisted submission file hashes without network, models or audio."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
hashes=json.loads((root/'SUBMISSION_SHA256.json').read_text(encoding='utf-8'))
for relative, expected in hashes.items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SystemExit(f'Missing or unsafe submission path: {relative}')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f'SHA256 mismatch: {relative}')
print(f'Verified {len(hashes)} submitted files.')
