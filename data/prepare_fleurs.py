"""Fetch a bounded official FLEURS shard and freeze unique-source Hindi QA inputs.

Questions/answers are researcher-authored separately, before any model inference.
The audio supplies evidence; the supplied written question asks about that evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
URL = 'https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/hi_in/test/0000.parquet'
EXPECTED_SHA256 = '8a0be26c5584c1478233a6f47fb90275c713f72565327ea1da76c9f1cd5dd38b'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.metadata_only:
        import fsspec
        with fsspec.open(URL, block_size=1024 * 1024).open() as f:
            rows = pq.read_table(f, columns=['id', 'transcription', 'raw_transcription']).to_pylist()
        # IDs identify parallel source sentences; repeated recordings stay in one split.
        distinct = {}
        for row_index, row in enumerate(rows):
            distinct.setdefault(str(row['id']), {'row_index': row_index, **row})
        ordered = sorted(distinct.values(), key=lambda r: hashlib.sha256(('vaani-2026:' + str(r['id'])).encode()).hexdigest())
        selected = ordered[:112]
        for i, row in enumerate(selected):
            row['split'] = 'dev' if i < 12 else 'test'
        (ROOT / 'candidates.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Frozen {len(selected)} unique source sentences from {len(rows)} recordings.')
        return
    shard = ROOT / 'fleurs-hi-test.parquet'
    if not shard.exists():
        temporary = shard.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(shard)
    with shard.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError('Upstream shard differs from the frozen benchmark. Preserve it and inspect; do not silently rebuild labels.')
    (ROOT / 'shard-provenance.json').write_text(json.dumps({'dataset': 'google/fleurs', 'config': 'hi_in', 'split': 'test', 'url': URL, 'bytes': shard.stat().st_size, 'sha256': digest, 'license': 'CC-BY-4.0'}, indent=2), encoding='utf-8')
    print(f'Shard verified: {shard.stat().st_size} bytes SHA256 {digest}')


if __name__ == '__main__':
    main()
