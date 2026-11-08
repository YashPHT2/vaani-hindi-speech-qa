"""Copy an explicit submission allowlist; never publish or delete originals.

python package_submission.py --output output/repository --include <reviewed-file> ...
The optional --include paths are deliberately individual files, not directories.
The reviewer selects prediction/config/metric/rating and sanitized provenance files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = [
    'README.txt', 'LICENSE-NOTICE.txt',
    'run.py', 'score.py', 'compare.py', 'smoke_metrics.py',
    'review_cases.py', 'select_secondary_review.py', 'integration_analysis.py', 'build_report.py',
    'package_submission.py', 'reproduce_analysis.py', 'verify_submission.py', 'requirements-measured.txt', 'requirements.txt', 'requirements-data.txt',
    'requirements-report.txt', 'data/prepare_fleurs.py',
    'data/build_manifest.py', 'data/candidates.json', 'data/qa.tsv',
    'data/manifest.jsonl', 'data/dev-freeze.json', 'data/full-freeze.json',
    'data/shard-provenance.json', 'data/archive/v1/qa.tsv',
    'data/archive/v1/manifest.jsonl', 'data/archive/v1/dev-freeze.json',
    'data/archive/v1/full-freeze.json',
]
# Directory prefixes copied wholesale. Used for the numeral evidence package,
# whose files are generated together and are meaningless if separated: the
# replay scripts, the records they read and EXPECTED.json form one unit.
TREES = [
    'numeral-evidence',
]
BLOCKED_NAMES = {'state.json', 'env.json', 'environment.json', 'job.json',
                 'gpu_controller.py', 'college_batch.py'}
BLOCKED_PARTS = {'controller', 'responses', 'processing', 'audio', '__pycache__',
                 '.git', '.cache', '.venv', 'team'}


def safe_file(value):
    rel = Path(value)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Expected project-relative file: {value}')
    path = (ROOT / rel).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'Outside project or not a regular file: {value}')
    lower = str(rel).lower()
    if (set(rel.parts) & BLOCKED_PARTS or rel.name in BLOCKED_NAMES
            or rel.suffix.lower() in {'.wav','.parquet','.pt','.bin','.safetensors','.pem','.key'}
            or any(term in lower for term in ('password','credential','.env','token.json'))):
        raise ValueError(f'Local/private or heavyweight file is excluded: {value}')
    if path.stat().st_size > 15 * 1024**2:
        raise ValueError(f'Unexpectedly large submission file: {value}')
    return rel, path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/repository')
    parser.add_argument('--include',nargs='*',default=[],help='Individual reviewed scientific artifact files')
    args=parser.parse_args()
    output=args.output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('Output must be a dedicated directory, not a project ancestor')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is nonempty. Choose a new directory; originals are never deleted.')
    names=list(dict.fromkeys(BASE+args.include))
    for tree in TREES:
        root=(ROOT/tree).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f'Allowlisted tree is not a directory: {tree}')
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            # Caches and editor debris are never part of the deliverable. They
            # would be rejected by safe_file anyway; skipping them keeps the
            # error message about real omissions rather than stale bytecode.
            if '__pycache__' in path.parts or path.suffix in {'.pyc','.pyo'}:
                continue
            names.append(path.relative_to(ROOT).as_posix())
    selected=[safe_file(name) for name in names]
    if not any(rel.name=='predictions.jsonl' for rel,_ in selected):
        raise ValueError('Include reviewed measured prediction artifacts before packaging the final submission.')
    output.mkdir(parents=True,exist_ok=True)
    hashes={}
    for rel,source in selected:
        destination=output/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        original=hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=original:
            raise RuntimeError(f'Copy hash mismatch: {rel}')
        hashes[rel.as_posix()]=original
    (output/'.gitignore').write_text('__pycache__/\n*.pyc\n.env\n.env.*\ndata/audio/\ndata/*.parquet\n.cache/\ntmp/\n',encoding='utf-8')
    (output/'SUBMISSION_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'repository':str(output),'copied_files':len(hashes),'bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file())},indent=2))


if __name__=='__main__':
    main()
