"""Recompute and verify all submitted metrics from immutable predictions/ratings."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/recomputed'))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = 'data/manifest.jsonl'
    predictions = {arm: f'artifacts/vaani-v2-test-{arm}-20261103/results/test-{arm}/predictions.jsonl' for arm in ('auto','hint','oracle')}
    ratings = {arm: f'artifacts/blinded-review/packet-final-adjudicated/{arm}-ratings.jsonl' for arm in predictions}
    commands = {f'{arm}-metrics.json': ['score.py', manifest, predictions[arm], '--split', 'test', '--ratings', ratings[arm]] for arm in predictions}
    commands['paired-asr-em.json'] = ['compare.py', manifest, predictions['auto'], predictions['hint'], '--split', 'test']
    commands['paired-semantic.json'] = ['integration_analysis.py', manifest, '--split', 'test', '--baseline', 'auto', predictions['auto'], ratings['auto'], '--intervention', 'hint', predictions['hint'], ratings['hint'], '--oracle', 'oracle', predictions['oracle'], ratings['oracle']]
    for name, command in commands.items():
        result = subprocess.run([sys.executable, '-X', 'utf8', *command], cwd=ROOT, capture_output=True, encoding='utf-8', check=True)
        value = json.loads(result.stdout)
        (output/name).write_text(result.stdout, encoding='utf-8')
        expected = json.loads((ROOT/'artifacts/final-analysis'/name).read_text(encoding='utf-8'))
        if value != expected:
            raise RuntimeError(f'Recomputed result differs: {name}')
        print('MATCH', name)


if __name__ == '__main__':
    main()
