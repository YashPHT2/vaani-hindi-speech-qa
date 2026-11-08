"""Build and validate researcher-annotated, speech-context QA and paired noise.

All questions, answer aliases, source selection and noise policy are frozen before
inference. Artificial noise is a stress test, not evidence about real environments.
"""
import argparse
import csv
import hashlib
import io
import json
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent


def write_wave(path, samples):
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as out:
        out.setparams((1, 2, 16000, len(pcm), 'NONE', 'not compressed'))
        out.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-only', action='store_true')
    args = parser.parse_args()
    selected = json.loads((ROOT / 'candidates.json').read_text(encoding='utf-8'))
    with (ROOT / 'qa.tsv').open(encoding='utf-8', newline='') as f:
        annotations = list(csv.DictReader(f, delimiter='\t'))
    assert len(annotations) == len(selected) == 112
    assert [int(r['index']) for r in annotations] == list(range(112))
    data = pq.read_table(ROOT / 'fleurs-hi-test.parquet').to_pylist()
    (ROOT / 'audio').mkdir(exist_ok=True)
    manifest = []
    for index, (chosen, qa) in enumerate(zip(selected, annotations)):
        if args.dev_only and chosen['split'] != 'dev':
            continue
        source = data[chosen['row_index']]
        assert source['id'] == chosen['id']
        assert source['transcription'] == chosen['transcription']
        assert source['raw_transcription'] == chosen['raw_transcription']
        rate, samples = wavfile.read(io.BytesIO(source['audio']['bytes']))
        assert rate == 16000 and samples.ndim == 1
        if np.issubdtype(samples.dtype, np.integer):
            samples = samples.astype(np.float64) / max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        else:
            samples = samples.astype(np.float64)
        assert np.isfinite(samples).all() and len(samples) > 0
        # A single deterministic noise realization is shared by all model arms.
        rng = np.random.default_rng(int(hashlib.sha256(('noise:' + str(source['id'])).encode()).hexdigest()[:16], 16))
        noise = rng.standard_normal(len(samples))
        noise *= np.sqrt(np.mean(samples**2) / (10 * np.mean(noise**2)))
        variants = {'clean': samples, 'noise_10db': samples + noise}
        common_gain = min(1.0, .98 / max(np.max(np.abs(v)) for v in variants.values()))
        for condition, signal in variants.items():
            identifier = f'fleurs-hi-{source["id"]}-{condition}'
            relative = f'audio/{identifier}.wav'
            audio_sha = write_wave(ROOT / relative, signal * common_gain)
            manifest.append({
                'id': identifier, 'source_id': f'fleurs-hi-{source["id"]}',
                'audio': relative, 'audio_sha256': audio_sha,
                'transcript': source['transcription'], 'language': 'hi',
                'split': chosen['split'], 'condition': condition,
                'question': qa['question'], 'answers': qa['answers'].split('|'),
                'answerable': qa['answers'] != 'अज्ञात',
                'task_mode': 'spoken_context_qa',
                'source_dataset': 'google/fleurs', 'source_config': 'hi_in',
                'source_original_split': 'test', 'source_row_index': chosen['row_index'],
                'original_audio_path': source['audio']['path'],
                'duration_s': len(samples) / rate, 'shared_gain': common_gain,
                'noise_snr_db': 10 if condition == 'noise_10db' else None,
                'task_instruction': 'उत्तर केवल रिकॉर्डिंग की जानकारी से दें। प्रश्न का छोटा उत्तर लिखें। जानकारी न होने पर केवल अज्ञात लिखें।',
            })
    expected = 24 if args.dev_only else 224
    assert len(manifest) == expected
    dev_ids = {r['source_id'] for r in manifest if r['split'] == 'dev'}
    test_ids = {r['source_id'] for r in manifest if r['split'] == 'test'}
    assert not dev_ids.intersection(test_ids)
    text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in manifest)
    temporary = ROOT / 'manifest.jsonl.tmp'
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(ROOT / 'manifest.jsonl')
    freeze = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['qa.tsv', 'candidates.json', 'manifest.jsonl']}
    (ROOT / ('dev-freeze.json' if args.dev_only else 'full-freeze.json')).write_text(json.dumps({'sha256': freeze, 'items': len(manifest), 'unique_sources': len(dev_ids | test_ids), 'dev_sources': len(dev_ids), 'test_sources': len(test_ids), 'notes': 'Researcher-authored labels; unanswerable items use the explicit answer अज्ञात. Paired conditions are correlated and are never counted as independent sources.'}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'items': len(manifest), 'dev_sources': len(dev_ids), 'test_sources': len(test_ids), 'sha256': freeze}, indent=2))


if __name__ == '__main__':
    main()
