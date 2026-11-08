"""Offline assessment of saved pilot answers; no model loading or paid compute.

Counts exact-match transitions, never equates exact match with semantic judgment.
Preserves the published pilot flag and reports Unicode-normalized EM separately.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from score import norm


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transitions(baseline, treatment, metric):
    if baseline.keys() != treatment.keys():
        raise ValueError('Paired output IDs must match exactly')
    rescue, regress, both, neither = [], [], [], []
    for key in sorted(baseline):
        a, b = metric(baseline[key]), metric(treatment[key])
        (both if a and b else rescue if b else regress if a else neither).append(key)
    n = len(baseline)
    return {'n': n, 'rescued': len(rescue), 'regressed': len(regress),
            'unchanged_correct': len(both), 'unchanged_incorrect': len(neither),
            'net_corrected': len(rescue) - len(regress),
            'accuracy_delta_pp': 100 * (len(rescue) - len(regress)) / n,
            'rescued_ids': rescue, 'regressed_ids': regress}


def main():
    source = HERE / 'artifacts' / 'pilot-result.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    groups = {}
    for row in data['rows']:
        tag = row['arm'] + '/' + row['control']
        by_id = groups.setdefault(tag, {})
        if row['id'] in by_id:
            raise ValueError('Duplicate output')
        by_id[row['id']] = row
    assert all(len(rows) == 24 for rows in groups.values())
    baseline, full = groups['full/text'], groups['full/correct']
    def raw(row):
        return bool(row['exact_match'])
    def normalized(row):
        return norm(row['answer']) in {norm(a) for a in row['answers']}
    summary = {'source_file': str(source.relative_to(HERE)),
        'source_sha256': digest(source),
        'scope': 'Historical 12-source development set, 24 paired views; no fresh inference',
        'metric_limitations': 'EM is diagnostic, not adjudicated correctness or hallucination. No human ratings.',
        'arms': {name: {'views': len(rows),
            'raw_exact': sum(raw(r) for r in rows.values()),
            'normalized_exact': sum(normalized(r) for r in rows.values())}
            for name, rows in groups.items()},
        'full_vs_text_raw_em': transitions(baseline, full, raw),
        'full_vs_text_normalized_em': transitions(baseline, full, normalized),
        'per_condition': {condition: transitions(
            {k: r for k, r in baseline.items() if r['condition'] == condition},
            {k: r for k, r in full.items() if r['condition'] == condition}, raw)
            for condition in ('clean', 'noise_10db')},
        'asr_attributable_recovery': {
            'status': 'not measurable from this pilot',
            'reason': 'Pilot has no matched gold-transcript control. Historical oracle has different prompt/decoding and cannot be joined as if matched.'},
        'parameter_count': {
            'actual_gpu_smoke_numel': json.loads((HERE / 'artifacts/smoke-result.json').read_text(encoding='utf-8'))['parameter_count'],
            'independent_source_shape_count': (1536*64+64) + 2*(768*64+64) + (133*64+64) + (64+1) + (64*1536+1536),
            'buffers': 'uncertainty_mean and uncertainty_std are buffers, not trainable parameters',
            'qualification': 'Runtime count and explicit layer shape calculation agree. This audit does not reload pretrained models or checkpoints.'},
        'input_and_objective_audit': {
            'question': 'Separately supplied written Hindi question about factual speech, not a gold transcription of a spoken question',
            'question_embedding': 'Masked mean of frozen Qwen input-token embeddings; word-order information is lost',
            'audio_features': 'Frozen Whisper encoder, predicted-token DTW blocks, not reference-aligned test spans',
            'training': 'Answer-token CE only, no KD/alignment warmup in completed pilot',
            'gate': 'Already implemented sigmoid gate and fixed acoustic prefix insertion; zero-logit gate is 0.5, not a near-zero corrective initialization',
            'candidate_coverage': 'Dependent on emitted tokens; deletion/degenerate-alignment regions may be unrepresented',
            'budget': 'Trained K=4; K=8 unsupported by current constructor. Any new budget experiment requires versioned implementation/protocol and matched training.'}}
    assert summary['parameter_count']['actual_gpu_smoke_numel'] == summary['parameter_count']['independent_source_shape_count'] == 305281
    destination = HERE / 'artifacts' / 'assessment-pilot-audit.json'
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(destination), 'raw_em': summary['full_vs_text_raw_em'],
                      'normalized_em': summary['full_vs_text_normalized_em'],
                      'parameter_count': 305281}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
