"""Prepare the historical alignment inputs locally for prospective review."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
RUN_ID = 'alignment-ce-kl-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def main():
    existing = ['alignment-job.json', 'alignment-result.json', 'alignment-freeze-v1.json']
    if any((ART / name).exists() for name in existing):
        raise RuntimeError('An existing alignment freeze or run must be preserved.')

    previous = read(ART / 'scale-freeze-v1.json')['config']
    manifest = HERE.parent / 'data/manifest.jsonl'
    assert sha(manifest) == previous['historical_manifest_sha256']
    rows = {row['id']: row for row in map(json.loads, manifest.read_text(encoding='utf-8').splitlines())}
    assert len(rows) == 224
    train = set(previous['train_source_ids'])
    primary = set(previous['primary_source_ids'])
    assert len(train) == 32 and len(primary) == 11 and not train & primary
    assert 'fleurs-hi-1806' not in train | primary
    inputs = [item for item in previous['inputs'] if rows[item['id']]['source_id'] in train | primary]
    ids = {item['id'] for item in inputs}
    assert len(inputs) == len(ids) == 86
    for item in inputs:
        row = rows[item['id']]
        assert item['audio_sha256'] == row['audio_sha256'] == sha(Path(item['audio_file']))
        assert row['split'] == ('test' if row['source_id'] in train else 'dev')
    train_ids = {row_id for row_id in ids if rows[row_id]['source_id'] in train}
    dev_ids = ids - train_ids
    assert len(train_ids) == 64 and len(dev_ids) == 22
    donors = {row_id: previous['donor_by_view'][row_id] for row_id in sorted(dev_ids)}
    for row_id, donor_id in donors.items():
        assert donor_id in train_ids and rows[row_id]['condition'] == rows[donor_id]['condition']
        assert rows[row_id]['transcript'] != rows[donor_id]['transcript']

    planned_files = [
        'connector.py', 'readout_candidates.py', 'scale_worker.py', 'alignment_worker.py', 'alignment_runner.py',
        'alignment_objective.py', 'alignment_analysis.py', 'prepare_alignment.py',
        'alignment_protocol.md', 'artifacts/alignment-scientific-decision.json',
        'artifacts/scale-freeze-v1.json', 'artifacts/scale-final-decision.json', '../score.py',
    ]
    decision_path = ART / 'alignment-scientific-decision.json'
    decision = read(decision_path) if decision_path.exists() else {}
    if decision:
        assert set(decision['train_sources']) == train and set(decision['primary_dev_sources']) == primary
        assert decision['protocol_sha256'] == sha(HERE / 'alignment_protocol.md')
    config = dict(
        run_id=RUN_ID, seed=17, max_new_tokens=48,
        pilot_checkpoint_sha256=previous['pilot_checkpoint_sha256'],
        historical_manifest_sha256=sha(manifest),
        train_source_ids=previous['train_source_ids'], primary_source_ids=previous['primary_source_ids'],
        inputs=inputs, cache_sha256_by_view={row_id: previous['cache_sha256_by_view'][row_id] for row_id in sorted(ids)},
        donor_by_view=donors, donor_alignment=previous['donor_alignment'],
        arms=['text', 'ce_real', 'ce_wrong', 'kl_real', 'kl_wrong'],
        fit=dict(optimizer='AdamW', lr=.001, weight_decay=.01, clip_norm=1., batch=8, epochs=4, updates=32),
        temperature=2., kl_lambda=.5, minimum_answerable_teacher_sources=4, memory_scale=1.,
        resource_limits=previous['resource_limits'], reservation_usd=2, worst_case_cost_usd=1.27294744,
        fresh_test_loaded=False,
        source_sha256={name: sha(HERE / name) for name in planned_files if (HERE / name).exists()},
    )
    draft = dict(
        state='prospective_review_pending', config=config, scientific_decision=decision,
        missing_files=[name for name in planned_files if not (HERE / name).exists()],
        launch_condition='Separate explicit user approval bound to the final reviewed freeze hash',
    )
    (ART / 'alignment-config-draft.json').write_bytes(json.dumps(draft, ensure_ascii=False, indent=2).encode())
    print(json.dumps(dict(training_sources=32, training_views=64, primary_sources=11,
                          primary_views=22, expected_outputs=110, missing_files=draft['missing_files']), indent=2))


if __name__ == '__main__':
    main()
