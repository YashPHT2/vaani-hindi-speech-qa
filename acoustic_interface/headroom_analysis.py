"""Exact-match diagnostics plus blinded semantic packet for matched headroom.

Use --labels to compute semantic recovery from explicitly reviewed case judgments.
No model evaluation, training, network or paid compute occurs in this program.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from score import norm


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def pairs(a, b, field):
    assert a.keys() == b.keys()
    rescued = [i for i in a if not a[i][field] and b[i][field]]
    regressed = [i for i in a if a[i][field] and not b[i][field]]
    return {'views': len(a), 'baseline_correct': sum(r[field] for r in a.values()),
            'treatment_correct': sum(r[field] for r in b.values()),
            'rescued': len(rescued), 'regressed': len(regressed),
            'net_accuracy_delta_pp': 100 * (len(rescued) - len(regressed)) / len(a),
            'rescued_ids': rescued, 'regressed_ids': regressed}


def report(groups, field):
    a, b, c, d = [groups[name] for name in ('A_question_only', 'B_asr_text', 'C_correct_text', 'D_existing_adapter')]
    attributable = [i for i in b if not b[i][field] and c[i][field]]
    recovered = [i for i in attributable if d[i][field]]
    return {'arm_correct': {k: sum(r[field] for r in v.values()) for k, v in groups.items()},
        'correct_transcript_vs_asr': pairs(b, c, field),
        'adapter_vs_asr': pairs(b, d, field),
        'asr_attributable': {'denominator': len(attributable), 'recovered': len(recovered),
            'recovery_rate': len(recovered) / len(attributable) if attributable else None,
            'subset_ids': attributable, 'recovered_ids': recovered},
        'conditions': {condition: {'correct_transcript_vs_asr': pairs(
            {k: r for k, r in b.items() if r['condition'] == condition},
            {k: r for k, r in c.items() if r['condition'] == condition}, field),
            'adapter_vs_asr': pairs({k: r for k, r in b.items() if r['condition'] == condition},
                                    {k: r for k, r in d.items() if r['condition'] == condition}, field)}
            for condition in ('clean', 'noise_10db')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--output', type=Path, help='Write a separate review summary without overwriting the primary.')
    args = parser.parse_args()
    source = HERE / 'artifacts/headroom-result.json'
    result = json.loads(source.read_text(encoding='utf-8'))
    assert result['status'] == 'matched_headroom_complete' and len(result['rows']) == 96
    output = HERE / 'artifacts/headroom-review'
    output.mkdir(exist_ok=True)
    groups, cases, mapping = {}, {}, []
    for r in result['rows']:
        by_id = groups.setdefault(r['arm'], {})
        assert r['id'] not in by_id
        r = dict(r)
        r['normalized_em'] = norm(r['answer']) in {norm(a) for a in r['answers']}
        by_id[r['id']] = r
        key = hashlib.sha256(json.dumps([r['source_id'], r['question'], r['answer']],
                                      ensure_ascii=False).encode()).hexdigest()[:20]
        case = {'case': key, 'source_id': r['source_id'], 'evidence': r['publisher_evidence'],
                'question': r['question'], 'reference': r['answers'], 'answer': r['answer']}
        assert key not in cases or cases[key] == case
        cases[key] = case
        mapping.append({'arm': r['arm'], 'id': r['id'], 'case': key})
    assert len(groups) == 4
    common = next(iter(groups.values())).keys()
    assert len(common) == 24 and all(v.keys() == common for v in groups.values())
    summary = {'result_sha256': sha(source), 'views_per_arm': 24, 'unique_sources': 12,
        'response_rows': 96, 'distinct_semantic_cases': len(cases),
        'normalization': 'Existing score.norm Unicode/digits/punctuation policy',
        'normalized_em': report(groups, 'normalized_em'),
        'limitations': ['Historical development, no fresh-test result',
                        'EM is diagnostic and must not substitute for semantic correctness',
                        'Paired clean/noisy copies are not independent examples']}
    if args.labels:
        labels = [json.loads(s) for s in args.labels.read_text(encoding='utf-8').splitlines() if s.strip()]
        by_case = {r['case']: r for r in labels}
        assert len(by_case) == len(labels) and by_case.keys() == cases.keys()
        for label in labels:
            assert type(label['correct']) is bool and type(label['hallucinated']) is bool and label['reason']
        for m in mapping:
            groups[m['arm']][m['id']].update({k: by_case[m['case']][k] for k in ('correct', 'hallucinated')})
        summary['semantic'] = report(groups, 'correct')
        summary['semantic']['hallucinated'] = {k: sum(r['hallucinated'] for r in v.values()) for k, v in groups.items()}
        summary['semantic']['labels_sha256'] = sha(args.labels)
    else:
        dump(output / 'blinded-cases.jsonl', [cases[k] for k in sorted(cases)])
        dump(output / 'private-mapping.jsonl', mapping)
    path = args.output or HERE / 'artifacts/headroom-analysis.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
