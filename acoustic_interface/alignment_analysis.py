"""Offline blinded QA packet and paired source-level alignment outcome."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
ARMS = ('text', 'ce_real', 'ce_wrong', 'kl_real', 'kl_wrong')
sys.path.insert(0, str(HERE.parent))
from score import norm


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def case_key(row):
    fields = [row['source_id'], row['question'], row['answer']]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()[:20]


def combine(primary, independent):
    conflict = any(primary[k] != independent[k] for k in ('correct', 'hallucinated'))
    return dict(correct=primary['correct'] and independent['correct'] and not conflict,
                hallucinated=primary['hallucinated'] or independent['hallucinated'],
                ambiguous=conflict or primary.get('ambiguous', False) or independent.get('ambiguous', False),
                primary=primary, independent=independent)


def paired_gate(rows, labels, strict):
    def supported(label):
        return label['correct'] and not label['hallucinated'] and not (strict and label.get('ambiguous', False))

    by_source = {}
    for row in rows:
        by_source.setdefault(row['source_id'], {}).setdefault(row['id'], {})[row['arm']] = labels[case_key(row)]
    sources = []
    for source_id, views in sorted(by_source.items()):
        assert len(views) == 2
        details = []
        for view_id, arms in sorted(views.items()):
            assert set(arms) == set(ARMS)
            correct = {arm: bool(supported(label)) for arm, label in arms.items()}
            specific = (not correct['ce_real'] and correct['kl_real'] and not correct['kl_wrong']
                        and all(not arms[arm].get('ambiguous', False) for arm in ('ce_real', 'kl_real', 'kl_wrong')))
            details.append(dict(id=view_id, correct=correct, specific_rescue=specific,
                                actual_delta=int(correct['kl_real']) - int(correct['ce_real']),
                                donor_delta=int(correct['kl_wrong']) - int(correct['ce_wrong'])))
        actual = statistics.mean(row['actual_delta'] for row in details)
        donor = statistics.mean(row['donor_delta'] for row in details)
        sources.append(dict(source_id=source_id, views=details, actual_delta=actual, donor_delta=donor))
    assert len(sources) == 11
    positive = sum(row['actual_delta'] > 0 and row['actual_delta'] > row['donor_delta'] for row in sources)
    regressions = sum(row['actual_delta'] < 0 for row in sources)
    actual_total = sum(row['actual_delta'] for row in sources)
    specificity_total = sum(row['actual_delta'] - row['donor_delta'] for row in sources)
    specific_sources = sum(any(view['specific_rescue'] for view in row['views']) for row in sources)
    passed = positive >= 2 and regressions <= 1 and actual_total >= 1 and specificity_total >= 1 and specific_sources >= 2
    correct = {arm: sum(view['correct'][arm] for row in sources for view in row['views']) for arm in ARMS}
    return dict(passed=passed, strict=strict, sources=sources, primary_views=22, correct=correct,
                positive_actual_over_donor_sources=positive, regressed_sources=regressions,
                actual_source_equivalents=actual_total, actual_minus_donor_source_equivalents=specificity_total,
                unambiguous_specific_rescue_sources=specific_sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-labels', type=Path)
    parser.add_argument('--independent-labels', type=Path)
    args = parser.parse_args()
    raw = (ART / 'alignment-result.json').read_bytes()
    result = json.loads(raw)
    job = json.loads((ART / 'alignment-job.json').read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    assert job['result_sha256'] == digest and len(result['teacher_rows']) == 32
    summary = dict(status=result['status'], result_sha256=digest, teacher_eligibility=result['teacher_eligibility'])
    rows = result['eval_rows']
    if result['status'] == 'alignment_teacher_gate_stopped':
        assert not rows and not any(result['fit_trace'].values())
    else:
        assert result['status'] == 'alignment_complete' and len(rows) == 110
        assert len({(row['id'], row['arm']) for row in rows}) == 110
        assert {row['arm'] for row in rows} == set(ARMS)
        manifest = {row['id']: row for row in jsonl(HERE.parent / 'data/manifest.jsonl')}
        known = {}
        for name in ('headroom-review/independent-labels.jsonl', 'token-limit-review/independent-labels.jsonl',
                     'readout-review/independent-labels.jsonl', 'scale-review/reconciled-labels.jsonl'):
            known.update({label['case']: label for label in jsonl(ART / name)})
        for label in jsonl(ART / 'headroom-review/model-labels.jsonl'):
            if label['case'] in known:
                known[label['case']] = combine(label, known[label['case']])
        cases = {}
        for row in rows:
            reference = manifest[row['id']]
            row['question'] = reference['question']
            assert reference['split'] == 'dev' and reference['source_id'] == row['source_id'] != 'fleurs-hi-1806'
            key = case_key(row)
            if key not in known:
                cases[key] = dict(case=key, source_id=row['source_id'], question=row['question'],
                                  answer=row['answer'], reference=reference['answers'], evidence=reference['transcript'])
        packet = ART / 'alignment-review'
        packet.mkdir(exist_ok=True)
        (packet / 'new-blinded-cases.jsonl').write_text(''.join(json.dumps(cases[key], ensure_ascii=False) + '\n'
                                                              for key in sorted(cases)), encoding='utf-8')
        (packet / 'private-mapping.jsonl').write_text(''.join(json.dumps(dict(id=row['id'], arm=row['arm'], case=case_key(row))) + '\n'
                                                             for row in rows), encoding='utf-8')
        summary['new_distinct_review_cases'] = len(cases)
        summary['normalized_em'] = {arm: sum(norm(row['answer']) in {norm(a) for a in manifest[row['id']]['answers']}
                                             for row in rows if row['arm'] == arm) for arm in ARMS}
        summary['mean_nll'] = {arm: statistics.mean(row['nll'] for row in rows if row['arm'] == arm) for arm in ARMS}
        summary['output_counts'] = {}
        for arm in ARMS:
            arm_rows = [row for row in rows if row['arm'] == arm]
            summary['output_counts'][arm] = dict(
                empty=sum(not row['answer'].strip() for row in arm_rows),
                abstained=sum(norm(row['answer']) in ('अज्ञात', 'unknown') for row in arm_rows),
                finish=dict(Counter(row.get('finish') or 'unreported' for row in arm_rows)),
            )
        if args.primary_labels or args.independent_labels:
            assert args.primary_labels and args.independent_labels
            primary = {label['case']: label for label in jsonl(args.primary_labels)}
            independent = {label['case']: label for label in jsonl(args.independent_labels)}
            assert primary.keys() == independent.keys() == cases.keys()
            known.update({key: combine(primary[key], independent[key]) for key in cases})
            for arm in ARMS:
                arm_labels = [known[case_key(row)] for row in rows if row['arm'] == arm]
                summary['output_counts'][arm]['unsupported_assertions'] = sum(label['hallucinated'] for label in arm_labels)
                summary['output_counts'][arm]['ambiguous'] = sum(label.get('ambiguous', False) for label in arm_labels)
            summary['reconciled'] = paired_gate(rows, known, strict=False)
            summary['strict'] = paired_gate(rows, known, strict=True)
            summary['utility_gate_passed'] = summary['reconciled']['passed'] and summary['strict']['passed']
    (ART / 'alignment-analysis.json').write_bytes(json.dumps(summary, ensure_ascii=False, indent=2).encode())
    print(json.dumps({k: v for k, v in summary.items() if k not in ('reconciled', 'strict')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
