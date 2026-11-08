"""Validate and preserve human listening annotations without inferring missing judgments."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--confirmation',type=Path,help='Separate explicit user confirmation of missed listened checkboxes')
    args=parser.parse_args()
    raw=args.input.read_bytes()
    imported=json.loads(raw.decode('utf-8-sig'))
    audit_path=HERE/'artifacts/recoverability-targets-reviewed.jsonl'
    audit_raw=audit_path.read_bytes()
    targets={r['id']:r for r in map(json.loads,audit_raw.decode('utf-8').splitlines())}
    manifest={r['id']:r for r in map(json.loads,(HERE.parent/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    assert imported['schema']=='recoverability-listening-v1'
    assert imported['target_audit_sha256']==sha(audit_raw)
    annotations=imported['annotations']
    assert len(annotations)==len(targets)==10 and {r['id'] for r in annotations}==targets.keys()
    confirmation=json.loads(args.confirmation.read_text(encoding='utf-8')) if args.confirmation else None
    confirmed=set(confirmation['confirmed_listened_ids']) if confirmation else set()
    assert confirmed<=targets.keys()
    if confirmation:
        assert confirmation['user_reply'] and confirmation['confirmation_type']=='explicit_chat_response'
    records=[]
    for a in annotations:
        t=targets[a['id']]
        m=manifest[a['id']]
        assert type(a['human_listened']) is bool and type(a['target_spoken_verified']) is bool
        assert a['source_id']==m['source_id'] and a['condition']==m['condition']
        assert a['audio_sha256']==m['audio_sha256']
        assert a['publisher_target']==t['answer_critical_reference_phrase']
        assert a['duration_s']==m['duration_s']
        assert a['verdict'] in {'audible','not_spoken','uncertain','unreviewed'}
        assert a['target_spoken_verified']==(a['human_listened'] and a['verdict']=='audible')
        interval=a['target_interval_s']
        if interval is not None:
            assert len(interval)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in interval)
            assert 0<=interval[0]<interval[1]<=m['duration_s']
        listened=a['human_listened'] or a['id'] in confirmed
        verified=listened and a['verdict']=='audible'
        record=dict(a)
        record.update(effective_human_listened=listened,effective_target_spoken_verified=verified,
            listening_confirmation_source='explicit_user_chat' if a['id'] in confirmed else 'original_form',
            verification_scope='Single-user reference-guided listening; not blinded or multi-annotator validation',
            original_form_ambiguity=(a['verdict']!='unreviewed' and not a['human_listened']),
            ambiguity_resolved_by_chat=a['id'] in confirmed,
            manual_timestamp_usable=verified and interval is not None,
            timestamp_precision='Approximate user interval; not forced alignment' if interval is not None else None,
            text_pair_eligible=t['pair_eligible_text_only'],
            diagnostic_candidate_ready=verified and t['pair_eligible_text_only'],
            asr_semantic_error_human_confirmed=a['asr_semantic_target_retained']=='no' and listened)
        records.append(record)
    directory=HERE/'artifacts/listening-imports'
    directory.mkdir(exist_ok=True)
    preserved=directory/(sha(raw)+'.json')
    if preserved.exists():
        assert preserved.read_bytes()==raw
    else:
        preserved.write_bytes(raw)
    conflicts=[r['id'] for r in records if r['original_form_ambiguity'] and not r['ambiguity_resolved_by_chat']]
    summary=dict(status='imported_with_unresolved_fields' if conflicts else 'imported',
        original_file=str(args.input.resolve()),preserved_file=str(preserved.relative_to(HERE)),
        input_sha256=sha(raw),target_audit_sha256=sha(audit_raw),views=10,source_recordings=5,
        human_listened_views=sum(r['effective_human_listened'] for r in records),
        target_spoken_verified_views=sum(r['effective_target_spoken_verified'] for r in records),
        target_spoken_verified_sources=len({r['source_id'] for r in records if r['effective_target_spoken_verified']}),
        manually_timed_verified_views=sum(r['manual_timestamp_usable'] for r in records),
        verdict_counts=dict(Counter(r['verdict'] for r in records)),
        unchecked_but_judged_ids=conflicts,
        listened_without_verdict_ids=[r['id'] for r in records if r['effective_human_listened'] and r['verdict']=='unreviewed'],
        eligible_verified_pair_ids=[r['id'] for r in records if r['diagnostic_candidate_ready']],
        provisional_pair_status=[dict(id=r['id'],listened=r['effective_human_listened'],verdict=r['verdict'],
            ready=r['diagnostic_candidate_ready']) for r in records if r['text_pair_eligible']],
        asr_semantic_judgments_known=sum(r['asr_semantic_target_retained']!='unknown' for r in records),
        no_gpu_launched=True,
        limitations=['Original form and notes preserved byte-for-byte; missing checkboxes/words are not inferred.',
            'Target phrase confirmation does not itself verify a semantic ASR error or a competing hypothesis.',
            'No forced localization or general population claims from a single user review.'])
    if args.confirmation:
        summary['chat_confirmation_sha256']=sha(args.confirmation.read_bytes())
    for name,content in [('listening-import-summary.json',summary),('listening-imported.json',dict(summary=summary,annotations=records))]:
        (HERE/'artifacts'/name).write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
