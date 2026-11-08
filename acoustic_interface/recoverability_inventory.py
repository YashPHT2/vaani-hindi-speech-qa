"""Offline, provenance-aware inventory. Exact text overlap is candidate screening only."""
from collections import Counter, defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
import wave

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT))
from score import norm


def readlines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_span(reference,aliases,hypothesis):
    matches=[(reference.find(a),a) for a in aliases if a and a in reference and a!='अज्ञात']
    if not matches:
        return dict(reference_span=None,reason='No literal alias match; needs linguistic review')
    start,phrase=max(matches,key=lambda x:len(x[1]))
    end=start+len(phrase)
    words=list(re.finditer(r'\S+',reference))
    indices=[i for i,w in enumerate(words) if w.start()<end and w.end()>start]
    rw=[norm(w.group()) for w in words]
    hw=[norm(w) for w in hypothesis.split()] if hypothesis is not None else []
    target=set(indices)
    edits=[]
    if hypothesis is not None:
        for tag,i,j,k,l in difflib.SequenceMatcher(a=rw,b=hw,autojunk=False).get_opcodes():
            if tag!='equal' and (target.intersection(range(i,j)) or tag=='insert' and min(indices)<=i<=max(indices)+1):
                edits.append(dict(operation=tag,reference_word_indices=[i,j],hypothesis_word_indices=[k,l],
                    reference_text=' '.join(w.group() for w in words[i:j]),
                    hypothesis_text=' '.join(hypothesis.split()[k:l])))
    return dict(reference_span=dict(text=phrase,char_start=start,char_end=end,word_indices=indices),
        target_intersecting_edit_candidates=edits,
        caveat='Orthographic alignment only; not phonetic equivalence, validated errors, timestamps or spoken truth')


def main():
    art=HERE/'artifacts'
    out=art/'recoverability-inventory'
    out.mkdir(exist_ok=True)
    mp=PROJECT/'data/manifest.jsonl'
    manifest=readlines(mp)
    assert len(manifest)==224 and len({r['source_id'] for r in manifest})==112
    paths=[PROJECT/'artifacts/vaani-v2-dev-20261103/results/dev-hint/predictions.jsonl',
           PROJECT/'artifacts/vaani-v2-test-hint-20261103/results/test-hint/predictions.jsonl']
    legacy={r['id']:r for p in paths for r in readlines(p)}
    assert len(legacy)==224
    headroom=json.loads((art/'headroom-result.json').read_text(encoding='utf-8'))
    current={r['id']:r for r in headroom['rows'] if r['arm']=='B_asr_text'}
    pilot=json.loads((art/'pilot-result.json').read_text(encoding='utf-8'))
    train_cache_expected=set(pilot['training_source_ids'])
    subset=set(json.loads((art/'headroom-analysis-independent.json').read_text(encoding='utf-8'))['semantic']['asr_attributable']['subset_ids'])
    predicted={r['id']:r['span_audit'] for r in pilot['rows'] if r['arm']=='full' and r['control']=='correct'}
    views=[]
    for row in manifest:
        old=legacy[row['id']]
        assert old['audio_sha256']==row['audio_sha256']
        audio=PROJECT/'data'/row['audio']
        assert sha(audio)==row['audio_sha256']
        with wave.open(str(audio),'rb') as stream:
            rate,channels,frames=stream.getframerate(),stream.getnchannels(),stream.getnframes()
        assert rate==16000 and channels==1 and abs(frames/rate-row['duration_s'])<1e-6
        is_current=row['id'] in current
        hypothesis=current[row['id']]['transcript_used'] if is_current else None
        categories=['numeric_value'] if any(re.search(r'\d',a) for a in row['answers']) else ['lexical_unspecified'] if row['answerable'] else ['unanswerable_excluded']
        literal_present=any(norm(a) in norm(hypothesis) for a in row['answers']) if hypothesis is not None and row['answerable'] else None
        view=dict(id=row['id'],source_id=row['source_id'],historical_split=row['split'],
            research_role='historical_development' if row['split']=='dev' else 'historical_training_pool',
            condition=row['condition'],source_row_index=row['source_row_index'],
            original_audio_path=row['original_audio_path'],audio=str(audio.resolve()),
            audio_sha256=row['audio_sha256'],audio_hash_verified=True,duration_s=row['duration_s'],
            source_recording_unit=row['source_id'],speaker_id=None,speaker_separation_established=False,
            reference=row['transcript'],question=row['question'],answers=row['answers'],answerable=row['answerable'],
            category_screen=categories,question_mentions_negation='नहीं' in row['question'],
            current_whisper_transcript=hypothesis,current_asr_provenance='matched-headroom B' if is_current else 'not_locally_exported',
            prior_feature_cache_expected=row['source_id'] in train_cache_expected or row['split']=='dev',
            prior_feature_cache_presence_verified=False,
            legacy_whisper_hint_transcript=old['hypothesis'],legacy_transcript_provenance=str(paths[0 if row['split']=='dev' else 1].relative_to(PROJECT)),
            legacy_is_current_baseline=False,current_literal_alias_present=literal_present,
            asr_attributable_view_from_model_QA=row['id'] in subset,
            orthographic_target_screen=candidate_span(row['transcript'],row['answers'],hypothesis) if row['answerable'] else None,
            predicted_token_timestamp_candidates=predicted.get(row['id']),
            actual_spoken_answer_timestamp=None,human_listened=False,
            independently_verified_spoken_error=False,
            localization_status='predicted_token_DTW_only_not_reference_or_listening_validation' if row['id'] in predicted else 'no_localization',
            listener_annotation_required=True)
        views.append(view)
    groups=defaultdict(list)
    for row in views:
        groups[row['source_id']].append(row)
    sources=[]
    for source,rows in sorted(groups.items()):
        assert len(rows)==2 and {r['condition'] for r in rows}=={'clean','noise_10db'}
        first=rows[0]
        sources.append(dict(source_id=source,research_role=first['research_role'],views=[r['id'] for r in rows],
            question=first['question'],answers=first['answers'],category_screen=first['category_screen'],
            target_value_set=sorted({norm(a) for a in first['answers']}),
            current_asr_views_available=sum(r['current_whisper_transcript'] is not None for r in rows),
            current_exact_alias_absent_candidate_views=[r['id'] for r in rows if r['current_literal_alias_present'] is False],
            headroom_view_ids=[r['id'] for r in rows if r['asr_attributable_view_from_model_QA']],
            speaker_id=None,spoken_verification='not_performed',manual_target_timestamp=None))
    for name,rows in [('views-v1.jsonl',views),('sources-v1.jsonl',sources)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summary=dict(phase='offline_inventory_complete_no_spoken_verification',source_recordings=112,views=224,
        current_asr_available_views=len(current),legacy_asr_available_views=len(legacy),
        verified_waveform_hashes=len(views),human_listened_sources=0,verified_answer_timestamps=0,
        independently_verified_spoken_errors=0,speakers_known=0,
        categorical_source_support={role:dict(Counter(c for r in sources if r['research_role']==role for c in r['category_screen']))
            for role in ['historical_training_pool','historical_development']},
        current_QA_headroom_sources=[r['source_id'] for r in sources if r['headroom_view_ids']],
        source_hashes={str(p.relative_to(PROJECT)):sha(p) for p in [mp,*paths,art/'headroom-result.json',art/'pilot-result.json']},
        limitations=['No speaker IDs supplied; source grouping does not establish speaker-disjoint evaluation.',
            'Legacy ASR is auxiliary candidate material and is never silently joined as the current baseline.',
            'Exact alias absence and edit distance do not establish loss of spoken meaning.',
            'Publisher reference is not listening verification. Predicted token timestamps do not localize omissions.',
            'No fresh100 data or predictions were loaded; no GPU or classifier used.'],
        next_gate='Review target pairs; obtain actual listening verification/localization before claiming independently verified spoken errors.')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_hashes','limitations']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
