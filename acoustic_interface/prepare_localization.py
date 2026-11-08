"""Validate explicit listening boundaries and prepare a draft, never launch or reserve."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
RATE=16000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('annotation',type=Path)
    args=parser.parse_args()
    art=HERE/'artifacts'
    assert not (art/'localization-job.json').exists() and not (art/'localization-result.json').exists()
    raw=args.annotation.read_bytes()
    record=json.loads(raw.decode('utf-8-sig'))
    chat=record['schema']=='vaani-localization-boundaries-chat-v1'
    assert record['schema'] in {'vaani-localization-boundaries-v1','vaani-localization-boundaries-chat-v1'} and record['source_id']=='fleurs-hi-1806'
    assert record['target_phrase']=='प्रकाश प्रदूषण'
    assert record['placebo_checked_unrelated'] is True
    capture_files=[]
    if chat:
        assert record['complete_phrase_checked_both_versions'] is None
        assert record['phrase_bounds_reported_by_user'] is True and record['prior_target_audible_both_versions'] is True
        capture=record['capture_provenance']
        assert capture['method']=='user_screenshots_and_chat'
        assert float(capture['user_reply'])==record['boundary_uncertainty_s']
        for evidence in capture['screenshots']:
            ep=HERE/evidence['path']
            assert digest(ep)==evidence['sha256']
            capture_files.append(evidence['path'])
        assert len(capture_files)==2
        lp=HERE/capture['prior_listening_file']
        assert digest(lp)==capture['prior_listening_sha256']
        prior=json.loads(lp.read_text(encoding='utf-8'))['annotations']
        relevant=[r for r in prior if r['source_id']=='fleurs-hi-1806']
        assert len(relevant)==2 and all(r['effective_target_spoken_verified'] for r in relevant)
        capture_files.append(capture['prior_listening_file'])
    else:
        assert record['complete_phrase_checked_both_versions'] is True
    s,e=record['phrase_interval_s']
    u=record['boundary_uncertainty_s']
    assert all(type(v) in (int,float) and math.isfinite(v) for v in [s,e,u])
    assert 0<=s<e<=13.68 and 0<u<=1
    target=record['target_mask_interval_s']
    placebo=record['placebo_mask_interval_s']
    assert all(type(v) in (int,float) and math.isfinite(v) for v in [*target,*placebo])
    assert len(target)==len(placebo)==2
    assert abs(target[0]-max(0,s-u))<1e-8 and abs(target[1]-min(13.68,e+u))<1e-8
    assert 0<=placebo[0]<placebo[1]<=13.68
    assert abs((target[1]-target[0])-(placebo[1]-placebo[0]))<1e-8
    assert max(target[0],placebo[0])>=min(target[1],placebo[1])
    manifest={r['id']:r for r in map(json.loads,(HERE.parent/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    clips=record['clips']
    assert len(clips)==2 and {r['id'] for r in clips}=={'fleurs-hi-1806-clean','fleurs-hi-1806-noise_10db'}
    for clip in clips:
        row=manifest[clip['id']]
        assert clip['audio_sha256']==row['audio_sha256'] and clip['duration_s']==row['duration_s']
        assert clip['condition']==row['condition']
        assert digest(HERE.parent/'data'/row['audio'])==row['audio_sha256']
    lo,hi=round(target[0]*RATE),round(target[1]*RATE)
    count=hi-lo
    plo=round(placebo[0]*RATE)
    phi=plo+count
    assert count>160 and 0<=lo<hi<=218880 and 0<=plo<phi<=218880
    assert max(lo,plo)>=min(hi,phi)
    directory=art/'localization-boundary-imports'
    directory.mkdir(exist_ok=True)
    preserved=directory/(hashlib.sha256(raw).hexdigest()+'.json')
    if preserved.exists():
        assert preserved.read_bytes()==raw
    else:
        preserved.write_bytes(raw)
    status='human_reported_bounds' if chat else 'human_validated'
    validated=dict(schema='vaani-localization-boundaries-validated-v1',validation_status=status,human_review=True,
        full_phrase_extent_verified=not chat,
        source_id='fleurs-hi-1806',phrase_interval_s=[s,e],boundary_uncertainty_s=u,
        target_mask_interval_s=[lo/RATE,hi/RATE],placebo_mask_interval_s=[plo/RATE,phi/RATE],
        target_sample_bounds=[lo,hi],placebo_sample_bounds=[plo,phi],equal_sample_count=count,
        raw_annotation_file=str(preserved.relative_to(HERE)),raw_annotation_sha256=hashlib.sha256(raw).hexdigest(),
        raw_clips=clips,method=record['method'],notes=record.get('notes',''),
        rounding='Target bounds round to nearest16000Hz sample; placebo start rounds, end=start+targetN',
        precision='Approximate single-user listening with stated margin, not exact forced alignment',validated_unix=time.time())
    bp=art/'localization-boundaries-validated.json'
    bp.write_text(json.dumps(validated,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    original=json.loads((art/'omission-freeze-v1.json').read_text(encoding='utf-8'))['config']
    source_names=['localization_worker.py','localization_runner.py','prepare_localization.py',
        'artifacts/localization-boundaries-validated.json','artifacts/omission-freeze-v1.json',validated['raw_annotation_file'],*capture_files]
    config=dict(run_id='localized-omission-sensitivity-v1',seed=17,model=original['model'],model_revision=original['model_revision'],
        pair=original['pair'],inputs=original['inputs'],target_mask_interval_s=validated['target_mask_interval_s'],
        placebo_mask_interval_s=validated['placebo_mask_interval_s'],fade_s=.005,
        boundary_provenance=dict(file='artifacts/localization-boundaries-validated.json',sha256=digest(bp),validation_status=status),
        source_sha256={n:digest(HERE/n) for n in source_names},
        old_result_sha256='8877c11b117f3db234fd49e505c39877fdd4be65d4b7340b3f3050acebdf12f7',
        resource_limits=original['resource_limits'],reservation_usd=1,worst_case_cost_usd=.63168344,
        independent_sources=1,training_updates=0,fresh_test_loaded=False)
    draft=dict(state='prospective_review_pending',created_unix=time.time(),config=config,
        estimand='S(placebo-masked)-S(target-masked), equivalently targetdrop-placebodrop, per acoustic condition',
        primary_measure='Full-sequence lexical+EOS summed preference; token-normalized measures separate',
        auxiliary_measure='Firstdivergent token odds from a SINGLE shared-prefix forward distribution',
        reporting='Signed descriptive contrasts, no population significance or trainingGO from one source',
        boundary_scope='User-reported uncertain window; fullphraseextent not independently established; placebo equalduration and listener-checked unrelated' if chat else 'Padded target interval includes annotated uncertainty; placebo equalduration and listener-checked unrelated',
        limitations=['Masking perturbations can alter alignment and contextual encoder states.',
            'Unequal hypothesis lengths remain; source1 with2conditions not2independentcases.',
            'Candidatecoverage andlocalizeddecoder sensitivity do not prove four-vector readout.'])
    (art/'localization-config-draft.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='draft_prepared_no_GPU',phrase=[s,e],uncertainty=u,
        target=validated['target_mask_interval_s'],placebo=validated['placebo_mask_interval_s'],mask_samples=count),ensure_ascii=False))


if __name__=='__main__':
    main()
