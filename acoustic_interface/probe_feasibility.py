"""Text-only support audit for a future probe; does not load acoustic/test features."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent


def numeric(answers):
    return any(re.search(r'\d',answer) is not None for answer in answers)


def main():
    manifest=[json.loads(s) for s in (ROOT.parent/'data/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    by={}
    for row in manifest:
        by.setdefault(row['source_id'],row)
    analysis=json.loads((ROOT/'artifacts/headroom-analysis-independent.json').read_text(encoding='utf-8'))
    selected=set(analysis['semantic']['asr_attributable']['subset_ids'])
    sources=sorted({r['source_id'] for r in manifest if r['id'] in selected})
    summary=dict(scope='Offline textual feasibility audit, not an acoustic probe',independent_sources=len(by),
        source_groups={split:dict(sources=sum(r['split']==split for r in by.values()),
            numeric_reference_answer_sources=sum(r['split']==split and numeric(r['answers']) for r in by.values()),
            transcript_contains_digit_sources=sum(r['split']==split and bool(re.search(r'\d',r['transcript'])) for r in by.values()),
            transcript_contains_nahin_sources=sum(r['split']==split and 'नहीं' in r['transcript'] for r in by.values()))
            for split in sorted({r['split'] for r in by.values()})},
        headroom_sources=[dict(source_id=sid,question=by[sid]['question'],answers=by[sid]['answers'],
            numeric_answer=numeric(by[sid]['answers'])) for sid in sources],
        limitations=['Presence of digits or nahin is a textual screening heuristic, not validated target labeling.',
            'Historical100test rows are historicaltraining only in thisnewround; fresh100notread.',
            'Probe target classes need train/dev support; a unique1966 label cannot establish cross-source generalization.'])
    assert numeric(['1966']) and numeric(['१९६६']) and not numeric(['रॉस्बी संख्या'])
    (ROOT/'artifacts/probe-feasibility-text-audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'headroom_source_count':len(sources),'headroom_sources_with_numeric_answer':sum(r['numeric_answer'] for r in summary['headroom_sources']),
                      'historical_support':summary['source_groups']},ensure_ascii=False))


if __name__=='__main__':
    main()
