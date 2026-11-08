"""Persist the assistant's arm-masked semantic review; no inference or training.

These are model judgments against publisher text, not human/listening validation.
Flags preserve ambiguous/length-truncated responses for strict sensitivity analysis.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / 'artifacts/headroom-review'
# case prefix, correctness, unsupported factual assertion, ambiguous, rationale
JUDGMENTS = [
('067570',False,True,False,'Names senders rather than tourists and sellers; incomplete and unsupported.'),
('09c422',False,True,False,'Names sound rather than the people whose voices are heard.'),
('1102b0',False,False,False,'Abstains despite an explicitly supplied year.'),
('24344b',True,False,False,'1966 se is semantically the stated year.'),
('281f38',False,False,True,'Ros does not unambiguously identify Rossby; preserve for sensitivity review.'),
('3b9c50',True,False,False,'Correctly identifies the toilet.'),
('3bf6bf',True,False,True,'Shengen is a recognizable spelling variant of Schengen; spelling-sensitive.'),
('4a515e',False,True,False,'Invents a count of 12 absent from the passage.'),
('510916',False,False,False,'In telescopes does not identify the requested remote regions.'),
('51e3b3',True,False,True,'Identifies Schengen and separate visa application, but the clause is cut off.'),
('530101',False,False,False,'Abstains although Earth is stated.'),
('5febf1',False,False,False,'Uninterpretable fragment; no identifiable answer or factual claim.'),
('60efdb',False,False,False,'Uninterpretable fragment does not identify the toilet.'),
('70e184',False,False,False,'Repeats small number without identifying Rossby.'),
('73bd5d',False,True,False,'Incorrectly names the group social group rather than pride.'),
('7b60b8',False,True,False,'Incorrectly names the group largest social group rather than pride.'),
('8f281b',True,False,True,'Recognizable Schengen spelling identifies the requested visa, with garbled suffix.'),
('935c22',False,False,False,'Abstains although pride is stated.'),
('9baeb3',False,False,False,'Abstains although across the river is stated.'),
('9ef987',False,False,False,'Restates the relation without naming the number; truncated nonanswer.'),
('9f93b9',False,False,True,'Ros alone is underidentified; not accepted as Rossby without guessing.'),
('a56dbd',False,False,False,'Abstains although Schengen is stated.'),
('ab91ee',False,False,False,'Repeats question context but never supplies a region.'),
('afd569',True,False,False,'Correct abstention: the source does not supply a species count.'),
('b43d80',True,False,True,'Light pollution is recognizable despite the final word being truncated.'),
('b70ba0',False,False,False,'No information is an incorrect abstention for an answerable question.'),
('c52989',False,True,False,'Adds unsupported numerical examples and fails to identify Rossby.'),
('c66185',False,False,False,'Abstains although tourists and sellers are stated.'),
('ca80b5',False,False,False,'Abstains although light pollution is stated.'),
('d860f2',True,False,False,'Correctly identifies Rossby number.'),
('e1604c',True,False,True,'Shuchale is recognizable as toilet in this context; spelling-sensitive.'),
('e3cdc8',False,True,False,'Claims the group is named social cats rather than pride.'),
('e44a47',False,False,False,'Repeats the question without giving the requested region.'),
('e5120a',False,False,False,'Abstains although remote regions is stated.'),
('e63cbe',False,False,False,'Abstains although toilet is stated.'),
('ed14b1',False,False,False,'Uninterpretable fragment; no recoverable answer or clear factual assertion.'),
('f03ab3',False,True,False,'Names senders rather than tourists and sellers.'),
('f2c54f',True,False,False,'Correct abstention: no elevation in metres is supplied.'),
('f4619c',False,True,False,'Confuses bus destinations with the station location across the river.'),
('f66585',True,False,True,'Contains the correct across-the-river location despite trailing truncation.'),
]

def main():
    cases = [json.loads(s) for s in (PACKET/'blinded-cases.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases) == len(JUDGMENTS) == 40
    rows = []
    for prefix, correct, hallucinated, ambiguous, reason in JUDGMENTS:
        matching = [c for c in cases if c['case'].startswith(prefix)]
        assert len(matching) == 1
        rows.append(dict(case=matching[0]['case'], correct=correct, hallucinated=hallucinated,
                         ambiguous=ambiguous, reason=reason, reviewer='Codex primary assistant (model)',
                         human_review=False, audio_listened=False))
    assert len({r['case'] for r in rows}) == 40
    for name, labels in [('model-labels.jsonl', rows), ('model-labels-strict.jsonl',
            [dict(r, correct=False, reason=r['reason']+' Strict sensitivity rejects flagged variants.')
             if r['ambiguous'] and r['correct'] else r for r in rows])]:
        (PACKET/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in labels), encoding='utf-8')
    provenance = dict(review_type='Single assistant, arm-masked model review against publisher text',
        independently_reviewed=False, human_review=False, audio_listened=False,
        case_count=40, response_count=96,
        rubric={'correct':'Identifies the requested entity/relation or abstains when evidence is absent; recognizable spelling variants accepted; ambiguous cases flagged.',
                'hallucinated':'Makes an identifiable factual assertion unsupported by publisher evidence. Abstentions, uninterpretable fragments and mere failure to answer are not automatically hallucinations.'},
        limitations=['Primary assistant implemented the experiment; this is not independent review.',
            'Arm labels and clean/noisy labels omitted from review packet; assistant knew the overall design and aggregate EM.',
            'Variant and truncated-answer acceptance is subjective; strict sensitivity rejects every flagged accepted case.',
            'Reference text is not human listening validation. Question-only answers are judged against source evidence, so unsupported is not necessarily false in the world.'])
    (PACKET/'review-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'labels':len(rows),'ambiguous':sum(r['ambiguous'] for r in rows)}))

if __name__ == '__main__':
    main()
