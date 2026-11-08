"""Tiny constructed gold/prediction test. No model outputs or accuracy claims."""
import json
from score import summarize, norm


def main():
    assert norm("१२, आम।")=="12 आम"
    rows=[({"id":"a","transcript":"चार लाल आम","answers":["चार"]},{"hypothesis":"दो लाल आम","mode":"auto","answer":"दो","seconds_audio":2,"asr_s":0.4,"llm_s":0.6,"total_s":1}),
          ({"id":"b","transcript":"पाँच","answers":["पाँच"]},{"hypothesis":"पाँच","mode":"auto","answer":"पाँच।","seconds_audio":2,"asr_s":0.4,"llm_s":0.6,"total_s":1})]
    result=summarize(rows,{"a":{"correct":False,"hallucinated":True},"b":{"correct":True,"hallucinated":False}})
    for name,want in {"word_edits":1,"reference_words":4,"wer":.25,"answer_em":.5,"pipeline_rtf":.5,"hallucinated_rate":.5}.items():
        assert result[name]==want, (name,result[name],want)
    print(json.dumps({"kind":"constructed scoring fixture","rows":2,"checks_passed":7,"metrics":result},indent=2))


if __name__=="__main__":
    main()

