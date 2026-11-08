"""Historical C-only length diagnostic; no training, ASR or adapter inference."""
import hashlib
import json
from pathlib import Path
import time

RUN_ID = 'gold-token-limit-dev24-v1'
LLM = ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306')


def run(payload, config, commit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from connector import pack_prompt
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.monotonic()
    assert hashlib.sha256(payload).hexdigest() == config['input_sha256']
    rows = json.loads(payload)
    assert len(rows) == 24 and len({r['source_id'] for r in rows}) == 12
    assert len({r['id'] for r in rows}) == 24
    assert all(r['arm'] == 'C_correct_text' and r['memory_tokens'] == 0 for r in rows)
    assert {r['condition'] for r in rows} == {'clean', 'noise_10db'}
    out = Path('/cache/runs') / RUN_ID
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    commit()
    tokenizer = AutoTokenizer.from_pretrained(LLM[0], revision=LLM[1])
    lm = AutoModelForCausalLM.from_pretrained(LLM[0], revision=LLM[1],
        torch_dtype=torch.float16, attn_implementation='sdpa', use_safetensors=True).cuda().eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    empty = torch.empty(0,1536,device='cuda')
    def generate(evidence, question, cap):
        # Function has no answer/reference argument. pack_prompt answer defaults None.
        prompt = pack_prompt(lm,tokenizer,evidence,question,empty)
        assert prompt.pop('labels') is None
        prompt.pop('position_ids')
        return lm.generate(**prompt,max_new_tokens=cap,do_sample=False,
            eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)[0]
    outputs = []
    with torch.inference_mode():
        generate('',rows[0]['question'],1)  # Identical text-only baseline warmup.
        load_seconds = time.monotonic()-started
        for index,row in enumerate(rows):
            if time.monotonic()-started > 540:
                raise TimeoutError('Internal token-limit deadline')
            assert row['transcript_used'] == row['publisher_evidence']
            torch.cuda.synchronize(); tick = time.monotonic()
            generated = generate(row['publisher_evidence'],row['question'],128)
            torch.cuda.synchronize()
            generation_s = time.monotonic()-tick
            token_ids = generated.tolist()
            answer = tokenizer.decode(generated,skip_special_tokens=True).strip()
            prefix = tokenizer.decode(generated[:48],skip_special_tokens=True).strip()
            capped_before = row['finish'] == 'length'
            match = prefix == row['answer'] if capped_before else answer == row['answer']
            record = {k:row[k] for k in ('id','source_id','condition','question','publisher_evidence',
                'answers','audio_sha256','transcript_used')}
            record.update(arm='C_correct_text_128',answer=answer,token_ids=token_ids,
                old_answer=row['answer'],old_generated_tokens=row['generated_tokens'],
                old_finish=row['finish'],old_cap_hit=capped_before,decoded_first48=prefix,
                baseline_prefix_match=match,memory_tokens=0,generated_tokens=len(token_ids),
                finish='length' if len(token_ids)==128 and token_ids[-1]!=tokenizer.eos_token_id else 'stop',
                generation_s=generation_s)
            outputs.append(record)
            with (out/'predictions.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            if index%4 == 0:
                commit()
                print(json.dumps({'phase':'token_limit','rows_done':len(outputs)}),flush=True)
    mismatches = [r['id'] for r in outputs if not r['baseline_prefix_match']]
    result = dict(status='token_limit_complete',config=config,rows=outputs,
        baseline_prefix_mismatches=mismatches,interpretation_allowed=not mismatches,
        trainable_parameters_updated=0,fresh_test_loaded=False,adapter_loaded=False,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        load_seconds=load_seconds,elapsed_s=time.monotonic()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        scope='C-only historical dev; max_new_tokens is the sole prospective intervention')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    commit()
    return result
