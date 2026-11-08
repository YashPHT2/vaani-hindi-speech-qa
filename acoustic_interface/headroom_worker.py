"""Inference-only matched headroom diagnostic on historical development views."""
import hashlib
import io
import json
from pathlib import Path
import time

RUN_ID = 'matched-headroom-dev24-v1'
PILOT_RUN = 'paired-pilot-st-seed17-v1-r1'
CHECKPOINT_SHA = '8474698e8123d52e4ea96e310691062e49ce76f62d9e924d1976102ffbad2345'
LLM = ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306')


def run(manifest_bytes, config, commit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from connector import AcousticMemory, pack_prompt
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    started = time.monotonic()
    out = Path('/cache/runs') / RUN_ID
    out.mkdir(parents=True, exist_ok=False)
    assert hashlib.sha256(manifest_bytes).hexdigest() == config['historical_manifest_sha256']
    rows = [json.loads(s) for s in manifest_bytes.decode('utf-8').splitlines() if s.strip()]
    rows = [r for r in rows if r['split'] == 'dev']
    assert len(rows) == 24 and len({r['source_id'] for r in rows}) == 12
    assert len({r['id'] for r in rows}) == 24
    assert {r['condition'] for r in rows} == {'clean', 'noise_10db'}
    checkpoint = Path('/cache/runs') / PILOT_RUN / 'full-best.pt'
    checkpoint_bytes = checkpoint.read_bytes()
    assert hashlib.sha256(checkpoint_bytes).hexdigest() == CHECKPOINT_SHA
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    commit()
    tokenizer = AutoTokenizer.from_pretrained(LLM[0], revision=LLM[1])
    lm = AutoModelForCausalLM.from_pretrained(LLM[0], revision=LLM[1],
        torch_dtype=torch.float16, attn_implementation='sdpa', use_safetensors=True).cuda().eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    adapter = AcousticMemory().cuda().eval()
    state = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    adapter.load_state_dict(state['state_dict'], strict=True)
    assert sum(p.numel() for p in adapter.parameters()) == 305281
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    results = []
    arms = ('A_question_only', 'B_asr_text', 'C_correct_text', 'D_existing_adapter')
    def forward(row, arm, features):
        question = row['question']
        evidence = '' if arm == 'A_question_only' else row['transcript'] if arm == 'C_correct_text' else features['transcript']
        memory = torch.empty(0, 1536, device='cuda')
        selected = []
        if arm == 'D_existing_adapter':
            keys = ('audio', 'frame_mask', 'span_mask', 'uncertainty', 'uncertainty_available')
            inputs = {key: features[key].cuda() for key in keys}
            ids = tokenizer(question, return_tensors='pt').input_ids.cuda()
            inputs.update(question=lm.get_input_embeddings()(ids).float(),
                          question_mask=torch.ones_like(ids, dtype=torch.bool))
            memories, chosen, _ = adapter(**inputs, condition='full')
            memory, selected = memories[0], chosen[0].tolist()
        return evidence, memory, selected
    with torch.inference_mode():
        # Unscored, text-only warmup; this updates no parameters or cache evidence.
        warm = pack_prompt(lm, tokenizer, '', rows[0]['question'], torch.empty(0, 1536, device='cuda'))
        warm.pop('labels'); warm.pop('position_ids')
        lm.generate(**warm, max_new_tokens=1, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        for row_index, row in enumerate(rows):
            if time.monotonic() - started > 540:
                raise TimeoutError('Internal headroom diagnostic deadline')
            cache = Path('/cache/features-v1') / (row['id'] + '.pt')
            # Trusted own saved tensors only. No ASR, training, data download or new selection.
            features = torch.load(cache, map_location='cpu', weights_only=False)
            for arm in arms:
                evidence, memory, selected = forward(row, arm, features)
                target = pack_prompt(lm, tokenizer, evidence, row['question'], memory, row['answers'][0])
                nll = float(lm(**target, use_cache=False).loss)
                assert torch.isfinite(torch.tensor(nll))
                prompt = pack_prompt(lm, tokenizer, evidence, row['question'], memory)
                assert prompt.pop('labels') is None
                prompt.pop('position_ids')
                torch.cuda.synchronize(); generation_start = time.monotonic()
                generated = lm.generate(**prompt, max_new_tokens=48, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)[0]
                torch.cuda.synchronize()
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                record = {'id': row['id'], 'source_id': row['source_id'], 'condition': row['condition'],
                    'arm': arm, 'question': row['question'], 'publisher_evidence': row['transcript'],
                    'answers': row['answers'], 'answer': answer, 'nll': nll,
                    'transcript_used': evidence, 'audio_sha256': row['audio_sha256'],
                    'selected_spans': selected, 'memory_tokens': len(memory),
                    'alignment_available': bool(features['frame_mask'].any()),
                    'generated_tokens': len(generated),
                    'finish': 'length' if len(generated) == 48 and int(generated[-1]) != tokenizer.eos_token_id else 'stop',
                    'generation_s': time.monotonic() - generation_start}
                results.append(record)
                with (out / 'predictions.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            if row_index % 4 == 0:
                commit()
                print(json.dumps({'phase': 'headroom', 'views_done': row_index + 1, 'rows_done': len(results)}), flush=True)
    assert len(results) == 96
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA
    result = {'status': 'matched_headroom_complete', 'config': config, 'rows': results,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'trainable_parameters_updated': 0, 'adapter_parameter_count': 305281,
        'source_pilot_checkpoint_sha256': CHECKPOINT_SHA,
        'load_seconds': load_seconds, 'elapsed_s': time.monotonic() - started,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'scope': 'Historical development only, inference-only, fixed prompt and decoding; no fresh test or architecture change'}
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    commit()
    return result
