"""Prospectively declared single-source synthetic-deletion sensitivity study."""
import hashlib
import io
import json
from pathlib import Path
import time
import wave

REVISION='973afd24965f72e36ca33b3055d56a652f456b4d'


def token_regions(plus,minus):
    common=0
    while common<min(len(plus),len(minus)) and plus[common]==minus[common]:
        common+=1
    suffix=0
    while suffix<min(len(plus)-common,len(minus)-common) and plus[-suffix-1]==minus[-suffix-1]:
        suffix+=1
    return dict(common_prefix_lexical_tokens=common,common_suffix_lexical_tokens=suffix,
        plus_differing_span=[common,len(plus)-suffix],minus_differing_span=[common,len(minus)-suffix])


def run(inputs,config,commit):
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    from transformers.modeling_outputs import BaseModelOutput
    torch.set_num_threads(2)
    torch.manual_seed(17)
    assert torch.cuda.is_available() and len(inputs)==2
    started=time.monotonic()
    out=Path('/cache/runs')/config['run_id']
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    processor=WhisperProcessor.from_pretrained('openai/whisper-small',revision=REVISION)
    tokenizer=processor.tokenizer
    tokenizer.set_prefix_tokens(language='hi',task='transcribe',predict_timestamps=False)
    prefix=list(tokenizer.prefix_tokens)
    candidates={name:tokenizer.encode(config['pair'][name],add_special_tokens=False) for name in ('plus','minus')}
    regions=token_regions(candidates['plus'],candidates['minus'])
    common=regions['common_prefix_lexical_tokens']
    assert common<min(map(len,candidates.values()))
    tokenization=dict(prefix_tokens=prefix,eos_token_id=tokenizer.eos_token_id,
        candidates=candidates,regions=regions,scoring='Exclude forced prefix; lexical and EOS separate; no truncation',
        first_difference='next lexical token immediately after exact common token prefix',
        differing_span='Token LCP/LCS-derived region, not a timestamp or exact reference-character mask')
    # Resolve and persist pinned tokenization BEFORE acoustic encoding or scoring.
    token_bytes=json.dumps(tokenization,ensure_ascii=False,indent=2).encode()
    (out/'tokenization.json').write_bytes(token_bytes)
    commit()
    asr=WhisperForConditionalGeneration.from_pretrained('openai/whisper-small',revision=REVISION,
        torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    for parameter in asr.parameters():
        parameter.requires_grad_(False)
    assert prefix[0]==asr.config.decoder_start_token_id
    def waveform(raw,expected):
        assert hashlib.sha256(raw).hexdigest()==expected
        with wave.open(io.BytesIO(raw),'rb') as f:
            assert f.getnchannels()==1 and f.getsampwidth()==2 and f.getframerate()==16000
            samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
        assert 0<len(samples)<=480000 and np.isfinite(samples).all()
        return samples
    def encode(samples):
        batch=processor(samples,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        assert batch.input_features.shape[-1]==3000
        return asr.model.encoder(batch.input_features.to('cuda',dtype=torch.float16),return_dict=True).last_hidden_state
    def score(hidden,name):
        lexical=candidates[name]
        sequence=prefix+lexical+[tokenizer.eos_token_id]
        ids=torch.tensor([sequence],device='cuda')
        logits=asr(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            decoder_input_ids=ids[:,:-1],use_cache=False,return_dict=True).logits.float()
        logp=logits.log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)[0]
        lex=logp[len(prefix)-1:len(prefix)-1+len(lexical)]
        eos=logp[-1]
        assert len(lex)==len(lexical) and torch.isfinite(logp).all()
        lo,hi=regions[name+'_differing_span']
        diff=lex[lo:hi]
        return dict(candidate=name,lexical_token_ids=lexical,lexical_token_logp=lex.tolist(),
            lexical_token_count=len(lexical),lexical_sum_logp=float(lex.sum()),lexical_mean_logp=float(lex.mean()),
            eos_logp=float(eos),sequence_sum_logp=float(lex.sum()+eos),
            sequence_mean_logp=float((lex.sum()+eos)/(len(lexical)+1)),
            differing_span_sum_logp=float(diff.sum()),
            differing_span_mean_logp=float(diff.mean()) if len(diff) else None,
            differing_span_token_count=len(diff),first_divergent_token_id=lexical[common],
            first_divergent_token_logp=float(lex[common]))
    rows=[]
    with torch.inference_mode():
        for item in inputs:
            actual=waveform(item['audio_bytes'],item['audio_sha256'])
            donor=waveform(item['donor_bytes'],item['donor_audio_sha256'])
            for control,samples in [('actual',actual),('silence',np.zeros_like(actual)),('mismatch',donor)]:
                if time.monotonic()-started>540:
                    raise TimeoutError('Internal diagnostic deadline')
                hidden=encode(samples)
                scores={name:score(hidden,name) for name in ('plus','minus')}
                metrics=['lexical_sum_logp','lexical_mean_logp','sequence_sum_logp','sequence_mean_logp',
                    'differing_span_sum_logp','first_divergent_token_logp']
                row=dict(id=item['id'],source_id=item['source_id'],condition=item['condition'],control=control,
                    input_duration_s=len(samples)/16000,actual_audio_sha256=item['audio_sha256'],
                    donor_id=item['donor_id'] if control=='mismatch' else None,
                    donor_audio_sha256=item['donor_audio_sha256'] if control=='mismatch' else None,
                    scores=scores,preference={m:scores['plus'][m]-scores['minus'][m] for m in metrics})
                rows.append(row)
                with (out/'scores.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                commit()
                print(json.dumps({'phase':'scoring','id':item['id'],'control':control}),flush=True)
    result=dict(status='omission_sensitivity_complete',config=config,tokenization=tokenization,
        tokenization_sha256=hashlib.sha256(token_bytes).hexdigest(),rows=rows,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        elapsed_s=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        training_updates=0,fresh_test_loaded=False,independent_sources=1,
        interpretation='Frozen-decoder synthetic-deletion sensitivity; not encoder recovery, local feature proof or QA gain')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    commit()
    return result
