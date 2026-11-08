"""Prospective scalar-only calibration of detached existing acoustic memories."""
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import unicodedata
import wave

RUN_ID="scale-calibration-v1"
CHECKPOINT_SHA="8474698e8123d52e4ea96e310691062e49ce76f62d9e924d1976102ffbad2345"
ASR=("openai/whisper-small","973afd24965f72e36ca33b3055d56a652f456b4d")
LLM=("Qwen/Qwen2.5-1.5B-Instruct","989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
GRID=(1/32,1/8,.5,1.,2.)
KEYS=("audio","frame_mask","span_mask","uncertainty","uncertainty_available")


def sha(payload):return hashlib.sha256(payload).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
def normalized_answer(value):return " ".join(unicodedata.normalize("NFC",value).lower().split())


def preflight_gate(rows):
    """Declared four-source, two-view sensitivity gate; no outcome selection."""
    grouped={}
    for row in rows:grouped.setdefault(row["source_id"],{}).setdefault(row["id"],[]).append(row)
    assert len(grouped)==4 and len(rows)==40
    source_metrics=[]
    for source,views in grouped.items():
        assert len(views)==2
        shifts=[];variation=False
        for records in views.values():
            assert len(records)==5 and {r["alpha"] for r in records}==set(GRID)
            baseline=next(r for r in records if r["alpha"]==1.)
            shifts.append(max(abs(r["first_token_logprob"]-baseline["first_token_logprob"]) for r in records))
            variation=variation or len({normalized_answer(r["answer"]) for r in records})>1
        source_metrics.append({"source_id":source,"mean_view_max_abs_first_token_logp_shift":sum(shifts)/2,
            "normalized_answer_variation":variation})
    logp_sources=sum(r["mean_view_max_abs_first_token_logp_shift"]>=.1 for r in source_metrics)
    answer_sources=sum(r["normalized_answer_variation"] for r in source_metrics)
    return {"passed":logp_sources>=2 or answer_sources>=2,"logp_threshold_sources":logp_sources,
        "answer_variation_sources":answer_sources,"source_stats":source_metrics,
        "answer_normalization":"Unicode NFC, lowercase, collapse whitespace; no spelling or semantic normalization"}


def run(inputs,historical_bytes,config,commit):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM,WhisperProcessor,WhisperForConditionalGeneration
    from connector import AcousticMemory,pack_prompt
    from readout_candidates import grid_windows,pack_windows,cached_windows,replace_geometry_audio
    assert torch.cuda.is_available() and config["run_id"]==RUN_ID
    assert sha(historical_bytes)==config["historical_manifest_sha256"]
    assert config["alphas"]==list(GRID) and config["seed"]==17
    assert config["fit"]=={"parameter":"log_alpha only","initial":0,"optimizer":"Adam","lr":.1,"batch":8,"epochs":4,"updates":32,"alpha_bounds":[1/32,2],"selection":"final update only"}
    assert config["preflight_logprob_threshold"]==.1 and config["preflight_min_sources"]==2
    rows=[json.loads(line) for line in historical_bytes.decode("utf-8").splitlines()]
    assert len(rows)==224 and len({r["id"] for r in rows})==224
    by_id={r["id"]:r for r in rows}
    train_sources=set(config["train_source_ids"]);primary_sources=set(config["primary_source_ids"])
    illustrative=config["illustrative_source_id"]
    assert len(train_sources)==32 and len(primary_sources)==11 and illustrative=="fleurs-hi-1806"
    assert not train_sources&primary_sources and illustrative not in train_sources|primary_sources
    train=[r for r in rows if r["source_id"] in train_sources]
    primary=[r for r in rows if r["source_id"] in primary_sources]
    case=[r for r in rows if r["source_id"]==illustrative]
    assert len(train)==64 and len(primary)==22 and len(case)==2
    assert all(r["split"]=="test" for r in train) and all(r["split"]=="dev" for r in primary+case)
    donors=config["donor_by_view"]
    assert set(donors)=={r["id"] for r in primary+case}
    for row in primary+case:
        donor=by_id[donors[row["id"]]]
        assert donor["source_id"] in train_sources and donor["source_id"]!=row["source_id"]
        assert donor["condition"]==row["condition"] and donor["transcript"]!=row["transcript"]
        eligible=[r for r in train if r["condition"]==row["condition"] and r["source_id"]!=row["source_id"] and r["transcript"]!=row["transcript"]]
        expected=min(eligible,key=lambda d:(abs(d["duration_s"]-row["duration_s"]),int(d["source_id"].rsplit("-",1)[-1])))
        assert donor["id"]==expected["id"]
    assert config["donor_alignment"]=="linear_interpolation_of_complete_valid_encoder_sequence_align_corners_false"
    waves={item["id"]:item for item in inputs}
    assert len(waves)==len(inputs) and set(waves)<={r["id"] for r in train+primary+case}
    assert {r["id"] for r in train+primary+case}<=set(waves)
    for identity,item in waves.items():assert sha(item["audio_bytes"])==by_id[identity]["audio_sha256"]==item["audio_sha256"]
    baseline_rows=config["baseline_rows"]
    assert sha(canonical(baseline_rows))==config["baseline_rows_semantic_sha256"]
    baseline={(r["id"],r["arm"]):r for r in baseline_rows}
    torch.set_num_threads(2);torch.manual_seed(17)
    started=time.monotonic();out=Path("/cache/runs")/RUN_ID;out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        payload=json.dumps(value,ensure_ascii=False,indent=2).encode("utf-8")
        (out/name).write_bytes(payload);return sha(payload)
    write("config.json",config)
    def check():
        if time.monotonic()-started>1740:raise TimeoutError("Scalar work deadline; no automatic retry")
    def event(record):
        record["elapsed_s"]=time.monotonic()-started
        with (out/"events.jsonl").open("ab") as stream:stream.write(canonical(record)+b"\n")
        print(json.dumps(record,ensure_ascii=False),flush=True)
    checkpoint=Path("/cache/runs/paired-pilot-st-seed17-v1-r1/full-best.pt")
    checkpoint_bytes=checkpoint.read_bytes();assert sha(checkpoint_bytes)==config["pilot_checkpoint_sha256"]==CHECKPOINT_SHA
    tokenizer=AutoTokenizer.from_pretrained(LLM[0],revision=LLM[1])
    lm=AutoModelForCausalLM.from_pretrained(LLM[0],revision=LLM[1],torch_dtype=torch.float16,attn_implementation="sdpa",use_safetensors=True).cuda().eval()
    adapter=AcousticMemory().cuda().eval()
    adapter.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes),map_location="cpu",weights_only=True)["state_dict"],strict=True)
    processor=WhisperProcessor.from_pretrained(ASR[0],revision=ASR[1])
    asr=WhisperForConditionalGeneration.from_pretrained(ASR[0],revision=ASR[1],torch_dtype=torch.float16,attn_implementation="eager",use_safetensors=True).cuda().eval()
    for model in (lm,adapter,asr):
        for parameter in model.parameters():parameter.requires_grad_(False)
    adapter_initial={n:v.detach().cpu().clone() for n,v in adapter.state_dict().items()}
    versions={label:[p._version for p in model.parameters()] for label,model in (("lm",lm),("adapter",adapter),("asr",asr))}
    def proof():
        assert sha(checkpoint.read_bytes())==CHECKPOINT_SHA
        assert all(torch.equal(v.detach().cpu(),adapter_initial[n]) for n,v in adapter.state_dict().items())
        for label,model in (("lm",lm),("adapter",adapter),("asr",asr)):
            assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert [p._version for p in model.parameters()]==versions[label]
        for (identity,kind),memory in memories.items():
            assert not memory.requires_grad and memory.grad_fn is None
            assert sha(memory.numpy().tobytes())==metadata[identity][kind+"_memory_sha256"]
        return {"detached_memory_bytes_unchanged":True,"checkpoint_before_sha256":sha(checkpoint_bytes),"checkpoint_after_sha256":sha(checkpoint.read_bytes()),
            "adapter_state_unchanged":True,"lm_asr_adapter_parameter_versions_unchanged":True,"backbone_gradients_none":True}
    def decode(identity):
        item=waves[identity]
        with wave.open(io.BytesIO(item["audio_bytes"]),"rb") as source:
            assert source.getnchannels()==1 and source.getsampwidth()==2 and source.getframerate()==16000
            count=source.getnframes();samples=np.frombuffer(source.readframes(count),dtype="<i2").astype(np.float32)/32768
        assert count==round(by_id[identity]["duration_s"]*16000) and 0<count<=480000 and np.isfinite(samples).all()
        return samples
    def encode(samples):
        features=processor(samples,sampling_rate=16000,return_tensors="pt").input_features.cuda().half()
        return asr.model.encoder(features,return_dict=True).last_hidden_state[0].float()
    memories={};metadata={};missing=[]
    # No alpha enters candidate construction, routing, gates, question or scaffold.
    with torch.no_grad():
        for row in train+primary+case:
            check();path=Path("/cache/features-v1")/(row["id"]+".pt")
            if not path.exists():
                missing.append(row["id"]);continue
            blob=path.read_bytes();assert sha(blob)==config["cache_sha256_by_view"][row["id"]]
            cached=torch.load(io.BytesIO(blob),map_location="cpu",weights_only=False)
            valid_frames=math.ceil(row["duration_s"]/.02)
            valid=bool((cached["span_mask"]&cached["frame_mask"].any(-1)).any())
            if valid:payload={k:cached[k].cuda() for k in KEYS};windows,_=cached_windows(cached,valid_frames)
            else:payload=pack_windows(encode(decode(row["id"])),valid_frames,grid_windows(valid_frames));windows=payload["candidate_windows"]
            ids=tokenizer(row["question"],return_tensors="pt").input_ids.cuda()
            question=lm.get_input_embeddings()(ids).float();qmask=torch.ones_like(ids,dtype=torch.bool)
            vector,chosen,gates=adapter(**{k:payload[k] for k in KEYS},question=question,question_mask=qmask,condition="full")
            memory=vector[0].detach().cpu().clone();assert torch.isfinite(memory).all()
            if not len(memory):
                missing.append(row["id"]);continue
            assert 1<=len(memory)<=4
            memories[(row["id"],"real")]=memory
            metadata[row["id"]]={"transcript":cached["transcript"],"feature_cache_sha256":sha(blob),"fallback_used":not valid,
                "candidate_windows_frames":windows,"selected_indices":chosen[0].tolist(),"selected_windows_frames":[windows[i] for i in chosen[0].tolist()],
                "gates":gates[0].tolist(),"uncertainty_available":payload["uncertainty_available"].cpu().tolist(),
                "question_embedding_norm_mean":float(question.norm(dim=-1).mean()),"real_memory_sha256":sha(memory.numpy().tobytes()),
                "real_memory_norms":memory.norm(dim=-1).tolist()}
            if row["id"] in donors:
                donor_id=donors[row["id"]];samples=decode(donor_id)
                donor_valid_frames=math.ceil(by_id[donor_id]["duration_s"]/.02)
                donor_hidden=encode(samples)[:donor_valid_frames]
                if donor_valid_frames!=valid_frames:
                    donor_hidden=F.interpolate(donor_hidden.T[None],size=valid_frames,mode="linear",align_corners=False)[0].T
                assert len(donor_hidden)==valid_frames
                if valid:donor_payload=replace_geometry_audio(cached,donor_hidden.cpu(),valid_frames)
                else:donor_payload=pack_windows(donor_hidden,valid_frames,windows)
                wrong,selected,amplitude=adapter(**{k:donor_payload[k].cuda() for k in KEYS},question=question,question_mask=qmask,condition="full")
                wrong_memory=wrong[0].detach().cpu().clone();assert len(wrong_memory)==4 and torch.isfinite(wrong_memory).all()
                memories[(row["id"],"wrong")]=wrong_memory
                metadata[row["id"]].update(donor_id=donor_id,donor_duration_ratio=by_id[donor_id]["duration_s"]/row["duration_s"],
                    donor_original_valid_frames=donor_valid_frames,recipient_valid_frames=valid_frames,
                    donor_alignment=config["donor_alignment"],donor_resampling_ratio=valid_frames/donor_valid_frames,
                    wrong_memory_sha256=sha(wrong_memory.numpy().tobytes()),wrong_memory_norms=wrong_memory.norm(dim=-1).tolist(),
                    wrong_selected_indices=selected[0].tolist(),wrong_selected_windows_frames=[windows[i] for i in selected[0].tolist()],wrong_gates=amplitude[0].tolist())
            event({"phase":"detached_memory","id":row["id"],"fallback":not valid})
    write("memory-metadata.json",metadata);commit()
    if missing:
        result={"status":"scale_preflight_stopped","config":config,"preflight_rows":[],
            "preflight_gate":{"passed":False,"incomplete":True,"missing_view_ids":missing},"fit_trace":[],"eval_rows":[],
            "memory_metadata":metadata,"weight_invariance":proof(),"scalar_updates":0,"backbone_training_updates":0,"fresh_test_loaded":False,"elapsed_s":time.monotonic()-started}
        write("result.json",result);commit();return result
    def evaluate(row,alpha,kind,phase,arm):
        check()
        base=memories[(row["id"],kind)].cuda() if kind!="text" else torch.empty(0,1536,device="cuda")
        with torch.no_grad():
            memory=base*alpha
            prompt=pack_prompt(lm,tokenizer,metadata[row["id"]]["transcript"],row["question"],memory,answer=None)
            assert prompt.pop("labels") is None
            # This distribution sees precisely the answer-free generation prefix.
            target_token=tokenizer(row["answers"][0]+"<|im_end|>",add_special_tokens=False).input_ids[0]
            logits=lm(**prompt,use_cache=False).logits[0,-1].float()
            first_logit=float(logits[target_token]);first_logp=float(logits.log_softmax(-1)[target_token])
            prompt_hash=sha(prompt["inputs_embeds"].detach().cpu().contiguous().numpy().tobytes())
            prefix_tokens=int(prompt["inputs_embeds"].shape[1])
            if phase=="preflight":
                logit_path=out/(row["id"]+"-alpha"+format(alpha,".5g")+"-prefix-logits.pt")
                torch.save({"first_position_logits":logits.detach().cpu(),"first_gold_token_id":target_token,
                    "prompt_embedding_sha256":prompt_hash,"prompt_tokens":prefix_tokens,
                    "question":row["question"],"transcript":metadata[row["id"]]["transcript"],
                    "detached_memory_sha256":metadata[row["id"]]["real_memory_sha256"],"alpha":alpha},logit_path)
            prompt.pop("position_ids")
            torch.cuda.synchronize();stamp=time.monotonic()
            generated=lm.generate(**prompt,max_new_tokens=48,do_sample=False,use_cache=True,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)[0]
            torch.cuda.synchronize();generation_s=time.monotonic()-stamp
            answer=tokenizer.decode(generated,skip_special_tokens=True).strip()
            target=pack_prompt(lm,tokenizer,metadata[row["id"]]["transcript"],row["question"],memory,answer=row["answers"][0])
            nll=float(lm(**target,use_cache=False).loss)
            assert all(math.isfinite(v) for v in (nll,first_logit,first_logp))
        record={"role":phase,"id":row["id"],"source_id":row["source_id"],"condition":row["condition"],"arm":arm,"alpha":alpha,
            "question":row["question"],"prompt_embedding_sha256":prompt_hash,"prompt_tokens":prefix_tokens,
            "memory_kind":kind,"answer":answer,"answers":row["answers"],"nll":nll,"first_token_id":target_token,
            "first_token_logit":first_logit,"first_token_logprob":first_logp,"first_token_probability":math.exp(first_logp),"top_token_id":int(logits.argmax()),"generated_token_ids":generated.tolist(),"generated_tokens":len(generated),
            "target_tokens_including_end":int((target["labels"]!=-100).sum()),"finish":"length" if len(generated)==48 and int(generated[-1])!=tokenizer.eos_token_id else "stop",
            "generation_s":generation_s,"pre_alpha_memory_norms":base.norm(dim=-1).tolist(),"post_alpha_memory_norms":memory.norm(dim=-1).tolist(),
            "pre_alpha_memory_sha256":sha(base.cpu().numpy().tobytes()),"memory_metadata":metadata[row["id"]],"transcript_used":metadata[row["id"]]["transcript"]}
        with (out/(phase+".jsonl")).open("ab") as stream:stream.write(canonical(record)+b"\n")
        return record
    pre_sources=sorted(train_sources,key=lambda s:sha(("scale-preflight-v1:"+s.rsplit("-",1)[-1]).encode()))[:4]
    assert pre_sources==config["preflight_source_ids"]
    preflight=[]
    for row in train:
        if row["source_id"] in pre_sources:
            for alpha in GRID:preflight.append(evaluate(row,alpha,"real","preflight","real_alpha_grid"))
    gate=preflight_gate(preflight);write("preflight-gate.json",gate);event({"phase":"preflight_gate",**gate});commit()
    if not gate["passed"]:
        result={"status":"scale_preflight_stopped","config":config,"preflight_rows":preflight,"preflight_gate":gate,"fit_trace":[],"eval_rows":[],
            "memory_metadata":metadata,"weight_invariance":proof(),"scalar_updates":0,"backbone_training_updates":0,"fresh_test_loaded":False,"elapsed_s":time.monotonic()-started}
        write("result.json",result);commit();return result
    theta=torch.nn.Parameter(torch.zeros((),device="cuda",dtype=torch.float32))
    optimizer=torch.optim.Adam([theta],lr=.1);updates=[]
    def train_nll(row,alpha):
        vector=memories[(row["id"],"real")].cuda()*alpha
        packed=pack_prompt(lm,tokenizer,metadata[row["id"]]["transcript"],row["question"],vector,row["answers"][0])
        return lm(**packed,use_cache=False).loss
    def train_mean(alpha):
        with torch.no_grad():return sum(float(train_nll(row,alpha)) for row in train)/64
    alpha1_train_nll=train_mean(1.)
    for epoch in range(4):
        order=list(train);random.Random(17+epoch).shuffle(order)
        for start in range(0,64,8):
            check();optimizer.zero_grad();losses=[]
            for row in order[start:start+8]:
                loss=train_nll(row,theta.exp());assert torch.isfinite(loss)
                (loss/8).backward();losses.append(float(loss.detach()))
            assert theta.grad is not None and torch.isfinite(theta.grad)
            assert all(p.grad is None for model in (lm,adapter,asr) for p in model.parameters())
            before=float(theta.exp().detach());gradient=float(theta.grad);optimizer.step()
            assert torch.isfinite(theta)
            with torch.no_grad():theta.clamp_(math.log(1/32),math.log(2))
            assert torch.isfinite(theta)
            update={"update":len(updates)+1,"epoch":epoch,"mean_loss":sum(losses)/8,"alpha_before":before,
                "theta_gradient":gradient,"alpha_after":float(theta.exp().detach()),"gradient_clipping":None}
            updates.append(update);event({"phase":"scalar_update",**update})
        write("scalar-progress.json",{"theta":float(theta.detach()),"alpha":float(theta.exp().detach()),"updates":updates});commit()
    assert len(updates)==32
    alpha_fit=float(theta.exp().detach());final_train_nll=train_mean(alpha_fit)
    calibration={"theta":float(theta.detach()),"alpha":alpha_fit,"updates":updates,"scalar_updates":32,
        "alpha1_train_nll":alpha1_train_nll,"final_train_nll":final_train_nll,"at_lower_bound":abs(alpha_fit-1/32)<1e-6,"at_upper_bound":abs(alpha_fit-2)<1e-6,"optimizer_parameter_count":1,"train_view_passes":256,"final_endpoint_selection":True}
    write("scalar-final.json",calibration);commit()
    evaluation=[];guards=[]
    for group,subset in (("primary",primary),("illustrative",case)):
        for row in subset:
            for arm,kind,alpha in (("text","text",1.),("real_one","real",1.),("real_fitted","real",alpha_fit),
                ("wrong_one","wrong",1.),("wrong_fitted","wrong",alpha_fit)):
                record=evaluate(row,alpha,kind,group,arm);evaluation.append(record)
                prior=baseline.get((row["id"],"text" if kind=="text" else "real"))
                if prior is not None and arm in ("text","real_one"):
                    guards.append({"id":row["id"],"arm":arm,"answer_match":record["answer"]==prior["answer"],"prior_source":prior.get("source"),
                        "transcript_match":record["transcript_used"]==prior["transcript"]})
            commit()
    assert len(evaluation)==120 and sum(r["role"]=="primary" for r in evaluation)==110
    result={"status":"scale_calibration_complete","config":config,"preflight_rows":preflight,"preflight_gate":gate,"fit_trace":updates,"fitted_alpha":alpha_fit,"calibration":calibration,"eval_rows":evaluation,
        "memory_metadata":metadata,"baseline_guards":guards,"baseline_guard_mismatch":any(not g["answer_match"] or not g["transcript_match"] for g in guards),
        "weight_invariance":proof(),"scalar_updates":32,"backbone_training_updates":0,"fresh_test_loaded":False,"gpu":torch.cuda.get_device_name(),
        "elapsed_s":time.monotonic()-started,"peak_cuda_bytes":torch.cuda.max_memory_allocated(),"scope":"Scalar-only historical calibration; primary11source dev,1806 illustrative separate; no fresh evaluation"}
    result_sha=write("result.json",result);write("artifact-manifest.json",{"result_sha256":result_sha,"scalar_sha256":sha((out/"scalar-final.json").read_bytes()),"memory_metadata_sha256":sha((out/"memory-metadata.json").read_bytes())});commit();return result
