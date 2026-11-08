"""Runs inside the selected college pod, supervised by college_batch.py."""
import json
import os
import subprocess
import sys
import time
import traceback
import argparse
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EVALUATION_SPLITS=('test',)
EVALUATION_MODES=('hint',)
CACHE_REUSE_RUN='vaani-v2-dev-20261103'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['preflight','experiment'],default='preflight')
    cfg=parser.parse_args()
    os.chdir(ROOT)
    env=os.environ.copy()
    cache=ROOT/'cache' if CACHE_REUSE_RUN is None else ROOT.parent/CACHE_REUSE_RUN/'cache'
    if CACHE_REUSE_RUN is not None:
        if Path(CACHE_REUSE_RUN).name!=CACHE_REUSE_RUN or not CACHE_REUSE_RUN.startswith('vaani-') or not cache.is_dir():
            raise ValueError('Expected verified prior assignment cache directory')
        previous=json.loads((cache.parent/'experiment-status.json').read_text())
        if previous.get('state')!='completed' or previous.get('exit')!=0:
            raise ValueError('Reuse requires a successful prior assignment experiment')
    env.update(HF_HOME=str(cache),HF_HUB_DISABLE_XET="1",HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONNOUSERSITE="1",USE_TF="0",USE_FLAX="0",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="4")
    runtime={}
    for name in ['memory.max','memory.current','cpu.max','memory.events']:
        path=Path('/sys/fs/cgroup')/name
        if path.exists(): runtime[name]=path.read_text().strip()
    runtime['work_bytes']=subprocess.check_output(['du','-sb','/home/jovyan/work'],text=True).split()[0]
    (ROOT/(cfg.stage+'-resources.json')).write_text(json.dumps(runtime,indent=2))
    # Keep code dependencies and weights within the experiment directory. The existing CUDA torch is reused.
    deps=ROOT/"deps"
    env["PYTHONPATH"]=str(deps)
    (ROOT/(cfg.stage+'-provenance.json')).write_text(json.dumps({
        'stage':cfg.stage,'python':sys.version,'started_unix':time.time(),
        'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['run.py','score.py','gpu_batch.py','gpu_supervisor.py']},
        'manifest_sha256':hashlib.sha256((ROOT/'data/manifest.jsonl').read_bytes()).hexdigest(),
        'environment':{k:env[k] for k in ['HF_HOME','HF_HUB_DISABLE_XET','PYTHONPATH','PYTHONNOUSERSITE','USE_TF','USE_FLAX','OMP_NUM_THREADS']}
    },indent=2))
    if cfg.stage=='preflight':
        # Reuse the image's matching CUDA Torch / NumPy / SciPy stack. In particular,
        # never let pip shadow NumPy or upgrade Torch via transitive dependencies.
        subprocess.run([sys.executable,"-m","pip","install","--target",str(deps),"--no-cache-dir","--no-deps","--only-binary=:all:",
                        "transformers==4.57.2","huggingface-hub==0.36.0","tokenizers==0.22.2",
                        "safetensors==0.8.0","regex==2026.9.29"],env=env,check=True,timeout=420)
        check=[sys.executable,"-c","import json,torch,numpy,scipy,transformers; from transformers import WhisperProcessor,WhisperForConditionalGeneration,Qwen2ForCausalLM; a=torch.ones(1,device='cuda'); print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),free_vram=torch.cuda.mem_get_info()[0],smoke=float(a.sum()))))"]
        result=subprocess.check_output(check,env=env,text=True,timeout=70)
        (ROOT/"env.json").write_text(result)
        print(result,flush=True)
        return
    if not (ROOT/'preflight-status.json').exists() or json.loads((ROOT/'preflight-status.json').read_text()).get('exit')!=0:
        raise RuntimeError('Successful import/CUDA preflight required')
    def infer(name,mode,split,extra=()):
        print('START',name,flush=True)
        subprocess.run([sys.executable,"run.py","data/manifest.jsonl",f"results/{name}","--split",split,"--mode",mode,*extra],check=True,env=env)
        print('COMPLETE',name,flush=True)
    infer("smoke","auto","dev",["--limit","1"])
    for split in EVALUATION_SPLITS:
        for mode in EVALUATION_MODES:
            name=split+'-'+mode
            infer(name,mode,split)
            with open(f"results/{name}/metrics.json","w") as f:
                subprocess.run([sys.executable,"score.py","data/manifest.jsonl",f"results/{name}/predictions.jsonl","--split",split],stdout=f,check=True,env=env)


if __name__=="__main__":
    started=time.time()
    code=0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code=1
    finally:
        (ROOT/"exit.json").write_text(json.dumps({"exit":code,"elapsed_s":time.time()-started}))
    sys.exit(code)

