"""Bounded detached process supervisor; records timeout and immutable start marker."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import fcntl

ROOT=Path(__file__).resolve().parent

def main():
    os.chdir(ROOT)
    stage=sys.argv[1]
    if stage not in ('preflight','experiment'): raise ValueError(stage)
    lock=(ROOT.parent/'.gpu-assignment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=ROOT/(stage+'-status.json')
    # Exclusive marker refuses duplicate launches, including prior failed runs.
    with status.open('x') as f:
        json.dump({'state':'starting','started_unix':time.time(),'supervisor_pid':os.getpid()},f)
    command=[sys.executable,'gpu_batch.py','--stage',stage]
    started=time.time()
    p=subprocess.Popen(command,stdin=subprocess.DEVNULL,start_new_session=True)
    data={'state':'running','started_unix':started,'supervisor_pid':os.getpid(),'child_pid':p.pid,'argv':command,'timeout_s':2700}
    status.write_text(json.dumps(data,indent=2))
    try:
        code=p.wait(timeout=2700)
        reason='completed' if code==0 else 'failed'
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try: p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL); p.wait()
        code=124; reason='timeout'
    data.update(state=reason,exit=code,finished_unix=time.time(),elapsed_s=time.time()-started)
    status.write_text(json.dumps(data,indent=2))
    sys.exit(code)

if __name__=='__main__': main()
