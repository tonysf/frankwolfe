"""Bounded, low-priority local batch, at most two single-core worker processes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
def atomic(path,value):
    tmp=path.with_suffix('.pending')
    with tmp.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    os.replace(tmp,path)

def main(plan_path):
    plan=json.loads(plan_path.read_text())
    phase=plan['phase']
    inventory=ROOT/f'inventory_{phase}.json'
    assert not inventory.exists(), 'No overwriting or duplicate batch.'
    logs=ROOT/f'logs_{phase}'
    logs.mkdir(exist_ok=False)
    workers=plan['resources']['max_workers']
    assert workers in (1,2)
    deadline=time.monotonic()+plan['resources']['maximum_wall_seconds_per_batch']
    state={'phase':phase,'status':'running','trials':{},'plan':str(plan_path)}
    atomic(inventory,state)
    def worker(trial):
        tid=trial['id']
        if deadline-time.monotonic()<120:return tid,{'state':'not_started_deadline'}
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
        with (logs/f'{tid}.log').open('x') as log:
            # Two-worker process affinity confines the entire batch to two logical CPUs.
            p=subprocess.run(['nice','-n','10','taskset','-c','0,1',sys.executable,str(ROOT/'run_local.py'),
                              '--plan',str(plan_path),'--trial',tid],env=env,stdout=log,stderr=subprocess.STDOUT,
                              timeout=max(1,deadline-time.monotonic()))
        return tid,{'state':'finished' if p.returncode==0 else 'failed','exit_code':p.returncode,'log':str(logs/f'{tid}.log')}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks={pool.submit(worker,t):t['id'] for t in plan['trials']}
        for future in as_completed(tasks):
            try:tid,result=future.result()
            except Exception as e:tid,result=tasks[future],{'state':'infrastructure_failure','error':repr(e)}
            state['trials'][tid]=result
            atomic(inventory,state)
            summary=ROOT/phase/tid/'summary.json'
            if summary.exists():
                s=json.loads(summary.read_text())
                print(json.dumps({'id':tid,'state':result['state'],'gate':s['gate_pass'],'rmse':s['rmse'],
                                  'tp':s['normalized_tp'],'seconds':s['total_seconds']}),flush=True)
            else:print(json.dumps({'id':tid,**result}),flush=True)
    state['status']='complete' if all(v['state']=='finished' for v in state['trials'].values()) else 'finished_with_failures'
    atomic(inventory,state)
    return 0 if state['status']=='complete' else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    sys.exit(main(p.parse_args().plan))
