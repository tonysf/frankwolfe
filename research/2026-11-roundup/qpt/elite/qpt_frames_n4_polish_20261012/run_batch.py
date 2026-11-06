"""Bounded local-only polish driver; every declared trial remains in inventory."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent


def set_local_limits(r):
    if r['max_workers'] not in (1,2) or r['cpu_affinity']!=[0,1] or not 10<=r['nice']<=19:
        raise ValueError('Only low-priority two-CPU local work is allowed.')
    os.sched_setaffinity(0,{0,1}); priority=os.getpriority(os.PRIO_PROCESS,0)
    if priority<r['nice']:os.nice(r['nice']-priority)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'


def update_inventory(path,state):
    if json.loads(path.read_text())['batch_id']!=state['batch_id']:raise ValueError('Inventory ownership changed.')
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    with temp.open('x') as f:json.dump(state,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    os.replace(temp,path)


def stop_group(p):
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=5)


def worker_outcome(code,summary):
    if code==2 and summary.get('status')=='failed':return 'method_failure'
    return 'finished' if code==0 and summary.get('status')=='complete' else 'infrastructure_failure'


def main(plan_path):
    plan_path=Path(plan_path).resolve()
    if plan_path.parent!=ROOT:raise ValueError('Plan must belong to this campaign root.')
    plan=json.loads(plan_path.read_text());set_local_limits(plan['resources'])
    from run_polish import validate_plan,confirmation_guard,runtime_guard,path_guard,digest,now,write
    path_guard(plan_path);runtime_guard();validate_plan(plan);confirmation_guard(plan)
    phase=plan['phase'];r=plan['resources'];inventory=ROOT/f'inventory_{phase}.json';logs=ROOT/f'logs_{phase}'
    if inventory.exists() or logs.exists() or any((ROOT/phase/t['id']).exists() for t in plan['trials']):
        raise FileExistsError('Refusing repeated batch, overwrite or implicit resumption.')
    logs.mkdir(exist_ok=False);deadline=time.monotonic()+r['batch_wall_seconds']
    state=dict(batch_id=uuid.uuid4().hex,phase=phase,plan_sha256=digest(plan_path),status='running',started_utc=now(),
        total_charged_row_budget=plan['total_charged_row_budget'],
        trials={t['id']:dict(state='pending',charged_row_budget=t['charged_row_budget']) for t in plan['trials']})
    write(inventory,state)
    def worker(t):
        tid=t['id'];remaining=deadline-time.monotonic();budget=t['charged_row_budget']
        if remaining<=1:return tid,dict(state='not_started_deadline',charged_row_budget=budget)
        timeout=min(r['trial_wall_seconds'],remaining);start=time.monotonic()
        with (logs/f'{tid}.log').open('x') as log:
            p=subprocess.Popen([sys.executable,str(ROOT/'run_polish.py'),'--plan',str(plan_path),'--trial',tid],
                env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_group(p);return tid,dict(state='timeout',timeout_seconds=timeout,seconds=time.monotonic()-start,charged_row_budget=budget)
        path=ROOT/phase/tid/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {}
        result=dict(state=worker_outcome(code,summary),exit_code=code,seconds=time.monotonic()-start,charged_row_budget=budget)
        for key in ('gate_pass','charged_rows','rmse','normalized_tp','failure_reason'):
            if key in summary:result[key]=summary[key]
        return tid,result
    with ThreadPoolExecutor(max_workers=r['max_workers']) as pool:
        jobs={pool.submit(worker,t):t for t in plan['trials']}
        for future in as_completed(jobs):
            t=jobs[future]
            try:tid,result=future.result()
            except Exception as error:tid,result=t['id'],dict(state='infrastructure_failure',error=repr(error),charged_row_budget=t['charged_row_budget'])
            state['trials'][tid]=result;update_inventory(inventory,state);print(json.dumps({'id':tid,**result}),flush=True)
    state['status']='complete' if all(x['state']=='finished' for x in state['trials'].values()) else 'finished_with_failures'
    state['finished_utc']=now();update_inventory(inventory,state)
    return 0 if state['status']=='complete' else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);raise SystemExit(main(p.parse_args().plan))
