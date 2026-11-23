"""Bounded, exclusive local CPU launcher with inclusive failure inventories."""
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

ROOT = Path(__file__).resolve().parent


def set_local_limits(resources):
    if resources['max_workers'] not in (1, 2) or not set(resources['cpu_affinity']) <= {0, 1}:
        raise ValueError('Only CPUs0,1 and at most two workers are allowed.')
    if not 10 <= resources['nice'] <= 19:
        raise ValueError('Local jobs must be low priority.')
    os.sched_setaffinity(0, set(resources['cpu_affinity']))
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority < resources['nice']: os.nice(resources['nice']-priority)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def update_inventory(path, state):
    """Replace only the current batch's own already-created inventory."""
    if json.loads(path.read_text())['batch_id'] != state['batch_id']:
        raise ValueError('Inventory ownership changed.')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with temporary.open('x') as stream:
        json.dump(state, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    os.replace(temporary, path)


def stop_process_group(process):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=5)


def worker_outcome(returncode, summary):
    if summary.get('status') == 'failed': return 'method_failure'
    return 'finished' if returncode == 0 and summary.get('status') == 'complete' else 'infrastructure_failure'


def main(plan_path):
    plan_path = Path(plan_path).resolve()
    if plan_path.parent != ROOT: raise ValueError('Plan outside campaign root.')
    plan = json.loads(plan_path.read_text()); set_local_limits(plan['resources'])
    from run_tuning import validate_plan, runtime_guard, plan_path_in_root, execution_prerequisites, now, digest, write
    plan_path_in_root(plan_path); runtime_guard(); spec = validate_plan(plan); execution_prerequisites(plan,spec)
    phase = plan['phase']; resources = plan['resources']; batch_id = uuid.uuid4().hex
    inventory, logs = ROOT/f'inventory_{phase}.json', ROOT/f'logs_{phase}'
    if inventory.exists() or logs.exists(): raise FileExistsError('Refusing to repeat or overwrite a batch.')
    if any((ROOT/phase/t['id']).exists() for t in plan['trials']):
        raise FileExistsError('A planned trial already exists; no implicit resumption.')
    logs.mkdir(exist_ok=False); deadline = time.monotonic()+resources['batch_wall_seconds']
    state = dict(batch_id=batch_id, phase=phase, plan_sha256=digest(plan_path), status='running', started_utc=now(),
                 total_charged_row_budget=plan['total_charged_row_budget'],
                 trials={t['id']: {'state':'pending','charged_row_budget':t['charged_row_budget']} for t in plan['trials']})
    write(inventory, state)

    def worker(trial):
        tid = trial['id']; remaining = deadline-time.monotonic()
        if remaining <= 1: return tid, dict(state='not_started_deadline', charged_row_budget=trial['charged_row_budget'])
        timeout = min(resources['trial_wall_seconds'], remaining)
        cmd = [sys.executable,str(ROOT/'run_tuning.py'),'--plan',str(plan_path),'--trial',tid]
        start = time.monotonic()
        with (logs/f'{tid}.log').open('x') as stream:
            process = subprocess.Popen(cmd, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try: code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stop_process_group(process)
                return tid, dict(state='timeout', timeout_seconds=timeout, seconds=time.monotonic()-start,
                                 charged_row_budget=trial['charged_row_budget'])
        path = ROOT/phase/tid/'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {}
        result = dict(state=worker_outcome(code,summary),exit_code=code,seconds=time.monotonic()-start,
                      charged_row_budget=trial['charged_row_budget'])
        for key in ('gate_pass','charged_rows','rmse','normalized_tp','failure_reason'):
            if key in summary: result[key] = summary[key]
        return tid, result

    with ThreadPoolExecutor(max_workers=resources['max_workers']) as pool:
        jobs = {pool.submit(worker,t):t for t in plan['trials']}
        for future in as_completed(jobs):
            trial = jobs[future]
            try: tid, outcome = future.result()
            except Exception as error:
                tid, outcome = trial['id'], dict(state='infrastructure_failure',error=repr(error),charged_row_budget=trial['charged_row_budget'])
            state['trials'][tid] = outcome; update_inventory(inventory,state)
            print(json.dumps({'id':tid,**outcome}),flush=True)
    state['status'] = 'complete' if all(r['state']=='finished' for r in state['trials'].values()) else 'finished_with_failures'
    state['finished_utc'] = now(); update_inventory(inventory,state)
    return 0 if state['status']=='complete' else 2


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--plan',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().plan))
