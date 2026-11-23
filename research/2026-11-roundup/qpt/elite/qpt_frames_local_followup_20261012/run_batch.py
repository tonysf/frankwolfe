"""Exclusive bounded low-priority local execution, never more than two CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def atomic(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main(plan_path):
    from run_followup import validate_plan, now, digest
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    phase = plan['phase']
    inventory = ROOT / f'inventory_{phase}.json'
    assert not inventory.exists(), 'Do not duplicate or overwrite a batch.'
    logs = ROOT / f'logs_{phase}'
    logs.mkdir(exist_ok=False)
    resources = plan['resources']
    assert resources['max_workers'] in (1, 2) and resources['cpu_affinity'] == [0, 1]
    deadline = time.monotonic() + resources['batch_wall_seconds']
    state = {'phase': phase, 'plan_sha256': digest(plan_path), 'status': 'running',
             'started_utc': now(), 'trials': {}}
    atomic(inventory, state)

    def worker(trial):
        trial_id = trial['id']
        if deadline - time.monotonic() < 120:
            return trial_id, {'state': 'not_started_deadline'}
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', JAX_PLATFORMS='cpu',
                   XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
        with (logs / f'{trial_id}.log').open('x') as stream:
            p = subprocess.run(['nice', '-n', '10', 'taskset', '-c', '0,1', sys.executable,
                                str(ROOT / 'run_followup.py'), '--plan', str(plan_path), '--trial', trial_id],
                               env=env, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=min(resources['trial_wall_seconds'], max(1, deadline-time.monotonic())))
        return trial_id, {'state': 'finished' if p.returncode == 0 else 'failed', 'exit_code': p.returncode}

    with ThreadPoolExecutor(max_workers=resources['max_workers']) as pool:
        jobs = {pool.submit(worker, t): t['id'] for t in plan['trials']}
        for future in as_completed(jobs):
            try:
                trial_id, result = future.result()
            except Exception as error:
                trial_id, result = jobs[future], {'state': 'infrastructure_failure', 'error': repr(error)}
            state['trials'][trial_id] = result
            atomic(inventory, state)
            path = ROOT / phase / trial_id / 'summary.json'
            summary = json.loads(path.read_text()) if path.exists() else {}
            print(json.dumps({'id': trial_id, **result,
                              **{k: summary[k] for k in ('gate_pass', 'rmse', 'normalized_tp', 'total_seconds') if k in summary}}), flush=True)
    state['status'] = 'complete' if all(v['state'] == 'finished' for v in state['trials'].values()) else 'finished_with_failures'
    state['finished_utc'] = now()
    atomic(inventory, state)
    return 0 if state['status'] == 'complete' else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    sys.exit(main(p.parse_args().plan.resolve()))
