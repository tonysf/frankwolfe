"""Bounded local CPU FraMES comparison on the hash-verified SCOT instance.

The four prefactor candidates and fixed horizon are frozen before tuning.
Tuning uses two single-thread processes; confirmation uses three fresh seeds,
sequentially, for interpretable CPU solver times. No loss-based early stopping.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/scot_cpu_20261014'
REFERENCE = OUT / 'reference'
sys.path.insert(0, str(ROOT / 'extensions/scot_reproduction_20261014'))
from reference import verify_reference, write_json

CANDIDATES = [
    dict(id='fw04', gamma0=.3, rho0=.3, mu_scale=1.),
    dict(id='fw05', gamma0=.3, rho0=.3, mu_scale=10.),
    dict(id='fw07', gamma0=1., rho0=.3, mu_scale=1.),
    dict(id='fw08', gamma0=1., rho0=.3, mu_scale=10.),
]
HORIZON = 2000
CHECKPOINTS = [1, 10, 50, 100, 200, 500, 1000, 1500, 2000]


def execute(phase, candidate, seed):
    name = f'{phase}_{candidate["id"]}_seed{seed}'
    output = OUT / 'runs' / name
    if output.exists():
        raise ValueError(f'Refusing to reuse {output}')
    command = [sys.executable, str(ROOT / 'extensions/pair_sampling_20261011/run_pair.py'),
        '--instance', str(REFERENCE / 'prepared/scot1047'), '--out', str(output),
        '--method', 'smoothing-fw', '--device', 'cpu', '--dtype', 'float32',
        '--iterations', str(HORIZON), '--seed', str(seed),
        '--output-seed', str(seed+20000), '--init-seed', str(seed+10000),
        '--initialization', 'product', '--oracle', 'pair-sampling',
        '--pairs-per-call', '1024', '--save-plans', 'all',
        '--gamma0', str(candidate['gamma0']), '--rho0', str(candidate['rho0']),
        '--mu-scale', str(candidate['mu_scale']), '--progress-every', '1000',
        '--checkpoint-iterations', ','.join(map(str, CHECKPOINTS)),
        '--checkpoints', '30,60,120,180,240,300,600']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    started = time.perf_counter()
    with (OUT / f'{name}.log').open('x') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=3600)
    receipt = dict(name=name, command=command, exit_code=result.returncode,
                   process_elapsed_seconds=time.perf_counter()-started,
                   concurrent_tuning=(phase == 'tune'), cpu_threads_per_process=1)
    write_json(OUT / f'{name}_receipt.json', receipt)
    if result.returncode:
        raise RuntimeError(f'{name} failed with exit {result.returncode}; see log.')
    metadata = json.loads((output / 'metadata.json').read_text())
    count = metadata['counters']
    if (metadata['status'] != 'completed' or count['completed_updates'] != HORIZON
            or count['stochastic_gradient_calls'] != HORIZON
            or count['pair_samples'] != HORIZON*1024
            or metadata['ranks'][0]['device'] != 'cpu'):
        raise ValueError(f'{name} failed completion/oracle/device checks')
    if metadata['array_files_sha256'] != verify_reference(REFERENCE)['prepared_instance']['array_files_sha256']:
        raise ValueError(f'{name} input hash mismatch')
    loss = metadata['final_float64_validation']['plans']['last_rounded']['loss']
    print(json.dumps(dict(name=name, loss=loss,
                          solver_seconds=metadata['timing']['solver_seconds'])), flush=True)
    return dict(candidate=candidate, seed=seed, directory=str(output),
                last_rounded_loss_float64=loss, receipt=receipt)


def main():
    reference = verify_reference(REFERENCE)
    (OUT / 'runs').mkdir(exist_ok=True)
    config = dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        fixed_horizon=HORIZON, candidates=CANDIDATES, tuning_seed=810,
        confirmation_seeds=[811, 812, 813], pairs_per_call=1024,
        selected_metric='independent_float64_last_rounded_loss_at_fixed_horizon',
        threads_per_process=1, tuning_workers=2, confirmation_workers=1,
        optimizer_dtype='float32', final_evaluation_dtype='float64',
        initialization='uniform product', checkpoint_iterations=CHECKPOINTS,
        selection_scope='Four of the nine frozen Ruche FraMES candidates; local preliminary comparison only.',
        theory_scope='Unchanged decay exponents .75/.5/.25; empirical prefactor tuning. Final loss is not a theorem-output guarantee.',
        timing_scope='Tuning timings excluded from CPU comparisons because two runs overlap. Confirmation runs sequentially; original runner reports solver+current output repair separately from end-to-end time.',
        reference_loss=reference['rounded']['loss'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (OUT / 'local_config.json').open('x') as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tuned = list(pool.map(lambda c: execute('tune', c, 810), CANDIDATES))
    selected = min(tuned, key=lambda r: r['last_rounded_loss_float64'])['candidate']
    write_json(OUT / 'selection.json', dict(selected=selected, tuning=tuned,
                                          selection_metric=config['selected_metric']))
    confirmed = [execute('confirm', selected, seed) for seed in config['confirmation_seeds']]
    write_json(OUT / 'completed.json', dict(status='completed', config=config,
                                           tuning=tuned, selected=selected, confirmation=confirmed))


if __name__ == '__main__':
    main()
