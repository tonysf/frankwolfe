"""One reproducible CPU run, with exact per-iterate gap diagnostics outside optimizer timing."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from core import Problem, schedule, update

ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def run(config, out, cpu=None):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    problem = Problem.load(ROOT/'data.npz')
    rng = np.random.default_rng(config['seed'])
    x = np.zeros_like(problem.Y)
    momentum = None
    trace = []
    gap_sum = 0.
    optimizer_seconds = diagnostic_seconds = 0.
    stream_hash = hashlib.sha256()
    n = config['iterations']
    log_at = set([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, n])
    log_at.update(range(1000, n+1, 500))
    kwargs = {k: config[k] for k in ['method', 'beta0', 'gamma0', 'rho0']}

    def observe(iteration):
        nonlocal diagnostic_seconds
        begin = time.perf_counter()
        beta = schedule(iteration=iteration, **kwargs)[2]
        row = dict(iteration=iteration, beta=beta, sampled_entries=iteration*config['batch'],
                   mean_true_smoothed_gap=gap_sum/iteration if iteration else None,
                   **problem.metrics(x, beta))
        if row['nuclear_norm'] > problem.tau+1e-8 or row['feasible_nuclear_norm'] > problem.tau+1e-8:
            raise ValueError('Nuclear-ball feasibility violated')
        if row['feasible_max_abs'] > 1+1e-12 or not all(np.isfinite(v) for v in row.values() if v is not None):
            raise ValueError('Invalid metric or repaired output')
        diagnostic_seconds += time.perf_counter()-begin
        row.update(optimizer_seconds=optimizer_seconds, diagnostic_seconds=diagnostic_seconds,
                   total_seconds=time.perf_counter()-started)
        trace.append(row)

    observe(0)
    for k in range(n):
        begin = time.perf_counter()
        beta = schedule(iteration=k, **kwargs)[2]
        gap = problem.true_gap(x, beta)
        if gap < -1e-9 or not np.isfinite(gap):
            raise ValueError('Invalid true gap')
        gap_sum += gap
        diagnostic_seconds += time.perf_counter()-begin
        begin = time.perf_counter()
        draws = rng.integers(problem.count, size=config['batch'])
        sampled = problem.sample_gradient(x, draws)
        x, momentum = update(problem, x, momentum, sampled, iteration=k, **kwargs)
        optimizer_seconds += time.perf_counter()-begin
        stream_hash.update(draws.astype('<i8').tobytes())
        if k+1 in log_at:
            observe(k+1)
    np.savez_compressed(out/'final_state.npz', x=x, momentum=momentum,
                        feasible_x=problem.feasible_repair(x))
    result = dict(status='completed', config=config, trace=trace,
        completed_updates=n, sampled_entries=n*config['batch'],
        optimizer_seconds=optimizer_seconds, diagnostic_seconds=diagnostic_seconds,
        total_seconds=time.perf_counter()-started, sampled_stream_sha256=stream_hash.hexdigest(),
        data_sha256=hashlib.sha256((ROOT/'data.npz').read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['run.py','core.py']},
        cpu_affinity=sorted(os.sched_getaffinity(0)), numpy_version=np.__version__,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        timing_scope='optimizer includes sampling, momentum, penalty gradient, full-SVD LMO and update; diagnostics timed separately; concurrent pinned single-thread workers',
        statistic_scope='mean_true_smoothed_gap at N averages G_k for ALL pre-update states k=0,...,N-1 with their own beta_k; loss is current radial-repaired feasible output')
    atomic_json(out/'result.json', result)
    return dict(out=str(out), final=trace[-1], config=config,
                optimizer_seconds=optimizer_seconds, diagnostic_seconds=diagnostic_seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = run(config, args.out, args.cpu)
    print(json.dumps(result))
