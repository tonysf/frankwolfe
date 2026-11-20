"""Fixed tuning and fresh-seed protocol, using four pinned physical CPU cores."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import hashlib
import itertools
import json
import multiprocessing as mp
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from run import atomic_json, run

ROOT = Path(__file__).resolve().parent


def worker(task):
    cpu = [0, 2, 4, 6][(mp.current_process()._identity[0]-1) % 4]
    name, config = task
    path = ROOT/'runs'/name
    if (path/'result.json').exists():
        old = json.loads((path/'result.json').read_text())
        assert old['config'] == config and old['status'] == 'completed'
        assert old['data_sha256'] == hashlib.sha256((ROOT/'data.npz').read_bytes()).hexdigest()
        assert old['source_sha256'] == {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                        for name in ['core.py','run.py']}
        return dict(out=str(path), final=old['trace'][-1], config=config,
                    optimizer_seconds=old['optimizer_seconds'], diagnostic_seconds=old['diagnostic_seconds'])
    return run(config, path, cpu)


def execute(tasks, phase):
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        pending = {pool.submit(worker, task): task[0] for task in tasks}
        for future in as_completed(pending):
            try:
                results.append(future.result())
            except Exception as exc:
                atomic_json(ROOT/'failure.json', dict(phase=phase, run=pending[future], error=repr(exc)))
                raise
            atomic_json(ROOT/'progress.json', dict(phase=phase, completed=len(results), total=len(tasks),
                latest_run=pending[future], updated_utc=datetime.now(timezone.utc).isoformat()))
    return results


def main():
    started = time.perf_counter()
    with np.load(ROOT/'data.npz') as d:
        count = int(d['mask'].sum())
    frames_candidates = [dict(method='sframes', gamma0=g, rho0=r, beta0=count*b, beta_factor=b,
                              candidate=f'f{i:02d}')
        for i, (g, r, b) in enumerate(itertools.product([.3, 1.], [.3, 1.], [.1, 1., 10., 100.]))]
    sh_candidates = [dict(method='shcgm', gamma0=1., rho0=1., beta0=count*float(b), beta_factor=float(b),
                          candidate=f'h{i:02d}')
        for i, b in enumerate(np.geomspace(.01, 1000., 16))]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        iterations=20000, tuning_batch=64, confirmation_batches=[16,64,256],
        tuning_seeds=[101,102], confirmation_seeds=[201,202,203,204,205],
        frames_candidates=frames_candidates, shcgm_candidates=sh_candidates,
        selection='lowest median final radial-repaired feasible training loss at20k on two tuning seeds; per-method selection; batch64 tuning reused unchanged across confirmation batches',
        literal_frames_control='also confirm best gamma0=rho0=1 setting from its four beta0 candidates',
        data_sha256=hashlib.sha256((ROOT/'data.npz').read_bytes()).hexdigest(),
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['core.py','run.py','campaign.py']},
        workers=4, physical_cpu_affinity=[0,2,4,6], threads_per_worker=1,
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        initialization='zero for every method; no method sees reference solution',
        sampling='IID uniform observed-entry indices, common RNG seed and stream for methods at each batch/seed; gradients depend on each method own iterate',
        schedules=dict(sframes='gamma0/(k+1)^.75,rho0/(k+1)^.5,beta0/(k+1)^.25; k>=0; first sample initializes momentum',
                       shcgm='9/(k+9),4/(k+8)^(2/3),beta0/sqrt(k+9); zero-based k; d0=0; equivalent to published one-based Algorithm1'),
        normalization='half-mean squared loss; beta0 grid is expressed as beta_factor times number of observations; not a reproduction of published MovieLens loss values')
    atomic_json(ROOT/'campaign_config.json', manifest)
    tasks = []
    for c in frames_candidates+sh_candidates:
        for seed in manifest['tuning_seeds']:
            config = dict(c, seed=seed, batch=64, iterations=20000, phase='tuning')
            tasks.append((f"tuning/{c['candidate']}_seed{seed}", config))
    tuned = execute(tasks, 'tuning')
    scores = {}
    for c in frames_candidates+sh_candidates:
        vals = [r['final']['feasible_loss'] for r in tuned if r['config']['candidate']==c['candidate']]
        scores[c['candidate']] = float(np.median(vals))
    bestframes = min(frames_candidates, key=lambda c:scores[c['candidate']])
    bestsh = min(sh_candidates, key=lambda c:scores[c['candidate']])
    unitframes = min([c for c in frames_candidates if c['gamma0']==c['rho0']==1.], key=lambda c:scores[c['candidate']])
    selected = {'sframes_tuned':bestframes, 'shcgm':bestsh, 'sframes_unit':unitframes}
    atomic_json(ROOT/'selection.json', dict(selected=selected, scores=scores, selection=manifest['selection']))
    tasks = []
    for name, c in selected.items():
        for batch, seed in itertools.product(manifest['confirmation_batches'], manifest['confirmation_seeds']):
            config = dict(c, seed=seed, batch=batch, iterations=20000, phase='confirmation', label=name)
            tasks.append((f'confirmation/{name}_q{batch}_seed{seed}', config))
    confirmed = execute(tasks, 'confirmation')
    for batch, seed in itertools.product(manifest['confirmation_batches'], manifest['confirmation_seeds']):
        hashes = [json.loads((Path(r['out'])/'result.json').read_text())['sampled_stream_sha256']
                  for r in confirmed if r['config']['batch']==batch and r['config']['seed']==seed]
        assert len(set(hashes)) == 1, 'Methods did not share the intended sample-index stream'
    atomic_json(ROOT/'completed.json', dict(status='completed', tuning_runs=len(tuned), confirmation_runs=len(confirmed),
        elapsed_seconds=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
        common_stream_validation='passed', selected=selected))
    print(json.dumps(json.loads((ROOT/'completed.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    main()
