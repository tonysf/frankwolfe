"""Single-trajectory, elapsed-time runner for instrumented installed POT Sampled GW.

The native estimator, conditional NumPy sampling, cost normalization, Sinkhorn
call, product initialization, and native stopping rule are preserved. CUDA tensors
select POT's torch backend, while native CPU transfers remain algorithmic work.
This is an instrumented hybrid native solver, not a new GPU implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import os
import resource
import shutil
import time
import warnings

import numpy as np
import torch

from gw_core import GWProblem, assignment_lmo, marginal_residuals, round_transport
from run_benchmark import (resolve_device, CheckpointGrid, _assert_plan, _command, _json_safe,
                           _write_csv, environment, sha256_file, sync)

ROOT = Path(__file__).resolve().parent


def run(args):
    from native_sampled_gw import sampled_gromov_wasserstein

    total_start = time.perf_counter()
    if not math.isfinite(args.budget) or args.budget <= 0:
        raise ValueError('A finite positive solver-time budget is required.')
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError('Native Sampled GW requires positive proximal epsilon here.')
    if args.batch < 1 or args.max_iter < 1 or args.threads < 1:
        raise ValueError('batch, max-iter, and threads must be positive.')
    targets = ([float(value) for value in args.checkpoints.split(',')] if args.checkpoints else
               [value for value in (1., 3., 10., 30., 100., 300.) if value <= args.budget])
    if any(value > args.budget for value in targets):
        raise ValueError('Checkpoint targets must not exceed the budget.')
    grid = CheckpointGrid(sorted(set(targets + [args.budget])))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, dtype = resolve_device(args.device), getattr(torch, args.dtype)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    instance = Path(args.instance).resolve()
    with np.load(instance, allow_pickle=False) as archive:
        canonical = {key: np.array(archive[key], copy=True) for key in ('A', 'B', 'a', 'b')}
    hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in canonical.items()}
    manifest = json.loads(instance.with_suffix('.json').read_text()) if instance.with_suffix('.json').exists() else None
    if manifest:
        for key in ('A', 'B'):
            if manifest.get(key + '_sha256') != hashes[key]:
                raise ValueError(f'{key} differs from the frozen manifest hash.')
    problem = GWProblem(*(torch.as_tensor(canonical[key], dtype=dtype, device=device)
                          for key in ('A', 'B', 'a', 'b')))
    n, m = len(problem.a), len(problem.b)
    metadata = dict(method='native-sampled-gw', config=vars(args),
                    instance=str(instance), instance_sha256=sha256_file(instance),
                    array_sha256=hashes, instance_manifest=manifest,
                    environment=environment(device), pot_version=importlib.metadata.version('POT'),
                    source_commit=_command(['git', 'rev-parse', 'HEAD']),
                    source_dirty=bool(_command(['git', 'status', '--porcelain'])),
                    source_file_sha256={name: sha256_file(ROOT / name) for name in
                                        ('run_native_sampled.py', 'native_sampled_gw.py', 'run_benchmark.py', 'gw_core.py')},
                    stochastic_seed=args.seed, initialization_seed=args.init_seed,
                    initialization='native product a*b^T; initialization seed is recorded but consumes no randomness',
                    backend_description='installed POT hybrid native solver with torch backend; native NumPy sampling and to_numpy transfers preserved',
                    solver_semantics='one uninterrupted native trajectory; no restart per checkpoint',
                    sample_semantics='native nb_samples_grad; not equated with sketch probes or scalar-anchor samples',
                    output_selection='current feasible rounded output; no best-so-far selection',
                    checkpoint_policy='first completed native iteration crossing earliest target; other crossed windows missed',
                    inner_iterations=None, inner_stopping_failures=None,
                    unknown_inner_counts_reason='native sinkhorn call supplies no inner log; counts are not inferred from feasibility or warnings',
                    preparation_policy='shared input loading/device placement and objective square caches are outside solver; all native preparation and native transfers use module solver timing',
                    native_source_archive=str(out / 'native_source'))
    sync(device)
    preparation_seconds = time.perf_counter() - total_start
    repair_seconds = diagnostic_seconds = validation_seconds = 0.
    diagnostic_counts = dict(loss_calls=0, full_gradient_calls=0, assignment_lmo_calls=0)
    trace, failure_events = [], []
    last_info = dict(completed_iterations=0, completed_updates=0, solver_seconds=0., updated_transport=False)
    last_evaluated_raw = last_rounded = None
    last_evaluated_iteration = None
    returned_plan = None
    final_validation = None
    report = None
    status, error = 'running', None

    def evaluate(P, kind, target, info):
        nonlocal repair_seconds, diagnostic_seconds, last_evaluated_raw, last_rounded, last_evaluated_iteration
        # Snapshot allocation/transfer is done by the instrumented module and is
        # separately reported there. round_transport itself leaves P unchanged.
        sync(device)
        started = time.perf_counter()
        _assert_plan(P)
        R = round_transport(P, problem.a, problem.b)
        _assert_plan(R)
        sync(device)
        repair_current = time.perf_counter() - started
        repair_seconds += repair_current
        started = time.perf_counter()
        raw_residuals = marginal_residuals(P, problem.a, problem.b)
        rounded_residuals = marginal_residuals(R, problem.a, problem.b)
        if rounded_residuals['both_l1'] > 100 * torch.finfo(dtype).eps:
            raise FloatingPointError('Rounded native output fails the marginal tolerance.')
        raw_loss, rounded_loss = problem.loss(P).item(), problem.loss(R).item()
        diagnostic_counts['loss_calls'] += 2
        if not math.isfinite(raw_loss) or not math.isfinite(rounded_loss):
            raise FloatingPointError('Nonfinite native output objective rejected.')
        solver_seconds = float(info['solver_seconds'])
        row = dict(kind=kind, target_solver_seconds=target,
                   actual_solver_seconds=solver_seconds, iteration=int(info['completed_iterations']),
                   native_completed_updates=int(info['completed_updates']),
                   native_updated_transport=bool(info['updated_transport']),
                   raw_loss=raw_loss, rounded_loss=rounded_loss,
                   raw_zero_entries=int((P == 0).sum().item()),
                   repair_l1=(R - P).abs().sum().item(), repair_seconds_current=repair_current,
                   repair_seconds_cumulative=repair_seconds,
                   time_to_this_feasible_output=solver_seconds + repair_current,
                   **raw_residuals,
                   **{'rounded_' + key: value for key, value in rounded_residuals.items()})
        uniform = (torch.allclose(problem.a, torch.full_like(problem.a, 1 / n))
                   and torch.allclose(problem.b, torch.full_like(problem.b, 1 / m)))
        if n == m and n <= args.gap_max_n and uniform:
            G = problem.gradient(R)
            S = assignment_lmo(G, problem.a, problem.b)
            diagnostic_counts['full_gradient_calls'] += 1
            diagnostic_counts['assignment_lmo_calls'] += 1
            row['original_fw_gap'] = (G * (R - S)).sum().item()
        sync(device)
        diagnostic_seconds += time.perf_counter() - started
        row['diagnostic_seconds_cumulative'] = diagnostic_seconds
        trace.append(row)
        last_evaluated_raw, last_rounded = P, R
        last_evaluated_iteration = int(info['completed_iterations'])

    def callback(snapshot_T, info):
        nonlocal last_info
        required = ('completed_iterations', 'completed_updates', 'solver_seconds', 'updated_transport')
        if any(key not in info for key in required):
            raise ValueError(f'Instrumentation callback omitted one of {required}.')
        last_info = dict(info)
        P = snapshot_T if isinstance(snapshot_T, torch.Tensor) else torch.as_tensor(snapshot_T, dtype=dtype, device=device)
        target, missed = grid.consume(float(info['solver_seconds']))
        if target is not None:
            evaluate(P, 'checkpoint', target, info)
            for missed_target in missed:
                trace.append(dict(kind='missed', target_solver_seconds=missed_target,
                                  actual_solver_seconds=info['solver_seconds'],
                                  iteration=info['completed_iterations'],
                                  native_completed_updates=info['completed_updates'],
                                  reason='same_completed_iteration_already_used_for_earlier_target'))

    captured_warnings = []
    try:
        # A shared diagnostic at the known native product start. The native
        # function still performs and times its own unchanged initialization.
        evaluate(torch.outer(problem.a, problem.b), 'initial', None, last_info)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            try:
                native_result, report = sampled_gromov_wasserstein(
                    problem.A, problem.B, problem.a, problem.b,
                    loss_fun=lambda x, y: (x - y) ** 2,
                    nb_samples_grad=args.batch, epsilon=args.epsilon,
                    max_iter=args.max_iter, log=False, verbose=False,
                    random_state=args.seed, callback=callback, time_budget=args.budget,
                    archive_dir=out / 'native_source')
            finally:
                captured_warnings.extend(dict(category=type(item.message).__name__,
                                              message=str(item.message), filename=item.filename,
                                              lineno=item.lineno) for item in recorded)
        returned_plan = native_result if isinstance(native_result, torch.Tensor) else torch.as_tensor(native_result, dtype=dtype, device=device)
        final_info = dict(completed_iterations=int(report['completed_iterations']),
                          completed_updates=int(report['completed_updates']),
                          solver_seconds=float(report['solver_seconds']),
                          updated_transport=last_info['updated_transport'])
        sync(device)
        started = time.perf_counter()
        _assert_plan(returned_plan)
        same_output = (last_evaluated_iteration == final_info['completed_iterations']
                       and last_evaluated_raw is not None and torch.equal(returned_plan, last_evaluated_raw))
        sync(device)
        diagnostic_seconds += time.perf_counter() - started
        if not same_output:
            evaluate(returned_plan, 'final', None, final_info)
        # The final solver clock can include epilogue work after the last callback.
        # This does not create a new completed iteration or backfill a checkpoint.
        last_info = final_info
        sync(device)
        started = time.perf_counter()
        problem64 = GWProblem(*(torch.as_tensor(canonical[key], dtype=torch.float64, device=device)
                                 for key in ('A', 'B', 'a', 'b')))
        raw64, rounded64 = returned_plan.double(), last_rounded.double()
        _assert_plan(raw64)
        _assert_plan(rounded64)
        final_validation = dict(dtype='float64', device=str(device),
                                raw_loss=problem64.loss(raw64).item(), rounded_loss=problem64.loss(rounded64).item(),
                                raw=marginal_residuals(raw64, problem64.a, problem64.b),
                                rounded=marginal_residuals(rounded64, problem64.a, problem64.b),
                                plan='actual native returned plan and saved rounded output promoted without re-rounding',
                                loss_calls=2)
        if not math.isfinite(final_validation['raw_loss']) or not math.isfinite(final_validation['rounded_loss']):
            raise FloatingPointError('Native final float64 objective validation failed.')
        if final_validation['rounded']['both_l1'] > 100 * torch.finfo(dtype).eps:
            raise FloatingPointError('Native final float64 marginal validation failed.')
        sync(device)
        validation_seconds = time.perf_counter() - started
        status = 'completed_with_warnings' if captured_warnings else 'completed'
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        failure_events.append(dict(type='run_rejected', error=error, **last_info))
    for target in grid.remaining():
        trace.append(dict(kind='missed', target_solver_seconds=target,
                          actual_solver_seconds=last_info['solver_seconds'],
                          iteration=last_info['completed_iterations'],
                          native_completed_updates=last_info['completed_updates'],
                          reason='native_stopped_before_checkpoint_or_no_new_completed_iteration'))
    metadata.update(status=status, error=error, instrumentation=report,
                    counters=dict(completed_iterations=last_info['completed_iterations'],
                                  completed_updates=last_info['completed_updates'],
                                  inner_iterations=None, inner_stopping_failures=None),
                    diagnostic_counts=diagnostic_counts,
                    final_float64_validation=final_validation,
                    warnings=captured_warnings,
                    memory=dict(cpu_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                cpu_peak_scope='process lifetime including input preparation and validation',
                                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None),
                    timing=dict(preparation_seconds=preparation_seconds,
                                solver_seconds=last_info['solver_seconds'], repair_seconds=repair_seconds,
                                exact_diagnostic_seconds=diagnostic_seconds,
                                final_float64_validation_seconds=validation_seconds),
                    timing_scope='module solver time excludes its copies and callbacks; callback times contain runner repair/diagnostics, so do not add them twice',
                    numerical_policy='raw negatives in loss retained; negative/nonfinite plans rejected; native warnings and checkpoint support zeros preserved',
                    failure_timing_complete=status != 'failed',
                    failure_timing_note='on failure solver_seconds is the last completed callback clock; total elapsed still includes rejected work')
    if report is not None:
        metadata['timing']['native_instrumentation_preparation_seconds'] = report['instrumentation_preparation_seconds']
        metadata['timing']['native_callback_seconds_overlapping_repair_and_diagnostics'] = report['callback_seconds']
    sync(device)
    started = time.perf_counter()
    _write_csv(out / 'trace.csv', trace)
    (out / 'failures.json').write_text(json.dumps(_json_safe(failure_events), indent=2, allow_nan=False))
    if status != 'failed':
        np.savez(out / 'final.npz', P_raw=returned_plan.detach().cpu().numpy(),
                 P_rounded=last_rounded.detach().cpu().numpy())
    elif last_evaluated_raw is not None:
        np.savez(out / 'last_observed_diagnostic.npz', P_last_observed=last_evaluated_raw.detach().cpu().numpy())
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for name in ('run_native_sampled.py', 'native_sampled_gw.py', 'run_benchmark.py', 'gw_core.py', 'smoothing_fw.py'):
        shutil.copy2(ROOT / name, snapshot / name)
    sync(device)
    metadata['timing']['serialization_seconds'] = time.perf_counter() - started
    metadata['timing']['total_elapsed_seconds_before_final_metadata_write'] = time.perf_counter() - total_start
    metadata['timing']['unclassified_instrumentation_and_python_seconds'] = (
        metadata['timing']['total_elapsed_seconds_before_final_metadata_write']
        - sum(metadata['timing'][key] for key in ('preparation_seconds', 'solver_seconds', 'repair_seconds',
              'exact_diagnostic_seconds', 'final_float64_validation_seconds', 'serialization_seconds'))
        - (report['instrumentation_preparation_seconds'] if report is not None else 0.))
    (out / 'metadata.json').write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False))
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance', required=True)
    p.add_argument('--out', required=True, help='New output directory; existing runs are refused.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--epsilon', type=float, default=.1)
    p.add_argument('--budget', type=float, required=True)
    p.add_argument('--checkpoints', help='Comma-separated seconds <= budget; standard grid is truncated to budget by default.')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--init-seed', type=int, default=0, help='Recorded only: native product initialization consumes no RNG.')
    p.add_argument('--max-iter', type=int, default=1000000)
    p.add_argument('--gap-max-n', type=int, default=512)
    return p


def main():
    args = parser().parse_args()
    result = run(args)
    print(json.dumps(dict(status=result['status'], iterations=result['counters']['completed_iterations'],
                          solver_seconds=result['timing']['solver_seconds'], error=result['error'],
                          out=str(Path(args.out).resolve())), indent=2))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
