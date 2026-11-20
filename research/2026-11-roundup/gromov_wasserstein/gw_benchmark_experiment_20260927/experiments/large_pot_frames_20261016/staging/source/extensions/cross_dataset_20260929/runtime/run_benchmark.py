"""Practical fixed-time GW comparisons on identical frozen arrays.

This harness provides current feasible outputs, never an uncounted best-of-run
selection. Native Sampled GW uses its separate product-start runner and is omitted
from randomized-start comparisons because its API has no G0 argument. Constants and oracle selection must be frozen
on development data before interpreting a held-out comparison.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from gw_core import (GWProblem, assignment_lmo, kl_prox, marginal_residuals,
                     project_transport_dykstra, rademacher, round_transport)
from smoothing_fw import MomentumConfig, initialize_state, step
from measurements import load_correspondence, correspondence_metrics

ROOT = Path(__file__).resolve().parent
METHODS = ('smoothing-fw', 'smoothing-row-projection', 'full-kl', 'stochastic-kl',
           'stochastic-kl-momentum', 'fw', 'pg')


class CheckpointGrid:
    """One output for the earliest crossed target; remaining crossed windows miss."""
    def __init__(self, targets):
        self.targets = tuple(float(t) for t in targets)
        if not self.targets or any(not math.isfinite(t) or t <= 0 for t in self.targets):
            raise ValueError('Checkpoint targets must be finite and positive.')
        if list(self.targets) != sorted(set(self.targets)):
            raise ValueError('Checkpoint targets must be strictly increasing.')
        self.next = 0

    def consume(self, completed_solver_seconds):
        crossed = []
        while self.next < len(self.targets) and self.targets[self.next] <= completed_solver_seconds:
            crossed.append(self.targets[self.next])
            self.next += 1
        return (crossed[0], crossed[1:]) if crossed else (None, [])

    def remaining(self):
        return list(self.targets[self.next:])


def resolve_device(name):
    """Resolve unsuffixed CUDA to logical device zero inside Slurm visibility."""
    device = torch.device(name)
    return torch.device('cuda', 0) if device.type == 'cuda' and device.index is None else device


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _command(arguments):
    try:
        result = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True,
                                timeout=15, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment(device):
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    packages = {}
    for package in ('numpy', 'scipy', 'torch', 'POT'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata = dict(python=sys.version, platform=platform.platform(), hostname=platform.node(),
                    cpu_model=cpu, torch_num_threads=torch.get_num_threads(),
                    torch_num_interop_threads=torch.get_num_interop_threads(),
                    thread_environment={key: os.environ.get(key) for key in
                                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                    packages=packages, cuda_runtime=torch.version.cuda,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    float32_matmul_precision=torch.get_float32_matmul_precision(),
                    cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    amp=False, device=str(device),
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        metadata['gpu'] = dict(name=properties.name, total_memory_bytes=properties.total_memory,
                               capability=[properties.major, properties.minor])
        metadata['nvidia_smi'] = _command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                           '--format=csv,noheader'])
    return metadata


def _assert_plan(P):
    if not torch.isfinite(P).all():
        raise FloatingPointError('Nonfinite coupling rejected; no feasible output is claimed.')
    if (P < 0).any():
        raise FloatingPointError('Negative coupling rejected; no silent clipping is permitted.')


def make_initial_plan(problem, initialization, init_seed):
    """Protocol section 12 start; reads only marginals, never costs or labels.

    The private NumPy generator is independent of the Torch oracle generator. It
    also gives the same permutation on CPU and CUDA. Every practical method calls
    this constructor with the same initialization mode/seed before its first step.
    """
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or not 0 <= init_seed < 2**63:
        raise ValueError('init_seed must be an explicit integer in [0,2**63).')
    product = torch.outer(problem.a, problem.b)
    if initialization == 'product':
        return product, {'mode': 'product', 'formula': 'P0=a*b^T',
                         'rng': None, 'independent_permutation': None,
                         'seed_used_for_randomness': False}
    if initialization != 'randomized':
        raise ValueError('initialization must be product or randomized.')
    n, m = product.shape
    if n != m or not torch.equal(problem.a, torch.full_like(problem.a, 1/n)) or not torch.equal(problem.b, torch.full_like(problem.b, 1/n)):
        raise ValueError('Randomized permutation starts require square uniform marginals.')
    init_rng = np.random.default_rng(init_seed)
    permutation = init_rng.permutation(n)
    P0 = .9 * product
    P0[torch.arange(n, device=P0.device), torch.as_tensor(permutation, device=P0.device)] += .1/n
    if not torch.isfinite(P0).all() or (P0 <= 0).any():
        raise ValueError('The randomized initial coupling is not finite and strictly positive.')
    tolerance = 32 * torch.finfo(P0.dtype).eps
    if (P0.sum(1)-problem.a).abs().sum().item() > tolerance or (P0.sum(0)-problem.b).abs().sum().item() > tolerance:
        raise ValueError('The randomized initial coupling failed both-marginal feasibility validation.')
    return P0, {'mode': 'randomized', 'formula': 'P0=0.9*a*b^T+0.1*R/n',
                'rng': 'numpy.default_rng/'+type(init_rng.bit_generator).__name__,
                'independent_permutation': permutation.tolist(),
                'seed_used_for_randomness': True,
                'marginal_l1_tolerance': tolerance,
                'permutation_definition': 'R[i,independent_permutation[i]]=1; independent of frozen costs, planted matching, and oracle samples'}


def _kl_momentum_gradient(problem, P, previous, generator, rho, batch):
    """One fresh smooth-gradient sample, with the same EMA as smoothing_fw.

    previous=None initializes d0 directly. The runner reuses d0 on update zero,
    so this helper is called only once per new oracle point. No marginal penalty
    enters the KL comparator. Timing and call/probe counters belong to the runner.
    """
    fresh = problem.sketch_gradient(P, rademacher(P.shape[1], batch,
                                                 like=P, generator=generator))
    if not torch.isfinite(fresh).all():
        raise FloatingPointError('Nonfinite gradient rejected.')
    return fresh if previous is None else (1 - rho) * previous + rho * fresh


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    total_start = time.perf_counter()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.budget <= 0 or not math.isfinite(args.budget):
        raise ValueError('budget must be finite and positive.')
    supplied_grid = ([float(value) for value in args.checkpoints.split(',')]
                     if args.checkpoints is not None else
                     [value for value in (1., 3., 10., 30., 100., 300.) if value <= args.budget])
    if any(value > args.budget for value in supplied_grid):
        raise ValueError('All checkpoint targets must be at most the solver-time budget.')
    grid = CheckpointGrid(sorted(set(supplied_grid + [args.budget])))
    if args.threads < 1 or args.batch < 1 or args.inner_max < 1 or args.inner_tol <= 0:
        raise ValueError('Positive threads, batch, inner-max, and inner-tol are required.')
    if args.eta <= 0 or not math.isfinite(args.eta) or args.eta_power < 0:
        raise ValueError('eta must be finite and positive, and eta-power nonnegative.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dtype = getattr(torch, args.dtype)
    instance = Path(args.instance).resolve()
    with np.load(instance, allow_pickle=False) as archive:
        canonical = {key: np.array(archive[key], copy=True) for key in ('A', 'B', 'a', 'b')}
    hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in canonical.items()}
    manifest = json.loads(instance.with_suffix('.json').read_text()) if instance.with_suffix('.json').exists() else None
    if manifest:
        for key in ('A', 'B'):
            if manifest.get(key + '_sha256') != hashes[key]:
                raise ValueError(f'The frozen {key} array does not match its manifest hash.')
    cast = lambda value: torch.as_tensor(value, dtype=dtype, device=device)
    problem = GWProblem(*(cast(canonical[key]) for key in ('A', 'B', 'a', 'b')))
    n, m = len(problem.a), len(problem.b)
    correspondence = load_correspondence(args.correspondence_file, args.correspondence_sha256, n, m, device)
    if args.method == 'pg' and max(n, m) > args.pg_max_n:
        raise ValueError('Euclidean PG is restricted to small instances by --pg-max-n.')
    if args.method == 'fw' and (n != m or not np.allclose(canonical['a'], 1 / n)
                                or not np.allclose(canonical['b'], 1 / n)):
        raise ValueError('Exact assignment FW requires square uniform marginals.')
    theory = json.loads((ROOT / 'configs/theory_config.json').read_text())
    smooth_config = None
    if args.method.startswith('smoothing-'):
        mu0 = args.mu0 if args.mu0 is not None else args.mu_scale / m
        smooth_config = MomentumConfig(
            alpha=theory['alpha'], beta=theory['beta'], delta=theory['delta'],
            gamma0=args.gamma0 if args.gamma0 is not None else theory['gamma0'],
            rho0=args.rho0 if args.rho0 is not None else theory['rho0'], mu0=mu0,
            oracle=args.oracle, batch=args.batch,
            update='row_projection' if args.method == 'smoothing-row-projection' else 'fw',
            eta0=args.eta if args.method == 'smoothing-row-projection' else None,
            eta_exponent=args.eta_power if args.method == 'smoothing-row-projection' else None,
            theory_id=theory['theorem_identifier'], output_statistic=theory['proved_primary_statistic'])
    kl_momentum_config = None
    if args.method == 'stochastic-kl-momentum':
        if args.oracle != 'sketch' or args.batch != 128:
            raise ValueError('stochastic-kl-momentum requires the frozen sketch oracle with batch 128.')
        if args.rho0 is None or not math.isfinite(args.rho0) or not 0 < args.rho0 <= 1:
            raise ValueError('stochastic-kl-momentum requires explicit selected smoothing rho0 in (0,1].')
        kl_momentum_config = dict(
            rho0=args.rho0, beta=theory['beta'], rho_formula='rho0/(k+1)**beta',
            indexing='k=0,1,...', oracle='sketch', batch=128,
            initialization='d0=first smooth gradient estimate at P0; reused for update zero',
            recurrence='d_k=(1-rho_k)*d_(k-1)+rho_k*g_k(P_k), k>=1',
            penalty='none; full-transport KL prox uses only the smooth-gradient momentum',
            parameter_policy='independent equal-budget grid for each method and instance; shared fixed momentum schedule; see frozen campaign selection',
            theorem_claim='none; this matched KL ablation is not the smoothing theorem algorithm')
    metadata = dict(config=vars(args), instance=str(instance), instance_sha256=sha256_file(instance),
                    array_sha256=hashes, instance_manifest=manifest, environment=environment(device),
                    source_commit=_command(['git', 'rev-parse', 'HEAD']),
                    source_dirty=bool(_command(['git', 'status', '--porcelain'])),
                    source_file_sha256={path.name: sha256_file(path) for path in
                                       (ROOT / 'run_benchmark.py', ROOT / 'smoothing_fw.py', ROOT / 'gw_core.py', ROOT / 'measurements.py')},
                    initialization=('product a*b^T; initialization_seed is recorded but consumes no randomness'
                                    if args.initialization == 'product' else
                                    'P0=0.9*a*b^T+0.1*R/n; independent permutation drawn only from initialization_seed'),
                    initialization_seed=args.init_seed, stochastic_seed=args.seed,
                    oracle_selection_status=args.oracle_selection_status,
                    theory_config=theory, smoothing_config=(smooth_config.metadata(
                        initialization='provided' if args.initialization == 'randomized' else 'product')
                        if smooth_config else None),
                    mu0=smooth_config.mu0 if smooth_config else None,
                    m_mu0=m * smooth_config.mu0 if smooth_config else None,
                    momentum_error_policy=('same-point error at retained last-update input P_(k-1), or P0 initially; '
                                          'point indices distinguish it from the saved current output P_k; '
                                          'stored-momentum/current-output tracking error is separately labeled'
                                          if smooth_config else 'not applicable without a momentum state'),
                    output_selection='current output; no best-so-far selection',
                    checkpoint_policy='first completed iterate crossing earliest target; additional crossed windows marked missed',
                    timing_policy='synchronize CUDA around algorithm, repair, diagnostics, and validation; all update checks and transfers included',
                    native_sampled_gw=('omitted from randomized-start panel: native API has no G0 argument'
                                       if args.initialization == 'randomized' else
                                       'not included here; use the separate native product-start runner'))
    metadata['correspondence_sha256'] = args.correspondence_sha256
    metadata['correspondence_policy'] = 'withheld from solver updates and parameter selection; measured only on current rounded checkpoint outputs'
    if kl_momentum_config:
        metadata['momentum_config'] = kl_momentum_config
        metadata['momentum_error_policy'] = (
            'same-point error at stored last-update input P_(k-1), or P0 initially; '
            'momentum_point_iteration labels this point separately from current output P_k')
    sync(device)
    preparation_seconds = time.perf_counter() - total_start
    counters = dict(oracle_evaluations=0, exact_gradient_calls=0, stochastic_gradient_calls=0,
                    sketch_probes=0, anchor_samples=0, initialization_oracle_calls=0,
                    row_lmo_calls=0, row_projection_calls=0, transport_lmo_calls=0,
                    kl_projection_calls=0, euclidean_projection_calls=0,
                    inner_iterations_total=0, inner_stopping_failures=0,
                    underflow_steps=0, underflow_entries_total=0, underflow_entries_max=0,
                    attempted_updates=0, completed_updates=0)
    if kl_momentum_config:
        counters.update(momentum_initializations=0, momentum_ema_updates=0)
    diagnostic_counts = dict(loss_calls=0, full_gradient_calls=0, assignment_lmo_calls=0)
    trace, failure_events = [], []
    solver_seconds = repair_seconds = diagnostic_seconds = validation_seconds = 0.
    P = logP = state = final_R = None
    smooth_momentum_point = None
    smooth_momentum_iteration = None
    kl_momentum = kl_momentum_point = None
    kl_momentum_iteration = None
    final_evaluated_iteration = None
    final_validation = None
    status, error = 'running', None

    def evaluate(kind, target, iteration):
        nonlocal repair_seconds, diagnostic_seconds, final_R, final_evaluated_iteration
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
        tolerance = 100 * torch.finfo(dtype).eps
        if rounded_residuals['both_l1'] > tolerance:
            raise FloatingPointError(f'Rounded output failed marginal validation: {rounded_residuals}')
        raw_loss, rounded_loss = problem.loss(P).item(), problem.loss(R).item()
        diagnostic_counts['loss_calls'] += 2
        if not math.isfinite(raw_loss) or not math.isfinite(rounded_loss):
            raise FloatingPointError('Nonfinite exact objective rejected.')
        row = dict(kind=kind, target_solver_seconds=target, actual_solver_seconds=solver_seconds,
                   iteration=iteration, raw_loss=raw_loss, rounded_loss=rounded_loss,
                   repair_l1=(R - P).abs().sum().item(), repair_seconds_current=repair_current,
                   repair_seconds_cumulative=repair_seconds,
                   time_to_this_feasible_output=solver_seconds + repair_current,
                   **raw_residuals,
                   **{'rounded_' + key: value for key, value in rounded_residuals.items()})
        if correspondence is not None:
            row.update(correspondence_metrics(R, correspondence))
        if smooth_config:
            mu = smooth_config.schedule(iteration)['mu']
            G = problem.gradient(P)
            diagnostic_counts['full_gradient_calls'] += 1
            H = G + (P.sum(0) - problem.b)[None, :] / mu
            row.update(mu=mu, smoothing_point_iteration=iteration,
                       true_smoothed_gap=((H * P).sum() - (problem.a * H.min(1).values).sum()).item())
            if smooth_momentum_iteration == iteration:
                exact_at_momentum_point = G
            else:
                exact_at_momentum_point = problem.gradient(smooth_momentum_point)
                diagnostic_counts['full_gradient_calls'] += 1
            row.update(
                momentum_point_iteration=smooth_momentum_iteration,
                momentum_error_matches_current_output=(smooth_momentum_iteration == iteration),
                momentum_error_frobenius=(state.d - exact_at_momentum_point).norm().item(),
                stored_momentum_current_output_tracking_error_frobenius=(state.d - G).norm().item(),
                stored_momentum_iteration=smooth_momentum_iteration,
                tracking_error_point_iteration=iteration)
        if kl_momentum_config:
            # d_(k-1) and the retained input P_(k-1) match. The current output
            # remains P_k; never label this error as a diagnostic at P_k.
            exact_at_momentum_point = problem.gradient(kl_momentum_point)
            diagnostic_counts['full_gradient_calls'] += 1
            row.update(momentum_point_iteration=kl_momentum_iteration,
                       momentum_error_matches_current_output=(kl_momentum_iteration == iteration),
                       momentum_error_frobenius=(kl_momentum - exact_at_momentum_point).norm().item())
        if n == m and max(n, m) <= args.gap_max_n:
            G = problem.gradient(R)
            S = assignment_lmo(G, problem.a, problem.b)
            diagnostic_counts['full_gradient_calls'] += 1
            diagnostic_counts['assignment_lmo_calls'] += 1
            row['original_fw_gap'] = (G * (R - S)).sum().item()
        sync(device)
        diagnostic_seconds += time.perf_counter() - started
        if smooth_config and iteration > 0:
            previous_schedule = smooth_config.schedule(iteration - 1)
            row.update({'last_update_' + key: value for key, value in previous_schedule.items()})
        elif iteration > 0 and args.method in ('full-kl', 'stochastic-kl', 'stochastic-kl-momentum', 'pg'):
            row['last_update_eta'] = args.eta / iteration ** args.eta_power
        if kl_momentum_config and iteration > 0:
            row['last_update_rho'] = args.rho0 / iteration ** kl_momentum_config['beta']
            row['last_update_effective_momentum_weight'] = 1. if iteration == 1 else row['last_update_rho']
        row.update({'cumulative_' + key: value for key, value in counters.items()})
        row['diagnostic_seconds_cumulative'] = diagnostic_seconds
        trace.append(row)
        final_R, final_evaluated_iteration = R, iteration

    try:
        # P0 construction, RNG setup, KL log-state, and the first momentum gradient
        # are all solver work. Distance-square caching occurred in preparation.
        sync(device)
        started = time.perf_counter()
        P0, initialization_details = make_initial_plan(problem, args.initialization, args.init_seed)
        if smooth_config:
            state = initialize_state(problem, smooth_config, seed=args.seed, P0=P0)
            P = state.P
            smooth_momentum_point, smooth_momentum_iteration = P, 0
            counters.update(asdict(state.counts))
        else:
            P = P0
            generator = torch.Generator(device=device).manual_seed(args.seed)
            if args.method in ('full-kl', 'stochastic-kl', 'stochastic-kl-momentum'):
                logP = P.log()
            if kl_momentum_config:
                counters['oracle_evaluations'] += 1
                counters['stochastic_gradient_calls'] += 1
                counters['sketch_probes'] += args.batch
                counters['initialization_oracle_calls'] += 1
                kl_momentum = _kl_momentum_gradient(problem, P, None, generator, args.rho0, args.batch)
                kl_momentum_point, kl_momentum_iteration = P, 0
                counters['momentum_initializations'] += 1
        sync(device)
        solver_seconds += time.perf_counter() - started
        # Reproducibility auditing is diagnostic work, not hidden solver work.
        # The actual initial plan hash lets all matched methods verify P0 equality.
        started = time.perf_counter()
        metadata['initialization_details'] = initialization_details
        metadata['initialization_plan_sha256'] = hashlib.sha256(P.detach().cpu().numpy().tobytes()).hexdigest()
        metadata['initialization_residuals'] = marginal_residuals(P, problem.a, problem.b)
        sync(device)
        diagnostic_seconds += time.perf_counter() - started
        del P0
        evaluate('initial', None, 0)
        while counters['completed_updates'] == 0 or solver_seconds < args.budget:
            counters['attempted_updates'] += 1
            sync(device)
            started = time.perf_counter()
            try:
                k = counters['completed_updates']
                info = {}
                if smooth_config:
                    result = step(problem, state)
                    P = state.P
                    # Retain only the latest original input. Step never mutates
                    # these tensors; diagnostics neither sample nor advance state.
                    smooth_momentum_point = result.P_before
                    smooth_momentum_iteration = result.update_index
                    counters.update(asdict(state.counts))
                    del result
                else:
                    if kl_momentum_config:
                        if k > 0:
                            counters['oracle_evaluations'] += 1
                            counters['stochastic_gradient_calls'] += 1
                            counters['sketch_probes'] += args.batch
                            rho = args.rho0 / (k + 1) ** kl_momentum_config['beta']
                            kl_momentum = _kl_momentum_gradient(problem, P, kl_momentum, generator, rho, args.batch)
                            kl_momentum_point, kl_momentum_iteration = P, k
                            counters['momentum_ema_updates'] += 1
                        # Initialization already sampled d0; update zero reuses it.
                        G = kl_momentum
                    else:
                        counters['oracle_evaluations'] += 1
                        if args.method == 'stochastic-kl':
                            counters['stochastic_gradient_calls'] += 1
                            if args.oracle == 'sketch':
                                counters['sketch_probes'] += args.batch
                                G = problem.sketch_gradient(P, rademacher(m, args.batch, like=P, generator=generator))
                            elif args.oracle == 'anchor':
                                counters['anchor_samples'] += args.batch
                                flat = torch.randint(n * m, (args.batch,), device=device, generator=generator)
                                G = problem.anchor_gradient(P, torch.stack((flat // m, flat % m), dim=1))
                            else:
                                raise ValueError('stochastic-kl requires sketch or anchor oracle.')
                        else:
                            counters['exact_gradient_calls'] += 1
                            G = problem.gradient(P)
                    if not torch.isfinite(G).all():
                        raise FloatingPointError('Nonfinite gradient rejected.')
                    eta = args.eta / (k + 1) ** args.eta_power
                    if args.method == 'fw':
                        counters['transport_lmo_calls'] += 1
                        S = assignment_lmo(G, problem.a, problem.b)
                        D = S - P
                        linear = (G * D).sum().item()
                        permutation = S.argmax(1)
                        fS = (problem.A - problem.B[permutation][:, permutation]).square().mean().item()
                        fP = .5 * (G * P).sum().item()
                        quadratic = fS - fP - linear
                        gamma = min(1., max(0., -linear / (2 * quadratic))) if quadratic > 0 else float(linear + quadratic < 0)
                        P = P + gamma * D
                    elif args.method in ('full-kl', 'stochastic-kl', 'stochastic-kl-momentum'):
                        counters['kl_projection_calls'] += 1
                        P, logP, info = kl_prox(logP, G, problem.a, problem.b, eta=eta,
                                               tol=args.inner_tol, max_iter=args.inner_max)
                    else:
                        counters['euclidean_projection_calls'] += 1
                        P, info = project_transport_dykstra(P - eta * G, problem.a, problem.b,
                                                           gap_tol=args.inner_tol, max_iter=args.inner_max)
                    counters['inner_iterations_total'] += info.get('inner_iterations', 0)
                    underflow = info.get('underflow_entries', 0)
                    counters['underflow_steps'] += int(underflow > 0)
                    counters['underflow_entries_total'] += underflow
                    counters['underflow_entries_max'] = max(counters['underflow_entries_max'], underflow)
                    if info.get('converged') is False:
                        counters['inner_stopping_failures'] += 1
                        failure_events.append(dict(iteration=k + 1, type='inner_tolerance_not_attained', **info))
                    if underflow:
                        failure_events.append(dict(iteration=k + 1, type='exponentiation_underflow_logstate_retained', **info))
                _assert_plan(P)
                counters['completed_updates'] += 1
            finally:
                sync(device)
                solver_seconds += time.perf_counter() - started
            target, missed = grid.consume(solver_seconds)
            if target is not None:
                evaluate('checkpoint', target, counters['completed_updates'])
                for missed_target in missed:
                    trace.append(dict(kind='missed', target_solver_seconds=missed_target,
                                      actual_solver_seconds=solver_seconds, iteration=counters['completed_updates'],
                                      reason='same_completed_iterate_already_used_for_earlier_target'))
        if final_evaluated_iteration != counters['completed_updates']:
            evaluate('final', None, counters['completed_updates'])
        status = 'completed_with_inner_failures' if counters['inner_stopping_failures'] else 'completed'
        # Promote the canonical stored costs and the actual output; do not regenerate
        # geometry or silently round a second plan before validation.
        sync(device)
        started = time.perf_counter()
        validation_problem = GWProblem(*(torch.as_tensor(canonical[key], dtype=torch.float64, device=device)
                                         for key in ('A', 'B', 'a', 'b')))
        raw64, rounded64 = P.double(), final_R.double()
        final_validation = dict(dtype='float64', device=str(device),
                                raw_loss=validation_problem.loss(raw64).item(),
                                rounded_loss=validation_problem.loss(rounded64).item(),
                                raw=marginal_residuals(raw64, validation_problem.a, validation_problem.b),
                                rounded=marginal_residuals(rounded64, validation_problem.a, validation_problem.b),
                                plan='same saved raw and rounded plans, promoted without re-rounding',
                                loss_calls=2)
        _assert_plan(rounded64)
        if final_validation['rounded']['both_l1'] > 100 * torch.finfo(dtype).eps:
            raise FloatingPointError('Final rounded plan failed float64 marginal validation.')
        if not math.isfinite(final_validation['raw_loss']) or not math.isfinite(final_validation['rounded_loss']):
            raise FloatingPointError('Final float64 objective validation failed.')
        sync(device)
        validation_seconds = time.perf_counter() - started
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        failure_events.append(dict(iteration=counters['completed_updates'], type='run_rejected', error=error))
    for target in grid.remaining():
        trace.append(dict(kind='missed', target_solver_seconds=target, actual_solver_seconds=solver_seconds,
                          iteration=counters['completed_updates'], reason='run_ended_before_target'))
    metadata.update(status=status, error=error, counters=counters, diagnostic_counts=diagnostic_counts,
                    final_float64_validation=final_validation,
                    memory=dict(cpu_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                cpu_peak_scope='process lifetime, includes preparation and float64 validation',
                                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None),
                    timing=dict(preparation_seconds=preparation_seconds, solver_seconds=solver_seconds,
                                repair_seconds=repair_seconds, exact_diagnostic_seconds=diagnostic_seconds,
                                final_float64_validation_seconds=validation_seconds),
                    numerical_policy='negative losses retained without clamping; negative/nonfinite plans rejected',
                    failure_accounting_complete=status != 'failed')
    sync(device)
    serialization_start = time.perf_counter()
    _write_csv(out / 'trace.csv', trace)
    (out / 'failures.json').write_text(json.dumps(_json_safe(failure_events), indent=2, allow_nan=False))
    if P is not None:
        arrays = {'P_raw': P.detach().cpu().numpy()}
        if status != 'failed' and final_R is not None and final_evaluated_iteration == counters['completed_updates']:
            arrays['P_rounded'] = final_R.detach().cpu().numpy()
        np.savez(out / 'final.npz', **arrays)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for name in ('run_benchmark.py', 'smoothing_fw.py', 'gw_core.py', 'measurements.py'):
        shutil.copy2(ROOT / name, snapshot / name)
    shutil.copy2(ROOT / 'configs/theory_config.json', snapshot / 'theory_config.json')
    sync(device)
    metadata['timing']['serialization_seconds'] = time.perf_counter() - serialization_start
    metadata['timing']['total_elapsed_seconds_before_final_metadata_write'] = time.perf_counter() - total_start
    metadata['timing']['unclassified_python_overhead_seconds'] = (
        metadata['timing']['total_elapsed_seconds_before_final_metadata_write']
        - sum(metadata['timing'][name] for name in ('preparation_seconds', 'solver_seconds', 'repair_seconds',
              'exact_diagnostic_seconds', 'final_float64_validation_seconds', 'serialization_seconds')))
    (out / 'metadata.json').write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False))
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance', required=True)
    p.add_argument('--correspondence-file', help='Optional withheld correspondence NPZ, only used for output diagnostics.')
    p.add_argument('--correspondence-sha256', help='Required SHA256 when correspondence-file is supplied.')
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--out', required=True, help='New output directory; existing runs are never overwritten.')
    p.add_argument('--budget', type=float, required=True, help='Algorithm wall-clock seconds, excluding diagnostics.')
    p.add_argument('--checkpoints', help='Comma-separated solver seconds <= budget; default uses 1,3,10,30,100,300 within budget.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, required=True, help='Stochastic seed; never used for evaluation.')
    p.add_argument('--init-seed', type=int, required=True, help='Independent permutation seed; product initialization uses no RNG.')
    p.add_argument('--initialization', choices=('product', 'randomized'), default='product',
                   help='Product a*b^T or protocol start 0.9*a*b^T+0.1*R/n; native Sampled GW is omitted from randomized panels.')
    p.add_argument('--oracle', choices=('exact', 'sketch', 'anchor'), default='sketch')
    p.add_argument('--oracle-selection-status', default='development_candidate_not_frozen')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--mu0', type=float)
    p.add_argument('--mu-scale', type=float, default=1., help='m*mu0 when --mu0 is absent; practical constant.')
    p.add_argument('--gamma0', type=float)
    p.add_argument('--rho0', type=float)
    p.add_argument('--eta', type=float, default=1.)
    p.add_argument('--eta-power', type=float, default=0.)
    p.add_argument('--inner-tol', type=float, default=1e-8)
    p.add_argument('--inner-max', type=int, default=5000)
    p.add_argument('--gap-max-n', type=int, default=512)
    p.add_argument('--pg-max-n', type=int, default=128)
    return p


def main():
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({'status': result['status'], 'updates': result['counters']['completed_updates'],
                      'solver_seconds': result['timing']['solver_seconds'], 'out': str(Path(args.out).resolve()),
                      'error': result['error']}, indent=2))
    if result['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
