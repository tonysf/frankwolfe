"""Fixed-horizon, row-sharded stochastic GW comparison with certified KL solves.

SMD follows the fresh-gradient entropic specialization of Fatkhullin--He
(AISTATS 2024), equation (2); its independently drawn P_R is distinct from
P_T and from any practical current-iterate diagnostic. Finite inner tolerances
are numerical choices, not an inexact convergence theorem. GPU execution is
restricted to an allocated Slurm job by the existing setup helper.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
HISTORICAL_ROOT = ROOT.parent / 'shareseq_25000_20260930'
sys.path.insert(0, str(HISTORICAL_ROOT))
from run_sharded import (Clock, initialization, load_problem, memory,
                         setup, sha256, write_json)
from certified_prox import CertifiedKLProx
from smd import (cost_bound_from_extrema, output_probabilities, select_output_index,
                 smd_step)

METHOD_NAMES = {
    'published-smd': 'Stochastic entropic mirror descent (fresh sketch gradient; no momentum)',
    'smoothing-fw': 'Smoothed row Frank-Wolfe with gradient momentum (sketch gradient)',
}
ARRAY_NAMES = ('A.npy', 'B.npy', 'a.npy', 'b.npy')


def assert_plan(problem, plan):
    problem.require_finite(plan, 'coupling')
    if problem.global_min(plan).item() < 0:
        raise FloatingPointError('Negative coupling rejected.')


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def parser():
    defaults = json.loads((ROOT/'inner_solver.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance', '--data-dir', dest='instance', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=list(METHOD_NAMES), required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--dtype', choices=('float32', 'float64'), default='float32')
    p.add_argument('--iterations', type=int, required=True, help='Fixed horizon, independent of iterates and timing.')
    p.add_argument('--seed', '--probe-seed', dest='seed', type=int, required=True)
    p.add_argument('--output-seed', type=int, required=True)
    p.add_argument('--init-seed', type=int, default=0)
    p.add_argument('--initialization', choices=('product', 'randomized'), default='product')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--prefactor-fraction', type=float, default=1.)
    p.add_argument('--rho0', type=float, default=.3)
    p.add_argument('--gamma0', type=float, default=.1)
    p.add_argument('--mu-scale', type=float, default=.1)
    p.add_argument('--inner-tol', type=float, default=defaults['summed_marginal_l1_tolerance'])
    p.add_argument('--inner-gap-tol', type=float, default=defaults['normalized_primal_dual_gap_tolerance'])
    p.add_argument('--inner-max', type=int, default=defaults.get('maximum_sinkhorn_iterations', 10000))
    p.add_argument('--check-every', type=int, default=defaults.get('check_every', 10))
    p.add_argument('--certificate-block-rows', type=int, default=256)
    p.add_argument('--exact-block-rows', type=int, default=256)
    p.add_argument('--save-plans', choices=('all', 'none'), default='all',
                   help='Persist dense selected/last plan shards, or retain only hashes and validation results.')
    p.add_argument('--progress-every', type=int, default=100,
                   help='Write progress after this many completed updates; never changes the fixed horizon.')
    p.add_argument('--checkpoint-iterations', default='', help='Additional current-iterate diagnostics; never select output.')
    p.add_argument('--checkpoints', default='', help='Solver-second diagnostic thresholds; do not determine horizon.')
    return p


def validate_args(args):
    for name in ('iterations', 'inner_max', 'check_every', 'certificate_block_rows', 'exact_block_rows', 'progress_every'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if args.batch != 128:
        raise ValueError('The matched Rademacher oracle uses exactly 128 probes.')
    for name in ('prefactor_fraction', 'rho0', 'gamma0', 'mu_scale', 'inner_tol', 'inner_gap_tol'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    if max(args.prefactor_fraction, args.rho0, args.gamma0) > 1:
        raise ValueError('Prefactor fraction, rho0 and gamma0 must be at most one.')
    for name in ('seed', 'init_seed', 'output_seed'):
        if not 0 <= getattr(args, name) < 2**63:
            raise ValueError(f'{name} must lie in [0,2**63).')
    for value in args.checkpoint_iterations.split(','):
        if value and not 0 <= int(value) <= args.iterations:
            raise ValueError('Iteration checkpoints must belong to [0,T].')
    for value in args.checkpoints.split(','):
        if value and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError('Time diagnostic thresholds must be finite and positive.')


def verified_input(directory, device):
    """Hash once, broadcasting failures as well as successful verification."""
    packet = [None]
    if dist.get_rank() == 0:
        try:
            manifest = json.loads((directory / 'manifest.json').read_text())
            hashes = {name: sha256(directory / name) for name in ARRAY_NAMES}
            if hashes != manifest.get('array_files_sha256'):
                raise ValueError('Frozen instance array hashes do not match manifest.')
            packet[0] = dict(manifest=manifest, hashes=hashes,
                             manifest_sha256=sha256(directory / 'manifest.json'))
        except Exception as exc:
            packet[0] = dict(error=repr(exc))
    dist.broadcast_object_list(packet, src=0, device=device)
    if 'error' in packet[0]:
        raise ValueError(packet[0]['error'])
    return packet[0]


def source_hashes():
    paths = [ROOT / name for name in ('run_published.py', 'smd.py', 'certified_prox.py', 'inner_solver.json')]
    paths += [HISTORICAL_ROOT / name for name in ('run_sharded.py', 'sharded_gw.py')]
    return {str(path.relative_to(ROOT.parent)): sha256(path) for path in paths}


@torch.no_grad()
def run(args, device):
    validate_args(args)
    total_start = time.perf_counter()
    rank, world = dist.get_rank(), dist.get_world_size()
    out = Path(args.out)
    creation = [None]
    if rank == 0:
        try:
            out.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            creation[0] = repr(exc)
    dist.broadcast_object_list(creation, src=0, device=device)
    if creation[0]:
        raise ValueError(f'Cannot create new result directory: {creation[0]}')
    rank_dir = out / f'rank{rank}'
    rank_dir.mkdir()
    clock = Clock(device)
    clock.start()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    inputs = verified_input(Path(args.instance), device)
    dtype = getattr(torch, args.dtype)
    problem = load_problem(args.instance, dtype, device, args.exact_block_rows)
    cmax = cost_bound_from_extrema(problem.global_min(problem.A).item(),
                                   problem.global_max(problem.A).item(),
                                   problem.global_min(problem.B).item(),
                                   problem.global_max(problem.B).item())
    preparation_seconds = clock.stop()
    preparation_memory = memory(device)
    clock.start()
    P = initialization(problem, args.initialization, args.init_seed)
    logP = None
    if args.method == 'published-smd':
        logP = torch.empty(P.shape, dtype=torch.float64, device=device)
        for start in range(0, problem.local_rows, args.certificate_block_rows):
            block = slice(start, start+args.certificate_block_rows)
            logP[block] = P[block].double().log()
    momentum = None
    probes_rng = torch.Generator(device=device).manual_seed(args.seed)
    output_rng = torch.Generator(device='cpu').manual_seed(args.output_seed)
    selected_index = select_output_index(args.iterations, generator=output_rng)
    selected_probability = output_probabilities(args.iterations)[selected_index].item()
    proximal = (CertifiedKLProx(problem, gap_tol=args.inner_gap_tol,
                               certificate_block_rows=args.certificate_block_rows)
                if args.method == 'published-smd' else None)
    solver_seconds = clock.stop()
    repair_seconds = diagnostic_seconds = serialization_seconds = progress_logging_seconds = 0.
    progress_iterations = []
    counters = dict(completed_updates=0, stochastic_gradient_calls=0, sketch_probes=0,
                    initialization_oracle_calls=0, inner_iterations_total=0,
                    certificate_checks=0, inner_stopping_failures=0,
                    underflow_entries_total=0, underflow_steps=0,
                    exact_diagnostic_loss_calls=0)
    trace, inner_trace, saved = [], [], {}
    arrays_for_validation = {}
    initial_shard_hash = hashlib.sha256(P.cpu().numpy().tobytes()).hexdigest()
    iteration_targets = {int(x) for x in args.checkpoint_iterations.split(',') if x}
    time_targets = sorted(set(float(x) for x in args.checkpoints.split(',') if x))
    next_time_target = 0
    maxima = dict(marginal_l1=0., normalized_primal_dual_gap=0.,
                  normalized_gap_upper_bound=0., primal_witness_marginal_l1=0.)
    dimensions = dict(source=problem.n, target=problem.m)

    def account_inner(info):
        counters['inner_iterations_total'] += int(info.get('inner_iterations', 0))
        counters['certificate_checks'] += int(info.get('certificate_checks', 0))
        zeros = int(info.get('underflow_entries', 0))
        counters['underflow_entries_total'] += zeros
        counters['underflow_steps'] += int(zeros > 0)

    def evaluate(kind, iteration, *, save=False, crossed_times=None):
        nonlocal repair_seconds, diagnostic_seconds, serialization_seconds
        clock.start()
        assert_plan(problem, P)
        rounded = problem.round_transport(P)
        assert_plan(problem, rounded)
        repair_current = clock.stop()
        repair_seconds += repair_current
        clock.start()
        raw_residuals = problem.marginal_residuals(P)
        rounded_residuals = problem.marginal_residuals(rounded)
        if rounded_residuals['both_l1'] > 100 * torch.finfo(dtype).eps:
            raise FloatingPointError('Rounded diagnostic coupling failed feasibility validation.')
        raw_loss, rounded_loss = problem.loss(P).item(), problem.loss(rounded).item()
        counters['exact_diagnostic_loss_calls'] += 2
        if not math.isfinite(raw_loss) or not math.isfinite(rounded_loss):
            raise FloatingPointError('Nonfinite exact GW diagnostic.')
        row = dict(kind=kind, iteration=iteration, actual_solver_seconds=solver_seconds,
                   time_to_this_feasible_output=solver_seconds+repair_current,
                   repair_seconds_current=repair_current, raw_loss=raw_loss, rounded_loss=rounded_loss,
                   raw_residuals=raw_residuals, rounded_residuals=rounded_residuals,
                   repair_l1=problem.global_sum((rounded-P).abs()).item(),
                   crossed_time_thresholds=crossed_times or [])
        diagnostic_seconds += clock.stop()
        trace.append(row)
        if save:
            clock.start()
            for name, plan in (('raw', P), ('rounded', rounded)):
                key = f'{kind}_{name}'
                array = plan.cpu().numpy().copy()
                arrays_for_validation[key] = array
                persisted = args.save_plans == 'all'
                saved[key] = dict(persisted=persisted, file=None, file_sha256=None, bytes=0,
                                  array_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                                  shape=list(array.shape), dtype=str(array.dtype), array_bytes=array.nbytes)
                if persisted:
                    path = rank_dir / f'{key}.npy'
                    np.save(path, array, allow_pickle=False)
                    saved[key].update(file=str(path), file_sha256=sha256(path), bytes=path.stat().st_size)
            serialization_seconds += clock.stop()
        return row

    def progress():
        nonlocal progress_logging_seconds
        # Synchronize after rank-zero I/O, outside the solver clock, so peers
        # cannot charge waiting for this logging to their next update.
        clock.start()
        progress_iterations.append(counters['completed_updates'])
        if rank == 0:
            write_json(out/'progress.json', dict(status='running', config=vars(args),
                       fixed_horizon=args.iterations, selected_index=selected_index,
                       solver_seconds=solver_seconds, counters=counters, last_inner=inner_trace[-1] if inner_trace else None))
        progress_logging_seconds += clock.stop()

    evaluate('initial', 0)
    progress()
    for k in range(args.iterations):
        if k == selected_index:
            evaluate('selected', k, save=True)
        clock.start()
        gradient = problem.sketch_gradient(P, problem.rademacher(args.batch, generator=probes_rng))
        counters['stochastic_gradient_calls'] += 1
        counters['sketch_probes'] += args.batch
        problem.require_finite(gradient, 'fresh stochastic gradient')
        if args.method == 'published-smd':
            try:
                P, logP, info = smd_step(logP, gradient, iteration=k, cmax=cmax,
                    prox=proximal, marginal_residuals=proximal.marginal_residuals,
                    inner_tol=args.inner_tol, gap_tol=args.inner_gap_tol,
                    inner_max=args.inner_max, check_every=args.check_every,
                    prefactor_fraction=args.prefactor_fraction, require=problem._require)
            except Exception as exc:
                counters['inner_stopping_failures'] += 1
                failed_certificate = getattr(exc, 'certificate', None) or {}
                account_inner(failed_certificate)
                solver_seconds += clock.stop()
                progress()
                write_json(rank_dir/'inner_failure.json', dict(status='failed', iteration=k,
                    error=repr(exc), counters=counters, solver_seconds=solver_seconds,
                    certificate=json_safe(failed_certificate),
                    retained_log_plan_in_memory_until_exit=True, accepted_failed_update=False))
                raise
            account_inner(info)
            for key in maxima:
                maxima[key] = max(maxima[key], float(info[key]))
            inner_trace.append(dict(info))
        else:
            if k == 0:
                momentum = gradient
                counters['initialization_oracle_calls'] = 1
            else:
                rho = args.rho0 / math.sqrt(k+1)
                momentum = (1-rho)*momentum + rho*gradient
            problem.require_finite(momentum, 'gradient momentum')
            P = problem.fw_step(P, momentum, gamma=args.gamma0/(k+1)**.75,
                                mu=args.mu_scale/problem.m/(k+1)**.25)
        del gradient
        assert_plan(problem, P)
        counters['completed_updates'] += 1
        solver_seconds += clock.stop()
        crossed = []
        while next_time_target < len(time_targets) and time_targets[next_time_target] <= solver_seconds:
            crossed.append(time_targets[next_time_target])
            next_time_target += 1
        if k+1 in iteration_targets or crossed:
            evaluate('current_diagnostic', k+1, crossed_times=crossed)
        if k+1 in iteration_targets or crossed or (k+1) % args.progress_every == 0 or k+1 == args.iterations:
            progress()
    evaluate('last', args.iterations, save=True)
    optimization_memory = memory(device)
    del P, logP, momentum, proximal, problem
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    clock.start()
    validator = load_problem(args.instance, torch.float64, device, args.exact_block_rows)
    final_validation = dict(dtype='float64', exact_objective=True,
                            plan_values='Promoted CPU snapshot values; no additional re-rounding',
                            optimizer_state_released=True, plans={})
    for name, array in arrays_for_validation.items():
        promoted = torch.tensor(array, dtype=torch.float64, device=device)
        assert_plan(validator, promoted)
        loss = validator.loss(promoted).item()
        residuals = validator.marginal_residuals(promoted)
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite float64 final loss.')
        if name.endswith('_rounded') and residuals['both_l1'] > 100*torch.finfo(dtype).eps:
            raise FloatingPointError('Float64 validation rejected a rounded saved output.')
        final_validation['plans'][name] = dict(loss=loss, marginal_residuals=residuals)
        del promoted
    validation_seconds = clock.stop()
    validation_memory = memory(device)
    del validator, arrays_for_validation
    rank_record = dict(rank=rank, saved_plans=saved, initial_shard_sha256=initial_shard_hash,
                       memory=dict(preparation=preparation_memory, optimization_and_diagnostics=optimization_memory,
                                   float64_validation=validation_memory),
                       device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None)
    write_json(rank_dir/'metadata.json', rank_record)
    ranks = [None]*world
    dist.all_gather_object(ranks, rank_record)
    metadata = dict(status='completed', method=METHOD_NAMES[args.method], method_key=args.method,
        config=vars(args), counters=counters, ranks=ranks, cmax=cmax,
        array_files_sha256=inputs['hashes'], instance_manifest=inputs['manifest'],
        instance_manifest_sha256=inputs['manifest_sha256'], source_file_sha256=source_hashes(),
        output_selection=dict(index=selected_index, probability=selected_probability,
            candidates='P_0,...,P_(T-1); snapshot before its update', horizon=args.iterations,
            distribution='eta_k/sum_j eta_j = (k+1)^(-1/2)/sum_j (j+1)^(-1/2)',
            independent_rng=True, output_seed=args.output_seed,
            scope='Paper-specified SMD output' if args.method == 'published-smd' else 'Diagnostic only; no FW theorem-output claim',
            last_iterate_index=args.iterations, rounded_selected_is_diagnostic=True,
            selected_raw_is_the_algorithm_output=True),
        random_streams=dict(probes=dict(seed=args.seed, device=str(device)),
                            output=dict(seed=args.output_seed, device='cpu', independent_generator=True),
                            initialization=dict(seed=args.init_seed, generator='numpy independent if randomized')),
        schedule=dict(smd='eta_k = prefactor_fraction/(4*Cmax*sqrt(k+1)); Cmax=0 uses prefactor_fraction/sqrt(k+1)',
                      gamma_exponent=.75, rho_exponent=.5, mu_exponent=.25,
                      smoothing_mu0='mu_scale / number_of_target_cells'),
        inner_acceptance_maxima=maxima, inner_trace=inner_trace, trace=trace, dimensions=dimensions,
        certificate_work=dict(checks=counters['certificate_checks'],
                              global_plan_entries_per_check=dimensions['source']*dimensions['target'],
                              block_rows=args.certificate_block_rows,
                              all_certificate_elapsed_time_in_solver=True),
        final_float64_validation=final_validation,
        plan_storage=dict(mode=args.save_plans, dense_arrays_persisted=args.save_plans == 'all',
                          cpu_snapshots_and_hashes=True, all_four_outputs_validated_in_float64=True),
        progress_reporting=dict(interval_updates=args.progress_every,
                                completed_updates_recorded=progress_iterations,
                                excluded_from_solver_time=True),
        timing=dict(preparation_seconds=preparation_seconds, solver_seconds=solver_seconds,
                    repair_seconds=repair_seconds, exact_diagnostic_seconds=diagnostic_seconds,
                    serialization_seconds=serialization_seconds, progress_logging_seconds=progress_logging_seconds,
                    final_float64_validation_seconds=validation_seconds,
                    total_elapsed_seconds=time.perf_counter()-total_start,
                    solver_device_seconds=world*solver_seconds),
        environment=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                         hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                         world_size=world, backend=dist.get_backend(), cuda_runtime=torch.version.cuda,
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                         deterministic_algorithms=True, tf32=False),
        scientific_scope='Fixed-horizon published SMD structure with explicit finite inner tolerances; no theorem for this finite inexactness is claimed.',
        timing_policy='All gradient work, Sinkhorn iterations, certificate checks and acceptance audits are solver time. Exact GW diagnostics, output rounding, snapshot copying/hashing/persistence, progress logging and final float64 validation are timed separately.',
        coupling_partition='Contiguous source rows; no full coupling gather on any GPU',
        momentum_initialization='No momentum in SMD. FW d0=first fresh gradient, reused at update zero; one oracle call per update.')
    if rank == 0:
        write_json(out/'metadata.json', metadata)
        write_json(out/'progress.json', dict(status='completed', counters=counters, solver_seconds=solver_seconds))
        print(json.dumps(dict(out=str(out), status='completed', updates=args.iterations,
                              selected_index=selected_index, solver_seconds=solver_seconds)), flush=True)
    return metadata


def main():
    args = parser().parse_args()
    device = None
    rendezvous = None
    try:
        validate_args(args)
        if args.device == 'cpu' and 'RANK' not in os.environ:
            # Standalone CPU correctness tests, without requiring torchrun.
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
            torch.use_deterministic_algorithms(True)
            rendezvous = tempfile.TemporaryDirectory(prefix='published_smd_rendezvous_')
            dist.init_process_group('gloo', init_method='file://' + str(Path(rendezvous.name)/'store'), rank=0, world_size=1)
            device = torch.device('cpu')
        else:
            device = setup(args.device)
        run(args, device)
    except Exception as exc:
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get('RANK', '0'))
        write_json(directory/f'failure_rank{rank}.json', dict(status='failed', rank=rank,
                   error=repr(exc), traceback=traceback.format_exc(), config=vars(args)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous is not None:
            rendezvous.cleanup()


if __name__ == '__main__':
    main()
