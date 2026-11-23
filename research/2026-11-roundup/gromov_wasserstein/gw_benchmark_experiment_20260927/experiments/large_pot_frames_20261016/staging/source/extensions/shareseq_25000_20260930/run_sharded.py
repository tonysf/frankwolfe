"""Equal-time stochastic GW comparisons with one row partition per GPU.

Launch with torchrun. No dense coupling is gathered on any GPU. Exact losses
and diagnostics are distributed, and final float64 validation follows release
of optimizer state. Source and input manifests are recorded in every result.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

from sharded_gw import RowShardedGW

ROOT = Path(__file__).resolve().parent
METHOD_NAMES = {
    'smoothing-fw': 'Smoothed row FW + momentum (sketch gradient, row-sharded)',
    'stochastic-kl-momentum': 'Transport KL-proximal gradient + momentum (sketch gradient, row-sharded)',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def setup(device_name):
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device_name == 'cuda':
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Ruche GPU execution requires a Slurm allocation.')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        if int(os.environ['WORLD_SIZE']) != int(os.environ['LOCAL_WORLD_SIZE']) or torch.cuda.device_count() != int(os.environ['LOCAL_WORLD_SIZE']):
            raise RuntimeError('Use one node and exactly one process per allocated visible GPU.')
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device, backend = torch.device('cpu'), 'gloo'
    dist.init_process_group(backend, timeout=datetime.timedelta(minutes=15))
    return device


class Clock:
    """Maximum synchronized rank wall-time; identical value on every rank.

    Kernel communication and boundary barriers are timed. The scalar collective
    used to combine clock readings is instrumentation, reported in total time.
    """
    def __init__(self, device):
        self.device = device

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def start(self):
        self.started = time.perf_counter()
        self.sync()
        dist.barrier()

    def stop(self):
        self.sync()
        dist.barrier()
        elapsed = torch.tensor(time.perf_counter() - self.started,
                               dtype=torch.float64, device=self.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return elapsed.item()


def shard_bounds(n, rank, world):
    q, remainder = divmod(n, world)
    start = rank * q + min(rank, remainder)
    return start, start + q + int(rank < remainder)


def load_problem(directory, dtype, device, block_rows):
    directory = Path(directory)
    a = np.load(directory / 'a.npy', mmap_mode='r', allow_pickle=False)
    b = np.load(directory / 'b.npy', mmap_mode='r', allow_pickle=False)
    lo, hi = shard_bounds(len(a), dist.get_rank(), dist.get_world_size())
    A = np.load(directory / 'A.npy', mmap_mode='r', allow_pickle=False)
    B = np.load(directory / 'B.npy', mmap_mode='r', allow_pickle=False)
    cast = lambda x: torch.tensor(np.asarray(x), dtype=dtype, device=device)
    return RowShardedGW(cast(A[lo:hi]), cast(B), cast(a), cast(b),
                        exact_block_rows=block_rows)


def assert_plan(problem, P):
    problem.require_finite(P, 'coupling')
    minimum = P.min()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    if minimum.item() < 0:
        raise FloatingPointError('Negative coupling rejected.')


def initialization(problem, mode, seed):
    P = torch.outer(problem.a_local, problem.b)
    if mode == 'randomized':
        n, m = len(problem.a), len(problem.b)
        if n != m or not torch.allclose(problem.a, torch.full_like(problem.a, 1/n)) or not torch.allclose(problem.b, torch.full_like(problem.b, 1/n)):
            raise ValueError('Randomized permutation starts require square uniform marginals.')
        permutation = np.random.default_rng(seed).permutation(n)
        indices = torch.as_tensor(permutation[problem.row_start:problem.row_stop], device=P.device)
        P *= .9
        P[torch.arange(len(P), device=P.device), indices] += .1/n
    return P


def matching_metrics(problem, P, correspondence, chunk=256):
    """Symmetric exact tie-aware metrics with sharded source rows."""
    n, m = len(problem.a), len(problem.b)
    if n != m:
        raise ValueError('Paired correspondence requires square costs.')
    forward = torch.as_tensor(correspondence, device=P.device, dtype=torch.long)
    local_map = forward[problem.row_start:problem.row_stop]
    truth_local = P[torch.arange(len(P), device=P.device), local_map]
    # Each target has exactly one true source; collect only this thin vector.
    target_truth = torch.zeros(m, device=P.device, dtype=P.dtype)
    target_truth[local_map] = truth_local
    dist.all_reduce(target_truth)
    f_accuracy = torch.zeros((), device=P.device, dtype=torch.float64)
    f_rank = torch.zeros_like(f_accuracy)
    inverse_equal = torch.zeros(m, device=P.device, dtype=torch.int64)
    inverse_higher = torch.zeros_like(inverse_equal)
    for start in range(0, len(P), chunk):
        block = P[start:start+chunk]
        truths = truth_local[start:start+chunk]
        equal = (block == truths[:, None]).sum(1)
        higher = (block > truths[:, None]).sum(1)
        f_accuracy += ((higher == 0).double() / equal).sum()
        f_rank += ((higher.double() + .5*(equal.double()-1)) / max(1, n-1)).sum()
        inverse_equal += (block == target_truth[None, :]).sum(0)
        inverse_higher += (block > target_truth[None, :]).sum(0)
    dist.all_reduce(f_accuracy); dist.all_reduce(f_rank)
    dist.all_reduce(inverse_equal); dist.all_reduce(inverse_higher)
    i_accuracy = ((inverse_higher == 0).double()/inverse_equal).mean().item()
    i_rank = ((inverse_higher.double()+.5*(inverse_equal.double()-1))/max(1, n-1)).mean().item()
    return dict(correspondence_top1_accuracy=(f_accuracy.item()/n+i_accuracy)/2,
                correspondence_rank_fraction=(f_rank.item()/n+i_rank)/2,
                correspondence_forward_top1_accuracy=f_accuracy.item()/n,
                correspondence_inverse_top1_accuracy=i_accuracy,
                correspondence_true_pair_mass=problem.global_sum(truth_local.double()).item())


def memory(device):
    value = dict(rank=dist.get_rank(), cpu_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if device.type == 'cuda':
        value.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                     cuda_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    return value


@torch.no_grad()
def run(args, device):
    total_start = time.perf_counter()
    rank, world = dist.get_rank(), dist.get_world_size()
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    rank_dir = out / f'rank{rank}'
    rank_dir.mkdir()
    clock = Clock(device)
    clock.start()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    instance_manifest = json.loads((Path(args.instance)/'manifest.json').read_text())
    # Verify every file once per job on rank zero, before any scientific work.
    checked = [None]
    if rank == 0:
        checked[0] = {name: sha256(Path(args.instance)/name) for name in ('A.npy','B.npy','a.npy','b.npy')}
    dist.broadcast_object_list(checked, src=0, device=device)
    recorded = instance_manifest.get('array_files_sha256')
    if not isinstance(recorded, dict) or set(recorded) != {'A.npy','B.npy','a.npy','b.npy'} or checked[0] != recorded:
        raise ValueError('Frozen instance array hashes do not match manifest.')
    problem = load_problem(args.instance, torch.float32, device, args.exact_block_rows)
    correspondence = None
    if args.correspondence:
        if not args.correspondence_sha256 or sha256(args.correspondence) != args.correspondence_sha256:
            raise ValueError('Correspondence hash is required and must match.')
        with np.load(args.correspondence, allow_pickle=False) as data:
            forward, inverse = data['forward'], data['inverse']
            n = len(problem.a)
            if len(problem.b) != n or forward.shape != (n,) or inverse.shape != (n,) or forward.dtype.kind not in 'iu' or inverse.dtype.kind not in 'iu':
                raise ValueError('Invalid correspondence shape or dtype.')
            if not np.array_equal(np.sort(forward), np.arange(n)) or not np.array_equal(inverse[forward], np.arange(n)):
                raise ValueError('Correspondence must be a mutually inverse permutation.')
            correspondence = forward.copy()
    preparation_seconds = clock.stop()
    preparation_memory = memory(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    clock.start()
    P = initialization(problem, args.initialization, args.init_seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    d = problem.sketch_gradient(P, problem.rademacher(args.batch, generator=generator))
    problem.require_finite(d, 'initial momentum')
    logP = P.log() if args.method == 'stochastic-kl-momentum' else None
    momentum_point, momentum_iteration = P, 0
    solver_seconds = clock.stop()
    repair_seconds = diagnostic_seconds = 0.
    counters = dict(completed_updates=0, stochastic_gradient_calls=1,
                    initialization_oracle_calls=1, sketch_probes=args.batch,
                    inner_iterations_total=0, inner_stopping_failures=0,
                    underflow_entries_total=0, underflow_steps=0)
    trace, failure_events = [], []
    targets = sorted(set([float(x) for x in args.checkpoints.split(',') if x] + [args.budget]))
    next_target = 0
    final_R = None
    last_evaluated = None
    initial_shard_hash = hashlib.sha256(P.cpu().numpy().tobytes()).hexdigest()

    def evaluate(kind, target):
        nonlocal repair_seconds, diagnostic_seconds, final_R, last_evaluated
        clock.start()
        assert_plan(problem, P)
        R = problem.round_transport(P)
        assert_plan(problem, R)
        repair_current = clock.stop()
        repair_seconds += repair_current
        clock.start()
        raw_residuals, rounded_residuals = problem.marginal_residuals(P), problem.marginal_residuals(R)
        if rounded_residuals['both_l1'] > 100*torch.finfo(P.dtype).eps:
            raise FloatingPointError('Rounded coupling failed feasibility validation.')
        row = dict(kind=kind, target_solver_seconds=target, actual_solver_seconds=solver_seconds,
                   iteration=counters['completed_updates'], raw_loss=problem.loss(P).item(),
                   rounded_loss=problem.loss(R).item(),
                   repair_l1=problem.global_sum((R-P).abs()).item(),
                   repair_seconds_current=repair_current, repair_seconds_cumulative=repair_seconds,
                   time_to_this_feasible_output=solver_seconds+repair_current,
                   **raw_residuals, **{'rounded_'+k:v for k,v in rounded_residuals.items()})
        if not math.isfinite(row['raw_loss']) or not math.isfinite(row['rounded_loss']):
            raise FloatingPointError('Nonfinite exact objective.')
        G = problem.gradient(momentum_point)
        row.update(momentum_point_iteration=momentum_iteration,
                   momentum_error_matches_current_output=(momentum_iteration == counters['completed_updates']),
                   momentum_error_frobenius=problem.global_norm(d-G).item())
        del G
        if args.method == 'smoothing-fw':
            mu = args.mu_scale/len(problem.b)/(counters['completed_updates']+1)**.25
            G = problem.gradient(P)
            _, c = problem.marginals(P)
            H = G + (c-problem.b)[None, :]/mu
            row.update(mu=mu, true_smoothed_gap=problem.global_sum(
                (H*P).sum()- (problem.a_local*H.min(1).values).sum()).item())
            del G, H
        if correspondence is not None:
            row.update(matching_metrics(problem, R, correspondence))
        diagnostic_seconds += clock.stop()
        row.update(diagnostic_seconds_cumulative=diagnostic_seconds,
                   **{'cumulative_'+k:v for k,v in counters.items()})
        trace.append(row)
        final_R, last_evaluated = R, counters['completed_updates']

    evaluate('initial', None)
    while counters['completed_updates'] == 0 or solver_seconds < args.budget:
        clock.start()
        k = counters['completed_updates']
        momentum_point, momentum_iteration = P, k
        if k > 0:
            fresh = problem.sketch_gradient(P, problem.rademacher(args.batch, generator=generator))
            rho = args.rho0/(k+1)**.5
            d = (1-rho)*d + rho*fresh
            del fresh
            counters['stochastic_gradient_calls'] += 1
            counters['sketch_probes'] += args.batch
        problem.require_finite(d, 'momentum')
        if args.method == 'smoothing-fw':
            P = problem.fw_step(P, d, gamma=args.gamma0/(k+1)**.75,
                                mu=args.mu_scale/len(problem.b)/(k+1)**.25)
        else:
            P, logP, info = problem.kl_prox(logP, d, eta=args.eta, tol=args.inner_tol,
                                            max_iter=args.inner_max, check_every=10)
            counters['inner_iterations_total'] += info['inner_iterations']
            counters['inner_stopping_failures'] += int(not info['converged'])
            counters['underflow_entries_total'] += info['underflow_entries']
            counters['underflow_steps'] += int(info['underflow_entries'] > 0)
            if not info['converged'] or info['underflow_entries']:
                failure_events.append(dict(iteration=k+1, **info))
        assert_plan(problem, P)
        counters['completed_updates'] += 1
        solver_seconds += clock.stop()
        crossed = []
        while next_target < len(targets) and targets[next_target] <= solver_seconds:
            crossed.append(targets[next_target]); next_target += 1
        if crossed:
            evaluate('checkpoint', crossed[0])
            for missed in crossed[1:]:
                trace.append(dict(kind='missed', target_solver_seconds=missed,
                                  actual_solver_seconds=solver_seconds, iteration=counters['completed_updates'],
                                  reason='same_completed_iterate_used_for_earlier_target'))
        if args.max_updates and counters['completed_updates'] >= args.max_updates:
            break
    if last_evaluated != counters['completed_updates']:
        evaluate('final', None)
    for missed in targets[next_target:]:
        trace.append(dict(kind='missed', target_solver_seconds=missed, actual_solver_seconds=solver_seconds,
                          iteration=counters['completed_updates'], reason='fixed_update_limit'))
    solver_and_diagnostic_memory = memory(device)
    clock.start()
    raw_cpu, rounded_cpu = P.cpu().numpy(), final_R.cpu().numpy()
    saved = {}
    for name, array in (('P_raw', raw_cpu), ('P_rounded', rounded_cpu)):
        saved[name] = dict(array_sha256=hashlib.sha256(array.tobytes()).hexdigest(), shape=list(array.shape))
        if args.save_plans:
            path = rank_dir/(name+'.npy')
            np.save(path, array, allow_pickle=False)
            saved[name].update(file=str(path), file_sha256=sha256(path), bytes=path.stat().st_size)
    serialization_seconds = clock.stop()
    del P, final_R, d, momentum_point, logP, problem
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    clock.start()
    validator = load_problem(args.instance, torch.float64, device, args.exact_block_rows)
    final_validation = dict(dtype='float64', plan='same output values promoted without re-rounding',
                            exact_objective=True, loss_calls=2)
    for name, array in (('raw',raw_cpu), ('rounded',rounded_cpu)):
        promoted = torch.tensor(array, dtype=torch.float64, device=device)
        assert_plan(validator, promoted)
        final_validation[name+'_loss'] = validator.loss(promoted).item()
        final_validation[name] = validator.marginal_residuals(promoted)
        del promoted
    if not all(math.isfinite(final_validation[k]) for k in ('raw_loss','rounded_loss')) or final_validation['rounded']['both_l1'] > 100*torch.finfo(torch.float32).eps:
        raise FloatingPointError('Float64 final validation failed.')
    validation_seconds = clock.stop()
    validation_memory = memory(device)
    del validator, raw_cpu, rounded_cpu
    rank_record = dict(rank=rank, initial_shard_sha256=initial_shard_hash, saved_plans=saved,
                       memory=dict(preparation=preparation_memory, solver_and_diagnostics=solver_and_diagnostic_memory,
                                   float64_validation=validation_memory),
                       device=str(device), gpu=(torch.cuda.get_device_name(device) if device.type == 'cuda' else None))
    write_json(rank_dir/'metadata.json', rank_record)
    ranks = [None]*world
    dist.all_gather_object(ranks, rank_record)
    metadata = dict(status='completed_with_inner_failures' if counters['inner_stopping_failures'] else 'completed',
        method=METHOD_NAMES[args.method], config=vars(args), counters=counters, ranks=ranks,
        array_files_sha256=checked[0], instance_manifest=instance_manifest,
        source_file_sha256={p.name:sha256(p) for p in (ROOT/'run_sharded.py',ROOT/'sharded_gw.py',ROOT/'theory_config.json')},
        source_revision=(ROOT/'SOURCE_REVISION').read_text().strip() if (ROOT/'SOURCE_REVISION').exists() else None,
        environment=dict(torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                         hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                         world_size=world, backend=dist.get_backend(), cuda_runtime=torch.version.cuda,
                         cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                         cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                         deterministic_algorithms=True, tf32=False),
        final_float64_validation=final_validation, trace=trace, failure_events=failure_events,
        timing=dict(preparation_seconds=preparation_seconds,solver_seconds=solver_seconds,repair_seconds=repair_seconds,
                    exact_diagnostic_seconds=diagnostic_seconds,serialization_seconds=serialization_seconds,
                    final_float64_validation_seconds=validation_seconds,
                    total_elapsed_seconds=time.perf_counter()-total_start,
                    solver_gpu_seconds=world*solver_seconds),
        scientific_scope='practical current-output comparison; no full-trajectory theorem statistic claimed',
        numerical_equivalence='same estimator and updates; distributed reductions change floating-point summation order',
        timing_policy='maximum rank synchronized wall time including algorithm collectives; clock scalar reduction is instrumentation',
        coupling_partition='contiguous source rows; B and B squared replicated; no full coupling GPU gather',
        output_selection='current output; no best-so-far or correspondence-based selection',
        momentum_initialization='d0=first sketch gradient; reused at update zero; rho0 recorded but unused there')
    if rank == 0:
        write_json(out/'metadata.json', metadata)
        print(json.dumps(dict(out=str(out),status=metadata['status'],updates=counters['completed_updates'],
                              solver_seconds=solver_seconds,rounded_loss=final_validation['rounded_loss'])),flush=True)
    return metadata


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--method',choices=list(METHOD_NAMES),required=True)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--budget',type=float,required=True)
    p.add_argument('--checkpoints',default='10,30')
    p.add_argument('--max-updates',type=int,default=0,help='Correctness/resource probes only; zero disables.')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--init-seed',type=int,required=True)
    p.add_argument('--initialization',choices=('product','randomized'),default='product')
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--rho0',type=float,default=.3)
    p.add_argument('--gamma0',type=float,default=.1)
    p.add_argument('--mu-scale',type=float,default=.1)
    p.add_argument('--eta',type=float,default=.1)
    p.add_argument('--inner-tol',type=float,default=1e-4)
    p.add_argument('--inner-max',type=int,default=1000)
    p.add_argument('--exact-block-rows',type=int,default=1024)
    p.add_argument('--correspondence')
    p.add_argument('--correspondence-sha256')
    p.add_argument('--save-plans',action='store_true')
    return p


def main():
    args = parser().parse_args()
    if not math.isfinite(args.budget) or args.budget <= 0 or args.batch != 128 or args.inner_max < 1 or args.exact_block_rows < 1 or args.max_updates < 0:
        raise ValueError('Invalid budget, frozen sketch size, or iteration bounds.')
    for value in (args.rho0,args.gamma0,args.mu_scale,args.eta,args.inner_tol):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Schedule and proximal constants must be finite and positive.')
    if args.rho0 > 1 or args.gamma0 > 1 or any(float(x) <= 0 or float(x) > args.budget or not math.isfinite(float(x)) for x in args.checkpoints.split(',') if x):
        raise ValueError('Invalid FW/momentum coefficient or checkpoint grid.')
    if not 0 <= args.seed < 2**63 or not 0 <= args.init_seed < 2**63:
        raise ValueError('Seeds must lie in [0,2**63).')
    device = setup(args.device)
    try:
        run(args, device)
    except Exception as exc:
        directory = Path(args.out)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory/f'failure_rank{dist.get_rank()}.json',
                   dict(status='failed', rank=dist.get_rank(), error=repr(exc),
                        traceback=traceback.format_exc(), config=vars(args)))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
