"""Independent dense-reference validation of the row-sharded GW implementation.

Run with torchrun --standalone --nproc-per-node=4 gpu_correctness.py --out FILE
--device cuda (in Slurm), or --device cpu for a tiny local Gloo validation.
These fixed-size correctness checks are not scientific performance measurements.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist

from reference.gw_core import GWProblem, fw_row_update, kl_prox, marginal_residuals, rademacher, round_transport
from sharded_gw import RowShardedGW


class Checks:
    """Each comparison is reduced before branching so every rank participates."""

    def __init__(self, device):
        self.device = device
        self.rows = []

    def close(self, name, actual, expected, *, atol, rtol):
        actual = torch.as_tensor(actual, device=self.device)
        expected = torch.as_tensor(expected, device=self.device)
        shape_ok = actual.shape == expected.shape
        if shape_ok and actual.numel():
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
            if finite:
                delta = (actual.double() - expected.double()).abs()
                max_abs = delta.max().item()
                scaled = delta / (atol + rtol * expected.double().abs()).clamp_min(1e-300)
                max_scaled = scaled.max().item()
                passed = bool((scaled <= 1).all())
            else:
                max_abs = max_scaled = float('inf')
                passed = False
        elif shape_ok:
            passed, max_abs, max_scaled = True, 0., 0.
        else:
            passed, max_abs, max_scaled = False, float('inf'), float('inf')
        summary = torch.tensor([float(not passed), max_abs, max_scaled],
                               dtype=torch.float64, device=self.device)
        dist.all_reduce(summary, op=dist.ReduceOp.MAX)
        values = summary.cpu().tolist()
        self.rows.append(dict(name=name, passed=values[0] == 0,
                              max_absolute_error=values[1] if np.isfinite(values[1]) else None,
                              max_tolerance_ratio=values[2] if np.isfinite(values[2]) else None,
                              atol=atol, rtol=rtol))

    def true(self, name, condition):
        self.close(name, int(bool(condition)), 1, atol=0., rtol=0.)


def _initial_plan(dense, mode):
    P = torch.outer(dense.a, dense.b)
    if mode == 'randomized':
        n = dense.a.numel()
        permutation = np.random.default_rng(91507).permutation(n)
        P *= .9
        P[torch.arange(n, device=P.device), torch.as_tensor(permutation, device=P.device)] += .1 / n
    return P


def _shared_probes(n, q, dtype, device, generator):
    # Only rank zero consumes this stream. Every rank receives identical probes.
    if dist.get_rank() == 0:
        Z = (2 * torch.randint(0, 2, (n, q), generator=generator, device=device) - 1).to(dtype)
    else:
        Z = torch.empty((n, q), device=device, dtype=dtype)
    dist.broadcast(Z, src=0)
    return Z


def _compare_residuals(checks, prefix, actual, expected, *, atol, rtol):
    for key in ('row_l1', 'col_l1', 'col_l2', 'both_l1', 'mass_error', 'min_entry'):
        checks.close(prefix + '/' + key, actual[key], expected[key], atol=atol, rtol=rtol)


@torch.no_grad()
def validate_precision(checks, device, dtype, steps):
    n, q = 17, 128
    name = str(dtype).split('.')[-1]
    # Costs are frozen in float32 first, then promoted for both reference and core.
    rng = np.random.default_rng(92517)
    matrices = []
    for _ in range(2):
        raw = rng.uniform(.2, 1., size=(n, n))
        raw = ((raw + raw.T) / 2).astype(np.float32)
        np.fill_diagonal(raw, 0.)
        matrices.append(torch.as_tensor(raw, device=device, dtype=dtype))
    A, B = matrices
    a = torch.full((n,), 1 / n, device=device, dtype=dtype)
    b = a.clone()
    rank, world = dist.get_rank(), dist.get_world_size()
    # Contiguous near-equal partition, including the deliberately uneven n=17.
    sizes = [n // world + (int(r < n % world)) for r in range(world)]
    start, stop = sum(sizes[:rank]), sum(sizes[:rank + 1])
    problem = RowShardedGW(A[start:stop].clone(), B.clone(), a.clone(), b.clone(),
                           exact_block_rows=3)
    dense = GWProblem(A, B, a, b)
    checks.true(name + '/row_partition', (problem.row_start, problem.row_stop) == (start, stop))
    tol = dict(atol=2e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=2e-11, rtol=3e-10)
    ptol = dict(atol=8e-8, rtol=3e-5) if dtype == torch.float32 else dict(atol=2e-12, rtol=3e-10)
    inner_tol = 1e-6 if dtype == torch.float32 else 1e-11

    oracle_generator = torch.Generator(device=device).manual_seed(7435)
    reference_generator = torch.Generator(device=device).manual_seed(7435)
    nonroot_initial_state = oracle_generator.get_state().clone()
    for draw in range(3):
        actual_probes = problem.rademacher(q, generator=oracle_generator)
        expected_probes = rademacher(n, q, like=A, generator=reference_generator)
        checks.close(name + f'/rademacher_reference_stream_{draw}', actual_probes,
                     expected_probes, atol=0., rtol=0.)
    checks.true(name + '/nonroot_oracle_stream_not_consumed', rank == 0 or
                torch.equal(oracle_generator.get_state(), nonroot_initial_state))
    invalid = torch.ones((stop-start, n), device=device, dtype=dtype)
    if rank == world-1:
        invalid[0, 0] = float('nan')
    rejected = False
    try:
        problem.require_finite(invalid, 'injected validation error')
    except ValueError:
        rejected = True
    checks.true(name + '/one_rank_nonfinite_rejected_collectively', rejected)

    # Explicitly nonfeasible P tests actual marginals rather than feasible shortcuts.
    raw = rng.uniform(.1, 2., size=(n, n)).astype(np.float32)
    raw *= 1.23 / raw.sum()
    P_dense = torch.as_tensor(raw, device=device, dtype=dtype)
    P = P_dense[start:stop].clone()
    probes = torch.Generator(device=device).manual_seed(7131)
    Z = _shared_probes(n, q, dtype, device, probes)
    checks.close(name + '/infeasible_loss', problem.loss(P), dense.loss(P_dense), **tol)
    checks.close(name + '/infeasible_gradient', problem.gradient(P), dense.gradient(P_dense)[start:stop], **tol)
    checks.close(name + '/infeasible_sketch_loss', problem.sketch_loss(P, Z), dense.sketch_loss(P_dense, Z), **tol)
    checks.close(name + '/infeasible_sketch_gradient', problem.sketch_gradient(P, Z),
                 dense.sketch_gradient(P_dense, Z)[start:stop], **tol)
    _compare_residuals(checks, name + '/infeasible_residuals', problem.marginal_residuals(P),
                       marginal_residuals(P_dense, a, b), **tol)
    before = P.clone()
    rounded = problem.round_transport(P)
    expected = round_transport(P_dense, a, b)
    checks.close(name + '/rounding', rounded, expected[start:stop], **ptol)
    checks.close(name + '/rounding_does_not_mutate_raw', P, before, atol=0., rtol=0.)
    _compare_residuals(checks, name + '/rounded_residuals', problem.marginal_residuals(rounded),
                       marginal_residuals(expected, a, b), **tol)
    checks.true(name + '/rounded_feasibility', problem.marginal_residuals(rounded)['both_l1']
                < (1e-5 if dtype == torch.float32 else 1e-10))
    checks.close(name + '/global_sum', problem.global_sum(P), P_dense.sum(), **tol)
    checks.close(name + '/global_norm', problem.global_norm(P), P_dense.norm(), **tol)

    for mode in ('product', 'randomized'):
        for method in ('smoothing-fw', 'stochastic-kl-momentum'):
            prefix = name + '/' + mode + '/' + method
            P_dense = _initial_plan(dense, mode)
            P = P_dense[start:stop].clone()
            logP, logP_dense = P.log(), P_dense.log()
            probes = torch.Generator(device=device).manual_seed(83125)
            Z = _shared_probes(n, q, dtype, device, probes)
            d, d_dense = problem.sketch_gradient(P, Z), dense.sketch_gradient(P_dense, Z)
            sampled_points = 1
            for k in range(steps):
                # Initialization computes d0 once; update zero reuses that estimate.
                if k:
                    Z = _shared_probes(n, q, dtype, device, probes)
                    rho = .3 / (k + 1) ** .5
                    d = (1 - rho) * d + rho * problem.sketch_gradient(P, Z)
                    d_dense = (1 - rho) * d_dense + rho * dense.sketch_gradient(P_dense, Z)
                    sampled_points += 1
                checks.close(prefix + f'/momentum_{k}', d, d_dense[start:stop], **tol)
                if method == 'smoothing-fw':
                    gamma, mu = .3 / (k + 1) ** .75, (1. / n) / (k + 1) ** .25
                    P = problem.fw_step(P, d, gamma=gamma, mu=mu)
                    H_dense = d_dense + (P_dense.sum(0) - b)[None, :] / mu
                    P_dense = fw_row_update(P_dense, H_dense, a, gamma)
                else:
                    P, logP, info = problem.kl_prox(logP, d, eta=3., tol=inner_tol,
                                                  max_iter=1000, check_every=10)
                    P_dense, logP_dense, expected_info = kl_prox(
                        logP_dense, d_dense, a, b, eta=3., tol=inner_tol,
                        max_iter=1000, check_every=10)
                    checks.close(prefix + f'/log_state_{k}', logP, logP_dense[start:stop], **tol)
                    checks.true(prefix + f'/convergence_flag_{k}', info['converged'] == expected_info['converged'])
                    checks.close(prefix + f'/reported_marginal_l1_{k}', info['marginal_l1'],
                                 problem.marginal_residuals(P)['both_l1'], **tol)
                    checks.close(prefix + f'/underflow_count_{k}', info['underflow_entries'],
                                 expected_info['underflow_entries'], atol=0., rtol=0.)
                checks.close(prefix + f'/iterate_{k + 1}', P, P_dense[start:stop], **ptol)
                checks.close(prefix + f'/loss_{k + 1}', problem.loss(P), dense.loss(P_dense), **tol)
            checks.true(prefix + '/first_estimate_reuse_probe_count', sampled_points == steps)
            untouched = P.clone()
            R = problem.round_transport(P)
            R_dense = round_transport(P_dense, a, b)
            checks.close(prefix + '/final_rounding', R, R_dense[start:stop], **ptol)
            checks.close(prefix + '/final_rounded_loss', problem.loss(R), dense.loss(R_dense), **tol)
            checks.close(prefix + '/checkpoint_does_not_mutate_iterate', P, untouched, atol=0., rtol=0.)
            _compare_residuals(checks, prefix + '/final_residuals', problem.marginal_residuals(R),
                               marginal_residuals(R_dense, a, b), **tol)

    # Force an inexact inner solve; do not confuse the cap with convergence.
    positive = _initial_plan(dense, 'randomized')
    G = torch.as_tensor(rng.normal(size=(n, n)).astype(np.float32), device=device, dtype=dtype)
    capped, capped_log, capped_info = problem.kl_prox(
        positive[start:stop].log(), G[start:stop], eta=20., tol=1e-14,
        max_iter=1, check_every=10)
    reference_capped, reference_log, reference_info = kl_prox(
        positive.log(), G, a, b, eta=20., tol=1e-14, max_iter=1, check_every=10)
    checks.close(name + '/capped_kl_plan', capped, reference_capped[start:stop], **ptol)
    checks.close(name + '/capped_kl_logstate', capped_log, reference_log[start:stop], **tol)
    checks.true(name + '/capped_kl_reports_failure', not capped_info['converged'] and not reference_info['converged'])
    checks.true(name + '/capped_kl_iteration_count', capped_info['inner_iterations'] == 1)

    # Exponentiation can underflow, but a finite log-state must survive unchanged.
    underflow, underflow_log, underflow_info = problem.kl_prox(
        positive[start:stop].log(), G[start:stop], eta=2000., tol=1e-14,
        max_iter=10, check_every=10)
    underflow_reference, underflow_reference_log, underflow_reference_info = kl_prox(
        positive.log(), G, a, b, eta=2000., tol=1e-14, max_iter=10, check_every=10)
    checks.close(name + '/underflow_kl_plan', underflow, underflow_reference[start:stop], **ptol)
    checks.close(name + '/underflow_kl_logstate', underflow_log, underflow_reference_log[start:stop], **tol)
    checks.true(name + '/underflow_logstate_is_finite', bool(torch.isfinite(underflow_log).all()))
    checks.true(name + '/underflow_was_exercised', underflow_info['underflow_entries'] > 0)
    checks.close(name + '/underflow_global_count', underflow_info['underflow_entries'],
                 underflow_reference_info['underflow_entries'], atol=0., rtol=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.steps < 2 or args.threads < 1:
        parser.error('steps must be at least 2 and threads must be positive')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_world = int(os.environ.get('LOCAL_WORLD_SIZE', world))
    if world != local_world or rank != local_rank or not 1 <= world <= 17:
        raise RuntimeError('Correctness driver requires a single node with unique local ranks, at most 17 ranks')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('CUDA correctness checks require a Slurm allocation')
        if torch.cuda.device_count() != local_world:
            raise RuntimeError('Visible GPU count must equal the torchrun local process count')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(seconds=180))
    output = Path(args.out).resolve()
    checks = Checks(device)
    try:
        for dtype in (torch.float64, torch.float32):
            validate_precision(checks, device, dtype, args.steps)
        local = dict(rank=rank, local_rank=local_rank, hostname=socket.gethostname(), device=str(device))
        if device.type == 'cuda':
            p = torch.cuda.get_device_properties(device)
            local.update(gpu_name=p.name, total_memory_bytes=p.total_memory,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        ranks = [None] * world
        dist.all_gather_object(ranks, local)
        status = 'passed' if all(row['passed'] for row in checks.rows) else 'failed'
        report = dict(status=status, purpose='distributed correctness only; no scientific timing claim',
                      n=17, m=17, batch=128, momentum_steps=args.steps, world_size=world,
                      ranks=ranks, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      tf32=False, precision=['float64', 'float32'], checks=checks.rows,
                      failed_checks=[row for row in checks.rows if not row['passed']],
                      source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                     (Path(__file__), Path(__file__).with_name('sharded_gw.py'),
                                      Path(__file__).parent / 'reference' / 'gw_core.py')})
        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps(dict(status=status, checks=len(checks.rows),
                                  failed_checks=len(report['failed_checks']), out=str(output))), flush=True)
        dist.barrier()
        if status != 'passed':
            raise SystemExit(1)
    except Exception as exc:
        # torchrun terminates peers on fatal worker failure. Preserve a rank-local
        # traceback rather than risking an unmatched collective in this handler.
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(output.name + f'.rank{rank}.failure.json').write_text(json.dumps(
            dict(status='failed', rank=rank, error=repr(exc), traceback=traceback.format_exc(),
                 completed_checks=checks.rows), indent=2, allow_nan=False) + '\n')
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
