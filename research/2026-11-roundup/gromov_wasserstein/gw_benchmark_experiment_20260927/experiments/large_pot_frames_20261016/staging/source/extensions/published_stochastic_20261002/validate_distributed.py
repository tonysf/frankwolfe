"""Small allocated GPU/Gloo validation of the published baseline's KL step."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / 'shareseq_25000_20260930'))
from gw_core import kl_prox as reference_prox
from sharded_gw import RowShardedGW, row_partition
from certified_prox import CertifiedKLProx
from smd import smd_step, ProximalSolveError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if args.device == 'cuda':
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('GPU validation must run in a Slurm allocation.')
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        if torch.cuda.device_count() != int(os.environ['LOCAL_WORLD_SIZE']):
            raise RuntimeError('Use one process per visible allocated GPU.')
    else:
        device = torch.device('cpu')
    dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                            timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    rows = []
    started = time.perf_counter()
    def check(name, condition, **metrics):
        failure = torch.tensor(int(not condition), device=device)
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        rows.append({'name': name, 'passed': failure.item() == 0, **metrics})
        if failure.item():
            raise AssertionError(name)
    try:
        for dtype in (torch.float64, torch.float32):
            for n in (2, 3, 17):
                label = f'{dtype}/n{n}'
                lo, hi = row_partition(n, world, rank)
                rng = np.random.default_rng(713+n)
                x = rng.normal(size=(n, 3))
                A = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
                A = (A/A.max()).astype(np.float32)
                B = A[::-1, ::-1].copy()
                A = torch.tensor(A, dtype=dtype, device=device)
                B = torch.tensor(B, dtype=dtype, device=device)
                a = torch.full((n,), 1/n, dtype=dtype, device=device)
                problem = RowShardedGW(A[lo:hi].clone(), B, a, a.clone(), exact_block_rows=3)
                solver = CertifiedKLProx(problem, gap_tol=1e-7, certificate_block_rows=2)
                fullP = torch.outer(a, a)
                logP = fullP[lo:hi].double().log()
                fullG = torch.tensor(rng.normal(size=(n, n)), dtype=dtype, device=device)
                if n == 2:
                    fullG = torch.tensor([[0., 3.], [3., 0.]], dtype=dtype, device=device)
                Q, logQ, info = smd_step(logP, fullG[lo:hi], iteration=0, cmax=1.,
                    prox=solver, marginal_residuals=solver.marginal_residuals,
                    inner_tol=1e-7, gap_tol=1e-7, inner_max=10000,
                    check_every=1, require=problem._require)
                check(label+'/both_criteria', info['marginal_l1'] <= 1e-7
                      and info['normalized_primal_dual_gap'] <= 1e-7,
                      marginal_l1=info['marginal_l1'], normalized_gap=info['normalized_primal_dual_gap'],
                      inner_iterations=info['inner_iterations'])
                check(label+'/retained_log64', logQ.dtype == torch.float64)
                check(label+'/witness_feasibility', info['primal_witness_marginal_l1'] <= 1e-12)
                if n == 2:
                    t = .5 / (1 + math.exp(-.25*3.))
                    expected = torch.tensor([[t, .5-t], [.5-t, t]], dtype=torch.float64, device=device)
                else:
                    expected, _, ref_info = reference_prox(fullP.double().log(), fullG.double(),
                        a.double(), a.double(), .25, tol=1e-13, max_iter=10000, check_every=1)
                    check(label+'/dense_reference_converged', ref_info['converged'])
                check(label+'/independent_solution', bool(torch.allclose(Q.double(), expected[lo:hi], atol=2e-8, rtol=2e-6)))
                # A second update exercises FP64 log state with FP32 oracle/plan.
                Q2, logQ2, info2 = smd_step(logQ, -fullG[lo:hi], iteration=1, cmax=1.,
                    prox=solver, marginal_residuals=solver.marginal_residuals,
                    inner_tol=1e-7, gap_tol=1e-7, inner_max=10000,
                    check_every=1, require=problem._require)
                check(label+'/second_update', info2['converged'] and info2['marginal_l1'] <= 1e-7
                      and info2['normalized_primal_dual_gap'] <= 1e-7)
                bad = logQ2.clone()
                if rank == 0:
                    bad[0, 0] = float('nan')
                rejected = False
                try:
                    solver(bad, fullG[lo:hi], eta=.25, tol=1e-7, max_iter=10, check_every=1)
                except (ValueError, FloatingPointError):
                    rejected = True
                check(label+'/one_rank_nonfinite_rejected', rejected)
        payload = {'status': 'passed', 'checks': rows, 'check_count': len(rows),
                   'world_size': world, 'device': str(device), 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'job_id': os.environ.get('SLURM_JOB_ID'),
                   'elapsed_seconds': time.perf_counter()-started,
                   'scope': 'Small kernel/integration checks; not scientific performance results.'}
        if rank == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
            print(json.dumps({'status': 'passed', 'checks': len(rows), 'world_size': world}))
    except BaseException as exc:
        if rank == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({'status': 'failed', 'checks': rows, 'error': repr(exc)}, indent=2)+'\n')
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
