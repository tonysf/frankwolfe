"""Bounded numerical oracle/update proof on the requested allocated topology.

CPU: standalone or torchrun/Gloo. CUDA: torchrun inside Slurm/NCCL. These
are tiny correctness fixtures, never timing or scientific campaign results.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_pair
from oracle import PairSample, sample_pairs, pair_gradient
from sharded_gw import RowShardedGW, row_partition
from run_sharded import setup, write_json, sha256
from smd import smd_step, cost_bound_from_extrema
from certified_prox import CertifiedKLProx
from stable_prox import StableCertifiedKLProx
from msbpg import constants_from_problem, msbpg_step


def fixture(device, dtype, n=7, m=5):
    # Integer distances are exactly symmetric even under low-precision loading.
    x = torch.arange(n, dtype=dtype, device=device)
    y = torch.arange(m, dtype=dtype, device=device)
    A = (x[:, None]-x[None, :]).abs()/4
    B = (y[:, None]-y[None, :]).abs()/3
    a = (x+1)/(x+1).sum()
    b = (m-y)/(m-y).sum()
    counts = (torch.arange(n*m, device=device).reshape(n, m) % 5).to(torch.int64)
    counts[0] = 0
    P = counts.to(dtype)/32  # Exact binary fractions; nonunit mass, sparse/zero row.
    return A, B, a, b, P, counts


def dense_slice_gradient(A, B, K, L, mass):
    # Direct squared differences independently check expanded GEMM arithmetic.
    return (A[:, K, None]-B[:, L].T[None, :, :]).square().mean(1)*(2*mass)


def dense_gradient(A, B, P):
    result = torch.zeros_like(P)
    for k in range(P.shape[0]):
        for l in range(P.shape[1]):
            result += 2*P[k, l]*(A[:, k, None]-B[None, :, l]).square()
    return result


def dense_prox(P, gradient, a, b, eta):
    K = P.double()*torch.exp(-eta*gradient.double())
    v = torch.ones_like(b, dtype=torch.float64)
    for _ in range(10000):
        u = a.double()/(K@v)
        v = b.double()/(K.T@u)
        Q = u[:, None]*K*v[None, :]
        if ((Q.sum(1)-a.double()).abs().sum()+(Q.sum(0)-b.double()).abs().sum()).item() < 2e-14:
            return Q.to(P.dtype)
    raise AssertionError('Tiny independent dense scaling failed.')


@torch.no_grad()
def check_oracle(device, dtype, n=7, m=5):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    lo, hi = row_partition(n, world, rank)
    A, B, a, b, P, counts = fixture(device, dtype, n, m)
    problem = RowShardedGW(A[lo:hi], B, a, b, exact_block_rows=2)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-12
    pair_ids = torch.repeat_interleave(torch.arange(n*m, device=device), counts.flatten())
    enumerated = PairSample(pair_ids//m, pair_ids % m, P.double().sum())
    expected = dense_gradient(A.double(), B.double(), P.double())[lo:hi]
    observed = pair_gradient(problem, enumerated)
    torch.testing.assert_close(observed.double(), expected, atol=tolerance, rtol=tolerance)
    maximum_error = (observed.double()-expected).abs().max().item() if observed.numel() else 0.
    # Prevent accidental communication of the full plan or dense thin-product.
    original_gather = problem._gather_rows
    def vector_only(value):
        assert value.ndim == 1, 'Pair sampler attempted a nonvector gather.'
        return original_gather(value)
    problem._gather_rows = vector_only
    def no_plan_broadcast(*args, **kwargs):
        raise AssertionError('Pair sampler attempted to broadcast the coupling.')
    problem._broadcast_rows = no_plan_broadcast
    rng = torch.Generator(device=device).manual_seed(891)
    reference_rng = torch.Generator(device=device).manual_seed(891)
    for draw in range(2):
        before = rng.get_state().clone()
        sample = sample_pairs(problem, P[lo:hi], 37, generator=rng, conditional_block_rows=2)
        uniforms = torch.rand((2, 37), dtype=torch.float64, device=device, generator=reference_rng)
        row_cdf = P.double().sum(1).cumsum(0)
        K = torch.searchsorted(row_cdf/row_cdf[-1], uniforms[0], right=True)
        L = torch.empty_like(K)
        for t in range(37):
            cdf = P[K[t]].double().cumsum(0)
            L[t] = torch.searchsorted(cdf/cdf[-1], uniforms[1, t], right=True)
        assert torch.equal(sample.source, K) and torch.equal(sample.target, L)
        assert torch.equal(rng.get_state(), reference_rng.get_state() if rank == 0 else before)
        assert bool((P[K, L] > 0).all())
        actual = pair_gradient(problem, sample)
        reference = dense_slice_gradient(A.double(), B.double(), K, L, P.double().sum())[lo:hi]
        torch.testing.assert_close(actual.double(), reference, atol=tolerance, rtol=tolerance)
    # Exactly-zero uniforms must skip zero rows and leading zero entries.
    zeros = torch.zeros((2, 3), dtype=torch.float64, device=device)
    sample = sample_pairs(problem, P[lo:hi], 3, uniforms=zeros)
    assert bool((P[sample.source, sample.target] > 0).all())
    invalids = [torch.zeros_like(P[lo:hi]), P[lo:hi].clone(), P[lo:hi].clone()]
    if rank == 0:
        invalids[1][0, 0] = -1
        invalids[2][0, 0] = float('nan')
    for invalid in invalids:
        try:
            sample_pairs(problem, invalid, 3, generator=rng)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid coupling accepted by pair sampler.')
    return dict(dtype=str(dtype), n=n, m=m, exact_enumeration_maximum_abs_error=maximum_error,
                nonunit_mass=P.double().sum().item(), checks=9,
                coverage=['weighted exact enumeration', 'off-marginal/nonunit/sparse/nonuniform',
                          'direct squared-distance sample gradient', 'two fresh rank-zero streams',
                          'no full-plan communication', 'zero CDF boundary',
                          'collective zero mass rejection', 'collective negative rejection', 'collective nonfinite rejection'])


@torch.no_grad()
def check_updates(device):
    rank, world = dist.get_rank(), dist.get_world_size()
    A, B, a, b, _, _ = fixture(device, torch.float64, n=5, m=4)
    lo, hi = row_partition(5, world, rank)
    problem = RowShardedGW(A[lo:hi], B, a, b, exact_block_rows=2)
    constants = constants_from_problem(problem, block_rows=2)
    cmax = cost_bound_from_extrema(A.min().item(), A.max().item(), B.min().item(), B.max().item())
    evidence = []
    for method in ('smoothing-fw', 'published-smd', 'published-msbpg'):
        full = torch.outer(a, b)
        local, logP, momentum, reference_momentum = full[lo:hi].clone(), full[lo:hi].log(), None, None
        rng = torch.Generator(device=device).manual_seed(651)
        proximal = (StableCertifiedKLProx if method == 'published-msbpg' else CertifiedKLProx)(
            problem, gap_tol=1e-9, certificate_block_rows=2)
        for k in range(3):
            samples = sample_pairs(problem, local, 41, generator=rng)
            G = pair_gradient(problem, samples)
            exact_samples = dense_slice_gradient(A, B, samples.source, samples.target, samples.mass)
            torch.testing.assert_close(G, exact_samples[lo:hi], atol=3e-13, rtol=3e-13)
            if method == 'smoothing-fw':
                rho = .3/(k+1)**.5
                momentum = G if k == 0 else (1-rho)*momentum+rho*G
                reference_momentum = exact_samples if k == 0 else (1-rho)*reference_momentum+rho*exact_samples
                gamma, mu = .1/(k+1)**.75, .1/len(b)/(k+1)**.25
                local = problem.fw_step(local, momentum, gamma=gamma, mu=mu)
                H = reference_momentum+(full.sum(0)-b)[None, :]/mu
                full = (1-gamma)*full
                full[torch.arange(len(a), device=device), H.argmin(1)] += gamma*a
            else:
                if method == 'published-smd':
                    local, logP, info = smd_step(logP, G, iteration=k, cmax=cmax, prox=proximal,
                        marginal_residuals=proximal.marginal_residuals, inner_tol=1e-11, gap_tol=1e-9,
                        inner_max=10000, check_every=1, require=problem._require)
                    direction, eta = exact_samples, info['eta']
                else:
                    local, logP, momentum, info = msbpg_step(logP, G, momentum, iteration=k,
                        constants=constants, prox=proximal, marginal_residuals=proximal.marginal_residuals,
                        inner_tol=1e-11, gap_tol=1e-9, inner_max=10000, check_every=1, require=problem._require)
                    beta = None if k == 0 else 1/k**.5
                    reference_momentum = exact_samples if k == 0 else (1-beta)*reference_momentum+beta*exact_samples
                    direction, eta = reference_momentum, info['proximal_eta']
                full = dense_prox(full, direction, a, b, eta)
            torch.testing.assert_close(local, full[lo:hi], atol=3e-10, rtol=3e-9)
        evidence.append(dict(method=method, updates=3, oracle_calls=3, pairs_per_call=41,
                             independent_dense_update_agreement=True))
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rendezvous = None
    evidence = dict(status='failed', scope='tiny correctness proof; no performance or campaign results')
    try:
        if args.device == 'cpu' and 'RANK' not in os.environ:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            rendezvous = tempfile.TemporaryDirectory(prefix='pair_oracle_proof_')
            dist.init_process_group('gloo', init_method='file://'+str(Path(rendezvous.name)/'store'), rank=0, world_size=1)
            device = torch.device('cpu')
        else:
            device = setup(args.device)
        evidence.update(world_size=dist.get_world_size(), backend=dist.get_backend(), device=str(device),
                        source_file_sha256=run_pair.source_hashes(),
                        validation_source_sha256=sha256(__file__), torch=torch.__version__)
        evidence['oracle_checks'] = [check_oracle(device, torch.float64), check_oracle(device, torch.float32),
                                     check_oracle(device, torch.float64, n=2, m=3)]
        evidence['update_checks'] = check_updates(device)
        evidence['status'] = 'passed'
    except Exception as exc:
        evidence.update(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get('RANK', 0))
        if rank == 0:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, evidence)
        if dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous:
            rendezvous.cleanup()


if __name__ == '__main__':
    main()
