"""Tiny CPU parity checks; the scientific 25k computation belongs on Slurm."""
from __future__ import annotations

import itertools
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cross_dataset_20260929" / "runtime"))
from gw_core import GWProblem, fw_row_update, kl_prox, marginal_residuals, round_transport
from sharded_gw import RowShardedGW, row_partition


def _fixture(dtype, n=7, m=5):
    generator = torch.Generator().manual_seed(751)
    x, y = torch.rand((n, 3), generator=generator, dtype=dtype), torch.rand((m, 2), generator=generator, dtype=dtype)
    A, B = torch.cdist(x, x), torch.cdist(y, y)
    a, b = torch.arange(1, n + 1, dtype=dtype), torch.arange(m, 0, -1, dtype=dtype)
    a, b = a / a.sum(), b / b.sum()
    P = torch.rand((n, m), generator=generator, dtype=dtype) / (n * m)
    Z = (2 * torch.randint(0, 2, (m, 9), generator=generator) - 1).to(dtype)
    return A, B, a, b, P, Z


def _exercise(rank, world_size, dtype, *, n=7, m=5):
    torch.set_num_threads(1)
    A, B, a, b, P, Z = _fixture(dtype, n, m)
    start, stop = row_partition(n, world_size, rank)
    sl = slice(start, stop)
    dense = GWProblem(A, B, a, b)
    problem = RowShardedGW(A[sl].clone(), B.clone(), a.clone(), b.clone(), exact_block_rows=2)
    rtol, atol = ((2e-11, 2e-12) if dtype == torch.float64 else (2e-5, 2e-6))

    def close(actual, expected):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    # Off-feasible and signed inputs expose accidentally substituted marginals.
    for full in (P, 4 * P, P - .03):
        local = full[sl].clone()
        close(problem.loss(local), dense.loss(full))
        close(problem.gradient(local), dense.gradient(full)[sl])
        close(problem.sketch_loss(local, Z), dense.sketch_loss(full, Z))
        close(problem.sketch_gradient(local, Z), dense.sketch_gradient(full, Z)[sl])
        rr, cc = problem.marginals(local)
        close(rr, full.sum(1))
        close(cc, full.sum(0))
        expected = marginal_residuals(full, a, b)
        actual = problem.marginal_residuals(local)
        for key in expected:
            close(torch.tensor(actual[key], dtype=dtype), torch.tensor(expected[key], dtype=dtype))

    # Enumerating the fixed Rademacher distribution proves the unbiased formula
    # for this nonsymmetric, off-feasible plan (no noisy empirical assertion).
    full_distribution = torch.tensor(list(itertools.product((-1., 1.), repeat=m)), dtype=dtype).T
    close(problem.sketch_loss(P[sl], full_distribution), dense.loss(P))
    close(problem.sketch_gradient(P[sl], full_distribution), dense.gradient(P)[sl])

    for full in (P, P * 10, torch.zeros_like(P)):
        actual = problem.round_transport(full[sl])
        close(actual, round_transport(full, a, b)[sl])
        residual = problem.marginal_residuals(actual)
        assert residual["both_l1"] < (1e-12 if dtype == torch.float64 else 1e-6)

    generator = torch.Generator().manual_seed(144)
    reference = torch.Generator().manual_seed(144)
    for _ in range(3):
        probes = problem.rademacher(9, generator=generator)
        expected = (2 * torch.randint(0, 2, (m, 9), generator=reference) - 1).to(dtype)
        assert torch.equal(probes, expected)

    # Both starts and five momentum steps retain the first-draw reuse at k=0.
    starts = [torch.outer(a, b), round_transport(P + .0001, a, b)]
    for initial in starts:
        for method in ("fw", "kl"):
            full = initial.clone()
            local = full[sl].clone()
            dense_log, local_log = full.log(), local.log()
            full_d = dense.sketch_gradient(full, Z)
            local_d = problem.sketch_gradient(local, Z)
            for k in range(5):
                rho = .3 / (k + 1) ** .5
                probes = Z.roll(k, dims=1).roll(k, dims=0)
                if k:
                    full_d = (1 - rho) * full_d + rho * dense.sketch_gradient(full, probes)
                    local_d = (1 - rho) * local_d + rho * problem.sketch_gradient(local, probes)
                close(local_d, full_d[sl])
                if method == "fw":
                    gamma, mu = .3 / (k + 1) ** .75, .2 / (k + 1) ** .25
                    H = full_d + (full.sum(0) - b)[None, :] / mu
                    full = fw_row_update(full, H, a, gamma)
                    local = problem.fw_step(local, local_d, gamma, mu)
                else:
                    tolerance = 1e-11 if dtype == torch.float64 else 1e-5
                    full, dense_log, dense_info = kl_prox(dense_log, full_d, a, b, .7,
                                                         tol=tolerance, max_iter=200, check_every=3)
                    local, local_log, local_info = problem.kl_prox(local_log, local_d, .7,
                                                                   tol=tolerance, max_iter=200, check_every=3)
                    close(local_log, dense_log[sl])
                    for key in ("inner_iterations", "converged", "underflow_entries"):
                        assert local_info[key] == dense_info[key], (key, local_info, dense_info)
                    close(torch.tensor(local_info["marginal_l1"], dtype=dtype),
                          torch.tensor(dense_info["marginal_l1"], dtype=dtype))
                close(local, full[sl])
                close(problem.loss(local), dense.loss(full))

    # Exact row minima choose the first column, including gamma=1.
    tied = torch.zeros_like(P)
    product = torch.outer(a, b)
    # Cancel the column penalty exactly: rounded product marginal sums need
    # not be bitwise identical in the two reduction orders.
    actual = problem.fw_step(torch.zeros_like(P[sl]), b.expand(stop - start, -1), 1., 1.)
    expected = fw_row_update(torch.zeros_like(P), tied, a, 1.)
    close(actual, expected[sl])

    # Max-iteration exits and underflow diagnostics are global and preserve logs.
    strong = torch.arange(n * m, dtype=dtype).reshape(n, m).square()
    for G, limit in ((strong / 30, 1), (strong, 7)):
        full, full_log, info = kl_prox(product.log(), G, a, b, 1.,
                                       tol=1e-13, max_iter=limit, check_every=3)
        local, local_log, local_info = problem.kl_prox(product[sl].log(), G[sl], 1.,
                                                      tol=1e-13, max_iter=limit, check_every=3)
        close(local, full[sl])
        # Strong float32 gradients amplify differences in subtraction; compare
        # finite retained logs at their actual scale, not a fixed 1e-6 tolerance.
        torch.testing.assert_close(local_log, full_log[sl], rtol=rtol * 5, atol=atol * 100)
        assert local_info["inner_iterations"] == info["inner_iterations"]
        assert local_info["underflow_entries"] == info["underflow_entries"]
        assert local_info["converged"] == info["converged"]
        assert torch.isfinite(local_log).all()

    assert not problem.gradient(P[sl].requires_grad_()).requires_grad
    invalid = P[sl].clone()
    if rank == 0:
        invalid[0, 0] = float("nan")
    try:
        problem.loss(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("One-rank nonfinite plan must raise collectively.")

    invalid = A[sl].clone()
    if rank == 0:
        invalid[0, 1] += 1
    try:
        RowShardedGW(invalid, B, a, b, exact_block_rows=2)
    except ValueError:
        pass
    else:
        raise AssertionError("An asymmetric distributed A must be rejected.")


def _worker(rank, world_size, init_file):
    from datetime import timedelta
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=90))
    try:
        for dtype in (torch.float64, torch.float32):
            _exercise(rank, world_size, dtype)
        # More ranks than rows exercises empty-partition reduction identities.
        _exercise(rank, world_size, torch.float64, n=2, m=3)
    finally:
        dist.destroy_process_group()


class ShardedGWTests(unittest.TestCase):
    def test_single_process(self):
        for dtype in (torch.float64, torch.float32):
            _exercise(0, 1, dtype)

    def test_uneven_gloo_three_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(3, str(Path(directory) / "store")), nprocs=3, join=True)

    def test_partition(self):
        self.assertEqual([row_partition(7, 3, rank) for rank in range(3)], [(0, 3), (3, 5), (5, 7)])
        self.assertEqual([row_partition(2, 3, rank) for rank in range(3)], [(0, 1), (1, 2), (2, 2)])


if __name__ == "__main__":
    unittest.main()
