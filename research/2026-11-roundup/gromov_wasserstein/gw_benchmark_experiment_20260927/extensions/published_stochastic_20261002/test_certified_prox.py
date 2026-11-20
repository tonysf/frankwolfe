"""Bounded CPU tests, including independent optima and two-rank collectives."""
from pathlib import Path
import math
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shareseq_25000_20260930"))
from sharded_gw import RowShardedGW, row_partition
from certified_prox import CertifiedKLProx


def _problem(a=None, b=None, dtype=torch.float64):
    if a is None:
        a = torch.tensor([.5, .5], dtype=dtype)
    if b is None:
        b = a.clone()
    return RowShardedGW(torch.zeros((a.numel(), a.numel()), dtype=dtype),
                        torch.zeros((b.numel(), b.numel()), dtype=dtype), a, b)


def _scalar_optimum(reference, gradient, a, b, eta):
    # A one-dimensional derivative bisection, independent of Sinkhorn.
    lo = max(0., a[0].item() + b[0].item() - a.sum().item())
    hi = min(a[0].item(), b[0].item())
    direction = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
    for _ in range(100):
        t = (lo + hi) / 2
        Q = torch.tensor([[t, a[0].item() - t],
                          [b[0].item() - t, a[1].item() - b[0].item() + t]],
                         dtype=torch.float64)
        derivative = ((gradient + (Q.log() - reference.log()) / eta) * direction).sum()
        if derivative > 0:
            hi = t
        else:
            lo = t
    return Q


def test_matches_independent_scalar_optimum_and_direct_objective():
    a = torch.tensor([.35, .65], dtype=torch.float64)
    b = torch.tensor([.6, .4], dtype=torch.float64)
    problem = _problem(a, b)
    prox = CertifiedKLProx(problem, gap_tol=1e-11, certificate_block_rows=1)
    # A deliberately non-unit reference detects omitted mass constants.
    reference = torch.tensor([[.5, .2], [.1, .7]], dtype=torch.float64)
    G = torch.tensor([[3., -2.], [-4., 1.]], dtype=torch.float64)
    eta = .37
    Q, logQ, info = prox(reference.log(), G, eta, tol=1e-12, check_every=1)
    expected = _scalar_optimum(reference, G, a, b, eta)
    torch.testing.assert_close(Q, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(logQ.exp(), Q, rtol=0, atol=0)
    objective = ((G * expected).sum()
                 + (expected * (expected.log() - reference.log()) - expected + reference).sum() / eta)
    assert info["primal_objective"] == pytest.approx(objective.item(), abs=1e-11)
    assert info["dual_objective"] <= objective.item() + 1e-11
    assert info["reference_mass"] == pytest.approx(1.5)
    assert info["converged"] and info["proximal_optimality_certified"]
    assert info["primal_witness_marginal_l1"] < 1e-13


def test_feasible_nonoptimal_plan_fails_gap_despite_zero_marginal_error():
    problem = _problem()
    prox = CertifiedKLProx(problem)
    logP = torch.full((2, 2), .25, dtype=torch.float64).log()
    G = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.float64)
    eta = .5
    # Analytic optimum and its exact dual potentials for this symmetric case.
    diagonal = .5 / (1 + math.exp(-eta * 2.))
    alpha = torch.full((2,), math.log(diagonal / .25), dtype=torch.float64)
    feasible_but_wrong = torch.full((2, 2), .25, dtype=torch.float64)
    cert = prox.certificate(feasible_but_wrong, logP, G, eta,
                            row_potential=alpha, col_potential=torch.zeros(2, dtype=torch.float64))
    assert prox.marginal_residuals(feasible_but_wrong)["both_l1"] == 0
    assert cert["certificate_valid"]
    assert cert["normalized_primal_dual_gap"] > .1
    assert not cert["proximal_optimality_certified"]
    assert cert["primal_objective"] == pytest.approx(1.)


def test_iteration_cap_reports_failure_without_relaxing_criteria():
    a = torch.tensor([.25, .75], dtype=torch.float64)
    b = torch.tensor([.5, .5], dtype=torch.float64)
    problem = _problem(a, b)
    prox = CertifiedKLProx(problem)
    G = torch.tensor([[0., 40.], [40., 0.]], dtype=torch.float64)
    _, _, info = prox(torch.outer(a, b).log(), G, 1., max_iter=1, check_every=10)
    assert info["inner_iterations"] == info["certificate_checks"] == 1
    assert not info["converged"]
    assert info["marginal_l1"] > 1e-7
    assert "cap exhausted" in info["failure_reason"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_solution_and_actual_plan_precision(dtype):
    problem = _problem(dtype=dtype)
    prox = CertifiedKLProx(problem, certificate_block_rows=1)
    G = torch.tensor([[0., 2.], [2., 0.]], dtype=dtype)
    Q, logQ, info = prox(torch.full((2, 2), .25, dtype=dtype).log(), G, .5)
    diagonal = .5 / (1 + math.exp(-1))
    expected = torch.tensor([[diagonal, .5 - diagonal], [.5 - diagonal, diagonal]], dtype=dtype)
    torch.testing.assert_close(Q, expected, rtol=0, atol=1e-7 if dtype == torch.float32 else 1e-14)
    assert Q.dtype == dtype and logQ.dtype == torch.float64
    assert info["converged"]
    assert info["marginal_l1"] == prox.marginal_residuals(Q)["both_l1"]
    assert info["marginal_l1"] <= 1e-7
    assert info["normalized_primal_dual_gap"] <= 1e-7


def test_row_cost_shifts_change_objectives_but_not_optimizer():
    problem = _problem()
    prox = CertifiedKLProx(problem, certificate_block_rows=1)
    G = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.float64)
    lp = torch.full((2, 2), .25, dtype=torch.float64).log()
    Q, _, info = prox(lp, G, .5, tol=1e-12)
    shifts = torch.tensor([10000., -9998.], dtype=torch.float64)
    shifted, _, shifted_info = prox(lp, G + shifts[:, None], .5, tol=1e-12)
    torch.testing.assert_close(shifted, Q, rtol=0, atol=1e-12)
    for name in ("primal_objective", "dual_objective"):
        assert shifted_info[name] - info[name] == pytest.approx(1., abs=1e-11)
    assert shifted_info["converged"]


def test_dual_gauge_shift_is_invariant():
    problem = _problem()
    prox = CertifiedKLProx(problem)
    lp = torch.full((2, 2), .25, dtype=torch.float64).log()
    Q = lp.exp()
    zero = torch.zeros(2, dtype=torch.float64)
    info = prox.certificate(Q, lp, torch.zeros_like(Q), .3, row_potential=zero, col_potential=zero)
    shifted = prox.certificate(Q, lp, torch.zeros_like(Q), .3,
                               row_potential=zero + 13, col_potential=zero - 13)
    assert shifted["primal_objective"] == pytest.approx(info["primal_objective"], abs=1e-13)
    assert shifted["dual_objective"] == pytest.approx(info["dual_objective"], abs=1e-13)
    assert shifted["certificate_valid"]


def test_underflow_preserves_finite_log_state():
    problem = _problem(dtype=torch.float32)
    prox = CertifiedKLProx(problem)
    lp = torch.full((2, 2), .25, dtype=torch.float32).log()
    G = torch.tensor([[0., 2000.], [2000., 0.]], dtype=torch.float32)
    Q, logQ, info = prox(lp, G, .5)
    assert info["underflow_entries"] == 2
    assert torch.isfinite(logQ).all()
    assert (Q == 0).sum() == 2
    assert info["converged"]
    # A following opposite cost restores entries that underflowed in Q32;
    # reconstructing log-state with log(Q32) would fail this test.
    restored, restored_log, next_info = prox(logQ, -G, .5)
    torch.testing.assert_close(restored, torch.full_like(restored, .25), rtol=0, atol=1e-7)
    assert torch.isfinite(restored_log).all()
    assert next_info["underflow_entries"] == 0 and next_info["converged"]


def test_gap_scales_with_inverse_step_for_fixed_primal_and_dual():
    problem = _problem()
    prox = CertifiedKLProx(problem)
    lp = torch.full((2, 2), .25, dtype=torch.float64).log()
    Q = torch.tensor([[.4, .1], [.1, .4]], dtype=torch.float64)
    G = torch.zeros_like(Q)
    zero = torch.zeros(2, dtype=torch.float64)
    certificates = [prox.certificate(Q, lp, G, eta,
                                     row_potential=zero, col_potential=zero)
                    for eta in (.2, .8)]
    assert certificates[0]["primal_dual_gap"] == pytest.approx(
        4 * certificates[1]["primal_dual_gap"], abs=1e-13)
    for info in certificates:
        assert info["certificate_valid"] and not info["proximal_optimality_certified"]
        assert info["gap_arithmetic_crosscheck_error"] <= info["gap_roundoff_tolerance"]


def test_inconsistent_exact_marginal_masses_fail_explicitly():
    a = torch.tensor([.5, .5], dtype=torch.float64)
    b = torch.tensor([.5, .5 + 1e-9], dtype=torch.float64)
    problem = _problem(a, b)
    with pytest.raises(ValueError, match="masses must agree"):
        CertifiedKLProx(problem)


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for a_values, b_values in (([.25, .25, .5], [.125, .125, .25, .5]), ([1.], [.25, .75])):
            a = torch.tensor(a_values, dtype=torch.float64)
            b = torch.tensor(b_values, dtype=torch.float64)
            n, m = len(a), len(b)
            start, stop = row_partition(n, 2, rank)
            problem = RowShardedGW(torch.zeros((stop - start, n), dtype=torch.float64),
                                   torch.zeros((m, m), dtype=torch.float64), a, b,
                                   exact_block_rows=1)
            prox = CertifiedKLProx(problem, certificate_block_rows=1)
            # Row/column separable costs have constant objective on U(a,b),
            # hence their proximal optimizer is exactly the feasible reference.
            row_cost = torch.arange(n, dtype=torch.float64) * .7
            col_cost = torch.arange(m, dtype=torch.float64) * -.3
            G = row_cost[start:stop, None] + col_cost[None, :]
            reference = torch.outer(a[start:stop], b)
            Q, logQ, info = prox(reference.log(), G, .37, tol=1e-12, check_every=1)
            torch.testing.assert_close(Q, reference, rtol=0, atol=1e-14)
            torch.testing.assert_close(logQ.exp(), Q, rtol=0, atol=0)
            expected = (a @ row_cost + b @ col_cost).item()
            assert info["primal_objective"] == pytest.approx(expected, abs=1e-12)
            assert info["dual_objective"] == pytest.approx(expected, abs=1e-12)
            assert info["converged"]
            assert info["primal_witness_marginal_l1"] < 1e-13
            assert info["reference_mass"] == pytest.approx(1.)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_uneven_and_empty_shards():
    with tempfile.TemporaryDirectory(prefix="certified-prox-gloo-") as directory:
        rendezvous = "file://" + str(Path(directory) / "rendezvous")
        mp.spawn(_distributed_worker, args=(rendezvous,), nprocs=2, join=True)
