"""Numerical parity and failure-budget tests for the sharded fast inner solve."""
from pathlib import Path
import sys

import numpy as np
import ot
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prox import DistributedCertifiedPOTProx, CertifiedKLProx, ProximalSolveError
from validation import fixture, dense_gradient, RowShardedGW


def problem_fixture():
    torch.set_num_threads(1)
    values = [torch.tensor(x, dtype=torch.float64) for x in fixture()]
    problem = RowShardedGW(*values, exact_block_rows=2)
    plan = torch.outer(problem.a, problem.b)
    return problem, plan, plan.log()


def test_four_full_updates_match_pot_entropic_pgd():
    p, initial, logref = problem_fixture()
    prox = DistributedCertifiedPOTProx(p, gap_tol=1e-11, certificate_block_rows=2)
    plan = initial.clone()
    for _ in range(4):
        plan, logplan, info = prox(logref, p.gradient(plan), 1000., tol=1e-13, check_every=1)
        assert info["converged"]
        assert info["kernel_released_before_candidate"]
        torch.testing.assert_close(logplan.exp(), plan, atol=0., rtol=0.)
    expected = ot.gromov.entropic_gromov_wasserstein(
        p.A.numpy(), p.B.numpy(), p.a.numpy(), p.b.numpy(), loss_fun="square_loss",
        epsilon=.001, solver="PGD", G0=initial.numpy(), max_iter=4, tol=0.,
        numItermax=100000, stopThr=1e-14)
    np.testing.assert_allclose(plan.numpy(), expected, rtol=0., atol=2e-11)


def test_matches_original_certificate_and_log_domain_plan():
    p, initial, logref = problem_fixture()
    gradient = p.gradient(initial)
    fast = DistributedCertifiedPOTProx(p, gap_tol=1e-11, certificate_block_rows=2)
    slow = CertifiedKLProx(p, gap_tol=1e-11, certificate_block_rows=2)
    actual = fast(logref, gradient, 1000., tol=1e-12, check_every=1)
    expected = slow(logref, gradient, 1000., tol=1e-12, check_every=1)
    assert actual[2]["converged"] and expected[2]["converged"]
    torch.testing.assert_close(actual[0], expected[0], atol=1e-13, rtol=0.)
    assert actual[2]["reference_mass"] == pytest.approx(1., abs=1e-14)
    assert actual[2]["primal_objective"] == pytest.approx(expected[2]["primal_objective"], abs=1e-13)


def test_kernel_underflow_uses_original_log_domain_and_counts_zeros():
    p, initial, logref = problem_fixture()
    gradient = torch.zeros_like(initial)
    gradient[0, 0] = 1.
    prox = DistributedCertifiedPOTProx(p, certificate_block_rows=2)
    Q, _, info = prox(logref, gradient, 1000.)
    assert info["converged"]
    assert info["sinkhorn_path"] == "original-log-domain-fallback"
    assert info["fast_scaling_iterations"] == 0
    assert info["inner_iterations"] == info["log_domain_iterations"]
    assert info["underflow_entries"] == 1
    assert Q[0, 0] == 0


def test_late_failure_counts_both_paths_against_one_budget():
    p, plan, logref = problem_fixture()
    class FailThird(DistributedCertifiedPOTProx):
        calls = 0
        def _scaling_iteration(self, kernel, v):
            self.calls += 1
            if self.calls == 3:
                return None, None, None, "injected failure"
            return super()._scaling_iteration(kernel, v)
    _, _, info = FailThird(p)(logref, p.gradient(plan), 1000., max_iter=5, check_every=10)
    assert info["fast_scaling_iterations"] == 3
    assert info["log_domain_iterations"] == 2
    assert info["inner_iterations"] == info["total_inner_budget"] == 5
    assert not info["converged"]


def test_exhausted_budget_does_not_start_fallback():
    p, plan, logref = problem_fixture()
    class FailImmediately(DistributedCertifiedPOTProx):
        def _scaling_iteration(self, kernel, v):
            return None, None, None, "injected failure"
    with pytest.raises(ProximalSolveError) as error:
        FailImmediately(p)(logref, p.gradient(plan), 1000., max_iter=1)
    assert error.value.certificate["inner_iterations"] == 1
    assert error.value.certificate["total_inner_budget"] == 1


def test_cap_reports_nonconvergence_with_original_certificate():
    p, plan, logref = problem_fixture()
    _, _, info = DistributedCertifiedPOTProx(p)(logref, p.gradient(plan), 1000., max_iter=1)
    assert info["inner_iterations"] == info["certificate_checks"] == 1
    assert not info["converged"]
    assert "primal_objective" in info and "stable_primal_dual_gap" in info


def test_expanded_product_reference_and_inputs_unchanged():
    marginal = torch.full((4,), .25, dtype=torch.float64)
    distances = torch.zeros((4, 4), dtype=torch.float64)
    p = RowShardedGW(distances, distances, marginal, marginal)
    logref = torch.tensor(1 / 16, dtype=torch.float64).log().expand(4, 4)
    gradient = torch.arange(16, dtype=torch.float64).reshape(4, 4) * .0001
    saved = gradient.clone()
    Q, _, info = DistributedCertifiedPOTProx(p)(logref, gradient, 1000.)
    assert info["converged"]
    torch.testing.assert_close(Q, logref.exp(), atol=1e-14, rtol=0.)
    assert torch.equal(gradient, saved) and logref.stride() == (0, 0)


@pytest.mark.parametrize("options", [{"eta": 0.}, {"eta": float("nan")},
                                    {"eta": 1000., "max_iter": 0},
                                    {"eta": 1000., "check_every": 0}])
def test_invalid_arguments_rejected(options):
    p, plan, logref = problem_fixture()
    with pytest.raises(ValueError):
        DistributedCertifiedPOTProx(p)(logref, p.gradient(plan), **options)
