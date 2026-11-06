"""Small CPU checks against an independent one-dimensional constrained solve."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gw_core import marginal_residuals
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shareseq_25000_20260930"))
from sharded_gw import RowShardedGW
from certified_prox import CertifiedKLProx
from smd import (ProximalSolveError, cost_bound_from_distances,
                 cost_bound_from_extrema, output_probabilities,
                 select_output_index, smd_step, step_size)


def _scalar_transport_optimum(reference, gradient, a, b, eta):
    """Solve the strictly convex 2x2 problem by bisection on its derivative.

    Every feasible plan is [[t,a0-t],[b0-t,1-a0-b0+t]]. This uses no
    Sinkhorn iteration, matrix scaling, or implementation under test.
    """
    lo, hi = max(0., a[0].item() + b[0].item() - 1), min(a[0].item(), b[0].item())
    direction = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
    for _ in range(100):
        t = (lo + hi) / 2
        candidate = torch.tensor([[t, a[0].item() - t],
                                  [b[0].item() - t, 1 - a[0].item() - b[0].item() + t]],
                                 dtype=torch.float64)
        derivative = ((gradient + (candidate.log() - reference.log()) / eta) * direction).sum()
        if derivative > 0:
            hi = t
        else:
            lo = t
    return candidate


def _problem():
    a = torch.tensor([.35, .65], dtype=torch.float64)
    b = torch.tensor([.6, .4], dtype=torch.float64)
    return a, b, a[:, None] * b[None, :]


def _prox(a, b):
    return CertifiedKLProx(RowShardedGW(torch.zeros((len(a), len(a)), dtype=a.dtype),
                                       torch.zeros((len(b), len(b)), dtype=b.dtype), a, b))


def test_two_fresh_gradient_steps_match_independent_constrained_optima():
    a, b, plan = _problem()
    log_plan = plan.log()
    gradients = [torch.tensor([[3., -2.], [-4., 1.]], dtype=torch.float64),
                 torch.tensor([[-1., 4.], [2., -3.]], dtype=torch.float64)]
    for k, gradient in enumerate(gradients):
        expected = _scalar_transport_optimum(plan, gradient, a, b, step_size(k, 1.))
        plan, log_plan, info = smd_step(
            log_plan, gradient, iteration=k, cmax=1.,
            prox=_prox(a, b),
            marginal_residuals=lambda Q: marginal_residuals(Q, a, b),
            inner_tol=1e-13, inner_max=10000)
        torch.testing.assert_close(plan, expected, rtol=0, atol=1e-13)
        torch.testing.assert_close(log_plan.exp(), plan, rtol=0, atol=0)
        assert info["marginal_l1"] <= 1e-13
        assert info["proximal_optimality_certified"]
        assert info["normalized_primal_dual_gap"] <= 1e-7


def test_extrema_bound_matches_enumerated_squared_gw_costs():
    A = torch.tensor([[0., 2., 1.], [2., 0., 1.5], [1., 1.5, 0.]], dtype=torch.float64)
    B = torch.tensor([[0., 3.], [3., 0.]], dtype=torch.float64)
    all_costs = (A[:, :, None, None] - B[None, None, :, :]).square()
    assert cost_bound_from_distances(A, B) == all_costs.max().item() == 9.
    assert cost_bound_from_extrema(.5, 2., .2, 4.) == 3.5**2
    assert step_size(0, 9.) == pytest.approx(1/36)
    assert step_size(3, 9.) == pytest.approx(1/72)
    assert step_size(0, 0.) == 1.
    assert step_size(3, 0.) == .5
    assert step_size(3, 9., prefactor_fraction=.4) == pytest.approx(.4/72)


@pytest.mark.parametrize("fraction", [0., -.2, 1.01, float("nan"), float("inf")])
def test_prefactor_tuning_stays_within_the_bound(fraction):
    with pytest.raises(ValueError, match="prefactor_fraction"):
        step_size(0, 1., prefactor_fraction=fraction)


@pytest.mark.parametrize("iteration", [-1, .5, True])
def test_invalid_schedule_index(iteration):
    with pytest.raises(ValueError, match="iteration"):
        step_size(iteration, 1.)


@pytest.mark.parametrize("extrema", [(2., 1., 0., 1.), (-1., 1., 0., 1.),
                                     (0., float("nan"), 0., 1.), (0., 1e300, 0., 1.)])
def test_invalid_distance_ranges(extrema):
    with pytest.raises(ValueError):
        cost_bound_from_extrema(*extrema)


@pytest.mark.parametrize("bad", [torch.tensor([[0., 1.], [2., 0.]]),
                                 torch.tensor([[1., 0.], [0., 0.]]),
                                 torch.tensor([[0., float("inf")], [float("inf"), 0.]]),
                                 torch.tensor([[0., -1.], [-1., 0.]]), torch.ones(2, 3)])
def test_invalid_canonical_matrices(bad):
    with pytest.raises(ValueError):
        cost_bound_from_distances(bad, torch.zeros(2, 2))


def test_output_distribution_is_step_weighted_and_rng_is_separate():
    weights = output_probabilities(4)
    torch.testing.assert_close(weights[0] / weights[3], torch.tensor(2., dtype=torch.float64))
    oracle_rng = torch.Generator().manual_seed(97)
    oracle_state = oracle_rng.get_state().clone()
    output_rng = torch.Generator().manual_seed(2031)
    choices = torch.tensor([select_output_index(4, generator=output_rng) for _ in range(12000)])
    observed = torch.bincount(choices, minlength=4) / choices.numel()
    assert set(choices.tolist()) == {0, 1, 2, 3}  # excludes P_T, includes P_0
    torch.testing.assert_close(observed.double(), weights, atol=.02, rtol=0)
    assert torch.equal(oracle_state, oracle_rng.get_state())
    assert select_output_index(1, generator=output_rng) == 0
    with pytest.raises(ValueError, match="horizon"):
        select_output_index(0, generator=output_rng)


def test_actual_marginals_are_audited_even_if_solver_claims_success():
    a, b, plan = _problem()
    def inaccurate_prox(logP, G, **kwargs):
        wrong = plan * .9
        return wrong, wrong.log(), {"converged": True, "marginal_l1": 0.}
    with pytest.raises(ProximalSolveError, match="Actual summed marginal residual"):
        smd_step(plan.log(), torch.zeros_like(plan), iteration=0, cmax=1.,
                 prox=inaccurate_prox, marginal_residuals=lambda Q: marginal_residuals(Q, a, b),
                 inner_tol=1e-12, inner_max=10000)


def test_iteration_cap_failure_is_explicit():
    a, b, plan = _problem()
    G = torch.tensor([[20., -15.], [-20., 10.]], dtype=torch.float64)
    with pytest.raises(ProximalSolveError):
        smd_step(plan.log(), G, iteration=0, cmax=1.,
                 prox=_prox(a, b),
                 marginal_residuals=lambda Q: marginal_residuals(Q, a, b),
                 inner_tol=1e-13, inner_max=1)


def test_feasible_result_without_optimality_certificate_is_rejected():
    a, b, plan = _problem()
    def feasibility_only(logP, G, **kwargs):
        return plan, plan.log(), {"converged": True, "marginal_l1": 0.}
    with pytest.raises(ProximalSolveError, match="Missing numerical"):
        smd_step(plan.log(), torch.zeros_like(plan), iteration=0, cmax=1.,
                 prox=feasibility_only, marginal_residuals=lambda Q: marginal_residuals(Q, a, b),
                 inner_tol=1e-7, inner_max=100)


def test_feasible_result_with_excessive_gap_is_rejected():
    a, b, plan = _problem()
    def excessive_gap(logP, G, **kwargs):
        return plan, plan.log(), {"converged": True, "proximal_optimality_certified": True,
            "primal_objective": .1, "dual_objective": .099999,
            "primal_dual_gap": 1e-6, "normalized_primal_dual_gap": 1e-6,
            "primal_witness_marginal_l1": 0.,
            "stable_primal_dual_gap": 1e-6, "gap_roundoff_tolerance": 1e-12}
    with pytest.raises(ProximalSolveError, match="Normalized primal-dual gap"):
        smd_step(plan.log(), torch.zeros_like(plan), iteration=0, cmax=1.,
                 prox=excessive_gap, marginal_residuals=lambda Q: marginal_residuals(Q, a, b),
                 inner_tol=1e-7, inner_max=100)


def test_small_step_roundoff_certificate_remains_accepted():
    a = torch.full((2,), .5, dtype=torch.float64)
    plan = torch.outer(a, a)
    gradient = torch.tensor([[0., 3.], [3., 0.]], dtype=torch.float64)
    solver = _prox(a, a)
    actual, _, info = smd_step(plan.log(), gradient, iteration=0, cmax=125000.,
        prox=solver, marginal_residuals=solver.marginal_residuals,
        inner_tol=1e-7, inner_max=100, check_every=1)
    expected = _scalar_transport_optimum(plan, gradient, a, a, 2e-6)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)
    assert info["normalized_gap_upper_bound"] <= 1e-7


@pytest.mark.parametrize("raw_gap,stable_gap,roundoff", [
    (-5e-11, 0., 6e-8),       # expected FP64 cancellation: permitted
    (-5e-8, 0., 1e-9),        # signed error exceeds its claimed allowance
    (8e-8, 8e-8, 3e-8),      # raw gap alone passes but guarded gap does not
    (0., 0., 2e-7),           # arithmetic uncertainty consumes the tolerance
])
def test_wrapper_independently_audits_roundoff(raw_gap, stable_gap, roundoff):
    a, b, plan = _problem()
    def supplied_certificate(logP, G, **kwargs):
        return plan, plan.log(), {"converged": True, "proximal_optimality_certified": True,
            "primal_objective": raw_gap, "dual_objective": 0.,
            "primal_dual_gap": raw_gap, "normalized_primal_dual_gap": raw_gap,
            "primal_witness_marginal_l1": 0.,
            "stable_primal_dual_gap": stable_gap, "gap_roundoff_tolerance": roundoff}
    def attempt():
        return smd_step(plan.log(), torch.zeros_like(plan), iteration=0, cmax=1.,
            prox=supplied_certificate, marginal_residuals=lambda Q: marginal_residuals(Q,a,b),
            inner_tol=1e-7, inner_max=100)
    if raw_gap == -5e-11:
        assert attempt()[2]["normalized_gap_upper_bound"] <= 1e-7
    else:
        with pytest.raises(ProximalSolveError):
            attempt()
