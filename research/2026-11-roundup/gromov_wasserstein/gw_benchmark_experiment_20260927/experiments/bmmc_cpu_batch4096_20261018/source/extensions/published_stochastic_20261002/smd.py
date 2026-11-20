"""Reference components for Fatkhullin--He (AISTATS 2024), equation (2).

The entropic specialization uses a FRESH unbiased gradient estimate in
    argmin_{Q in U(a,b)} <G_k,Q> + KL(Q || P_k)/eta_k.
There is no gradient averaging or iterate extrapolation. For the squared GW
objective without a factor of 1/2, the conservative schedule here is
eta_k = 1/(4*Cmax*sqrt(k+1)), with zero-based k. The experiment runner uses this update and records numerical solve certificates.

The paper's mathematical update requires an exact minimizer. A finite
Sinkhorn tolerance is a numerical implementation choice, NOT a theorem
certificate. This module requires a numerical primal-dual certificate as
well as an independent actual-plan marginal audit. These certify the
selected numerical stopping criteria, not an outer convergence theorem
for inexact arithmetic.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Callable

import torch


class ProximalSolveError(RuntimeError):
    """A failed numerical solve, with its certificate retained for the run log."""

    def __init__(self, message, *, certificate=None):
        super().__init__(message)
        self.certificate = ({key: (value if not isinstance(value, float)
                                  or math.isfinite(value) else str(value))
                             for key, value in certificate.items()}
                            if certificate is not None else None)


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def cost_bound_from_extrema(min_a: float, max_a: float,
                            min_b: float, max_b: float) -> float:
    """Maximum squared pairwise distance difference from GLOBAL extrema.

    Distributed callers must reduce extrema over every row shard first.
    All supplied extrema must describe finite, nonnegative distance entries.
    """
    lo_a, hi_a, lo_b, hi_b = (
        _nonnegative_finite(x, name) for x, name in zip(
            (min_a, max_a, min_b, max_b), ("min_a", "max_a", "min_b", "max_b")))
    if lo_a > hi_a or lo_b > hi_b:
        raise ValueError("Distance extrema must be ordered.")
    difference = max(abs(lo_a - hi_b), abs(hi_a - lo_b))
    result = difference * difference
    if not math.isfinite(result):
        raise ValueError("Squared GW cost bound overflowed.")
    return result


def cost_bound_from_distances(A: torch.Tensor, B: torch.Tensor,
                             *, symmetry_atol: float = 1e-12) -> float:
    """Validate complete canonical distance matrices and obtain Cmax.

    A canonical distance matrix here is square, finite, nonnegative,
    symmetric, with zero diagonal (absolute tolerance ``symmetry_atol``).
    No normalization or clipping is performed. Use global extrema instead
    when matrices are sharded; this helper requires the complete matrices.
    """
    tolerance = _nonnegative_finite(symmetry_atol, "symmetry_atol")
    for name, matrix in (("A", A), ("B", B)):
        if not isinstance(matrix, torch.Tensor) or not matrix.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor.")
        if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be a nonempty square matrix.")
        if not bool(torch.isfinite(matrix).all()) or bool((matrix < 0).any()):
            raise ValueError(f"{name} must have finite, nonnegative entries.")
        if (not torch.allclose(matrix, matrix.T, atol=tolerance, rtol=0)
                or bool((matrix.diagonal().abs() > tolerance).any())):
            raise ValueError(f"{name} must be symmetric with zero diagonal.")
    return cost_bound_from_extrema(A.min().item(), A.max().item(),
                                  B.min().item(), B.max().item())


def step_size(iteration: int, cmax: float, *, prefactor_fraction: float = 1.) -> float:
    """Zero-based safe schedule; eta0=1 if the entire GW cost is zero.

    A fraction in (0,1] allows prefactor tuning within the analytic upper
    bound. The cmax=0 branch still follows 1/sqrt(k+1); the objective is identically
    zero, so it does not need division by its zero smoothness bound.
    """
    k = _integer(iteration, "iteration", 0)
    bound = _nonnegative_finite(cmax, "cmax")
    fraction = float(prefactor_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("prefactor_fraction must lie in (0,1].")
    value = fraction * (1.0 if bound == 0 else 0.25 / bound) / math.sqrt(k + 1)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Step size is not representable as a positive finite float.")
    return value


def output_probabilities(horizon: int) -> torch.Tensor:
    """Weights for P_0,...,P_{T-1}; P_T is NOT a candidate output.

    The common step-size constant cancels, leaving probabilities
    proportional to 1/sqrt(k+1). Horizon T must be fixed independently of
    the iterates; choosing an objective-minimizing checkpoint is different.
    """
    T = _integer(horizon, "horizon", 1)
    weights = torch.arange(1, T + 1, dtype=torch.float64).rsqrt()
    return weights / weights.sum()


def select_output_index(horizon: int, *, generator: torch.Generator) -> int:
    """Draw an output using an explicitly supplied independent CPU RNG.

    The caller must seed/keep this RNG independently of the gradient RNG.
    Reusing the oracle RNG or choosing T adaptively is outside this helper's
    fixed-horizon output rule. No global random generator is touched.
    """
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("An explicit independent CPU torch.Generator is required.")
    return int(torch.multinomial(output_probabilities(horizon), 1,
                                 generator=generator).item())


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@torch.no_grad()
def smd_step(log_plan: torch.Tensor, fresh_gradient: torch.Tensor, *,
             iteration: int, cmax: float, prox: Callable,
             marginal_residuals: Callable, inner_tol: float, inner_max: int,
             check_every: int = 1, prefactor_fraction: float = 1., gap_tol: float = 1e-7,
             require: Callable = _require
             ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Apply one fresh-gradient entropic SMD step with explicit auditing.

    ``prox(logP,G,eta=...,tol=...,max_iter=...,check_every=...)`` returns
    ``(Q,logQ,info)``. Use ``CertifiedKLProx``; historical feasibility-only proximal solvers
    do not provide the required numerical certificate. ``marginal_residuals``
    must independently recompute ``both_l1`` from the actual stored Q with
    FP64 accumulation, globally if sharded.
    Distributed callers must also supply a collective ``require`` callback
    so rank-local validation failures do not strand peers in collectives;
    all ranks must use identical iteration, cmax and inner-solve settings.

    ``inner_tol`` and ``inner_max`` have no inherited defaults. Retain the
    returned log-plan as state even if entries of Q underflow. Underflow is
    reported by the supplied proximal solver. Freshness/unbiasedness of G
    is the caller's responsibility; this function has no momentum state.
    The retained log-state may be FP64 while the physical plan/gradient are
    FP32. Acceptance requires both the marginal and normalized gap criteria.
    """
    eta = step_size(iteration, cmax, prefactor_fraction=prefactor_fraction)
    tolerance = _nonnegative_finite(inner_tol, "inner_tol")
    gap_tolerance = _nonnegative_finite(gap_tol, "gap_tol")
    if tolerance == 0 or gap_tolerance == 0:
        raise ValueError("inner_tol and gap_tol must be strictly positive.")
    maximum = _integer(inner_max, "inner_max", 1)
    interval = _integer(check_every, "check_every", 1)
    require(isinstance(log_plan, torch.Tensor) and isinstance(fresh_gradient, torch.Tensor),
            "The log-plan and fresh gradient must be tensors.")
    require(log_plan.ndim == 2 and log_plan.shape == fresh_gradient.shape
            and log_plan.dtype in (torch.float32, torch.float64)
            and fresh_gradient.dtype in (torch.float32, torch.float64)
            and torch.finfo(log_plan.dtype).bits >= torch.finfo(fresh_gradient.dtype).bits
            and fresh_gradient.device == log_plan.device,
            "The log-plan and gradient must be compatible floating-point matrices.")
    require(bool(torch.isfinite(log_plan).all() and torch.isfinite(fresh_gradient).all()),
            "The retained log-plan and fresh gradient must be finite.")
    plan, log_next, metadata = prox(log_plan, fresh_gradient, eta=eta, tol=tolerance,
                                    max_iter=maximum, check_every=interval)
    require(isinstance(plan, torch.Tensor) and isinstance(log_next, torch.Tensor),
            "The proximal solver must return tensors.")
    require(plan.shape == log_plan.shape and log_next.shape == log_plan.shape
            and plan.dtype == fresh_gradient.dtype and log_next.dtype == torch.float64
            and plan.device == log_plan.device and log_next.device == log_plan.device,
            "The proximal solver returned incompatible tensors.")
    require(bool(torch.isfinite(plan).all() and torch.isfinite(log_next).all()
                 and (plan >= 0).all()), "The proximal solve returned invalid values.")
    audited = float(marginal_residuals(plan)["both_l1"])
    if not math.isfinite(audited) or audited < 0 or audited > tolerance:
        raise ProximalSolveError(
            f"Actual summed marginal residual {audited:.17g} exceeds tolerance {tolerance:.17g}.",
            certificate=metadata)
    if not metadata.get("converged", False):
        raise ProximalSolveError("The proximal solver reported nonconvergence.", certificate=metadata)
    if metadata.get("proximal_optimality_certified") is not True:
        raise ProximalSolveError("Missing numerical primal-dual optimality certificate.", certificate=metadata)
    required = ("primal_objective", "dual_objective", "primal_dual_gap",
                "normalized_primal_dual_gap", "primal_witness_marginal_l1",
                "stable_primal_dual_gap", "gap_roundoff_tolerance")
    try:
        primal, dual, gap, normalized, witness_residual, stable_gap, roundoff = (
            float(metadata[key]) for key in required)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProximalSolveError("Incomplete numerical primal-dual certificate.", certificate=metadata) from exc
    if not all(math.isfinite(value) for value in
               (primal, dual, gap, normalized, witness_residual, stable_gap, roundoff)):
        raise ProximalSolveError("Nonfinite numerical primal-dual certificate.", certificate=metadata)
    scale = max(1., abs(primal), abs(dual))
    identity_tolerance = 1e-12 * scale
    if witness_residual < 0 or witness_residual > 1e-12:
        raise ProximalSolveError("The primal witness is not feasible to FP64 audit tolerance.", certificate=metadata)
    if (roundoff < 0 or roundoff / scale > gap_tolerance
            or min(gap, stable_gap) < -roundoff
            or abs(gap - stable_gap) > roundoff
            or abs((primal - dual) - gap) > identity_tolerance):
        raise ProximalSolveError("Invalid primal-dual gap or inconsistent objective values.", certificate=metadata)
    if abs(normalized - gap / scale) > 1e-12:
        raise ProximalSolveError("Inconsistent normalized primal-dual gap.", certificate=metadata)
    # Independently enforce the conservative upper gap used by the kernel.
    # Cancellation may make raw J-D slightly negative; the reported FP64
    # roundoff allowance must itself fit within the selected tolerance.
    upper_normalized = (max(gap, stable_gap) + roundoff) / scale
    if upper_normalized > gap_tolerance:
        raise ProximalSolveError(
            f"Normalized primal-dual gap upper bound {upper_normalized:.17g} exceeds {gap_tolerance:.17g}.",
            certificate=metadata)
    info = dict(metadata)
    info.update(iteration=int(iteration), eta=eta, prefactor_fraction=float(prefactor_fraction),
                marginal_l1=audited, inner_tolerance=tolerance,
                normalized_gap_tolerance=gap_tolerance,
                normalized_gap_upper_bound=upper_normalized, proximal_optimality_certified=True)
    return plan, log_next, info
