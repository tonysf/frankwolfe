"""Ding--Li--Toh (JMLR 2025), equations (8)--(9), Theorem 24.

The constrained entropy proximal update uses the published momentum recursion
and its coupled theoretical step sizes. The numerical certificate is the same
as the independently validated plain-SMD implementation; finite tolerances do
not supply an inexact version of the paper's exact-proximal theorem.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Callable

import torch

PUBLISHED_ROOT = Path(__file__).resolve().parents[1] / 'published_stochastic_20261002'
sys.path.insert(0, str(PUBLISHED_ROOT))
from smd import (ProximalSolveError, _integer, _nonnegative_finite, _require,
                 output_probabilities, select_output_index)


def _fraction(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f'{name} must lie in (0,1].')
    return value



def theorem_constants(*, hessian_bound: float, maximum_entry: float, **evidence) -> dict:
    """Constants for phi=M sum psi_M, with M=max feasible coupling entry.

    psi_M agrees with t log(t)-t up to M and has quadratic tail curvature 1/M.
    Thus phi is globally 1-strongly convex and Dphi=M KL on feasible pairs.
    The ambient GW Hessian norm bound L gives LF=L and kappa=2 L^2.
    """
    L, M = float(hessian_bound), float(maximum_entry)
    if not all(math.isfinite(x) and x > 0 for x in (L, M)):
        raise ValueError('hessian_bound and maximum_entry must be positive finite.')
    kappa = 2*L*L
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError('kappa is not representable as a positive finite float.')
    return dict(hessian_bound=L, maximum_feasible_entry=M, kernel_scale=M, mu=1.,
                relative_smoothness=L, kappa=kappa,
                coupling_c_max=1/(2*math.sqrt(kappa)),
                kernel='phi=M*sum(psi_M), entropy up to M and quadratic tail with curvature 1/M',
                feasible_divergence='D_phi(Q,P)=M*KL(Q||P)',
                norm='Frobenius', **evidence)


@torch.no_grad()
def constants_from_problem(problem, *, block_rows=256) -> dict:
    """Compute a bounded-memory, row-sharded ambient GW Hessian bound.

    W_(ij,kl)=(A_ik-B_jl)^2 is symmetric nonnegative, hence ||2W||_2 is
    bounded by twice its maximum row sum. We form only row statistics and
    blocks of n-by-m expressions, never W or a full FP64 distance copy.
    The explicit FP64 guard accounts for cancellation in that expression;
    it is a conservative numerical guard, not directed interval arithmetic.
    """
    block_rows = _integer(block_rows, 'block_rows', 1)
    exact_symmetric = True
    for start in range(0, problem.m, block_rows):
        stop = min(start+block_rows, problem.m)
        exact_symmetric = bool(torch.equal(problem.B[start:stop], problem.B[:,start:stop].T)) and exact_symmetric
    for start, block in problem._broadcast_rows(problem.A):
        stop = start+block.shape[0]
        exact_symmetric = bool(torch.equal(problem.A[:,start:stop],
                                          block[:,problem.row_start:problem.row_stop].T)) and exact_symmetric
    problem._require(exact_symmetric,
                     'The MSBPG Hessian bound and oracle require exactly symmetric loaded distance matrices.')
    def row_stats(matrix):
        sums = torch.empty(matrix.shape[0], dtype=torch.float64, device=matrix.device)
        squares = torch.empty_like(sums)
        for start in range(0, matrix.shape[0], block_rows):
            block = slice(start, start+block_rows)
            values = matrix[block].double()
            sums[block] = values.sum(1)
            squares[block] = values.square().sum(1)
        return sums, squares
    a_sum, a_square = row_stats(problem.A)
    b_sum, b_square = row_stats(problem.B)
    local_max = torch.zeros((), dtype=torch.float64, device=problem.device)
    local_positive = torch.zeros_like(local_max)
    for start in range(0, problem.local_rows, block_rows):
        block = slice(start, start+block_rows)
        positive = problem.m*a_square[block,None] + problem.n*b_square[None,:]
        cross = 2*a_sum[block,None]*b_sum[None,:]
        local_max = torch.maximum(local_max, (positive-cross).max())
        local_positive = torch.maximum(local_positive, (positive+cross.abs()).max())
    row_max = problem.global_max(local_max).item()
    positive_max = problem.global_max(local_positive).item()
    unit_roundoff = torch.finfo(torch.float64).eps/2
    arithmetic_operations = 8*(problem.n+problem.m)+64
    product = arithmetic_operations*unit_roundoff
    if product >= .5:
        raise ValueError('Dimensions exceed the supported FP64 bound calculation.')
    gamma = product/(1-product)
    guard = 2*gamma/(1-gamma)*positive_max
    raw_bound = 2*row_max
    L = math.nextafter(raw_bound+guard, math.inf) if positive_max else 1.
    M = min(problem.a.max().item(), problem.b.max().item())
    result = theorem_constants(hessian_bound=L, maximum_entry=M,
        hessian_bound_method='2*max_ij(m*sum_k A_ik^2+n*sum_l B_jl^2-2*sum_k A_ik*sum_l B_jl)',
        raw_hessian_bound=raw_bound, hessian_fp64_absolute_guard=guard,
        hessian_fp64_guard_gamma=gamma, hessian_bound_zero_objective_convention=(positive_max == 0),
        hessian_bound_input_dtype=str(problem.dtype), hessian_statistics_dtype='torch.float64',
        exact_loaded_distance_symmetry_verified=True,
        bound_arithmetic_scope='Conservative FP64 guard, not a formal interval certificate',
        parameter_scope='Constants apply to the distance and marginal values loaded by the optimizer')
    return result


def schedule(iteration: int, constants: dict, *, coupling_fraction: float = 1.,
             momentum_prefactor: float = 1.) -> dict:
    """Zero-indexed equations (8)--(9): alpha_k uses beta_(k+1), not beta_k.

    beta_j=theta/sqrt(j) for j>=1; v_0 is initialized by the first gradient,
    so no beta_0 is used. alpha is the paper's step for the scaled kernel;
    eta=alpha/kernel_scale is the effective step passed to the KL solver.
    """
    k = _integer(iteration, 'iteration', 0)
    fraction = _fraction(coupling_fraction, 'coupling_fraction')
    theta = _fraction(momentum_prefactor, 'momentum_prefactor')
    mu, lf, kappa = (float(constants[name]) for name in ('mu', 'relative_smoothness', 'kappa'))
    scale, maximum = float(constants['kernel_scale']), float(constants['coupling_c_max'])
    if not all(math.isfinite(x) and x > 0 for x in (mu, lf, kappa, scale, maximum)):
        raise ValueError('Theorem constants must be positive finite scalars.')
    c = fraction * maximum
    beta_next = theta / math.sqrt(k+1)
    alpha = c * mu * beta_next
    eta = alpha / scale
    theorem_cap = 1 / (2 * math.sqrt(mu*kappa))
    if c > theorem_cap*(1+1e-14) or not alpha < min(1., 1/mu)/lf:
        raise ValueError('The parameter choice violates the theorem or small-step restriction.')
    if not all(math.isfinite(x) and x > 0 for x in (c, beta_next, alpha, eta)):
        raise ValueError('Schedule is not representable as positive finite numbers.')
    return dict(alpha=alpha, eta=eta, proximal_eta=eta,
                beta=(None if k == 0 else theta/math.sqrt(k)),
                beta_next=beta_next, coupling_c=c, coupling_fraction=fraction,
                momentum_prefactor=theta, theorem_c_upper_bound=theorem_cap,
                alpha_small_step_upper_bound=min(1.,1/mu)/lf,
                theorem_c_bound_satisfied=True, alpha_small_step_bound_satisfied=True)


@torch.no_grad()
def normalized_displacement(log_before, log_after, alpha, *, block_rows=256,
                            global_sum=lambda x: x) -> float:
    """Squared Frobenius displacement/alpha^2 from the retained FP64 plans.

    exp(log_before)*expm1(log_after-log_before) avoids cancellation between
    nearby plan entries. Blocked evaluation avoids a dense FP64 scratch plan.
    ``global_sum`` reduces a scalar across row shards, including empty shards.
    This is a numerical diagnostic, not an exact-proximal theorem certificate.
    """
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be positive and finite.')
    block_rows = _integer(block_rows, 'block_rows', 1)
    value = torch.zeros((), dtype=torch.float64, device=log_before.device)
    for start in range(0, log_before.shape[0], block_rows):
        block = slice(start, start+block_rows)
        before, after = log_before[block].double(), log_after[block].double()
        hi, lo = torch.maximum(before, after), torch.minimum(before, after)
        difference = hi.exp() * (-torch.expm1(lo-hi)) / alpha
        value.add_(difference.square().sum())
    result = float(global_sum(value).item())
    if not math.isfinite(result) or result < 0:
        raise FloatingPointError('Invalid normalized update displacement.')
    return result


@torch.no_grad()
def msbpg_step(log_plan: torch.Tensor, fresh_gradient: torch.Tensor,
               momentum: torch.Tensor | None, *, iteration: int, constants: dict,
               prox: Callable, marginal_residuals: Callable, inner_tol: float,
               inner_max: int, check_every: int = 1, coupling_fraction: float = 1.,
               momentum_prefactor: float = 1., gap_tol: float = 1e-7,
               require: Callable = _require
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """One published EMA/KL step with the existing audited numerical criteria.

    The initial fresh gradient is reused at update zero, so initialization adds
    no oracle call. ``require`` must be collective for a distributed caller.
    All ranks must share constants and scalar settings.
    """
    parameters = schedule(iteration, constants, coupling_fraction=coupling_fraction,
                          momentum_prefactor=momentum_prefactor)
    eta = parameters['eta']
    if iteration == 0:
        require(momentum is None, 'Initial momentum must be absent; v0 is the first fresh gradient.')
        next_momentum = fresh_gradient
    else:
        require(isinstance(momentum, torch.Tensor), 'An existing momentum is required after update zero.')
        require(momentum.shape == fresh_gradient.shape and momentum.dtype == fresh_gradient.dtype
                and momentum.device == fresh_gradient.device,
                'Momentum and gradient must be compatible tensors.')
        beta = parameters['beta']
        next_momentum = (1-beta)*momentum + beta*fresh_gradient
    tolerance = _nonnegative_finite(inner_tol, "inner_tol")
    gap_tolerance = _nonnegative_finite(gap_tol, "gap_tol")
    if tolerance == 0 or gap_tolerance == 0:
        raise ValueError("inner_tol and gap_tol must be strictly positive.")
    maximum = _integer(inner_max, "inner_max", 1)
    interval = _integer(check_every, "check_every", 1)
    require(isinstance(log_plan, torch.Tensor) and isinstance(next_momentum, torch.Tensor),
            "The log-plan and fresh gradient must be tensors.")
    require(log_plan.ndim == 2 and log_plan.shape == next_momentum.shape
            and log_plan.dtype in (torch.float32, torch.float64)
            and next_momentum.dtype in (torch.float32, torch.float64)
            and torch.finfo(log_plan.dtype).bits >= torch.finfo(next_momentum.dtype).bits
            and next_momentum.device == log_plan.device,
            "The log-plan and gradient must be compatible floating-point matrices.")
    require(bool(torch.isfinite(log_plan).all() and torch.isfinite(next_momentum).all()),
            "The retained log-plan and fresh gradient must be finite.")
    plan, log_next, metadata = prox(log_plan, next_momentum, eta=eta, tol=tolerance,
                                    max_iter=maximum, check_every=interval)
    require(isinstance(plan, torch.Tensor) and isinstance(log_next, torch.Tensor),
            "The proximal solver must return tensors.")
    require(plan.shape == log_plan.shape and log_next.shape == log_plan.shape
            and plan.dtype == next_momentum.dtype and log_next.dtype == torch.float64
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
    info.update(iteration=int(iteration), **parameters,
                marginal_l1=audited, inner_tolerance=tolerance,
                normalized_gap_tolerance=gap_tolerance,
                normalized_gap_upper_bound=upper_normalized, proximal_optimality_certified=True)
    return plan, log_next, next_momentum, info
