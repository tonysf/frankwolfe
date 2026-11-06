"""Read-only diagnostics of the actual preupdate FraMES momentum and penalty."""
from __future__ import annotations

import math
import torch


def _ratio(numerator, denominator):
    """A zero reference norm has no relative error; preserve that fact as null."""
    return numerator / denominator if denominator > 0 else None


@torch.no_grad()
def direction_diagnostics(problem, P, momentum, k, args):
    # All actual LMO decisions are recomputed in optimizer precision, including
    # its first-column tie convention. No input tensor is changed in place.
    gamma = args.gamma0 / (k+1)**.75
    rho_scheduled = args.rho0 / math.sqrt(k+1)
    mu = args.mu_scale / problem.m / (k+1)**.25
    columns = problem._reduce(P.sum(0))
    column_error = columns - problem.b
    penalty_vector = column_error / mu
    penalized = momentum + penalty_vector[None, :]
    with_penalty = penalized.argmin(1)
    without_penalty = momentum.argmin(1)

    # Promote the current optimizer snapshot and frozen costs. This full GW
    # gradient is analytical, including the actual off-feasible marginals.
    # Constructing the diagnostic problem and all ensuing work is outside the
    # optimizer clock. It has no random generator and no writable optimizer view.
    reference = type(problem)(problem.A.double(), problem.B.double(),
                              problem.a.double(), problem.b.double(),
                              group=problem.group, exact_block_rows=problem.exact_block_rows)
    promoted = P.double()
    full_gradient = reference.gradient(promoted)
    full_centered = full_gradient - full_gradient.mean(1, keepdim=True)
    d64 = momentum.double()
    momentum_centered = d64 - d64.mean(1, keepdim=True)
    h64 = penalty_vector.double()
    penalty_centered = h64 - h64.mean()
    full_norm = reference.global_norm(full_centered).item()
    momentum_norm = reference.global_norm(momentum_centered).item()
    penalty_norm = math.sqrt(problem.n) * penalty_centered.norm().item()
    error_norm = reference.global_norm(momentum_centered-full_centered).item()
    _, columns64 = reference.marginals(promoted)
    structure_l1 = reference.global_sum((promoted-reference.a_local[:, None]*columns64[None, :]).abs()).item()
    residuals = problem.marginal_residuals(P)
    agreement = problem.global_sum((with_penalty == without_penalty).double()).item()/problem.n

    def choices(indices):
        counts = problem._reduce(torch.bincount(indices, minlength=problem.m))
        masses = reference.a_local.new_zeros(problem.m)
        masses.scatter_add_(0, indices, reference.a_local)
        masses = reference._reduce(masses)
        return dict(selected_column_count=int((counts > 0).sum().item()),
                    largest_row_fraction=counts.max().item()/problem.n,
                    largest_source_mass=masses.max().item())

    g_indices = full_gradient.argmin(1)
    g_penalized_indices = (full_gradient+penalty_vector.double()[None, :]).argmin(1)
    return dict(iteration=k, plan_index=k, momentum_index=k,
        position='after momentum update, before FW step', gamma=gamma, rho_scheduled=rho_scheduled,
        rho_effective=1. if k == 0 else rho_scheduled,
        momentum_initialization_reuses_first_gradient=(k == 0), mu=mu,
        optimizer_dtype=str(P.dtype).removeprefix('torch.'), full_gradient_dtype='float64',
        norm_dtype='float64', row_centering='subtract each row arithmetic mean; unweighted Frobenius norm',
        residuals=residuals, penalty_value=column_error.double().square().sum().item()/(2*mu),
        row_centered_momentum_norm=momentum_norm, row_centered_full_gradient_norm=full_norm,
        row_centered_penalty_norm=penalty_norm,
        row_centered_momentum_error_norm=error_norm,
        penalty_to_momentum_ratio=_ratio(penalty_norm,momentum_norm),
        row_centered_momentum_relative_error=_ratio(error_norm,full_norm),
        lmo_agreement_with_without_penalty=agreement,
        lmo_agreement_momentum_full_gradient_without_penalty=
            reference.global_sum((without_penalty == g_indices).double()).item()/problem.n,
        lmo_agreement_momentum_full_gradient_with_penalty=
            reference.global_sum((with_penalty == g_penalized_indices).double()).item()/problem.n,
        lmo_with_penalty=choices(with_penalty), lmo_without_penalty=choices(without_penalty),
        raw_structure_l1=structure_l1,
        raw_structure_definition='||P_k-a(P_k^T 1)^T||_1, evaluated in float64')
