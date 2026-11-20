"""Composable ordinary-momentum Moreau-smoothed reference solver.

This module implements Protocol section 7, not a theorem or a tuned configuration.
Every exponent and leading schedule constant is supplied explicitly. Updates use
k=0,1,... with gamma_k=gamma0/(k+1)**alpha, rho_k=rho0/(k+1)**beta,
and mu_k=mu0/(k+1)**delta. The first smooth estimate initializes d_0 directly;
rho_0 is recorded but not applied to that initialization.

The sketch and anchor distributions stay fixed and each subsequent iterate draws
fresh independent samples. The anchor implementation is the starter's transparent
Python-loop reference, not a high-performance implementation. Initialization uses
P_0=a b^T by default; a supplied strictly positive feasible P0 is also supported.
PyTorch argmin selects the first column in an exact row tie.

For the supplied manuscript Section 2, use the matrix-space affine constraint
D_tilde={P:P.T@1=b}. Its squared Frobenius distance is ||P.T@1-b||^2/n,
so the manuscript smoothing parameter is beta_k=mu_k/n. This identity
transfers the implemented penalty to the manuscript formulation; it is not
a claim that arbitrary supplied schedules meet the theorem assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Literal

import torch

from gw_core import GWProblem, fw_row_update, rademacher, simplex_rows


@dataclass(frozen=True)
class MomentumConfig:
    alpha: float
    beta: float
    delta: float
    gamma0: float
    rho0: float
    mu0: float
    oracle: Literal['exact', 'sketch', 'anchor']
    batch: int
    update: Literal['fw', 'row_projection'] = 'fw'
    eta0: float | None = None
    eta_exponent: float | None = None
    theory_id: str | None = None
    output_statistic: str | None = None

    def __post_init__(self):
        for name in ('alpha', 'beta', 'delta'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite nonnegative exponent.')
        for name in ('gamma0', 'rho0'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must lie in (0,1].')
        if not math.isfinite(self.mu0) or self.mu0 <= 0:
            raise ValueError('mu0 must be finite and positive.')
        if self.oracle not in ('exact', 'sketch', 'anchor'):
            raise ValueError('oracle must be exact, sketch, or anchor.')
        if isinstance(self.batch, bool) or not isinstance(self.batch, int) or self.batch < 1:
            raise ValueError('batch must be a positive integer (ignored for exact gradients).')
        if self.update not in ('fw', 'row_projection'):
            raise ValueError('update must be fw or row_projection.')
        if self.update == 'row_projection':
            if self.eta0 is None or not math.isfinite(self.eta0) or self.eta0 <= 0:
                raise ValueError('The row-projection ablation requires its own positive eta0.')
            if self.eta_exponent is None or not math.isfinite(self.eta_exponent) or self.eta_exponent < 0:
                raise ValueError('The row-projection ablation requires an explicit nonnegative eta_exponent.')
        elif self.eta0 is not None or self.eta_exponent is not None:
            raise ValueError('eta0 and eta_exponent apply only to the row-projection ablation.')

    def schedule(self, k: int) -> dict[str, float | None]:
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError('The update index k must be a nonnegative integer.')
        index = k + 1
        return {
            'gamma': self.gamma0 / index ** self.alpha,
            'rho': self.rho0 / index ** self.beta,
            'mu': self.mu0 / index ** self.delta,
            'eta': self.eta0 / index ** self.eta_exponent if self.update == 'row_projection' else None,
        }

    def metadata(self, *, initialization: Literal['product', 'provided'] = 'product') -> dict:
        """Describe the default or an explicitly supplied initial coupling."""
        if initialization not in ('product', 'provided'):
            raise ValueError("initialization metadata must be product or provided.")
        return asdict(self) | {
            'schedule_index': 'k=0,1,...; denominator=(k+1)',
            'gamma_formula': 'gamma0/(k+1)**alpha',
            'rho_formula': 'rho0/(k+1)**beta; unused at k=0 because d0=g0',
            'mu_formula': 'mu0/(k+1)**delta',
            'batch_schedule': 'constant; exact oracle does not sample',
            'momentum_initialization': ('first smooth gradient estimate at product P0'
                                        if initialization == 'product' else
                                        'first smooth gradient estimate at supplied feasible P0'),
            'initialization': ('P0=outer(a,b)' if initialization == 'product' else
                               'supplied strictly positive feasible P0'),
            'tie_breaking': 'first column among exact row minima (torch.argmin)',
            'theory_metadata_complete': bool(self.theory_id and self.output_statistic),
            'theory_verification': 'not_verified_by_this_implementation',
        }


@dataclass(frozen=True)
class OracleCounts:
    oracle_evaluations: int = 0
    exact_gradient_calls: int = 0
    stochastic_gradient_calls: int = 0
    sketch_probes: int = 0
    anchor_samples: int = 0
    initialization_oracle_calls: int = 0
    row_lmo_calls: int = 0
    row_projection_calls: int = 0


@dataclass
class MomentumState:
    P: torch.Tensor
    d: torch.Tensor
    iteration: int
    generator: torch.Generator
    seed: int
    config: MomentumConfig
    counts: OracleCounts
    problem: GWProblem


@dataclass(frozen=True)
class StepInfo:
    """References to the pre-update point and estimator, without dense copies.

    P_before, gradient_estimate, and d_used all refer to update_index k. Evaluate
    momentum error as ||d_used-grad f(P_before)||. After step, state.P is P_{k+1}
    while state.d is d_k. Treat returned tensors as read-only. The solver never
    mutates earlier P/d tensors in place; the caller can release this record after
    logging to avoid retaining old dense arrays.
    """
    update_index: int
    completed_updates: int
    P_before: torch.Tensor
    gradient_estimate: torch.Tensor
    d_used: torch.Tensor
    gamma: float
    rho: float
    mu: float
    eta: float | None
    effective_momentum_weight: float
    oracle_evaluations_this_step: int
    counts: OracleCounts


def _sample_gradient(problem: GWProblem, P: torch.Tensor, config: MomentumConfig,
                     generator: torch.Generator, counts: OracleCounts):
    if config.oracle == 'exact':
        gradient = problem.gradient(P)
        counts = replace(counts, exact_gradient_calls=counts.exact_gradient_calls + 1)
    elif config.oracle == 'sketch':
        probes = rademacher(P.shape[1], config.batch, like=P, generator=generator)
        gradient = problem.sketch_gradient(P, probes)
        counts = replace(counts, stochastic_gradient_calls=counts.stochastic_gradient_calls + 1,
                         sketch_probes=counts.sketch_probes + config.batch)
    else:
        n, m = P.shape
        flat = torch.randint(n * m, (config.batch,), device=P.device, generator=generator)
        pairs = torch.stack((flat // m, flat % m), dim=1)
        gradient = problem.anchor_gradient(P, pairs)
        counts = replace(counts, stochastic_gradient_calls=counts.stochastic_gradient_calls + 1,
                         anchor_samples=counts.anchor_samples + config.batch)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError('Smooth gradient estimate contains nonfinite values.')
    return gradient, replace(counts, oracle_evaluations=counts.oracle_evaluations + 1)


@torch.no_grad()
def initialize_state(problem: GWProblem, config: MomentumConfig, *, seed: int,
                     P0: torch.Tensor | None = None) -> MomentumState:
    """Evaluate the first oracle at P0 and initialize d0=g0.

    The default is the product coupling. A supplied point is checked for matching
    shape/dtype/device, strict positivity, and both marginals, then copied. It is
    never repaired or projected. Its construction must use a separate RNG stream
    from the stochastic oracle; this function consumes only the oracle stream.

    This initialization is algorithmic work: its oracle call and samples are
    included in state.counts before any update. A runner must time it as solver
    work, rather than hiding it in diagnostic/setup time.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise ValueError('An explicit integer seed in [0,2**63) is required.')
    if P0 is None:
        P = torch.outer(problem.a, problem.b)
    else:
        if not isinstance(P0, torch.Tensor):
            raise TypeError('P0 must be a torch.Tensor.')
        if P0.shape != (problem.a.numel(), problem.b.numel()):
            raise ValueError('P0 shape must match the problem marginals.')
        if P0.dtype != problem.A.dtype or P0.device != problem.A.device:
            raise ValueError('P0 dtype and device must match the frozen problem.')
        if not torch.isfinite(P0).all() or (P0 <= 0).any():
            raise ValueError('P0 must be finite and strictly positive.')
        tolerance = 32 * torch.finfo(P0.dtype).eps
        row_error = (P0.sum(1) - problem.a).abs().sum().item()
        col_error = (P0.sum(0) - problem.b).abs().sum().item()
        if row_error > tolerance or col_error > tolerance:
            raise ValueError('P0 must satisfy both row and column marginals within '
                             f'entrywise-L1 tolerance {tolerance}; '
                             f'row={row_error}, column={col_error}.')
        P = P0.clone()
    generator = torch.Generator(device=P.device).manual_seed(seed)
    d, counts = _sample_gradient(problem, P, config, generator, OracleCounts())
    counts = replace(counts, initialization_oracle_calls=1)
    return MomentumState(P=P, d=d, iteration=0, generator=generator,
                         seed=seed, config=config, counts=counts, problem=problem)


@torch.no_grad()
def step(problem: GWProblem, state: MomentumState,
         config: MomentumConfig | None = None) -> StepInfo:
    """Advance state once and return matching pre-update diagnostic references.

    The config is frozen at initialization. The optional argument permits callers
    using step(problem,state,config) and must equal that initial configuration.
    No transport rounding, column normalization, or support truncation is applied.
    """
    if problem is not state.problem:
        raise ValueError('Use the same frozen GWProblem object as at initialization.')
    if config is not None and config != state.config:
        raise ValueError('A run configuration cannot change after initialization.')
    config = state.config
    k = state.iteration
    values = config.schedule(k)
    gamma, rho, mu, eta = (values[key] for key in ('gamma', 'rho', 'mu', 'eta'))
    before = state.P
    counts = state.counts
    if k == 0:
        # initialize_state already sampled g(P0), so do not consume a second call.
        gradient = d = state.d
        calls_this_step = 0
    else:
        gradient, counts = _sample_gradient(problem, before, config, state.generator, counts)
        d = (1 - rho) * state.d + rho * gradient
        calls_this_step = 1
    # The penalty is current and exact. It never enters the smooth-gradient EMA.
    H = d + (before.sum(0) - problem.b)[None, :] / mu
    if config.update == 'fw':
        after = fw_row_update(before, H, problem.a, gamma)
        counts = replace(counts, row_lmo_calls=counts.row_lmo_calls + 1)
    else:
        after = simplex_rows(before - eta * H, problem.a)
        counts = replace(counts, row_projection_calls=counts.row_projection_calls + 1)
    if not torch.isfinite(after).all():
        raise FloatingPointError('The update produced nonfinite coupling entries.')
    state.P, state.d, state.iteration, state.counts = after, d, k + 1, counts
    return StepInfo(update_index=k, completed_updates=k + 1, P_before=before,
                    gradient_estimate=gradient, d_used=d, gamma=gamma, rho=rho,
                    mu=mu, eta=eta, effective_momentum_weight=1. if k == 0 else rho,
                    oracle_evaluations_this_step=calls_this_step, counts=counts)
