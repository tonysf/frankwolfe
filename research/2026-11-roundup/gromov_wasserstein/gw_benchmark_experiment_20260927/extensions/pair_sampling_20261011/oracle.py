"""Conditional iid pair oracle for the unregularized squared GW objective.

For a nonnegative positive-mass coupling P, draw (K,L) with replacement from
P/s, s=sum(P), and return 2*s/q sum_t (A[i,K_t]-B[j,L_t])**2.  This is the
ambient gradient, including when P has incorrect row/column marginals or
nonunit mass.  Symmetric distances are required by RowShardedGW itself.

Only actual row masses, uniforms and pair indices are communicated.  No
coupling is gathered.  Rank zero owns the random stream; conditional CDFs
are resolved on the rank holding each sampled row, in bounded row blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent
METHOD_NAMES = {
    'smoothing-fw': 'FraMES: smoothed row Frank-Wolfe with momentum (pair-sampled gradient)',
    'published-smd': 'Stochastic entropic mirror descent (fresh pair-sampled gradient; no momentum)',
    'published-msbpg': 'MSBPG: stochastic Bregman proximal gradient (pair-sampled gradient; published momentum and coupled schedules)',
}


@dataclass(frozen=True)
class PairSample:
    source: torch.Tensor
    target: torch.Tensor
    mass: torch.Tensor
    row_marginal_l1: torch.Tensor | None = None


def add_arguments(parser):
    parser.add_argument('--oracle', choices=('sketch', 'pair-sampling'), default='sketch')
    parser.add_argument('--pairs-per-call', type=int, default=1024,
                        help='IID coupling pairs per oracle call, with replacement; used only by pair-sampling.')
    return parser


def validate_arguments(args):
    oracle = getattr(args, 'oracle', 'sketch')
    if oracle not in ('sketch', 'pair-sampling'):
        raise ValueError('Unknown gradient oracle.')
    q = getattr(args, 'pairs_per_call', 1024)
    if not isinstance(q, int) or isinstance(q, bool) or q < 1:
        raise ValueError('pairs_per_call must be a positive integer.')
    if args.batch != 128:
        raise ValueError('The legacy sketch batch must remain128; pair-sampling uses --pairs-per-call.')


def _positive_count(problem, value, name):
    problem._require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                     f'{name} must be a positive integer.')
    count = torch.tensor(value, dtype=torch.int64, device=problem.device)
    problem._require_replicated(count, name)


@torch.no_grad()
def fixed_tree_prefix_sum(values):
    """Inclusive last-axis prefixes with a fixed Sklansky addition tree.

    PyTorch2.5.1 rejects *all* floating CUDA cumsum calls under strict
    determinism, before choosing its scan kernel (ScanKernels.cpp). Merely
    adding a second row therefore cannot fix the affected environment.
    https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/ScanKernels.cpp

    Pad each row to a power of two. At each stage both half-blocks already
    contain their local prefixes; add the completed left total to the right
    prefixes. Clone those totals before the in-place, disjoint right-half
    writes. Only deterministic copy/add kernels are used, with no cumsum,
    atomics, dtype conversion, probability quantization or global-mode change.
    The result uses <2 times the input storage, plus <=half that padded size
    for the temporary totals. The same tree applies independently to each row.
    """
    if not isinstance(values, torch.Tensor) or values.ndim not in (1, 2):
        raise ValueError('Prefix scan requires a vector or matrix.')
    columns = values.shape[-1]
    if columns <= 1:
        return values.clone()
    width = 1 << (columns-1).bit_length()
    rows = values.reshape(-1, columns)
    result = values.new_zeros((rows.shape[0], width))
    result[:, :columns].copy_(rows)
    half = 1
    while half < width:
        blocks = result.view(rows.shape[0], width//(2*half), 2*half)
        left_totals = blocks[:, :, half-1:half].clone()
        blocks[:, :, half:].add_(left_totals)
        del left_totals
        half *= 2
    return result[:, :columns].reshape(values.shape)


def cdf_prefix_sum(values):
    """Keep CPU arithmetic unchanged; use strict deterministic CUDA additions."""
    return fixed_tree_prefix_sum(values) if values.is_cuda else values.cumsum(-1)


@torch.no_grad()
def sample_pairs(problem, P, count, *, generator=None, conditional_block_rows=16,
                 uniforms=None):
    """Draw common pairs; ``uniforms`` is an explicit deterministic test input.

    FP64 sums/CDFs use the actual represented plan values, not target a/b.
    Each rank consumes RNG only if it is rank zero.  The two independent
    uniform vectors select a row and a conditional column respectively.
    Zero mass is rejected because P/s has no probability-law definition.
    """
    _positive_count(problem, count, 'pair count')
    _positive_count(problem, conditional_block_rows, 'conditional block rows')
    problem._validate_plan(P)
    nonnegative = torch.ones((), dtype=torch.bool, device=P.device)
    for start in range(0, problem.local_rows, problem.exact_block_rows):
        nonnegative.logical_and_((P[start:start+problem.exact_block_rows] >= 0).all())
    problem._require(nonnegative, 'Pair sampling requires a nonnegative coupling.')
    row_mass = problem._gather_rows(P.sum(1, dtype=torch.float64))
    row_cdf = cdf_prefix_sum(row_mass)
    mass = row_cdf[-1]
    problem._require(torch.isfinite(mass) & (mass > 0),
                     'Pair sampling requires finite positive total coupling mass.')
    if uniforms is None:
        uniforms = torch.empty((2, count), dtype=torch.float64, device=problem.device)
        if problem.rank == 0:
            uniforms.copy_(torch.rand((2, count), dtype=torch.float64,
                                      device=problem.device, generator=generator))
        if problem.world_size > 1:
            dist.broadcast(uniforms, src=problem._global_rank(0), group=problem.group)
    else:
        problem._require(isinstance(uniforms, torch.Tensor)
                         and uniforms.shape == (2, count)
                         and uniforms.dtype == torch.float64 and uniforms.device == problem.device,
                         'Explicit uniforms require shape(2,q), float64 and the problem device.')
        problem._require(torch.isfinite(uniforms).all() & (uniforms >= 0).all() & (uniforms < 1).all(),
                         'Uniform draws must lie in [0,1).')
        problem._require_replicated(uniforms, 'pair uniforms')
    # Normalized CDF endpoints are exactly1. right=True skips zero-mass
    # prefixes/plateaus even for an exactly-zero uniform. No index clipping.
    source = torch.searchsorted(row_cdf / mass, uniforms[0].contiguous(), right=True)
    problem._require((source < problem.n).all() & (row_mass[source] > 0).all(),
                     'Row categorical draw did not select positive represented mass.')
    target = torch.zeros(count, dtype=torch.int64, device=problem.device)
    owned = torch.nonzero((source >= problem.row_start) & (source < problem.row_stop)).flatten()
    local_valid = torch.ones((), dtype=torch.bool, device=problem.device)
    for begin in range(0, owned.numel(), conditional_block_rows):
        positions = owned[begin:begin+conditional_block_rows]
        local_rows = source[positions]-problem.row_start
        cdf = cdf_prefix_sum(P[local_rows].to(torch.float64))
        cdf.div_(cdf[:, -1:].clone())
        columns = torch.searchsorted(cdf, uniforms[1, positions, None].contiguous(), right=True).flatten()
        local_valid.logical_and_((columns < problem.m).all())
        target[positions] = columns
    problem._require(local_valid, 'Conditional categorical draw exceeded a coupling row.')
    problem._reduce(target)
    return PairSample(source, target, mass,
                      (row_mass-problem.a.to(torch.float64)).abs().sum())


@torch.no_grad()
def pair_gradient(problem, sample):
    """Evaluate one specified empirical pair distribution on local output rows.

    The expanded squared-distance expression needs two thin feature matrices
    and one local gradient.  There is no q-by-n-by-m temporary, clipping,
    maximum normalization, feasible-column substitution or hidden oracle draw.
    """
    K, L, mass = sample.source, sample.target, sample.mass
    problem._require(isinstance(K, torch.Tensor) and isinstance(L, torch.Tensor)
                     and K.ndim == L.ndim == 1 and K.numel() == L.numel() and K.numel() > 0
                     and K.dtype == L.dtype == torch.int64
                     and K.device == L.device == problem.device,
                     'Pair indices require equal positive-length int64 vectors on the problem device.')
    _positive_count(problem, K.numel(), 'pair count')
    problem._require((K >= 0).all() & (K < problem.n).all()
                     & (L >= 0).all() & (L < problem.m).all(), 'Pair index out of range.')
    problem._require(isinstance(mass, torch.Tensor) and mass.ndim == 0
                     and mass.device == problem.device and mass.dtype == torch.float64,
                     'Sample mass must be an FP64 scalar on the problem device.')
    problem._require(torch.isfinite(mass) & (mass > 0), 'Sample mass must be positive finite.')
    problem._require_replicated(K, 'sampled source indices')
    problem._require_replicated(L, 'sampled target indices')
    problem._require_replicated(mass, 'sample mass')
    X, Y = problem.A[:, K], problem.B[:, L]
    result = X.square().sum(1)[:, None] + Y.square().sum(1)[None, :]
    result.addmm_(X, Y.T, beta=1., alpha=-2.)
    result.mul_((2*mass/K.numel()).to(problem.dtype))
    problem.require_finite(result, 'pair-sampled gradient')
    return result


def oracle_metadata(args):
    if getattr(args, 'oracle', 'sketch') == 'sketch':
        return dict(name='sketch', probes_per_call=args.batch, pair_samples_per_call=0,
                    gradient_normalization='none', randomness='fresh rank-zero common Rademacher probes')
    return dict(name='pair-sampling', probes_per_call=0, pair_samples_per_call=args.pairs_per_call,
                law='iid with replacement (K,L)~P/s, s=sum(P), conditional on current P',
                estimator='(2*s/q)*sum_t (A[i,K_t]-B[j,L_t])^2',
                gradient_normalization='none; actual total mass factor2s retained',
                target_marginals_substituted=False, sampling_arithmetic='FP64 row sums and CDFs',
                prefix_scan='CUDA fixed-tree FP64 additions; CPU cumsum; strict determinism retained',
                gradient_arithmetic=args.dtype, conditional_block_rows=16,
                randomness='rank-zero independent row and conditional-column uniforms; common global pairs',
                zero_mass_policy='reject undefined P/s',
                communication='row mass vector and O(q) uniforms/indices; no full coupling gather',
                mass_min=None, mass_max=None, mass_last=None, maximum_row_marginal_l1=None)


@torch.no_grad()
def stochastic_gradient(problem, P, args, generator, counters, metadata):
    """Single shared dispatch; initialization reuses its first returned gradient."""
    if getattr(args, 'oracle', 'sketch') == 'sketch':
        gradient = problem.sketch_gradient(P, problem.rademacher(args.batch, generator=generator))
        counters['sketch_probes'] += args.batch
    else:
        sample = sample_pairs(problem, P, args.pairs_per_call, generator=generator)
        gradient = pair_gradient(problem, sample)
        counters['pair_samples'] += args.pairs_per_call
        mass, row_l1 = sample.mass.item(), sample.row_marginal_l1.item()
        metadata['mass_last'] = mass
        metadata['mass_min'] = mass if metadata['mass_min'] is None else min(mass, metadata['mass_min'])
        metadata['mass_max'] = mass if metadata['mass_max'] is None else max(mass, metadata['mass_max'])
        metadata['maximum_row_marginal_l1'] = row_l1 if metadata['maximum_row_marginal_l1'] is None else max(row_l1, metadata['maximum_row_marginal_l1'])
    counters['stochastic_gradient_calls'] += 1
    return gradient
