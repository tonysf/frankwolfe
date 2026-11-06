"""Cancellation-aware arithmetic for the SAME unscaled KL proximal certificate.

Only certificate evaluation changes. Sinkhorn, the FP32-derived feasible
witness, and both acceptance tolerances are inherited unchanged. These are
FP64 numerical error allowances, not directed-rounding interval certificates.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'published_stochastic_20261002'))
from certified_prox import CertifiedKLProx


def _expm1_minus_x(x):
    """exp(x)-1-x, avoiding cancellation at zero with a degree-12 series."""
    polynomial = torch.full_like(x, 1 / math.factorial(12))
    for order in range(11, 1, -1):
        polynomial.mul_(x).add_(1 / math.factorial(order))
    series = x.square() * polynomial
    return torch.where(x.abs() <= .125, series, torch.expm1(x) - x)


def _kl_with_error(witness, reference, ratio, ratio_error):
    """KL entry and an absolute numerical error allowance.

    ratio approximates log(witness/reference). Close entries use the quadratic
    remainder witness*(exp(-ratio)-1+ratio). The allowance exploits that its
    derivative is reference-witness; it does not charge a unit mass when that
    derivative vanishes. Far/zero entries use the ordinary noncancelling form.
    """
    unit = torch.finfo(torch.float64).eps / 2
    positive = witness > 0
    close = positive & (ratio.abs() <= .125)
    safe_ratio = torch.where(positive, ratio, torch.zeros_like(ratio))
    near = witness * _expm1_minus_x(-safe_ratio)
    general = witness * safe_ratio - witness + reference
    kl = torch.where(close, near, general)
    arithmetic_scale = torch.where(close, kl.abs(),
                                    witness * safe_ratio.abs() + witness + reference)
    error = 32 * unit * arithmetic_scale
    error = error + (witness - reference).abs() * ratio_error
    error = error + .5 * reference * ratio_error.square() * ratio_error.exp()
    return kl, error


class StableCertifiedKLProx(CertifiedKLProx):
    """Use stable KL remainders and dual mass differences without changing J.

    The inherited primal witness still comes from the returned plan, including
    FP32 quantization. A genuinely inaccurate witness can therefore still fail
    at a small eta; this evaluator does not replace it by the FP64 candidate.
    """

    @torch.no_grad()
    def certificate(self, Q, logP, G, eta, *, row_potential, col_potential):
        p = self.problem
        rows, cols = self._masses(Q)
        row_scale = torch.minimum(torch.ones_like(self.a), self.a / rows.clamp_min(
            torch.finfo(torch.float64).tiny))
        scaled_cols = self._zero(p.m)
        for block in self._blocks():
            scaled_cols.add_((Q[block].double() * row_scale[block, None]).sum(0))
        p._reduce(scaled_cols)
        col_scale = torch.minimum(torch.ones_like(self.b), self.b / scaled_cols.clamp_min(
            torch.finfo(torch.float64).tiny))
        base_rows, base_cols = self._zero(p.local_rows), self._zero(p.m)
        for block in self._blocks():
            base = Q[block].double() * row_scale[block, None] * col_scale[None, :]
            base_rows[block] = base.sum(1)
            base_cols.add_(base.sum(0))
        p._reduce(base_cols)
        r = (self.a - base_rows).clamp_min(0)
        s = (self.b - base_cols).clamp_min(0)
        deficit = p._reduce(r.sum()).item()
        correction_col = s / deficit if deficit > 0 else torch.zeros_like(s)
        # Gradient, primal KL, dual mass difference, independent gap KL,
        # reference/dual mass, witness correction, four absolute term sums,
        # and three entrywise arithmetic-error allowances.
        totals = self._zero(14)
        witness_rows, witness_cols = self._zero(p.local_rows), self._zero(p.m)
        unit = torch.finfo(torch.float64).eps / 2
        for block in self._blocks():
            original = Q[block].double()
            witness = original * row_scale[block, None] * col_scale[None, :]
            witness.add_(r[block, None] * correction_col[None, :])
            lp, g = logP[block].double(), G[block].double()
            u, v = row_potential[block, None], col_potential[None, :]
            reference = lp.exp()
            # Form this directly: subtracting two large logarithms first would
            # lose O(eps*|logP|), then amplify that loss by 1/eta in the dual.
            delta = -eta * g + u + v
            delta_error = 8 * unit * (eta * g.abs() + u.abs() + v.abs())
            dual_plan = reference * delta.exp()
            mass_difference = -reference * torch.expm1(delta)
            positive = witness > 0
            logw = torch.where(positive, witness.log(), torch.zeros_like(witness))
            nearby_reference = positive & (reference >= torch.finfo(torch.float64).tiny) & (
                (witness - reference).abs() <= .5 * reference)
            relative = (witness - reference) / reference.clamp_min(torch.finfo(torch.float64).tiny)
            # Clamp only the unselected expression, never a reference or plan.
            near_ratio = torch.log1p(relative.clamp_min(-.75))
            ratio = torch.where(nearby_reference, near_ratio, logw - lp)
            ratio = torch.where(positive, ratio, torch.zeros_like(ratio))
            ratio_error = 16 * unit * (1 + lp.abs() + logw.abs())
            primal_kl, primal_kl_error = _kl_with_error(witness, reference, ratio, ratio_error)
            gap_ratio = torch.where(positive, ratio - delta, torch.zeros_like(ratio))
            gap_kl, gap_kl_error = _kl_with_error(
                witness, dual_plan, gap_ratio, ratio_error + delta_error)
            dual_error = 16 * unit * mass_difference.abs() + dual_plan * (
                delta_error + .5 * delta_error.square() * delta_error.exp())
            gradient_terms = g * witness
            entries = (gradient_terms, primal_kl, mass_difference, gap_kl,
                       reference, dual_plan, (witness-original).abs(),
                       gradient_terms.abs(), primal_kl.abs(), mass_difference.abs(),
                       gap_kl.abs(), primal_kl_error, gap_kl_error, dual_error)
            for index, entry in enumerate(entries):
                totals[index].add_(entry.sum())
            witness_rows[block] = witness.sum(1)
            witness_cols.add_(witness.sum(0))
        p._reduce(totals)
        p._reduce(witness_cols)
        row_error = p._reduce((witness_rows - self.a).abs().sum())
        witness_error = (row_error + (witness_cols - self.b).abs().sum()).item()
        row_dual = p._reduce((self.a * row_potential).sum()).item()
        col_dual = (self.b * col_potential).sum().item()
        row_dual_abs = p._reduce((self.a * row_potential).abs().sum()).item()
        col_dual_abs = (self.b * col_potential).abs().sum().item()
        row_correction = row_potential * (witness_rows - self.a)
        col_correction = col_potential * (witness_cols - self.b)
        correction = (p._reduce(row_correction.sum()) + col_correction.sum()).item()
        correction_abs = (p._reduce(row_correction.abs().sum()) + col_correction.abs().sum()).item()
        correction_sensitivity = (p._reduce((row_potential.abs() * (
            witness_rows.abs() + self.a.abs())).sum()) + (col_potential.abs() * (
            witness_cols.abs() + self.b.abs())).sum()).item()
        values = totals.tolist()
        primal = values[0] + values[1] / eta
        dual = (row_dual + col_dual + values[2]) / eta
        gap = primal - dual
        stable_gap = (values[3] + correction) / eta
        normalizer = max(1., abs(primal), abs(dual))
        # A serial reduction of one block is a conservative depth bound for
        # any ordinary parallel tree. Include block accumulation, rank scalar
        # reductions, vector reductions, and scalar assembly. This deliberately
        # avoids assuming a particular CUDA reduction tree.
        block_entries = min(self.block_rows, p.n) * p.m
        operation_depth = max(block_entries, p.n, p.m) + math.ceil(p.n / self.block_rows) + p.world_size + 128
        product = operation_depth * unit
        if product >= .25:
            raise ValueError('Certificate dimensions exceed supported FP64 error model.')
        gamma = product / (1 - product)
        primal_error = gamma * (values[7] + values[8] / eta) + values[11] / eta
        dual_error = (gamma * (row_dual_abs + col_dual_abs + values[9]) + values[13]) / eta
        gap_error = (gamma * (values[10] + correction_abs + correction_sensitivity) + values[12]) / eta
        roundoff = (primal_error + dual_error + gap_error +
                    8 * unit * (abs(primal) + abs(dual) + abs(stable_gap))) / (1 - gamma)
        finite = all(math.isfinite(x) for x in (primal, dual, gap, stable_gap, witness_error, roundoff))
        valid = (finite and witness_error <= self.feasibility_roundoff and
                 min(gap, stable_gap) >= -roundoff and abs(gap - stable_gap) <= roundoff and
                 roundoff / normalizer <= self.gap_tol)
        return dict(primal_objective=primal, dual_objective=dual, primal_dual_gap=gap,
                    normalized_primal_dual_gap=gap / normalizer, gap_normalizer=normalizer,
                    gap_roundoff_tolerance=roundoff, stable_primal_dual_gap=stable_gap,
                    gap_arithmetic_crosscheck_error=abs(gap-stable_gap),
                    primal_witness_marginal_l1=witness_error,
                    primal_witness_feasibility_tolerance=self.feasibility_roundoff,
                    primal_witness_plan_l1_difference=values[6], reference_mass=values[4],
                    dual_candidate_mass=values[5], certificate_valid=bool(valid),
                    proximal_optimality_certified=bool(valid and (max(gap,stable_gap)+roundoff)/normalizer <= self.gap_tol),
                    arithmetic_certificate='FP64 stable KL remainders and direct dual mass differences; not interval-certified',
                    certificate_primal='FP64 rounded feasible witness of returned plan',
                    certificate_method='stable_unscaled_kl_v1',
                    certificate_reduction_gamma=gamma,
                    certificate_reduction_depth_bound=operation_depth,
                    certificate_primal_roundoff=primal_error,
                    certificate_dual_roundoff=dual_error,
                    certificate_gap_roundoff=gap_error,
                    certificate_error_scope='Ordinary IEEE reduction model and conservative FP64 transcendental-error allowances; same unscaled J and tolerances',
                    plan_dtype=str(Q.dtype), retained_log_dtype='torch.float64')
