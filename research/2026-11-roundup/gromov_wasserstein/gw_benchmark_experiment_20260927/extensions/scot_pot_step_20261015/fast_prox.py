"""CPU FP64 Sinkhorn acceleration retaining the original numerical certificate.

The positive row-shifted kernel is formed once, then Sinkhorn scaling uses
matrix-vector products. The accepted physical plan is reconstructed from the
ORIGINAL log-kernel and scaling potentials. The unchanged CertifiedKLProx
certificate checks that original subproblem, not a clipped/truncated kernel.
Kernel underflow or unsafe scalings switch to the original log-domain solver.
Every attempted scaling iteration counts against the same total inner cap.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "extensions/published_stochastic_20261002"))
from certified_prox import CertifiedKLProx
from smd import ProximalSolveError


class FastCertifiedKLProx(CertifiedKLProx):
    """A local CPU FP64 fast path with original-log-domain fallback."""

    @staticmethod
    def _positive_finite(value):
        return bool(torch.isfinite(value).all() and (value > 0).all())

    def _scaling_iteration(self, kernel, v):
        row_denominator = kernel @ v
        if not self._positive_finite(row_denominator):
            raise ArithmeticError("unsafe row scaling denominator")
        u = self.a / row_denominator
        if not self._positive_finite(u):
            raise ArithmeticError("unsafe row scaling factor")
        column_denominator = kernel.T @ u
        if not self._positive_finite(column_denominator):
            raise ArithmeticError("unsafe column scaling denominator")
        v = self.b / column_denominator
        if not self._positive_finite(v):
            raise ArithmeticError("unsafe column scaling factor")
        return u, v

    def _fallback(self, logP, G, eta, tol, max_iter, check_every, *,
                  reason, attempted, certificate_checks, cheap_checks):
        remaining = max_iter - attempted
        if remaining < 1:
            raise ProximalSolveError(
                "Unsafe fast scaling exhausted the total inner-iteration budget",
                certificate=dict(converged=False, failure_reason=reason,
                                 inner_iterations=attempted, certificate_checks=certificate_checks,
                                 fast_scaling_iterations=attempted,
                                 cheap_marginal_checks=cheap_checks, total_inner_budget=max_iter))
        Q, logQ, info = super().__call__(
            logP, G, eta, tol=tol, max_iter=remaining, check_every=check_every)
        info = dict(info)
        info.update(inner_iterations=attempted + info["inner_iterations"],
                    certificate_checks=certificate_checks + info["certificate_checks"],
                    fast_scaling_iterations=attempted,
                    log_domain_iterations=info["inner_iterations"],
                    cheap_marginal_checks=cheap_checks,
                    fast_path_used=attempted > 0, fallback_reason=reason,
                    total_inner_budget=max_iter,
                    sinkhorn_path="original-log-domain-fallback")
        return Q, logQ, info

    @torch.no_grad()
    def __call__(self, logP, G, eta, tol=1e-7, max_iter=10000, check_every=10):
        p = self.problem
        p._validate_plan(G, "gradient")
        p._require(isinstance(logP, torch.Tensor) and logP.shape == G.shape
                   and logP.device == p.device and logP.dtype in (torch.float32, torch.float64),
                   "Retained log-plan must have the local shape/device and floating dtype.")
        p.require_finite(logP, "retained log-plan")
        p._require(math.isfinite(eta) and eta > 0 and math.isfinite(tol) and tol > 0
                   and isinstance(max_iter, int) and not isinstance(max_iter, bool) and max_iter > 0
                   and isinstance(check_every, int) and not isinstance(check_every, bool)
                   and check_every > 0, "Positive step, tolerances and integer limits are required.")

        def fallback(reason, attempted=0, certificate_checks=0, cheap_checks=0):
            return self._fallback(logP, G, eta, tol, max_iter, check_every,
                                  reason=reason, attempted=attempted,
                                  certificate_checks=certificate_checks, cheap_checks=cheap_checks)

        if p.device.type != "cpu" or p.dtype != torch.float64 or p.world_size != 1:
            return fallback("fast path requires local CPU FP64")
        raw_kernel = logP.double() - eta * G.double()
        p.require_finite(raw_kernel, "original log-kernel")
        shift = raw_kernel.max(1).values
        kernel = (raw_kernel - shift[:, None]).exp()
        del raw_kernel
        # Never silently solve a kernel with dropped or nonfinite entries.
        if not self._positive_finite(kernel):
            return fallback("row-shifted kernel has nonpositive or nonfinite entries")
        v = torch.ones_like(self.b)
        checks = cheap_checks = 0
        for iteration in range(1, max_iter + 1):
            try:
                u, v = self._scaling_iteration(kernel, v)
            except ArithmeticError as exc:
                return fallback(str(exc), iteration, checks, cheap_checks)
            if iteration % check_every != 0 and iteration != max_iter:
                continue
            cheap_checks += 1
            row_mass = u * (kernel @ v)
            col_mass = v * (kernel.T @ u)
            quick_residual = ((row_mass - self.a).abs().sum()
                              + (col_mass - self.b).abs().sum()).item()
            if not math.isfinite(quick_residual):
                return fallback("nonfinite scaling marginal residual", iteration, checks, cheap_checks)
            if quick_residual > tol and iteration != max_iter:
                continue
            logu, logv = u.log(), v.log()
            Q, logQ = self._candidate(logP, G, eta, shift, logu, logv)
            if not bool(torch.isfinite(Q).all() and torch.isfinite(logQ).all()):
                return fallback("nonfinite original-log-kernel candidate", iteration, checks, cheap_checks)
            marginal = self.marginal_residuals(Q)["both_l1"]
            info = self.certificate(Q, logP, G, eta,
                                    row_potential=logu - shift, col_potential=logv)
            checks += 1
            if not info["certificate_valid"]:
                return fallback("original-objective certificate invalid after fast scaling",
                                iteration, checks, cheap_checks)
            converged = marginal <= tol and info["proximal_optimality_certified"]
            info.update(inner_iterations=iteration, certificate_checks=checks,
                        marginal_l1=marginal, converged=bool(converged),
                        marginal_tolerance=tol, normalized_gap_tolerance=self.gap_tol,
                        sinkhorn_arithmetic="FP64 CPU matrix-vector scaling; candidate and certificate use original log-kernel",
                        sinkhorn_path="positive-kernel-matrix-vector",
                        fast_path_used=True, fast_scaling_iterations=iteration,
                        log_domain_iterations=0, cheap_marginal_checks=cheap_checks,
                        total_inner_budget=max_iter,
                        underflow_entries=int((Q == 0).sum().item()))
            if converged:
                return Q, logQ, info
            if iteration == max_iter:
                info["failure_reason"] = "inner iteration cap exhausted without both criteria"
                return Q, logQ, info
            del Q, logQ
        raise AssertionError("A final iteration must return a checked candidate")
