"""FP64 row-sharded Sinkhorn scaling with the original objective certificate.

Only column matrix-vector products are summed across ranks; vectors and
failure decisions are collective. The positive row-shifted kernel is never
clipped. Any zero/nonfinite kernel or unsafe scaling switches every rank to
the original log-domain solver, within the SAME total iteration budget.

Kernel construction and certificate work use bounded row blocks. The kernel
is released before materializing Q and logQ; accepted plans are reconstructed
from the original log-kernel, not from a truncated exponential kernel.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "extensions/published_stochastic_20261002"))
from certified_prox import CertifiedKLProx
from smd import ProximalSolveError


class DistributedCertifiedPOTProx(CertifiedKLProx):
    """Positive-kernel CPU/CUDA solver; inputs and output remain row-sharded."""

    def _all_good(self, local_condition):
        flag = torch.as_tensor(local_condition, device=self.problem.device,
                               dtype=torch.int32).reshape(())
        return bool(self.problem._reduce(flag, dist.ReduceOp.MIN).item())

    @staticmethod
    def _positive_finite(value):
        return torch.isfinite(value).all() & (value > 0).all()

    def _build_kernel(self, log_reference, gradient, eta, shift):
        kernel = torch.empty(gradient.shape, device=self.problem.device,
                             dtype=torch.float64)
        valid = torch.ones((), device=self.problem.device, dtype=torch.bool)
        for block in self._blocks():
            value = log_reference[block].double() - eta * gradient[block].double()
            shift[block] = value.max(1).values
            value.sub_(shift[block, None]).exp_()
            kernel[block].copy_(value)
            valid.logical_and_(self._positive_finite(value))
        return kernel, self._all_good(valid)

    def _scaling_iteration(self, kernel, v):
        row_denominator = kernel @ v
        u = self.a / row_denominator
        if not self._all_good(self._positive_finite(row_denominator)
                              & self._positive_finite(u)):
            return None, None, None, "unsafe row denominator or scaling factor"
        column_denominator = self.problem._reduce(kernel.T @ u)
        v = self.b / column_denominator
        if not self._all_good(self._positive_finite(column_denominator)
                              & self._positive_finite(v)):
            return None, None, None, "unsafe column denominator or scaling factor"
        return u, v, column_denominator, None

    def _fallback(self, log_reference, gradient, eta, tol, max_iter, check_every,
                  *, reason, attempted, checks, cheap_checks):
        remaining = max_iter - attempted
        if remaining < 1:
            raise ProximalSolveError(
                "Unsafe Sinkhorn scaling exhausted the total inner budget",
                certificate=dict(converged=False, failure_reason=reason,
                                 inner_iterations=attempted, certificate_checks=checks,
                                 fast_scaling_iterations=attempted,
                                 cheap_marginal_checks=cheap_checks,
                                 total_inner_budget=max_iter))
        Q, logQ, info = super().__call__(
            log_reference, gradient, eta, tol=tol, max_iter=remaining,
            check_every=check_every)
        log_iterations = info["inner_iterations"]
        info = dict(info)
        info.update(inner_iterations=attempted + log_iterations,
                    certificate_checks=checks + info["certificate_checks"],
                    fast_scaling_iterations=attempted,
                    log_domain_iterations=log_iterations,
                    cheap_marginal_checks=cheap_checks,
                    fast_path_used=attempted > 0, fallback_reason=reason,
                    total_inner_budget=max_iter,
                    sinkhorn_path="original-log-domain-fallback",
                    collective_fallback=True,
                    kernel_released_before_candidate=True)
        return Q, logQ, info

    @torch.no_grad()
    def __call__(self, log_reference, gradient, eta, tol=1e-7,
                 max_iter=100000, check_every=10):
        p = self.problem
        p._require(p.dtype == torch.float64, "This campaign requires FP64 plans and gradients.")
        p._validate_plan(gradient, "gradient")
        p._require(isinstance(log_reference, torch.Tensor)
                   and log_reference.shape == gradient.shape
                   and log_reference.device == p.device
                   and log_reference.dtype == torch.float64,
                   "The fixed log-reference must have the local FP64 shape and device.")
        p.require_finite(log_reference, "fixed log-reference")
        p._require(math.isfinite(eta) and eta > 0 and math.isfinite(tol) and tol > 0
                   and isinstance(max_iter, int) and not isinstance(max_iter, bool)
                   and max_iter > 0 and isinstance(check_every, int)
                   and not isinstance(check_every, bool) and check_every > 0,
                   "Positive finite step/tolerances and positive integer limits are required.")
        settings = torch.tensor([eta, tol, self.gap_tol, max_iter, check_every,
                                 self.block_rows], device=p.device, dtype=torch.float64)
        p._require_replicated(settings, "distributed proximal settings")

        def fallback(reason, attempted=0, checks=0, cheap_checks=0):
            return self._fallback(log_reference, gradient, eta, tol, max_iter,
                                  check_every, reason=reason, attempted=attempted,
                                  checks=checks, cheap_checks=cheap_checks)

        shift = self._zero(p.local_rows)
        kernel, valid = self._build_kernel(log_reference, gradient, eta, shift)
        if not valid:
            del kernel
            return fallback("row-shifted kernel has zero or nonfinite entries")
        v = torch.ones_like(self.b)
        checks = cheap_checks = rebuilds = 0
        for iteration in range(1, max_iter + 1):
            u, v, column_denominator, failure = self._scaling_iteration(kernel, v)
            if failure is not None:
                del kernel
                return fallback(failure, iteration, checks, cheap_checks)
            if iteration % check_every != 0 and iteration != max_iter:
                continue
            cheap_checks += 1
            row_error = ((u * (kernel @ v)) - self.a).abs().sum()
            column_error = (v * column_denominator - self.b).abs().sum()
            residual = (p._reduce(row_error) + column_error).item()
            if not math.isfinite(residual):
                del kernel
                return fallback("nonfinite scaling residual", iteration, checks, cheap_checks)
            if residual > tol and iteration != max_iter:
                continue
            # At SHARE31890, each dense local FP64 matrix is approximately 2GB.
            # Avoid holding the kernel alongside the two materialized outputs.
            del kernel
            logu, logv = u.log(), v.log()
            Q, logQ = self._candidate(log_reference, gradient, eta, shift, logu, logv)
            valid = torch.ones((), dtype=torch.bool, device=p.device)
            underflow = self._zero()
            for block in self._blocks():
                valid.logical_and_(torch.isfinite(Q[block]).all()
                                   & torch.isfinite(logQ[block]).all())
                underflow.add_((Q[block] == 0).sum())
            if not self._all_good(valid):
                del Q, logQ
                return fallback("nonfinite original-log-kernel candidate", iteration,
                                checks, cheap_checks)
            marginal = self.marginal_residuals(Q)["both_l1"]
            info = self.certificate(Q, log_reference, gradient, eta,
                                    row_potential=logu - shift, col_potential=logv)
            checks += 1
            if not self._all_good(info["certificate_valid"]):
                del Q, logQ
                return fallback("invalid original-objective certificate", iteration,
                                checks, cheap_checks)
            converged = self._all_good(marginal <= tol
                                       and info["proximal_optimality_certified"])
            info.update(inner_iterations=iteration, certificate_checks=checks,
                        marginal_l1=marginal, converged=converged,
                        marginal_tolerance=tol, normalized_gap_tolerance=self.gap_tol,
                        sinkhorn_arithmetic="FP64 row-sharded matrix-vector scaling; original log-kernel candidate/certificate",
                        sinkhorn_path="distributed-positive-kernel-matrix-vector",
                        fast_path_used=True, fast_scaling_iterations=iteration,
                        log_domain_iterations=0, cheap_marginal_checks=cheap_checks,
                        total_inner_budget=max_iter,
                        underflow_entries=int(p._reduce(underflow).item()),
                        kernel_released_before_candidate=True,
                        kernel_rebuilds=rebuilds, collective_fallback=True)
            if converged or iteration == max_iter:
                if not converged:
                    info["failure_reason"] = "inner iteration cap exhausted without both criteria"
                return Q, logQ, info
            del Q, logQ
            kernel, valid = self._build_kernel(log_reference, gradient, eta, shift)
            rebuilds += 1
            if not valid:
                del kernel
                return fallback("unsafe rebuilt kernel", iteration, checks, cheap_checks)
        raise AssertionError("The final iteration must produce a checked candidate")
