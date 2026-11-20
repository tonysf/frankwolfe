"""Row-sharded entropic proximal solves with two numerical acceptance tests.

The objective is J(Q)=<G,Q>+sum(Q*(log(Q)-logP)-Q+exp(logP))/eta.
Sinkhorn vectors, retained log-state and all certificate arithmetic are FP64;
only bounded row blocks are promoted, and no dense plan is gathered. The plan
used by the GW oracle has the problem dtype. Its actual FP64-summed marginal
residual is checked. A separately rounded FP64 feasible witness supplies the
primal upper bound; the unrounded Sinkhorn potentials supply the dual lower
bound for the ORIGINAL objective, including mass constants and row shifts.

These are floating-point numerical certificates, not interval bounds or a
proof of outer convergence with finite inner error. In particular, the rounded
witness need not equal the returned plan (their discrepancy is reported).
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


class CertifiedKLProx:
    """Adapter for ``RowShardedGW`` in local, Gloo and NCCL modes."""

    def __init__(self, problem, *, gap_tol: float = 1e-7,
                 certificate_block_rows: int = 256):
        self.problem = problem
        self.gap_tol = float(gap_tol)
        self.block_rows = certificate_block_rows
        problem._require(math.isfinite(self.gap_tol) and self.gap_tol > 0,
                         "The normalized gap tolerance must be positive and finite.")
        problem._require(isinstance(self.block_rows, int)
                         and not isinstance(self.block_rows, bool) and self.block_rows > 0,
                         "certificate_block_rows must be a positive integer.")
        self.a = problem.a_local.double()
        self.b = problem.b.double()
        self.mass_a = problem._reduce(self.a.sum()).item()
        self.mass_b = self.b.sum().item()
        self.feasibility_roundoff = 512 * torch.finfo(torch.float64).eps * max(
            1., self.mass_a, self.mass_b)
        problem._require(abs(self.mass_a - self.mass_b) <= self.feasibility_roundoff,
                         "FP64 marginal masses must agree; unequal masses have no feasible coupling.")

    def _blocks(self):
        for start in range(0, self.problem.local_rows, self.block_rows):
            yield slice(start, min(start + self.block_rows, self.problem.local_rows))

    def _zero(self, *shape):
        return torch.zeros(shape, dtype=torch.float64, device=self.problem.device)

    def _masses(self, Q):
        rows = self._zero(self.problem.local_rows)
        cols = self._zero(self.problem.m)
        for block in self._blocks():
            value = Q[block].double()
            rows[block] = value.sum(1)
            cols.add_(value.sum(0))
        self.problem._reduce(cols)
        return rows, cols

    @torch.no_grad()
    def marginal_residuals(self, Q):
        """Audit the ACTUAL returned plan, using FP64 reductions."""
        self.problem._validate_plan(Q)
        rows, cols = self._masses(Q)
        row_l1 = self.problem._reduce((rows - self.a).abs().sum()).item()
        col_l1 = (cols - self.b).abs().sum().item()
        return {"row_l1": row_l1, "col_l1": col_l1,
                "both_l1": row_l1 + col_l1,
                "col_l2": (cols - self.b).norm().item(),
                "mass_error": abs(cols.sum().item() - self.mass_b),
                "min_entry": self.problem.global_min(Q).item()}

    def _kernel(self, logP, G, eta, block, shift):
        return logP[block].double() - eta * G[block].double() - shift[block, None]

    def _column_lse(self, logP, G, eta, shift, u):
        local = self._zero(self.problem.m).fill_(-math.inf)
        for block in self._blocks():
            value = self._kernel(logP, G, eta, block, shift)
            value.add_(u[block, None])
            local = torch.logaddexp(local, torch.logsumexp(value, dim=0))
        maximum = self.problem._reduce(local.clone(), dist.ReduceOp.MAX)
        scaled = torch.where(torch.isfinite(maximum), (local - maximum).exp(),
                             torch.zeros_like(local))
        total = self.problem._reduce(scaled)
        return maximum + total.log()

    def _candidate(self, logP, G, eta, shift, u, v):
        logQ = torch.empty(logP.shape, dtype=torch.float64, device=logP.device)
        Q = torch.empty(logP.shape, dtype=self.problem.dtype, device=logP.device)
        for block in self._blocks():
            value = self._kernel(logP, G, eta, block, shift)
            value.add_(u[block, None]).add_(v[None, :])
            logQ[block] = value
            Q[block] = value.exp()
        return Q, logQ

    @torch.no_grad()
    def certificate(self, Q, logP, G, eta, *, row_potential, col_potential):
        """Evaluate a numerical certificate, useful independently in tests.

        Potentials here refer to the unshifted kernel ``logP-eta*G``.
        The dual candidate is ``exp(logP-eta*G+row_potential+col_potential)``;
        it need not equal Q. This deliberately does not infer optimality from
        feasibility or from Q's logarithm.
        """
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
        # An O(eps64) imbalance can remain because the target input masses
        # themselves agree only to FP64 roundoff. Verify the witness below.
        correction_col = s / deficit if deficit > 0 else torch.zeros_like(s)
        # primal, reference mass, dual mass, L1 correction, abs terms, stable KL
        totals = self._zero(6)
        witness_rows, witness_cols = self._zero(p.local_rows), self._zero(p.m)
        for block in self._blocks():
            original = Q[block].double()
            witness = original * row_scale[block, None] * col_scale[None, :]
            witness.add_(r[block, None] * correction_col[None, :])
            lp = logP[block].double()
            g = G[block].double()
            log_dual = lp - eta * g + row_potential[block, None] + col_potential[None, :]
            reference = lp.exp()
            dual_plan = log_dual.exp()
            # xlogy implements the zero-mass convention without log(0) NaNs.
            kl = torch.xlogy(witness, witness) - witness * lp - witness + reference
            objective_terms = g * witness + kl / eta
            totals[0].add_(objective_terms.sum())
            totals[1].add_(reference.sum())
            totals[2].add_(dual_plan.sum())
            totals[3].add_((witness - original).abs().sum())
            totals[4].add_(objective_terms.abs().sum())
            # Independent gap computation avoids cancellation of two nearly
            # equal objectives. For nearby positive entries use expm1; the
            # general expression also covers exact zeros/underflow correctly.
            log_ratio = witness.log() - log_dual
            general_kl = torch.xlogy(witness, witness) - witness * log_dual - witness + dual_plan
            nearby_kl = witness * (log_ratio + torch.expm1(-log_ratio))
            stable_kl = torch.where((witness > 0) & (log_ratio.abs() < .5), nearby_kl, general_kl)
            totals[5].add_(stable_kl.sum())
            witness_rows[block] = witness.sum(1)
            witness_cols.add_(witness.sum(0))
        p._reduce(totals)
        p._reduce(witness_cols)
        row_error = p._reduce((witness_rows - self.a).abs().sum())
        witness_error = (row_error + (witness_cols - self.b).abs().sum()).item()
        row_dual = p._reduce((self.a * row_potential).sum())
        row_dual_abs = p._reduce((self.a * row_potential).abs().sum())
        constraint_correction = p._reduce((row_potential * (witness_rows - self.a)).sum())
        constraint_correction.add_(col_potential @ (witness_cols - self.b))
        stable_gap = ((totals[5] + constraint_correction) / eta).item()
        dual = ((row_dual + self.b @ col_potential - totals[2] + totals[1]) / eta).item()
        primal = totals[0].item()
        gap = primal - dual
        normalizer = max(1., abs(primal), abs(dual))
        # Explicitly report signed roundoff; a significantly negative gap is
        # an invalid certificate, never silently clipped to zero.
        roundoff = 256 * torch.finfo(torch.float64).eps * max(
            normalizer, totals[4].item(), (row_dual_abs.item()
            + (self.b * col_potential).abs().sum().item() + totals[1].item()
            + totals[2].item()) / eta)
        finite = all(math.isfinite(x) for x in (primal, dual, gap, stable_gap, witness_error, roundoff))
        valid = (finite and witness_error <= self.feasibility_roundoff
                 and min(gap, stable_gap) >= -roundoff and abs(gap - stable_gap) <= roundoff
                 and roundoff / normalizer <= self.gap_tol)
        normalized = gap / normalizer
        return {"primal_objective": primal, "dual_objective": dual,
                "primal_dual_gap": gap, "normalized_primal_dual_gap": normalized,
                "gap_normalizer": normalizer,
                "gap_roundoff_tolerance": roundoff,
                "stable_primal_dual_gap": stable_gap,
                "gap_arithmetic_crosscheck_error": abs(gap - stable_gap),
                "primal_witness_marginal_l1": witness_error,
                "primal_witness_feasibility_tolerance": self.feasibility_roundoff,
                "primal_witness_plan_l1_difference": totals[3].item(),
                "reference_mass": totals[1].item(), "dual_candidate_mass": totals[2].item(),
                "certificate_valid": valid,
                "proximal_optimality_certified": valid and (max(gap, stable_gap) + roundoff) / normalizer <= self.gap_tol,
                "arithmetic_certificate": "FP64 arithmetic; not interval-certified",
                "certificate_primal": "FP64 rounded feasible witness of returned plan",
                "plan_dtype": str(Q.dtype), "retained_log_dtype": "torch.float64"}

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
        settings = torch.tensor([eta, tol, self.gap_tol, max_iter, check_every, self.block_rows],
                                device=p.device, dtype=torch.float64)
        p._require_replicated(settings, "certified proximal settings")
        shift = self._zero(p.local_rows)
        for block in self._blocks():
            shift[block] = (logP[block].double() - eta * G[block].double()).max(1).values
        p.require_finite(shift, "kernel row shifts")
        u, v = self._zero(p.local_rows), self._zero(p.m)
        loga, logb = self.a.log(), self.b.log()
        checks = 0
        for iteration in range(1, max_iter + 1):
            for block in self._blocks():
                value = self._kernel(logP, G, eta, block, shift)
                value.add_(v[None, :])
                u[block] = loga[block] - torch.logsumexp(value, dim=1)
            v = logb - self._column_lse(logP, G, eta, shift, u)
            if iteration % check_every != 0 and iteration != max_iter:
                continue
            Q, logQ = self._candidate(logP, G, eta, shift, u, v)
            marginal = self.marginal_residuals(Q)["both_l1"]
            # A certificate at every stopping check also records informative
            # gap evidence when a cap is exhausted before feasibility passes.
            info = self.certificate(Q, logP, G, eta,
                                    row_potential=u - shift, col_potential=v)
            checks += 1
            converged = marginal <= tol and info["proximal_optimality_certified"]
            info.update(inner_iterations=iteration, certificate_checks=checks,
                        marginal_l1=marginal, converged=bool(converged),
                        marginal_tolerance=tol, normalized_gap_tolerance=self.gap_tol,
                        sinkhorn_arithmetic="FP64 bounded blocks and vector reductions")
            if not info["certificate_valid"]:
                info["failure_reason"] = "invalid primal-dual certificate"
                break
            if converged:
                break
            # Avoid retaining the previous dense candidates through the next
            # allocation; only final/accepted candidates leave the loop.
            if iteration < max_iter:
                del Q, logQ
        underflow = self._zero()
        for block in self._blocks():
            underflow.add_((Q[block] == 0).sum())
        info["underflow_entries"] = int(p._reduce(underflow).item())
        if not info["converged"]:
            info.setdefault("failure_reason", "inner iteration cap exhausted without both criteria")
        return Q, logQ, info
