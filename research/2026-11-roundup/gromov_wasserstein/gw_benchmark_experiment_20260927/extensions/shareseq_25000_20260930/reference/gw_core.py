"""Reference kernels for a fixed, unregularized squared-loss GW problem.

All objectives use the full quadratic at the actual (possibly infeasible) P.
No factor of 1/2, entropy term, low-rank constraint, or hidden marginal substitution.
The routines are reference implementations, not tuned large-GPU implementations.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

Tensor = torch.Tensor


@dataclass
class GWProblem:
    A: Tensor
    B: Tensor
    a: Tensor
    b: Tensor

    def __post_init__(self) -> None:
        n, m = self.a.numel(), self.b.numel()
        if self.A.shape != (n, n) or self.B.shape != (m, m):
            raise ValueError("Distance matrices and marginals have incompatible shapes.")
        if not (self.A.dtype == self.B.dtype == self.a.dtype == self.b.dtype):
            raise ValueError("All arrays must have the same dtype.")
        if not (self.A.device == self.B.device == self.a.device == self.b.device):
            raise ValueError("All arrays must be on the same device.")
        for x in (self.A, self.B, self.a, self.b):
            if not torch.isfinite(x).all():
                raise ValueError("Input contains nonfinite values.")
        if not torch.allclose(self.A, self.A.T) or not torch.allclose(self.B, self.B.T):
            raise ValueError("These kernels require symmetric distance matrices.")
        if (self.a <= 0).any() or (self.b <= 0).any():
            raise ValueError("Use strictly positive marginals.")
        one = self.a.new_tensor(1.)
        if not torch.allclose(self.a.sum(), one) or not torch.allclose(self.b.sum(), one):
            raise ValueError("Marginals must each have total mass one.")
        self.A2 = self.A.square()  # Cache explicitly; count these arrays in memory.
        self.B2 = self.B.square()

    def loss(self, P: Tensor) -> Tensor:
        """Valid also off feasibility and for signed directions in line searches."""
        r, c = P.sum(1), P.sum(0)
        return r @ (self.A2 @ r) + c @ (self.B2 @ c) - 2 * ((self.A @ P @ self.B) * P).sum()

    def gradient(self, P: Tensor) -> Tensor:
        r, c = P.sum(1), P.sum(0)
        return 2 * ((self.A2 @ r)[:, None] + (self.B2 @ c)[None, :] - 2 * self.A @ P @ self.B)

    def sketch_loss(self, P: Tensor, Z: Tensor) -> Tensor:
        """Average scalar sampled loss. Columns of Z are unscaled +/-1 probes.

        E[zz^T]=I makes this loss unbiased for the ORIGINAL full GW objective.
        This is a fixed-distribution oracle derived for this protocol, not the
        published Sampled GW estimator. Sampled losses can be negative.
        """
        if Z.ndim != 2 or Z.shape[0] != self.b.numel() or Z.shape[1] < 1:
            raise ValueError("Z must have shape (m, positive_batch_size).")
        r, c = P.sum(1), P.sum(0)
        U, V = P @ Z, P @ (self.B @ Z)
        return r @ (self.A2 @ r) + c @ (self.B2 @ c) - 2 * (U * (self.A @ V)).sum() / Z.shape[1]

    def sketch_gradient(self, P: Tensor, Z: Tensor) -> Tensor:
        """Gradient of sketch_loss; does not form AP or APB."""
        if Z.ndim != 2 or Z.shape[0] != self.b.numel() or Z.shape[1] < 1:
            raise ValueError("Z must have shape (m, positive_batch_size).")
        r, c = P.sum(1), P.sum(0)
        BZ = self.B @ Z
        AU, AV = self.A @ (P @ Z), self.A @ (P @ BZ)
        return (2 * ((self.A2 @ r)[:, None] + (self.B2 @ c)[None, :])
                - (2 / Z.shape[1]) * (AV @ Z.T + AU @ BZ.T))

    def anchor_gradient(self, P: Tensor, pairs: Tensor) -> Tensor:
        """Gradient of the uniformly sampled scalar anchor losses phi_kl.

        phi_kl(P) = nm P_kl <K^kl,P>. This transparent reference estimator
        loops over anchors. It can have severe variance and is NOT claimed
        to be an efficient default. Enumerate all pairs for an exact test.
        """
        n, m = P.shape
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("pairs must be a nonempty (batch,2) integer tensor.")
        out = torch.zeros_like(P)
        for k, l in pairs.tolist():
            K = (self.A[:, k, None] - self.B[:, l][None, :]).square()
            out += (n * m) * P[k, l] * K
            out[k, l] += (n * m) * (K * P).sum()
        return out / len(pairs)


def rademacher(m: int, batch: int, *, like: Tensor,
               generator: torch.Generator | None = None) -> Tensor:
    if m < 1 or batch < 1:
        raise ValueError("Dimensions must be positive.")
    return (2 * torch.randint(0, 2, (m, batch), device=like.device,
                              generator=generator) - 1).to(like.dtype)


def row_lmo(G: Tensor, a: Tensor) -> Tensor:
    S = torch.zeros_like(G)
    S[torch.arange(G.shape[0], device=G.device), G.argmin(1)] = a
    return S


def fw_row_update(P: Tensor, H: Tensor, a: Tensor, gamma: float) -> Tensor:
    """FW update without allocating the dense atom S."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1].")
    out = (1 - gamma) * P
    out[torch.arange(P.shape[0], device=P.device), H.argmin(1)] += gamma * a
    return out


def marginal_residuals(P: Tensor, a: Tensor, b: Tensor) -> dict[str, float]:
    er, ec = P.sum(1) - a, P.sum(0) - b
    return {
        "row_l1": er.abs().sum().item(),
        "col_l1": ec.abs().sum().item(),
        "col_l2": ec.norm().item(),
        "both_l1": (er.abs().sum() + ec.abs().sum()).item(),
        "mass_error": abs(P.sum().item() - 1),
        "min_entry": P.min().item(),
    }


def round_transport(P: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """General nonnegative-plan rounding, also usable for inexact baselines.

    Scale down excess row mass, then excess column mass, then fill deficits
    by a rank-one outer product. Feasibility is up to arithmetic precision.
    This is NOT Euclidean projection. Use as output-only postprocessing.
    """
    if not torch.isfinite(P).all() or (P < 0).any():
        raise ValueError("Rounding needs a finite, nonnegative matrix.")
    tiny = torch.finfo(P.dtype).tiny
    row_scale = torch.minimum(torch.ones_like(a), a / P.sum(1).clamp_min(tiny))
    Q = P * row_scale[:, None]
    col_scale = torch.minimum(torch.ones_like(b), b / Q.sum(0).clamp_min(tiny))
    Q = Q * col_scale[None, :]
    # Clamps address roundoff in theoretically nonnegative deficits only.
    r = (a - Q.sum(1)).clamp_min(0)
    s = (b - Q.sum(0)).clamp_min(0)
    delta = r.sum()
    if delta.item() > 0:
        Q = Q + torch.outer(r, s) / delta
    return Q


def _fixed_order_row_cumsum(values: Tensor) -> Tensor:
    """Inclusive row scan with a fixed addition tree, in the input precision.

    Hillis--Steele doubling uses only disjoint, out-of-place elementwise sums.
    Unlike CUDA cumsum in some supported PyTorch versions, it works with
    deterministic algorithms enabled. The O(log(width)) full-array passes and
    temporary storage remain inside the calling projection/solver timer.
    This helper also accepts CPU tensors for direct numerical validation.
    """
    result = values
    offset = 1
    while offset < values.shape[1]:
        result = torch.cat((result[:, :offset],
                            result[:, offset:] + result[:, :-offset]), dim=1)
        offset *= 2
    return result


def _simplex_row_cumsum(values: Tensor) -> Tensor:
    # Keep the established CPU accumulation unchanged; no host transfer/promotion.
    return _fixed_order_row_cumsum(values) if values.is_cuda else values.cumsum(1)


def simplex_rows(V: Tensor, mass: Tensor) -> Tensor:
    """Euclidean projection of each row onto a simplex of the given mass."""
    if V.ndim != 2 or mass.shape != (V.shape[0],) or (mass <= 0).any():
        raise ValueError("Positive row masses of matching size are required.")
    U = torch.sort(V, dim=1, descending=True).values
    cssv = _simplex_row_cumsum(U) - mass[:, None]
    j = torch.arange(1, V.shape[1] + 1, dtype=V.dtype, device=V.device)
    active = U - cssv / j > 0
    rho = active.sum(1).clamp_min(1) - 1
    theta = cssv.gather(1, rho[:, None]).squeeze(1) / (rho + 1).to(V.dtype)
    return (V - theta[:, None]).clamp_min(0)


def assignment_lmo(G: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Exact full-polytope LMO for n=m and uniform marginals.

    SciPy is a CPU solver. Include device transfers in timed baseline runs.
    """
    from scipy.optimize import linear_sum_assignment
    n, m = G.shape
    if n != m or not torch.allclose(a, torch.full_like(a, 1/n)) or not torch.allclose(b, torch.full_like(b, 1/n)):
        raise ValueError("Assignment LMO is only for square, uniform problems.")
    row, col = linear_sum_assignment(G.detach().cpu().double().numpy())
    S = torch.zeros_like(G)
    S[torch.as_tensor(row, device=G.device), torch.as_tensor(col, device=G.device)] = 1/n
    return S


def original_fw_gap(problem: GWProblem, P: Tensor) -> float:
    G = problem.gradient(P)
    S = assignment_lmo(G, problem.a, problem.b)
    return (G * (P - S)).sum().item()


def smoothed_gap(problem: GWProblem, P: Tensor, mu: float) -> float:
    if mu <= 0:
        raise ValueError("mu must be positive.")
    H = problem.gradient(P) + (P.sum(0) - problem.b)[None, :] / mu
    return ((H * P).sum() - (problem.a * H.min(1).values).sum()).item()


def kl_prox(logP: Tensor, G: Tensor, a: Tensor, b: Tensor, eta: float,
            tol: float = 1e-9, max_iter: int = 5000,
            check_every: int = 10) -> tuple[Tensor, Tensor, dict]:
    """Log-domain KL/Bregman proximal gradient step.

    Solves min_<U(a,b)> <G,Q> + (1/eta) KL(Q || P).
    Return logQ as solver state; do not reconstruct logQ by logging exp(logQ).
    The reported tolerance is summed absolute marginal error, not POT's tol.
    This is a reference implementation, not a reproduction of a named package.
    """
    if eta <= 0 or tol <= 0 or max_iter < 1 or check_every < 1:
        raise ValueError("Positive step size, tolerance and iteration limits required.")
    if not torch.isfinite(logP).all():
        raise ValueError("Use a strictly positive initialization and retain log-state.")
    logK = logP - eta * G
    logK = logK - logK.max(1, keepdim=True).values  # Invariant row scaling.
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    loga, logb = a.log(), b.log()
    residual = float('inf')
    for it in range(1, max_iter + 1):
        u = loga - torch.logsumexp(logK + v[None, :], dim=1)
        v = logb - torch.logsumexp(logK + u[:, None], dim=0)
        if it % check_every == 0 or it == max_iter:
            logQ = logK + u[:, None] + v[None, :]
            rr = torch.logsumexp(logQ, 1).exp() - a
            cc = torch.logsumexp(logQ, 0).exp() - b
            residual = (rr.abs().sum() + cc.abs().sum()).item()
            if residual <= tol:
                break
    logQ = logK + u[:, None] + v[None, :]
    Q = logQ.exp()
    actual = marginal_residuals(Q, a, b)["both_l1"]
    return Q, logQ, {"inner_iterations": it, "marginal_l1": actual,
                     "converged": actual <= tol,
                     "underflow_entries": int((Q == 0).sum().item())}


def project_transport_dykstra(Z: Tensor, a: Tensor, b: Tensor,
                               gap_tol: float = 1e-10, max_iter: int = 10000,
                               check_every: int = 10) -> tuple[Tensor, dict]:
    """Reference Euclidean projection with a primal-dual gap certificate.

    Dykstra over row and column simplices; reset corrections for each new Z.
    Not a reproduction of Liang--Yang iPG. Intended for small float64 checks.
    Marginal feasibility ALONE does not certify Euclidean projection accuracy.
    """
    if gap_tol <= 0 or max_iter < 1:
        raise ValueError("Positive tolerances/iteration limits required.")
    X, U, V = Z.clone(), torch.zeros_like(Z), torch.zeros_like(Z)
    gap = float('inf')
    for it in range(1, max_iter + 1):
        Y = simplex_rows(X + U, a)
        U = X + U - Y
        X = simplex_rows((Y + V).T, b).T
        V = Y + V - X
        if it % check_every == 0 or it == max_iter:
            R = round_transport(X, a, b)
            # Valid support functions of the two simplex-product sets.
            support_a = (a * U.max(1).values).sum()
            support_b = (b * V.max(0).values).sum()
            dual = ((U+V)*Z).sum() - 0.5*(U+V).square().sum() - support_a - support_b
            primal = 0.5*(R-Z).square().sum()
            gap = (primal-dual).item()
            if gap >= -1e-10 * max(1., abs(primal.item())) and gap <= gap_tol:
                break
    return R, {"inner_iterations": it, "projection_primal_dual_gap": gap,
               "converged": 0 <= gap <= gap_tol or (-1e-12 <= gap <= gap_tol)}
