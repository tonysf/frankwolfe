"""Row-sharded kernels for the unchanged, unregularized squared-loss GW problem.

Every rank holds contiguous rows of A, A**2 and the coupling P. B, B**2 and
the two marginal vectors are replicated. The stochastic oracle communicates
only row marginals and thin n-by-batch products. Exact diagnostics broadcast
bounded blocks of P; they never gather the full coupling onto a device.

All ranks must call methods in the same order. NCCL/CUDA is the production
backend; Gloo/CPU and an uninitialized, single-process mode support tests.
These are collective kernels, not a distributed stochastic data sampler:
every rank uses the same probes and computes its rows of the same estimator.
Floating-point reduction order differs from the single-device reference.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist

Tensor = torch.Tensor


def row_partition(n: int, world_size: int, rank: int) -> tuple[int, int]:
    """Contiguous quotient/remainder partition, allowing empty final shards."""
    if n < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Positive n/world_size and a valid rank are required.")
    width, remainder = divmod(n, world_size)
    start = rank * width + min(rank, remainder)
    return start, start + width + int(rank < remainder)


class RowShardedGW:
    """A local A row block plus replicated B/a/b defines one global problem.

    Local plans have shape ``(row_stop-row_start, m)``. ``loss`` and
    ``sketch_loss`` return the same global scalar on every rank. Gradient,
    rounding, FW and KL results retain the local row partition. Constructors
    validate finite inputs, unit positive marginals and distance symmetry;
    invalid input on one rank raises on all ranks after a collective check.
    """

    @torch.no_grad()
    def __init__(self, A_local: Tensor, B: Tensor, a: Tensor, b: Tensor, *,
                 group=None, exact_block_rows: int = 1024):
        if group is not None and not dist.is_initialized():
            raise ValueError("A process group requires initialized torch.distributed.")
        self.group = group
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank(group) if self.distributed else 0
        self.world_size = dist.get_world_size(group) if self.distributed else 1
        self.backend = str(dist.get_backend(group)) if self.distributed else "none"
        self._check_device = (torch.device("cuda", torch.cuda.current_device())
                              if self.backend == "nccl" else torch.device("cpu"))
        self.A, self.B, self.a, self.b = A_local, B, a, b
        basic = (all(isinstance(x, Tensor) for x in (A_local, B, a, b))
                 and a.ndim == b.ndim == 1 and a.numel() > 0 and b.numel() > 0
                 and A_local.ndim == B.ndim == 2
                 and A_local.dtype in (torch.float32, torch.float64)
                 and all(x.dtype == A_local.dtype for x in (B, a, b))
                 and all(x.device == A_local.device for x in (B, a, b)))
        self._require(basic, "Inputs must be compatible float32/float64 tensors with vector marginals.")
        self.device, self.dtype = A_local.device, A_local.dtype
        self._require(self.backend != "nccl" or self.device.type == "cuda",
                      "NCCL requires CUDA tensors on the rank's current device.")
        self._require(self.backend != "nccl" or self.device == self._check_device,
                      "Call torch.cuda.set_device(local_rank) before constructing the problem.")
        self._require(self.backend != "gloo" or self.device.type == "cpu",
                      "The Gloo test backend requires CPU tensors.")
        self.n, self.m = a.numel(), b.numel()
        self._require(isinstance(exact_block_rows, int) and not isinstance(exact_block_rows, bool)
                      and exact_block_rows > 0, "exact_block_rows must be a positive integer.")
        self.exact_block_rows = exact_block_rows
        metadata = torch.tensor([self.n, self.m, 32 if self.dtype == torch.float32 else 64,
                                 self.exact_block_rows], dtype=torch.int64, device=self._check_device)
        if self.world_size > 1:
            all_metadata = [torch.empty_like(metadata) for _ in range(self.world_size)]
            dist.all_gather(all_metadata, metadata, group=self.group)
            self._require(all(torch.equal(metadata, entry) for entry in all_metadata),
                          "Every rank must agree on global dimensions, dtype and exact block size.")
        self.row_start, self.row_stop = row_partition(self.n, self.world_size, self.rank)
        self.local_rows = self.row_stop - self.row_start
        self.partitions = [row_partition(self.n, self.world_size, r)
                           for r in range(self.world_size)]
        self.row_counts = [stop - start for start, stop in self.partitions]
        self.a_local = self.a[self.row_start:self.row_stop]
        self._require(A_local.shape == (self.local_rows, self.n) and B.shape == (self.m, self.m),
                      "Distance matrix shapes must match the fixed row partition and marginals.")
        for name, value in (("A", A_local), ("B", B), ("a", a), ("b", b)):
            self.require_finite(value, name)
        one = a.new_tensor(1.)
        self._require(bool((a > 0).all() and (b > 0).all()
                           and torch.allclose(a.sum(), one) and torch.allclose(b.sum(), one)),
                      "Marginals must be strictly positive and each have total mass one.")
        # Replication is a semantic invariant, not merely a shape requirement.
        self._require_replicated(a, "a")
        self._require_replicated(b, "b")
        symmetric = True
        for start in range(0, self.m, self.exact_block_rows):
            stop = min(start + self.exact_block_rows, self.m)
            symmetric = bool(torch.allclose(B[start:stop], B[:, start:stop].T)) and symmetric
            self._require_replicated(B[start:stop], "B block")
        for start, block in self._broadcast_rows(A_local):
            stop = start + block.shape[0]
            symmetric = bool(torch.allclose(A_local[:, start:stop],
                                             block[:, self.row_start:self.row_stop].T)) and symmetric
        self._require(symmetric, "GW kernels require symmetric distance matrices.")
        self.A2, self.B2 = self.A.square(), self.B.square()
        self.require_finite(self.A2, "A squared")
        self.require_finite(self.B2, "B squared")

    def _require(self, condition, message: str) -> None:
        flag = torch.as_tensor(condition, dtype=torch.int32, device=self._check_device).reshape(())
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.group)
        if not bool(flag.item()):
            raise ValueError(message)

    def _global_rank(self, group_rank: int) -> int:
        return (dist.get_global_rank(self.group, group_rank)
                if self.group is not None else group_rank)

    def _reduce(self, tensor: Tensor, op=dist.ReduceOp.SUM) -> Tensor:
        if self.world_size > 1:
            dist.all_reduce(tensor, op=op, group=self.group)
        return tensor

    def _require_replicated(self, tensor: Tensor, name: str) -> None:
        if self.world_size == 1:
            return
        reference = tensor.contiguous().clone()
        dist.broadcast(reference, src=self._global_rank(0), group=self.group)
        self._require(torch.equal(tensor, reference), f"{name} must be identical on all ranks.")

    def _validate_plan(self, P: Tensor, name: str = "P") -> None:
        self._require(isinstance(P, Tensor) and P.shape == (self.local_rows, self.m)
                      and P.dtype == self.dtype and P.device == self.device,
                      f"{name} must have the local plan shape, dtype and device.")
        self.require_finite(P, name)

    @torch.no_grad()
    def require_finite(self, value: Tensor, name: str = "tensor") -> None:
        # Row blocks avoid a full-size boolean temporary during validation.
        finite = torch.ones((), dtype=torch.bool, device=value.device)
        if value.ndim >= 2:
            for start in range(0, value.shape[0], getattr(self, "exact_block_rows", 1024)):
                finite.logical_and_(torch.isfinite(value[start:start + getattr(self, "exact_block_rows", 1024)]).all())
        else:
            finite = torch.isfinite(value).all()
        self._require(finite, f"{name} contains nonfinite values on at least one rank.")

    @torch.no_grad()
    def require_nonnegative(self, value: Tensor, name: str = "tensor") -> None:
        self.require_finite(value, name)
        self._require((value >= 0).all(), f"{name} must be nonnegative on every rank.")

    @torch.no_grad()
    def global_sum(self, value: Tensor) -> Tensor:
        """Sum entries of a row-sharded tensor over all ranks (scalar)."""
        return self._reduce(value.sum())

    @torch.no_grad()
    def global_norm(self, value: Tensor) -> Tensor:
        return self.global_sum(value.square()).sqrt()

    @torch.no_grad()
    def global_min(self, value: Tensor) -> Tensor:
        local = value.min() if value.numel() else value.new_tensor(float("inf"))
        return self._reduce(local, dist.ReduceOp.MIN)

    @torch.no_grad()
    def global_max(self, value: Tensor) -> Tensor:
        local = value.max() if value.numel() else value.new_tensor(float("-inf"))
        return self._reduce(local, dist.ReduceOp.MAX)

    def _gather_rows(self, local: Tensor) -> Tensor:
        """Gather vectors/thin sketch products with padding for uneven shards."""
        if self.world_size == 1:
            return local
        maximum = max(self.row_counts)
        padded = local.new_zeros((maximum,) + local.shape[1:])
        padded[:self.local_rows].copy_(local)
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded, group=self.group)
        return torch.cat([part[:count] for part, count in zip(gathered, self.row_counts)], dim=0)

    def _broadcast_rows(self, local: Tensor):
        """Yield (global_start, row_block), never replicating the full array."""
        for owner, (start, stop) in enumerate(self.partitions):
            for global_start in range(start, stop, self.exact_block_rows):
                width = min(self.exact_block_rows, stop - global_start)
                if self.rank == owner:
                    block = local[global_start - start:global_start - start + width].contiguous()
                else:
                    block = local.new_empty((width,) + local.shape[1:])
                if self.world_size > 1:
                    dist.broadcast(block, src=self._global_rank(owner), group=self.group)
                yield global_start, block

    @torch.no_grad()
    def marginals(self, P: Tensor) -> tuple[Tensor, Tensor]:
        """Return replicated global row and column mass vectors."""
        self._validate_plan(P)
        return self._gather_rows(P.sum(1)), self._reduce(P.sum(0))

    def _left_product(self, P: Tensor) -> Tensor:
        AP = torch.zeros_like(P)
        for start, block in self._broadcast_rows(P):
            AP.addmm_(self.A[:, start:start + block.shape[0]], block)
        return AP

    def _marginal_terms(self, P: Tensor):
        r, c = self.marginals(P)
        return r, c, self.A2 @ r, self.B2 @ c

    @torch.no_grad()
    def loss(self, P: Tensor) -> Tensor:
        """Full GW loss, including off-feasibility and signed-plan terms."""
        r, c, ar, bc = self._marginal_terms(P)
        AP = self._left_product(P)
        cross = ((AP @ self.B) * P).sum()
        return self._reduce((r[self.row_start:self.row_stop] * ar).sum() - 2 * cross) + c @ bc

    @torch.no_grad()
    def gradient(self, P: Tensor) -> Tensor:
        """Exact local gradient rows; AP is computed by bounded broadcasts."""
        _, _, ar, bc = self._marginal_terms(P)
        AP = self._left_product(P)
        result = AP @ self.B
        result.mul_(-4).add_(2 * ar[:, None]).add_(2 * bc[None, :])
        return result

    def _validate_probes(self, Z: Tensor) -> None:
        self._require(isinstance(Z, Tensor) and Z.ndim == 2 and Z.shape[0] == self.m
                      and Z.shape[1] > 0 and Z.dtype == self.dtype and Z.device == self.device,
                      "Z must have shape (m, positive batch) with the problem dtype/device.")
        # Check collective sizes before using them in an all-gather.
        batch = torch.tensor(Z.shape[1], dtype=torch.int64, device=self._check_device)
        smallest, largest = batch.clone(), batch.clone()
        self._reduce(smallest, dist.ReduceOp.MIN)
        self._reduce(largest, dist.ReduceOp.MAX)
        self._require(smallest == largest, "All ranks must use the same probe batch size.")
        self.require_finite(Z, "Z")
        self._require_replicated(Z, "Z")

    @torch.no_grad()
    def rademacher(self, batch: int, *, generator: Optional[torch.Generator] = None) -> Tensor:
        """Rank zero draws the reference unscaled +/-1 probes and broadcasts."""
        self._require(isinstance(batch, int) and not isinstance(batch, bool) and batch > 0,
                      "The probe batch must be a positive integer.")
        # Batch is normally frozen in the runner; validate it before broadcast.
        size = torch.tensor(batch, dtype=torch.int64, device=self._check_device)
        minimum, maximum = size.clone(), size.clone()
        self._reduce(minimum, dist.ReduceOp.MIN)
        self._reduce(maximum, dist.ReduceOp.MAX)
        self._require(minimum == maximum, "All ranks must request the same probe batch size.")
        if self.rank == 0:
            Z = (2 * torch.randint(0, 2, (self.m, batch), device=self.device,
                                   generator=generator) - 1).to(self.dtype)
        else:
            Z = torch.empty((self.m, batch), device=self.device, dtype=self.dtype)
        if self.world_size > 1:
            dist.broadcast(Z, src=self._global_rank(0), group=self.group)
        return Z

    @torch.no_grad()
    def sketch_loss(self, P: Tensor, Z: Tensor) -> Tensor:
        self._validate_probes(Z)
        r, c, ar, bc = self._marginal_terms(P)
        U = P @ Z
        V = self._gather_rows(P @ (self.B @ Z))
        cross = (U * (self.A @ V)).sum() / Z.shape[1]
        return self._reduce((r[self.row_start:self.row_stop] * ar).sum() - 2 * cross) + c @ bc

    @torch.no_grad()
    def sketch_gradient(self, P: Tensor, Z: Tensor) -> Tensor:
        """The gradient of the same unbiased scalar Rademacher sketch loss."""
        self._validate_probes(Z)
        _, _, ar, bc = self._marginal_terms(P)
        BZ = self.B @ Z
        U = self._gather_rows(P @ Z)
        V = self._gather_rows(P @ BZ)
        AU, AV = self.A @ U, self.A @ V
        # Accumulating into one local output avoids two simultaneous products.
        result = AV @ Z.T
        result.addmm_(AU, BZ.T)
        result.mul_(-2 / Z.shape[1]).add_(2 * ar[:, None]).add_(2 * bc[None, :])
        return result

    @torch.no_grad()
    def fw_step(self, P: Tensor, d: Tensor, gamma: float, mu: float) -> Tensor:
        """Local row FW update with the exact global column penalty."""
        self._validate_plan(P)
        self._validate_plan(d, "momentum")
        self._require(math.isfinite(gamma) and 0 <= gamma <= 1
                      and math.isfinite(mu) and mu > 0, "Use gamma in [0,1] and positive finite mu.")
        column_error = self._reduce(P.sum(0)) - self.b
        H = d + column_error[None, :] / mu
        out = (1 - gamma) * P
        # torch.argmin preserves the reference's first-column tie convention.
        out[torch.arange(self.local_rows, device=self.device), H.argmin(1)] += gamma * self.a_local
        return out

    @torch.no_grad()
    def marginal_residuals(self, P: Tensor) -> dict[str, float]:
        self._validate_plan(P)
        row_l1 = self.global_sum((P.sum(1) - self.a_local).abs())
        columns = self._reduce(P.sum(0))
        column_error = columns - self.b
        col_l1 = column_error.abs().sum()
        return {"row_l1": row_l1.item(), "col_l1": col_l1.item(),
                "col_l2": column_error.norm().item(), "both_l1": (row_l1 + col_l1).item(),
                "mass_error": abs(columns.sum().item() - 1),
                "min_entry": self.global_min(P).item()}

    @torch.no_grad()
    def round_transport(self, P: Tensor) -> Tensor:
        """Output-only row/column scaling and rank-one deficit correction."""
        self._validate_plan(P)
        self.require_nonnegative(P, "Rounding input")
        tiny = torch.finfo(P.dtype).tiny
        row_scale = torch.minimum(torch.ones_like(self.a_local),
                                  self.a_local / P.sum(1).clamp_min(tiny))
        Q = P * row_scale[:, None]
        columns = self._reduce(Q.sum(0))
        col_scale = torch.minimum(torch.ones_like(self.b), self.b / columns.clamp_min(tiny))
        Q.mul_(col_scale[None, :])
        r = (self.a_local - Q.sum(1)).clamp_min(0)
        s = (self.b - self._reduce(Q.sum(0))).clamp_min(0)
        delta = self.global_sum(r)
        if delta.item() > 0:
            # addmm avoids allocating a second dense local rank-one correction.
            Q.addmm_(r[:, None], (s / delta)[None, :])
        return Q

    def _column_logsumexp(self, values: Tensor) -> Tensor:
        local = torch.logsumexp(values, dim=0)
        maximum = self._reduce(local.clone(), dist.ReduceOp.MAX)
        finite = torch.isfinite(maximum)
        scaled = torch.where(finite, (local - maximum).exp(), torch.zeros_like(local))
        total = self._reduce(scaled)
        return torch.where(finite, maximum + total.log(), maximum)

    @torch.no_grad()
    def kl_prox(self, logP: Tensor, G: Tensor, eta: float, tol: float = 1e-9,
                max_iter: int = 5000, check_every: int = 10) -> tuple[Tensor, Tensor, dict]:
        """Distributed log-domain KL proximal step with reference stopping rules.

        Every column normalization is an exact global logsumexp (up to floating
        point arithmetic), using max and sum all-reductions. The stopping error
        is global row L1 plus column L1 once, never world_size times column L1.
        Retain returned logQ as state, including entries whose exp underflows.
        """
        self._validate_plan(logP, "logP")
        self._validate_plan(G, "gradient")
        self._require(math.isfinite(eta) and eta > 0 and math.isfinite(tol) and tol > 0
                      and isinstance(max_iter, int) and max_iter >= 1
                      and isinstance(check_every, int) and check_every >= 1,
                      "Positive finite step/tolerance and integer iteration limits are required.")
        logK = logP - eta * G
        logK.sub_(logK.max(1, keepdim=True).values)
        self.require_finite(logK, "KL log kernel")
        u, v = torch.zeros_like(self.a_local), torch.zeros_like(self.b)
        loga, logb = self.a_local.log(), self.b.log()
        for it in range(1, max_iter + 1):
            u = loga - torch.logsumexp(logK + v[None, :], dim=1)
            v = logb - self._column_logsumexp(logK + u[:, None])
            if it % check_every == 0 or it == max_iter:
                logQ = logK + u[:, None] + v[None, :]
                rr = torch.logsumexp(logQ, dim=1).exp() - self.a_local
                cc = self._column_logsumexp(logQ).exp() - self.b
                residual = self.global_sum(rr.abs()) + cc.abs().sum()
                if residual.item() <= tol:
                    break
        logQ = logK + u[:, None] + v[None, :]
        Q = logQ.exp()
        actual = self.marginal_residuals(Q)["both_l1"]
        underflow = self.global_sum((Q == 0).to(torch.int64)).item()
        return Q, logQ, {"inner_iterations": it, "marginal_l1": actual,
                         "converged": actual <= tol, "underflow_entries": int(underflow)}
