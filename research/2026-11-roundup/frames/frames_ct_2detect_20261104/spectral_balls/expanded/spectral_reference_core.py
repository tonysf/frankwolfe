"""FP64 factorized least squares with spectral and convex side constraints.

The primary finite sum is over IMAGE COLUMNS, not individual pixels:
    f(U,V) = (1/n) sum_j ||U v_j - M[:,j]||^2 / (2*m).
U has shape (pixels, rank), V has shape (images, rank).
All gradients here are gradients in the product Frobenius inner product.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd, svdvals


def pair_norm_sq(U, V):
    return float(np.sum(np.square(U)) + np.sum(np.square(V)))


def _matrix(A):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("Expected a matrix")
    return A


def spectral_lmo(G, radius, return_certificate=False):
    """Minimum-Frobenius spectral-ball LMO with a numerical support certificate.

    Nonzero singular directions receive singular value -radius; null directions
    receive zero. This is not the rank-one LMO of a nuclear-norm ball. Floating
    point singular values <= eps*max(shape)*sigma_max are treated as zero. The
    consequent departure from the exact LMO support value is bounded by radius
    times the sum of those discarded singular values, explicitly reported.
    """
    G = _matrix(G)
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and nonnegative")
    if radius == 0 or not np.any(G):
        result = np.zeros_like(G)
        certificate = {"rank": 0, "rank_tolerance": 0.0, "support_gap_bound": 0.0,
                       "discarded_singular_sum": 0.0}
    else:
        P, singular, Qt = svd(G, full_matrices=False, check_finite=True)
        tolerance = np.finfo(np.float64).eps * max(G.shape) * float(singular[0])
        retained = singular > tolerance
        result = -float(radius) * (P[:, retained] @ Qt[retained])
        discarded = float(np.sum(singular[~retained]))
        certificate = {"rank": int(np.count_nonzero(retained)),
                       "rank_tolerance": tolerance,
                       "support_gap_bound": float(radius) * discarded,
                       "discarded_singular_sum": discarded}
    return (result, certificate) if return_certificate else result


def project_spectral_ball(A, radius):
    """Euclidean/Frobenius projection by clipping singular values."""
    A = _matrix(A)
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and nonnegative")
    if radius == 0:
        return np.zeros_like(A)
    P, singular, Qt = svd(A, full_matrices=False, check_finite=True)
    if singular[0] <= radius:
        return A.copy()
    return (P * np.minimum(singular, radius)) @ Qt


def project_l1_ball(A, radius):
    """Euclidean projection onto an ENTRYWISE l1 ball, preserving signs."""
    A = np.asarray(A, dtype=np.float64)
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and nonnegative")
    if radius == 0:
        return np.zeros_like(A)
    values = np.abs(A).ravel()
    if values.sum() <= radius:
        return A.copy()
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    index = np.arange(1, ordered.size + 1)
    active = np.flatnonzero(ordered > (cumulative - radius) / index)
    rho = active[-1]
    threshold = (cumulative[rho] - radius) / (rho + 1)
    return np.sign(A) * np.maximum(np.abs(A) - threshold, 0.0)


@dataclass
class FactorizationProblem:
    M: np.ndarray
    rank: int
    RU: float
    RV: float
    l1_radius: float
    nonnegative_V: bool = False

    def __post_init__(self):
        self.M = np.array(self.M, dtype=np.float64, order="C", copy=True)
        if self.M.ndim != 2 or min(self.M.shape) == 0:
            raise ValueError("M must be a nonempty matrix")
        if not np.all(np.isfinite(self.M)):
            raise ValueError("M must be finite")
        self.m, self.n = self.M.shape
        self.N = self.m * self.n
        self.n_components = self.n
        if not isinstance(self.rank, (int, np.integer)) or not 1 <= self.rank <= min(self.m, self.n):
            raise ValueError("rank must be between 1 and min(M.shape)")
        for name in ("RU", "RV", "l1_radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)

    def _factors(self, U, V):
        U, V = _matrix(U), _matrix(V)
        if U.shape != (self.m, self.rank) or V.shape != (self.n, self.rank):
            raise ValueError("Factor shapes must be (pixels,rank) and (images,rank)")
        return U, V

    @staticmethod
    def _indices(indices, size):
        values = np.asarray(indices)
        if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Sample indices must be a nonempty integer vector")
        if np.any(values < 0) or np.any(values >= size):
            raise ValueError("Sample index out of bounds")
        return values.astype(np.intp, copy=False)

    def objective(self, U, V):
        U, V = self._factors(U, V)
        residual = U @ V.T - self.M
        return float(np.sum(residual * residual) / (2 * self.N))

    def full_grad(self, U, V):
        U, V = self._factors(U, V)
        residual = U @ V.T - self.M
        return (residual @ V) / self.N, (residual.T @ U) / self.N

    def sampled_grad_columns(self, U, V, cols):
        """Average image-component gradients, including repeated image indices."""
        U, V = self._factors(U, V)
        cols = self._indices(cols, self.n)
        Vbatch = V[cols]
        residual = U @ Vbatch.T - self.M[:, cols]
        scale = 1.0 / (self.m * cols.size)
        gU = (residual @ Vbatch) * scale
        gV = np.zeros_like(V)
        np.add.at(gV, cols, (residual.T @ U) * scale)
        return gU, gV

    def sampled_grad(self, U, V, rng, batch_size):
        """IID uniform IMAGE sampling with replacement; unbiased for full_grad."""
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        cols = rng.integers(self.n, size=batch_size)
        return self.sampled_grad_columns(U, V, cols)

    def sampled_grad_indices(self, U, V, rows, cols):
        """Diagnostic ENTRY-component oracle; not the primary campaign oracle."""
        U, V = self._factors(U, V)
        rows = self._indices(rows, self.m)
        cols = self._indices(cols, self.n)
        if rows.shape != cols.shape:
            raise ValueError("Entry row and column arrays must have equal length")
        Ubatch, Vbatch = U[rows], V[cols]
        residual = np.einsum("ij,ij->i", Ubatch, Vbatch) - self.M[rows, cols]
        residual = residual / rows.size
        gU, gV = np.zeros_like(U), np.zeros_like(V)
        np.add.at(gU, rows, residual[:, None] * Vbatch)
        np.add.at(gV, cols, residual[:, None] * Ubatch)
        return gU, gV

    def sampled_grad_entries(self, U, V, rng, batch_size):
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        rows = rng.integers(self.m, size=batch_size)
        cols = rng.integers(self.n, size=batch_size)
        return self.sampled_grad_indices(U, V, rows, cols)

    def lmo_C(self, gU, gV, return_certificate=False):
        gU, gV = self._factors(gU, gV)
        if return_certificate:
            sU, certU = spectral_lmo(gU, self.RU, return_certificate=True)
            sV, certV = spectral_lmo(gV, self.RV, return_certificate=True)
            return sU, sV, {"U": certU, "V": certV,
                            "support_gap_bound": certU["support_gap_bound"] + certV["support_gap_bound"]}
        return spectral_lmo(gU, self.RU), spectral_lmo(gV, self.RV)

    def project_C(self, U, V):
        U, V = self._factors(U, V)
        return project_spectral_ball(U, self.RU), project_spectral_ball(V, self.RV)

    def project_D(self, U, V):
        U, V = self._factors(U, V)
        Vbase = np.maximum(V, 0.0) if self.nonnegative_V else V
        return np.maximum(U, 0.0), project_l1_ball(Vbase, self.l1_radius)

    def smoothed_gradient(self, U, V, mu, gradient=None):
        """Gradient of f + dist_D^2/(2mu), with exact penalty gradient.

        gradient=None requests the true full gradient of f. A supplied pair
        may be a momentum estimate; the penalty is never put in momentum.
        """
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("mu must be finite and positive")
        U, V = self._factors(U, V)
        gU, gV = self.full_grad(U, V) if gradient is None else self._factors(*gradient)
        pU, pV = self.project_D(U, V)
        return gU + (U - pU) / mu, gV + (V - pV) / mu

    def smoothed_objective(self, U, V, mu):
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("mu must be finite and positive")
        U, V = self._factors(U, V)
        pU, pV = self.project_D(U, V)
        return self.objective(U, V) + pair_norm_sq(U - pU, V - pV) / (2 * mu)

    def smoothed_fw_gap(self, U, V, mu):
        """Deterministic true full-gradient FW gap for the smoothed objective."""
        U, V = self._factors(U, V)
        gU, gV = self.smoothed_gradient(U, V, mu)
        # The nuclear-norm dual value uses ALL singular values, so diagnostic
        # stationarity does not depend on the optimizer's nullspace tolerance.
        return float(np.sum(gU * U) + np.sum(gV * V)
                     + self.RU * svdvals(gU).sum() + self.RV * svdvals(gV).sum())

    def feasibility(self, U, V, tolerance=1e-10):
        """Constraint violations and exact product-Frobenius distances to C,D."""
        U, V = self._factors(U, V)
        su, sv = svdvals(U), svdvals(V)
        pU, pV = self.project_D(U, V)
        U_negative = float(np.linalg.norm(np.minimum(U, 0.0)))
        V_negative = float(np.linalg.norm(np.minimum(V, 0.0))) if self.nonnegative_V else 0.0
        u_violation = max(0.0, float(su[0]) - self.RU)
        v_violation = max(0.0, float(sv[0]) - self.RV)
        v_l1 = float(np.abs(V).sum())
        l1_violation = max(0.0, v_l1 - self.l1_radius)
        maximum = max(u_violation, v_violation, U_negative, V_negative, l1_violation)
        return {
            "U_spectral_norm": float(su[0]), "V_spectral_norm": float(sv[0]),
            "U_spectral_violation": u_violation, "V_spectral_violation": v_violation,
            "U_negative_norm": U_negative, "V_negative_norm": V_negative,
            "V_l1_norm": v_l1, "V_l1_violation": l1_violation,
            "C_distance": float(np.sqrt(np.square(np.maximum(su - self.RU, 0)).sum()
                                        + np.square(np.maximum(sv - self.RV, 0)).sum())),
            "D_distance": float(np.sqrt(pair_norm_sq(U - pU, V - pV))),
            "max_constraint_violation": maximum, "feasible": maximum <= tolerance,
        }

    def repair(self, U, V):
        """Return a feasible DIAGNOSTIC point; never use as an optimizer update.

        Positive-part clipping can increase the operator norm, so it precedes
        radial spectral rescaling. Radial scaling preserves signs and l1 bounds.
        This map is not claimed to be a Euclidean projection onto C intersect D.
        """
        U, V = self._factors(U, V)
        fixedU = np.maximum(U, 0.0)
        fixedV = np.maximum(V, 0.0) if self.nonnegative_V else V.copy()
        normU, normV = float(svdvals(fixedU)[0]), float(svdvals(fixedV)[0])
        if normU > self.RU:
            fixedU *= self.RU / normU
        v_l1 = float(np.abs(fixedV).sum())
        scale = min(1.0, self.RV / normV if normV else 1.0,
                    self.l1_radius / v_l1 if v_l1 else 1.0)
        fixedV *= scale
        return fixedU, fixedV

    def certificates(self):
        """Conservative uniform constants on C; loss normalization is explicit.

        Literal global smoothness/variance in the SFRAMES manuscript requires
        a smooth compactly supported extension equal to these component losses
        on a neighborhood of C. No extension is numerically evaluated, because
        every gradient query is in C. The raw quartic is not globally smooth.
        """
        max_entry = float(np.max(np.abs(self.M)))
        max_col_norm = float(np.max(np.linalg.norm(self.M, axis=0)))
        radii_sq = self.RU ** 2 + self.RV ** 2
        image_residual = self.RU * self.RV + max_col_norm
        entry_residual = self.RU * self.RV + max_entry
        return {
            "objective": "sum_entries((U@V.T-M)^2)/(2*m*n)",
            "primary_oracle": "uniform_image_columns",
            "sampled_grad_default_sampling": "iid_with_replacement",
            "batch_sampling_note": "Explicit columns may instead be a fresh uniform subset without replacement; the same variance upper bound holds",
            "n_components": self.n, "m_pixels": self.m, "n_images": self.n,
            "rank": int(self.rank), "RU": self.RU, "RV": self.RV,
            "l1_radius": self.l1_radius, "nonnegative_V": bool(self.nonnegative_V),
            "C_diameter_bound": float(2 * np.sqrt(self.rank * radii_sq)),
            "C_product_norm_bound": float(np.sqrt(self.rank * radii_sq)),
            "max_entry_abs": max_entry, "max_image_l2": max_col_norm,
            "image_gradient_squared_bound": float(image_residual ** 2 * radii_sq / self.m ** 2),
            "image_L_bound": float((radii_sq + image_residual) / self.m),
            "image_U_partial_L_bound": self.RV ** 2 / self.m,
            "image_V_partial_L_bound": self.RU ** 2 / self.m,
            "entry_gradient_squared_bound": float(entry_residual ** 2 * radii_sq),
            "entry_L_bound": float(radii_sq + entry_residual),
            "batch_variance_bound": "image_gradient_squared_bound/batch_size",
            "ri_witness": ("U=epsilon*ones(m,rank), V=eta*ones(n,rank); "
                           "0<epsilon<RU/sqrt(m*rank), "
                           "0<eta<min(RV/sqrt(n*rank),l1_radius/(n*rank))"
                           if self.nonnegative_V else
                           "U=epsilon*ones(m,rank), V=0; 0<epsilon<RU/sqrt(m*rank)"),
            "raw_quartic_globally_smooth": False,
            "global_assumption_resolution": "C2 compact cutoff of each component, equal to one on a neighborhood of C; never evaluated outside C",
            "spectral_lmo": "minimum_Frobenius_polar; numerical_null_directions_zeroed; not_rank_one",
            "spectral_lmo_null_tolerance": "float64_eps*max(matrix.shape)*largest_singular_value",
            "spectral_lmo_support_error_bound": "radius*sum(discarded_singular_values); available per call",
            "repair_is_projection": False,
        }
