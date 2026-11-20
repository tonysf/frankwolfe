"""Data-only problem geometry and reproducible shared feasible initialization.

No optimization iterations or outcome-dependent parameter choices occur here.
Run callers with their intended BLAS thread limits set before importing NumPy.
"""
import numpy as np
from scipy.linalg import svdvals


def make_problem_config(A, rank):
    """Return JSON-safe geometry from a matrix of image columns.

    The intended input has each nonzero image scaled to Euclidean norm one;
    this helper records whether that preprocessing has actually happened.
    The rank-r SVD floor is an unconstrained lower bound, not a promise that
    the spectral, positivity and l1 constraints admit its minimizer.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or min(A.shape) == 0 or not np.all(np.isfinite(A)):
        raise ValueError("A must be a finite nonempty matrix")
    m, n = A.shape
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(m, n):
        raise ValueError("rank must be between 1 and min(A.shape)")
    singular = svdvals(A, check_finite=False)
    sigma1 = float(singular[0])
    if sigma1 <= 0:
        raise ValueError("A must contain a nonzero observation")
    radius = float(1.25 * np.sqrt(sigma1))
    l1_radius = float(.35 * np.sqrt(n * rank) * radius)
    # Summing the actual tail avoids cancellation of nearly equal energies.
    tail_energy = float(np.sum(np.square(singular[rank:])))
    data_energy = float(np.sum(np.square(singular)))
    column_norms = np.linalg.norm(A, axis=0)
    zero_loss = data_energy / (2 * m * n)
    rank_floor = tail_energy / (2 * m * n)
    return {
        "rank": int(rank), "RU": radius, "RV": radius,
        "l1_radius": l1_radius, "nonnegative_V": False,
        "m": int(m), "n": int(n), "n_components": int(n),
        "sigma1": sigma1,
        "rank_r_loss_lower_bound": rank_floor,
        "zero_prediction_loss": zero_loss,
        "mean_data_square": data_energy / (m * n),
        "rank_r_relative_squared_error_lower_bound": tail_energy / data_energy,
        "rank_r_loss_floor_note": "Unconstrained rank-r approximation; constraints may increase the attainable loss",
        "L_full_bound": float((3 * radius ** 2 + sigma1) / (m * n)),
        "full_gradient_bound_formula": "(RU^2+RV^2+RU*RV+sigma1)/(m*n)",
        "radius_rule": "RU=RV=1.25*sqrt(sigma1(A))",
        "l1_rule": "0.35*sqrt(n_images*rank)*RV",
        "l1_budget_selection": "Frozen before optimizer runs at 35 percent of the nonnegative-code spectral-ball l1 upper bound",
        "column_norm_min": float(column_norms.min()),
        "column_norm_max": float(column_norms.max()),
        "columns_unit_norm": bool(np.allclose(column_norms, 1.0, rtol=1e-8, atol=1e-10)),
        "data_nonnegative": bool(np.all(A >= 0)),
        "spectral_caps_admit_balanced_unconstrained_svd_factors": True,
        "spectral_cap_margin_over_svd": 1.25,
        "signed_V_l1_global_upper_bound": float(rank * np.sqrt(n) * radius),
        "nonnegative_V_l1_global_upper_bound": float(np.sqrt(n * rank) * radius),
        "l1_budget_over_nonnegative_upper_bound": .35,
        "constraints_guaranteed_active": False,
        "initialization_rule": "independent positive uniform factors; each operator norm at half its cap; exact least-squares scalar along their product; feasible radial repair",
    }


def initialize(problem, seed):
    """Return the same nonzero feasible FP64 point for every method at a seed.

    Both factors are random matrices, not repeated copies of a single column.
    The scalar computation fits the full data only along this one fixed ray;
    it is an initialization, not an optimization loop.
    """
    rng = np.random.default_rng(seed)
    U = rng.uniform(.1, 1.0, size=(problem.m, problem.rank))
    V = rng.uniform(.1, 1.0, size=(problem.n, problem.rank))
    U *= (.5 * problem.RU) / float(svdvals(U, check_finite=False)[0])
    V *= (.5 * problem.RV) / float(svdvals(V, check_finite=False)[0])
    # ||UV^T||_F^2 and <A,UV^T> without allocating the entire product.
    denominator = float(np.sum((U.T @ U) * (V.T @ V)))
    numerator = float(np.sum(U * (problem.M @ V)))
    if not np.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError("Degenerate random factor initialization")
    product_scale = numerator / denominator
    # Face data are nonnegative/nonzero, hence the optimum is strictly positive.
    # Keep a nonzero point even if the helper is reused on signed data whose
    # best nonnegative scalar is zero, rather than initialize at the zero saddle.
    product_scale = max(product_scale, np.finfo(np.float64).eps)
    factor_scale = np.sqrt(product_scale)
    U, V = problem.repair(U * factor_scale, V * factor_scale)
    if not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)):
        raise FloatingPointError("Nonfinite factor initialization")
    return U, V
