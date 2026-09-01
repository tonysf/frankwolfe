"""E6: factorized real correlation-matrix sensing.

This benchmark is the small real-valued analogue of factorized quantum
process tomography.  Its quadratic-lift composite is an experimental
extension of the linear-map FRAMES formulation.
"""

from __future__ import annotations

import numpy as np

from ..components import (
    FrobeniusBallConstraint,
    QuadraticLiftMap,
    UnitDiagonalIndicator,
)
from ..problem import BenchmarkProblem


class FactorizedCorrelationSmooth:
    """Least-squares sensing loss evaluated through ``U @ U.T``."""

    def __init__(self, sensing_matrices, observations):
        self.sensing_matrices = np.asarray(sensing_matrices, dtype=float)
        self.observations = np.asarray(observations, dtype=float)
        if self.sensing_matrices.ndim != 3:
            raise ValueError("sensing_matrices must be a three-dimensional array.")
        sample_count, rows, columns = self.sensing_matrices.shape
        if rows != columns:
            raise ValueError("Each sensing matrix must be square.")
        if self.observations.shape != (sample_count,):
            raise ValueError(
                "observations must have one entry per sensing matrix."
            )
        if not np.allclose(
            self.sensing_matrices,
            np.swapaxes(self.sensing_matrices, 1, 2),
        ):
            raise ValueError("The sensing matrices must be symmetric.")
        # Concise aliases commonly used in the mathematical formulation.
        self.matrices = self.sensing_matrices
        self.targets = self.observations

    def predictions(self, u, batch=None):
        if batch is None:
            matrices = self.sensing_matrices
        else:
            matrices = self.sensing_matrices[np.asarray(batch, dtype=int)]
        lifted = np.asarray(u) @ np.asarray(u).T
        return np.einsum("kij,ij->k", matrices, lifted, optimize=True)

    def value(self, u):
        residuals = self.predictions(u) - self.observations
        return 0.5 * float(np.mean(residuals**2))

    @staticmethod
    def _mean_gradient(matrices, residuals, u):
        # Since every A_i is symmetric, J_{<A_i, UU^T>}(U)^*[1]
        # equals 2 A_i U.
        weighted = np.einsum(
            "k,kij->ij", residuals, matrices, optimize=True
        )
        return 2.0 * (weighted @ np.asarray(u)) / residuals.size

    def gradient(self, u):
        residuals = self.predictions(u) - self.observations
        return self._mean_gradient(
            self.sensing_matrices, residuals, u
        )

    def gradient_for_batch(self, u, batch):
        batch = np.asarray(batch, dtype=int)
        matrices = self.sensing_matrices[batch]
        residuals = self.predictions(u, batch) - self.observations[batch]
        return self._mean_gradient(matrices, residuals, u)


def make_problem(profile="tiny", problem_seed=0, initialization_seed=0):
    """Build a reproducible nonlinear-composite sensing benchmark."""

    sizes = {
        "tiny": dict(dimension=6, rank=2, measurements=48),
        "small": dict(dimension=18, rank=3, measurements=256),
    }
    if profile not in sizes:
        raise ValueError("Unknown factorized-correlation profile.")
    size = sizes[profile]
    rng = np.random.default_rng(problem_seed)
    init_rng = np.random.default_rng(initialization_seed)
    dimension = size["dimension"]
    rank = size["rank"]

    truth_factor = rng.normal(size=(dimension, rank))
    truth_factor /= np.linalg.norm(truth_factor, axis=1, keepdims=True)
    truth_lift = truth_factor @ truth_factor.T

    raw_sensing = rng.normal(
        scale=1.0 / np.sqrt(dimension),
        size=(size["measurements"], dimension, dimension),
    )
    sensing_matrices = 0.5 * (
        raw_sensing + np.swapaxes(raw_sensing, 1, 2)
    )
    observations = np.einsum(
        "kij,ij->k", sensing_matrices, truth_lift, optimize=True
    )
    smooth = FactorizedCorrelationSmooth(sensing_matrices, observations)
    lift = QuadraticLiftMap()
    penalty = UnitDiagonalIndicator()
    radius = 1.1 * float(np.linalg.norm(truth_factor))
    constraint = FrobeniusBallConstraint(radius=radius)

    x0 = init_rng.normal(size=(dimension, rank))
    row_norms = np.linalg.norm(x0, axis=1, keepdims=True)
    zero_rows = row_norms[:, 0] == 0.0
    if np.any(zero_rows):  # Defensive for custom/random bit generators.
        x0[zero_rows, 0] = 1.0
        row_norms = np.linalg.norm(x0, axis=1, keepdims=True)
    x0 /= row_norms

    metric_beta = 1.0

    def metrics(u):
        u = np.asarray(u)
        lifted = lift.forward(u)
        residuals = smooth.predictions(u) - observations
        diagonal_residual = np.diag(lifted) - 1.0
        smooth_gradient = smooth.gradient(u)
        projected = penalty.prox(lifted, metric_beta)
        moreau_gradient = lift.jacobian_adjoint(
            u, (lifted - projected) / metric_beta
        )
        combined_gradient = smooth_gradient + moreau_gradient
        atom = constraint.lmo(combined_gradient)
        singular_values = np.linalg.svd(lifted, compute_uv=False)
        rank_threshold = 1e-7 * max(1.0, singular_values[0])
        return {
            "lifted_recovery_error": float(
                np.linalg.norm(lifted - truth_lift, ord="fro")
                / np.linalg.norm(truth_lift, ord="fro")
            ),
            "sensing_rmse": float(np.sqrt(np.mean(residuals**2))),
            "diagonal_violation": float(np.linalg.norm(diagonal_residual)),
            "rank": float(np.count_nonzero(singular_values > rank_threshold)),
            "nonlinear_composite_fw_gap": float(
                np.vdot(combined_gradient, u - atom).real
            ),
        }

    return BenchmarkProblem(
        name="factorized_correlation",
        smooth=smooth,
        composite_map=lift,
        penalty=penalty,
        constraint=constraint,
        x0=x0,
        population_size=size["measurements"],
        metrics=metrics,
        feasibility_or_regularizer=lambda u: float(
            np.linalg.norm(np.diag(lift.forward(u)) - 1.0)
        ),
        metadata={
            "experiment": "E6",
            "profile": profile,
            "dimension": dimension,
            "rank": rank,
            "measurements": size["measurements"],
            "radius": radius,
            "metric_beta": metric_beta,
            "nonlinear_composite": True,
            "nonlinear_composite_status": "experimental",
        },
    )


__all__ = ["FactorizedCorrelationSmooth", "make_problem"]
