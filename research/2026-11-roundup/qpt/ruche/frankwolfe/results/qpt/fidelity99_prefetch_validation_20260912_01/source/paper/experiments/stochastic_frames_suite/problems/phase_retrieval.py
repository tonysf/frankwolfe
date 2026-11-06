"""E5: total-variation regularized real phase retrieval."""

from __future__ import annotations

import numpy as np

from ..components import FiniteDifferenceMap, L1Penalty, L2BallConstraint
from ..problem import BenchmarkProblem


class PhaseRetrievalSmooth:
    """Finite-sum quartic phase-retrieval loss.

    For rows ``a_i`` of :attr:`measurements` and intensity observations
    ``y_i``, the loss is

    ``mean(((a_i @ x) ** 2 - y_i) ** 2) / 4``.
    """

    def __init__(self, measurements, intensities):
        self.measurements = np.asarray(measurements, dtype=float)
        self.intensities = np.asarray(intensities, dtype=float)
        if self.measurements.ndim != 2:
            raise ValueError("measurements must be a two-dimensional array.")
        if self.intensities.shape != (self.measurements.shape[0],):
            raise ValueError(
                "intensities must have one entry per measurement row."
            )
        # Descriptive aliases used by small analysis scripts.
        self.features = self.measurements
        self.targets = self.intensities

    def residuals(self, x, batch=None):
        if batch is None:
            measurements = self.measurements
            intensities = self.intensities
        else:
            batch = np.asarray(batch, dtype=int)
            measurements = self.measurements[batch]
            intensities = self.intensities[batch]
        amplitudes = measurements @ np.asarray(x)
        return amplitudes, amplitudes**2 - intensities

    def value(self, x):
        _, residuals = self.residuals(x)
        return 0.25 * float(np.mean(residuals**2))

    def gradient(self, x):
        amplitudes, residuals = self.residuals(x)
        return (
            self.measurements.T @ (residuals * amplitudes)
            / self.measurements.shape[0]
        )

    def gradient_for_batch(self, x, batch):
        batch = np.asarray(batch, dtype=int)
        measurements = self.measurements[batch]
        amplitudes, residuals = self.residuals(x, batch)
        return measurements.T @ (residuals * amplitudes) / batch.size


def make_problem(profile="tiny", problem_seed=0, initialization_seed=0):
    """Build a reproducible TV phase-retrieval benchmark."""

    sizes = {
        "tiny": dict(dimension=16, measurements=64, segments=4),
        "small": dict(dimension=64, measurements=512, segments=8),
    }
    if profile not in sizes:
        raise ValueError("Unknown phase-retrieval profile.")
    size = sizes[profile]
    rng = np.random.default_rng(problem_seed)
    init_rng = np.random.default_rng(initialization_seed)
    dimension = size["dimension"]

    boundaries = np.linspace(
        0, dimension, size["segments"] + 1, dtype=int
    )
    levels = rng.uniform(-1.0, 1.0, size=size["segments"])
    # Avoid an accidentally tiny signal, which would make a relative recovery
    # metric ill-conditioned for an otherwise valid seed.
    if np.linalg.norm(levels) < 0.5:
        levels[0] += 1.0
    truth = np.empty(dimension, dtype=float)
    for start, stop, level in zip(boundaries[:-1], boundaries[1:], levels):
        truth[start:stop] = level
    truth /= np.linalg.norm(truth)

    measurements = rng.normal(
        size=(size["measurements"], dimension)
    )
    intensities = (measurements @ truth) ** 2
    smooth = PhaseRetrievalSmooth(measurements, intensities)
    difference = FiniteDifferenceMap(dimension)
    penalty_weight = 0.02
    penalty = L1Penalty(weight=penalty_weight)
    radius = 1.1 * float(np.linalg.norm(truth))
    constraint = L2BallConstraint(radius=radius)

    raw_x0 = init_rng.normal(size=dimension)
    raw_norm = float(np.linalg.norm(raw_x0))
    if raw_norm == 0.0:  # Defensive for custom/random bit generators.
        raw_x0[0] = 1.0
        raw_norm = 1.0
    x0 = raw_x0 * (0.5 * radius / raw_norm)

    metric_beta = 1.0

    def metrics(x):
        x = np.asarray(x)
        prediction_residual = (measurements @ x) ** 2 - intensities
        differences = difference.forward(x)
        smooth_gradient = smooth.gradient(x)
        mapped = difference.forward(x)
        prox_mapped = penalty.prox(mapped, metric_beta)
        moreau_gradient = difference.jacobian_adjoint(
            x, (mapped - prox_mapped) / metric_beta
        )
        combined_gradient = smooth_gradient + moreau_gradient
        atom = constraint.lmo(combined_gradient)
        return {
            "sign_invariant_recovery_error": float(
                min(np.linalg.norm(x - truth), np.linalg.norm(x + truth))
                / np.linalg.norm(truth)
            ),
            "measurement_rmse": float(
                np.sqrt(np.mean(prediction_residual**2))
            ),
            "tv": float(np.sum(np.abs(differences))),
            "phase_exact_smoothed_gap": float(
                np.vdot(combined_gradient, x - atom).real
            ),
        }

    return BenchmarkProblem(
        name="phase_retrieval",
        smooth=smooth,
        composite_map=difference,
        penalty=penalty,
        constraint=constraint,
        x0=x0,
        population_size=size["measurements"],
        metrics=metrics,
        feasibility_or_regularizer=lambda x: float(
            np.sum(np.abs(difference.forward(x)))
        ),
        metadata={
            "experiment": "E5",
            "profile": profile,
            "dimension": dimension,
            "measurements": size["measurements"],
            "segments": size["segments"],
            "lambda": penalty_weight,
            "radius": radius,
            "metric_beta": metric_beta,
        },
    )


__all__ = ["PhaseRetrievalSmooth", "make_problem"]
