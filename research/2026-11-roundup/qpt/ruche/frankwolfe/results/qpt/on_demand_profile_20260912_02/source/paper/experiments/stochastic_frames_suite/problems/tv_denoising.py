"""E2: one-dimensional total-variation denoising with coordinate sampling."""

from __future__ import annotations

import numpy as np

from ..components import BoxConstraint, FiniteDifferenceMap, L1Penalty
from ..problem import BenchmarkProblem


class CoordinateDenoisingSmooth:
    def __init__(self, observation):
        self.observation = np.asarray(observation, dtype=float)

    def value(self, x):
        residual = np.asarray(x) - self.observation
        return 0.5 * float(np.mean(residual**2))

    def gradient(self, x):
        return (np.asarray(x) - self.observation) / self.observation.size

    def gradient_for_batch(self, x, batch):
        batch = np.asarray(batch, dtype=int)
        gradient = np.zeros_like(self.observation)
        residual = np.asarray(x) - self.observation
        np.add.at(gradient, batch, residual[batch] / batch.size)
        return gradient


def make_problem(profile="tiny", problem_seed=0, initialization_seed=0):
    sizes = {"tiny": 24, "small": 160}
    if profile not in sizes:
        raise ValueError("Unknown TV-denoising profile.")
    dimension = sizes[profile]
    rng = np.random.default_rng(problem_seed)
    init_rng = np.random.default_rng(initialization_seed)
    segment_count = 4 if profile == "tiny" else 8
    boundaries = np.linspace(0, dimension, segment_count + 1, dtype=int)
    levels = rng.uniform(0.1, 0.9, size=segment_count)
    truth = np.empty(dimension)
    for start, stop, level in zip(boundaries[:-1], boundaries[1:], levels):
        truth[start:stop] = level
    observation = np.clip(truth + rng.normal(scale=0.12, size=dimension), 0, 1)
    smooth = CoordinateDenoisingSmooth(observation)
    difference = FiniteDifferenceMap(dimension)
    penalty_weight = 0.04
    penalty = L1Penalty(weight=penalty_weight)
    constraint = BoxConstraint(lower=0.0, upper=1.0)
    # Separate initialization randomness while remaining close to the data.
    x0 = np.clip(
        observation + init_rng.normal(scale=0.01, size=dimension), 0, 1
    )

    def metrics(x):
        mse = float(np.mean((x - truth) ** 2))
        differences = difference.forward(x)
        return {
            "mse": mse,
            "psnr": float(-10.0 * np.log10(max(mse, np.finfo(float).tiny))),
            "tv": float(np.sum(np.abs(differences))),
            "change_points": float(np.count_nonzero(np.abs(differences) > 0.05)),
        }

    return BenchmarkProblem(
        name="tv_denoising",
        smooth=smooth,
        composite_map=difference,
        penalty=penalty,
        constraint=constraint,
        x0=x0,
        population_size=dimension,
        metrics=metrics,
        feasibility_or_regularizer=lambda x: float(
            np.sum(np.abs(difference.forward(x)))
        ),
        metadata={
            "experiment": "E2",
            "profile": profile,
            "dimension": dimension,
            "lambda": penalty_weight,
        },
    )
