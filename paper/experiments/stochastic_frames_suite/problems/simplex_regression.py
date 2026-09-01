"""E0: online least-squares regression over the probability simplex."""

from __future__ import annotations

import numpy as np

from ..components import IdentityMap, SimplexConstraint, ZeroPenalty
from ..problem import BenchmarkProblem


class SimplexRegressionSmooth:
    def __init__(self, features, targets):
        self.features = np.asarray(features, dtype=float)
        self.targets = np.asarray(targets, dtype=float)

    def value(self, x):
        residual = self.features @ x - self.targets
        return 0.5 * float(np.mean(residual**2))

    def gradient(self, x):
        residual = self.features @ x - self.targets
        return self.features.T @ residual / self.features.shape[0]

    def gradient_for_batch(self, x, batch):
        features = self.features[np.asarray(batch)]
        residual = features @ x - self.targets[np.asarray(batch)]
        return features.T @ residual / len(batch)


def make_problem(profile="tiny", problem_seed=0, initialization_seed=0):
    sizes = {
        "tiny": dict(dimension=8, train=32, test=32),
        "small": dict(dimension=24, train=256, test=256),
    }
    if profile not in sizes:
        raise ValueError("Unknown simplex-regression profile.")
    size = sizes[profile]
    rng = np.random.default_rng(problem_seed)
    init_rng = np.random.default_rng(initialization_seed)
    dimension = size["dimension"]
    support = min(3, dimension)
    support_indices = rng.choice(dimension, support, replace=False)
    weights = rng.uniform(0.2, 1.0, support)
    weights /= weights.sum()
    truth = np.zeros(dimension)
    truth[support_indices] = weights

    train_features = rng.normal(size=(size["train"], dimension))
    test_features = rng.normal(size=(size["test"], dimension))
    noise_scale = 0.05
    train_targets = train_features @ truth + rng.normal(
        scale=noise_scale, size=size["train"]
    )
    test_targets = test_features @ truth + rng.normal(
        scale=noise_scale, size=size["test"]
    )
    smooth = SimplexRegressionSmooth(train_features, train_targets)
    constraint = SimplexConstraint(radius=1.0)
    x0 = init_rng.dirichlet(np.ones(dimension))

    def metrics(x):
        train_residual = train_features @ x - train_targets
        test_residual = test_features @ x - test_targets
        gradient = smooth.gradient(x)
        atom = constraint.lmo(gradient)
        return {
            "train_mse": float(np.mean(train_residual**2)),
            "test_mse": float(np.mean(test_residual**2)),
            "parameter_error": float(np.linalg.norm(x - truth)),
            "exact_fw_gap": float(np.vdot(gradient, x - atom).real),
        }

    return BenchmarkProblem(
        name="simplex_regression",
        smooth=smooth,
        composite_map=IdentityMap(),
        penalty=ZeroPenalty(),
        constraint=constraint,
        x0=x0,
        population_size=size["train"],
        metrics=metrics,
        metadata={
            "experiment": "E0",
            "profile": profile,
            "dimension": dimension,
            "train_size": size["train"],
            "test_size": size["test"],
        },
    )
