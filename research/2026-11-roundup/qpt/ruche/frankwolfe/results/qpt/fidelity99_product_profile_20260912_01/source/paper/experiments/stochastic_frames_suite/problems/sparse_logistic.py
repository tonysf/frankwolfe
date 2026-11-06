"""E1: sparse logistic regression with an L1 composite penalty."""

from __future__ import annotations

import numpy as np

from ..components import IdentityMap, L1Penalty, L2BallConstraint
from ..problem import BenchmarkProblem


def _negative_logistic_factor(margin):
    """Return ``1 / (1 + exp(margin))`` without overflow."""

    margin = np.asarray(margin, dtype=float)
    result = np.empty_like(margin)
    nonnegative = margin >= 0
    exp_negative = np.exp(-margin[nonnegative])
    result[nonnegative] = exp_negative / (1.0 + exp_negative)
    exp_positive = np.exp(margin[~nonnegative])
    result[~nonnegative] = 1.0 / (1.0 + exp_positive)
    return result


class LogisticSmooth:
    def __init__(self, features, labels):
        self.features = np.asarray(features, dtype=float)
        self.labels = np.asarray(labels, dtype=float)

    def value(self, x):
        margins = self.labels * (self.features @ x)
        return float(np.mean(np.logaddexp(0.0, -margins)))

    def gradient(self, x):
        margins = self.labels * (self.features @ x)
        coefficients = -self.labels * _negative_logistic_factor(margins)
        return self.features.T @ coefficients / self.features.shape[0]

    def gradient_for_batch(self, x, batch):
        batch = np.asarray(batch)
        features = self.features[batch]
        labels = self.labels[batch]
        margins = labels * (features @ x)
        coefficients = -labels * _negative_logistic_factor(margins)
        return features.T @ coefficients / batch.size


def make_problem(profile="tiny", problem_seed=0, initialization_seed=0):
    sizes = {
        "tiny": dict(samples=40, dimension=10),
        "small": dict(samples=320, dimension=40),
    }
    if profile not in sizes:
        raise ValueError("Unknown sparse-logistic profile.")
    size = sizes[profile]
    rng = np.random.default_rng(problem_seed)
    init_rng = np.random.default_rng(initialization_seed)
    n, dimension = size["samples"], size["dimension"]
    truth = np.zeros(dimension)
    support = rng.choice(dimension, min(4, dimension), replace=False)
    truth[support] = rng.normal(scale=1.0, size=support.size)
    features = rng.normal(size=(n, dimension))
    logits = features @ truth
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    labels = np.where(rng.random(n) < probabilities, 1.0, -1.0)
    radius = max(1.0, 1.25 * np.linalg.norm(truth))
    penalty_weight = 0.04
    smooth = LogisticSmooth(features, labels)
    penalty = L1Penalty(weight=penalty_weight)
    constraint = L2BallConstraint(radius=radius)
    raw_x0 = init_rng.normal(size=dimension)
    x0 = raw_x0 * (0.05 * radius / max(np.linalg.norm(raw_x0), 1e-15))

    def metrics(x):
        scores = features @ x
        predictions = np.where(scores >= 0, 1.0, -1.0)
        return {
            "log_loss": smooth.value(x),
            "accuracy": float(np.mean(predictions == labels)),
            "support_size": float(np.count_nonzero(np.abs(x) > 1e-6)),
            "parameter_error": float(np.linalg.norm(x - truth)),
        }

    return BenchmarkProblem(
        name="sparse_logistic",
        smooth=smooth,
        composite_map=IdentityMap(),
        penalty=penalty,
        constraint=constraint,
        x0=x0,
        population_size=n,
        metrics=metrics,
        feasibility_or_regularizer=lambda x: float(np.sum(np.abs(x))),
        metadata={
            "experiment": "E1",
            "profile": profile,
            "dimension": dimension,
            "samples": n,
            "lambda": penalty_weight,
            "radius": radius,
        },
    )
