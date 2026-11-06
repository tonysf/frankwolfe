"""E4: nonnegative matrix completion over a nuclear-norm ball."""

from __future__ import annotations

import numpy as np

from ..components import IdentityMap, NonnegativeIndicator, NuclearNormBallConstraint
from ..problem import BenchmarkProblem


class ObservedEntrySmooth:
    def __init__(self, shape, rows, columns, values):
        self.shape = tuple(shape)
        self.rows = np.asarray(rows, dtype=int)
        self.columns = np.asarray(columns, dtype=int)
        self.values = np.asarray(values, dtype=float)

    def residuals(self, x, batch=None):
        if batch is None:
            rows, columns, values = self.rows, self.columns, self.values
        else:
            batch = np.asarray(batch, dtype=int)
            rows, columns, values = (
                self.rows[batch],
                self.columns[batch],
                self.values[batch],
            )
        return np.asarray(x)[rows, columns] - values

    def value(self, x):
        residual = self.residuals(x)
        return 0.5 * float(np.mean(residual**2))

    def gradient(self, x):
        gradient = np.zeros(self.shape, dtype=float)
        residual = self.residuals(x)
        np.add.at(
            gradient,
            (self.rows, self.columns),
            residual / self.values.size,
        )
        return gradient

    def gradient_for_batch(self, x, batch):
        batch = np.asarray(batch, dtype=int)
        gradient = np.zeros(self.shape, dtype=float)
        residual = self.residuals(x, batch)
        np.add.at(
            gradient,
            (self.rows[batch], self.columns[batch]),
            residual / batch.size,
        )
        return gradient


def make_problem(profile="tiny", problem_seed=0, initialization_seed=0):
    sizes = {
        "tiny": dict(rows=7, columns=6, rank=2, train_fraction=0.55),
        "small": dict(rows=24, columns=20, rank=3, train_fraction=0.45),
    }
    if profile not in sizes:
        raise ValueError("Unknown matrix-completion profile.")
    size = sizes[profile]
    rng = np.random.default_rng(problem_seed)
    init_rng = np.random.default_rng(initialization_seed)
    left = rng.uniform(0.0, 1.0, size=(size["rows"], size["rank"]))
    right = rng.uniform(0.0, 1.0, size=(size["columns"], size["rank"]))
    truth = left @ right.T
    noisy = truth + rng.normal(scale=0.01, size=truth.shape)
    total = truth.size
    permutation = rng.permutation(total)
    train_count = max(1, int(size["train_fraction"] * total))
    train_flat = permutation[:train_count]
    test_flat = permutation[train_count:]
    train_rows, train_columns = np.unravel_index(train_flat, truth.shape)
    test_rows, test_columns = np.unravel_index(test_flat, truth.shape)
    smooth = ObservedEntrySmooth(
        truth.shape,
        train_rows,
        train_columns,
        noisy[train_rows, train_columns],
    )
    tau = 1.1 * float(np.linalg.svd(truth, compute_uv=False).sum())
    constraint = NuclearNormBallConstraint(radius=tau)
    raw_x0 = init_rng.uniform(size=truth.shape)
    raw_nuclear = float(np.linalg.svd(raw_x0, compute_uv=False).sum())
    x0 = raw_x0 * min(0.05 * tau / max(raw_nuclear, 1e-15), 1.0)

    def rmse(x, rows, columns, target):
        if rows.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((x[rows, columns] - target) ** 2)))

    def metrics(x):
        singular_values = np.linalg.svd(x, compute_uv=False)
        return {
            "train_rmse": rmse(
                x,
                train_rows,
                train_columns,
                noisy[train_rows, train_columns],
            ),
            "test_rmse": rmse(
                x, test_rows, test_columns, truth[test_rows, test_columns]
            ),
            "negative_part_norm": float(
                np.linalg.norm(np.minimum(x, 0.0))
            ),
            "nuclear_norm": float(singular_values.sum()),
            "numerical_rank": float(
                np.count_nonzero(singular_values > 1e-7)
            ),
        }

    return BenchmarkProblem(
        name="matrix_completion",
        smooth=smooth,
        composite_map=IdentityMap(),
        penalty=NonnegativeIndicator(),
        constraint=constraint,
        x0=x0,
        population_size=train_count,
        metrics=metrics,
        feasibility_or_regularizer=lambda x: float(
            np.linalg.norm(np.minimum(x, 0.0))
        ),
        metadata={
            "experiment": "E4",
            "profile": profile,
            "rows": size["rows"],
            "columns": size["columns"],
            "rank": size["rank"],
            "observations": train_count,
            "tau": tau,
        },
    )
