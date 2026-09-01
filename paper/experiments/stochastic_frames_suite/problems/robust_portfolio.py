"""E3: simplex-constrained robust portfolio selection."""

from __future__ import annotations

import numpy as np

from ..components import DenseLinearMap, MaxPenalty, SimplexConstraint
from ..problem import BenchmarkProblem


class PortfolioSmooth:
    def __init__(self, mean_return, centered_scenarios, risk_weight):
        self.mean_return = np.asarray(mean_return, dtype=float)
        self.scenarios = np.asarray(centered_scenarios, dtype=float)
        self.risk_weight = float(risk_weight)

    def value(self, x):
        exposures = self.scenarios @ x
        return float(
            -self.mean_return @ x
            + 0.5 * self.risk_weight * np.mean(exposures**2)
        )

    def gradient(self, x):
        exposures = self.scenarios @ x
        return (
            -self.mean_return
            + self.risk_weight
            * (self.scenarios.T @ exposures)
            / self.scenarios.shape[0]
        )

    def gradient_for_batch(self, x, batch):
        scenarios = self.scenarios[np.asarray(batch)]
        return (
            -self.mean_return
            + self.risk_weight
            * (scenarios.T @ (scenarios @ x))
            / len(batch)
        )


def make_problem(profile="tiny", problem_seed=0, initialization_seed=0):
    sizes = {
        "tiny": dict(assets=8, scenarios=40, factors=2, stresses=5),
        "small": dict(assets=30, scenarios=400, factors=4, stresses=12),
    }
    if profile not in sizes:
        raise ValueError("Unknown robust-portfolio profile.")
    size = sizes[profile]
    rng = np.random.default_rng(problem_seed)
    init_rng = np.random.default_rng(initialization_seed)
    loadings = rng.normal(scale=0.05, size=(size["assets"], size["factors"]))
    factors = rng.normal(size=(size["scenarios"], size["factors"]))
    idiosyncratic = rng.normal(
        scale=0.025, size=(size["scenarios"], size["assets"])
    )
    realized_returns = factors @ loadings.T + idiosyncratic
    mean_return = rng.uniform(0.01, 0.08, size=size["assets"])
    centered = realized_returns - realized_returns.mean(axis=0, keepdims=True)
    stress = np.abs(
        rng.normal(scale=0.12, size=(size["stresses"], size["assets"]))
    )
    risk_weight = 8.0
    stress_weight = 0.25
    smooth = PortfolioSmooth(mean_return, centered, risk_weight)
    composite_map = DenseLinearMap(stress)
    penalty = MaxPenalty(weight=stress_weight)
    constraint = SimplexConstraint(radius=1.0)
    x0 = init_rng.dirichlet(np.ones(size["assets"]))

    def metrics(x):
        exposures = centered @ x
        stress_losses = stress @ x
        return {
            "mean_return": float(mean_return @ x),
            "volatility": float(np.sqrt(np.mean(exposures**2))),
            "worst_stress_loss": float(np.max(stress_losses)),
            "concentration": float(np.sum(x**2)),
        }

    return BenchmarkProblem(
        name="robust_portfolio",
        smooth=smooth,
        composite_map=composite_map,
        penalty=penalty,
        constraint=constraint,
        x0=x0,
        population_size=size["scenarios"],
        metrics=metrics,
        feasibility_or_regularizer=lambda x: float(np.max(stress @ x)),
        metadata={
            "experiment": "E3",
            "profile": profile,
            "assets": size["assets"],
            "scenarios": size["scenarios"],
            "stresses": size["stresses"],
            "gamma": risk_weight,
            "kappa": stress_weight,
        },
    )
