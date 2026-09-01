"""Problem contracts and the adapter consumed by ``StochasticFrames``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from frank_wolfe import ObjectiveFunction


@dataclass(frozen=True)
class SamplePlan:
    """Precomputed minibatches shared across method comparisons."""

    indices: np.ndarray
    population_size: int
    batch_size: int
    seed: int

    def __post_init__(self):
        indices = np.asarray(self.indices)
        if indices.ndim != 2:
            raise ValueError("SamplePlan indices must have shape (steps, batch).")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("SamplePlan indices must be integers.")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.batch_size <= 0 or indices.shape[1] != self.batch_size:
            raise ValueError("batch_size must match the second array dimension.")
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")
        if np.any(indices < 0) or np.any(indices >= self.population_size):
            raise IndexError("SamplePlan contains an out-of-range index.")
        normalized_indices = indices.astype(np.int64, copy=True)
        normalized_indices.setflags(write=False)
        object.__setattr__(self, "indices", normalized_indices)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def steps(self) -> int:
        return int(self.indices.shape[0])

    def batch(self, iteration: int) -> np.ndarray:
        return self.indices[iteration].copy()

    @classmethod
    def generate(cls, population_size, steps, batch_size, seed):
        if population_size <= 0 or steps < 0 or batch_size <= 0:
            raise ValueError(
                "population_size and batch_size must be positive and steps "
                "must be nonnegative."
            )
        rng = np.random.default_rng(seed)
        indices = rng.integers(
            0, population_size, size=(steps, batch_size), dtype=np.int64
        )
        return cls(indices, int(population_size), int(batch_size), int(seed))


@dataclass
class BenchmarkProblem:
    """A finite-sum stochastic composite optimization benchmark."""

    name: str
    smooth: Any
    composite_map: Any
    penalty: Any
    constraint: Any
    x0: np.ndarray
    population_size: int
    metrics: Callable[[np.ndarray], Mapping[str, float]]
    task_loss: Optional[Callable[[np.ndarray], float]] = None
    feasibility_or_regularizer: Optional[Callable[[np.ndarray], float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self):
        self.x0 = np.asarray(self.x0, dtype=float)
        if not self.name:
            raise ValueError("A benchmark must have a nonempty name.")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if not self.constraint.contains(self.x0):
            raise ValueError(f"Initial iterate for {self.name} is infeasible.")
        if self.task_loss is None:
            self.task_loss = lambda x: float(self.smooth.value(x))
        if self.feasibility_or_regularizer is None:
            if self.penalty.kind == "indicator":
                self.feasibility_or_regularizer = self._indicator_residual
            else:
                self.feasibility_or_regularizer = lambda x: float(
                    self.penalty.value(self.composite_map.forward(x))
                )

    def _indicator_residual(self, x):
        y = self.composite_map.forward(x)
        projection = self.penalty.prox(y, 1.0)
        return float(np.linalg.norm(np.asarray(y) - np.asarray(projection)))


class ObjectiveAdapter(ObjectiveFunction):
    """Translate suite component contracts to the legacy optimizer contract.

    The adapter also records lightweight optimizer-time traces and oracle
    counts.  Exact gradients and task metrics are deliberately computed only
    after timing has stopped.
    """

    def __init__(self, problem, sample_plan, *, deterministic=False):
        super().__init__()
        self.problem = problem
        self.sample_plan = sample_plan
        self.deterministic = bool(deterministic)
        if sample_plan.population_size != problem.population_size:
            raise ValueError("Sample plan population does not match problem.")
        self.iteration = 0
        self.stochastic_gradient_calls = 0
        self.sampled_observations = 0
        self.full_gradient_calls = 0
        self.map_forward_calls = 0
        self.map_adjoint_calls = 0
        self.prox_calls = 0
        self.lmo_calls = 0
        self.stochastic_points = []
        self.stochastic_gradients = []
        self.realized_batches = []

    def evaluate(self, x):
        return float(self.problem.smooth.value(x))

    def gradient(self, x):
        self.full_gradient_calls += 1
        return np.asarray(self.problem.smooth.gradient(x))

    def stochastic_gradient(self, x):
        if self.iteration >= self.sample_plan.steps:
            raise RuntimeError("The precomputed SamplePlan is exhausted.")
        if self.deterministic:
            batch = np.arange(self.problem.population_size, dtype=np.int64)
        else:
            batch = self.sample_plan.batch(self.iteration)
        point = np.array(x, copy=True)
        gradient = np.asarray(
            self.problem.smooth.gradient_for_batch(point, batch)
        )
        self.iteration += 1
        self.stochastic_gradient_calls += 1
        self.sampled_observations += int(batch.size)
        self.stochastic_points.append(point)
        self.stochastic_gradients.append(gradient.copy())
        self.realized_batches.append(batch.copy())
        return gradient

    def linear_operator(self, x):
        self.map_forward_calls += 1
        return self.problem.composite_map.forward(x)

    def linear_operator_adjoint(self, y):
        self.map_adjoint_calls += 1
        # Kept for ObjectiveFunction compatibility. Linear maps ignore x.
        return self.problem.composite_map.jacobian_adjoint(None, y)

    def linear_operator_adjoint_at(self, x, y):
        self.map_adjoint_calls += 1
        return self.problem.composite_map.jacobian_adjoint(x, y)

    def minimal_norm_selection(self, y):
        selection = getattr(self.problem.penalty, "minimal_norm_subgradient", None)
        if not callable(selection):
            return np.zeros_like(y, dtype=float)
        return selection(y)

    def prox(self, y, beta):
        self.prox_calls += 1
        return self.problem.penalty.prox(y, beta)

    def lmo(self, gradient):
        self.lmo_calls += 1
        return self.problem.constraint.lmo(gradient)

    def counts(self) -> Dict[str, int]:
        return {
            "stochastic_gradient": self.stochastic_gradient_calls,
            "sampled_observation": self.sampled_observations,
            "full_gradient": self.full_gradient_calls,
            "lmo": self.lmo_calls,
            "prox": self.prox_calls,
            "map_forward": self.map_forward_calls,
            "map_adjoint": self.map_adjoint_calls,
        }


def moreau_value(penalty, y, beta):
    """Evaluate the Moreau envelope from a penalty's prox and value."""

    prox_y = np.asarray(penalty.prox(y, beta))
    penalty_value = float(penalty.value(prox_y))
    if not np.isfinite(penalty_value):
        raise ValueError("prox(y, beta) must lie in the penalty domain.")
    residual = np.asarray(y) - prox_y
    return penalty_value + 0.5 * float(np.vdot(residual, residual).real) / beta
