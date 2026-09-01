"""Configuration and schedule semantics for stochastic-FRAMES experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np


Schedule = Optional[Union[float, Callable[[int], float]]]
SUPPORTED_METHODS = (
    "momentum",
    "no-momentum",
    "deterministic",
    "fixed-smoothing",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """One benchmark/method/seed run.

    Callable schedules are intentionally supported by the Python API.  The
    archive stores only the realized values, so loading a result never needs
    to unpickle a callable.
    """

    problem: str
    method: str = "momentum"
    profile: str = "tiny"
    steps: int = 100
    batch_size: int = 1
    problem_seed: int = 0
    initialization_seed: int = 0
    sampling_seed: int = 0
    beta0: float = 1.0
    reference_beta: float = 1.0
    rho_scale: float = 1.0
    smoothing_scale: float = 1.0
    step_scale: float = 1.0
    rho_schedule: Schedule = None
    smoothing_schedule: Schedule = None
    step_size_schedule: Schedule = None
    checkpoint_frequency: int = 0
    checkpoint_steps: Optional[Tuple[int, ...]] = None
    show_progress: bool = False

    def __post_init__(self):
        if not self.problem:
            raise ValueError("problem must be a nonempty registry name.")
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {SUPPORTED_METHODS}; got "
                f"{self.method!r}."
            )
        if self.method == "fixed-smoothing" and self.smoothing_schedule is not None:
            raise ValueError(
                "fixed-smoothing does not accept smoothing_schedule; use "
                "beta0 and smoothing_scale for its constant beta."
            )
        if self.profile not in {"tiny", "small"}:
            raise ValueError("profile must be either 'tiny' or 'small'.")
        if not isinstance(self.steps, (int, np.integer)) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer.")
        if (
            not isinstance(self.batch_size, (int, np.integer))
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer.")
        if (
            not isinstance(self.checkpoint_frequency, (int, np.integer))
            or self.checkpoint_frequency < 0
        ):
            raise ValueError(
                "checkpoint_frequency must be a nonnegative integer."
            )
        for name in ("problem_seed", "initialization_seed", "sampling_seed"):
            if not isinstance(getattr(self, name), (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in (
            "beta0",
            "reference_beta",
            "rho_scale",
            "smoothing_scale",
            "step_scale",
        ):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite scalar.")
        if not isinstance(self.show_progress, (bool, np.bool_)):
            raise TypeError("show_progress must be a boolean.")

        if self.checkpoint_steps is not None:
            steps = tuple(int(step) for step in self.checkpoint_steps)
            if any(step < 0 or step > self.steps for step in steps):
                raise ValueError("checkpoint steps must lie in [0, steps].")
            if tuple(sorted(set(steps))) != steps:
                raise ValueError(
                    "checkpoint_steps must be strictly increasing and unique."
                )
            object.__setattr__(self, "checkpoint_steps", steps)

    def resolved_checkpoint_steps(self) -> np.ndarray:
        """Return sorted iteration checkpoints plus the terminal endpoint."""

        if self.checkpoint_steps is not None:
            values = set(self.checkpoint_steps)
        elif self.checkpoint_frequency > 0:
            values = set(range(0, self.steps + 1, self.checkpoint_frequency))
        else:
            # At most eleven intervals keeps post-hoc metrics cheap by default.
            frequency = max(1, self.steps // 10) if self.steps else 1
            values = set(range(0, self.steps + 1, frequency))
        values.update((0, self.steps))
        return np.asarray(sorted(values), dtype=np.int64)

    @staticmethod
    def _scaled_callable(schedule, scale, *, upper=None):
        def resolved(iteration):
            value = float(schedule(iteration)) * scale
            return min(upper, value) if upper is not None else value

        return resolved

    def resolved_schedules(self):
        """Build the actual rho, smoothing, and FW step schedules."""

        default_rho = lambda k: min(1.0, 4.0 / (k + 8) ** (2.0 / 3.0))
        default_beta = lambda k: self.beta0 / (k + 1) ** 0.25
        default_step = lambda k: 1.0 / (k + 1) ** 0.5

        if self.method in {"no-momentum", "deterministic"}:
            rho = 1.0
        elif self.rho_schedule is not None:
            rho = self.rho_schedule
        else:
            rho = self._scaled_callable(
                default_rho, self.rho_scale, upper=1.0
            )

        if self.smoothing_schedule is not None:
            smoothing = self.smoothing_schedule
        elif self.method == "fixed-smoothing":
            smoothing = self.beta0 * self.smoothing_scale
        else:
            smoothing = self._scaled_callable(
                default_beta, self.smoothing_scale
            )

        if self.step_size_schedule is not None:
            step = self.step_size_schedule
        else:
            step = self._scaled_callable(
                default_step, self.step_scale, upper=1.0
            )
        return rho, smoothing, step
