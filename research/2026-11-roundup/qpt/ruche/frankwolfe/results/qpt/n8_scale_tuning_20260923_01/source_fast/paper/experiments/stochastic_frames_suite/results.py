"""Structured results and seed aggregation for stochastic-FRAMES runs.

This module deliberately contains no optimizer or problem dependencies.  The
small, NumPy-only data model is shared by runners, archive I/O, and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np


SCHEMA_VERSION = 1
"""Current on-disk and in-memory result schema version."""

RESULT_SCHEMA_VERSION = SCHEMA_VERSION
"""Descriptive alias used by callers that also have problem schemas."""


_METADATA_FIELDS = {
    "problem_name",
    "n_steps",
    "problem_seed",
    "initialization_seed",
    "sampling_seed",
    "batch_size",
    "method",
    "schema_version",
    "reference_beta",
    "extra",
}


def _integer(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _json_value(value: Any, path: str = "metadata") -> Any:
    """Normalize metadata to the strict, portable JSON value subset."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings.")
            normalized[key] = _json_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported value {type(value).__name__}; "
        "use JSON scalars, lists, and string-keyed mappings."
    )


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    """Return the canonical representation used by :mod:`.io`."""

    return json.dumps(
        _json_value(metadata), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class RunMetadata:
    """Identity and reproducibility settings for one optimization run."""

    problem_name: str
    n_steps: int
    problem_seed: int = 0
    initialization_seed: int = 0
    sampling_seed: int = 0
    batch_size: int = 1
    method: str = "momentum"
    schema_version: int = SCHEMA_VERSION
    reference_beta: Optional[float] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.problem_name, str) or not self.problem_name.strip():
            raise ValueError("problem_name must be a non-empty string.")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string.")

        object.__setattr__(self, "problem_name", self.problem_name.strip())
        object.__setattr__(self, "method", self.method.strip())
        object.__setattr__(self, "n_steps", _integer(self.n_steps, "n_steps", minimum=0))
        object.__setattr__(
            self, "problem_seed", _integer(self.problem_seed, "problem_seed", minimum=0)
        )
        object.__setattr__(
            self,
            "initialization_seed",
            _integer(self.initialization_seed, "initialization_seed", minimum=0),
        )
        object.__setattr__(
            self,
            "sampling_seed",
            _integer(self.sampling_seed, "sampling_seed", minimum=0),
        )
        object.__setattr__(
            self, "batch_size", _integer(self.batch_size, "batch_size", minimum=1)
        )
        version = _integer(self.schema_version, "schema_version", minimum=1)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported result schema version {version}; "
                f"this code supports version {SCHEMA_VERSION}."
            )
        object.__setattr__(self, "schema_version", version)

        if self.reference_beta is not None:
            beta = float(self.reference_beta)
            if not np.isfinite(beta) or beta <= 0.0:
                raise ValueError("reference_beta must be positive and finite.")
            object.__setattr__(self, "reference_beta", beta)

        normalized_extra = _json_value(self.extra, "metadata.extra")
        conflicting = sorted(set(normalized_extra).intersection(_METADATA_FIELDS))
        if conflicting:
            raise ValueError(
                "metadata.extra uses reserved keys: " + ", ".join(conflicting)
            )
        object.__setattr__(self, "extra", normalized_extra)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RunMetadata":
        """Create metadata from either nested or flat dictionary data.

        Unknown flat keys are retained in :attr:`extra`, which makes the
        constructor convenient for CLI configuration dictionaries without
        weakening validation of the required reproducibility fields.
        """

        if not isinstance(values, Mapping):
            raise TypeError("metadata must be RunMetadata or a mapping.")
        values = dict(values)
        explicit_extra = values.pop("extra", {})
        if not isinstance(explicit_extra, Mapping):
            raise TypeError("metadata.extra must be a mapping.")
        known = {key: values.pop(key) for key in list(values) if key in _METADATA_FIELDS}
        extra = dict(explicit_extra)
        duplicate = sorted(set(extra).intersection(values))
        if duplicate:
            raise ValueError("duplicate metadata keys: " + ", ".join(duplicate))
        extra.update(values)
        known["extra"] = extra
        try:
            return cls(**known)
        except TypeError as error:
            raise TypeError(f"invalid run metadata: {error}") from error

    def to_dict(self, *, flatten_extra: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "problem_name": self.problem_name,
            "n_steps": self.n_steps,
            "problem_seed": self.problem_seed,
            "initialization_seed": self.initialization_seed,
            "sampling_seed": self.sampling_seed,
            "batch_size": self.batch_size,
            "method": self.method,
            "schema_version": self.schema_version,
            "reference_beta": self.reference_beta,
        }
        if flatten_extra:
            result.update(self.extra)
        else:
            result["extra"] = dict(self.extra)
        return result


def validate_metadata(
    metadata: Any, expected: Optional[Any] = None
) -> RunMetadata:
    """Normalize metadata and optionally check an expected subset.

    ``expected`` may be another :class:`RunMetadata` or a mapping.  Mapping
    keys can name core metadata fields or entries in ``metadata.extra``.
    """

    if isinstance(metadata, RunMetadata):
        # Reconstruct so validation remains meaningful if a caller mutated
        # the mapping held in ``extra`` after the frozen dataclass was made.
        result = RunMetadata.from_mapping(metadata.to_dict())
    elif isinstance(metadata, Mapping):
        result = RunMetadata.from_mapping(metadata)
    else:
        raise TypeError("metadata must be RunMetadata or a mapping.")

    if expected is None:
        return result
    actual = result.to_dict(flatten_extra=True)
    if isinstance(expected, RunMetadata):
        expected_values = expected.to_dict(flatten_extra=True)
    elif isinstance(expected, Mapping):
        expected_values = dict(expected)
        nested_extra = expected_values.pop("extra", {})
        if not isinstance(nested_extra, Mapping):
            raise TypeError("expected metadata 'extra' value must be a mapping.")
        duplicate = set(nested_extra).intersection(expected_values)
        if duplicate:
            raise ValueError(
                "duplicate expected metadata keys: " + ", ".join(sorted(duplicate))
            )
        expected_values.update(nested_extra)
    else:
        raise TypeError("expected metadata must be RunMetadata or a mapping.")
    for key, wanted in expected_values.items():
        if key not in actual:
            raise ValueError(f"archive metadata has no {key!r} field.")
        if actual[key] != _json_value(wanted, f"expected.{key}"):
            raise ValueError(
                f"archive metadata mismatch for {key!r}: "
                f"expected {wanted!r}, found {actual[key]!r}."
            )
    return result


def _numeric_array(
    value: Any,
    name: str,
    *,
    ndim: Optional[int] = None,
    finite: bool = False,
    real: bool = False,
    copy: bool = True,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"{name} must be a numeric array, not {array.dtype}.")
    if real and array.dtype.kind == "c":
        raise TypeError(f"{name} must be real-valued.")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got {array.ndim}.")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, copy=copy)


def _float_trace(
    value: Any,
    name: str,
    length: int,
    *,
    finite: bool = True,
) -> np.ndarray:
    array = _numeric_array(value, name, ndim=1, finite=finite, real=True)
    if len(array) != length:
        raise ValueError(f"{name} must have length {length}, got {len(array)}.")
    return array.astype(float, copy=False)


def _count_trace(value: Any, name: str) -> np.ndarray:
    array = _numeric_array(value, name, ndim=1, finite=True, real=True)
    if np.any(array < 0) or np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain nonnegative integer counts.")
    result = array.astype(np.int64, copy=False)
    if len(result) > 1 and np.any(np.diff(result) < 0):
        raise ValueError(f"{name} must be cumulative and nondecreasing.")
    return result


@dataclass(eq=False)
class OracleCounts:
    """Cumulative oracle work, recorded either per step or per checkpoint."""

    stochastic_gradients: np.ndarray
    sampled_observations: np.ndarray
    lmo_calls: np.ndarray
    prox_calls: np.ndarray
    map_calls: np.ndarray
    metric_calls: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "stochastic_gradients",
            "sampled_observations",
            "lmo_calls",
            "prox_calls",
            "map_calls",
            "metric_calls",
        ):
            setattr(self, name, _count_trace(getattr(self, name), f"oracle_counts.{name}"))
        lengths = {len(getattr(self, name)) for name in self.field_names()}
        if len(lengths) != 1:
            raise ValueError("all oracle count traces must have the same length.")

    @staticmethod
    def field_names() -> Tuple[str, ...]:
        return (
            "stochastic_gradients",
            "sampled_observations",
            "lmo_calls",
            "prox_calls",
            "map_calls",
            "metric_calls",
        )

    @classmethod
    def zeros(cls, length: int) -> "OracleCounts":
        length = _integer(length, "length", minimum=0)
        values = [np.zeros(length, dtype=np.int64) for _ in cls.field_names()]
        return cls(*values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OracleCounts":
        if not isinstance(values, Mapping):
            raise TypeError("oracle_counts must be OracleCounts or a mapping.")
        aliases = {
            "stochastic_gradient_calls": "stochastic_gradients",
            "stochastic_gradient": "stochastic_gradients",
            "sampled_observation_calls": "sampled_observations",
            "sampled_observation": "sampled_observations",
            "lmo": "lmo_calls",
            "prox": "prox_calls",
            "map": "map_calls",
            "metric": "metric_calls",
        }
        normalized: Dict[str, Any] = {}
        for key, value in values.items():
            canonical = aliases.get(key, key)
            if canonical in normalized:
                raise ValueError(f"duplicate oracle count {canonical!r}.")
            normalized[canonical] = value
        unknown = sorted(set(normalized).difference(cls.field_names()))
        if unknown:
            raise ValueError("unknown oracle counts: " + ", ".join(unknown))
        missing = sorted(set(cls.field_names()).difference(normalized))
        if missing:
            raise ValueError("missing oracle counts: " + ", ".join(missing))
        return cls(**normalized)

    def __len__(self) -> int:
        return len(self.stochastic_gradients)

    @property
    def stochastic_gradient_calls(self) -> np.ndarray:
        return self.stochastic_gradients

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.field_names()}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OracleCounts) and all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in self.field_names()
        )


_CHECKPOINT_TRACE_FIELDS = (
    "task_loss",
    "composite_objective",
    "smoothed_objective",
    "exact_smoothed_gap",
    "reference_objective",
    "reference_gap",
    "estimator_error",
    "feasibility_or_regularizer",
    "optimizer_time",
)


@dataclass(eq=False)
class RunResult:
    """Complete reproducible trace for one stochastic-FRAMES run.

    Schedule and estimated-gap arrays have one entry per optimizer step.
    Expensive metrics, iterates, and optimizer time have one entry per
    checkpoint.  Oracle counts may use either convention; use
    :meth:`oracle_counts_at_checkpoints` when checkpoint alignment is needed.
    """

    metadata: RunMetadata
    checkpoint_steps: np.ndarray
    checkpoint_iterates: np.ndarray
    momentum_weights: np.ndarray
    smoothing_parameters: np.ndarray
    step_sizes: np.ndarray
    estimated_gaps: np.ndarray
    task_loss: np.ndarray
    composite_objective: np.ndarray
    smoothed_objective: np.ndarray
    exact_smoothed_gap: np.ndarray
    reference_objective: np.ndarray
    reference_gap: np.ndarray
    estimator_error: np.ndarray
    feasibility_or_regularizer: np.ndarray
    oracle_counts: OracleCounts
    optimizer_time: np.ndarray
    problem_metrics: Mapping[str, np.ndarray]
    final_iterate: np.ndarray
    final_gradient_estimate: np.ndarray

    def __post_init__(self) -> None:
        self.metadata = validate_metadata(self.metadata)
        self.checkpoint_steps = _count_trace(
            self.checkpoint_steps, "checkpoint_steps"
        )
        if len(self.checkpoint_steps) == 0:
            raise ValueError("checkpoint_steps must not be empty.")
        if self.checkpoint_steps[0] != 0:
            raise ValueError("checkpoint_steps must begin at zero.")
        if self.checkpoint_steps[-1] != self.metadata.n_steps:
            raise ValueError("checkpoint_steps must end at metadata.n_steps.")
        if len(np.unique(self.checkpoint_steps)) != len(self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be strictly increasing.")

        n_checkpoints = len(self.checkpoint_steps)
        self.checkpoint_iterates = _numeric_array(
            self.checkpoint_iterates,
            "checkpoint_iterates",
            finite=True,
        )
        if self.checkpoint_iterates.ndim < 1:
            raise ValueError("checkpoint_iterates must have a checkpoint axis.")
        if self.checkpoint_iterates.shape[0] != n_checkpoints:
            raise ValueError(
                "checkpoint_iterates first dimension must match checkpoint_steps."
            )

        self.momentum_weights = _float_trace(
            self.momentum_weights,
            "momentum_weights",
            self.metadata.n_steps,
        )
        self.smoothing_parameters = _float_trace(
            self.smoothing_parameters,
            "smoothing_parameters",
            self.metadata.n_steps,
        )
        self.step_sizes = _float_trace(
            self.step_sizes, "step_sizes", self.metadata.n_steps
        )
        self.estimated_gaps = _float_trace(
            self.estimated_gaps, "estimated_gaps", self.metadata.n_steps
        )
        if np.any((self.momentum_weights <= 0.0) | (self.momentum_weights > 1.0)):
            raise ValueError("momentum_weights must lie in (0, 1].")
        if np.any(self.smoothing_parameters <= 0.0):
            raise ValueError("smoothing_parameters must be positive.")
        if np.any((self.step_sizes < 0.0) | (self.step_sizes > 1.0)):
            raise ValueError("step_sizes must lie in [0, 1].")

        for name in _CHECKPOINT_TRACE_FIELDS:
            allow_nonfinite = name == "composite_objective"
            setattr(
                self,
                name,
                _float_trace(
                    getattr(self, name),
                    name,
                    n_checkpoints,
                    finite=not allow_nonfinite,
                ),
            )
        if np.any(self.optimizer_time < 0.0) or np.any(np.diff(self.optimizer_time) < 0.0):
            raise ValueError("optimizer_time must be nonnegative and nondecreasing.")
        if np.any(self.estimator_error < 0.0):
            raise ValueError("estimator_error must be nonnegative.")
        if np.any(self.feasibility_or_regularizer < 0.0):
            raise ValueError("feasibility_or_regularizer must be nonnegative.")

        if isinstance(self.oracle_counts, Mapping):
            self.oracle_counts = OracleCounts.from_mapping(self.oracle_counts)
        elif not isinstance(self.oracle_counts, OracleCounts):
            raise TypeError("oracle_counts must be OracleCounts or a mapping.")
        count_length = len(self.oracle_counts)
        if count_length not in {self.metadata.n_steps, n_checkpoints}:
            raise ValueError(
                "oracle count traces must be recorded per step or per checkpoint."
            )

        if not isinstance(self.problem_metrics, Mapping):
            raise TypeError("problem_metrics must be a mapping.")
        metrics: Dict[str, np.ndarray] = {}
        reserved = set(self.trajectory_names())
        for name, values in self.problem_metrics.items():
            if not isinstance(name, str) or not name:
                raise TypeError("problem metric names must be non-empty strings.")
            if name in metrics:
                raise ValueError(f"duplicate problem metric {name!r}.")
            if name in reserved:
                raise ValueError(f"problem metric name {name!r} is reserved.")
            metric = _numeric_array(
                values, f"problem_metrics[{name!r}]", finite=True, real=True
            )
            if metric.ndim < 1 or metric.shape[0] != n_checkpoints:
                raise ValueError(
                    f"problem metric {name!r} must have checkpoint-leading "
                    f"shape ({n_checkpoints}, ...)."
                )
            metrics[name] = metric
        self.problem_metrics = metrics

        self.final_iterate = _numeric_array(
            self.final_iterate, "final_iterate", finite=True
        )
        self.final_gradient_estimate = _numeric_array(
            self.final_gradient_estimate,
            "final_gradient_estimate",
            finite=True,
        )
        iterate_shape = self.checkpoint_iterates.shape[1:]
        if self.final_iterate.shape != iterate_shape:
            raise ValueError(
                "final_iterate shape must match a checkpoint iterate: "
                f"expected {iterate_shape}, got {self.final_iterate.shape}."
            )
        if self.final_gradient_estimate.shape != iterate_shape:
            raise ValueError(
                "final_gradient_estimate shape must match final_iterate."
            )
        if not np.array_equal(
            self.final_iterate, self.checkpoint_iterates[-1], equal_nan=True
        ):
            raise ValueError("final_iterate must equal the terminal checkpoint iterate.")

    @property
    def problem_name(self) -> str:
        return self.metadata.problem_name

    @property
    def schema_version(self) -> int:
        return self.metadata.schema_version

    @property
    def method(self) -> str:
        return self.metadata.method

    @property
    def problem_seed(self) -> int:
        return self.metadata.problem_seed

    @property
    def initialization_seed(self) -> int:
        return self.metadata.initialization_seed

    @property
    def sampling_seed(self) -> int:
        return self.metadata.sampling_seed

    @property
    def batch_size(self) -> int:
        return self.metadata.batch_size

    @property
    def n_steps(self) -> int:
        return self.metadata.n_steps

    @property
    def rho_values(self) -> np.ndarray:
        return self.momentum_weights

    @property
    def rho(self) -> np.ndarray:
        return self.momentum_weights

    @property
    def optimizer_seconds(self) -> np.ndarray:
        return self.optimizer_time

    @property
    def cumulative_stochastic_gradients(self) -> np.ndarray:
        return self.oracle_counts.stochastic_gradients

    @property
    def cumulative_sampled_observations(self) -> np.ndarray:
        return self.oracle_counts.sampled_observations

    @property
    def cumulative_lmo_calls(self) -> np.ndarray:
        return self.oracle_counts.lmo_calls

    @property
    def cumulative_prox_calls(self) -> np.ndarray:
        return self.oracle_counts.prox_calls

    @property
    def cumulative_map_calls(self) -> np.ndarray:
        return self.oracle_counts.map_calls

    @property
    def cumulative_metric_calls(self) -> np.ndarray:
        return self.oracle_counts.metric_calls

    @property
    def fixed_reference_objective(self) -> np.ndarray:
        return self.reference_objective

    @property
    def fixed_reference_gap(self) -> np.ndarray:
        return self.reference_gap

    @staticmethod
    def trajectory_names() -> Tuple[str, ...]:
        return (
            "momentum_weights",
            "smoothing_parameters",
            "step_sizes",
            "estimated_gaps",
            "task_loss",
            "composite_objective",
            "smoothed_objective",
            "exact_smoothed_gap",
            "reference_objective",
            "reference_gap",
            "estimator_error",
            "feasibility_or_regularizer",
            "optimizer_time",
            "stochastic_gradients",
            "sampled_observations",
            "lmo_calls",
            "prox_calls",
            "map_calls",
            "metric_calls",
        )

    def oracle_counts_at_checkpoints(self) -> OracleCounts:
        """Return counts aligned to ``checkpoint_steps``.

        Suite archives store counts directly at pre-update checkpoints.  For
        ``k < T`` these include the work used to form ``(x_k, d_k, beta_k)``;
        the terminal checkpoint reuses the last optimizer count.  Legacy
        per-completed-step traces instead assign zero to checkpoint zero.
        """

        if self.metadata.extra.get("oracle_count_axis") == "checkpoint_pre_update":
            return OracleCounts(
                *(values.copy() for values in self.oracle_counts.to_dict().values())
            )
        # Prefer the per-step interpretation in an otherwise ambiguous legacy
        # case where n_steps equals the number of checkpoints.
        if len(self.oracle_counts) == self.metadata.n_steps:
            indices = self.checkpoint_steps - 1
            data = []
            for values in self.oracle_counts.to_dict().values():
                aligned = np.zeros(len(self.checkpoint_steps), dtype=np.int64)
                positive = self.checkpoint_steps > 0
                aligned[positive] = values[indices[positive]]
                data.append(aligned)
            return OracleCounts(*data)
        return OracleCounts(
            *(values.copy() for values in self.oracle_counts.to_dict().values())
        )

    def trajectory(self, name: str) -> np.ndarray:
        if name in _CHECKPOINT_TRACE_FIELDS or name in {
            "momentum_weights",
            "smoothing_parameters",
            "step_sizes",
            "estimated_gaps",
        }:
            return getattr(self, name)
        if name in OracleCounts.field_names():
            return getattr(self.oracle_counts, name)
        if name in self.problem_metrics:
            return self.problem_metrics[name]
        raise KeyError(f"unknown result trajectory {name!r}.")

    def trajectories(self) -> Dict[str, np.ndarray]:
        values = {name: self.trajectory(name) for name in self.trajectory_names()}
        values.update(self.problem_metrics)
        return values

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunResult) or self.metadata != other.metadata:
            return False
        for name in (
            "checkpoint_steps",
            "checkpoint_iterates",
            "final_iterate",
            "final_gradient_estimate",
        ) + self.trajectory_names():
            left = self.trajectory(name) if name in self.trajectory_names() else getattr(self, name)
            right = other.trajectory(name) if name in self.trajectory_names() else getattr(other, name)
            if not np.array_equal(left, right, equal_nan=True):
                return False
        if set(self.problem_metrics) != set(other.problem_metrics):
            return False
        return all(
            np.array_equal(
                self.problem_metrics[name], other.problem_metrics[name], equal_nan=True
            )
            for name in self.problem_metrics
        )


# The descriptive name reads better in public APIs; keep both for convenience.
ExperimentResult = RunResult


def validate_result(result: Any) -> RunResult:
    """Validate that ``result`` is a fully normalized :class:`RunResult`."""

    if not isinstance(result, RunResult):
        raise TypeError("result must be a RunResult.")
    # Construction performs full validation.  Reconstructing also catches
    # mutation of an array or mutable problem-metrics mapping after creation.
    return RunResult(
        metadata=result.metadata,
        checkpoint_steps=result.checkpoint_steps,
        checkpoint_iterates=result.checkpoint_iterates,
        momentum_weights=result.momentum_weights,
        smoothing_parameters=result.smoothing_parameters,
        step_sizes=result.step_sizes,
        estimated_gaps=result.estimated_gaps,
        task_loss=result.task_loss,
        composite_objective=result.composite_objective,
        smoothed_objective=result.smoothed_objective,
        exact_smoothed_gap=result.exact_smoothed_gap,
        reference_objective=result.reference_objective,
        reference_gap=result.reference_gap,
        estimator_error=result.estimator_error,
        feasibility_or_regularizer=result.feasibility_or_regularizer,
        oracle_counts=result.oracle_counts.to_dict(),
        optimizer_time=result.optimizer_time,
        problem_metrics=result.problem_metrics,
        final_iterate=result.final_iterate,
        final_gradient_estimate=result.final_gradient_estimate,
    )


@dataclass(frozen=True, eq=False)
class QuantileSummary:
    """Median and requested pointwise quantiles for one numeric trajectory."""

    median: np.ndarray
    quantile_levels: np.ndarray
    quantile_values: np.ndarray

    def __post_init__(self) -> None:
        median = _numeric_array(self.median, "median", real=True)
        levels = _numeric_array(
            self.quantile_levels, "quantile_levels", ndim=1, finite=True, real=True
        ).astype(float, copy=False)
        values = _numeric_array(self.quantile_values, "quantile_values", real=True)
        if np.any((levels < 0.0) | (levels > 1.0)) or np.any(np.diff(levels) <= 0.0):
            raise ValueError("quantile_levels must be strictly increasing in [0, 1].")
        if values.shape != (len(levels),) + median.shape:
            raise ValueError("quantile_values shape does not match levels and median.")
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "quantile_levels", levels)
        object.__setattr__(self, "quantile_values", values)

    @property
    def lower(self) -> np.ndarray:
        return self.quantile_values[0]

    @property
    def upper(self) -> np.ndarray:
        return self.quantile_values[-1]

    def at(self, quantile: float) -> np.ndarray:
        matches = np.flatnonzero(np.isclose(self.quantile_levels, quantile))
        if len(matches) == 0:
            raise KeyError(f"quantile {quantile} was not requested.")
        return self.quantile_values[int(matches[0])]


@dataclass(frozen=True)
class AggregatedResult:
    """Pointwise seed summary for compatible runs."""

    problem_name: str
    method: str
    schema_version: int
    n_runs: int
    checkpoint_steps: np.ndarray
    quantile_levels: np.ndarray
    summaries: Mapping[str, QuantileSummary]
    problem_seeds: np.ndarray
    initialization_seeds: np.ndarray
    sampling_seeds: np.ndarray

    @property
    def metrics(self) -> Mapping[str, QuantileSummary]:
        return self.summaries

    @property
    def median(self) -> Dict[str, np.ndarray]:
        return {name: summary.median for name, summary in self.summaries.items()}

    @property
    def lower(self) -> Dict[str, np.ndarray]:
        return {name: summary.lower for name, summary in self.summaries.items()}

    @property
    def upper(self) -> Dict[str, np.ndarray]:
        return {name: summary.upper for name, summary in self.summaries.items()}

    def metric(self, name: str) -> QuantileSummary:
        try:
            return self.summaries[name]
        except KeyError as error:
            raise KeyError(f"unknown aggregated trajectory {name!r}.") from error

    def __getattr__(self, name: str) -> QuantileSummary:
        summaries = object.__getattribute__(self, "summaries")
        if name in summaries:
            return summaries[name]
        raise AttributeError(name)


def _quantile_levels(quantiles: Sequence[float]) -> np.ndarray:
    levels = np.asarray(tuple(quantiles), dtype=float)
    if levels.ndim != 1 or len(levels) == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(levels)) or np.any((levels < 0.0) | (levels > 1.0)):
        raise ValueError("quantiles must be finite and lie in [0, 1].")
    levels = np.unique(levels)
    if len(levels) == 0:
        raise ValueError("at least one quantile is required.")
    return levels


def _compatible_runs(results: Sequence[RunResult]) -> None:
    first = results[0]
    first_metric_names = set(first.problem_metrics)
    for index, result in enumerate(results[1:], start=1):
        if not isinstance(result, RunResult):
            raise TypeError(f"results[{index}] is not a RunResult.")
        mismatches = []
        for field_name in ("problem_name", "method", "schema_version", "n_steps", "batch_size"):
            if getattr(result, field_name) != getattr(first, field_name):
                mismatches.append(field_name)
        if result.metadata.reference_beta != first.metadata.reference_beta:
            mismatches.append("reference_beta")
        for schedule_name in (
            "momentum_weights",
            "smoothing_parameters",
            "step_sizes",
        ):
            if not np.array_equal(
                getattr(result, schedule_name), getattr(first, schedule_name)
            ):
                mismatches.append(schedule_name)
        for setting_name in (
            "profile",
            "problem_schema_version",
            "actual_batch_size",
        ):
            if result.metadata.extra.get(setting_name) != first.metadata.extra.get(
                setting_name
            ):
                mismatches.append(setting_name)
        if mismatches:
            raise ValueError(
                f"results[{index}] is incompatible in: " + ", ".join(mismatches)
            )
        if not np.array_equal(result.checkpoint_steps, first.checkpoint_steps):
            raise ValueError(f"results[{index}] has different checkpoint_steps.")
        if set(result.problem_metrics) != first_metric_names:
            raise ValueError(f"results[{index}] has different problem metrics.")


def aggregate_results(
    results: Iterable[RunResult],
    quantiles: Sequence[float] = (0.25, 0.75),
    metric_names: Optional[Iterable[str]] = None,
) -> AggregatedResult:
    """Aggregate compatible runs pointwise using median and quantiles.

    Problem, initialization, and sampling seeds are intentionally allowed to
    differ.  All non-seed run settings, checkpoints, metric names, and metric
    shapes must match.
    """

    runs = list(results)
    if not runs:
        raise ValueError("at least one result is required for aggregation.")
    if not isinstance(runs[0], RunResult):
        raise TypeError("results[0] is not a RunResult.")
    _compatible_runs(runs)
    levels = _quantile_levels(quantiles)

    available = runs[0].trajectories()
    if metric_names is None:
        names = tuple(available)
    else:
        names = tuple(metric_names)
        if len(set(names)) != len(names):
            raise ValueError("metric_names contains duplicates.")
        missing = sorted(set(names).difference(available))
        if missing:
            raise KeyError("unknown trajectories: " + ", ".join(missing))

    summaries: Dict[str, QuantileSummary] = {}
    for name in names:
        arrays = [result.trajectory(name) for result in runs]
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(f"trajectory {name!r} has inconsistent shapes.")
        stack = np.stack(arrays, axis=0).astype(float, copy=False)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(stack, axis=0)
            values = np.nanquantile(stack, levels, axis=0)
        summaries[name] = QuantileSummary(median, levels, values)

    first = runs[0]
    return AggregatedResult(
        problem_name=first.problem_name,
        method=first.method,
        schema_version=first.schema_version,
        n_runs=len(runs),
        checkpoint_steps=first.checkpoint_steps.copy(),
        quantile_levels=levels.copy(),
        summaries=summaries,
        problem_seeds=np.asarray([result.problem_seed for result in runs], dtype=int),
        initialization_seeds=np.asarray(
            [result.initialization_seed for result in runs], dtype=int
        ),
        sampling_seeds=np.asarray(
            [result.sampling_seed for result in runs], dtype=int
        ),
    )


def aggregate_seeds(
    results: Iterable[RunResult],
    quantiles: Sequence[float] = (0.25, 0.75),
    metric_names: Optional[Iterable[str]] = None,
) -> AggregatedResult:
    """Alias emphasizing the intended aggregation dimension."""

    return aggregate_results(results, quantiles=quantiles, metric_names=metric_names)


def aggregate_metric(
    results: Iterable[RunResult],
    metric_name: str,
    quantiles: Sequence[float] = (0.25, 0.75),
) -> QuantileSummary:
    """Return the seed summary for one named trajectory."""

    return aggregate_results(
        results, quantiles=quantiles, metric_names=(metric_name,)
    ).metric(metric_name)


__all__ = [
    "SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "RunMetadata",
    "OracleCounts",
    "RunResult",
    "ExperimentResult",
    "QuantileSummary",
    "AggregatedResult",
    "validate_metadata",
    "validate_result",
    "aggregate_results",
    "aggregate_seeds",
    "aggregate_metric",
]
