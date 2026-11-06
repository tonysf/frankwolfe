"""Safe NPZ persistence for :mod:`.results`.

Archives contain only numeric arrays and NumPy Unicode scalars/arrays.  The
loader always disables pickle and validates both the archive envelope and the
reconstructed result before returning it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from .results import (
    OracleCounts,
    RunMetadata,
    RunResult,
    SCHEMA_VERSION,
    _metadata_json,
    validate_metadata,
    validate_result,
)


ARCHIVE_FORMAT = "stochastic_frames_suite.run_result"
ARCHIVE_VERSION = 1


class ResultArchiveError(ValueError):
    """Raised when a result archive is malformed or incompatible."""


_ARRAY_FIELDS = (
    "checkpoint_steps",
    "checkpoint_iterates",
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
    "final_iterate",
    "final_gradient_estimate",
)

_COUNT_FIELDS = OracleCounts.field_names()

_ENVELOPE_KEYS = {
    "archive_format",
    "archive_version",
    "meta_schema_version",
    "meta_problem_name",
    "meta_method",
    "meta_problem_seed",
    "meta_initialization_seed",
    "meta_sampling_seed",
    "meta_batch_size",
    "meta_n_steps",
    "meta_has_reference_beta",
    "meta_reference_beta",
    "meta_extra_json",
    "problem_metric_names",
}


def _effective_metadata(
    base: RunMetadata, additional: Optional[Any]
) -> RunMetadata:
    if additional is None:
        return base
    if isinstance(additional, RunMetadata):
        core_fields = (
            "problem_name",
            "n_steps",
            "problem_seed",
            "initialization_seed",
            "sampling_seed",
            "batch_size",
            "method",
            "schema_version",
            "reference_beta",
        )
        mismatches = [
            name
            for name in core_fields
            if getattr(additional, name) != getattr(base, name)
        ]
        if mismatches:
            raise ValueError(
                "additional metadata conflicts with the result in: "
                + ", ".join(mismatches)
            )
        extra_values = dict(additional.extra)
    elif isinstance(additional, Mapping):
        extra_values = dict(additional)
        core = base.to_dict(flatten_extra=False)
        for name in tuple(extra_values):
            if name not in core or name == "extra":
                continue
            wanted = extra_values.pop(name)
            if core[name] != wanted:
                raise ValueError(
                    f"additional metadata {name!r} conflicts with the result."
                )
        nested = extra_values.pop("extra", {})
        if not isinstance(nested, Mapping):
            raise TypeError("additional metadata 'extra' value must be a mapping.")
        overlap = set(nested).intersection(extra_values)
        if overlap:
            raise ValueError(
                "duplicate additional metadata keys: " + ", ".join(sorted(overlap))
            )
        extra_values = {**nested, **extra_values}
    else:
        raise TypeError("additional metadata must be a mapping or RunMetadata.")

    merged = dict(base.extra)
    for name, value in extra_values.items():
        if name in merged and merged[name] != value:
            raise ValueError(f"additional metadata would overwrite {name!r}.")
        merged[name] = value
    return RunMetadata(
        problem_name=base.problem_name,
        n_steps=base.n_steps,
        problem_seed=base.problem_seed,
        initialization_seed=base.initialization_seed,
        sampling_seed=base.sampling_seed,
        batch_size=base.batch_size,
        method=base.method,
        schema_version=base.schema_version,
        reference_beta=base.reference_beta,
        extra=merged,
    )


def result_to_payload(
    result: RunResult, metadata: Optional[Any] = None
) -> Dict[str, np.ndarray]:
    """Convert a result to a validated, pickle-free archive payload."""

    normalized = validate_result(result)
    run_metadata = _effective_metadata(normalized.metadata, metadata)
    metric_names = tuple(sorted(normalized.problem_metrics))
    metric_name_dtype = max(1, max((len(name) for name in metric_names), default=1))

    payload: Dict[str, np.ndarray] = {
        "archive_format": np.asarray(ARCHIVE_FORMAT),
        "archive_version": np.asarray(ARCHIVE_VERSION, dtype=np.int64),
        "meta_schema_version": np.asarray(
            run_metadata.schema_version, dtype=np.int64
        ),
        "meta_problem_name": np.asarray(run_metadata.problem_name),
        "meta_method": np.asarray(run_metadata.method),
        "meta_problem_seed": np.asarray(run_metadata.problem_seed, dtype=np.int64),
        "meta_initialization_seed": np.asarray(
            run_metadata.initialization_seed, dtype=np.int64
        ),
        "meta_sampling_seed": np.asarray(
            run_metadata.sampling_seed, dtype=np.int64
        ),
        "meta_batch_size": np.asarray(run_metadata.batch_size, dtype=np.int64),
        "meta_n_steps": np.asarray(run_metadata.n_steps, dtype=np.int64),
        "meta_has_reference_beta": np.asarray(
            run_metadata.reference_beta is not None, dtype=np.bool_
        ),
        "meta_reference_beta": np.asarray(
            np.nan
            if run_metadata.reference_beta is None
            else run_metadata.reference_beta,
            dtype=float,
        ),
        "meta_extra_json": np.asarray(_metadata_json(run_metadata.extra)),
        "problem_metric_names": np.asarray(
            metric_names, dtype=f"<U{metric_name_dtype}"
        ),
    }
    for name in _ARRAY_FIELDS:
        payload[name] = np.asarray(getattr(normalized, name))
    for name in _COUNT_FIELDS:
        payload[f"count_{name}"] = np.asarray(
            getattr(normalized.oracle_counts, name), dtype=np.int64
        )
    for index, name in enumerate(metric_names):
        payload[f"problem_metric_{index:04d}"] = np.asarray(
            normalized.problem_metrics[name]
        )

    for key, value in payload.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"archive payload {key!r} is not a NumPy array.")
        if value.dtype.hasobject:
            raise TypeError(
                f"archive payload {key!r} has object dtype; pickle is forbidden."
            )
        if key not in _ENVELOPE_KEYS and key != "problem_metric_names":
            if value.dtype.kind not in "biufc":
                raise TypeError(f"archive data {key!r} must be numeric.")
    return payload


def save_result(
    path: Union[str, os.PathLike],
    result: RunResult,
    metadata: Optional[Any] = None,
) -> Path:
    """Atomically save ``result`` as a compressed, pickle-free NPZ archive."""

    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result_to_payload(result, metadata=metadata)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary_name, "wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def _archive_arrays(path: Union[str, os.PathLike]) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {}
            for key in archive.files:
                try:
                    value = archive[key]
                except ValueError as error:
                    raise ResultArchiveError(
                        f"archive field {key!r} cannot be loaded without pickle."
                    ) from error
                if value.dtype.hasobject:
                    raise ResultArchiveError(
                        f"archive field {key!r} has forbidden object dtype."
                    )
                arrays[key] = np.array(value, copy=True)
            return arrays
    except ResultArchiveError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise ResultArchiveError(f"could not read result archive: {error}") from error


def _scalar(
    arrays: Mapping[str, np.ndarray],
    key: str,
    *,
    kind: Optional[str] = None,
) -> Any:
    try:
        value = arrays[key]
    except KeyError as error:
        raise ResultArchiveError(f"archive is missing required field {key!r}.") from error
    if value.shape != ():
        raise ResultArchiveError(f"archive field {key!r} must be scalar.")
    if kind is not None and value.dtype.kind not in kind:
        raise ResultArchiveError(
            f"archive field {key!r} has invalid dtype {value.dtype}."
        )
    return value.item()


def _decode_metadata(arrays: Mapping[str, np.ndarray]) -> RunMetadata:
    archive_format = _scalar(arrays, "archive_format", kind="US")
    if archive_format != ARCHIVE_FORMAT:
        raise ResultArchiveError(
            f"unknown archive format {archive_format!r}; expected {ARCHIVE_FORMAT!r}."
        )
    archive_version = _scalar(arrays, "archive_version", kind="iu")
    if archive_version != ARCHIVE_VERSION:
        raise ResultArchiveError(
            f"unsupported archive version {archive_version}; "
            f"this code supports version {ARCHIVE_VERSION}."
        )
    schema_version = _scalar(arrays, "meta_schema_version", kind="iu")
    if schema_version != SCHEMA_VERSION:
        raise ResultArchiveError(
            f"unsupported result schema version {schema_version}; "
            f"this code supports version {SCHEMA_VERSION}."
        )

    has_reference_beta = _scalar(
        arrays, "meta_has_reference_beta", kind="b"
    )
    reference_beta_value = _scalar(arrays, "meta_reference_beta", kind="f")
    if not isinstance(has_reference_beta, (bool, np.bool_)):
        raise ResultArchiveError("meta_has_reference_beta must be boolean.")
    if has_reference_beta:
        reference_beta: Optional[float] = float(reference_beta_value)
    else:
        if not np.isnan(reference_beta_value):
            raise ResultArchiveError(
                "meta_reference_beta must be NaN when no reference beta is present."
            )
        reference_beta = None

    extra_text = _scalar(arrays, "meta_extra_json", kind="US")
    try:
        extra = json.loads(extra_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ResultArchiveError("meta_extra_json is not valid JSON.") from error
    if not isinstance(extra, dict):
        raise ResultArchiveError("meta_extra_json must encode a mapping.")
    try:
        return RunMetadata(
            problem_name=_scalar(arrays, "meta_problem_name", kind="US"),
            n_steps=_scalar(arrays, "meta_n_steps", kind="iu"),
            problem_seed=_scalar(arrays, "meta_problem_seed", kind="iu"),
            initialization_seed=_scalar(
                arrays, "meta_initialization_seed", kind="iu"
            ),
            sampling_seed=_scalar(arrays, "meta_sampling_seed", kind="iu"),
            batch_size=_scalar(arrays, "meta_batch_size", kind="iu"),
            method=_scalar(arrays, "meta_method", kind="US"),
            schema_version=schema_version,
            reference_beta=reference_beta,
            extra=extra,
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveError(f"invalid run metadata: {error}") from error


def _expected_keys(metric_count: int) -> set:
    return (
        set(_ENVELOPE_KEYS)
        | set(_ARRAY_FIELDS)
        | {f"count_{name}" for name in _COUNT_FIELDS}
        | {f"problem_metric_{index:04d}" for index in range(metric_count)}
    )


def payload_to_result(
    arrays: Mapping[str, np.ndarray],
    expected_metadata: Optional[Any] = None,
) -> RunResult:
    """Validate an archive payload and reconstruct its :class:`RunResult`."""

    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping.")
    for key, value in arrays.items():
        if not isinstance(key, str):
            raise ResultArchiveError("archive field names must be strings.")
        if not isinstance(value, np.ndarray):
            raise ResultArchiveError(f"archive field {key!r} is not an array.")
        if value.dtype.hasobject:
            raise ResultArchiveError(
                f"archive field {key!r} has forbidden object dtype."
            )

    metadata = _decode_metadata(arrays)
    try:
        names_array = arrays["problem_metric_names"]
    except KeyError as error:
        raise ResultArchiveError(
            "archive is missing required field 'problem_metric_names'."
        ) from error
    if names_array.ndim != 1 or names_array.dtype.kind not in "US":
        raise ResultArchiveError(
            "problem_metric_names must be a one-dimensional Unicode array."
        )
    metric_names = tuple(str(name) for name in names_array.tolist())
    if any(not name for name in metric_names) or len(set(metric_names)) != len(
        metric_names
    ):
        raise ResultArchiveError(
            "problem_metric_names must contain unique, non-empty names."
        )
    if metric_names != tuple(sorted(metric_names)):
        raise ResultArchiveError("problem_metric_names must be sorted.")

    expected_keys = _expected_keys(len(metric_names))
    missing = sorted(expected_keys.difference(arrays))
    unknown = sorted(set(arrays).difference(expected_keys))
    if missing:
        raise ResultArchiveError("archive is missing fields: " + ", ".join(missing))
    if unknown:
        raise ResultArchiveError("archive contains unknown fields: " + ", ".join(unknown))

    numeric_keys = (
        set(_ARRAY_FIELDS)
        | {f"count_{name}" for name in _COUNT_FIELDS}
        | {f"problem_metric_{index:04d}" for index in range(len(metric_names))}
    )
    for key in numeric_keys:
        if arrays[key].dtype.kind not in "biufc":
            raise ResultArchiveError(f"archive field {key!r} must be numeric.")

    problem_metrics = {
        name: arrays[f"problem_metric_{index:04d}"]
        for index, name in enumerate(metric_names)
    }
    counts = OracleCounts(
        *(arrays[f"count_{name}"] for name in _COUNT_FIELDS)
    )
    try:
        result = RunResult(
            metadata=metadata,
            checkpoint_steps=arrays["checkpoint_steps"],
            checkpoint_iterates=arrays["checkpoint_iterates"],
            momentum_weights=arrays["momentum_weights"],
            smoothing_parameters=arrays["smoothing_parameters"],
            step_sizes=arrays["step_sizes"],
            estimated_gaps=arrays["estimated_gaps"],
            task_loss=arrays["task_loss"],
            composite_objective=arrays["composite_objective"],
            smoothed_objective=arrays["smoothed_objective"],
            exact_smoothed_gap=arrays["exact_smoothed_gap"],
            reference_objective=arrays["reference_objective"],
            reference_gap=arrays["reference_gap"],
            estimator_error=arrays["estimator_error"],
            feasibility_or_regularizer=arrays["feasibility_or_regularizer"],
            oracle_counts=counts,
            optimizer_time=arrays["optimizer_time"],
            problem_metrics=problem_metrics,
            final_iterate=arrays["final_iterate"],
            final_gradient_estimate=arrays["final_gradient_estimate"],
        )
    except (TypeError, ValueError) as error:
        raise ResultArchiveError(f"invalid result data: {error}") from error
    validate_metadata(result.metadata, expected_metadata)
    return result


def load_result(
    path: Union[str, os.PathLike],
    *,
    expected_metadata: Optional[Any] = None,
) -> RunResult:
    """Load and fully validate a result archive with pickle disabled."""

    return payload_to_result(
        _archive_arrays(path), expected_metadata=expected_metadata
    )


def load_metadata(
    path: Union[str, os.PathLike],
    *,
    expected_metadata: Optional[Any] = None,
) -> RunMetadata:
    """Load archive metadata after validating the complete archive."""

    return load_result(path, expected_metadata=expected_metadata).metadata


def validate_archive(
    path: Union[str, os.PathLike],
    *,
    expected_metadata: Optional[Any] = None,
) -> RunMetadata:
    """Validate an archive and return its metadata."""

    return load_metadata(path, expected_metadata=expected_metadata)


# Explicit aliases make call sites self-documenting and preserve a compact API.
save_run_result = save_result
load_run_result = load_result


__all__ = [
    "ARCHIVE_FORMAT",
    "ARCHIVE_VERSION",
    "ResultArchiveError",
    "result_to_payload",
    "payload_to_result",
    "save_result",
    "load_result",
    "save_run_result",
    "load_run_result",
    "load_metadata",
    "validate_archive",
]
