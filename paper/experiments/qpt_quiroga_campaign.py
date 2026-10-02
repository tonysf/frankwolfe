"""Reproducible comparisons on the Quiroga-style QPT sensing design.

This deliberately small CPU runner compares three optimizers on exactly the
same rows, observations and initial factor:

* ``printed-adafgd`` loops the adaptive update printed by Quiroga and
  Kyrillidis;
* ``armijo-fgd`` uses the same factor direction with an Armijo line search on
  the summed penalized objective; and
* ``frames`` calls the structured stochastic-FRAMES adapter.

The common objective used for reporting is

    0.5 * sum_s (A(U U^H)_s - f_s)^2 + lambda * ||Tr_out(U U^H) - I||_F^2.

FRAMES internally uses a mean measurement loss, so its constant smoothing
parameter is fixed to ``beta = m / (2 lambda)``.  Multiplying its smoothed
objective by ``m`` then gives the objective above.  Result archives contain no
pickled objects and are published without overwriting existing paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import scipy

from .qpt_quiroga_frames import (
    quiroga_adafgd_step,
    run_quiroga_sensing_stochastic_frames,
)
from .qpt_quiroga_sensing import (
    QuirogaSensingData,
    generate_quiroga_sensing_data,
    trace_preserving_loss_and_gradient,
    trace_preserving_residual,
)
from .quantum_process_tomography import (
    PowerSchedule,
    checkpoint_iterations,
    make_factor_initial_point,
    pack_factor,
    unpack_factor,
)


SCHEMA_VERSION = 1
FORMAT = "qpt_quiroga_campaign"
METHODS = ("printed-adafgd", "armijo-fgd", "frames")
COMMON_METRICS = (
    "measurement_loss_sum",
    "measurement_loss_mean",
    "rmse",
    "tp_violation",
    "normalized_tp_violation",
    "penalized_objective",
    "process_fidelity",
)


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, str(array.dtype))).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _finite_positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive finite number.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _canonical_all_rows(data, rows):
    return (rows.size == data.m
            and np.array_equal(rows, np.arange(data.m, dtype=np.int64)))


def _active_rows_and_targets(data, rows):
    if not isinstance(data, QuirogaSensingData):
        raise TypeError("data must be QuirogaSensingData.")
    if rows is None:
        rows = np.arange(data.m, dtype=np.int64)
    else:
        rows = data.design.validate_rows(rows)
    targets = (data.all_observations() if _canonical_all_rows(data, rows)
               else data.observations_for_rows(rows))
    return rows, np.asarray(targets, dtype=np.float64)


def evaluate_common_metrics(data, factor, rows, targets, tp_weight):
    """Evaluate method-independent metrics on one factor and active row set."""
    factor = np.asarray(factor, dtype=np.complex128)
    predicted = (data.design.full_values(factor) if _canonical_all_rows(data, rows)
                 else data.design.values(factor, rows))
    residual = predicted - targets
    measurement_sum = 0.5 * float(np.dot(residual, residual))
    measurement_mean = measurement_sum / rows.size
    tp_residual = trace_preserving_residual(factor)
    tp_violation = float(np.linalg.norm(tp_residual, ord="fro"))
    values = {
        "measurement_loss_sum": measurement_sum,
        "measurement_loss_mean": measurement_mean,
        "rmse": math.sqrt(2.0 * measurement_mean),
        "tp_violation": tp_violation,
        "normalized_tp_violation": tp_violation / math.sqrt(data.d),
        "penalized_objective": measurement_sum + tp_weight * tp_violation**2,
        "process_fidelity": float(data.fidelity(factor)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("A common checkpoint metric is nonfinite.")
    return values


@dataclass
class MethodTrace:
    name: str
    status: str
    failure_reason: str
    final_factor: np.ndarray
    checkpoint_steps: np.ndarray
    optimizer_seconds: np.ndarray
    cumulative_data_row_accesses: np.ndarray
    measurement_loss_sum: np.ndarray
    measurement_loss_mean: np.ndarray
    rmse: np.ndarray
    tp_violation: np.ndarray
    normalized_tp_violation: np.ndarray
    penalized_objective: np.ndarray
    process_fidelity: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def endpoint(self):
        result = {
            "status": self.status,
            "failure_reason": self.failure_reason or None,
            "completed_steps": int(self.checkpoint_steps[-1]),
            "optimizer_seconds": float(self.optimizer_seconds[-1]),
            "cumulative_data_row_accesses": int(self.cumulative_data_row_accesses[-1]),
        }
        for name in COMMON_METRICS:
            value = float(getattr(self, name)[-1])
            result[name] = value if math.isfinite(value) else None
        totals = (
            ("spectral_norm_seconds", "spectral_norm_seconds_total", float),
            ("spectral_norm_operator_calls", "spectral_norm_operator_calls_total", int),
            ("spectral_norm_adjoint_column_equivalent",
             "spectral_norm_adjoint_column_equivalent_total", int),
        )
        for source, destination, conversion in totals:
            if source in self.diagnostics:
                result[destination] = conversion(np.sum(self.diagnostics[source]))
        return result


def _make_trace(name, status, failure_reason, factor, steps, times, work, metric_rows, diagnostics=None):
    return MethodTrace(
        name=name,
        status=status,
        failure_reason=failure_reason,
        final_factor=np.asarray(factor, dtype=np.complex128).copy(),
        checkpoint_steps=np.asarray(steps, dtype=np.int64),
        optimizer_seconds=np.asarray(times, dtype=np.float64),
        cumulative_data_row_accesses=np.asarray(work, dtype=np.int64),
        diagnostics={} if diagnostics is None else {
            key: np.asarray(value) for key, value in diagnostics.items()
        },
        **{
            name_: np.asarray([row[name_] for row in metric_rows], dtype=np.float64)
            for name_ in COMMON_METRICS
        },
    )


def _append_checkpoint(data, factor, rows, targets, tp_weight, step, elapsed, work,
                       steps, times, work_rows, metric_rows):
    metrics = evaluate_common_metrics(data, factor, rows, targets, tp_weight)
    steps.append(int(step))
    times.append(float(elapsed))
    work_rows.append(int(work))
    metric_rows.append(metrics)

def _try_append_checkpoint(data, factor, rows, targets, tp_weight, step, elapsed, work,
                           steps, times, work_rows, metric_rows):
    try:
        _append_checkpoint(data, factor, rows, targets, tp_weight, step, elapsed, work,
                           steps, times, work_rows, metric_rows)
    except Exception as error:
        if steps and steps[-1] == step:
            times[-1], work_rows[-1] = float(elapsed), int(work)
        else:
            steps.append(int(step))
            times.append(float(elapsed))
            work_rows.append(int(work))
            metric_rows.append({name: np.nan for name in COMMON_METRICS})
        return f"checkpoint metrics failed ({type(error).__name__}: {error})"
    return ""


def _update_terminal_attempt(step, elapsed, work, steps, times, work_rows):
    if steps and steps[-1] == step:
        times[-1] = float(elapsed)
        work_rows[-1] = int(work)



def run_printed_adafgd(
    data,
    factor0,
    rows,
    targets,
    *,
    n_steps,
    metrics_frequency,
    eta_scale,
    tp_weight,
    spectral_norm_method="auto",
    dense_max_process_dimension=1024,
    spectral_norm_tolerance=1e-10,
    spectral_norm_maxiter=None,
):
    """Loop the literal printed-rule update, retaining numerical failures."""
    n_steps = _positive_integer(n_steps, "n_steps")
    requested = set(checkpoint_iterations(n_steps, metrics_frequency).tolist())
    factor = np.asarray(factor0, dtype=np.complex128).copy()
    steps, times, work_rows, metric_rows = [], [], [], []
    initial_error = _try_append_checkpoint(
        data, factor, rows, targets, tp_weight, 0, 0.0, 0,
        steps, times, work_rows, metric_rows)
    if initial_error:
        return _make_trace(
            "printed-adafgd", "failed", initial_error, factor,
            steps, times, work_rows, metric_rows, diagnostics={})
    adaptive_steps, numerators, denominators = [], [], []
    spectral_seconds, spectral_calls, spectral_columns = [], [], []
    elapsed = 0.0
    measurement_rows = 0
    status, failure_reason = "complete", ""
    methods_used = []

    for step in range(1, n_steps + 1):
        began = perf_counter()
        measurement_rows += rows.size
        try:
            following, info = quiroga_adafgd_step(
                data,
                factor,
                eta_scale=eta_scale,
                tp_weight=tp_weight,
                rows=rows,
                observations=targets,
                spectral_norm_method=spectral_norm_method,
                dense_max_process_dimension=dense_max_process_dimension,
                spectral_norm_tolerance=spectral_norm_tolerance,
                spectral_norm_maxiter=spectral_norm_maxiter,
            )
            if not np.all(np.isfinite(following)):
                raise FloatingPointError("The printed adaFGD update is nonfinite.")
        except Exception as error:  # retain the failed trajectory as evidence
            elapsed += perf_counter() - began
            status = "failed"
            failure_reason = f"{type(error).__name__}: {error}"
            break
        elapsed += perf_counter() - began
        factor = following
        adaptive_steps.append(info["eta"])
        numerators.append(info["spectral_norm_numerator"])
        denominators.append(info["prediction_norm_denominator"])
        methods_used.append(info["spectral_norm_method"])
        spectral_seconds.append(info["spectral_norm_seconds"])
        spectral_calls.append(info["spectral_norm_operator_calls"])
        spectral_columns.append(info["spectral_norm_adjoint_column_equivalent"])
        if step in requested:
            metric_error = _try_append_checkpoint(
                data, factor, rows, targets, tp_weight, step, elapsed,
                measurement_rows, steps, times, work_rows, metric_rows)
            if metric_error:
                status, failure_reason = "failed", metric_error
                break

    completed = len(adaptive_steps)
    if steps[-1] != completed:
        metric_error = _try_append_checkpoint(
            data, factor, rows, targets, tp_weight, completed, elapsed,
            measurement_rows, steps, times, work_rows, metric_rows)
        if metric_error:
            status, failure_reason = "failed", metric_error
    else:
        _update_terminal_attempt(
            completed, elapsed, measurement_rows, steps, times, work_rows)
    return _make_trace(
        "printed-adafgd", status, failure_reason, factor, steps, times, work_rows,
        metric_rows,
        diagnostics={
            "step_size": np.asarray(adaptive_steps, dtype=np.float64),
            "spectral_norm_numerator": np.asarray(numerators, dtype=np.float64),
            "prediction_norm_denominator": np.asarray(denominators, dtype=np.float64),
            "spectral_norm_method": np.asarray(methods_used, dtype="U16"),
            "spectral_norm_seconds": np.asarray(spectral_seconds, dtype=np.float64),
            "spectral_norm_operator_calls": np.asarray(spectral_calls, dtype=np.int64),
            "spectral_norm_adjoint_column_equivalent": np.asarray(
                spectral_columns, dtype=np.int64),
        },
    )


def _summed_objective_and_half_gradient(data, factor, rows, targets, tp_weight):
    if _canonical_all_rows(data, rows):
        measurement_loss, measurement_gradient = data.design.full_loss_and_gradient(
            factor, targets, reduction="sum")
    else:
        measurement_loss, measurement_gradient = data.design.loss_and_gradient(
            factor, rows, targets, reduction="sum")
    tp_half_loss, tp_half_gradient = trace_preserving_loss_and_gradient(factor)
    # Each returned real-Frobenius gradient is twice the matrix-space direction
    # used in the printed update.  For H=||R||^2, grad(H)/2 is tp_half_gradient.
    direction = 0.5 * measurement_gradient + tp_weight * tp_half_gradient
    objective = float(measurement_loss + 2.0 * tp_weight * tp_half_loss)
    return objective, np.asarray(direction)


def _summed_objective(data, factor, rows, targets, tp_weight):
    predicted = (data.design.full_values(factor) if _canonical_all_rows(data, rows)
                 else data.design.values(factor, rows))
    residual = predicted - targets
    tp_residual = trace_preserving_residual(factor)
    return float(0.5 * np.dot(residual, residual) + tp_weight * np.vdot(tp_residual, tp_residual).real)


def run_armijo_fgd(
    data,
    factor0,
    rows,
    targets,
    *,
    n_steps,
    metrics_frequency,
    tp_weight,
    initial_step=1.0,
    shrink=0.5,
    armijo_constant=1e-4,
    max_backtracks=30,
):
    """Factor GD with Armijo backtracking on the shared summed objective.

    ``direction`` is one half of the real-Frobenius gradient, matching the
    direction in the printed update.  Therefore the exact descent slope in
    the Armijo test is ``-2 * ||direction||_F**2``.
    """
    n_steps = _positive_integer(n_steps, "n_steps")
    initial_step = _finite_positive(initial_step, "initial_step")
    if not 0.0 < shrink < 1.0:
        raise ValueError("shrink must lie strictly between zero and one.")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie strictly between zero and one.")
    if isinstance(max_backtracks, (bool, np.bool_)) or not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer.")

    requested = set(checkpoint_iterations(n_steps, metrics_frequency).tolist())
    factor = np.asarray(factor0, dtype=np.complex128).copy()
    steps, times, work_rows, metric_rows = [], [], [], []
    initial_error = _try_append_checkpoint(
        data, factor, rows, targets, tp_weight, 0, 0.0, 0,
        steps, times, work_rows, metric_rows)
    if initial_error:
        return _make_trace(
            "armijo-fgd", "failed", initial_error, factor,
            steps, times, work_rows, metric_rows, diagnostics={})
    accepted_steps, backtracks, objective_evaluations = [], [], []
    elapsed = 0.0
    measurement_rows = 0
    status, failure_reason = "complete", ""

    for step in range(1, n_steps + 1):
        began = perf_counter()
        try:
            objective, direction = _summed_objective_and_half_gradient(
                data, factor, rows, targets, tp_weight)
            measurement_rows += rows.size
            slope_magnitude = 2.0 * float(np.vdot(direction, direction).real)
            if not math.isfinite(objective) or not math.isfinite(slope_magnitude):
                raise FloatingPointError("The Armijo objective or slope is nonfinite.")
            if slope_magnitude == 0.0:
                status = "converged"
                elapsed += perf_counter() - began
                break
            candidate = None
            trial_step = initial_step
            trials = 0
            for backtrack in range(int(max_backtracks) + 1):
                trial = factor - trial_step * direction
                trial_objective = _summed_objective(data, trial, rows, targets, tp_weight)
                measurement_rows += rows.size
                trials += 1
                if (math.isfinite(trial_objective)
                        and trial_objective <= objective - armijo_constant * trial_step * slope_magnitude):
                    candidate = trial
                    break
                trial_step *= shrink
            if candidate is None:
                raise RuntimeError(
                    f"Armijo line search failed after {max_backtracks + 1} trials."
                )
        except Exception as error:  # retain the last finite point
            elapsed += perf_counter() - began
            status = "failed"
            failure_reason = f"{type(error).__name__}: {error}"
            break
        elapsed += perf_counter() - began
        factor = candidate
        accepted_steps.append(trial_step)
        backtracks.append(backtrack)
        objective_evaluations.append(trials)
        if step in requested:
            metric_error = _try_append_checkpoint(
                data, factor, rows, targets, tp_weight, step, elapsed,
                measurement_rows, steps, times, work_rows, metric_rows)
            if metric_error:
                status, failure_reason = "failed", metric_error
                break

    completed = len(accepted_steps)
    if steps[-1] != completed:
        metric_error = _try_append_checkpoint(
            data, factor, rows, targets, tp_weight, completed, elapsed,
            measurement_rows, steps, times, work_rows, metric_rows)
        if metric_error:
            status, failure_reason = "failed", metric_error
    else:
        _update_terminal_attempt(
            completed, elapsed, measurement_rows, steps, times, work_rows)
    return _make_trace(
        "armijo-fgd", status, failure_reason, factor, steps, times, work_rows,
        metric_rows,
        diagnostics={
            "step_size": np.asarray(accepted_steps, dtype=np.float64),
            "backtracks": np.asarray(backtracks, dtype=np.int64),
            "objective_evaluations": np.asarray(objective_evaluations, dtype=np.int64),
        },
    )


def run_frames(
    data,
    factor0,
    rows,
    targets,
    *,
    n_steps,
    metrics_frequency,
    rank,
    tau,
    batch_size,
    sampling_seed,
    tp_weight,
    rho_schedule,
    step_size_schedule,
    show_progress=False,
):
    """Run FRAMES with the smoothing value exactly matched to ``tp_weight``."""
    beta = rows.size / (2.0 * tp_weight)
    result = run_quiroga_sensing_stochastic_frames(
        data,
        n_steps=n_steps,
        rank=rank,
        tau=tau,
        batch_size=batch_size,
        sampling_seed=sampling_seed,
        x0=pack_factor(factor0),
        beta0=beta,
        rho_schedule=rho_schedule,
        smoothing_schedule=beta,
        step_size_schedule=step_size_schedule,
        metrics_frequency=metrics_frequency,
        show_progress=show_progress,
        rows=rows,
        observations=targets,
    )
    measurement_sum = rows.size * result.measurement_loss
    metrics = []
    for index in range(result.checkpoint_steps.size):
        tp = float(result.tp_violation[index])
        mean_loss = float(result.measurement_loss[index])
        metrics.append({
            "measurement_loss_sum": float(measurement_sum[index]),
            "measurement_loss_mean": mean_loss,
            "rmse": math.sqrt(2.0 * mean_loss),
            "tp_violation": tp,
            "normalized_tp_violation": tp / math.sqrt(data.d),
            "penalized_objective": float(measurement_sum[index] + tp_weight * tp**2),
            "process_fidelity": float(result.process_fidelity[index]),
        })
    work = result.checkpoint_steps.astype(np.int64) * int(batch_size)
    return _make_trace(
        "frames", "complete", "", result.final_factor,
        result.checkpoint_steps, result.optimizer_seconds, work, metrics,
        diagnostics={
            "exact_smoothed_gap": result.exact_smoothed_gap,
            "checkpoint_smoothing_parameter": result.checkpoint_smoothing_parameters,
            "estimated_gap": result.estimated_gaps,
            "momentum_weight": result.momentum_weights,
            "smoothing_parameter": result.smoothing_parameters,
            "step_size": result.step_sizes,
        },
    )


def _schedule_metadata(schedule):
    if isinstance(schedule, PowerSchedule):
        return {"type": "PowerSchedule", **vars(schedule)}
    if np.isscalar(schedule):
        return {"type": "constant", "value": float(schedule)}
    return {"type": "callable", "module": getattr(schedule, "__module__", None),
            "qualname": getattr(schedule, "__qualname__", type(schedule).__name__)}

def run_campaign(
    data,
    *,
    methods=METHODS,
    rows=None,
    rank=1,
    initialization_seed=0,
    sampling_seed=0,
    n_steps=400,
    frames_steps=None,
    metrics_frequency=20,
    frames_metrics_frequency=None,
    eta_scale,
    tp_weight,
    tau=None,
    batch_size=32,
    spectral_norm_method="auto",
    dense_max_process_dimension=1024,
    spectral_norm_tolerance=1e-10,
    spectral_norm_maxiter=None,
    armijo_initial_step=1.0,
    armijo_shrink=0.5,
    armijo_constant=1e-4,
    armijo_max_backtracks=30,
    rho_schedule=None,
    step_size_schedule=None,
    show_progress=False,
):
    """Run selected methods and return ``(metadata, payload, traces)``."""
    if data.truth_factor is None or data.truth_factor.ndim != 2 or data.truth_factor.shape[1] != 1:
        raise ValueError("Quiroga campaigns require a rank-one truth factor for fidelity metrics.")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(name not in METHODS for name in methods):
        raise ValueError(f"methods must be unique names from {METHODS}.")
    rank = _positive_integer(rank, "rank")
    n_steps = _positive_integer(n_steps, "n_steps")
    frames_steps = n_steps if frames_steps is None else _positive_integer(frames_steps, "frames_steps")
    metrics_frequency = _positive_integer(metrics_frequency, "metrics_frequency")
    frames_metrics_frequency = (metrics_frequency if frames_metrics_frequency is None
                                else _positive_integer(frames_metrics_frequency, "frames_metrics_frequency"))
    eta_scale = _finite_positive(eta_scale, "eta_scale")
    tp_weight = _finite_positive(tp_weight, "tp_weight")
    batch_size = _positive_integer(batch_size, "batch_size")
    active_rows, targets = _active_rows_and_targets(data, rows)
    x0 = make_factor_initial_point(data, rank=rank, seed=initialization_seed)
    factor0 = unpack_factor(x0, data.process_dimension, rank)
    tau = math.sqrt(data.d) if tau is None else _finite_positive(tau, "tau")
    if np.linalg.norm(factor0, ord=2) > tau + 1e-10 and "frames" in methods:
        raise ValueError("tau is smaller than the shared initial factor's operator norm.")

    if rho_schedule is None:
        rho_schedule = PowerSchedule(2.0, 4.0, 0.6, cap=1.0)
    if step_size_schedule is None:
        step_size_schedule = PowerSchedule(2.0, 2.0, 1.0, cap=1.0)

    traces = {}
    for method in methods:
        if method == "printed-adafgd":
            trace = run_printed_adafgd(
                data, factor0, active_rows, targets, n_steps=n_steps,
                metrics_frequency=metrics_frequency, eta_scale=eta_scale,
                tp_weight=tp_weight, spectral_norm_method=spectral_norm_method,
                dense_max_process_dimension=dense_max_process_dimension,
                spectral_norm_tolerance=spectral_norm_tolerance,
                spectral_norm_maxiter=spectral_norm_maxiter,
            )
        elif method == "armijo-fgd":
            trace = run_armijo_fgd(
                data, factor0, active_rows, targets, n_steps=n_steps,
                metrics_frequency=metrics_frequency, tp_weight=tp_weight,
                initial_step=armijo_initial_step, shrink=armijo_shrink,
                armijo_constant=armijo_constant,
                max_backtracks=armijo_max_backtracks,
            )
        else:
            try:
                trace = run_frames(
                    data, factor0, active_rows, targets, n_steps=frames_steps,
                    metrics_frequency=frames_metrics_frequency, rank=rank, tau=tau,
                    batch_size=batch_size, sampling_seed=sampling_seed,
                    tp_weight=tp_weight, rho_schedule=rho_schedule,
                    step_size_schedule=step_size_schedule,
                    show_progress=show_progress,
                )
            except Exception as error:
                metric = evaluate_common_metrics(
                    data, factor0, active_rows, targets, tp_weight)
                trace = _make_trace(
                    "frames", "failed", f"{type(error).__name__}: {error}",
                    factor0, [0], [0.0], [0], [metric], diagnostics={})
        traces[method] = trace

    configuration = {
        "methods": list(methods), "n_steps": n_steps, "frames_steps": frames_steps,
        "metrics_frequency": metrics_frequency,
        "frames_metrics_frequency": frames_metrics_frequency,
        "rank": rank, "initialization_seed": int(initialization_seed),
        "sampling_seed": int(sampling_seed), "eta_scale": eta_scale,
        "tp_weight_lambda": tp_weight,
        "frames_beta": active_rows.size / (2.0 * tp_weight),
        "tau": tau, "batch_size": batch_size,
        "spectral_norm_method": spectral_norm_method,
        "dense_max_process_dimension": int(dense_max_process_dimension),
        "spectral_norm_tolerance": float(spectral_norm_tolerance),
        "spectral_norm_maxiter": spectral_norm_maxiter,
        "armijo_initial_step": float(armijo_initial_step),
        "armijo_shrink": float(armijo_shrink),
        "armijo_constant": float(armijo_constant),
        "armijo_max_backtracks": int(armijo_max_backtracks),
        "rho_schedule": _schedule_metadata(rho_schedule),
        "step_size_schedule": _schedule_metadata(step_size_schedule),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": data.design.describe(),
        "data_metadata": data.metadata,
        "observation_mode": data.observation_mode,
        "active_row_count": int(active_rows.size),
        "active_rows_sha256": _array_sha256(active_rows),
        "active_observations_sha256": _array_sha256(targets),
        "initial_factor_sha256": _array_sha256(factor0),
        "truth_factor_sha256": None if data.truth_factor is None else _array_sha256(data.truth_factor),
        "configuration": configuration,
        "objective": "0.5*sum(residual^2)+lambda*||Tr_out(UU^H)-I||_F^2",
        "armijo_slope": "-2*||half_gradient_direction||_F^2",
        "method_domains": {
            "frames": {"constraint": "operator_norm_ball", "tau": tau},
            "printed-adafgd": {"constraint": "none", "projection": "none"},
            "armijo-fgd": {"constraint": "none", "projection": "none"},
        },
        "work_counter": (
            "nominal optimizer active-row accesses; not a total compute/work-equivalent; "
            "excludes checkpoint passes, TP work, full structured forward/adjoint overhead, "
            "and separately reported adaFGD spectral-norm operations"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "methods": {name: trace.endpoint() for name, trace in traces.items()},
        "scipy_version": scipy.__version__,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("qpt_quiroga_campaign.py", "qpt_quiroga_frames.py", "qpt_quiroga_sensing.py")
        },
    }
    payload = {
        "active_rows": active_rows,
        "active_observations": targets,
        "initial_factor": factor0,
        "truth_factor": (np.empty((0, 0), dtype=np.complex128)
                         if data.truth_factor is None else data.truth_factor),
    }
    for name, trace in traces.items():
        prefix = name.replace("-", "_")
        for field_name in (
            "final_factor", "checkpoint_steps", "optimizer_seconds",
            "cumulative_data_row_accesses", *COMMON_METRICS,
        ):
            payload[f"{prefix}_{field_name}"] = np.asarray(getattr(trace, field_name))
        for diagnostic, values in trace.diagnostics.items():
            payload[f"{prefix}_diagnostic_{diagnostic}"] = np.asarray(values)
    return metadata, payload, traces


def _temporary_path(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    return Path(name)


def save_campaign(npz_path, metadata_path, metadata, payload):
    """Publish individually atomic, no-clobber pickle-free NPZ and JSON files."""
    npz_path, metadata_path = Path(npz_path), Path(metadata_path)
    if npz_path == metadata_path:
        raise ValueError("The NPZ and JSON paths must differ.")
    for path in (npz_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    archive_payload = {key: np.asarray(value) for key, value in payload.items()}
    archive_payload["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    )
    if any(value.dtype.hasobject for value in archive_payload.values()):
        raise TypeError("Object arrays are forbidden in campaign archives.")

    npz_temp, json_temp = _temporary_path(npz_path), _temporary_path(metadata_path)
    published_npz = False
    try:
        with npz_temp.open("wb") as handle:
            np.savez_compressed(handle, **archive_payload)
            handle.flush()
            os.fsync(handle.fileno())
        with json_temp.open("w", encoding="utf-8") as handle:
            handle.write(metadata_text)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails rather than overwriting.
        os.link(npz_temp, npz_path)
        published_npz = True
        try:
            os.link(json_temp, metadata_path)
        except BaseException:
            npz_path.unlink()
            published_npz = False
            raise
    finally:
        for temporary in (npz_temp, json_temp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        if published_npz and not metadata_path.exists():
            try:
                npz_path.unlink()
            except FileNotFoundError:
                pass
    return npz_path, metadata_path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--steps", type=int, default=400,
                        help="Full-batch steps for printed adaFGD and Armijo FGD.")
    parser.add_argument("--frames-steps", type=int, default=None)
    parser.add_argument("--metrics-every", type=int, default=20)
    parser.add_argument("--frames-metrics-every", type=int, default=None)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--subset-size", type=int, default=0,
                        help="Zero uses all rows; otherwise select a fixed subset without replacement.")
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--channel-seed", type=int, default=0)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--observation-mode", choices=("noiseless", "gaussian", "shots"), default="noiseless")
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--shot-seed", type=int, default=0)
    parser.add_argument("--eta-scale", type=float, required=True)
    parser.add_argument("--tp-weight", type=float, required=True,
                        help="Lambda in the shared summed penalized objective.")
    parser.add_argument("--tau", type=float, default=None,
                        help="FRAMES factor operator-norm radius; default sqrt(d).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--spectral-norm-method", choices=("auto", "dense", "matrix-free"), default="auto")
    parser.add_argument("--dense-max-process-dimension", type=int, default=1024)
    parser.add_argument("--spectral-norm-tolerance", type=float, default=1e-10)
    parser.add_argument("--spectral-norm-maxiter", type=int, default=None)
    parser.add_argument("--armijo-initial-step", type=float, default=1.0)
    parser.add_argument("--armijo-shrink", type=float, default=0.5)
    parser.add_argument("--armijo-constant", type=float, default=1e-4)
    parser.add_argument("--armijo-max-backtracks", type=int, default=30)
    parser.add_argument("--rho-scale", type=float, default=2.0)
    parser.add_argument("--rho-offset", type=float, default=4.0)
    parser.add_argument("--rho-exponent", type=float, default=0.6)
    parser.add_argument("--step-scale", type=float, default=2.0)
    parser.add_argument("--step-offset", type=float, default=2.0)
    parser.add_argument("--step-exponent", type=float, default=1.0)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None,
                        help="JSON path; defaults to --save with a .json suffix.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fail-on-method-failure", action="store_true",
                        help="After saving artifacts, exit nonzero if any method failed.")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    metadata_path = args.metadata if args.metadata is not None else args.save.with_suffix(".json")
    if args.save.exists() or metadata_path.exists():
        parser.error("refusing to overwrite an existing NPZ or JSON result")
    try:
        data = generate_quiroga_sensing_data(
            args.n_qubits,
            channel_seed=args.channel_seed,
            observation_mode=args.observation_mode,
            noise_std=args.noise_std,
            noise_seed=args.noise_seed,
            shots=args.shots,
            shot_seed=args.shot_seed,
        )
        if args.subset_size < 0:
            raise ValueError("subset_size must be nonnegative.")
        rows = (None if args.subset_size == 0
                else data.fixed_row_subset(args.subset_size, args.subset_seed))
        rho = PowerSchedule(args.rho_scale, args.rho_offset, args.rho_exponent, cap=1.0)
        step = PowerSchedule(args.step_scale, args.step_offset, args.step_exponent, cap=1.0)
        metadata, payload, traces = run_campaign(
            data,
            methods=args.methods,
            rows=rows,
            rank=args.rank,
            initialization_seed=args.initialization_seed,
            sampling_seed=args.sampling_seed,
            n_steps=args.steps,
            frames_steps=args.frames_steps,
            metrics_frequency=args.metrics_every,
            frames_metrics_frequency=args.frames_metrics_every,
            eta_scale=args.eta_scale,
            tp_weight=args.tp_weight,
            tau=args.tau,
            batch_size=args.batch_size,
            spectral_norm_method=args.spectral_norm_method,
            dense_max_process_dimension=args.dense_max_process_dimension,
            spectral_norm_tolerance=args.spectral_norm_tolerance,
            spectral_norm_maxiter=args.spectral_norm_maxiter,
            armijo_initial_step=args.armijo_initial_step,
            armijo_shrink=args.armijo_shrink,
            armijo_constant=args.armijo_constant,
            armijo_max_backtracks=args.armijo_max_backtracks,
            rho_schedule=rho,
            step_size_schedule=step,
            show_progress=not args.quiet,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    metadata["configuration"].update({
        "subset_size": int(args.subset_size), "subset_seed": int(args.subset_seed),
        "channel_seed": int(args.channel_seed), "noise_seed": int(args.noise_seed),
        "shot_seed": int(args.shot_seed), "shots": args.shots,
        "observation_mode": args.observation_mode, "noise_std": float(args.noise_std),
    })
    save_campaign(args.save, metadata_path, metadata, payload)
    summary = {name: trace.endpoint() for name, trace in traces.items()}
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), flush=True)
        print(f"Saved {args.save} and {metadata_path}", flush=True)
    if args.fail_on_method_failure and any(trace.status == "failed" for trace in traces.values()):
        raise SystemExit(2)
    return metadata, payload, traces


if __name__ == "__main__":
    main()
