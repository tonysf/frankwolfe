"""Experimental exact-ray FGD for Quiroga's fixed-penalty objective.

This is NOT the published adaFGD step. It keeps its half-gradient direction
and fixed TP penalty, but minimizes the quartic objective along the positive
factor ray. Roots are computed in a unit-Frobenius direction to avoid tiny
gradient powers and cancellation from polarizing with a tiny direction.
There is no Armijo fallback, step floor, projection, or penalty continuation.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from .qpt_quiroga_sensing import (
    choi_output_partial_trace,
    trace_preserving_loss_and_gradient,
    trace_preserving_residual,
)
from .quantum_process_tomography import checkpoint_iterations


METHOD = "exact-linesearch-fgd"


def positive_stationary_points(coefficients):
    """Positive real roots of a quartic derivative, in increasing order.

    ``coefficients`` are ascending powers. Nearly real numerical roots use a
    relative imaginary tolerance of 1e-8. Every returned root must still pass
    direct residual-objective evaluation; roots alone never certify descent.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (5,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("quartic coefficients must be five finite real numbers.")
    derivative = coefficients[1:] * np.arange(1.0, 5.0)
    nonzero = np.flatnonzero(derivative)
    if not nonzero.size:
        return np.empty(0, dtype=np.float64)
    derivative = derivative[:nonzero[-1] + 1]
    if derivative.size == 1:
        return np.empty(0, dtype=np.float64)
    roots = np.roots((derivative / np.max(np.abs(derivative)))[::-1])
    positive = [float(root.real) for root in roots
                if np.isfinite(root) and root.real > 0
                and abs(root.imag) <= 1e-8 * max(1.0, abs(root.real))]
    return np.unique(np.asarray(positive, dtype=np.float64))


def factor_ray_coefficients(data, factor, ray, rows, targets, tp_weight,
                            *, predicted=None, forward=None):
    """Coefficients of the exact objective at ``factor - distance * ray``.

    A provided ``predicted`` is reused; two more forward evaluations suffice
    by polarization. ``forward`` optionally counts these actual evaluations.
    No process-space matrix or dense sensing tensor is formed.
    """
    if forward is None:
        forward = lambda value: data.design.full_values(value)[rows]
    predicted = forward(factor) if predicted is None else np.asarray(predicted)
    b = forward(ray)
    a = forward(factor + ray) - predicted - b
    residual = predicted - targets
    current_tp = trace_preserving_residual(factor)
    ray_tp = choi_output_partial_trace(ray)
    d = data.d
    left = factor.reshape(d, -1)
    right = ray.reshape(d, -1)
    cross_tp = left @ right.conj().T + right @ left.conj().T
    inner = lambda left_, right_: float(np.vdot(left_, right_).real)
    return np.asarray([
        0.5 * inner(residual, residual) + tp_weight * inner(current_tp, current_tp),
        -inner(residual, a) - 2.0 * tp_weight * inner(current_tp, cross_tp),
        0.5 * inner(a, a) + inner(residual, b)
        + tp_weight * (inner(cross_tp, cross_tp) + 2.0 * inner(current_tp, ray_tp)),
        -inner(a, b) - 2.0 * tp_weight * inner(cross_tp, ray_tp),
        0.5 * inner(b, b) + tp_weight * inner(ray_tp, ray_tp),
    ], dtype=np.float64)


def run_exact_linesearch_fgd(
    data, factor0, rows, targets, *, n_steps, metrics_frequency, tp_weight,
    gradient_atol=1e-12, gradient_rtol=1e-10,
):
    """Positive-ray quartic line search with a true full-gradient stop.

    One nominal direction pass comprises current predictions and adjoint
    application. Polynomial construction costs two additional forward passes;
    each directly evaluated candidate costs one more. Completed-update pass
    arrays and explicit terminal-attempt counts reconstruct every checkpoint's
    row charge, including failed or stationary attempts. Checkpoint metrics,
    TP contractions and full-design overhead are excluded as in the campaign.
    """
    # Campaign dispatch imports this module locally; importing its archive
    # helpers here therefore does not create an import-time cycle.
    from .qpt_quiroga_campaign import (
        _finite_nonnegative, _finite_positive, _make_trace, _positive_integer,
        _try_append_checkpoint, _update_terminal_attempt,
    )

    n_steps = _positive_integer(n_steps, "n_steps")
    metrics_frequency = _positive_integer(metrics_frequency, "metrics_frequency")
    tp_weight = _finite_positive(tp_weight, "tp_weight")
    gradient_atol = _finite_nonnegative(gradient_atol, "gradient_atol")
    gradient_rtol = _finite_nonnegative(gradient_rtol, "gradient_rtol")
    rows = data.design.validate_rows(rows)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != rows.shape or not np.all(np.isfinite(targets)):
        raise ValueError("targets must be finite and match rows.")
    factor = np.asarray(factor0, dtype=np.complex128).copy()
    requested = set(checkpoint_iterations(n_steps, metrics_frequency).tolist())
    steps, times, work_rows, metrics = [], [], [], []
    initial_error = _try_append_checkpoint(
        data, factor, rows, targets, tp_weight, 0, 0.0, 0,
        steps, times, work_rows, metrics)
    if initial_error:
        return _make_trace(METHOD, "failed", initial_error, factor,
                           steps, times, work_rows, metrics)
    etas, distances, gradient_norms, candidate_counts = [], [], [], []
    directions, forwards, trials = [], [], []
    terminal = dict(direction_passes=0, polynomial_forward_passes=0,
                    objective_evaluations=0)
    elapsed, work = 0.0, 0
    initial_gradient_norm = None
    termination = {}
    status, failure_reason = "complete", ""

    for step in range(1, n_steps + 1):
        began = perf_counter()
        counts = dict(direction_passes=0, polynomial_forward_passes=0,
                      objective_evaluations=0)

        def forward(value, kind):
            nonlocal work
            counts[kind] += 1
            work += rows.size
            return data.design.full_values(value)[rows]

        try:
            predicted = forward(factor, "direction_passes")
            residual = predicted - targets
            weights = np.bincount(rows, weights=residual, minlength=data.m)
            tp_half_loss, tp_term = trace_preserving_loss_and_gradient(factor)
            direction = data.design.full_adjoint(factor, weights) + tp_weight * tp_term
            objective = float(0.5 * np.dot(residual, residual) + 2.0 * tp_weight * tp_half_loss)
            direction_norm = float(np.linalg.norm(direction, ord="fro"))
            gradient_norm = 2.0 * direction_norm
            if initial_gradient_norm is None:
                initial_gradient_norm = gradient_norm
            threshold = gradient_atol + gradient_rtol * initial_gradient_norm
            if not all(math.isfinite(x) for x in (objective, gradient_norm, threshold)):
                raise FloatingPointError("Exact-ray objective, gradient or threshold is nonfinite.")
            if gradient_norm <= threshold:
                status = "converged"
                termination = {
                    "termination_reason": ["full_gradient_tolerance"],
                    "termination_full_gradient_norm": [gradient_norm],
                    "termination_gradient_threshold": [threshold],
                    "initial_full_gradient_norm": [initial_gradient_norm],
                }
                terminal = counts
                elapsed += perf_counter() - began
                break
            ray = direction / direction_norm
            coefficients = factor_ray_coefficients(
                data, factor, ray, rows, targets, tp_weight, predicted=predicted,
                forward=lambda value: forward(value, "polynomial_forward_passes"))
            roots = positive_stationary_points(coefficients)
            best_factor, best_distance, best_objective = None, None, objective
            for distance in roots:
                candidate = factor - distance * ray
                if not np.all(np.isfinite(candidate)):
                    continue
                candidate_residual = forward(candidate, "objective_evaluations") - targets
                candidate_tp = trace_preserving_residual(candidate)
                candidate_objective = float(
                    0.5 * np.dot(candidate_residual, candidate_residual)
                    + tp_weight * np.vdot(candidate_tp, candidate_tp).real)
                if math.isfinite(candidate_objective) and candidate_objective < best_objective:
                    best_factor, best_distance, best_objective = candidate, distance, candidate_objective
            if best_factor is None:
                raise RuntimeError(
                    "No positive quartic stationary point gave a strict direct-objective decrease; "
                    "no fallback or zero-step success was used.")
            eta = float(best_distance / direction_norm)
            if not math.isfinite(eta):
                raise FloatingPointError("Exact-ray half-gradient step multiplier is nonfinite.")
        except Exception as error:
            status, failure_reason = "failed", f"{type(error).__name__}: {error}"
            terminal = counts
            elapsed += perf_counter() - began
            break
        elapsed += perf_counter() - began
        factor = best_factor
        etas.append(eta)
        distances.append(best_distance)
        gradient_norms.append(gradient_norm)
        candidate_counts.append(roots.size)
        directions.append(counts["direction_passes"])
        forwards.append(counts["polynomial_forward_passes"])
        trials.append(counts["objective_evaluations"])
        if step in requested:
            error = _try_append_checkpoint(
                data, factor, rows, targets, tp_weight, step, elapsed, work,
                steps, times, work_rows, metrics)
            if error:
                status, failure_reason = "failed", error
                break
    completed = len(etas)
    if steps[-1] != completed:
        error = _try_append_checkpoint(
            data, factor, rows, targets, tp_weight, completed, elapsed, work,
            steps, times, work_rows, metrics)
        if error:
            status, failure_reason = "failed", error
    else:
        _update_terminal_attempt(completed, elapsed, work, steps, times, work_rows)
    return _make_trace(
        METHOD, status, failure_reason, factor, steps, times, work_rows, metrics,
        diagnostics={
            "step_size": np.asarray(etas, dtype=np.float64),
            "line_search_distance": np.asarray(distances, dtype=np.float64),
            "full_gradient_norm": np.asarray(gradient_norms, dtype=np.float64),
            "positive_stationary_points": np.asarray(candidate_counts, dtype=np.int64),
            "direction_passes": np.asarray(directions, dtype=np.int64),
            "polynomial_forward_passes": np.asarray(forwards, dtype=np.int64),
            "objective_evaluations": np.asarray(trials, dtype=np.int64),
            "terminal_attempt_rows": np.asarray([rows.size * sum(terminal.values())], dtype=np.int64),
            **{f"terminal_{key}": np.asarray([value], dtype=np.int64)
               for key, value in terminal.items()},
            **termination,
        })
