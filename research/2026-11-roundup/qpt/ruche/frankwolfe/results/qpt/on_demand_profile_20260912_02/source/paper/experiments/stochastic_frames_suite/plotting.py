"""Seed-aggregated plots for stochastic-FRAMES result traces."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .results import AggregatedResult, RunResult, aggregate_results


AXIS_LABELS = {
    "iterations": "Completed iterations",
    "sampled_observations": "Sampled observations",
    "lmo_calls": "LMO calls",
    "optimizer_time": "Optimizer time (seconds)",
}
PER_STEP_TRAJECTORIES = {
    "momentum_weights",
    "smoothing_parameters",
    "step_sizes",
    "estimated_gaps",
}


def _as_groups(results):
    if isinstance(results, AggregatedResult):
        return {(results.problem_name, results.method): results}
    if isinstance(results, RunResult):
        results = [results]
    groups = defaultdict(list)
    for result in results:
        if not isinstance(result, RunResult):
            raise TypeError("plot inputs must be RunResult instances.")
        groups[(result.problem_name, result.method)].append(result)
    if not groups:
        raise ValueError("At least one result is required for plotting.")
    return {
        key: aggregate_results(group, quantiles=(0.25, 0.75))
        for key, group in groups.items()
    }


def _checkpoint_aligned(values, checkpoint_steps):
    values = np.asarray(values)
    n_steps = int(checkpoint_steps[-1])
    # Suite count traces are checkpoint-aligned. Prefer that interpretation
    # when lengths are otherwise ambiguous.
    if values.shape[0] == len(checkpoint_steps):
        return values
    if values.shape[0] == n_steps:
        if values.shape[0] == 0:
            return np.zeros(
                (len(checkpoint_steps),) + values.shape[1:], dtype=values.dtype
            )
        indices = np.minimum(checkpoint_steps, values.shape[0] - 1)
        return values[indices]
    raise ValueError("Trace cannot be aligned to checkpoint steps.")


def _x_values(aggregate, x_axis, target_length):
    checkpoints = aggregate.checkpoint_steps
    n_steps = int(checkpoints[-1])
    if x_axis == "iterations":
        if target_length == len(checkpoints):
            return checkpoints
        if target_length == n_steps:
            return np.arange(1, n_steps + 1)
    elif x_axis in {"sampled_observations", "lmo_calls"}:
        summary = aggregate.metric(x_axis).median
        if target_length == len(checkpoints):
            return _checkpoint_aligned(summary, checkpoints)
        if target_length == n_steps:
            return summary
    elif x_axis == "optimizer_time":
        summary = aggregate.metric("optimizer_time").median
        if target_length == len(checkpoints):
            return summary
    else:
        raise ValueError(
            f"x_axis must be one of {tuple(AXIS_LABELS)}; got {x_axis!r}."
        )
    raise ValueError(
        f"Metric trace of length {target_length} cannot be plotted against "
        f"{x_axis!r}."
    )


def plot_metric(
    results,
    metric="exact_smoothed_gap",
    *,
    x_axis="iterations",
    ax=None,
    log_y=False,
    title=None,
):
    """Plot medians and interquartile bands, grouped by problem/method."""

    groups = _as_groups(results)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.0))
    for (problem, method), aggregate in groups.items():
        summary = aggregate.metric(metric)
        if summary.median.ndim != 1:
            raise ValueError("plot_metric requires a scalar-valued trace.")
        median = summary.median
        lower = summary.lower
        upper = summary.upper
        n_steps = int(aggregate.checkpoint_steps[-1])
        per_step = metric in PER_STEP_TRAJECTORIES
        if x_axis != "iterations" and per_step:
            if n_steps == 0:
                raise ValueError("An empty per-step trace cannot be plotted.")
            indices = np.minimum(aggregate.checkpoint_steps, n_steps - 1)
            median = median[indices]
            lower = lower[indices]
            upper = upper[indices]
        if x_axis == "iterations" and per_step:
            x = np.arange(n_steps)
        else:
            x = _x_values(aggregate, x_axis, len(median))
        label = method if len({key[0] for key in groups}) == 1 else f"{problem}: {method}"
        line = ax.plot(x, median, label=label)[0]
        ax.fill_between(
            x,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
    ax.set_xlabel(AXIS_LABELS[x_axis])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or metric.replace("_", " ").title())
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_common_axes(results, metric="exact_smoothed_gap", *, log_y=False):
    """Create the standard iteration/observation/LMO/time comparison panel."""

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    for axis, x_axis in zip(axes.flat, AXIS_LABELS):
        plot_metric(
            results,
            metric,
            x_axis=x_axis,
            ax=axis,
            log_y=log_y,
            title=f"Against {AXIS_LABELS[x_axis].lower()}",
        )
    return figure, axes


def save_common_plot(
    path,
    results,
    metric="exact_smoothed_gap",
    *,
    log_y=False,
    dpi=150,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_common_axes(results, metric=metric, log_y=log_y)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


__all__ = ["plot_metric", "plot_common_axes", "save_common_plot"]
