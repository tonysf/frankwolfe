"""Nonnegative-baseline extension of the frozen group-balanced mean/SD renderer.

This renderer consumes validated, within-budget native trajectories. It changes
only their display; optimizer outputs and the endpoint summaries are untouched.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np


AXES = (("iteration", "Iterations"), ("comparison_oracle_calls", "Oracle calls"),
        ("comparison_views", "Processed views"))
METRICS = (("raw_loss", "Raw training loss\n/ initial loss"),
           ("feasible_loss", "Evaluated training loss\n/ initial loss"),
           ("validation_loss", "Evaluated held-out loss"),
           ("d_distance", "Nonnegativity distance"))


def observed_values(run, metric):
    values = np.asarray([row[metric] for row in run["trace"]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The plotted metric contains nonfinite observations")
    if metric in ("raw_loss", "feasible_loss"):
        if values[0] <= 0:
            raise ValueError("Relative training loss requires positive initial loss")
        values = values / values[0]
    return values


def select_and_weight(runs, slices, groups, size, variant):
    """Give every group equal weight, then every slice and its repetitions."""
    chosen = sorted((key, run) for key, run in runs.items()
                    if key[2] == size and key[3] == variant)
    if not chosen or {key[0] for key, _ in chosen} != set(slices):
        raise ValueError(f"Incomplete plotted cohort for {variant}, N={size}")
    per_slice = Counter(key[0] for key, _ in chosen)
    per_group = Counter(groups[sid] for sid in slices)
    weights = np.asarray([1.0 / (len(per_group) * per_group[groups[key[0]]] * per_slice[key[0]])
                          for key, _ in chosen])
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Plot weights do not sum to one")
    weights /= weights.sum()
    return chosen, weights


def summarize(runs, slices, groups, size, variant, metric, axis):
    """Return exact weighted mean/SD on a fixed cohort's shared work support.

    For work axes, an iterate remains current until the next recorded update.
    A right-continuous step lookup aligns those states without interpolating
    objective values. Truncating at the shortest observed support avoids either
    extrapolation or a changing set of runs near the end of a curve.
    """
    chosen, weights = select_and_weight(runs, slices, groups, size, variant)
    coordinates, observations = [], []
    for key, run in chosen:
        trace = run["trace"]
        iterations = np.asarray([row["iteration"] for row in trace], dtype=np.int64)
        if iterations[0] != 0 or np.any(np.diff(iterations) < 0) or np.any(np.diff(iterations) > 1):
            raise ValueError(f"Missing native accepted iterations: {key}")
        x = np.asarray([row[axis] for row in trace], dtype=np.int64)
        if x[0] != 0 or np.any(np.diff(x) < 0):
            raise ValueError(f"Work is not nondecreasing: {key}, {axis}")
        # Budget-stop records can charge failed trial work without accepting a
        # factor update. At duplicate coordinates the latest recorded state is
        # current; keep all distinct work coordinates, with no interpolation.
        keep = np.r_[np.diff(x) != 0, True]
        coordinates.append(x[keep])
        observations.append(observed_values(run, metric)[keep])
    end = min(int(x[-1]) for x in coordinates)
    grid = np.unique(np.concatenate([x[x <= end] for x in coordinates]))
    aligned = np.stack([values[np.searchsorted(x, grid, side="right") - 1]
                        for values, x in zip(observations, coordinates)])
    mean = np.sum(weights[:, None] * aligned, axis=0)
    sd = np.sqrt(np.sum(weights[:, None] * (aligned - mean) ** 2, axis=0))
    # Do not replace a true zero-variance cohort by numerical roundoff.
    identical = np.all(aligned == aligned[0], axis=0)
    mean[identical], sd[identical] = aligned[0, identical], 0.0
    return grid, mean, sd


def log_work_ticks(limit):
    ticks = [0]
    power = 1
    while power < limit / 3:
        ticks.append(power)
        power *= 10
    if limit not in ticks:
        ticks.append(limit)
    return ticks


def _signed_ticks(low, high, threshold):
    positive = [threshold * 10 ** k for k in range(8)
                if threshold * 10 ** k <= max(abs(low), abs(high)) * 1.1]
    return sorted([-v for v in positive if -v >= low] + [0] + [v for v in positive if v <= high])


def plot_size(analysis, runs, slices, groups, size, budget, original_cap, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {(metric, axis, variant): summarize(runs, slices, groups, size, variant, metric, axis)
                 for metric, _ in METRICS for axis, _ in AXES for variant in analysis.VARIANTS}
    stats = dict(training_views=size, renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        center="Arithmetic mean with equal group, within-group slice, and within-slice repetition weights",
        standard_deviation="Descriptive weighted population SD: sqrt(sum(w * (value - mean)^2)); not standard error",
        alignment="At each work value use the last recorded iterate at or below that value; no linear interpolation",
        support="Fixed cohort, truncated at minimum final native coordinate; duplicate coordinates retain latest state",
        band="Unmodified arithmetic mean minus/plus one standard deviation in the original metric units",
        individual_trajectories_drawn=False, methods={}, row_scales={})
    for variant, (label, color) in analysis.VARIANTS.items():
        chosen, weights = select_and_weight(runs, slices, groups, size, variant)
        stats["methods"][variant] = dict(label=label, color=color, physical_runs=len(chosen),
            runs=[dict(slice_id=key[0], seed=key[1], group=groups[key[0]], weight=float(weight))
                  for (key, _), weight in zip(chosen, weights)], axes={})
        for axis, _ in AXES:
            grid = summaries[(METRICS[0][0], axis, variant)][0]
            metrics = {}
            for metric, _ in METRICS:
                _, mean, sd = summaries[(metric, axis, variant)]
                metrics[metric] = dict(mean=mean.tolist(), standard_deviation=sd.tolist(),
                    lower_min=float(np.min(mean - sd)), upper_max=float(np.max(mean + sd)))
            stats["methods"][variant]["axes"][axis] = dict(grid=grid.tolist(), metrics=metrics)

    fig, axes = plt.subplots(4, 3, figsize=(6.8, 7.3), sharey="row")
    for row, (metric, ylabel) in enumerate(METRICS):
        low = min(float(np.min(center - sd)) for (m, _, _), (_, center, sd) in summaries.items() if m == metric)
        high = max(float(np.max(center + sd)) for (m, _, _), (_, center, sd) in summaries.items() if m == metric)
        # Signed bands are retained exactly; logarithmic axes are used only if
        # all band values are positive. Distance uses symlog to include zero.
        scale = "log" if low > 0 and metric != "d_distance" else "symlog"
        threshold = 0.001 if metric == "d_distance" else (0.01 if metric in ("raw_loss", "feasible_loss") else 0.001)
        stats["row_scales"][metric] = dict(scale=scale, lower_band_min=low, upper_band_max=high,
                                          negative_band_preserved=low < 0,
                                          linthresh=threshold if scale == "symlog" else None)
        for col, (axis, heading) in enumerate(AXES):
            ax = axes[row, col]
            if axis == "comparison_views" and 0 < original_cap < budget:
                ax.axvline(original_cap, color="0.5", linewidth=0.6, alpha=0.5, zorder=0)
            for variant, (label, color) in analysis.VARIANTS.items():
                x, mean, sd = summaries[(metric, axis, variant)]
                step = None if axis == "iteration" else "post"
                ax.fill_between(x, mean - sd, mean + sd, step=step,
                                color=color, alpha=0.13, linewidth=0, zorder=1)
                ax.plot(x, mean, color=color, linestyle="-", linewidth=1.2, label=label,
                        drawstyle="default" if axis == "iteration" else "steps-post", zorder=2)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(heading, weight="bold")
            if row == len(METRICS) - 1:
                ax.set_xlabel(heading)
            if axis == "comparison_views":
                ax.set_xlim(0, budget)
                ax.set_xticks(np.linspace(0, budget, 5))
            else:
                ax.set_xscale("symlog", linthresh=1, linscale=0.5)
                limit = max(summaries[(metric, axis, variant)][0][-1] for variant in analysis.VARIANTS)
                ax.set_xlim(0, limit)
                ax.set_xticks(log_work_ticks(limit))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if scale == "log":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
            else:
                ax.set_yscale("symlog", linthresh=threshold, linscale=0.3)
                lower_limit = -threshold * 0.1 if low >= 0 else low - max(abs(low) * 0.05, threshold * 0.1)
                upper_limit = high + max(abs(high) * 0.05, threshold * 0.1)
                ax.set_ylim(lower_limit, upper_limit)
                ax.set_yticks(_signed_ticks(lower_limit, upper_limit, threshold))
            if metric in ("raw_loss", "feasible_loss"):
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=0.20, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{size:,} training views", fontsize=10, weight="bold", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.974),
               ncol=getattr(analysis, "legend_columns", 4),
               frameon=False, fontsize=6.5, handlelength=1.7, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=1.0, w_pad=1.0)
    stem = output / f"trajectories_{size}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=190)
    plt.close(fig)
    (output / f"statistics_{size}.json").write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")
    return stats
