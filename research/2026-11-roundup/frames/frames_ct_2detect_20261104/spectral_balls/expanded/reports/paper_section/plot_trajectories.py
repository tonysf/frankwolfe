"""Plot complete per-iteration evaluation observations on three work axes."""

import argparse
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter
import numpy as np


OUTPUT = Path(__file__).resolve().parent
EXPANDED = OUTPUT.parents[1]
ROOT = EXPANDED / "joint_tuning"
sys.path.insert(0, str(EXPANDED))
from analyze import VARIANTS, slice_id


LINESTYLES = {
    "sframes": "-",
    "sframes_full": "--",
    "spring_sarah": "-.",
    "prox_sarah": (0, (5, 1, 1, 1)),
    "palm": ":",
}


AXES = (
    ("iteration", "Iterations"),
    ("gradient_calls", "Oracle calls"),
    ("forward_views", "Views processed"),
)
METRICS = (
    ("raw_loss", "Raw training loss\n/ initial loss"),
    ("feasible_loss", "Repaired training loss\n/ initial loss"),
    ("validation_loss", "Repaired held-out loss"),
    ("d_distance", r"Distance to $D$"),
)


def evaluation_traces(root, design, budget):
    """Require the frozen cohort and an observation after every update."""
    analysis_path = root / "analyze.py"
    if not analysis_path.is_file():
        analysis_path = EXPANDED / "analyze.py"
    spec = importlib.util.spec_from_file_location("ct_trajectory_analysis", analysis_path)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    runs = analysis.collect(root)
    slices = sorted(slice_id(value) for value in design["evaluation_slices"])
    seeds = sorted(design["evaluation_seeds"])
    sizes = sorted(design["evaluation_training_views"])
    deterministic = set(design["deterministic_variants"])
    expected = {
        (sid, seed, size, variant)
        for sid, seed, size, variant in product(slices, seeds, sizes, VARIANTS)
        if variant not in deterministic or seed == seeds[0]
    }
    if set(runs) != expected or any(run["status"] != "completed" for run in runs.values()):
        raise ValueError("Per-iteration plots require the complete frozen evaluation cohort")
    for sid, size in product(slices, sizes):
        group = [run for (s, _, n, _), run in runs.items() if (s, n) == (sid, size)]
        for run in group:
            analysis.verify_pair(group[0], run)
            trace = run["trace"]
            if [row["iteration"] for row in trace] != list(range(trace[-1]["iteration"] + 1)):
                raise ValueError(f"Missing per-iteration observations in {run['config']['run_id']}")
            if trace[-1]["forward_views"] < budget:
                raise ValueError("Completed trajectories must reach the data stopping budget")
            if any(row["forward_views"] != row["backward_views"] for row in trace):
                raise ValueError("The processed-view axis requires equal forward and adjoint counts")
            if any(b[axis] <= a[axis] for a, b in zip(trace, trace[1:])
                   for axis, _ in AXES):
                raise ValueError("Recorded iteration and work coordinates must increase strictly")
            if not np.isclose(trace[0]["raw_loss"], trace[0]["feasible_loss"]):
                raise ValueError("The shared feasible start must have equal raw and repaired losses")
            run["trace"] = [row for row in trace
                            if max(row["forward_views"], row["backward_views"]) <= budget]
    return runs, slices, sizes


def observed_values(run, metric):
    values = np.asarray([row[metric] for row in run["trace"]])
    if metric in ("raw_loss", "feasible_loss"):
        if values[0] <= 0:
            raise ValueError("Relative training loss requires a positive initial loss")
        values = values / values[0]
    return values


def summarize(runs, slices, size, variant, metric, axis):
    """Take coordinatewise nested medians at the same observed iteration."""
    group = {key: run for key, run in runs.items()
             if key[2] == size and key[3] == variant}
    length = min(len(run["trace"]) for run in group.values())
    slice_work, slice_values = [], []
    for sid in slices:
        seed_work, seed_values = [], []
        for key, run in group.items():
            if key[0] != sid:
                continue
            seed_work.append([row[axis] for row in run["trace"][:length]])
            seed_values.append(observed_values(run, metric)[:length])
        if not seed_work:
            raise ValueError(f"Missing slice {sid} for {variant}")
        slice_work.append(np.median(seed_work, axis=0))
        slice_values.append(np.median(seed_values, axis=0))
    return (np.median(slice_work, axis=0), np.median(slice_values, axis=0),
            np.quantile(slice_values, (0.25, 0.75), axis=0))


def plot_size(runs, slices, size, budget, output):
    fig, axes = plt.subplots(4, 3, figsize=(6.8, 7.3), sharey="row")
    for row, (metric, ylabel) in enumerate(METRICS):
        for col, (axis, heading) in enumerate(AXES):
            ax = axes[row, col]
            for variant, (label, color) in VARIANTS.items():
                for key, run in runs.items():
                    if key[2] == size and key[3] == variant:
                        ax.plot([point[axis] for point in run["trace"]],
                                observed_values(run, metric), color=color, alpha=0.10,
                                linewidth=0.4, zorder=1)
                x, center, band = summarize(runs, slices, size, variant, metric, axis)
                if variant in ("sframes", "sframes_full"):
                    ax.fill_between(x, *band, color=color, alpha=0.12, linewidth=0)
                ax.plot(x, center, color=color, linestyle=LINESTYLES[variant],
                        linewidth=1.25 if variant == "sframes" else 1.0, label=label)
                ax.plot(x[-1], center[-1], marker="o", markersize=2.3, color=color)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(heading, weight="bold")
            if row == len(METRICS) - 1:
                ax.set_xlabel(heading)
            if axis == "forward_views":
                ax.set_xlim(0, budget)
                ax.set_xticks(np.linspace(0, budget, 5))
            else:
                ax.set_xscale("symlog", linthresh=1, linscale=0.5)
                limit = max(run["trace"][-1][axis] for key, run in runs.items()
                            if key[2] == size)
                ax.set_xlim(0, limit)
                ax.set_xticks([0, 1, 10, 100, limit])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if metric in ("raw_loss", "feasible_loss"):
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            elif metric == "validation_loss":
                ax.set_yscale("log")
                ax.set_yticks([0.01, 0.02, 0.04, 0.08])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_yscale("symlog", linthresh=0.001, linscale=0.3)
                ax.set_ylim(bottom=-0.0001)
                ax.set_yticks([0, 0.001, 0.01, 0.1, 1])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.grid(alpha=0.20, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{size:,} training views", fontsize=10, weight="bold", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.974),
               ncol=5, frameon=False, fontsize=6.8, handlelength=2.0, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=1.0, w_pad=1.0)
    stem = output / f"trajectories_{size}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    design = json.loads((root / "manifest.json").read_text())["design"]
    budget = design["evaluation_view_budget"]
    runs, slices, sizes = evaluation_traces(root, design, budget)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7,
        "axes.edgecolor": "0.55", "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    for size in sizes:
        plot_size(runs, slices, size, budget, output)
    print(f"Plotted {len(runs)} completed per-iteration runs on all three work axes")


if __name__ == "__main__":
    main()
