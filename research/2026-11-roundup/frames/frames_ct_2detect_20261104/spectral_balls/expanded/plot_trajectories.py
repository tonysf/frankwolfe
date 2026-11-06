"""Plot the completed CT evaluation trajectories against projection work."""

import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter
import numpy as np

from analyze import VARIANTS, collect, slice_id, verify_pair


ROOT = Path(__file__).resolve().parent
LINESTYLES = {
    "sframes": "-",
    "sframes_full": "--",
    "spring_sarah": "-.",
    "prox_sarah": (0, (5, 1, 1, 1)),
    "palm": ":",
}


def evaluation_traces(root, design, budget):
    """Require the frozen evaluation cohort and common initial observations."""
    runs = collect(root)
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
        raise ValueError("Trajectory plots require the complete frozen evaluation cohort")
    for sid, size in product(slices, sizes):
        group = [run for (s, _, n, _), run in runs.items() if (s, n) == (sid, size)]
        for run in group:
            verify_pair(group[0], run)
            trace = run["trace"]
            if trace[0]["forward_views"] != 0 or trace[-1]["forward_views"] < budget:
                raise ValueError("Saved trajectories must cover the full plotting interval")
            if any(row["forward_views"] != row["backward_views"] for row in trace):
                raise ValueError("The per-direction work axis requires equal forward and adjoint counts")
            if any(b["forward_views"] <= a["forward_views"] for a, b in zip(trace, trace[1:])):
                raise ValueError("Interpolation requires strictly increasing projection counts")
            if not np.isclose(trace[0]["raw_loss"], trace[0]["feasible_loss"]):
                raise ValueError("The shared warm start must have equal raw and repaired losses")
    return runs, slices, sizes


def aggregate(runs, slices, size, variant, metric, grid, relative=False):
    """Interpolate each run, then take seed medians followed by slice medians."""
    slice_curves = []
    for sid in slices:
        seed_curves = []
        for (run_slice, _, n, method), run in runs.items():
            if (run_slice, n, method) != (sid, size, variant):
                continue
            trace = run["trace"]
            x = [row["forward_views"] for row in trace]
            y = np.asarray([row[metric] for row in trace])
            if relative:
                if y[0] <= 0:
                    raise ValueError("Relative loss needs a positive initial loss")
                y = y / y[0]
            seed_curves.append(np.interp(grid, x, y))
        slice_curves.append(np.median(seed_curves, axis=0))
    curves = np.asarray(slice_curves)
    return np.median(curves, axis=0), np.quantile(curves, (0.25, 0.75), axis=0)


def work_axis(ax, budget):
    ax.set_xlim(0, budget)
    ax.set_xticks(np.linspace(0, budget, 5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Projected views per direction")
    ax.grid(alpha=0.22, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def draw_curves(ax, runs, slices, size, metric, grid, relative=False):
    for variant, (label, color) in VARIANTS.items():
        center, band = aggregate(runs, slices, size, variant, metric, grid, relative)
        if variant in ("sframes", "sframes_full"):
            ax.fill_between(grid, *band, color=color, alpha=0.11, linewidth=0)
        ax.plot(grid, center, color=color, linestyle=LINESTYLES[variant],
                linewidth=2.0 if variant == "sframes" else 1.55, label=label)


def finish(fig, axes, title, output, stem):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, x=0.08, ha="left", fontsize=15, weight="bold", y=0.99)
    fig.legend(handles, labels, ncol=5, loc="upper left", bbox_to_anchor=(0.073, 0.947),
               frameon=False, handlelength=2.4, columnspacing=1.7, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.1, w_pad=3.0)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=190)
    plt.close(fig)


def plot_trajectories(root, output=None):
    design = json.loads((root / "manifest.json").read_text())["design"]
    budget = design["evaluation_view_budget"]
    runs, slices, sizes = evaluation_traces(root, design, budget)
    if len(sizes) != 2:
        raise ValueError("The report layout requires two training-view counts")
    grid = np.unique([0, budget] + [
        row["forward_views"] for run in runs.values() for row in run["trace"]
        if row["forward_views"] <= budget
    ])
    output = root / "reports" if output is None else output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": "0.55", "axes.linewidth": 0.65,
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.7), squeeze=False)
    for row, size in enumerate(sizes):
        for col, (metric, heading) in enumerate((
            ("raw_loss", "Raw training loss"), ("feasible_loss", "Repaired training loss")
        )):
            ax = axes[row, col]
            draw_curves(ax, runs, slices, size, metric, grid, relative=True)
            work_axis(ax, budget)
            ax.set_title(f"{size:,} training views")
            ax.set_ylabel(f"{heading} / initial loss")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    finish(fig, axes, "Training loss trajectories", output, "trajectory_training")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.7), squeeze=False)
    for col, size in enumerate(sizes):
        for row, metric in enumerate(("validation_loss", "d_distance")):
            ax = axes[row, col]
            draw_curves(ax, runs, slices, size, metric, grid)
            work_axis(ax, budget)
            if row == 0:
                ax.set_title(f"{size:,} training views")
                ax.set_ylabel("Repaired held-out view loss")
                ax.set_yscale("log")
                ax.set_ylim(bottom=0.0095)
                ax.set_yticks([0.01, 0.02, 0.04, 0.08])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylabel(r"Distance to nonnegativity set $D$")
                ax.set_yscale("symlog", linthresh=0.001, linscale=0.3)
                ax.set_ylim(bottom=0)
                ax.set_yticks([0, 0.001, 0.01, 0.1, 1])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    finish(fig, axes, "Held-out loss and feasibility trajectories", output, "trajectory_quality")
    return len(runs), len(slices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    count, slices = plot_trajectories(args.root.resolve(), args.output)
    print(f"Plotted {count} completed evaluation runs across {slices} slices")


if __name__ == "__main__":
    main()
