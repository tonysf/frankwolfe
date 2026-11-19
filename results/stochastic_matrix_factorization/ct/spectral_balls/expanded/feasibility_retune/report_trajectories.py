"""Render the two native-trajectory figures with color identifying each method."""

import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter
import numpy as np


AXES = (("iteration", "Iterations"), ("gradient_calls", "Oracle calls"),
        ("forward_views", "Views processed"))
METRICS = (("raw_loss", "Raw training loss\n/ initial loss"),
           ("feasible_loss", "Repaired training loss\n/ initial loss"),
           ("validation_loss", "Repaired held-out loss"), ("d_distance", r"Distance to $D$"))


def load_analysis(root):
    spec = importlib.util.spec_from_file_location("expanded_ct_plot_analysis", root / "analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    return analysis


def evaluation_traces(root, design, budget):
    analysis = load_analysis(root)
    runs = analysis.collect(root)
    slices = sorted(analysis.slice_id(value) for value in design["evaluation_slices"])
    seeds = sorted(design["evaluation_seeds"])
    sizes = sorted(design["evaluation_training_views"])
    deterministic = set(design["deterministic_variants"])
    expected = {(sid, seed, size, variant)
                for sid, seed, size, variant in product(slices, seeds, sizes, analysis.VARIANTS)
                if variant not in deterministic or seed == seeds[0]}
    if set(runs) != expected or any(run["status"] != "completed" for run in runs.values()):
        raise ValueError("Native trajectory figures require the complete frozen evaluation cohort")
    for sid, size in product(slices, sizes):
        peers = [run for (s, _, n, _), run in runs.items() if (s, n) == (sid, size)]
        for run in peers:
            analysis.verify_pair(peers[0], run)
            trace = run["trace"]
            if trace[-1]["forward_views"] < budget:
                raise ValueError("A completed evaluation run stopped before the work cap")
            if not np.isclose(trace[0]["raw_loss"], trace[0]["feasible_loss"]):
                raise ValueError("The shared feasible start must have equal raw and repaired losses")
            run["trace"] = [row for row in trace if max(row["forward_views"], row["backward_views"]) <= budget]
    return analysis, runs, slices, sizes


def observed_values(run, metric):
    values = np.asarray([row[metric] for row in run["trace"]])
    if metric in ("raw_loss", "feasible_loss"):
        if values[0] <= 0:
            raise ValueError("Relative training loss requires positive initial loss")
        values = values / values[0]
    return values


def summarize(runs, slices, groups, size, variant, metric, axis):
    """Use integer-iteration alignment and nested seed/slice/group medians."""
    chosen = {key: run for key, run in runs.items() if key[2] == size and key[3] == variant}
    length = min(len(run["trace"]) for run in chosen.values())
    per_group_work, per_group_values = {}, {}
    for sid in slices:
        peers = [run for key, run in chosen.items() if key[0] == sid]
        if not peers:
            raise ValueError(f"Missing {variant} for {sid}")
        work = np.median([[row[axis] for row in run["trace"][:length]] for run in peers], axis=0)
        values = np.median([observed_values(run, metric)[:length] for run in peers], axis=0)
        per_group_work.setdefault(groups[sid], []).append(work)
        per_group_values.setdefault(groups[sid], []).append(values)
    group_work = [np.median(values, axis=0) for _, values in sorted(per_group_work.items())]
    group_values = [np.median(values, axis=0) for _, values in sorted(per_group_values.items())]
    return (np.median(group_work, axis=0), np.median(group_values, axis=0),
            np.quantile(group_values, (0.25, 0.75), axis=0))


def log_work_ticks(limit):
    ticks = [0]
    power = 1
    while power < limit / 1.7:
        ticks.append(power)
        power *= 10
    if limit not in ticks:
        ticks.append(limit)
    return ticks


def plot_size(analysis, runs, slices, groups, size, budget, original_cap, output):
    fig, axes = plt.subplots(4, 3, figsize=(6.8, 7.3), sharey="row")
    for row, (metric, ylabel) in enumerate(METRICS):
        for col, (axis, heading) in enumerate(AXES):
            ax = axes[row, col]
            if axis == "forward_views" and 0 < original_cap < budget:
                ax.axvline(original_cap, color="0.5", linewidth=0.6, alpha=0.5, zorder=0)
            for variant, (label, color) in analysis.VARIANTS.items():
                for key, run in runs.items():
                    if key[2] == size and key[3] == variant:
                        ax.plot([point[axis] for point in run["trace"]], observed_values(run, metric),
                                color=color, linestyle="-", alpha=0.065, linewidth=0.35, zorder=1)
                x, center, band = summarize(runs, slices, groups, size, variant, metric, axis)
                if variant in ("sframes", "sframes_full", "tuned_sframes"):
                    ax.fill_between(x, *band, color=color, alpha=0.12, linewidth=0)
                ax.plot(x, center, color=color, linestyle="-", linewidth=1.2, label=label)
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
                limit = max(run["trace"][-1][axis] for key, run in runs.items() if key[2] == size)
                ax.set_xlim(0, limit)
                ax.set_xticks(log_work_ticks(limit))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if metric in ("raw_loss", "feasible_loss"):
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}%"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif metric == "validation_loss":
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.974), ncol=6,
               frameon=False, fontsize=6.5, handlelength=1.7, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.945), h_pad=1.0, w_pad=1.0)
    stem = output / f"trajectories_{size}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=190)
    plt.close(fig)


def add_joint_traces(runs, cases, groups, budget):
    expected = {(sid, seed, n) for sid, seed, n, variant in runs if variant == "sframes"}
    seen = set()
    for case in cases:
        key = (case["slice_id"], case["seed"], case["n_train"])
        if key not in expected or key in seen or groups[key[0]] != case["group"]:
            raise ValueError(f"Unexpected comparison case in the trajectory overlay: {key}")
        seen.add(key)
        record = case["runs"]["tuned_sframes"]
        raw = (Path(record["source_directory"]) / "trace.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["trace_sha256"]:
            raise ValueError(f"Tuned trace changed after analysis: {key}")
        trace = json.loads(raw)
        if (trace[0]["iteration"] != 0 or
                any(row["iteration"] != k for k, row in enumerate(trace)) or
                trace[-1]["forward_views"] != budget or trace[-1]["backward_views"] != budget):
            raise ValueError(f"Tuned run is not a complete native trajectory: {key}")
        runs[(*key, "tuned_sframes")] = {"trace": trace}
    if seen != expected:
        raise ValueError("Missing paired trajectories for the joint tune")


def render(root, cases, output):
    design = json.loads((root / "manifest.json").read_text())["design"]
    budget = design["evaluation_view_budget"]
    analysis, runs, slices, sizes = evaluation_traces(root, design, budget)
    groups = analysis.group_map(design)
    if cases is not None:
        add_joint_traces(runs, cases, groups, budget)
        variants = analysis.VARIANTS
        analysis.VARIANTS = {
            "sframes": ("SFRAMES (orig.)", variants["sframes"][1]),
            "tuned_sframes": ("SFRAMES (joint)", "#333333"),
            "sframes_full": ("Full-batch SF", variants["sframes_full"][1]),
            **{key: variants[key] for key in ("spring_sarah", "prox_sarah", "palm")},
        }
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
                        "xtick.labelsize": 6.5, "ytick.labelsize": 7,
                        "axes.edgecolor": "0.55", "axes.linewidth": 0.6, "pdf.fonttype": 42})
    for size in sizes:
        plot_size(analysis, runs, slices, groups, size, budget,
                  design.get("original_evaluation_view_budget", 17280), output)
    return len(runs), sum(len(run["trace"]) for run in runs.values())
