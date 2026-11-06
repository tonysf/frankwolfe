"""Summarize saved CT factorization probes without loading data or optimizers."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter


METHODS = {
    "sframes": ("SFRAMES", "#0072B2"),
    "spring_sarah": ("SPRING-SARAH", "#D55E00"),
    "prox_sarah": ("ProxSARAH", "#009E73"),
    "palm": ("PALM", "#AA4499"),
}
BUDGETS = {
    "iteration": "Optimizer iterations",
    "optimizer_seconds": "Optimizer time (s)",
    "projected_views": "Forward + backprojected views",
}
FIELDS = (
    "iteration", "optimizer_seconds", "forward_views", "backward_views", "gradient_calls",
    "raw_loss", "feasible_loss", "validation_loss", "reference_relative_error",
    "max_constraint_violation", "repair_distance",
)
WORK_NOTE = (
    "View counts include optimizer refreshes and recursive gradient differences. "
    "Timing excludes diagnostic probes and checkpoint I/O."
)
QUALITY_NOTE = (
    "Held-out loss and reference error use repaired factors. "
    "Constraint violation and repair distance describe the raw iterate."
)


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def read_runs(root):
    for path in sorted((root / "runs").glob("*/metadata.json")):
        config = read_json(path, {})["config"]
        method = config["method"]
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r} in {path}")
        trace = read_json(path.with_name("trace.json"), [])
        rows = []
        for record in trace:
            row = {key: record[key] for key in FIELDS}
            row["projected_views"] = row["forward_views"] + row["backward_views"]
            for field, value in row.items():
                if field == "reference_relative_error" and value is None:
                    continue
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {field} in {path.parent}")
            if rows and row["iteration"] <= rows[-1]["iteration"]:
                raise ValueError(f"Probe iterations are not increasing in {path.parent}")
            if rows and any(row[k] < rows[-1][k] for k in (
                "optimizer_seconds", "forward_views", "backward_views", "gradient_calls"
            )):
                raise ValueError(f"Optimizer work counts are not cumulative in {path.parent}")
            rows.append(row)
        progress = read_json(path.with_name("progress.json"), {})
        result = read_json(path.with_name("result.json"), {})
        status = progress.get("status", "awaiting probes")
        if result.get("status") == "completed" and rows and result["iteration"] == rows[-1]["iteration"]:
            status = "completed"
        elif status == "completed":
            status = "awaiting matching result"
        yield {
            "run_id": config.get("run_id", path.parent.name),
            "path": str(path.parent),
            "status": status,
            "checkpoint_iteration": progress.get("iteration"),
            "config": config,
            "rows": rows,
        }


def run_summary(run):
    rows = run["rows"]
    result = {key: value for key, value in run.items() if key != "rows"}
    result["probe_count"] = len(rows)
    if rows:
        result.update(
            initial=rows[0],
            last_probe=rows[-1],
            minimum_repaired_training_probe=min(rows, key=lambda row: row["feasible_loss"]),
            max_constraint_violation=max(row["max_constraint_violation"] for row in rows),
            max_repair_distance=max(row["repair_distance"] for row in rows),
        )
    return result


def cohort_summary(runs):
    probed = [run for run in runs if run["rows"]]
    methods = {run["config"]["method"] for run in probed}
    output = {
        "status_counts": dict(Counter(run["status"] for run in runs)),
        "methods_with_probes": sorted(methods),
        "run_ids": [run["run_id"] for run in runs],
    }
    if methods != set(METHODS):
        return output
    matched = {}
    for key in BUDGETS:
        cap = min(run["rows"][-1][key] for run in probed)
        probes = {
            run["run_id"]: next((row for row in reversed(run["rows"]) if row[key] <= cap), None)
            for run in probed
        }
        if all(row is not None for row in probes.values()):
            matched[key] = {"budget": cap, "last_probe_within_budget": probes}
    output["matched_budgets"] = matched
    return output


def plot_panel(ax, runs, xkey, xlabel, series, ylabel):
    plotted = False
    for run in runs:
        color = METHODS[run["config"]["method"]][1]
        for ykey, style in series:
            points = [(row[xkey], row[ykey]) for row in run["rows"] if row[ykey] is not None]
            if points:
                xs, ys = zip(*points)
                ax.plot(xs, ys, color=color, linestyle=style, linewidth=1.5,
                        marker="o" if len(points) == 1 else None, markersize=4)
                plotted = True
    if not plotted:
        ax.text(.5, .5, "Awaiting saved measurements", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.set_major_formatter(EngFormatter(sep="\N{THIN SPACE}"))
    ax.grid(alpha=.2)


def add_legend(fig, training=False):
    handles = [Line2D([0], [0], color=color, label=label) for label, color in METHODS.values()]
    if training:
        handles += [Line2D([0], [0], color=".25", linestyle=style, label=label)
                    for style, label in (("-", "Repaired loss"), ("--", "Raw loss"))]
    fig.legend(handles=handles, ncol=len(handles), loc="upper center", bbox_to_anchor=(.5, .91),
               frameon=False, fontsize=9)


def plot_cohort(pdf, output, case, runs):
    slice_id, n_train = case
    title = f"2DeteCT slice {slice_id} · {n_train:,} training views"
    suffix = f"{slice_id}_{n_train}"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), squeeze=False)
    for ax, (key, label) in zip(axes[0], BUDGETS.items()):
        plot_panel(ax, runs, key, label, [("feasible_loss", "-"), ("raw_loss", "--")], "Training loss")
        if any(row["raw_loss"] > 0 or row["feasible_loss"] > 0 for run in runs for row in run["rows"]):
            ax.set_yscale("log")
    fig.suptitle(title)
    add_legend(fig, training=True)
    fig.text(.05, .035, WORK_NOTE, fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, .82))
    fig.savefig(output / f"{suffix}_training.png", dpi=160)
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, (key, label, xkey) in zip(axes.flat, (
        ("validation_loss", "Held-out view loss", "projected_views"),
        ("reference_relative_error", "Relative reference-image error", "optimizer_seconds"),
        ("max_constraint_violation", "Maximum constraint violation", "iteration"),
        ("repair_distance", "Distance moved by feasibility repair", "iteration"),
    )):
        plot_panel(ax, runs, xkey, BUDGETS[xkey], [(key, "-")], label)
        if xkey == "iteration":
            ax.set_yscale("symlog", linthresh=1e-9)
            ax.set_ylim(bottom=0)
    fig.suptitle(title)
    add_legend(fig)
    fig.text(.05, .025, QUALITY_NOTE, fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .84))
    fig.savefig(output / f"{suffix}_quality.png", dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "reports"
    runs = list(read_runs(root))
    eval_runs = [run for run in runs if run["config"]["phase"] == "evaluation"]
    phase = "evaluation" if eval_runs else "development"
    cohorts = defaultdict(list)
    for run in eval_runs or runs:
        config = run["config"]
        cohorts[config["slice_id"], config["n_train"]].append(run)
    output.mkdir(parents=True, exist_ok=True)
    if cohorts:
        with PdfPages(output / "comparison.pdf") as pdf:
            for case, members in sorted(cohorts.items()):
                plot_cohort(pdf, output, case, members)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plot_phase": phase,
        "work_accounting": WORK_NOTE,
        "quality_accounting": QUALITY_NOTE,
        "runs": [run_summary(run) for run in runs],
        "cohorts": [
            {"slice_id": case[0], "n_train": case[1], **cohort_summary(members)}
            for case, members in sorted(cohorts.items())
        ],
    }
    tmp = output / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    tmp.replace(output / "summary.json")
    print(output.resolve())


if __name__ == "__main__":
    main()
