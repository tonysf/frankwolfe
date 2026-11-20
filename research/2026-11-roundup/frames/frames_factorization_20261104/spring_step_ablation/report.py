"""Compare the paired SPRING-SARAH step-size runs at their saved probes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter


HERE = Path(__file__).resolve().parent
ARMS = {
    "theorem": ("Corollary 3.3 parameters", "#0072B2"),
    "local_curvature": ("Practical local-curvature adaptation", "#D55E00"),
}
DATASETS = {"orl64": "ORL", "yaleb32": "Extended Yale-B"}
BUDGETS = {
    "iteration": "Optimizer iterations",
    "optimizer_seconds": "Optimizer time (s)",
    "component_gradient_calls": "Image-component gradient evaluations",
}
VIOLATIONS = (
    "U_spectral_violation", "V_spectral_violation", "U_negative_norm",
    "V_l1_violation", "max_constraint_violation",
)
POINT_FIELDS = (*BUDGETS, "raw_loss", "feasible_loss", "feasible", "repair_distance",
                *VIOLATIONS, "step_u", "step_v", "max_prox_distance_bound",
                "prox_numerical_floor_count", "cumulative_prox_iterations")
TIMING_NOTE = (
    "Optimizer time includes estimator, step-size computation, and projection work. "
    "The strict and practical runs were timed under different shared-host loads."
)
ORACLE_NOTE = (
    "Counts include both evaluations per SARAH difference and all image components per full refresh. "
    "Curvature estimation is timed, with no image-component gradient evaluations charged for it."
)


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def load_runs(manifest):
    project = manifest.parent.parent
    entries = json.loads(manifest.read_text())["runs"]
    datasets = {}
    for entry in entries:
        arm, seed, dataset = (entry[key] for key in ("step_rule", "seed", "dataset"))
        if arm not in ARMS:
            raise ValueError(f"Unknown step-size arm {arm!r}")
        path = project / entry["path"]
        result = read_json(path / "result.json", {})
        progress = read_json(path / "progress.json", {})
        metadata = read_json(path / "metadata.json", {})
        rows = read_json(path / "trace.json", [])
        config = metadata.get("config", entry.get("config", {}))
        if config and (config["dataset"] != dataset or config["seed"] != seed
                       or config.get("step_rule", config.get("baseline_mode")) != arm):
            raise ValueError(f"Manifest and run configuration disagree for {path}")
        probes = {row["iteration"]: row for row in rows}
        if list(probes) != sorted(probes) or len(probes) != len(rows):
            raise ValueError(f"Expected unique increasing probe iterations in {path}")
        target = config.get("iterations", entry.get("iterations"))
        status = progress.get("status", "awaiting probes" if metadata else "pending")
        if result.get("status") == "completed" and target in probes and result["iteration"] == target:
            status = "completed"
        elif status == "completed":
            status = "awaiting result"
        pair = datasets.setdefault(dataset, {}).setdefault(seed, {})
        if arm in pair:
            raise ValueError(f"Duplicate manifest entry for {dataset}, seed {seed}, {arm}")
        pair[arm] = {"path": str(path), "status": status, "probes": probes,
                     "config": config, "data_sha256": metadata.get("data_sha256"),
                     "target_iterations": target}
    return datasets


def point(row):
    return {name: row[name] for name in POINT_FIELDS if name in row}


def run_summary(run):
    summary = {key: run[key] for key in ("path", "status", "target_iterations", "data_sha256")}
    rows = list(run["probes"].values())
    summary["probe_count"] = len(rows)
    if rows:
        summary.update(
            initial=point(rows[0]), latest_probe=point(rows[-1]),
            feasible_probe_count=sum(row["feasible"] for row in rows),
            max_at_probes={name: max(row[name] for row in rows)
                           for name in (*VIOLATIONS, "repair_distance")},
            max_absolute_loss_change_from_repair=max(
                abs(row["raw_loss"] - row["feasible_loss"]) for row in rows),
        )
    return summary


def paired_summary(pair):
    summary = {"runs": {arm: run_summary(run) for arm, run in pair.items()}}
    if not all(arm in pair and pair[arm]["probes"] for arm in ARMS):
        return summary
    strict, practical = (pair[arm] for arm in ARMS)
    shared_fields = ("batch", "RU", "RV", "l1_radius", "rank", "iterations")
    if strict["data_sha256"] != practical["data_sha256"] or any(
        strict["config"].get(name) != practical["config"].get(name) for name in shared_fields
    ):
        raise ValueError("Paired runs disagree on the data, feasible geometry, batch size or horizon")
    common = strict["probes"].keys() & practical["probes"].keys()
    if not common:
        return summary
    last = max(common)
    matched = {arm: point(run["probes"][last]) for arm, run in pair.items()}
    summary.update(
        last_common_iteration=last,
        matched_probe=matched,
        component_counts_match_at_all_common_probes=all(
            strict["probes"][iteration]["component_gradient_calls"]
            == practical["probes"][iteration]["component_gradient_calls"]
            for iteration in common
        ),
    )
    for arm, run in pair.items():
        initial = run["probes"].get(0)
        if initial and initial["feasible_loss"] > 0:
            matched[arm]["repaired_loss_reduction"] = (
                1 - matched[arm]["feasible_loss"] / initial["feasible_loss"]
            )
    return summary


def plot_panel(ax, seeds, x, curves):
    has_points = False
    for pair in seeds.values():
        for arm, run in pair.items():
            rows = list(run["probes"].values())
            if not rows:
                continue
            has_points = True
            xs = [row[x] for row in rows]
            for y, style in curves:
                ax.plot(xs, [row[y] for row in rows], color=ARMS[arm][1], linestyle=style,
                        linewidth=1.5, marker="o" if len(rows) == 1 else None, markersize=4)
    if not has_points:
        ax.text(.5, .5, "Awaiting saved probes", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(BUDGETS[x])
    ax.xaxis.set_major_formatter(EngFormatter(sep="\N{THIN SPACE}"))


def plots(datasets, output):
    if not datasets:
        return
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    arm_handles = [Line2D([0], [0], color=color, label=label) for label, color in ARMS.values()]
    loss_handles = [Line2D([0], [0], color=".3", ls=style, label=label)
                    for style, label in (("-", "After diagnostic repair"), ("--", "Raw iterate"))]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(13, 3.0 * len(datasets) + 1.4), squeeze=False)
    for (dataset, seeds), row in zip(datasets.items(), axes):
        for x, ax in zip(BUDGETS, row):
            plot_panel(ax, seeds, x, (("feasible_loss", "-"), ("raw_loss", "--")))
            ax.set_yscale("log")
        row[0].set_ylabel(f"{DATASETS.get(dataset, dataset)}\nNormalized loss")
    fig.suptitle("SPRING-SARAH step-size ablation", y=.98)
    fig.legend(handles=arm_handles + loss_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(.5, .94), frameon=False)
    fig.text(.06, .012, TIMING_NOTE + " Raw and repaired curves may overlap.", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .88))
    save_plot(fig, output / "loss_by_budget")

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 3.0 * len(datasets) + 1), squeeze=False)
    for (dataset, seeds), row in zip(datasets.items(), axes):
        row[0].axhline(1e-10, color=".4", linewidth=1, linestyle=":")
        for ax, (metric, label) in zip(row, (("max_constraint_violation", "Max. constraint violation"),
                                             ("repair_distance", "Repair displacement"))):
            plot_panel(ax, seeds, "iteration", ((metric, "-"),))
            ax.set_ylabel(f"{DATASETS.get(dataset, dataset)}\n{label}")
            ax.set_yscale("symlog", linthresh=1e-15)
            ax.set_ylim(bottom=0)
    tolerance = Line2D([0], [0], color=".4", ls=":", label="Feasibility diagnostic tolerance")
    fig.suptitle("Raw iterate feasibility at saved probes", y=.98)
    fig.legend(handles=arm_handles + [tolerance], loc="upper center", ncol=3,
               bbox_to_anchor=(.5, .94), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save_plot(fig, output / "feasibility")


def save_plot(fig, stem):
    for extension in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--output", type=Path, default=HERE / "reports")
    args = parser.parse_args()
    datasets = load_runs(args.manifest.resolve())
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "timing": TIMING_NOTE,
        "component_gradient_evaluations": ORACLE_NOTE,
        "scope": "Paired step-size diagnostic at saved probes on the adapted factorization problem.",
        "datasets": {dataset: {str(seed): paired_summary(pair) for seed, pair in seeds.items()}
                     for dataset, seeds in datasets.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plots(datasets, args.output)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
