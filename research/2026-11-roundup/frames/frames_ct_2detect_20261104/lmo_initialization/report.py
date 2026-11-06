"""Compare paired SFRAMES traces from the feasible and Gaussian LMO starts."""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter
from problem_spec import add_problem_spec


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
STARTS = {
    "feasible": ("Original D-feasible start", "#0072B2"),
    "lmo": ("Gaussian-direction LMO start", "#D55E00"),
}
METADATA_FIELDS = (
    "optimizer", "caps", "data_sha256", "train_view_ids", "validation_view_ids",
)
FEASIBILITY_FIELDS = ("d_distance", "c_violation", "u_negative_norm", "v_negative_norm")
WORK = ("iteration", "forward_views", "backward_views", "gradient_calls")
TRAIN_LOSSES = (("feasible_loss", "-"), ("raw_loss", "--"))
PANELS = (
    ("forward_views", "Forward views", TRAIN_LOSSES, "Training loss", "log"),
    ("optimizer_seconds", "Optimizer time (s)", TRAIN_LOSSES, "Training loss", "log"),
    ("forward_views", "Forward views", (("validation_loss", "-"),), "Held-out view loss", "log"),
    ("forward_views", "Forward views", (("reference_relative_error", "-"),),
     "Relative reference-image error", None),
    ("iteration", "Optimizer iterations (symlog)", (("max_constraint_violation", "-"),),
     "Maximum constraint violation", "symlog"),
    ("iteration", "Optimizer iterations (symlog)", (("d_distance", "-"),),
     "Distance to D (Frobenius)", "symlog"),
)


def parent_report():
    spec = importlib.util.spec_from_file_location("ct_baseline_report", PARENT / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enrich(run):
    path = Path(run["path"])
    run["metadata"] = json.loads((path / "metadata.json").read_text())
    if run["rows"]:
        trace = json.loads((path / "trace.json").read_text())
        feasibility = {row["iteration"]: row for row in trace}
        for row in run["rows"]:
            saved = feasibility[row["iteration"]]
            row.update({key: saved[key] for key in FEASIBILITY_FIELDS if key in saved})
            # The original runs used C-feasible iterates, so their repair was orthant clipping.
            row.setdefault("d_distance", row["repair_distance"])
    return run


def match(original, rerun):
    left, right = original["config"], rerun["config"]
    if any(right.get(key) != value for key, value in left.items()):
        raise ValueError(f"Run configuration changed for {original['run_id']}")

    metadata = [run["metadata"] for run in (original, rerun)]
    if any(metadata[0][key] != metadata[1][key] for key in METADATA_FIELDS):
        raise ValueError(f"Paired run metadata differ for {original['run_id']}")

    probes = {row["iteration"]: row for row in original["rows"]}
    pairs = [(probes[row["iteration"]], row) for row in rerun["rows"] if row["iteration"] in probes]
    if any(old[key] != new[key] for old, new in pairs for key in WORK):
        raise ValueError(f"Paired probe work differs for {original['run_id']}")
    return pairs


def compare(original, rerun, run_summary):
    def summary(run):
        result = run_summary({key: val for key, val in run.items() if key != "metadata"})
        if "initialization" in run["metadata"]:
            result["initialization"] = run["metadata"]["initialization"]
        if run["rows"]:
            result["max_d_distance"] = max(row["d_distance"] for row in run["rows"])
            cap_excesses = [row["c_violation"] for row in run["rows"] if "c_violation" in row]
            if cap_excesses:
                result["max_c_violation"] = max(cap_excesses)
        return result

    output = {
        "run_id": original["run_id"],
        "config": original["config"],
        "feasible": summary(original),
        "lmo": summary(rerun) if rerun is not None else {"status": "waiting"},
    }
    if rerun is None:
        return output
    pairs = match(original, rerun)
    if pairs:
        old, new = pairs[-1]
        output["matched_work"] = {key: old[key] for key in WORK}
        output["matched_probes"] = {"feasible": old, "lmo": new}
    if original["rows"] and rerun["rows"]:
        cap = min(run["rows"][-1]["optimizer_seconds"] for run in (original, rerun))
        output["matched_optimizer_time"] = {
            "budget_seconds": cap,
            **{
                key: next(row for row in reversed(run["rows"]) if row["optimizer_seconds"] <= cap)
                for key, run in (("feasible", original), ("lmo", rerun))
            },
        }
    return output


def plot_panel(ax, runs, panel):
    xkey, xlabel, series, ylabel, scale = panel
    for start, run in runs.items():
        if run is None:
            continue
        for ykey, style in series:
            points = [(row[xkey], row[ykey]) for row in run["rows"] if row[ykey] is not None]
            if points:
                ax.plot(*zip(*points), color=STARTS[start][1], linestyle=style,
                        marker="o", markersize=2.5, linewidth=1.4)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if xkey == "iteration":
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0)
    else:
        ax.xaxis.set_major_formatter(EngFormatter(sep=" "))
    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_ylim(bottom=0)
    elif scale == "log":
        ax.set_yscale("log", nonpositive="mask")
    ax.grid(alpha=.22)


def plot_pair(pdf, original, rerun):
    config = original["config"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    fig.suptitle(
        f"{config['phase'].capitalize()} · {config['slice_id']} · "
        f"{config['n_train']:,} training views · γ₀ = {config['rate']:g}"
    )
    for ax, panel in zip(axes.flat, PANELS):
        plot_panel(ax, {"feasible": original, "lmo": rerun}, panel)
    legend = [
        Line2D([0], [0], color=color, label=label) for label, color in STARTS.values()
    ] + [
        Line2D([0], [0], color=".25", linestyle=style, label=label)
        for style, label in (("-", "Repaired loss"), ("--", "Raw loss"))
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .045), ncol=4,
               frameon=False, fontsize=9)
    fig.text(.05, .026, "Distance to D agrees with feasibility repair up to roundoff. "
             "The feasibility panels preserve zero and expand early iterations.", fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .95))
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, description, labels, rows, widths, note):
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_axis_off()
    fig.text(.08, .86, title, fontsize=22, weight="semibold")
    fig.text(.08, .78, description, fontsize=11, linespacing=1.5)
    table = ax.table(cellText=rows, colLabels=labels, colWidths=widths, cellLoc="center",
                     bbox=(0, .21, 1, .50))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#DDE9EF")
            cell.set_text_props(weight="semibold")
        elif row % 2:
            cell.set_facecolor("#F1F5F7")
    fig.text(.08, .15, note, fontsize=9, linespacing=1.5)
    pdf.savefig(fig)
    plt.close(fig)


def original_comparison(pdf, base, runs):
    cohorts = defaultdict(list)
    selected = defaultdict(list)
    for run in runs:
        config = run["config"]
        if config["phase"] != "evaluation" or not run["rows"]:
            continue
        cohorts[config["slice_id"], config["n_train"]].append(run)
        selected[config["method"]].append(next(
            row for row in reversed(run["rows"]) if row["forward_views"] <= config["view_budget"]
        ))
    if not cohorts:
        return
    rows = [
        [label, *(f"{median(row[key] for row in selected[method]):.5f}" for key in (
            "feasible_loss", "validation_loss", "reference_relative_error"
        ))]
        for method, (label, _) in base.METHODS.items() if selected[method]
    ]
    table_page(pdf, "Original four-method evaluation",
               "All methods began at the same D-feasible point for each evaluation case.",
               ["Method", "Repaired training\nloss", "Held-out view loss", "Relative reference\nerror"],
               rows, [.21, .26, .23, .30],
               "Medians across evaluation cases. Each value uses the last saved probe within its "
               "forward-view budget.\nThe next pages show the saved trajectories for each case.")
    with TemporaryDirectory(prefix="frames-original-ct-plots-") as folder:
        for case, members in sorted(cohorts.items()):
            base.plot_cohort(pdf, Path(folder), case, members)


def ablation_overview(pdf, summaries):
    rows = []
    for pair in summaries:
        config = pair["config"]
        initial = pair["lmo"].get("initial")
        probes = pair.get("matched_probes")
        rows.append([
            f"{config['slice_id']} / {config['n_train']}",
            f"{config['phase'].capitalize()} · {config['rate']:g}",
            f"{initial['d_distance']:.4g}" if initial else "—",
            f"{probes['feasible']['feasible_loss']:.5f}" if probes else "—",
            f"{probes['lmo']['feasible_loss']:.5f}" if probes else "—",
        ])
    table_page(pdf, "SFRAMES initialization ablation",
               "The new start uses independent Gaussian directions and one Frobenius-ball LMO per block.",
               ["Slice / views", "Phase · γ₀", "New initial\nD distance",
                "Original repaired\nloss", "LMO repaired\nloss"],
               rows, [.18, .19, .21, .21, .21],
               "The loss columns use the latest common saved iteration with identical view counts.\n"
               "The original initial D distance is zero in every pair.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=PARENT)
    parser.add_argument("--rerun", type=Path, default=HERE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.rerun / "reports"
    base = parent_report()
    original_runs = list(base.read_runs(args.original.resolve()))
    originals = [enrich(run) for run in original_runs if run["config"]["method"] == "sframes"]
    reruns = {run["run_id"]: enrich(run) for run in base.read_runs(args.rerun.resolve())}
    if not originals or reruns.keys() - {run["run_id"] for run in originals}:
        raise ValueError("Expected reruns that pair with original SFRAMES runs")

    pairs = [(run, reruns.get(run["run_id"])) for run in originals]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": [compare(old, new, base.run_summary) for old, new in pairs],
    }
    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / "comparison.pdf"
    temp_pdf = output / "comparison.pdf.tmp"
    with PdfPages(temp_pdf) as pdf:
        add_problem_spec(pdf, args.original.resolve(), args.rerun.resolve())
        original_comparison(pdf, base, original_runs)
        ablation_overview(pdf, summary["pairs"])
        for old, new in pairs:
            plot_pair(pdf, old, new)
    temp_pdf.replace(pdf_path)
    temp_summary = output / "summary.json.tmp"
    temp_summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temp_summary.replace(output / "summary.json")
    print(output.resolve())


if __name__ == "__main__":
    main()
