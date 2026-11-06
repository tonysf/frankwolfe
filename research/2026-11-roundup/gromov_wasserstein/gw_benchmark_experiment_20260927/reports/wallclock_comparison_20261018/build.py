"""Plot measured runtime endpoints, never infer unavailable timing trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_REPORT = ROOT / "experiments/large_pot_frames_20261016/status_20261018/reports/report.json"
COLORS = {"pot-full": "#2366a4", "frames": "#d26a19"}
LABELS = {
    "pot-full": r"Full-gradient POT ($\epsilon=10^{-3}$)",
    "frames": "Pair-sampled FRAMES (batch 1,024)",
}
NAMES = {"bmmc6224": "BMMC · 6,224 cells", "snare_full": "SNARE-seq · 10,000 cells"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_records(input_path, report_path):
    report = json.loads(report_path.read_text())
    records = json.loads(input_path.read_text())
    require(len(records) == 8, "Expected one POT and three FRAMES runs per dataset")
    rows, seen = [], set()
    for entry in records:
        dataset, task = entry["dataset"], entry["task"]
        require((dataset, task) not in seen, "Duplicate input run")
        seen.add((dataset, task))
        require(sha256(entry["raw"].encode()) == entry["sha256"], "Downloaded metadata hash differs")
        published = [r for r in report["datasets"][dataset]["runs"] if r["task_id"] == task]
        require(len(published) == 1, "Run is absent from archived report")
        require(published[0]["metadata_sha256"] == entry["sha256"], "Metadata differs from archived report")
        require(published[0]["metadata_path"] == entry["path"], "Metadata provenance path differs")
        meta = json.loads(entry["raw"])
        method = meta["method"]
        require(method in COLORS, "Unexpected method")
        expected_updates = 2_000 if method == "pot-full" else 20_000
        require(meta["status"] == "completed" and meta["completed_updates"] == expected_updates,
                "Incomplete or unexpected horizon")
        require(meta["world_size"] == 1 and meta["identity"]["device_model"] == "NVIDIA A100-SXM4-40GB",
                "Unexpected hardware")
        require(meta["config"]["dtype"] == "float64" and meta["config"]["epsilon"] == 0.001,
                "Unexpected precision or epsilon")
        require(meta["config"]["pairs"] == 1024, "Unexpected minibatch size")
        require(meta["final_loss"] == published[0]["final_loss"] == meta["trace"][-1]["rounded_loss"],
                "Final loss differs from archived report")
        require(math.isfinite(meta["elapsed_seconds"]) and meta["elapsed_seconds"] > 0, "Invalid runtime")
        require(all("elapsed_seconds" not in r for r in meta["trace"]),
                "Timing is now available in trace; update this endpoint-only report")
        rows.append(dict(dataset=dataset, task=task, method=method, seed=meta["seed"],
                         updates=meta["completed_updates"], elapsed_seconds=meta["elapsed_seconds"],
                         elapsed_minutes=meta["elapsed_seconds"] / 60, final_loss=meta["final_loss"],
                         initial_loss=meta["trace"][0]["rounded_loss"], metadata_sha256=entry["sha256"],
                         metadata_path=entry["path"], slurm_job_id=meta["slurm_job_id"], trace=meta["trace"]))
    for dataset in NAMES:
        methods = [r["method"] for r in rows if r["dataset"] == dataset]
        require(methods.count("pot-full") == 1 and methods.count("frames") == 3, "Incomplete dataset")
    return rows, report


def aggregate(rows):
    result = {}
    for dataset in NAMES:
        methods = {}
        for method in COLORS:
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            times = [r["elapsed_seconds"] for r in selected]
            losses = [r["final_loss"] for r in selected]
            methods[method] = dict(runs=len(selected), updates_per_run=selected[0]["updates"],
                                   median_elapsed_seconds=statistics.median(times),
                                   median_elapsed_minutes=statistics.median(times) / 60,
                                   elapsed_minutes_range=[min(times) / 60, max(times) / 60],
                                   median_final_loss=statistics.median(losses),
                                   final_loss_range=[min(losses), max(losses)],
                                   initial_loss=selected[0]["initial_loss"],
                                   median_loss_decrease_percent=100 * (1-statistics.median(losses)/selected[0]["initial_loss"]))
        result[dataset] = dict(methods=methods,
                               frames_over_pot_runtime_ratio=methods["frames"]["median_elapsed_seconds"] /
                               methods["pot-full"]["median_elapsed_seconds"],
                               frames_minus_pot_median_final_loss=methods["frames"]["median_final_loss"] -
                               methods["pot-full"]["median_final_loss"])
    return result


def plots(rows, report, stats, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, ScalarFormatter, StrMethodFormatter

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    artifacts = {}
    for dataset in NAMES:
        selected = [r for r in rows if r["dataset"] == dataset]
        fig, (runtime_ax, iteration_ax) = plt.subplots(1, 2, figsize=(12.6, 7.0))
        fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.38, wspace=.26)
        fig.suptitle(NAMES[dataset] + " | one 40-GB A100 per run | FP64", fontsize=16, y=.965, weight="bold")
        fig.text(.5, .91, "Recorded runtime endpoints and loss trajectories over iterations", ha="center", fontsize=12)
        runtime_ax.set_title("Measured final endpoints — no time trajectory", loc="left", pad=12)
        iteration_ax.set_title("Measured loss versus iterations", loc="left", pad=12)

        for row in selected:
            is_pot = row["method"] == "pot-full"
            runtime_ax.scatter(row["elapsed_minutes"], row["final_loss"], s=90 if is_pot else 65,
                               marker="s" if is_pot else "o", linewidth=1.7,
                               facecolors=COLORS[row["method"]] if is_pot else "white",
                               edgecolors=COLORS[row["method"]], zorder=4)
        fs = stats[dataset]["methods"]["frames"]
        runtime_ax.scatter(fs["median_elapsed_minutes"], fs["median_final_loss"], s=170, marker="D",
                           facecolors="none", edgecolors=COLORS["frames"], linewidth=2, zorder=3)
        y_values = [r["final_loss"] for r in selected]
        spread = max(y_values) - min(y_values)
        runtime_ax.set_ylim(min(y_values)-.27*spread, max(y_values)+.38*spread)
        runtime_ax.set_xlim(0, max(r["elapsed_minutes"] for r in selected)*1.18)
        runtime_ax.set_xlabel("Recorded total run time (minutes)")
        runtime_ax.set_ylabel("Final feasible GW loss · lower is better")
        runtime_ax.yaxis.set_major_locator(MaxNLocator(5))
        runtime_ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.5f}"))
        runtime_ax.text(.03, .96, "POT: 2,000 updates\nFRAMES: 20,000 updates", transform=runtime_ax.transAxes,
                        va="top", fontsize=10, bbox=dict(facecolor="white", edgecolor="none", alpha=.8))

        for method in COLORS:
            curves = report["datasets"][dataset]["curves"][method]
            xs = [r["iteration"] for r in curves]
            ys = [r["median"] for r in curves]
            iteration_ax.plot(xs, ys, color=COLORS[method], linewidth=2.1)
            if method == "frames":
                iteration_ax.fill_between(xs, [r["minimum"] for r in curves], [r["maximum"] for r in curves],
                                          color=COLORS[method], alpha=.14)
        iteration_ax.set_xlim(0, 20_000)
        iteration_ax.set_xlabel("Outer updates")
        iteration_ax.set_ylabel("Feasible GW loss · lower is better")
        iteration_ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        iteration_ax.yaxis.set_major_locator(MaxNLocator(5))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        iteration_ax.yaxis.set_major_formatter(formatter)
        for ax in (runtime_ax, iteration_ax):
            ax.grid(True, alpha=.2, linewidth=.7)
            ax.set_axisbelow(True)

        handles = [Line2D([], [], color=COLORS["pot-full"], marker="s", markersize=7,
                          label=LABELS["pot-full"] + " · one run"),
                   Line2D([], [], color=COLORS["frames"], marker="o", markerfacecolor="white", markersize=7,
                          label=LABELS["frames"] + " · three seeds"),
                   Line2D([], [], color="none", marker="D", markerfacecolor="none", markeredgecolor=COLORS["frames"], markersize=8,
                          label="FRAMES endpoint: median time and median loss")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .19), frameon=False, ncol=1, fontsize=10)
        fig.text(.08, .145, "Endpoint dots only: the original run logs did not record time at each loss diagnostic.",
                 fontsize=10, weight="bold")
        fig.text(.08, .07, "Time includes data loading, loss diagnostics and checkpoints; excludes queue wait and prefactor tuning.\n"
                 "Common metric: unregularized feasible GW loss. Iteration curve: median and observed seed range, not a confidence interval.",
                 fontsize=9, linespacing=1.45)
        artifacts[dataset] = []
        for extension in ("png", "pdf"):
            path = output / f"{dataset}_runtime_endpoints_and_iterations.{extension}"
            fig.savefig(path, dpi=180)
            artifacts[dataset].append(str(path))
        plt.close(fig)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE/"input/remote_records.json")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, report = load_records(args.input, args.report)
    stats = aggregate(rows)
    artifacts = plots(rows, report, stats, output)
    summary = dict(datasets=stats, artifacts=artifacts,
                   input_sha256=sha256(args.input.read_bytes()), archived_report_sha256=sha256(args.report.read_bytes()),
                   timing_scope="Recorded endpoint elapsed_seconds: data load, optimizer work, diagnostics and checkpoints; excludes queue wait, initialization before runner timer, final receipt publication/checkpoint cleanup and prefactor tuning.",
                   timing_limitations="No per-diagnostic timestamps; no loss-versus-time trajectories are inferred. A resumed run's cumulative clock omits serialization of the checkpoint used for resumption and the following shutdown tail.",
                   median_definition="FRAMES summary point pairs the median run duration with the median final loss; these coordinates need not come from the same seed.",
                   comparison_scope="One A100-SXM4-40GB per run, FP64; full-gradient POT epsilon=0.001 for 2,000 updates, pair-sampled FRAMES batch=1024 for 20,000 updates; selected FRAMES prefactors, development/tuning cost excluded.",
                   runs=[{k:v for k,v in r.items() if k != "trace"} for r in rows])
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    with (output/"summary.csv").open("w", newline="") as stream:
        fields = ["dataset", "method", "runs", "updates_per_run", "median_elapsed_minutes", "median_final_loss", "median_loss_decrease_percent"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for dataset, item in stats.items():
            for method, method_stats in item["methods"].items():
                row = dict(dataset=dataset, method=method, **method_stats)
                writer.writerow({key:row[key] for key in fields})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
