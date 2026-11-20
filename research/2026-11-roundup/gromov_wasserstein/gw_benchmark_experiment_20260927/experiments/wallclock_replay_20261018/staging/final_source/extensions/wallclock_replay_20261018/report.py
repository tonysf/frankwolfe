"""Plot measured feasible GW loss against actual elapsed active run time.

No iteration-to-time reconstruction is permitted. Three FRAMES seeds are
interpolated only within their shared observed time interval. The seed range
is descriptive and is not a confidence interval.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATASETS = ("bmmc6224", "snare_full")
TITLES = {"bmmc6224": "BMMC · 6,224 × 6,224 cells", "snare_full": "SNARE-seq · 10,000 × 10,000 cells"}
COLORS = {"pot-full": "#2266a8", "frames": "#ce5b19"}


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def tasks(config, dataset):
    baseline = config["baseline"]
    selected = config["frames"]["selected"][dataset]
    return [dict(task_id=f'baseline/pot-full_seed{baseline["full_seed"]}', method="pot-full",
                 seed=baseline["full_seed"], iterations=baseline["iterations"])] + [
        dict(task_id=f'confirmation/{selected["candidate_id"]}_seed{seed}', method="frames", seed=seed,
             iterations=config["frames"]["iterations"], **selected["prefactors"])
        for seed in config["frames"]["confirmation_seeds"]]


def validate_trace(meta, task, config, dataset):
    require(meta.get("status") == "completed", "Report requires completed runs")
    require(meta.get("completed_updates") == meta.get("target_updates") == task["iterations"],
            "Run has not completed its prescribed horizon")
    require(meta.get("method") == task["method"] and meta.get("seed") == task["seed"], "Run method/seed differs")
    for key in ("method", "seed", "iterations"):
        require(meta["config"][key] == task[key], f"Run configuration differs: {key}")
    for key in ("dtype", "epsilon", "pairs"):
        require(meta["config"][key] == config[key], f"Run configuration differs: {key}")
    if task["method"] == "frames":
        for key in ("gamma0", "rho0", "mu_scale"):
            require(meta["config"][key] == task[key], f"Selected FRAMES prefactor differs: {key}")
    require(meta.get("world_size") == 1 and meta["dimensions"]["n"] == meta["dimensions"]["m"] == config["instances"][dataset]["n"],
            "Dataset dimensions or one-GPU topology differ")
    require(meta.get("timing", {}).get("clock") == "time.monotonic", "Run lacks measured timing provenance")
    total = meta.get("elapsed_seconds")
    require(finite(total) and total > 0, "Invalid total elapsed time")
    trace = meta.get("trace")
    require(isinstance(trace, list) and len(trace) > 1, "Missing measured diagnostic trajectory")
    iterations = [row.get("iteration") for row in trace]
    require(all(isinstance(k, int) and not isinstance(k, bool) for k in iterations)
            and iterations == sorted(set(iterations)) and iterations[0] == 0
            and iterations[-1] == task["iterations"], "Incomplete or unordered iteration trajectory")
    elapsed = [row.get("elapsed_seconds") for row in trace]
    require(all(finite(value) and value >= 0 for value in elapsed)
            and all(left < right for left, right in zip(elapsed, elapsed[1:])),
            "Diagnostic times must be finite, actual, and strictly increasing")
    require(elapsed[-1] <= total, "Diagnostic time exceeds total elapsed time")
    require(all(finite(row.get("rounded_loss")) for row in trace), "Invalid feasible GW loss")
    require(meta.get("final_loss") == trace[-1]["rounded_loss"], "Final loss differs from the timed trace")
    return trace


def common_time_summary(traces):
    """Linear interpolation of actual measurements, never extrapolation."""
    left = max(trace[0]["elapsed_seconds"] for trace in traces)
    right = min(trace[-1]["elapsed_seconds"] for trace in traces)
    require(left < right, "FRAMES runs have no shared measured time interval")
    grid = np.unique([left, right] + [row["elapsed_seconds"] for trace in traces for row in trace
                                    if left <= row["elapsed_seconds"] <= right])
    values = np.array([np.interp(grid,
                                [row["elapsed_seconds"] for row in trace],
                                [row["rounded_loss"] for row in trace]) for trace in traces])
    return dict(elapsed_seconds=grid, median=np.median(values, axis=0),
                minimum=values.min(axis=0), maximum=values.max(axis=0), seeds=len(traces))


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def collect(root, config):
    root = Path(root)
    datasets, csv_rows, summaries = {}, [], {}
    for dataset in DATASETS:
        runs = []
        for task in tasks(config, dataset):
            path = root/"results"/dataset/task["task_id"]/"metadata.json"
            meta = read(path)
            trace = validate_trace(meta, task, config, dataset)
            runs.append(dict(task=task, meta=meta, path=path, trace=trace))
            csv_rows += [dict(dataset=dataset, method=task["method"], seed=task["seed"],
                              iteration=row["iteration"], elapsed_seconds=row["elapsed_seconds"],
                              rounded_loss=row["rounded_loss"]) for row in trace]
        full = next(run for run in runs if run["task"]["method"] == "pot-full")
        frames = [run for run in runs if run["task"]["method"] == "frames"]
        require(len(frames) == 3, "This report requires all three prescribed FRAMES seeds")
        aggregate = common_time_summary([run["trace"] for run in frames])
        endpoints = [dict(method=run["task"]["method"], seed=run["task"]["seed"],
                          updates=run["meta"]["completed_updates"],
                          final_loss=run["meta"]["final_loss"],
                          initial_loss=run["trace"][0]["rounded_loss"],
                          loss_reduction_fraction=1-run["meta"]["final_loss"]/run["trace"][0]["rounded_loss"],
                          last_diagnostic_elapsed_seconds=run["trace"][-1]["elapsed_seconds"],
                          total_elapsed_seconds=run["meta"]["elapsed_seconds"],
                          slurm_job_id=run["meta"].get("slurm_job_id"),
                          timing=run["meta"]["timing"],
                          metadata_path=str(run["path"]),
                          metadata_sha256=hashlib.sha256(run["path"].read_bytes()).hexdigest()) for run in runs]
        frames_endpoints = [row for row in endpoints if row["method"] == "frames"]
        pot_endpoint = next(row for row in endpoints if row["method"] == "pot-full")
        frame_stats = {key:stats([row[key] for row in frames_endpoints]) for key in (
            "final_loss", "loss_reduction_fraction", "last_diagnostic_elapsed_seconds", "total_elapsed_seconds")}
        summaries[dataset] = dict(runs=endpoints, full_gradient_pot=pot_endpoint,
                                  frames_three_seeds=frame_stats,
                                  frames_to_pot_total_elapsed_ratio=frame_stats["total_elapsed_seconds"]["median"]/pot_endpoint["total_elapsed_seconds"],
                                  frames_common_time_interval_seconds=[float(aggregate["elapsed_seconds"][0]), float(aggregate["elapsed_seconds"][-1])])
        datasets[dataset] = dict(full=full, frames=frames, aggregate=aggregate)
    return datasets, csv_rows, summaries


def render(dataset, data, output):
    full, frames, aggregate = data["full"], data["frames"], data["aggregate"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), sharey=True)
    for index, ax in enumerate(axes):
        trace = full["trace"]
        ax.plot(np.array([row["elapsed_seconds"] for row in trace])/60,
                [row["rounded_loss"] for row in trace], color=COLORS["pot-full"], linewidth=2,
                label=r"Full-gradient POT, $\epsilon=10^{-3}$ (one run)")
        ax.scatter([trace[-1]["elapsed_seconds"]/60], [trace[-1]["rounded_loss"]],
                   color=COLORS["pot-full"], s=26, zorder=4)
        for run in frames:
            trace = run["trace"]
            ax.plot(np.array([row["elapsed_seconds"] for row in trace])/60,
                    [row["rounded_loss"] for row in trace], color=COLORS["frames"], alpha=.25, linewidth=.8)
            ax.scatter([trace[-1]["elapsed_seconds"]/60], [trace[-1]["rounded_loss"]],
                       color=COLORS["frames"], marker="x", s=30, zorder=4)
        minutes = aggregate["elapsed_seconds"]/60
        ax.fill_between(minutes, aggregate["minimum"], aggregate["maximum"],
                        color=COLORS["frames"], alpha=.18, label="FRAMES min–max across three seeds")
        ax.plot(minutes, aggregate["median"], color=COLORS["frames"], linewidth=2,
                label="FRAMES median (1,024 pairs/update)")
        if index == 0:
            max_time = max(run["trace"][-1]["elapsed_seconds"] for run in [full, *frames])/60
            ax.set_xlim(0, max_time*1.025)
            ax.set_title("Full measured horizon")
        else:
            ax.set_xlim(0, 5)
            ax.set_title("First five minutes")
            if all(run["trace"][0]["elapsed_seconds"] > 300 for run in [full, *frames]):
                ax.text(.5, .5, "No completed diagnostic in this interval", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
        ax.set_xlabel("Measured active wall time (minutes)")
        ax.grid(True, alpha=.23)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set_ylabel("Feasible squared GW loss (rounded diagnostic)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .10), ncol=2,
               frameon=False, fontsize=9)
    fig.suptitle(TITLES[dataset]+" · one A100 per run, FP64", fontsize=13)
    fig.text(.5, .045,
             "Time includes data loading, diagnostics and checkpoint writes; excludes queueing and prefactor tuning.\n"
             "FRAMES median/range use interpolation only on shared measured time support; crosses mark individual endpoints.\n"
             "The three-seed range is not a confidence interval. POT: 2,000 updates; FRAMES: 20,000 updates per seed.",
             ha="center", va="center", fontsize=8)
    fig.subplots_adjust(left=.08, right=.985, top=.86, bottom=.29, wspace=.10)
    files = []
    for extension in ("png", "pdf"):
        path = output/f"{dataset}_loss_vs_time.{extension}"
        fig.savefig(path, dpi=180 if extension == "png" else None)
        files.append(path.name)
    plt.close(fig)
    return files


def report(root, output, config=None):
    config = read(HERE/"campaign.json") if config is None else config
    datasets, rows, summaries = collect(root, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"traces.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("dataset", "method", "seed", "iteration", "elapsed_seconds", "rounded_loss"))
        writer.writeheader()
        writer.writerows(rows)
    figures = {dataset:render(dataset, data, output) for dataset, data in datasets.items()}
    result = dict(status="completed", generated_at=datetime.now(timezone.utc).isoformat(),
                  root=str(Path(root).resolve()), metric="rounded_loss", time_axis="measured elapsed_seconds; no iteration-based estimates",
                  aggregation="linear interpolation of each FRAMES trace on the union of measured times within shared observed support; median and min–max across three seeds",
                  uncertainty="Seed range is descriptive, not a confidence interval; full-gradient POT has one observed trajectory",
                  timing="Includes loading, diagnostics and checkpoint writes. Excludes queueing, imports, CUDA initialization, final receipt/cleanup and prefactor tuning. Resumed time sums active chunks and may omit work after the last durable receipt before a crash.",
                  datasets=summaries, figures=figures, csv="traces.csv")
    (output/"summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = report(args.root, args.output)
    print(json.dumps(dict(status=result["status"], output=str(Path(args.output).resolve())), sort_keys=True))


if __name__ == "__main__":
    main()
