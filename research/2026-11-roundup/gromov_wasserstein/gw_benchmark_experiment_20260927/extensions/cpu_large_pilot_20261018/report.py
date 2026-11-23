"""Summarize completed timing events, including partial CPU pilot runs.

Standard library only. Default behavior reads one snapshot. An optional
bounded wait reads the campaign receipt once per minute before writing the
report. It never starts experiments or treats projections as measurements.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "experiments/cpu_large_pilot_20261018"
RUNS = [(dataset, method) for dataset in ("bmmc6224", "snare_full")
        for method in ("frames", "pot-full")]
STABLE_INNER = {"bmmc6224": 830, "snare_full": 250}


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_events(path):
    if not path.exists():
        return [], 0
    lines = path.read_text().splitlines()
    events, incomplete = [], 0
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError(f"Malformed interior event at {path}:{index + 1}")
            incomplete += 1
    return events, incomplete


def distribution(values):
    return (dict(count=len(values), mean=statistics.mean(values),
                 median=statistics.median(values), minimum=min(values), maximum=max(values))
            if values else dict(count=0, mean=None, median=None, minimum=None, maximum=None))


def diagnostic_iterations(horizon, every=500):
    fixed = {0, 1, 2, 5, 10, 20, 50, 100, 200, horizon}
    return sorted({k for k in fixed if k <= horizon} | set(range(every, horizon + 1, every)))


def gpu_reference(dataset, method):
    directory = ROOT / "reports/wallclock_comparison_20261018/input" / dataset
    records = []
    if method == "pot-full":
        paths = directory.glob("baseline/pot-full_seed810/metadata.json")
    else:
        paths = directory.glob("confirmation/*/metadata.json")
    for path in paths:
        record = read_json(path)
        if record and record.get("method") == method:
            records.append(dict(path=str(path.relative_to(ROOT)), seed=record["seed"],
                                seconds=record["elapsed_seconds"],
                                completed_updates=record["completed_updates"]))
    return dict(records=records,
                median_full_budget_seconds=statistics.median([r["seconds"] for r in records]) if records else None,
                note="Original one-A100 FP64 campaign; fixed-budget time, not time to an equal loss.")


def summarize_run(root, dataset, method):
    folder = root / "runs" / f"{dataset}_{method}"
    receipt = read_json(folder / "receipt.json") or {}
    metadata = read_json(folder / "metadata.json") or {}
    events, incomplete = read_events(folder / "profile.jsonl")
    updates = [e for e in events if e["phase"] == "update"]
    diagnostics = [e for e in events if e["phase"] == "diagnostic"]
    # Preserve sparse completed timing data, but do not let first-step warmup
    # stand in for sustained throughput when a later window is available.
    skipped = 2 if len(updates) > 2 else 0
    window = updates[skipped:][-20:]
    update_stats = distribution([e["duration_seconds"] for e in window])
    diagnostic_stats = distribution([e["duration_seconds"] for e in diagnostics])
    checkpoint_stats = distribution([e["duration_seconds"] for e in events if e["phase"] == "checkpoint"])
    component = {}
    for phase in ("load_problem", "checkpoint_load", "sample_pairs", "pair_gradient",
                  "full_gradient", "fw_step", "pot_step", "diagnostic", "checkpoint"):
        component[phase] = distribution([e["duration_seconds"] for e in events if e["phase"] == phase])
    completed = max([0] + [e.get("completed_updates", e.get("update_in_launch", 0)) for e in updates]
                    + [metadata.get("completed_updates", 0)])
    status = receipt.get("status", "not_started")
    pilot_finished = status in ("finished", "failed", "interrupted")
    inner = [e["inner_iterations"] for e in updates if e.get("inner_iterations") is not None]
    window_inner = [e["inner_iterations"] for e in window if e.get("inner_iterations") is not None]
    horizon = 20000 if method == "frames" else 2000
    diag_indices = diagnostic_iterations(horizon)
    tentative_reasons = []
    if method == "pot-full" and len(updates) < 10:
        tentative_reasons.append("Fewer than 10 POT updates; inner work increases after initialization.")
    if method == "pot-full" and window_inner and max(window_inner) < .9 * STABLE_INNER[dataset]:
        tentative_reasons.append("Observed inner iterations remain below 90% of the original GPU steady workload.")
    if len(updates) < 3:
        tentative_reasons.append("Too few completed updates to exclude the first two from throughput estimation.")
    if not updates:
        tentative_reasons.append("No completed outer update has been measured.")
    projection = dict(full_horizon_updates=horizon, diagnostic_every=500,
                      full_horizon_diagnostic_count=len(diag_indices), diagnostic_iterations=diag_indices,
                      tentative=bool(tentative_reasons), tentative_reasons=tentative_reasons,
                      status="unavailable", fixed_iteration_checkpoint_count=1 + len(set(range(5000, horizon + 1, 5000)) | {horizon}),
                      assumption="Update and diagnostic costs retain the observed medians. This is a duration extrapolation, not a measured full run or a prediction of final loss.")
    gpu = gpu_reference(dataset, method)
    if update_stats["count"] and diagnostic_stats["count"]:
        load_seconds = component["load_problem"]["median"] or 0.
        update_seconds = horizon * update_stats["median"]
        diagnostic_seconds = len(diag_indices) * diagnostic_stats["median"]
        base = load_seconds + update_seconds + diagnostic_seconds
        checkpoint_median = checkpoint_stats["median"] or 0.
        fixed = projection["fixed_iteration_checkpoint_count"]
        # The 30-minute checkpoint trigger can coincide with iteration-based
        # checkpoints. Bound its extra count rather than assert an exact count.
        extra = math.ceil(base / 1800)
        lower = base + fixed * checkpoint_median
        upper = base + (fixed + extra) * checkpoint_median
        projection.update(status="tentative" if tentative_reasons else "prefix_extrapolation",
                          load_seconds=load_seconds, optimizer_seconds=update_seconds,
                          diagnostics_seconds=diagnostic_seconds,
                          full_horizon_seconds_excluding_checkpoint_writes=base,
                          full_horizon_hours_excluding_checkpoint_writes=base / 3600,
                          checkpoint_seconds_median=checkpoint_median if checkpoint_stats["count"] else None,
                          checkpoint_overhead_estimated_seconds_lower_upper=[lower-base, upper-base] if checkpoint_stats["count"] else None,
                          full_horizon_hours_with_estimated_checkpoint_writes_lower_upper=[lower/3600, upper/3600] if checkpoint_stats["count"] else None,
                          ratio_to_gpu_fixed_budget_excluding_checkpoint_writes=base/gpu["median_full_budget_seconds"] if gpu["median_full_budget_seconds"] else None,
                          unmodeled="Imports, potential thermal/frequency drift, future changes in inner solve workload, and diagnostics/checkpoint cost changes. Checkpoint overhead range is a heuristic schedule estimate, not a bound on future runtime.")
    usage = dict(profile_interval_seconds=None, cpu_seconds=None, average_busy_cores=None,
                 fraction_of_four_cores=None)
    if len(events) >= 2:
        # Sum each launch separately so a future resume does not count idle
        # gaps between invocations or compare cumulative CPU counters across PIDs.
        launches = {}
        for event in events:
            launches.setdefault(event["launch_id"], []).append(event)
        wall = cpu = 0.
        for group in launches.values():
            first, last = group[0], group[-1]
            wall += last["elapsed_seconds"] - first["elapsed_seconds"]
            cpu += (last["cpu_user_seconds"] + last["cpu_system_seconds"]
                    - first["cpu_user_seconds"] - first["cpu_system_seconds"])
        if wall > 0:
            usage = dict(profile_interval_seconds=wall, cpu_seconds=cpu,
                         average_busy_cores=cpu/wall, fraction_of_four_cores=cpu/wall/4)
    peak = max([receipt.get("peak_rss_bytes_sampled", 0)] + [e["peak_rss_bytes"] for e in events])
    return dict(dataset=dataset, method=method, run_directory=str(folder.resolve()),
                pilot_status=status, pilot_finished=pilot_finished, not_finished=not pilot_finished,
                runner_status=metadata.get("status"), full_horizon_finished=metadata.get("status") == "completed",
                stop_reason=receipt.get("stop_reason"), process_seconds=receipt.get("process_seconds"),
                completed_updates=completed, measured_update_count=len(updates),
                timing_window=dict(first_updates_excluded=skipped, maximum_updates=20,
                                   outer_update_indices=[e.get("outer_update_index") for e in window]),
                update_seconds=update_stats, diagnostic_seconds=diagnostic_stats,
                component_seconds=component, cpu_usage=usage,
                peak_rss_bytes=peak, peak_rss_gib=peak/1024**3 if peak else None,
                diagnostic_trace=[e["diagnostic"] for e in diagnostics],
                observed_inner_iterations=inner, timing_window_inner_iterations=window_inner,
                original_gpu_steady_inner_iterations=STABLE_INNER[dataset] if method == "pot-full" else None,
                projection=projection, gpu_reference=gpu, incomplete_trailing_events_ignored=incomplete,
                receipt=receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--wait-for-completion", action="store_true",
                        help="Wait for the local campaign to finish, checking its receipt every 60 seconds for at most 4500 seconds, then write one report.")
    args = parser.parse_args(argv)
    wait_result = None
    if args.wait_for_completion:
        started = time.monotonic()
        deadline = started + 4500
        while True:
            observed = read_json(args.root / "campaign.json") or {}
            finished = observed.get("status") in ("finished", "stopped_memory_guard", "failed", "interrupted")
            if finished or time.monotonic() >= deadline:
                wait_result = dict(waited_seconds=time.monotonic()-started,
                                   maximum_seconds=4500, interval_seconds=60,
                                   completion_observed=finished, timed_out=not finished)
                break
            time.sleep(min(60., max(0., deadline-time.monotonic())))
    campaign = read_json(args.root / "campaign.json") or {}
    rows = [summarize_run(args.root, dataset, method) for dataset, method in RUNS]
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(),
                  campaign_status=campaign.get("status", "not_started"),
                  all_pilots_finished=all(r["pilot_finished"] for r in rows),
                  partial=not all(r["pilot_finished"] for r in rows),
                  completion_wait=wait_result,
                  notes=["Only completed profiling events are counted; an operation still running at this snapshot is absent.",
                         "CPU RNG and prefix arithmetic can differ from CUDA; same settings do not imply identical stochastic trajectories.",
                         "Projections compare the fixed experiment horizons, not equal achieved losses.",
                         "CPU busy-core utilization covers profiled intervals only, excluding imports and later unfinished operations."],
                  runs=rows)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = ["dataset", "method", "pilot_status", "not_finished", "completed_updates",
              "window_updates", "median_update_seconds", "median_diagnostic_seconds",
              "average_busy_cores", "peak_rss_gib", "projection_tentative",
              "full_horizon_updates", "full_horizon_diagnostic_count",
              "projected_hours_excluding_checkpoint_writes", "gpu_fixed_budget_seconds",
              "cpu_gpu_budget_ratio_excluding_checkpoint_writes", "observed_inner_iterations"]
    with (args.root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            projection = row["projection"]
            writer.writerow(dict(dataset=row["dataset"], method=row["method"],
                                 pilot_status=row["pilot_status"], not_finished=row["not_finished"],
                                 completed_updates=row["completed_updates"], window_updates=row["update_seconds"]["count"],
                                 median_update_seconds=row["update_seconds"]["median"],
                                 median_diagnostic_seconds=row["diagnostic_seconds"]["median"],
                                 average_busy_cores=row["cpu_usage"]["average_busy_cores"], peak_rss_gib=row["peak_rss_gib"],
                                 projection_tentative=projection["tentative"], full_horizon_updates=projection["full_horizon_updates"],
                                 full_horizon_diagnostic_count=projection["full_horizon_diagnostic_count"],
                                 projected_hours_excluding_checkpoint_writes=projection.get("full_horizon_hours_excluding_checkpoint_writes"),
                                 gpu_fixed_budget_seconds=row["gpu_reference"]["median_full_budget_seconds"],
                                 cpu_gpu_budget_ratio_excluding_checkpoint_writes=projection.get("ratio_to_gpu_fixed_budget_excluding_checkpoint_writes"),
                                 observed_inner_iterations=json.dumps(row["observed_inner_iterations"])))
    print(json.dumps(dict(partial=result["partial"], runs=[{key:r[key] for key in
        ("dataset", "method", "pilot_status", "completed_updates")} for r in rows],
        summary_json=str((args.root/"summary.json").resolve())), sort_keys=True))


if __name__ == "__main__":
    main()
