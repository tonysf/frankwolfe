"""Analyze joint-tuned SFRAMES with preserved dense baseline provenance."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
ANCHOR = "sframes"
VARIANTS = {
    "sframes": ("Minibatch SFRAMES", "#0072B2"),
    "sframes_full": ("Full-batch SFRAMES", "#E69F00"),
    "spring_sarah": ("SPRING-SARAH", "#D55E00"),
    "prox_sarah": ("ProxSARAH", "#009E73"),
    "palm": ("PALM", "#AA4499"),
}
METRICS = (
    "raw_loss", "feasible_loss", "validation_loss", "reference_relative_error",
    "max_constraint_violation", "repair_distance", "c_violation", "d_distance",
)
WORK = ("iteration", "forward_views", "backward_views", "gradient_calls")
PROJECTION = (
    "projection_calls", "projection_inner_iterations", "projection_dykstra_calls",
    "projection_floor_count", "max_projection_distance_bound",
    "max_lmo_support_gap_bound",
)
MATCHED_METADATA = (
    "data_sha256", "calibration_sha256", "train_view_ids", "validation_view_ids",
    "caps", "curvature", "initialization", "constraint_geometry", "source_sha256",
    "optimizer",
)
PAIRED_METADATA = (
    "data_sha256", "calibration_sha256", "train_view_ids", "validation_view_ids",
    "caps", "curvature", "initialization", "constraint_geometry",
)
NUMERICAL_SOURCES = (
    "ct_problem.py", "methods.py", "spectral_reference_core.py",
    "spectral_reference_baseline.py",
)
def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def slice_id(value):
    return f"slice{int(str(value).removeprefix('slice')):05d}"


def collect(root):
    """Read completed results and checkpoint status without touching optimizer state."""
    found = {}
    for metadata_path in sorted((root / "runs").glob("*/metadata.json")):
        meta = read_json(metadata_path, {})
        config = meta["config"]
        if config["phase"] != "evaluation":
            continue
        variant = config["variant"]
        expected_method = "sframes" if variant == "sframes_full" else variant
        if variant not in VARIANTS or config["method"] != expected_method:
            raise ValueError(f"Unexpected optimizer variant in {metadata_path}")
        key = slice_id(config["slice_id"]), int(config["seed"]), int(config["n_train"]), variant
        if key in found:
            raise ValueError(f"Duplicate evaluation arm {key}")
        folder = metadata_path.parent
        progress = read_json(folder / "progress.json", {})
        result = read_json(folder / "result.json", None)
        status = progress.get("status", "registered")
        trace = []
        if result is not None:
            trace = read_json(folder / "trace.json", [])
            if (result["status"] != "completed" or result["config"] != config or not trace
                    or result["iteration"] != trace[-1]["iteration"] or result["final"] != trace[-1]):
                raise ValueError(f"Completed result and saved probes disagree in {folder}")
            if any(meta[field] != result[field] for field in MATCHED_METADATA):
                raise ValueError(f"Completed result and metadata disagree in {folder}")
            validate_trace(trace, folder)
            status = "completed"
        elif status == "completed":
            status = "awaiting result"
        source_directory = folder.resolve()
        found[key] = dict(config=config, meta=meta, trace=trace, status=status,
                          source_directory=str(source_directory),
                          reused=source_directory.parent != (root / "runs").resolve(),
                          checkpoint_iteration=progress.get("iteration"), error=progress.get("error"))
    return found


def validate_trace(trace, folder):
    if any(trace[0][key] != 0 for key in WORK):
        raise ValueError(f"Expected an iteration-zero probe in {folder}")
    for row in trace:
        for key in WORK + METRICS:
            value = row[key]
            if key == "reference_relative_error" and value is None:
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} in {folder}")
    for previous, current in zip(trace, trace[1:]):
        if current["iteration"] <= previous["iteration"] or any(
            current[key] < previous[key] for key in WORK
        ):
            raise ValueError(f"Probe work is not increasing in {folder}")


def verify_pair(anchor, comparator):
    if anchor["config"]["calibration_seed"] != comparator["config"]["calibration_seed"]:
        raise ValueError("Paired evaluation arms have different calibration seeds")
    for key in PAIRED_METADATA:
        if anchor["meta"][key] != comparator["meta"][key]:
            raise ValueError(f"Paired evaluation arms disagree on {key}")
    a_source = anchor["meta"]["source_sha256"]
    b_source = comparator["meta"]["source_sha256"]
    for name in NUMERICAL_SOURCES:
        if name not in a_source or name not in b_source or a_source[name] != b_source[name]:
            raise ValueError(f"Paired evaluation arms use different numerical source {name}")
    a, b = anchor["trace"][0], comparator["trace"][0]
    for key in METRICS:
        if a[key] is None and b[key] is None:
            continue
        if a[key] is None or b[key] is None or not math.isclose(
            a[key], b[key], rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(f"Paired evaluation arms have different initial {key}")


def saved_probes(trace, view_cap):
    if not trace:
        return {}
    within_views = next(row for row in reversed(trace)
                        if max(row["forward_views"], row["backward_views"]) <= view_cap)
    fields = WORK + METRICS + PROJECTION
    return {budget: {key: row[key] for key in fields if key in row}
            for budget, row in (("view_cap", within_views), ("endpoint", trace[-1]))}


def slice_stat(values, mixes):
    """Collapse paired seeds within a slice, then summarize across slices."""
    by_slice = {sid: median(v) for sid, v in sorted(values.items()) if v}
    by_mix = defaultdict(list)
    for sid, value in by_slice.items():
        by_mix[mixes[sid]].append(value)
    return {
        "median_across_slices": median(by_slice.values()) if by_slice else None,
        "slice_medians": by_slice,
        "seed_ranges_by_slice": {sid: [min(values[sid]), max(values[sid])] for sid in by_slice},
        "within_mix": {mix: {"slices": len(v), "median": median(v), "range": [min(v), max(v)]}
                       for mix, v in sorted(by_mix.items())},
        "paired_seed_counts_by_slice": {sid: len(values[sid]) for sid in by_slice},
        "slices": len(by_slice),
        "seed_pairs": sum(len(values[sid]) for sid in by_slice),
        "slices_favoring_sframes": sum(v < 0 for v in by_slice.values()),
        "slices_tied": sum(v == 0 for v in by_slice.values()),
    }


def paired_summary(cases, comparator, budget, mixes):
    values = {field: defaultdict(list) for field in METRICS}
    percentages = {field: defaultdict(list) for field in METRICS}
    n = 0
    for case in cases:
        runs = case["arms"]
        if budget not in runs[ANCHOR]["probes"] or budget not in runs[comparator]["probes"]:
            continue
        a = runs[ANCHOR]["probes"][budget]
        b = runs[comparator]["probes"][budget]
        n += 1
        for key in METRICS:
            if a[key] is None or b[key] is None:
                continue
            delta = a[key] - b[key]
            values[key][case["slice_id"]].append(delta)
            if b[key] > 0:
                percentages[key][case["slice_id"]].append(100 * delta / b[key])
    return {
        "seed_pairs": n,
        "metrics": {key: {"absolute_difference": slice_stat(values[key], mixes),
                          "percent_difference": slice_stat(percentages[key], mixes)}
                    for key in METRICS},
    }


def marginal_summary(cases, budget):
    """Describe the common complete subset separately from pairwise estimates."""
    complete = [case for case in cases if all(
        budget in arm["probes"] for arm in case["arms"].values()
    )]
    methods = {}
    for variant in VARIANTS:
        group = defaultdict(list)
        for case in complete:
            group[case["slice_id"]].append(case["arms"][variant]["probes"][budget])
        summary = {}
        for key in WORK + METRICS + PROJECTION:
            per_slice = {}
            for sid, rows in group.items():
                values = [row[key] for row in rows if row.get(key) is not None]
                if values:
                    per_slice[sid] = median(values)
            summary[key] = median(per_slice.values()) if per_slice else None
        methods[variant] = summary
    return {"complete_cases": len(complete), "slices": len({c["slice_id"] for c in complete}),
            "median_of_slice_medians": methods}


def analyze(root, design, view_cap):
    found = collect(root)
    slices = sorted({slice_id(item) for item in design["evaluation_slices"]})
    seeds = sorted({int(seed) for seed in design["evaluation_seeds"]})
    training_sizes = sorted({int(n) for n in design["evaluation_training_views"]})
    mixes = {slice_id(sid): str(mix) for sid, mix in design["evaluation_slice_mixes"].items()}
    if set(mixes) != set(slices):
        raise ValueError("Every evaluation slice needs a material mix")
    deterministic = set(design["deterministic_variants"])
    if ANCHOR in deterministic or deterministic - set(VARIANTS):
        raise ValueError("Invalid deterministic variants in the frozen design")
    deterministic_seed = seeds[0]
    expected = list(product(slices, seeds, training_sizes))
    physical = {(sid, seed, n, variant) for sid, seed, n in expected for variant in VARIANTS
                if variant not in deterministic or seed == deterministic_seed}
    if set(found) - physical:
        raise ValueError("Saved evaluation arms include cases outside the frozen design")
    cases = []
    for sid, seed, n_train in expected:
        present = {v: found.get((sid, deterministic_seed if v in deterministic else seed, n_train, v))
                   for v in VARIANTS}
        complete = [r for r in present.values() if r and r["trace"]]
        for peer in complete[1:]:
            verify_pair(complete[0], peer)
        arms = {}
        for variant, run in present.items():
            if run is None:
                arms[variant] = {"status": "missing metadata", "probes": {}}
                continue
            arms[variant] = {
                "status": run["status"], "run_id": run["config"]["run_id"],
                "shared_deterministic_reference": variant in deterministic,
                "reused_existing_run": run["reused"],
                "source_directory": run["source_directory"],
                "source_sha256": run["meta"]["source_sha256"],
                "optimizer": run["meta"]["optimizer"],
                "checkpoint_iteration": run["checkpoint_iteration"],
                "error": run["error"], "probes": saved_probes(run["trace"], view_cap),
            }
        cases.append(dict(slice_id=sid, mix=mixes[sid], seed=seed, n_train=n_train, arms=arms))
    cohorts = {}
    for n in training_sizes:
        subset = [case for case in cases if case["n_train"] == n]
        cohorts[str(n)] = {
            budget: {
                "matched_all_methods": marginal_summary(subset, budget),
                "sframes_vs": {variant: paired_summary(subset, variant, budget, mixes)
                               for variant in VARIANTS if variant != ANCHOR},
            }
            for budget in ("view_cap", "endpoint")
        }
    campaign = read_json(root / "progress.json", {})
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "design": design,
        "budgets": {"forward_view_cap": view_cap, "backward_view_cap": view_cap},
        "comparison": {
            "direction": "Minibatch SFRAMES minus comparator. Negative favors SFRAMES.",
            "pairing": "Same slice and number of training views with shared calibration. "
                       "Stochastic arms pair by oracle seed. Each deterministic arm is run "
                       "once per slice and training size and serves as a shared reference.",
            "within_budget": "Last completed-run probe at or below the cap, without interpolation.",
            "endpoint": "Last probe. Different methods may overshoot the nominal stopping budget.",
            "slice_estimate": "Median paired difference across available seeds within each slice.",
            "summary_estimate": "Median of slice medians at each training-view count.",
            "mixes": "Slices within a material mix may be correlated. Overall and within-mix "
                     "estimates are descriptive summaries.",
            "percentages": "Percent difference uses the paired comparator value as the denominator. "
                           "Zero denominators are excluded from percent summaries.",
            "source_provenance": "Each arm retains its original runner hash and source directory. "
                                 "Joint-tuned SFRAMES and reused dense baselines may have different "
                                 "run.py hashes. Pairing requires identical data, calibration, "
                                 "initialization, constraints and shared numerical source hashes.",
        },
        "coverage": {
            "expected_physical_arms": len(physical),
            "expected_paired_seed_cases": len(expected),
            "case_counts_by_n_train": dict(Counter(c["n_train"] for c in cases)),
            "physical_arm_status_counts": dict(Counter(
                found[key]["status"] if key in found else "missing metadata" for key in physical)),
            "physical_arm_origin_counts": dict(Counter(
                "reused_existing_run" if run["reused"] else "executed_in_this_campaign"
                for run in found.values())),
            "campaign_phase": campaign.get("phase"),
            "campaign_status": campaign.get("status"),
            "campaign_failures": {key: value for key, value in campaign.get("runs", {}).items()
                                  if value.get("status") == "failed"},
        },
        "selection": read_json(root / "selection.json", None),
        "cohorts": cohorts,
        "cases": cases,
    }


def plot_effects(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = [v for v in VARIANTS if v != ANCHOR]
    headings = {"feasible_loss": "Repaired training loss", "validation_loss": "Held-out view loss"}
    mixes = {slice_id(sid): str(mix) for sid, mix in summary["design"]["evaluation_slice_mixes"].items()}
    markers = dict(zip(sorted(set(mixes.values())), ("o", "^", "s", "v", "P")))
    budget = "view_cap"
    title = f"{summary['budgets']['forward_view_cap']:,} views in each projection direction"
    cohorts = summary["cohorts"]
    fig, axes = plt.subplots(2, len(cohorts), figsize=(11.5, 6.5), squeeze=False)
    for j, (n, cohort) in enumerate(cohorts.items()):
        for i, (metric, heading) in enumerate(headings.items()):
            ax = axes[i, j]
            ax.axvline(0, color="0.45", linewidth=1)
            labels = []
            for k, variant in enumerate(variants):
                name, color = VARIANTS[variant]
                stats = cohort[budget]["sframes_vs"][variant]["metrics"][metric]["percent_difference"]
                values = stats["slice_medians"]
                labels.append(f"{name}  ({len(values)} slices)")
                if not values:
                    continue
                positions = np.linspace(k - 0.13, k + 0.13, len(values)) if len(values) > 1 else [k]
                for (sid, value), y in zip(values.items(), positions):
                    ax.scatter(value, y, color=color, marker=markers.get(mixes[sid], "o"),
                               s=23, alpha=0.70, zorder=2)
                ax.scatter(stats["median_across_slices"], k, marker="D", color=color,
                           edgecolor="white", linewidths=0.5, s=38, zorder=4)
            ax.set(yticks=range(len(variants)), yticklabels=labels, ylim=(len(variants)-0.4, -0.6),
                   title=f"{heading} · {int(n):,} training views",
                   xlabel="Paired difference from comparator (%)")
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="x", alpha=0.22)
            ax.set_axisbelow(True)
    fig.suptitle(f"Minibatch SFRAMES at {title}")
    legend = " and ".join(f"{symbol} {mix}" for symbol, mix in
                          zip(("circles", "triangles"), sorted(markers)))
    fig.text(.05, .022, "Small marks show within-slice seed medians "
             f"({legend}). Diamonds show the median across slices. Negative favors SFRAMES.",
             fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(destination, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forward-cap", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    design = read_json(root / "manifest.json", {})["design"]
    view_cap = args.forward_cap if args.forward_cap is not None else design["evaluation_view_budget"]
    if view_cap <= 0:
        parser.error("the projection-view cap must be positive")
    summary = analyze(root, design, view_cap)
    output = args.output or root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        pending = output / "comparison.pdf.tmp"
        plot_effects(summary, pending)
        pending.replace(output / "comparison.pdf")
    pending = output / "summary.json.tmp"
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pending.replace(output / "summary.json")
    print(output.resolve())


if __name__ == "__main__":
    main()
