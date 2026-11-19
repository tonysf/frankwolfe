"""Analyze native CT trajectories with specimen groups as the aggregation unit.

This module never loads an optimizer checkpoint or performs a projection.  It
uses only recorded observations, with no interpolation of trajectories or work
to accuracy targets.  Unreached targets are explicitly right censored.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
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
OPTIONAL_PAIRED_METADATA = (
    "projector", "factor_dtype", "initial_factor_frobenius_norm", "initial_factor_norm",
    "geometry", "detector", "detector_index", "preprocessing",
)
NUMERICAL_SOURCES = (
    "ct_problem.py", "methods.py", "spectral_reference_core.py",
    "spectral_reference_baseline.py",
)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slice_id(value):
    return f"slice{int(str(value).removeprefix('slice')):05d}"


def group_map(design):
    mapping = design.get("evaluation_slice_groups", design.get("evaluation_slice_mixes"))
    if mapping is None:
        raise ValueError("The frozen design must assign each evaluation slice to a specimen group")
    result = {}
    for sid, group in mapping.items():
        if isinstance(group, dict):
            group = group.get("group_id", group.get("group"))
        if not isinstance(group, str) or not group:
            raise ValueError(f"Invalid specimen group for {sid}")
        result[slice_id(sid)] = group
    expected = {slice_id(sid) for sid in design["evaluation_slices"]}
    if set(result) != expected:
        raise ValueError("The specimen-group map does not match the evaluation cohort")
    return result


def validate_trace(trace, folder):
    if not trace or any(trace[0][key] != 0 for key in WORK):
        raise ValueError(f"Expected an iteration-zero probe in {folder}")
    if [row["iteration"] for row in trace] != list(range(trace[-1]["iteration"] + 1)):
        raise ValueError(f"The evaluation trace is missing native iteration observations in {folder}")
    for row in trace:
        if any("seconds" in field or "optimizer_time" in field for field in row):
            raise ValueError(f"Optimizer timing is outside this experiment's protocol: {folder}")
        for key in WORK + METRICS:
            value = row[key]
            if key == "reference_relative_error" and value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} in {folder}")
        if row["forward_views"] != row["backward_views"]:
            raise ValueError(f"Forward and adjoint counters disagree in {folder}")
    for previous, current in zip(trace, trace[1:]):
        if any(current[key] <= previous[key] for key in WORK):
            raise ValueError(f"Iteration/work counters are not strictly increasing in {folder}")


def validate_continuation(meta, trace, folder):
    origin = meta.get("continuation")
    if not origin:
        return None
    source = Path(origin["source_directory"])
    if not source.is_absolute():
        raise ValueError(f"Continuation provenance needs an absolute source path in {folder}")
    source_trace = read_json(source / "trace.json")
    if source_trace is None:
        raise ValueError(f"Continuation source trace is unavailable in {folder}")
    expected_hash = origin.get("source_trace_sha256", origin.get("source_prefix_sha256"))
    if expected_hash and sha256(source / "trace.json") != expected_hash:
        raise ValueError(f"The source trace changed after continuation in {folder}")
    source_iteration = origin["source_final_iteration"]
    if (source_trace[-1]["iteration"] != source_iteration
            or trace[:len(source_trace)] != source_trace):
        raise ValueError(f"A continuation changed a recorded prefix in {folder}")
    for filename, field in (("checkpoint.pkl", "source_checkpoint_sha256"),
                            ("result.json", "source_result_sha256")):
        if field in origin and sha256(source / filename) != origin[field]:
            raise ValueError(f"Continuation source {filename} changed in {folder}")
    return origin


def collect(root):
    """Read result/metadata pairs and audit the unchanged prefixes of continuations."""
    audit_path = root / "audit_provenance.py"
    auditor = None
    if audit_path.is_file():
        spec = importlib.util.spec_from_file_location("expanded_ct_provenance_auditor", audit_path)
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
    elif (root / "cohort.json").is_file() and (root / "protocol_frozen.json").is_file():
        raise ValueError("The expanded campaign requires its frozen-input provenance audit gate")
    found = {}
    for path in sorted((root / "runs").glob("*/metadata.json")):
        meta = read_json(path)
        config = meta["config"]
        if config["phase"] != "evaluation":
            continue
        if auditor is not None:
            auditor.validate_evaluation_metadata(root, meta)
        variant = config["variant"]
        method = "sframes" if variant == "sframes_full" else variant
        if variant not in VARIANTS or config["method"] != method:
            raise ValueError(f"Unexpected optimizer variant in {path}")
        key = slice_id(config["slice_id"]), int(config["seed"]), int(config["n_train"]), variant
        if key in found:
            raise ValueError(f"Duplicate evaluation arm {key}")
        folder = path.parent
        progress = read_json(folder / "progress.json", {})
        result = read_json(folder / "result.json")
        status = progress.get("status", "registered")
        trace = []
        continuation = meta.get("continuation")
        if result is not None:
            trace = read_json(folder / "trace.json", [])
            if (result["status"] != "completed" or result["config"] != config or not trace
                    or result["iteration"] != trace[-1]["iteration"] or result["final"] != trace[-1]):
                raise ValueError(f"Completed result and saved observations disagree in {folder}")
            for field in MATCHED_METADATA:
                if meta[field] != result[field]:
                    raise ValueError(f"Completed result and metadata disagree on {field} in {folder}")
            for field in OPTIONAL_PAIRED_METADATA + ("continuation",):
                if (field in meta or field in result) and meta.get(field) != result.get(field):
                    raise ValueError(f"Completed result and metadata disagree on {field} in {folder}")
            validate_trace(trace, folder)
            continuation = validate_continuation(meta, trace, folder)
            status = "completed"
        elif status == "completed":
            status = "awaiting result"
        source_directory = folder.resolve()
        reused = source_directory.parent != (root / "runs").resolve()
        origin = ("reused_existing_run" if reused else "continued_existing_run" if continuation
                  else "executed_from_initialization")
        found[key] = dict(config=config, meta=meta, trace=trace, status=status,
                          source_directory=str(source_directory), reused=reused, origin=origin,
                          continuation=continuation,
                          checkpoint_iteration=progress.get("iteration"), error=progress.get("error"))
    return found


def verify_pair(anchor, comparator):
    if anchor["config"]["calibration_seed"] != comparator["config"]["calibration_seed"]:
        raise ValueError("Paired evaluation arms have different calibration seeds")
    for field in PAIRED_METADATA + OPTIONAL_PAIRED_METADATA:
        if (field in anchor["meta"] or field in comparator["meta"]) and (
                anchor["meta"].get(field) != comparator["meta"].get(field)):
            raise ValueError(f"Paired evaluation arms disagree on {field}")
    a_source, b_source = anchor["meta"]["source_sha256"], comparator["meta"]["source_sha256"]
    for name in NUMERICAL_SOURCES:
        if name not in a_source or name not in b_source or a_source[name] != b_source[name]:
            raise ValueError(f"Paired arms use different numerical source {name}")
    for field in METRICS:
        a, b = anchor["trace"][0][field], comparator["trace"][0][field]
        if a is None and b is None:
            continue
        if a is None or b is None or not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"Paired evaluation arms have different initial {field}")


def saved_probes(trace, view_cap, original_cap):
    if not trace:
        return {}
    rows = {"endpoint": trace[-1]}
    for name, cap in (("view_cap", view_cap), ("original_cap", original_cap)):
        rows[name] = next(row for row in reversed(trace)
                          if max(row["forward_views"], row["backward_views"]) <= cap)
    fields = WORK + METRICS + PROJECTION
    return {name: {key: row[key] for key in fields if key in row} for name, row in rows.items()}


def grouped_stat(values, groups):
    """Take seed medians, slice medians within groups, then a median over groups."""
    by_slice = {sid: median(v) for sid, v in sorted(values.items()) if v}
    grouped = defaultdict(list)
    for sid, value in by_slice.items():
        grouped[groups[sid]].append(value)
    by_group = {group: median(v) for group, v in sorted(grouped.items())}
    return {
        "median_across_groups": median(by_group.values()) if by_group else None,
        "group_medians": by_group,
        "group_range": [min(by_group.values()), max(by_group.values())] if by_group else None,
        "median_across_slices": median(by_slice.values()) if by_slice else None,
        "slice_medians": by_slice,
        "seed_ranges_by_slice": {sid: [min(values[sid]), max(values[sid])] for sid in by_slice},
        "within_group": {group: {"slices": len(v), "median": median(v), "range": [min(v), max(v)]}
                         for group, v in sorted(grouped.items())},
        "paired_seed_counts_by_slice": {sid: len(values[sid]) for sid in by_slice},
        "groups": len(by_group), "slices": len(by_slice),
        "seed_pairs": sum(len(values[sid]) for sid in by_slice),
        "groups_favoring_sframes": sum(v < 0 for v in by_group.values()),
        "groups_tied": sum(v == 0 for v in by_group.values()),
        "slices_favoring_sframes": sum(v < 0 for v in by_slice.values()),
        "slices_tied": sum(v == 0 for v in by_slice.values()),
    }


def paired_summary(cases, comparator, budget, groups):
    values = {field: defaultdict(list) for field in METRICS}
    percentages = {field: defaultdict(list) for field in METRICS}
    count = 0
    for case in cases:
        arms = case["arms"]
        if budget not in arms[ANCHOR]["probes"] or budget not in arms[comparator]["probes"]:
            continue
        a, b = arms[ANCHOR]["probes"][budget], arms[comparator]["probes"][budget]
        count += 1
        for field in METRICS:
            if a[field] is None or b[field] is None:
                continue
            delta = a[field] - b[field]
            values[field][case["slice_id"]].append(delta)
            if b[field] > 0:
                percentages[field][case["slice_id"]].append(100 * delta / b[field])
    return {"seed_pairs": count, "metrics": {
        field: {"absolute_difference": grouped_stat(values[field], groups),
                "percent_difference": grouped_stat(percentages[field], groups)}
        for field in METRICS}}


def marginal_summary(cases, budget, groups):
    complete = [case for case in cases if all(budget in arm["probes"]
                                             for arm in case["arms"].values())]
    methods, slice_methods, group_details = {}, {}, {}
    for variant in VARIANTS:
        by_slice = defaultdict(list)
        for case in complete:
            by_slice[case["slice_id"]].append(case["arms"][variant]["probes"][budget])
        summary, slice_summary, details = {}, {}, {}
        for field in WORK + METRICS + PROJECTION:
            values = {sid: [row[field] for row in rows if row.get(field) is not None]
                      for sid, rows in by_slice.items()}
            stats = grouped_stat(values, groups)
            summary[field] = stats["median_across_groups"]
            slice_summary[field] = stats["median_across_slices"]
            details[field] = stats["group_medians"]
        methods[variant], slice_methods[variant], group_details[variant] = summary, slice_summary, details
    return {"complete_cases": len(complete), "slices": len({c["slice_id"] for c in complete}),
            "groups": len({groups[c["slice_id"]] for c in complete}),
            "median_of_group_medians": methods, "group_medians": group_details,
            "median_of_slice_medians": slice_methods}


def initial_factor_norm(meta):
    for name in ("initial_factor_frobenius_norm", "initial_factor_norm", "initial_joint_factor_norm"):
        value = meta.get(name)
        if value is not None:
            break
    else:
        init = meta.get("initialization", {})
        value = init.get("factor_frobenius_norm")
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError("Relative feasibility targets require the calibrated initial factor norm")
    return value


def load_targets(root):
    frozen = read_json(root / "targets.json")
    if frozen is None:
        return None
    if frozen.get("status") not in ("frozen", "selected") and not frozen.get("frozen_utc"):
        raise ValueError("Joint accuracy targets must be frozen before evaluation")
    if not frozen.get("normalization", "initial_factor_frobenius_norm").startswith(
            ("initial_factor_frobenius_norm", "initial_factor_norm")):
        raise ValueError("Unsupported raw-feasibility target normalization")
    names = set()
    for target in frozen["targets"]:
        if target["name"] in names:
            raise ValueError("Accuracy target names must be unique")
        names.add(target["name"])
        if not (0 < target["relative_repaired_loss"] < 1
                and target["max_relative_d_distance"] >= 0):
            raise ValueError("Invalid loss or feasibility target")
    return frozen


def first_target_hit(run, target, view_cap):
    trace = run["trace"]
    if not trace:
        return {"status": "incomplete", "reached": False}
    scale = initial_factor_norm(run["meta"])
    initial_loss = trace[0]["feasible_loss"]
    if initial_loss <= 0:
        raise ValueError("Relative loss targets require positive initial loss")
    within = [row for row in trace if max(row["forward_views"], row["backward_views"]) <= view_cap]
    chosen = next((row for row in within
                   if row["feasible_loss"] / initial_loss <= target["relative_repaired_loss"]
                   and row["d_distance"] / scale <= target["max_relative_d_distance"]), None)
    if chosen is None:
        return {"status": "right_censored", "reached": False, "view_cap": view_cap,
                "first_views": None, "lower_bound_views": view_cap,
                "initial_factor_frobenius_norm": scale}
    return {"status": "reached", "reached": True, "first_views": chosen["forward_views"],
            "iteration": chosen["iteration"], "gradient_calls": chosen["gradient_calls"],
            "relative_repaired_loss": chosen["feasible_loss"] / initial_loss,
            "relative_d_distance": chosen["d_distance"] / scale,
            "initial_factor_frobenius_norm": scale}


def target_summary(cases, target_name, groups, view_cap):
    methods = {}
    for variant in VARIANTS:
        by_slice = defaultdict(list)
        for case in cases:
            hit = case["arms"][variant].get("target_hits", {}).get(target_name)
            if hit and hit["status"] != "incomplete":
                by_slice[case["slice_id"]].append(hit)
        lowers, uppers, rates = {}, {}, {}
        for sid, hits in by_slice.items():
            lowers[sid] = [hit["first_views"] if hit["reached"] else view_cap for hit in hits]
            uppers[sid] = [hit["first_views"] if hit["reached"] else math.inf for hit in hits]
            rates[sid] = sum(hit["reached"] for hit in hits) / len(hits)
        lower = grouped_stat(lowers, groups)
        upper = grouped_stat(uppers, groups)
        group_rates = defaultdict(list)
        for sid, value in rates.items():
            group_rates[groups[sid]].append(value)
        group_rates = {group: sum(values) / len(values) for group, values in sorted(group_rates.items())}
        lo, hi = lower["median_across_groups"], upper["median_across_groups"]
        censored = hi is None or not math.isfinite(hi)
        methods[variant] = {
            "seed_cases": sum(len(hits) for hits in by_slice.values()),
            "reached_seed_cases": sum(hit["reached"] for hits in by_slice.values() for hit in hits),
            "group_attainment_fractions": group_rates,
            "mean_group_attainment": sum(group_rates.values()) / len(group_rates) if group_rates else None,
            "median_first_views": None if censored else hi,
            "median_first_views_lower_bound": lo,
            "median_first_views_right_censored": censored,
            "group_median_first_views": {
                group: None if not math.isfinite(value) else value
                for group, value in upper["group_medians"].items()},
            "group_median_first_views_lower_bound": lower["group_medians"],
        }
    comparisons = {}
    for comparator in VARIANTS:
        if comparator == ANCHOR:
            continue
        ratios, categories = defaultdict(list), Counter()
        for case in cases:
            a = case["arms"][ANCHOR].get("target_hits", {}).get(target_name)
            b = case["arms"][comparator].get("target_hits", {}).get(target_name)
            if not a or not b or "incomplete" in (a["status"], b["status"]):
                continue
            if a["reached"] and b["reached"]:
                categories["both_reached"] += 1
                if b["first_views"] > 0:
                    ratios[case["slice_id"]].append(a["first_views"] / b["first_views"])
            elif a["reached"]:
                categories["only_sframes_reached"] += 1
            elif b["reached"]:
                categories["only_comparator_reached"] += 1
            else:
                categories["neither_reached"] += 1
        comparisons[comparator] = {"attainment_pair_counts": dict(categories),
                                   "first_views_ratio_when_both_reached": grouped_stat(ratios, groups)}
    return {"methods": methods, "sframes_vs": comparisons,
            "censoring": "No hit within the budget is right censored; it is never treated as a hit at the cap. "
                         "Median-work lower bounds propagate censored intervals through seed, slice and group medians. "
                         "Work ratios condition explicitly on both members reaching the target."}


def analyze(root, design, view_cap):
    found = collect(root)
    slices = sorted(slice_id(value) for value in design["evaluation_slices"])
    seeds = sorted(set(map(int, design["evaluation_seeds"])))
    sizes = sorted(set(map(int, design["evaluation_training_views"])))
    groups = group_map(design)
    deterministic = set(design["deterministic_variants"])
    if ANCHOR in deterministic or deterministic - set(VARIANTS):
        raise ValueError("Invalid deterministic variants")
    expected = list(product(slices, seeds, sizes))
    physical = {(sid, seed, n, variant) for sid, seed, n in expected for variant in VARIANTS
                if variant not in deterministic or seed == seeds[0]}
    if set(found) - physical:
        raise ValueError("Saved evaluation arms include cases outside the frozen design")
    original_cap = int(design.get("original_evaluation_view_budget", 17280))
    targets = load_targets(root)
    cases = []
    for sid, seed, n in expected:
        present = {v: found.get((sid, seeds[0] if v in deterministic else seed, n, v)) for v in VARIANTS}
        complete = [run for run in present.values() if run and run["trace"]]
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
                "origin": run["origin"], "source_directory": run["source_directory"],
                "source_sha256": run["meta"]["source_sha256"], "optimizer": run["meta"]["optimizer"],
                "continuation": run["continuation"], "checkpoint_iteration": run["checkpoint_iteration"],
                "error": run["error"], "probes": saved_probes(run["trace"], view_cap, original_cap),
            }
            if targets:
                arms[variant]["target_hits"] = {target["name"]: first_target_hit(run, target, view_cap)
                                                for target in targets["targets"]}
        cases.append(dict(slice_id=sid, group=groups[sid], seed=seed, n_train=n, arms=arms))
    cohorts = {}
    for n in sizes:
        subset = [case for case in cases if case["n_train"] == n]
        cohorts[str(n)] = {
            name: {"matched_all_methods": marginal_summary(subset, name, groups),
                   "sframes_vs": {variant: paired_summary(subset, variant, name, groups)
                                  for variant in VARIANTS if variant != ANCHOR}}
            for name in ("original_cap", "view_cap", "endpoint")}
        if targets:
            cohorts[str(n)]["accuracy_targets"] = {
                target["name"]: target_summary(subset, target["name"], groups, view_cap)
                for target in targets["targets"]}
    campaign = read_json(root / "evaluation_progress.json", read_json(root / "progress.json", {}))
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "design": design,
        "budgets": {"forward_view_cap": view_cap, "backward_view_cap": view_cap,
                    "original_forward_view_cap": original_cap},
        "comparison": {
            "direction": "Minibatch SFRAMES minus comparator. Negative favors SFRAMES.",
            "pairing": "Same slice and training-view count with shared calibration; stochastic arms pair by oracle seed. "
                       "Each deterministic arm runs once per slice/size and is shared across paired seed cases.",
            "within_budget": "Last recorded iterate at or below the cap, without interpolation.",
            "endpoint": "Last observation, possibly after overshooting the nominal work cap.",
            "summary_estimate": "Median over paired seeds within slices, then slices within specimen groups, "
                                "then over specimen groups. Groups have equal weight.",
            "percentages": "Paired comparator value is the denominator; zero denominators excluded.",
            "scope": "Specimen-group summaries are descriptive; slices and oracle seeds are not independent specimens.",
            "source_provenance": "Each run retains source hashes; continued runs retain an exact validated native prefix "
                                 "and source checkpoint/trace/result hashes. Pairing requires identical instance data, "
                                 "calibration, initialization, constraints and numerical source hashes.",
        },
        "coverage": {
            "expected_physical_arms": len(physical), "expected_paired_seed_cases": len(expected),
            "evaluation_slices": len(slices), "specimen_groups": len(set(groups.values())),
            "slices_by_group": {group: sorted(sid for sid in slices if groups[sid] == group)
                                for group in sorted(set(groups.values()))},
            "case_counts_by_n_train": dict(Counter(case["n_train"] for case in cases)),
            "physical_arm_status_counts": dict(Counter(found[key]["status"] if key in found else "missing metadata"
                                                       for key in physical)),
            "physical_arm_origin_counts": dict(Counter(run["origin"] for run in found.values())),
            "native_iteration_observations": sum(len(run["trace"]) for run in found.values()),
            "campaign_phase": campaign.get("phase"), "campaign_status": campaign.get("status"),
            "campaign_failures": {key: value for key, value in campaign.get("runs", {}).items()
                                  if value.get("status") == "failed"},
        },
        "selection": read_json(root / "selection.json"), "targets": targets,
        "provenance_anchors": {name: sha256(root / name) for name in
                               ("input_anchor.json", "audit_provenance.py", "manifest.json", "evaluation_plan.json")
                               if (root / name).is_file()},
        "cohorts": cohorts, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--forward-cap", type=int)
    parser.add_argument("--no-plots", action="store_true", help="Accepted for compatibility; this module creates no figures")
    args = parser.parse_args()
    root = args.root.resolve()
    design = read_json(root / "manifest.json")["design"]
    cap = args.forward_cap if args.forward_cap is not None else design["evaluation_view_budget"]
    if cap <= 0:
        parser.error("The work cap must be positive")
    summary = analyze(root, design, cap)
    output = (args.output or root / "reports").resolve()
    output.mkdir(parents=True, exist_ok=True)
    pending = output / "summary.json.tmp"
    pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pending.replace(output / "summary.json")
    print(json.dumps({"output": str(output), "coverage": summary["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
