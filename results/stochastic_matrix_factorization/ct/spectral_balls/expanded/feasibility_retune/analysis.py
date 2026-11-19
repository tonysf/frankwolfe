"""Compare feasibility-tuned SFRAMES with the published equal-view runs.

Only saved native iterates are used. The original evaluation cohort and target
definitions are frozen inputs. A new run is compared only with the corresponding
slice, training-view set, calibration and sampling-seed label.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
from itertools import product
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "convincing_4x_recovery"
TUNED = "tuned_sframes"
COMPARATORS = ("sframes", "prox_sarah")
METHODS = (TUNED, *COMPARATORS)
PCT_FIELDS = {
    "raw_loss_pct": "raw_loss",
    "repaired_loss_pct": "feasible_loss",
    "held_out_loss_pct": "validation_loss",
    "reference_error_pct": "reference_relative_error",
}


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_key(config):
    return config["slice_id"], config["n_train"], config["seed"]


def read_run(folder, legacy, auditor, baseline, cap, old_arm=None):
    """Audit completion, numerical pairing sources and native observations."""
    meta = read(folder / "metadata.json")
    result = read(folder / "result.json")
    trace = read(folder / "trace.json")
    config = meta["config"]
    require(result["status"] == "completed" and trace
            and result["iteration"] == trace[-1]["iteration"]
            and result["final"] == trace[-1]
            and all(result.get(field) == value for field, value in meta.items()),
            f"Completed result, metadata and native observations disagree in {folder}")
    require(config["phase"] == "evaluation" and config["record_every_iteration"] is True
            and config["view_budget"] == cap and config["run_id"] == folder.name,
            f"Run differs from the dense evaluation protocol in {folder}")
    legacy.validate_trace(trace, folder)
    require(trace[-1]["forward_views"] == trace[-1]["backward_views"] == cap,
            f"Expected the exact equal-view cap of {cap} per direction in {folder}")
    require(all(row["reference_relative_error"] is not None for row in trace),
            f"The evaluation reference is missing in {folder}")
    if config["method"] == "sframes":
        require(config["batch"] == 64
                and all(row["forward_views"] == 64 * row["iteration"]
                        and row["gradient_calls"] == row["iteration"] for row in trace),
                f"Expected one batch-64 joint gradient per SFRAMES update in {folder}")
    else:
        require(config["method"] == "prox_sarah"
                and all(row["d_distance"] == 0 for row in trace),
                f"Expected an orthant-feasible ProxSARAH reference in {folder}")

    canonical = {name: auditor.canonical_source_hash(baseline, name, meta["source_sha256"][name])
                 for name in legacy.NUMERICAL_SOURCES}
    if old_arm is not None:
        auditor.validate_evaluation_metadata(baseline, meta)
        legacy.validate_continuation(meta, trace, folder)
        legacy.validate_continuation({"continuation": meta.get("imported_checkpoint")}, trace, folder)
        require(old_arm["status"] == "completed"
                and old_arm["source_directory"] == str(folder.resolve())
                and old_arm["source_sha256"] == meta["source_sha256"]
                and old_arm["canonical_numerical_source_sha256"] == canonical
                and old_arm["optimizer"] == meta["optimizer"]
                and old_arm["probes"]["view_cap"] == trace[-1],
                f"The saved reference disagrees with its published audit in {folder}")
    else:
        require(not meta.get("continuation") and not meta.get("imported_checkpoint"),
                f"Feasibility tuning requires a fresh evaluation from the common initialization: {folder}")

    return dict(config=config, meta=meta, trace=trace, canonical_numerical_source_sha256=canonical)


def record_run(run, folder, legacy, targets, cap):
    trace = run["trace"]
    scale = legacy.initial_factor_norm(run["meta"])
    initial_loss = trace[0]["feasible_loss"]
    require(initial_loss > 0, f"Cannot normalize loss by {initial_loss} in {folder}")
    endpoint = dict(trace[-1], relative_raw_loss=trace[-1]["raw_loss"] / initial_loss,
                    relative_repaired_loss=trace[-1]["feasible_loss"] / initial_loss,
                    relative_d_distance=trace[-1]["d_distance"] / scale)
    joint = {}
    for target in targets:
        loss, feas = target["relative_repaired_loss"], target["max_relative_d_distance"]
        hit = legacy.first_target_hit(run, target, cap)
        joint[f"{loss:g}:{feas:g}"] = dict(
            hit, endpoint_hit=endpoint["relative_repaired_loss"] <= loss
            and endpoint["relative_d_distance"] <= feas,
        )
    return dict(source_directory=str(folder.resolve()), run_id=run["config"]["run_id"],
                trace_sha256=legacy.sha256(folder / "trace.json"),
                result_sha256=legacy.sha256(folder / "result.json"),
                initial_factor_frobenius_norm=scale, iteration_zero=trace[0], endpoint=endpoint,
                view_budget_slack=cap - endpoint["forward_views"], observations_count=len(trace),
                joint_targets=joint)


def grouped(cases, groups, legacy, get_value):
    values = defaultdict(list)
    for case in cases:
        value = get_value(case)
        if value is not None:
            values[case["slice_id"]].append(value)
    return legacy.grouped_stat(values, groups)


def balanced_rate(cases, get_value):
    """Average over seeds, then slices, then specimen groups."""
    seeds = defaultdict(list)
    for case in cases:
        seeds[case["group"], case["slice_id"]].append(bool(get_value(case)))
    groups = defaultdict(list)
    for (group, _), values in seeds.items():
        groups[group].append(sum(values) / len(values))
    rates = {group: sum(values) / len(values) for group, values in sorted(groups.items())}
    return sum(rates.values()) / len(rates), rates


def joint_summary(cases, groups, legacy, label, target_key):
    def observation(case):
        return case["runs"][label]["joint_targets"][target_key]

    endpoint, endpoint_groups = balanced_rate(cases, lambda c: observation(c)["endpoint_hit"])
    first_hit, first_hit_groups = balanced_rate(cases, lambda c: observation(c)["reached"])
    return dict(cases=len(cases), endpoint_hits=sum(observation(c)["endpoint_hit"] for c in cases),
                first_hits=sum(observation(c)["reached"] for c in cases),
                right_censored=sum(not observation(c)["reached"] for c in cases),
                endpoint_group_balanced_fraction=endpoint, endpoint_group_fractions=endpoint_groups,
                first_hit_group_balanced_fraction=first_hit, first_hit_group_fractions=first_hit_groups,
                first_views_when_reached=grouped(cases, groups, legacy, lambda c: observation(c)["first_views"]))


def compare(cases, groups, legacy, label, target_keys):
    def delta(case, metric, pct):
        a = case["runs"][TUNED]["endpoint"][metric]
        b = case["runs"][label]["endpoint"][metric]
        return 100 * (a - b) / b if pct and b else (None if pct else a - b)

    result = {name: grouped(cases, groups, legacy, lambda c: delta(c, metric, True))
              for name, metric in PCT_FIELDS.items()}
    result["absolute_differences"] = {
        metric: grouped(cases, groups, legacy, lambda c: delta(c, metric, False))
        for metric in (*PCT_FIELDS.values(), "relative_d_distance", "relative_repaired_loss")
    }
    joint = {}
    for key in target_keys:
        counts, ratios = Counter(), defaultdict(list)
        for case in cases:
            a, b = (case["runs"][variant]["joint_targets"][key] for variant in (TUNED, label))
            counts[(a["reached"], b["reached"])] += 1
            if a["reached"] and b["reached"]:
                ratios[case["slice_id"]].append(a["first_views"] / b["first_views"])
        joint[key] = dict(both_reached=counts[True, True], only_tuned_reached=counts[True, False],
                          only_reference_reached=counts[False, True], neither_reached=counts[False, False],
                          first_views_ratio_when_both_reached=legacy.grouped_stat(ratios, groups))
    result["joint_targets"] = joint
    return result


def analyze(root=ROOT, baseline=BASELINE):
    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    legacy = load_module(baseline / "analyze.py", "legacy_ct_analysis")
    auditor = load_module(baseline / "audit_provenance.py", "legacy_ct_provenance")
    reference_summary = read(baseline / "reports" / "summary.json")
    cohort = read(baseline / "cohort.json")
    cap = cohort["evaluation_view_budget"]
    require(reference_summary["budgets"]["forward_view_cap"] == cap
            and reference_summary["budgets"]["backward_view_cap"] == cap,
            "The published and frozen evaluation budgets disagree")
    groups = legacy.group_map(cohort)
    targets = legacy.load_targets(baseline)["targets"]
    target_keys = [f"{t['relative_repaired_loss']:g}:{t['max_relative_d_distance']:g}" for t in targets]
    require(set(target_keys) == {"0.3:0.01", "0.3:0.001", "0.4:0.01", "0.4:0.001"},
            "The four prespecified loss/feasibility targets changed")
    expected = set(product(cohort["evaluation_slices"], cohort["evaluation_training_views"],
                           cohort["evaluation_seeds"]))
    legacy_cases = {case_key(case): case for case in reference_summary["cases"]}
    require(set(legacy_cases) == expected and len(legacy_cases) == len(reference_summary["cases"]),
            "Saved reference cases do not match the frozen evaluation design")
    protocol_sha = legacy.sha256(root / "protocol.json")
    selection_sha = legacy.sha256(root / "selection.json")
    selection = read(root / "selection.json")
    require(selection["status"] == "selected" and selection["protocol_sha256"] == protocol_sha,
            "Selection does not link to the frozen protocol")
    selected = selection["selected"]
    evaluation = read(root / "evaluation_plan.json")
    plan = evaluation["runs"]
    require(evaluation["selection_sha256"] == selection_sha
            and Path(evaluation["comparison_root"]).resolve() == baseline
            and evaluation["expected_runs"] == len(plan),
            "The evaluation plan does not link to the frozen selection and references")
    for label, field in (("sframes", "current_sframes_run_ids"), ("prox_sarah", "prox_sarah_run_ids")):
        ids = {case["arms"][label]["run_id"] for case in legacy_cases.values()}
        require(set(evaluation[field]) == ids and len(evaluation[field]) == len(ids),
                f"The planned {label} references differ from the saved cases")
    planned = {case_key(config): config for config in plan}
    require(set(planned) == expected and len(planned) == len(plan),
            "New evaluation plan does not match the frozen evaluation design")
    found = {}
    for path in sorted((root / "runs").glob("*/metadata.json")):
        config = read(path)["config"]
        if config["phase"] != "evaluation":
            continue
        key = case_key(config)
        require(key not in found and config == planned.get(key),
                f"Unplanned or duplicate feasibility-tuned evaluation: {path}")
        found[key] = path.parent
    require(set(found) == expected, f"Expected {len(expected)} tuned cases, found {len(found)}")

    cases = []
    for key in sorted(expected):
        sid, n_train, seed = key
        tuned_folder = found[key]
        run = read_run(tuned_folder, legacy, auditor, baseline, cap)
        config, meta = run["config"], run["meta"]
        require(config["variant"] == config["method"] == "sframes"
                and config["beta_scale"] == selected["beta_scale"] and config["rate"] == selected["gamma0"],
                f"New evaluation does not use the frozen selected setting: {tuned_folder}")
        require(meta["optimizer"] == {"batch": 64, "gamma0": selected["gamma0"], "rho0": 1.0,
                                     "mu0": selected["beta_scale"] / max(meta["curvature"])},
                f"Realized optimizer parameters disagree with the selected setting: {tuned_folder}")
        records = {TUNED: record_run(run, tuned_folder, legacy, targets, cap)}
        old_case = legacy_cases[key]
        require(old_case["group"] == groups[sid], f"Specimen-group assignment differs for {sid}")
        for label in COMPARATORS:
            arm = old_case["arms"][label]
            folder = baseline / "runs" / arm["run_id"]
            reference = read_run(folder, legacy, auditor, baseline, cap, arm)
            require(reference["config"]["variant"] == label and case_key(reference["config"]) == key,
                    f"Unexpected reference variant or paired case in {folder}")
            for field in ("grid", "rank", "batch"):
                require(config[field] == reference["config"][field], f"Paired cases differ on {field}: {folder}")
            for field in ("astra", "numpy"):
                require(meta[field] == reference["meta"][field], f"Paired numerical environments differ: {folder}")
            legacy.verify_pair(run, reference)
            records[label] = record_run(reference, folder, legacy, targets, cap)
            for target, target_key in zip(targets, target_keys):
                require(arm["target_hits"][target["name"]]["first_views"]
                        == records[label]["joint_targets"][target_key]["first_views"],
                        f"Reference first-hit work changed in {folder}")
        cases.append(dict(slice_id=sid, n_train=n_train, seed=seed, group=groups[sid], runs=records))

    by_n = {}
    for n in cohort["evaluation_training_views"]:
        subset = [c for c in cases if c["n_train"] == n]
        methods = {}
        for label in METHODS:
            endpoint_fields = (*legacy.WORK, *legacy.METRICS, "relative_raw_loss",
                               "relative_repaired_loss", "relative_d_distance")
            endpoint = {metric: grouped(subset, groups, legacy, lambda c: c["runs"][label]["endpoint"][metric])
                        for metric in endpoint_fields}
            joints = {key: joint_summary(subset, groups, legacy, label, key) for key in target_keys}
            methods[label] = dict(endpoint=endpoint, joint_targets=joints)
        by_n[str(n)] = dict(methods=methods, cases=len(subset), pairs={
            label: compare(subset, groups, legacy, label, target_keys) for label in COMPARATORS
        })

    return dict(schema_version=1, status="completed_validated",
                generated_utc=datetime.now(timezone.utc).isoformat(), selection=selection,
                view_budget_per_direction=cap, development_view_budget_per_direction=cohort["development_view_budget"],
                evaluation_cases=len(cases), specimen_groups=len(set(groups.values())), cohort=cohort, targets=targets,
                by_n=by_n, cases=cases,
                aggregation="Endpoint and paired medians over seeds, then slices, then specimen groups. "
                            "Attainment rates use means at the same levels. Work ratios condition on common hits.",
                provenance=dict(
                    old_report_summary_sha256=legacy.sha256(baseline / "reports" / "summary.json"),
                    cohort_sha256=legacy.sha256(baseline / "cohort.json"),
                    targets_sha256=legacy.sha256(baseline / "targets.json"),
                    selection_sha256=selection_sha,
                    protocol_sha256=protocol_sha,
                    evaluation_plan_sha256=legacy.sha256(root / "evaluation_plan.json"),
                    analysis_sha256=legacy.sha256(Path(__file__)),
                ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    output = args.output or args.root / "summary.json"
    if output.is_file():
        previous = read(output)
        # A restart after report publication must preserve the audited summary.
        if {k: v for k, v in result.items() if k != "generated_utc"} == {
                k: v for k, v in previous.items() if k != "generated_utc"}:
            print(f"Revalidated unchanged paired comparison: {output}")
            return
        require(not (args.root / "report_update.json").exists(),
                "A published report already anchors a different comparison")
    pending = output.with_name(output.name + ".tmp")
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.replace(output)
    print(f"Validated {result['evaluation_cases']} cases per method at {result['view_budget_per_direction']:,} "
          f"views per direction: {output}")


if __name__ == "__main__":
    main()
