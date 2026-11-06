#!/usr/bin/env python3
"""Validate the complete Ruche calibration; freeze selection only on request.

No optimizer runs, remote access, or repository edits. --validate-only does not
rank candidates. --freeze requires a fresh output directory and a fully valid
eight-channel inventory. Analysis conventions were confirmed before outcomes
were inspected: nested channel means, channel medians, conservative missing
metric sentinels, and no deletion of numerical failures.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

COMMIT = "c7a8c9789b7814ede1c95e53eef13bfe1dd58c63"
METHODS = ("printed-adafgd", "frames")
ARRAYS = {0: "2014220", **{channel: "2014222" for channel in range(1, 8)}}
METRICS = ("process_fidelity", "rmse", "normalized_tp_violation")
CONVENTIONS = {
    "id": "gate_first_common_physical_metrics_v1_nested_missing_conventions_20261005",
    "agreement": "Root confirmed these conventions before inspecting candidate outcomes.",
    "experimental_unit": "channel; eight channels receive equal weight",
    "within_channel": "Mean sampling streams within initialization, then mean initializations; this plan has one initialization. Printed deterministic copies are counted once.",
    "scores": "Gate and gate-component scores: nested means across sampling, initialization, channel. Physical tie breakers: median across eight channel means after the same within-channel nesting.",
    "missing_metrics": "Undefined/nonfinite fidelity ranks as0; undefined/nonfinite full-design RMSE or normalized TP ranks as positive infinity. No outcome is omitted. Null with ranking_sentinel='+infinity' encodes the latter in JSON.",
    "failed_methods": "Gate remains false. Preserve finite recorded gate-component/physical diagnostics; replace only undefined secondary metrics by conservative sentinels.",
    "metric_source": "Archived gate.metric_values: process fidelity, full-design RMSE, normalized TP. Penalized objectives and timing never enter ranking.",
    "order": ["highest nested gate pass rate", "highest nested mean gate components passed",
              "highest median channel-mean fidelity", "lowest median channel-mean full-design RMSE",
              "lowest median channel-mean normalized TP", "lexicographically smallest config ID"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_launcher(path):
    spec = importlib.util.spec_from_file_location("frozen_calibration_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.COMMIT == COMMIT, "Launcher source pin mismatch")
    return module


def source_hashes(repo):
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   text=True).strip() == COMMIT, "Local source pin mismatch")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=repo, text=True).strip(), "Local source is dirty")
    sys.path.insert(0, str(repo))
    from paper.experiments.qpt_quiroga_direct_comparison import _SOURCE_PATHS
    return {name: digest(repo / name) for name in _SOURCE_PATHS}


def native_outcome(trial, metadata, method):
    comparison = metadata["comparison"]
    endpoint = metadata["methods"][method]
    gate = comparison["method_gate"][method]
    candidate_field = "printed_config_id" if method == "printed-adafgd" else "frames_config_id"
    sample = None if method == "printed-adafgd" else trial["sampling_index"]
    return {
        "method": method, "config_id": trial[candidate_field], "channel": trial["channel"],
        "initialization_index": trial["initialization_index"], "sampling_index": sample,
        "status": endpoint["status"], "failure_reason": endpoint.get("failure_reason"),
        "gate_passed": bool(gate["passed"]), "gate_components": gate["component_passes"],
        "metrics": gate["metric_values"], "rmse_source": gate["rmse_source"],
        "data_instance_sha256": comparison["data_instance_sha256"],
        "paired_instance_sha256": comparison["paired_instance_sha256"],
        "initial_factor_sha256": metadata["initial_factor_sha256"],
        "active_rows_sha256": metadata["active_rows_sha256"],
        "active_observations_sha256": metadata["active_observations_sha256"],
        "full_observations_sha256": comparison["full_observations_sha256"],
        "truth_factor_sha256": metadata["truth_factor_sha256"],
        "endpoint_without_timings": {k: v for k, v in endpoint.items() if "seconds" not in k},
        "endpoint_diagnostics": comparison["endpoint_diagnostics"][method],
        "source_trial_ids": [trial["trial_id"]],
    }


def deduplicate(records, plan):
    unique = {}
    for outcome in records:
        key = tuple(outcome[field] for field in (
            "method", "config_id", "channel", "initialization_index", "sampling_index"))
        comparable = {k: v for k, v in outcome.items() if k != "source_trial_ids"}
        if key in unique:
            previous = {k: v for k, v in unique[key].items() if k != "source_trial_ids"}
            require(canonical(comparable) == canonical(previous),
                    f"Repeated native endpoint/status/factor disagrees: {key}")
            unique[key]["source_trial_ids"].extend(outcome["source_trial_ids"])
        else:
            unique[key] = dict(outcome, source_trial_ids=list(outcome["source_trial_ids"]))
    for method, candidates, samples in (
        ("printed-adafgd", plan["printed_candidates"], [None]),
        ("frames", plan["frames_candidates"], plan["sampling_indices"])):
        expected = {(method, c["id"], channel, init, sample)
                    for c in candidates for channel in plan["channels"]
                    for init in plan["initialization_indices"] for sample in samples}
        require({key for key in unique if key[0] == method} == expected,
                f"Native candidate coverage is incomplete: {method}")
    identities = {}
    for outcome in unique.values():
        identity = (outcome["data_instance_sha256"], outcome["paired_instance_sha256"],
                    outcome["initial_factor_sha256"], outcome["active_rows_sha256"],
                    outcome["active_observations_sha256"], outcome["full_observations_sha256"],
                    outcome["truth_factor_sha256"])
        key = (outcome["channel"], outcome["initialization_index"])
        require(key not in identities or identities[key] == identity,
                f"Candidates do not share data/U0: {key}")
        identities[key] = identity
    return sorted(unique.values(), key=lambda row: (row["method"], row["config_id"],
                   row["channel"], row["initialization_index"], row["sampling_index"] or 0))


def validate_campaign(plan_path, results_root, repo, launcher_path):
    plan = load_json(plan_path)
    launcher = load_launcher(launcher_path)
    expected = launcher.make_inventory(plan)
    require(plan["source"]["commit"] == COMMIT, "Unexpected calibration source")
    require(plan["selection_rule_id"] == "gate_first_common_physical_metrics_v1",
            "Unexpected selection rule")
    parent = load_json(Path(plan["parent_plan"]))
    require(parent["candidate_identity"]["selection_rule_id"] == plan["selection_rule_id"],
            "Original selection rule differs")
    require(parent["candidate_identity"]["candidate_grid_id"] == plan["candidate_grid_id"],
            "Original candidate grid differs")
    plan_hash, wrapper_hash = digest(plan_path), digest(launcher_path)
    pinned_hashes = source_hashes(repo)
    artifacts, records, inventory_evidence = [], [], []
    for channel in plan["channels"]:
        array = ARRAYS[channel]
        channel_dir = results_root / f"calibration-{array}" / f"channel-{channel}"
        inventory_path = channel_dir / "inventory.json"
        inventory = load_json(inventory_path)
        require(inventory["channel"] == channel and inventory["git_commit"] == COMMIT,
                f"Inventory provenance mismatch: {channel}")
        require(inventory["slurm_array_job_id"] == array
                and inventory["slurm_array_task_id"] == str(channel), "Wrong array/task")
        require(inventory["plan_sha256"] == plan_hash
                and inventory["wrapper_sha256"] == wrapper_hash, "Plan/wrapper hash mismatch")
        require(digest(channel_dir / "calibration_plan.json") == plan_hash,
                "Archived plan bytes changed")
        require(inventory["status"] in {"all_trials_archived", "all_trials_archived_with_method_failures"},
                f"Incomplete channel inventory: {channel}")
        channel_expected = [t for t in expected if t["channel"] == channel]
        actual = {t["trial_id"]: t for t in inventory["trials"]}
        require(len(inventory["trials"]) == len(actual) == inventory["expected_trials"] == 44,
                "Missing/duplicate channel trials")
        require(set(actual) == {t["trial_id"] for t in channel_expected}, "Unexpected trial IDs")
        require({p.name for p in channel_dir.iterdir() if p.is_dir()} == set(actual),
                "Missing/extra trial directories")
        observed_counts = collections.Counter()
        for trial in channel_expected:
            entry = actual[trial["trial_id"]]
            for key, value in trial.items():
                require(entry.get(key) == value, f"Inventory candidate/seed mismatch: {trial['trial_id']}/{key}")
            out = channel_dir / trial["trial_id"]
            status_path = out / "execution_status.json"
            require(digest(status_path) == entry["status_record_sha256"], "Execution-status hash mismatch")
            status = load_json(status_path)
            require(status["trial_id"] == trial["trial_id"], "Status trial identity mismatch")
            require(status["status"] == entry["execution_status"]
                    and status["runner_exit_code"] == entry["runner_exit_code"], "Status/exit mismatch")
            require(status["status"] in {"complete", "method_failure_archived"}, "Incomplete trial")
            require(status["command"] == entry["command"], "Recorded command mismatch")
            remote_out = Path(plan["output_root"]) / f"calibration-{array}" / f"channel-{channel}" / trial["trial_id"]
            require(status["command"][1:] == launcher.command_for(plan, trial, remote_out)[1:],
                    "Executed command differs from the frozen candidate/flags")
            require(set(status["artifacts"]) == {"pair.json", "pair.npz", "stdout.log", "stderr.log"},
                    "Incomplete archived evidence")
            checksums = {}
            for name, evidence in status["artifacts"].items():
                path = out / name
                require(path.stat().st_size == evidence["bytes"]
                        and digest(path) == evidence["sha256"], f"Artifact checksum mismatch: {path}")
                checksums[name] = evidence
            validation = launcher.validate_archive(plan, trial, out, status["runner_exit_code"])
            require(validation == status["archive_validation"], "Saved/recomputed archive validation differs")
            metadata = load_json(out / "pair.json")
            require(metadata["provenance"]["source_sha256"] == pinned_hashes,
                    "Archive source-file hashes do not match pinned code")
            require(metadata["source_sha256"] == pinned_hashes, "Top-level source hashes differ")
            observed_counts[status["status"]] += 1
            artifacts.append({"trial_id": trial["trial_id"], "channel": channel, "array_id": array,
                              "status": status["status"], "runner_exit_code": status["runner_exit_code"],
                              "status_record_sha256": digest(status_path), "artifacts": checksums})
            records.extend(native_outcome(trial, metadata, method) for method in METHODS)
        require(dict(observed_counts) == inventory["trial_status_counts"], "Inventory status counts differ")
        expected_exit = 2 if observed_counts["method_failure_archived"] else 0
        require(inventory["scheduler_exit_code"] == expected_exit, "Unexpected inventory scheduler exit")
        inventory_evidence.append({"channel": channel, "array_id": array,
                                   "inventory_sha256": digest(inventory_path),
                                   "status_counts": dict(observed_counts)})
    outcomes = deduplicate(records, plan)
    report = {"schema_version": 1, "status": "fully_valid", "source_commit": COMMIT,
              "plan_sha256": plan_hash, "launcher_sha256": wrapper_hash,
              "source_sha256": pinned_hashes, "channel_count": 8,
              "paired_trial_count": len(artifacts), "unique_native_outcome_count": len(outcomes),
              "native_status_counts": {method: dict(collections.Counter(
                  row["status"] for row in outcomes if row["method"] == method)) for method in METHODS},
              "inventories": inventory_evidence, "trial_evidence": artifacts}
    return plan, report, outcomes


def rank_value(value, metric):
    finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return float(value) if finite else (0.0 if metric == "process_fidelity" else math.inf)


def mean(values):
    values = list(values)
    require(bool(values), "Empty aggregation group")
    return math.inf if any(math.isinf(v) for v in values) else math.fsum(v / len(values) for v in values)


def median(values):
    values = sorted(values)
    require(bool(values), "Empty median")
    half = len(values) // 2
    return values[half] if len(values) % 2 else mean([values[half - 1], values[half]])


def encode_rank(value):
    return {"value": None, "ranking_sentinel": "+infinity"} if math.isinf(value) else {
        "value": value, "ranking_sentinel": None}


def candidate_summary(outcomes, plan, method, candidate):
    rows = [row for row in outcomes if row["method"] == method and row["config_id"] == candidate["id"]]
    channels, channel_numbers = [], []
    for channel in plan["channels"]:
        initializations = []
        for initialization in plan["initialization_indices"]:
            sample_rows = [r for r in rows if r["channel"] == channel
                           and r["initialization_index"] == initialization]
            expected = 1 if method == "printed-adafgd" else len(plan["sampling_indices"])
            require(len(sample_rows) == expected, "Missing native sample during aggregation")
            initializations.append({
                "gate": mean(float(r["gate_passed"]) for r in sample_rows),
                "components": mean(sum(bool(v) for v in r["gate_components"].values()) for r in sample_rows),
                **{m: mean(rank_value(r["metrics"][m], m) for r in sample_rows) for m in METRICS}})
        numbers = {key: mean(row[key] for row in initializations) for key in ("gate", "components", *METRICS)}
        channel_numbers.append(numbers)
        channels.append({"channel": channel, "gate_pass_rate": numbers["gate"],
                         "mean_gate_components_passed": numbers["components"],
                         "physical_ranking_means": {m: encode_rank(numbers[m]) for m in METRICS}})
    gate = mean(row["gate"] for row in channel_numbers)
    components = mean(row["components"] for row in channel_numbers)
    medians = {m: median(row[m] for row in channel_numbers) for m in METRICS}
    rank_key = (-gate, -components, -medians["process_fidelity"], medians["rmse"],
                medians["normalized_tp_violation"], candidate["id"])
    result = {"method": method, "config_id": candidate["id"], "candidate": candidate,
              "unique_native_outcomes": len(rows), "channel_count": len(channels),
              "status_counts": dict(collections.Counter(row["status"] for row in rows)),
              "gate_pass_rate": gate, "mean_gate_components_passed": components,
              "physical_ranking_channel_medians": {m: encode_rank(medians[m]) for m in METRICS},
              "undefined_metric_counts": {m: sum(not isinstance(r["metrics"][m], (int, float))
                   or not math.isfinite(r["metrics"][m]) for r in rows) for m in METRICS},
              "channels": channels, "outcomes": rows}
    return rank_key, result


def select_candidates(outcomes, plan):
    summary, selected = {}, {}
    for method, candidates in (("printed-adafgd", plan["printed_candidates"]),
                               ("frames", plan["frames_candidates"])):
        ranked = sorted((candidate_summary(outcomes, plan, method, c) for c in candidates),
                        key=lambda item: item[0])
        for position, (_, row) in enumerate(ranked, 1):
            row["rank"] = position
        summary[method] = [row for _, row in ranked]
        selected[method] = ranked[0][1]["candidate"]
    return {"analysis_conventions": CONVENTIONS, "candidates": summary}, selected


def freeze_selection(out, plan, report, outcomes):
    require(report["status"] == "fully_valid" and report["paired_trial_count"] == 352,
            "Cannot freeze incomplete calibration")
    require(not out.exists(), "Refusing to overwrite an analysis directory")
    summary, selected = select_candidates(outcomes, plan)
    boundaries = {}
    for method, candidates, fields in (
        ("printed-adafgd", plan["printed_candidates"], ("alpha", "eta_scale")),
        ("frames", plan["frames_candidates"], ("alpha0",))):
        boundaries[method] = {
            field: {"selected": selected[method][field],
                    "grid_min": min(c[field] for c in candidates),
                    "grid_max": max(c[field] for c in candidates),
                    "at_lower_boundary": selected[method][field] == min(c[field] for c in candidates),
                    "at_upper_boundary": selected[method][field] == max(c[field] for c in candidates)}
            for field in fields}
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    manifest = {"schema_version": 1, "kind": "qpt_method_native_selection_manifest",
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "regime_id": "n2_full_128", "source_commit": COMMIT,
                "campaign_id": plan["campaign_id"], "candidate_grid_id": plan["candidate_grid_id"],
                "selection_rule_id": plan["selection_rule_id"], "analysis_conventions": CONVENTIONS,
                "plan_sha256": report["plan_sha256"],
                "analyzer_sha256": digest(__file__), "launcher_sha256": report["launcher_sha256"],
                "printed_config_id": selected["printed-adafgd"]["id"],
                "frames_config_id": selected["frames"]["id"],
                "selected_candidates": selected, "shared_calibration_configuration": plan["shared"],
                "selected_grid_boundaries": boundaries,
                "candidate_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "candidate_outcomes_and_ranking": summary["candidates"],
                "validation_evidence": report,
                "qualification": "Best within this predeclared calibration grid, not a claim of recovery or a held-out comparison. No validation/held-out outcomes used."}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    report_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    out.mkdir()
    for name, data in (("validation_report.json", report_bytes), ("candidate_summary.json", summary_bytes),
                       ("selection_manifest.json", manifest_bytes)):
        with (out / name).open("xb") as stream:
            stream.write(data)
    return {"selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "printed_config_id": manifest["printed_config_id"], "frames_config_id": manifest["frames_config_id"],
            "output_directory": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/tony/frankwolfe"))
    parser.add_argument("--launcher", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.freeze and args.out is None:
        parser.error("--freeze requires --out pointing to a fresh directory")
    plan, report, outcomes = validate_campaign(args.plan, args.results_root, args.repo, args.launcher)
    if args.validate_only:
        print(json.dumps({key: report[key] for key in ("status", "source_commit", "channel_count",
                          "paired_trial_count", "unique_native_outcome_count", "native_status_counts")},
                         sort_keys=True))
    else:
        print(json.dumps(freeze_selection(args.out, plan, report, outcomes), sort_keys=True))


if __name__ == "__main__":
    main()
