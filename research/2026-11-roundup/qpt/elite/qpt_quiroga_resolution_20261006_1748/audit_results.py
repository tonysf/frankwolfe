#!/usr/bin/env python3
"""Read-only audit of diagnostic archives; never optimize, select, or rerank.

The only write is a fresh output JSON. Partial mode reports missing trials
explicitly. Full mode requires all frozen IDs and both completed channel
inventories, including every result checksum. Archived failed status remains
failed even when a retained checkpoint meets the physical thresholds.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("diagnostic_runner", HERE / "run_diagnostics.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
require = runner.require


def array_hash(array, np, *, shape_header=True):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    if shape_header:
        digest.update(str((array.shape, str(array.dtype))).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def local_sources(repo):
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    require(head == runner.COMMIT and not dirty, "Local source is not the clean pinned checkout")
    return {name: runner.file_hash(repo / name) for name in runner.SOURCE_PATHS}


def expected_campaign_configuration(plan, trial):
    requested = runner.run_configuration(plan, trial)
    expected = {key: value for key, value in requested.items()
                if key not in {"show_progress", "tp_weight"}}
    expected["tp_weight_lambda"] = requested["tp_weight"]
    expected.update({
        "frames_beta": requested["frames_beta0"],
        "frames_initial_tp_weight_per_row_alpha": 1.0 / (2.0 * requested["frames_beta0"]),
        "frames_initial_tp_weight_lambda": 128 / (2.0 * requested["frames_beta0"]),
        "frames_penalty_mode": "independent_schedule",
    })
    for key in ("rho_schedule", "step_size_schedule", "frames_smoothing_schedule"):
        expected[key] = {"type": "PowerSchedule", **expected[key]}
    return expected


def same_number(actual, expected, np, label, *, atol=1e-13):
    if expected is None:
        require(actual is None, label + " must be undefined")
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual)
                and np.isclose(actual, expected, rtol=1e-10, atol=atol),
                label + " differs")


def first_physical_gate(arrays, prefix, gate, np):
    metrics = {key: arrays[prefix + "_" + key] for key in runner.PHYSICAL_METRICS}
    steps = arrays[prefix + "_checkpoint_steps"]
    rows = arrays[prefix + "_cumulative_data_row_accesses"]
    for index, step in enumerate(steps):
        point = {key: float(values[index]) for key, values in metrics.items()}
        if runner.gate_for("complete", point, gate)["pass"]:
            return {"step": int(step), "charged_rows": int(rows[index]),
                    "charged_row_epochs": float(rows[index] / 128),
                    "checkpoint_metrics": point}
    return None


def audit_case(json_path, trial, plan, provenance, runtime):
    np, Data, Design, evaluate = runtime
    npz_path = json_path.with_suffix(".npz")
    require(npz_path.is_file(), "Missing NPZ for " + str(json_path))
    metadata = runner.validate_saved(npz_path, json_path, trial, np)
    diagnostic = metadata["diagnostic"]
    for key, value in provenance.items():
        require(diagnostic[key] == value, f"Provenance {key} differs in {trial['id']}")
    require(diagnostic["role"] == plan["role"] and diagnostic["split"] == "calibration"
            and diagnostic["trial_id"] == trial["id"]
            and diagnostic["initialization"] == plan["shared"]["initialization"],
            "Trial role, split, ID or initialization differs")
    require(str(diagnostic["slurm_array_task_id"]) == str(trial["channel"]),
            "Wrong Slurm task/channel")
    require(bool(diagnostic["slurm_job_id"]) and bool(diagnostic["hostname"]),
            "Missing runtime provenance")
    requested = runner.run_configuration(plan, trial)
    require(diagnostic["requested_configuration"] == requested,
            "Requested configuration differs from frozen plan")
    require(metadata["configuration"] == expected_campaign_configuration(plan, trial),
            "Actual campaign configuration differs from plan")
    require(metadata["format"] == "qpt_quiroga_campaign" and metadata["schema_version"] == 1,
            "Unknown archive format")
    require(metadata["observation_mode"] == "noiseless" and metadata["active_row_count"] == 128,
            "Wrong observation design")
    require(metadata["data_metadata"]["channel_seed"] == trial["channel"]
            and metadata["data_metadata"]["truth_rank"] == 1,
            "Wrong truth/channel metadata")
    require(metadata["source_sha256"] == {
        Path(key).name: value for key, value in provenance["source_sha256"].items()
        if Path(key).name in {"qpt_quiroga_campaign.py", "qpt_quiroga_frames.py", "qpt_quiroga_sensing.py"}},
        "Campaign source checksums differ")
    method, prefix = trial["method"], trial["method"].replace("-", "_")
    endpoint = metadata["methods"][method]
    status = endpoint["status"]
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    identities = {}
    for name, shape, kind in (("active_rows", (128,), "i"),
                              ("active_observations", (128,), "f"),
                              ("initial_factor", (16, 1), "c"),
                              ("truth_factor", (16, 1), "c")):
        values = arrays[name]
        require(values.shape == shape and values.dtype.kind == kind
                and np.all(np.isfinite(values)), "Invalid shared array " + name)
        identities[name] = array_hash(values, np)
        require(metadata[name + "_sha256"] == identities[name], "Shared array hash differs: " + name)
    require(array_hash(arrays["truth_factor"], np, shape_header=False)
            == metadata["data_metadata"]["truth_factor_sha256"], "Generator truth hash differs")
    require(np.array_equal(arrays["active_rows"], np.arange(128)), "Full rows differ")
    require(np.isclose(float(np.vdot(arrays["initial_factor"], arrays["initial_factor"]).real),
                       4.0, rtol=1e-13, atol=1e-13), "Initial factor trace is not normalized")
    require(np.isclose(float(np.vdot(arrays["truth_factor"], arrays["truth_factor"]).real),
                       4.0, rtol=1e-13, atol=1e-13), "Truth trace is not normalized")
    final = arrays[prefix + "_final_factor"]
    require(final.shape == (16, 1) and final.dtype.kind == "c", "Wrong final factor")
    if status != "failed":
        require(np.all(np.isfinite(final)), "Successful factor is nonfinite")
    steps = arrays[prefix + "_checkpoint_steps"]
    rows = arrays[prefix + "_cumulative_data_row_accesses"]
    times = arrays[prefix + "_optimizer_seconds"]
    require(steps.dtype.kind == rows.dtype.kind == "i" and times.shape == steps.shape
            and np.all(np.isfinite(times)) and np.all(times >= 0)
            and np.all(np.diff(times) >= 0), "Malformed time/step/work arrays")
    for key in (*runner.PHYSICAL_METRICS, "penalized_objective"):
        values = arrays[prefix + "_" + key]
        require(values.shape == steps.shape and values.dtype.kind == "f", "Malformed trace " + key)
        require(not np.any(np.isinf(values)), "Trace contains infinity: " + key)
        if status != "failed":
            require(np.all(np.isfinite(values)), "Successful trace contains undefined metrics")
        last = float(values[-1])
        expected = last if math.isfinite(last) else None
        same_number(endpoint[key], expected, np, "Terminal trace " + key, atol=0)
    same_number(endpoint["optimizer_seconds"], float(times[-1]), np, "Terminal time", atol=0)
    checkpoint_stride = 400 if method == "frames" else 100
    require(all(int(s) % checkpoint_stride == 0 for s in steps[:-1]), "Unexpected checkpoint cadence")
    if method == "frames":
        require(np.array_equal(rows, steps * 32), "FRAMES checkpoint charge differs")
    elif method == "printed-adafgd":
        require(np.array_equal(rows[:-1], steps[:-1] * 128)
                and int(rows[-1]) in {int(steps[-1]) * 128, (int(steps[-1]) + 1) * 128},
                "Printed-rule checkpoint charge differs")
    else:
        require(np.all(rows >= steps * 128) and np.all(rows % 128 == 0),
                "Armijo row charge cannot match its full data passes")
    data = Data(Design(2), arrays["truth_factor"], "noiseless")
    require(metadata["design"] == metadata["data_metadata"]["design"], "Design metadata disagree")
    for key in ("convention", "n_qubits", "d", "probes", "outcomes", "rows", "povm_a", "povm_b"):
        require(metadata["design"][key] == data.design.describe()[key], "Wrong sensing design " + key)
    require(np.allclose(data.all_observations(), arrays["active_observations"], rtol=1e-12, atol=1e-14),
            "Stored observations differ from archived noiseless truth")
    trace = type("ArchivedTrace", (), {"status": status, "final_factor": final})()
    independent, errors = runner.independent_endpoint(
        data, trace, arrays["active_observations"], np, evaluate)
    archived = diagnostic["full_design_endpoint"]
    require(set(archived) == set(independent), "Endpoint metric keys differ")
    for key, value in independent.items():
        same_number(archived[key], value, np, "Independent endpoint " + key)
    require(set(diagnostic["endpoint_metric_errors"]) == set(errors), "Undefined metric evidence differs")
    gate = runner.gate_for(status, archived, plan["shared"]["gate"])
    require(diagnostic["gate"] == gate, "Archived gate differs")
    require(runner.gate_for(status, independent, plan["shared"]["gate"])["pass"] == gate["pass"],
            "Independent physical gate differs")
    budget = diagnostic["budget"]
    require(budget["primary_equal_work_comparison"] is False
            and budget["requested_iterations"] == (40000 if method == "frames" else 10000)
            and budget["requested_nominal_row_accesses"] == (None if method == "armijo-fgd" else 1280000)
            and budget["actual_optimizer_row_accesses"] == int(rows[-1])
            and budget["actual_optimizer_row_epochs"] == int(rows[-1]) / 128,
            "Budget metadata differs")
    if method == "armijo-fgd":
        require(bool(budget["armijo_budget_qualification"]), "Armijo work caveat missing")
    first = first_physical_gate(arrays, prefix, plan["shared"]["gate"], np)
    return {
        "trial_id": trial["id"], "channel": trial["channel"], "method": method,
        "trial": trial, "status": status, "failure_reason": endpoint["failure_reason"],
        "completed_steps": int(steps[-1]), "actual_optimizer_row_accesses": int(rows[-1]),
        "actual_optimizer_row_epochs": int(rows[-1]) / 128,
        "endpoint_physical": archived, "gate": gate,
        "first_recorded_physical_gate": first,
        "first_recorded_physical_gate_step": None if first is None else first["step"],
        "first_recorded_physical_gate_charged_rows": None if first is None else first["charged_rows"],
        "checkpoint_gate_qualification": (
            "Retrospective physical-checkpoint diagnostic, conditional on the retained trajectory. "
            "It does not replace terminal method status, create an early-stopping result, "
            "or alter the official endpoint gate."),
        "shared_array_hashes": identities,
        "final_factor_sha256": array_hash(final, np),
        "json_path": str(json_path.resolve()), "json_sha256": runner.file_hash(json_path),
        "npz_path": str(npz_path.resolve()), "npz_sha256": runner.file_hash(npz_path),
        "checks": "passed",
    }


def audit_inventories(paths, expected, cases, provenance, require_complete):
    evidence, channels, seen = [], set(), set()
    for path in sorted(set(paths)):
        inventory = json.loads(path.read_text(encoding="utf-8"))
        require(inventory["kind"] == "qpt_diagnostic_inventory" and inventory["schema_version"] == 1,
                "Unexpected inventory format")
        for key, value in provenance.items():
            require(inventory[key] == value, "Inventory provenance differs: " + key)
        channel = inventory["channel"]
        require(channel in {0, 1} and channel not in channels, "Duplicate/unexpected channel inventory")
        channels.add(channel)
        if require_complete:
            require(inventory["state"] == "complete", "Channel inventory not complete")
        ids = {record["trial"]["id"] for record in inventory["trials"]}
        require(ids == {key for key, trial in expected.items() if trial["channel"] == channel},
                "Inventory trial set differs")
        require(len(ids) == len(inventory["trials"]), "Duplicate inventory trial")
        for record in inventory["trials"]:
            trial_id = record["trial"]["id"]
            require(record["trial"] == expected[trial_id], "Inventory trial configuration differs")
            if trial_id not in cases:
                continue
            case = cases[trial_id]
            require(record["status"] == ("archived_method_failure" if case["status"] == "failed" else "complete"),
                    "Inventory method classification differs")
            require(record["method_status"] == case["status"]
                    and record["gate_pass"] == case["gate"]["pass"]
                    and record["actual_optimizer_row_accesses"] == case["actual_optimizer_row_accesses"],
                    "Inventory status, gate or work differs")
            require(record["json_sha256"] == case["json_sha256"]
                    and record["npz_sha256"] == case["npz_sha256"],
                    "Inventory artifact checksum differs")
            seen.add(trial_id)
        evidence.append({"path": str(path.resolve()), "sha256": runner.file_hash(path),
                         "channel": channel, "state": inventory["state"]})
    if require_complete:
        require(channels == {0, 1} and seen == set(expected), "Both complete inventory checksum sets are required")
    return evidence, seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=HERE / "diagnostic_plan.json")
    parser.add_argument("--repo", type=Path, default=Path("/home/tony/frankwolfe"))
    parser.add_argument("--inventory", action="append", type=Path, default=[])
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Refusing to overwrite an audit report")
    require(args.results_dir.is_dir(), "Results root does not exist")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    trials = runner.validate_plan(plan)
    expected = {trial["id"]: trial for trial in trials}
    sources = local_sources(args.repo)
    provenance = {"campaign_id": plan["campaign_id"], "plan_sha256": runner.file_hash(args.plan),
                  "runner_sha256": runner.file_hash(HERE / "run_diagnostics.py"),
                  "source_commit": runner.COMMIT, "source_sha256": sources}
    sys.path.insert(0, str(args.repo))
    import numpy as np
    from paper.experiments.qpt_quiroga_campaign import evaluate_common_metrics
    from paper.experiments.qpt_quiroga_sensing import QuirogaSensingData, QuirogaSensingDesign
    runtime = np, QuirogaSensingData, QuirogaSensingDesign, evaluate_common_metrics
    cases, channel_identity = {}, {}
    json_paths = sorted(args.results_dir.rglob("result.json"))
    require(json_paths, "No result JSON files found")
    for path in json_paths:
        identity = json.loads(path.read_text(encoding="utf-8"))["diagnostic"]["trial_id"]
        require(identity in expected and identity not in cases, "Unknown or duplicate trial ID")
        case = audit_case(path, expected[identity], plan, provenance, runtime)
        channel = case["channel"]
        if channel in channel_identity:
            require(channel_identity[channel] == case["shared_array_hashes"],
                    "Methods on the same channel do not share data, truth and initial factor")
        else:
            channel_identity[channel] = case["shared_array_hashes"]
        cases[identity] = case
    missing = sorted(set(expected) - set(cases))
    require(args.allow_partial or not missing, "Incomplete campaign: use --allow-partial to report a subset")
    inventory_paths = args.inventory or list(args.results_dir.rglob("inventory.json"))
    inventories, checked_inventory = audit_inventories(
        inventory_paths, expected, cases, provenance, not args.allow_partial)
    counts = {}
    for method in runner.METHODS:
        subset = [case for case in cases.values() if case["method"] == method]
        counts[method] = {
            "audited_cases": len(subset), "status_counts": dict(Counter(case["status"] for case in subset)),
            "official_endpoint_gate_passes": sum(case["gate"]["pass"] for case in subset),
            "terminal_physical_thresholds_met_including_failed": sum(
                all(case["gate"]["components"].values()) for case in subset),
            "ever_recorded_physical_thresholds_met_including_failed": sum(
                case["first_recorded_physical_gate"] is not None for case in subset),
        }
    report = {
        "schema_version": 1, "kind": "qpt_diagnostic_result_audit", "created_utc": runner.now(),
        "audit_passed": True, "partial_mode": args.allow_partial,
        "all_results_present": not missing, "all_inventory_checksums_verified": checked_inventory == set(expected),
        "expected_trials": len(expected), "audited_trials": len(cases), "missing_trial_ids": missing,
        "results_root": str(args.results_dir.resolve()), "provenance": provenance,
        "audit_script_sha256": runner.file_hash(__file__), "inventories": inventories,
        "trial_ids_without_inventory_checksum": sorted(set(cases) - checked_inventory),
        "method_counts": counts, "cases": [cases[key] for key in sorted(cases)],
        "qualification": (
            "Calibration diagnostic only: no winner selection or validation replacement. "
            "Failed terminal statuses remain failed even if earlier or final physical metrics meet thresholds. "
            "Armijo uses its actual counted line-search row charges and is not an equal-budget primary baseline."),
    }
    require(local_sources(args.repo) == sources, "Source changed during audit")
    runner.write_json(args.output, report)
    print(json.dumps({"audit_passed": True, "audited_trials": len(cases), "missing_trials": len(missing),
                      "method_counts": counts, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
