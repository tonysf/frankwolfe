#!/usr/bin/env python3
"""Audit immutable tuning archives and row charges without rerunning optimization."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import common as c


def check_work(arrays, prefix, ep, np):
    steps, rows, times = (arrays[prefix + "_" + k] for k in ("checkpoint_steps", "cumulative_data_row_accesses", "optimizer_seconds"))
    c.require(steps.dtype.kind == rows.dtype.kind == "i" and times.dtype.kind == "f"
              and times.shape == steps.shape == rows.shape and steps.ndim == 1 and len(steps)
              and steps[0] == 0 and np.all(np.diff(steps) > 0) and np.all(np.isfinite(times))
              and np.all(times >= 0) and np.all(np.diff(times) >= 0) and np.all(np.diff(rows) >= 0), "Malformed work/time arrays")
    if prefix == "printed_adafgd":
        expected = 128 * steps
        extra = int(rows[-1] - expected[-1])
        c.require(extra == 0 or (ep["status"] == "failed" and extra == 128), "Wrong failed-attempt charge")
    else:
        if prefix + "_diagnostic_direction_passes" not in arrays:
            c.require(ep["status"] == "failed" and steps.tolist() == [0] and rows.tolist() == [0]
                      and ep["cumulative_data_row_accesses"] == 0 and ep["optimizer_seconds"] == float(times[-1]),
                      "Missing exact-line work histories outside an initial checkpoint failure")
            return {"verified_rows": 0, "terminal_attempt_rows": 0}
        counts = []
        terminal = 0
        for name, ordinary in (("direction_passes", 1), ("polynomial_forward_passes", 2), ("objective_evaluations", None)):
            values = arrays[prefix + "_diagnostic_" + name]
            c.require(values.dtype.kind == "i" and values.shape == (int(steps[-1]),)
                      and np.all(values >= 1) and (ordinary is None or np.all(values == ordinary)), "Invalid exact-line work history")
            counts.append(values)
            last = arrays[prefix + "_diagnostic_terminal_" + name]
            c.require(last.dtype.kind == "i" and last.shape == (1,) and last[0] >= 0, "Invalid terminal work history")
            terminal += int(last[0])
        cumulative = np.concatenate(([0], np.cumsum(sum(counts))))
        expected = 128 * cumulative[steps]
        extra = 128 * terminal
        c.require(arrays[prefix + "_diagnostic_terminal_attempt_rows"].tolist() == [extra], "Terminal work total differs")
        c.require(ep["status"] != "complete" or extra == 0, "Complete method has terminal-attempt charge")
        c.require(ep["status"] != "converged" or extra == 128, "Convergence needs one terminal gradient pass")
    expected = expected.copy()
    expected[-1] += extra
    c.require(np.array_equal(rows, expected) and int(rows[-1]) == ep["cumulative_data_row_accesses"]
              and float(times[-1]) == ep["optimizer_seconds"], "Work/time endpoint differs")
    return {"verified_rows": int(rows[-1]), "terminal_attempt_rows": extra}


def validate_archive(path, plan, trial, provenance, np):
    metadata = json.loads(path.read_text())
    d = metadata["tuning"]
    for key, value in provenance.items():
        c.require(d[key] == value, "Archive provenance differs: " + key)
    c.require(d["trial"] == trial and d["requested_configuration"] == c.configuration(trial)
              and metadata["configuration"] == c.actual_configuration(trial), "Requested/actual configuration differs")
    from paper.experiments.qpt_quiroga_initialization import initialization_descriptor
    c.require(metadata["initialization"] == initialization_descriptor("gell-mann"), "Initializer descriptor differs")
    names = ["qpt_quiroga_campaign.py", "qpt_quiroga_frames.py", "qpt_quiroga_sensing.py", "qpt_quiroga_initialization.py", "quantum_process_tomography.py"]
    if trial["method"] == "exact-linesearch-fgd":
        names.append("qpt_quiroga_linesearch.py")
    c.require(metadata["source_sha256"] == {n: provenance["source_sha256"]["paper/experiments/" + n] for n in names}, "Native source hashes differ")
    method, prefix = trial["method"], trial["method"].replace("-", "_")
    c.require(list(metadata["methods"]) == [method], "Unplanned method")
    ep = metadata["methods"][method]
    c.require(ep["status"] in {"complete", "converged", "failed"}, "Unknown status")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
        c.require(json.loads(str(archive["metadata_json"].item())) == metadata, "NPZ/JSON metadata differ")
        c.require(all(not archive[k].dtype.hasobject for k in archive.files), "Object arrays forbidden")
        arrays = {k: archive[k] for k in archive.files}
    steps = arrays[prefix + "_checkpoint_steps"]
    work = check_work(arrays, prefix, ep, np)
    c.require(int(steps[-1]) == ep["completed_steps"] <= trial["n_steps"], "Endpoint step differs")
    c.require(ep["status"] != "complete" or int(steps[-1]) == trial["n_steps"], "Complete method stopped early")
    expected_steps = list(range(0, int(steps[-1]) + 1, trial["metrics_frequency"]))
    if expected_steps[-1] != int(steps[-1]):
        expected_steps.append(int(steps[-1]))
    c.require(steps.tolist() == expected_steps, "Checkpoint schedule differs")
    for key in (*c.base.PHYSICAL_METRICS, "penalized_objective"):
        values = arrays[prefix + "_" + key]
        last = float(values[-1])
        c.require(values.shape == steps.shape and ep[key] == (last if math.isfinite(last) else None), "Endpoint trace differs: " + key)
        c.require(ep["status"] == "failed" or np.all(np.isfinite(values)), "Successful trace nonfinite")
    c.require(ep["status"] != "failed" or (ep["failure_reason"] and not d["gate"]["pass"]), "Missing failure evidence")
    c.require(ep["status"] != "converged" or method == "exact-linesearch-fgd", "Unexpected convergence")
    return metadata, arrays, work


def audit_case(path, plan, trial, provenance, np):
    from paper.experiments.qpt_quiroga_campaign import evaluate_common_metrics, _summed_objective_and_half_gradient
    from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
    metadata, arrays, work = validate_archive(path, plan, trial, provenance, np)
    d, ep = metadata["tuning"], metadata["methods"][trial["method"]]
    prefix = trial["method"].replace("-", "_")
    data = generate_quiroga_sensing_data(2, channel_seed=trial["channel"], observation_mode="noiseless")
    c.require(metadata["design"] == data.design.describe() and metadata["observation_mode"] == "noiseless"
              and metadata["active_row_count"] == 128 and metadata["data_metadata"]["channel_seed"] == trial["channel"], "Wrong data design")
    identity = {}
    expected = {"active_rows": np.arange(128), "active_observations": data.all_observations(),
                "truth_factor": data.truth_factor, "initial_factor": c.initial_reference(trial["initialization_seed"], np)}
    for key, reference in expected.items():
        c.require(arrays[key].shape == reference.shape and np.allclose(arrays[key], reference, rtol=1e-13, atol=1e-14), "Shared array differs: " + key)
        identity[key] = c.array_hash(arrays[key], np)
        c.require(identity[key] == metadata[key + "_sha256"], "Shared array checksum differs")
    factor = arrays[prefix + "_final_factor"]
    c.require(factor.shape == (16, 1), "Wrong final factor shape")
    trace = type("ArchivedTrace", (), {"status": ep["status"], "final_factor": factor})()
    physical, errors = c.base.independent_endpoint(data, trace, arrays["active_observations"], np, evaluate_common_metrics)
    for key, value in physical.items():
        archived = d["full_design_endpoint"][key]
        c.require((value is None and archived is None) or (value is not None and archived is not None
                  and np.isclose(value, archived, rtol=1e-10, atol=1e-13)), "Independent endpoint differs: " + key)
        if key in c.base.PHYSICAL_METRICS and value is not None and ep[key] is not None:
            c.require(np.isclose(value, ep[key], rtol=1e-10, atol=1e-13), "Native physical endpoint differs")
    c.require(set(errors) == set(d["endpoint_metric_errors"]), "Undefined metric evidence differs")
    gate = c.base.gate_for(ep["status"], physical, plan["gate"])
    c.require(gate == d["gate"], "Independent gate differs")
    if ep["status"] == "converged":
        norms = []
        for point in (arrays["initial_factor"], factor):
            _, half = _summed_objective_and_half_gradient(data, point, arrays["active_rows"], arrays["active_observations"], 128 * trial["alpha"])
            norms.append(2 * float(np.linalg.norm(half)))
        stop = ep["termination"]
        threshold = 1e-12 + 1e-10 * norms[0]
        c.require(stop["reason"] == "full_gradient_tolerance" and np.isclose(stop["initial_full_gradient_norm"], norms[0], rtol=1e-10, atol=1e-13)
                  and np.isclose(stop["full_gradient_norm"], norms[1], rtol=1e-8, atol=1e-13)
                  and np.isclose(stop["gradient_threshold"], threshold, rtol=1e-10, atol=1e-13)
                  and norms[1] <= threshold + 1e-13, "Stationarity termination not reproducible")
    first = None
    for index, step in enumerate(arrays[prefix + "_checkpoint_steps"]):
        point = {key: float(arrays[prefix + "_" + key][index]) for key in c.base.PHYSICAL_METRICS}
        if c.base.gate_for("complete", point, plan["gate"])["pass"]:
            first = {"step": int(step), "charged_rows": int(arrays[prefix + "_cumulative_data_row_accesses"][index])}
            break
    return {"trial_id": trial["id"], "trial": trial, "channel": trial["channel"], "method": trial["method"],
            "status": ep["status"], "failure_reason": ep["failure_reason"], "completed_steps": ep["completed_steps"],
            "optimizer_seconds": ep["optimizer_seconds"], "endpoint_physical": physical, "gate": gate,
            "first_recorded_physical_gate": first, "work_accounting": work, "shared_array_hashes": identity,
            "json_path": str(path.resolve()), "json_sha256": c.base.file_hash(path), "npz_sha256": c.base.file_hash(path.with_suffix(".npz"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results-dir", "plan", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--calibration-audit", type=Path)
    args = parser.parse_args(argv)
    c.require(not args.output.exists(), "Refusing audit overwrite")
    plan = json.loads(args.plan.read_text())
    trials = c.validate_plan(plan)
    c.check_selection(plan, args.calibration_audit)
    provenance = c.provenance(plan, args.plan, args.plan_sha256, c.REPO)
    import numpy as np
    expected, cases, shared = {t["id"]: t for t in trials}, {}, {}
    channels = sorted({t["channel"] for t in trials})
    for path in sorted(args.results_dir.rglob("result.json")):
        raw = json.loads(path.read_text())
        trial_id = raw["tuning"]["trial"]["id"]
        c.require(trial_id in expected and trial_id not in cases, "Unknown/duplicate trial")
        trial = expected[trial_id]
        c.require(raw["tuning"]["slurm_array_task_id"] == channels.index(trial["channel"]), "Wrong Slurm task")
        case = audit_case(path, plan, trial, provenance, np)
        for key, digest in case["shared_array_hashes"].items():
            group = (trial["channel"], trial["initialization_seed"] if key == "initial_factor" else None, key)
            c.require(group not in shared or shared[group] == digest, "Cases do not share data/initial factor")
            shared[group] = digest
        cases[trial_id] = case
    c.require(cases, "No results")
    inventories, verified, seen_channels = [], set(), set()
    for path in sorted(args.results_dir.rglob("inventory.json")):
        inv = json.loads(path.read_text())
        c.require(inv["kind"] == "qpt_quiroga_tuning_inventory" and inv["schema_version"] == 1, "Wrong inventory")
        for key, value in provenance.items():
            c.require(inv[key] == value, "Inventory provenance differs")
        channel = inv["channel"]
        c.require(channel in channels and channel not in seen_channels, "Duplicate/unknown inventory channel")
        seen_channels.add(channel)
        c.require(args.allow_partial or inv["state"] == "complete", "Incomplete inventory")
        c.require([r["trial"] for r in inv["trials"]] == [t for t in trials if t["channel"] == channel], "Inventory trials differ")
        for record in inv["trials"]:
            key = record["trial"]["id"]
            if key not in cases:
                c.require(args.allow_partial, "Inventory case missing")
                continue
            case = cases[key]
            for kind in ("json", "npz"):
                c.require(record[kind + "_sha256"] == case[kind + "_sha256"]
                          and (path.parent / record[kind]).resolve() == Path(case["json_path"]).with_suffix("." + kind), "Inventory archive path/hash differs")
            c.require(record["method_status"] == case["status"] and record["gate_pass"] == case["gate"]["pass"]
                      and record["status"] == ("archived_method_failure" if case["status"] == "failed" else "complete"), "Inventory status differs")
            verified.add(key)
        inventories.append({"path": str(path.resolve()), "sha256": c.base.file_hash(path), "channel": channel, "state": inv["state"]})
    missing = sorted(set(expected) - set(cases))
    c.require(args.allow_partial or (not missing and verified == set(expected)), "Missing complete checksummed inventory")
    c.check_module_origins(c.REPO)
    c.require(c.source_hashes(c.REPO, plan["source"]["commit"]) == provenance["source_sha256"], "Source changed during audit")
    report = {"schema_version": 1, "kind": "qpt_quiroga_tuning_audit", "audit_passed": True,
              "created_utc": c.base.now(), "expected_trials": len(trials), "audited_trials": len(cases),
              "missing_trial_ids": missing, "partial_mode": args.allow_partial, "provenance": provenance,
              "inventories": inventories, "trial_ids_without_inventory_checksum": sorted(set(cases) - verified),
              "cases": [cases[key] for key in sorted(cases)], "status_counts": dict(Counter(case["status"] for case in cases.values())),
              "qualification": "Channels 100–107 are frozen historical revalidation, not unseen validation. Held-out 1000–1019 are excluded. Exact-line search is a separately labeled modification, not the printed step. Row work excludes spectral, TP, and checkpoint costs."}
    c.write_json(args.output, report)
    print(json.dumps({"audit_passed": True, "audited_trials": len(cases), "gate_passes": sum(case["gate"]["pass"] for case in cases.values())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
