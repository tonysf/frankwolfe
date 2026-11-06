#!/usr/bin/env python3
"""Execute the frozen initializer diagnostic only in a Slurm allocation.

Adapted from 2026-10-06 run_followup.py, SHA256
fb73fd3727dd151291c101d7c84ba5247244f7223cd9fe0db560a29756e39fdb.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from freeze_plan import build_plan
import support as base


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_plan(plan):
    require(plan == build_plan(plan["source"]["commit"], plan["frozen_utc"]),
            "Plan differs from the explicit reviewed follow-up")
    return plan["trials"]


def configuration(plan, trial):
    p = plan["shared"]
    beta0 = 1.0 / (2.0 * trial["alpha"])
    return {
        "methods": [trial["method"]], "rank": 1,
        "initialization_basis": "gell-mann",
        "initialization_seed": 200000 + 10 * trial["channel"],
        "sampling_seed": 300000 + 100 * trial["channel"],
        "n_steps": trial["row_epochs"], "frames_steps": trial["row_epochs"] * 4,
        "metrics_frequency": trial["checkpoint_epochs"],
        "frames_metrics_frequency": trial["checkpoint_epochs"] * 4,
        "eta_scale": trial["eta_scale"], "tp_weight": 128 * trial["alpha"],
        "tau": trial["tau"], "batch_size": 32,
        "spectral_norm_method": "dense", "dense_max_process_dimension": 1024,
        "spectral_norm_tolerance": 1e-10, "spectral_norm_maxiter": None,
        "armijo_initial_step": 1.0, "armijo_shrink": 0.5,
        "armijo_constant": 1e-4, "armijo_max_backtracks": 30,
        "armijo_gradient_atol": trial["armijo_gradient_atol"],
        "armijo_gradient_rtol": trial["armijo_gradient_rtol"],
        "frames_beta0": beta0, "rho_schedule": p["rho"],
        "step_size_schedule": trial.get("step_schedule", p["step"]),
        "frames_smoothing_schedule": {"scale": beta0, "offset": 1.0,
                                      "exponent": p["frames_penalty_exponent"], "cap": None},
        "show_progress": False,
    }


def actual_configuration(requested):
    expected = {key: value for key, value in requested.items() if key not in {"tp_weight", "show_progress"}}
    expected.update({"tp_weight_lambda": requested["tp_weight"], "frames_beta": requested["frames_beta0"],
                     "frames_initial_tp_weight_per_row_alpha": 1 / (2 * requested["frames_beta0"]),
                     "frames_initial_tp_weight_lambda": 128 / (2 * requested["frames_beta0"]),
                     "frames_penalty_mode": "independent_schedule"})
    for key in ("rho_schedule", "step_size_schedule", "frames_smoothing_schedule"):
        expected[key] = {"type": "PowerSchedule", **expected[key]}
    return expected


def source_hashes(repo, commit, base):
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    require(head == commit and not dirty, "Source checkout is not clean at the frozen commit")
    harness = sorted(str(path.relative_to(repo)) for path in
                     (repo / "scripts/qpt_gell_mann_diagnostic").glob("*.py"))
    paths = [*base.SOURCE_PATHS, "paper/experiments/qpt_quiroga_initialization.py", *harness,
             "scripts/qpt_gell_mann_diagnostic/diagnostic.sbatch"]
    return {name: base.file_hash(repo / name) for name in paths}


def validate_saved(npz_path, json_path, plan, trial, provenance, base, np):
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    d = metadata["diagnostic"]
    for key, value in provenance.items():
        require(d[key] == value, "Archived provenance differs: " + key)
    require(d["trial"] == trial and d["trial_id"] == trial["id"] and d["split"] == "calibration"
            and d["role"] == plan["role"], "Archived identity/role differs")
    cfg = configuration(plan, trial)
    require(d["requested_configuration"] == cfg and metadata["configuration"] == actual_configuration(cfg),
            "Requested/actual configuration differs")
    from paper.experiments.qpt_quiroga_initialization import initialization_descriptor
    require(metadata["initialization"] == initialization_descriptor("gell-mann"),
            "Initialization descriptor differs")
    expected_sources = {name: provenance["source_sha256"]["paper/experiments/" + name]
                        for name in ("qpt_quiroga_campaign.py", "qpt_quiroga_frames.py",
                                     "qpt_quiroga_sensing.py", "qpt_quiroga_initialization.py",
                                     "quantum_process_tomography.py")}
    require(metadata["source_sha256"] == expected_sources, "Campaign source hashes differ")
    method, prefix = trial["method"], trial["method"].replace("-", "_")
    require(list(metadata["methods"]) == [method], "Another method was included")
    ep = metadata["methods"][method]
    require(ep["status"] in {"complete", "converged", "failed"}, "Unknown method status")
    with np.load(npz_path, allow_pickle=False) as archive:
        require(json.loads(str(archive["metadata_json"].item())) == metadata, "NPZ/JSON differ")
        require(all(not archive[key].dtype.hasobject for key in archive.files), "Object array")
        steps = archive[prefix + "_checkpoint_steps"]
        rows = archive[prefix + "_cumulative_data_row_accesses"]
        require(steps.ndim == 1 and steps.size and steps[0] == 0 and np.all(np.diff(steps) > 0),
                "Malformed checkpoints")
        require(rows.shape == steps.shape and np.all(np.diff(rows) >= 0)
                and int(rows[-1]) == ep["cumulative_data_row_accesses"]
                and int(steps[-1]) == ep["completed_steps"], "Work/endpoint mismatch")
        require(archive[prefix + "_final_factor"].shape == (16, 1)
                and np.array_equal(archive["active_rows"], np.arange(128)), "Wrong factor/data shape")
        for name in (*base.PHYSICAL_METRICS, "penalized_objective"):
            values = archive[prefix + "_" + name]
            require(values.shape == steps.shape, "Metric trace length differs")
            last = float(values[-1])
            require(ep[name] == (last if math.isfinite(last) else None), "Endpoint/trace differs: " + name)
            if ep["status"] != "failed":
                require(np.all(np.isfinite(values)), "Successful metric trace is nonfinite")
        if ep["status"] == "complete":
            requested_steps = cfg["frames_steps"] if method == "frames" else cfg["n_steps"]
            require(int(steps[-1]) == requested_steps and len(steps) == 101, "Incomplete complete run")
            if method != "armijo-fgd":
                require(int(rows[-1]) == trial["row_epochs"] * 128, "Wrong nominal row budget")
        if ep["status"] == "converged":
            require(method == "armijo-fgd", "Unexpected non-Armijo convergence")
            termination = ep.get("termination", {})
            require(termination.get("reason") == "full_gradient_tolerance"
                    and math.isfinite(termination["full_gradient_norm"])
                    and 0 <= termination["full_gradient_norm"] <= termination["gradient_threshold"],
                    "Missing/inconsistent stationarity termination")
            require(np.isclose(termination["gradient_threshold"], trial["armijo_gradient_atol"]
                               + trial["armijo_gradient_rtol"] * termination["initial_full_gradient_norm"],
                               rtol=1e-13, atol=0), "Stopping threshold differs")
        if ep["status"] == "failed":
            require(bool(ep["failure_reason"]) and d["gate"]["pass"] is False, "Failed method missing failure evidence")
    require(d["gate"] == base.gate_for(ep["status"], d["full_design_endpoint"], plan["shared"]["gate"]),
            "Endpoint gate differs")
    return metadata


def execute_trial(plan, trial, out, provenance, base, runtime):
    np, generate, campaign, save, Schedule, evaluate = runtime
    destination = out / trial["id"]
    destination.mkdir()
    data = generate(2, channel_seed=trial["channel"], observation_mode="noiseless")
    requested = configuration(plan, trial)
    kwargs = dict(requested)
    for key in ("rho_schedule", "step_size_schedule", "frames_smoothing_schedule"):
        kwargs[key] = Schedule(**kwargs[key])
    metadata, payload, traces = campaign(data, **kwargs)
    trace = traces[trial["method"]]
    physical, errors = base.independent_endpoint(data, trace, data.all_observations(), np, evaluate)
    actual_rows = int(trace.cumulative_data_row_accesses[-1])
    metadata["diagnostic"] = {
        **provenance, "role": plan["role"], "split": "calibration", "trial": trial,
        "trial_id": trial["id"], "requested_configuration": requested,
        "initialization": plan["shared"]["initialization"],
        "full_design_endpoint": physical, "endpoint_metric_errors": errors,
        "gate": base.gate_for(trace.status, physical, plan["shared"]["gate"]),
        "budget": {"primary_equal_work_comparison": False,
                   "requested_iterations": requested["frames_steps"] if trial["method"] == "frames" else requested["n_steps"],
                   "requested_nominal_row_accesses": None if trial["method"] == "armijo-fgd" else trial["row_epochs"] * 128,
                   "actual_optimizer_row_accesses": actual_rows, "actual_optimizer_row_epochs": actual_rows / 128,
                   "qualification": plan["budget_qualification"]},
        "failure_archival_qualification": (
            "Existing campaign FRAMES exception fallback retains initial factor/step zero, not partial trajectory."
            if trial["method"] == "frames" and trace.status == "failed" else None),
    }
    npz, js = destination / "result.npz", destination / "result.json"
    save(npz, js, metadata, payload)
    validate_saved(npz, js, plan, trial, provenance, base, np)
    return {"status": "archived_method_failure" if trace.status == "failed" else "complete",
            "method_status": trace.status, "failure_reason": trace.failure_reason or None,
            "gate_pass": metadata["diagnostic"]["gate"]["pass"],
            "actual_optimizer_row_accesses": actual_rows,
            "json": str(js.relative_to(out)), "json_sha256": base.file_hash(js),
            "npz": str(npz.relative_to(out)), "npz_sha256": base.file_hash(npz), "completed_utc": base.now()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    trials = validate_plan(plan)
    if args.validate_only:
        print(json.dumps({"valid": True, "trials": len(trials), "plan_sha256": base.file_hash(args.plan)}))
        return 0
    require(bool(os.environ.get("SLURM_JOB_ID")) and bool(os.environ.get("SLURM_JOB_NODELIST")),
            "Scientific execution requires Slurm")
    require(args.array_index in (0, 1) and args.out is not None
            and args.deadline_epoch is not None and math.isfinite(args.deadline_epoch), "Missing execution controls")
    require(int(os.environ["SLURM_ARRAY_TASK_ID"]) == args.array_index, "Wrong Slurm array index")
    repo = Path(plan["source"]["repo"])
    hashes = source_hashes(repo, plan["source"]["commit"], base)
    require(Path(__file__).resolve().is_relative_to(repo.resolve()),
            "Runner must execute from the pinned checkout")
    args.out.mkdir(exist_ok=False)
    provenance = {"campaign_id": plan["campaign_id"], "plan_sha256": base.file_hash(args.plan),
                  "runner_sha256": base.file_hash(__file__),
                  "plan_builder_sha256": base.file_hash(Path(__file__).with_name("freeze_plan.py")),
                  "source_commit": plan["source"]["commit"], "source_sha256": hashes,
                  "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
                  "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                  "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"]}
    inventory = {"schema_version": 1, "kind": "qpt_gell_mann_inventory", **provenance,
                 "created_utc": base.now(), "channel": args.array_index, "state": "prepared",
                 "trials": [{"trial": t, "status": "pending"} for t in trials if t["channel"] == args.array_index]}
    inventory_path = args.out / "inventory.json"
    base.write_json(inventory_path, inventory)
    try:
        sys.path.insert(0, str(repo))
        import numpy as np
        from paper.experiments.qpt_quiroga_campaign import run_campaign, save_campaign, evaluate_common_metrics
        from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
        from paper.experiments.quantum_process_tomography import PowerSchedule
        for function in (run_campaign, generate_quiroga_sensing_data, PowerSchedule):
            require(Path(sys.modules[function.__module__].__file__).resolve().is_relative_to(repo.resolve()),
                    "Imported scientific code outside pinned checkout")
        runtime = np, generate_quiroga_sensing_data, run_campaign, save_campaign, PowerSchedule, evaluate_common_metrics
        inventory["state"] = "running"
        for record in inventory["trials"]:
            if args.deadline_epoch - time.time() <= plan["resources"]["minimum_seconds_before_new_trial"]:
                inventory["state"] = "time_guard_pending"
                base.write_json(inventory_path, inventory, replace=True)
                return 75
            record.update(status="running", started_utc=base.now())
            base.write_json(inventory_path, inventory, replace=True)
            print("START " + record["trial"]["id"], flush=True)
            record.update(execute_trial(plan, record["trial"], args.out, provenance, base, runtime))
            base.write_json(inventory_path, inventory, replace=True)
            print("END " + record["trial"]["id"] + " " + record["status"], flush=True)
        require(source_hashes(repo, plan["source"]["commit"], base) == hashes, "Source changed during run")
        inventory.update(state="complete", completed_utc=base.now())
        base.write_json(inventory_path, inventory, replace=True)
        return 2 if any(record["status"] == "archived_method_failure" for record in inventory["trials"]) else 0
    except Exception as error:
        inventory.update(state="infrastructure_or_invalid_archive", error=f"{type(error).__name__}: {error}",
                         stopped_utc=base.now())
        base.write_json(inventory_path, inventory, replace=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
