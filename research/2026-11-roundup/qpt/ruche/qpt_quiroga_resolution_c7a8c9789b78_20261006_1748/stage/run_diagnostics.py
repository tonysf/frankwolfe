#!/usr/bin/env python3
"""Frozen, calibration-only diagnostic cases using unmodified campaign code.

Plan inspection uses only the standard library. Scientific imports/execution
require a Slurm allocation and a clean, pinned source checkout. These cases do
not select configurations, replace validation, or form an equal-work comparison
with Armijo. All method failures are retained, never dropped from the inventory.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

COMMIT = "c7a8c9789b7814ede1c95e53eef13bfe1dd58c63"
METHODS = ("printed-adafgd", "armijo-fgd", "frames")
SOURCE_PATHS = (
    "paper/experiments/qpt_quiroga_campaign.py",
    "paper/experiments/qpt_quiroga_frames.py",
    "paper/experiments/qpt_quiroga_sensing.py",
    "paper/experiments/qpt_generate_data.py",
    "paper/experiments/qpt_observation_noise.py",
    "paper/experiments/quantum_process_tomography.py",
    "frank_wolfe/__init__.py", "frank_wolfe/algorithms/base.py",
    "frank_wolfe/algorithms/frames.py", "frank_wolfe/core/objective.py",
)
SHARED = {
    "n_qubits": 2, "rank": 1, "active_rows": 128, "row_epochs": 10000,
    "checkpoint_epochs": 100, "batch_size": 32, "frames_penalty_exponent": 0.25,
    "rho": {"scale": 2, "offset": 4, "exponent": 0.6, "cap": 1},
    "step": {"scale": 2, "offset": 2, "exponent": 1, "cap": 1},
    "observation_mode": "noiseless", "initialization": "trace_normalized_choi_uniform",
    "gate": {"fidelity_min": 0.99, "rmse_max": 1e-6, "normalized_tp_max": 0.001},
}
PHYSICAL_METRICS = (
    "measurement_loss_sum", "measurement_loss_mean", "rmse", "tp_violation",
    "normalized_tp_violation", "process_fidelity",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def finite_positive(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    target = path.with_name(path.name + ".pending") if replace else path
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(target, path)


def validate_plan(plan):
    require(plan["schema_version"] == 1
            and plan["kind"] == "qpt_calibration_resolution_diagnostic",
            "Unsupported diagnostic plan")
    require(isinstance(plan["campaign_id"], str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", plan["campaign_id"]),
            "Invalid campaign ID")
    require(plan["source"]["commit"] == COMMIT, "Unexpected source commit")
    require(plan["source"]["repo"] == "/gpfs/workdir/silvetian/frankwolfe-quiroga-" + COMMIT,
            "Unexpected pinned source directory")
    require(plan["shared"] == SHARED, "Shared diagnostic settings differ")
    require(plan["channels"] == [0, 1], "Only calibration channels 0 and 1 are allowed")
    require(isinstance(plan["role"], str) and "diagnostic" in plan["role"].lower(),
            "The diagnostic role must be explicit")
    guard = plan["resources"]["minimum_seconds_before_new_trial"]
    require(isinstance(guard, int) and not isinstance(guard, bool) and guard >= 120,
            "At least 120 seconds of walltime headroom is required")
    trials = plan["trials"]
    require(isinstance(trials, list) and trials, "The explicit trial list is empty")
    seen = set()
    for trial in trials:
        required = {"id", "channel", "method", "alpha", "eta_scale", "tau"}
        require(required <= set(trial) <= required | {"step_schedule"},
                "Unexpected or missing trial keys")
        trial_id = trial["id"]
        require(isinstance(trial_id, str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,179}", trial_id),
                "Unsafe trial ID")
        require(trial_id not in seen, "Duplicate trial ID")
        seen.add(trial_id)
        require(type(trial["channel"]) is int and trial["channel"] in (0, 1),
                "Trial is outside calibration channels")
        require(trial["method"] in METHODS, "Unknown method")
        require(all(finite_positive(trial[key]) for key in ("alpha", "eta_scale", "tau")),
                "Trial hyperparameters must be finite and positive")
        if trial["method"] == "frames":
            require(trial["tau"] >= 2, "FRAMES radius excludes the trace-normalized initial factor")
        if "step_schedule" in trial:
            require(trial["method"] == "frames", "Only FRAMES may override the step schedule")
            schedule = trial["step_schedule"]
            require(set(schedule) == {"scale", "offset", "exponent", "cap"}
                    and all(finite_positive(v) for v in schedule.values())
                    and schedule["exponent"] == schedule["cap"] == 1,
                    "Invalid explicit FRAMES step schedule")
    require({trial["channel"] for trial in trials} == {0, 1}, "Both calibration channels are required")
    if "expected_counts" in plan:
        counts = plan["expected_counts"]
        require(counts == {
            "trials": len(trials),
            "trials_per_channel": len(trials) // 2,
            "printed": sum(t["method"] == "printed-adafgd" for t in trials),
            "frames": sum(t["method"] == "frames" for t in trials),
            "armijo": sum(t["method"] == "armijo-fgd" for t in trials)},
            "Declared trial counts differ")
        require(all(sum(t["channel"] == c for t in trials) == counts["trials_per_channel"]
                    for c in (0, 1)), "Channels do not have the declared number of cases")
    if "seed_rule" in plan:
        require(plan["seed_rule"] == {"initialization": "200000+10*channel",
                                     "sampling": "300000+100*channel"},
                "Declared seed rule differs")
    return trials


def run_configuration(plan, trial):
    """Pure conversion; values remain explicit and testable without NumPy."""
    shared = plan["shared"]
    rows, batch = shared["active_rows"], shared["batch_size"]
    return {
        "methods": [trial["method"]], "rank": shared["rank"],
        "initialization_seed": 200000 + 10 * trial["channel"],
        "sampling_seed": 300000 + 100 * trial["channel"],
        "n_steps": shared["row_epochs"],
        "frames_steps": shared["row_epochs"] * rows // batch,
        "metrics_frequency": shared["checkpoint_epochs"],
        "frames_metrics_frequency": shared["checkpoint_epochs"] * rows // batch,
        "eta_scale": trial["eta_scale"], "tp_weight": rows * trial["alpha"],
        "tau": trial["tau"], "batch_size": batch,
        "spectral_norm_method": "dense", "dense_max_process_dimension": 1024,
        "spectral_norm_tolerance": 1e-10, "spectral_norm_maxiter": None,
        "armijo_initial_step": 1.0, "armijo_shrink": 0.5,
        "armijo_constant": 1e-4, "armijo_max_backtracks": 30,
        "frames_beta0": 1.0 / (2.0 * trial["alpha"]),
        "rho_schedule": shared["rho"],
        "step_size_schedule": trial.get("step_schedule", shared["step"]),
        "frames_smoothing_schedule": {
            "scale": 1.0 / (2.0 * trial["alpha"]), "offset": 1.0,
            "exponent": shared["frames_penalty_exponent"], "cap": None},
        "show_progress": False,
    }


def gate_for(status, metrics, gate):
    def meets(key, threshold, lower):
        value = metrics.get(key)
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)
                and (value >= threshold if lower else value <= threshold))
    components = {
        "fidelity": meets("process_fidelity", gate["fidelity_min"], True),
        "full_design_rmse": meets("rmse", gate["rmse_max"], False),
        "normalized_tp": meets("normalized_tp_violation", gate["normalized_tp_max"], False),
    }
    return {"thresholds": gate, "components": components,
            "pass": status in {"complete", "converged"} and all(components.values()),
            "method_status": status}


def verify_source(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == COMMIT, "Pinned source changed")
    require(not git("status", "--porcelain", "--untracked-files=normal"),
            "Source checkout is not clean")
    return {relative: file_hash(repo / relative) for relative in SOURCE_PATHS}


def independent_endpoint(data, trace, targets, np, evaluator):
    """No fallback from undefined full-design metrics to prior checkpoints."""
    errors = {}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            # Zero reporting penalty keeps these physical diagnostics independent
            # of the method-native objective and its potentially huge penalty.
            metrics = evaluator(data, trace.final_factor,
                                np.arange(data.m, dtype=np.int64), targets, 0.0)
        physical = {key: float(metrics[key]) for key in PHYSICAL_METRICS}
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        if trace.status != "failed":
            raise
        physical = {key: None for key in PHYSICAL_METRICS}
        errors["full_design_metrics"] = f"{type(error).__name__}: {error}"
    for name, order in (("factor_frobenius_norm", "fro"), ("factor_operator_norm", 2)):
        try:
            with np.errstate(over="raise", invalid="raise"):
                value = float(np.linalg.norm(trace.final_factor, ord=order))
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite factor norm")
            physical[name] = value
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            if trace.status != "failed":
                raise
            physical[name] = None
            errors[name] = f"{type(error).__name__}: {error}"
    return physical, errors


def validate_saved(npz_path, json_path, trial, np):
    """Validate the exact serialized artifact, not just its in-memory source."""
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    method, prefix = trial["method"], trial["method"].replace("-", "_")
    require(list(metadata["methods"]) == [method], "Archive includes another method")
    require(metadata["diagnostic"]["trial"] == trial, "Archived trial differs")
    endpoint = metadata["methods"][method]
    require(endpoint["status"] in {"complete", "converged", "failed"}, "Unknown terminal status")
    with np.load(npz_path, allow_pickle=False) as archive:
        require(json.loads(str(archive["metadata_json"].item())) == metadata,
                "NPZ/JSON metadata differ")
        require(all(not archive[key].dtype.hasobject for key in archive.files), "Object array")
        steps = archive[prefix + "_checkpoint_steps"]
        work = archive[prefix + "_cumulative_data_row_accesses"]
        require(steps.ndim == 1 and steps.size and steps[0] == 0
                and np.all(np.diff(steps) > 0), "Invalid checkpoint steps")
        require(work.shape == steps.shape and np.all(np.diff(work) >= 0),
                "Invalid cumulative work trace")
        require(int(steps[-1]) == endpoint["completed_steps"]
                and int(work[-1]) == endpoint["cumulative_data_row_accesses"],
                "Terminal work/steps differ")
        require(archive[prefix + "_final_factor"].shape == (16, 1), "Wrong factor shape")
        require(np.array_equal(archive["active_rows"], np.arange(128)), "Wrong sensing rows")
        require(archive["active_observations"].shape == (128,), "Wrong target shape")
        if endpoint["status"] == "complete":
            expected = 40000 if method == "frames" else 10000
            require(int(steps[-1]) == expected, "Completed method stopped early")
            require(steps.size == 101, "Unexpected complete checkpoint count")
            if method != "armijo-fgd":
                require(int(work[-1]) == 1280000, "Nominal row budget differs")
        if endpoint["status"] == "failed":
            require(bool(endpoint["failure_reason"]), "Missing method failure reason")
            require(metadata["diagnostic"]["gate"]["pass"] is False,
                    "A failed method passed the gate")
    return metadata


def execute_trial(plan, trial, out, provenance, runtime):
    np, generate, campaign, save, schedule, evaluator = runtime
    trial_out = out / trial["id"]
    trial_out.mkdir()
    data = generate(plan["shared"]["n_qubits"], channel_seed=trial["channel"],
                    observation_mode="noiseless", noise_std=0.0, noise_seed=0,
                    shots=None, shot_seed=0)
    require(data.m == 128, "Unexpected full-design row count")
    configuration = run_configuration(plan, trial)
    kwargs = dict(configuration)
    for key in ("rho_schedule", "step_size_schedule", "frames_smoothing_schedule"):
        kwargs[key] = schedule(**kwargs[key])
    metadata, payload, traces = campaign(data, **kwargs)
    trace = traces[trial["method"]]
    metrics, metric_errors = independent_endpoint(
        data, trace, data.all_observations(), np, evaluator)
    actual_rows = int(trace.cumulative_data_row_accesses[-1])
    metadata["diagnostic"] = {
        **provenance, "role": plan["role"], "split": "calibration",
        "trial_id": trial["id"], "trial": trial,
        "initialization": plan["shared"]["initialization"],
        "requested_configuration": configuration,
        "full_design_endpoint": metrics, "endpoint_metric_errors": metric_errors,
        "gate": gate_for(trace.status, metrics, plan["shared"]["gate"]),
        "budget": {
            "primary_equal_work_comparison": False,
            "requested_iterations": (configuration["frames_steps"] if trial["method"] == "frames"
                                     else configuration["n_steps"]),
            "requested_nominal_row_accesses": (None if trial["method"] == "armijo-fgd"
                                                else 1280000),
            "actual_optimizer_row_accesses": actual_rows,
            "actual_optimizer_row_epochs": actual_rows / 128,
            "armijo_budget_qualification": (
                "10000 optimizer iterations; line-search row accesses are counted, not row-budget-matched"
                if trial["method"] == "armijo-fgd" else None),
        },
        "failure_archival_qualification": (
            "Existing campaign FRAMES exception path retains initial factor and step zero; "
            "partial adapter trajectory is unavailable. No production code was patched."
            if trial["method"] == "frames" and trace.status == "failed" else None),
    }
    npz_path, json_path = trial_out / "result.npz", trial_out / "result.json"
    save(npz_path, json_path, metadata, payload)
    validate_saved(npz_path, json_path, trial, np)
    return {
        "status": "archived_method_failure" if trace.status == "failed" else "complete",
        "method_status": trace.status, "failure_reason": trace.failure_reason or None,
        "gate_pass": metadata["diagnostic"]["gate"]["pass"],
        "actual_optimizer_row_accesses": actual_rows,
        "json": str(json_path.relative_to(out)), "json_sha256": file_hash(json_path),
        "npz": str(npz_path.relative_to(out)), "npz_sha256": file_hash(npz_path),
        "completed_utc": now(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    all_trials = validate_plan(plan)
    if args.validate_only:
        print(json.dumps({"valid": True, "plan_sha256": file_hash(args.plan),
                          "trials": len(all_trials), "channels": plan["channels"]}, sort_keys=True))
        return 0
    require(args.array_index in (0, 1) and args.out is not None
            and args.deadline_epoch is not None and math.isfinite(args.deadline_epoch),
            "Execution requires array index, fresh output directory and deadline")
    require(bool(os.environ.get("SLURM_JOB_ID")) and bool(os.environ.get("SLURM_JOB_NODELIST")),
            "Scientific execution requires a Slurm allocation")
    if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        require(int(os.environ["SLURM_ARRAY_TASK_ID"]) == args.array_index,
                "Slurm array index differs")
    repo = Path(plan["source"]["repo"])
    sources = verify_source(repo)
    args.out.mkdir(exist_ok=False)
    provenance = {
        "campaign_id": plan["campaign_id"], "plan_sha256": file_hash(args.plan),
        "runner_sha256": file_hash(__file__), "source_commit": COMMIT,
        "source_sha256": sources, "hostname": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    inventory = {"schema_version": 1, "kind": "qpt_diagnostic_inventory", **provenance,
                 "created_utc": now(), "channel": args.array_index, "state": "prepared",
                 "trials": [{"trial": trial, "status": "pending"} for trial in all_trials
                            if trial["channel"] == args.array_index]}
    inventory_path = args.out / "inventory.json"
    write_json(inventory_path, inventory)
    sys.path.insert(0, str(repo))
    try:
        import numpy as np
        from paper.experiments.qpt_quiroga_campaign import (
            evaluate_common_metrics, run_campaign, save_campaign)
        from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
        from paper.experiments.quantum_process_tomography import PowerSchedule
        for callable_ in (run_campaign, generate_quiroga_sensing_data, PowerSchedule):
            module = sys.modules[callable_.__module__]
            require(Path(module.__file__).resolve().is_relative_to(repo.resolve()),
                    "A scientific module was imported outside the pinned checkout")
        runtime = (np, generate_quiroga_sensing_data, run_campaign, save_campaign,
                   PowerSchedule, evaluate_common_metrics)
        inventory["state"] = "running"
        for record in inventory["trials"]:
            if args.deadline_epoch - time.time() <= plan["resources"]["minimum_seconds_before_new_trial"]:
                inventory["state"] = "time_guard_pending"
                write_json(inventory_path, inventory, replace=True)
                return 75
            record.update({"status": "running", "started_utc": now()})
            write_json(inventory_path, inventory, replace=True)
            print("START " + record["trial"]["id"], flush=True)
            record.update(execute_trial(plan, record["trial"], args.out, provenance, runtime))
            write_json(inventory_path, inventory, replace=True)
            print("END " + record["trial"]["id"] + " " + record["status"], flush=True)
        require(verify_source(repo) == sources, "Source changed during diagnostic execution")
        inventory["state"] = "complete"
        inventory["completed_utc"] = now()
        write_json(inventory_path, inventory, replace=True)
        return 2 if any(r["status"] == "archived_method_failure" for r in inventory["trials"]) else 0
    except Exception as error:
        inventory["state"] = "infrastructure_or_invalid_archive"
        inventory["error"] = f"{type(error).__name__}: {error}"
        inventory["stopped_utc"] = now()
        write_json(inventory_path, inventory, replace=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
