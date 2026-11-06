#!/usr/bin/env python3
"""Execute a pinned, predeclared calibration inventory inside one Slurm task.

--validate-only is a standard-library-only inventory check, not an experiment.
This launcher never selects configurations or creates a selection manifest.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

COMMIT = "c7a8c9789b7814ede1c95e53eef13bfe1dd58c63"
METHODS = ("printed-adafgd", "frames")
PHYSICAL_METRICS = ("measurement_loss_sum", "measurement_loss_mean", "rmse",
                    "tp_violation", "normalized_tp_violation", "process_fidelity")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value, *, replace=False):
    path = Path(path)
    target = path.with_name(path.name + ".pending") if replace else path
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    if replace:
        os.replace(target, path)


def make_inventory(plan):
    require(plan["schema_version"] == 1, "Unsupported plan schema")
    require(plan["kind"] == "calibration_execution_plan_not_selection_manifest",
            "Plan must not be a selection manifest")
    require(plan["source"]["commit"] == COMMIT, "Unexpected source revision")
    require(plan["source"]["require_clean"] is True, "Clean source is mandatory")
    require(plan["channels"] == list(range(8)), "Expected calibration channels 0..7")
    require(plan["initialization_indices"] == [0], "Expected one initialization")
    require(plan["sampling_indices"] == [0, 1, 2], "Expected three FRAMES streams")
    require(plan["seed_formulas"] == {
        "initialization": "200000+10*channel_id+initialization_index",
        "sampling": "300000+100*channel_id+10*initialization_index+sampling_index",
        "subset": "400000+channel_id"}, "Unsupported seed formulas")
    require(plan["identity"] == {
        "cohort_template": "n2-full-calibration-{printed_config_id}__{frames_config_id}-c7a8c97",
        "trial_template": "qpt-cal-n2f-c{channel}-i{initialization_index}-s{sampling_index}-{pair_id}-20261005-r2"
    }, "Unsupported identity templates")
    shared = plan["shared"]
    expected_shared = {
        "n_qubits": 2, "active_rows": 128, "rank": 1, "subset_size": 0,
        "observation_mode": "noiseless", "noise_std": 0, "noise_seed": 0,
        "shot_seed": 0, "row_epochs": 2000, "checkpoint_epochs": 100,
        "batch_size": 32, "tau": 2, "frames_penalty_exponent": 0.25,
        "spectral_norm_method": "dense", "dense_max_process_dimension": 1024,
        "spectral_norm_tolerance": 1e-10,
        "rho_schedule": {"scale": 2, "offset": 4, "exponent": 0.6},
        "step_schedule": {"scale": 2, "offset": 2, "exponent": 1},
        "gate": {"name": "noiseless_recovery_v1", "fidelity_min": 0.99,
                 "rmse_max": 1e-6, "normalized_tp_max": 0.001}}
    require(shared == expected_shared, "Shared protocol differs from reviewed plan")
    require(plan["resources"] == {
        "account": "fwllm", "partition": "cpu_short", "nodes": 1, "tasks": 1,
        "cpus_per_task": 2, "memory": "1G", "walltime_seconds": 1800,
        "array": "0-7%4", "minimum_seconds_before_new_trial": 60},
        "Resource plan differs from reviewed allocation")
    printed = {row["id"]: row for row in plan["printed_candidates"]}
    frames = {row["id"]: row for row in plan["frames_candidates"]}
    alphas = [0.5, 0.05, 0.005, 0.0005, 0.00005]
    etas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
    require(len(printed) == len(plan["printed_candidates"]) == 30,
            "Printed candidate IDs must be unique")
    require(len(frames) == len(plan["frames_candidates"]) == 5,
            "FRAMES candidate IDs must be unique")
    require({(row["alpha"], row["eta_scale"]) for row in printed.values()}
            == {(a, e) for a in alphas for e in etas}, "Printed grid changed")
    require({row["alpha0"] for row in frames.values()} == set(alphas), "FRAMES grid changed")
    q_anchor = plan["anchors"]["printed_config_id"]
    f_anchor = plan["anchors"]["frames_config_id"]
    require((printed[q_anchor]["alpha"], printed[q_anchor]["eta_scale"])
            == (0.05, 0.0001), "Printed anchor changed")
    require(frames[f_anchor]["alpha0"] == 0.005, "FRAMES anchor changed")
    pairs = plan["pair_definitions"]
    require(len(pairs) == len({p["pair_id"] for p in pairs}) == 34,
            "Expected 34 unique pair definitions")
    expected_pairs = {(q, f_anchor) for q in printed} | {(q_anchor, f) for f in frames}
    require({(p["printed_config_id"], p["frames_config_id"]) for p in pairs}
            == expected_pairs, "Pair definitions do not cover the independent star grid")
    for pair in pairs:
        expected_samples = [0, 1, 2] if pair["printed_config_id"] == q_anchor else [0]
        require(pair["sampling_indices"] == expected_samples, "Pair replication changed")
    inventory = []
    for channel in plan["channels"]:
        for sample in plan["sampling_indices"]:
            for pair in pairs:
                if sample not in pair["sampling_indices"]:
                    continue
                row = dict(pair, channel=channel, initialization_index=0,
                           sampling_index=sample, initialization_seed=200000 + 10 * channel,
                           sampling_seed=300000 + 100 * channel + sample,
                           subset_seed=400000 + channel)
                row.pop("sampling_indices")
                row["printed"] = printed[pair["printed_config_id"]]
                row["frames"] = frames[pair["frames_config_id"]]
                row["cohort_id"] = plan["identity"]["cohort_template"].format(**row)
                row["trial_id"] = plan["identity"]["trial_template"].format(**row)
                inventory.append(row)
    require(len(inventory) == len({r["trial_id"] for r in inventory}) == 352,
            "Expected 352 unique scientific trials")
    require(set(collections.Counter(r["channel"] for r in inventory).values()) == {44},
            "Expected 44 trials per channel")
    q_keys = {(r["channel"], r["initialization_index"], r["printed_config_id"])
              for r in inventory}
    f_keys = {(r["channel"], r["initialization_index"], r["sampling_index"],
               r["frames_config_id"]) for r in inventory}
    require(plan["expected_counts"] == {
        "pair_definitions": 34, "paired_trials_per_channel": 44,
        "total_paired_trials": 352, "unique_printed_outcomes": len(q_keys),
        "unique_frames_outcomes": len(f_keys), "printed_executions": 352,
        "frames_executions": 352}, "Declared inventory counts disagree")
    require((len(q_keys), len(f_keys)) == (240, 120), "Native candidate coverage is incomplete")
    return inventory


def command_for(plan, trial, out):
    p = plan["shared"]
    values = {
        "n-qubits": p["n_qubits"], "row-epochs": p["row_epochs"],
        "checkpoint-epochs": p["checkpoint_epochs"],
        "printed-tp-weight-per-row": trial["printed"]["alpha"],
        "frames-initial-tp-weight-per-row": trial["frames"]["alpha0"],
        "frames-penalty-exponent": p["frames_penalty_exponent"],
        "eta-scale": trial["printed"]["eta_scale"], "rank": p["rank"],
        "tau": p["tau"], "batch-size": p["batch_size"],
        "subset-size": p["subset_size"], "subset-seed": trial["subset_seed"],
        "channel-seed": trial["channel"], "initialization-seed": trial["initialization_seed"],
        "sampling-seed": trial["sampling_seed"], "observation-mode": p["observation_mode"],
        "noise-std": p["noise_std"], "noise-seed": p["noise_seed"],
        "shot-seed": p["shot_seed"], "spectral-norm-method": p["spectral_norm_method"],
        "dense-max-process-dimension": p["dense_max_process_dimension"],
        "spectral-norm-tolerance": p["spectral_norm_tolerance"],
        "gate-name": p["gate"]["name"], "gate-fidelity-min": p["gate"]["fidelity_min"],
        "gate-rmse-max": p["gate"]["rmse_max"],
        "gate-normalized-tp-max": p["gate"]["normalized_tp_max"],
        "cohort-id": trial["cohort_id"], "trial-id": trial["trial_id"],
        "split": "calibration", "expected-commit": COMMIT, "save": str(out / "pair.npz")}
    for name in ("rho", "step"):
        for key, value in p[name + "_schedule"].items():
            values[f"{name}-{key}"] = value
    command = [sys.executable, "-m", "paper.experiments.qpt_quiroga_direct_comparison"]
    for flag, value in values.items():
        command.extend(["--" + flag, str(value)])
    command.extend(["--require-clean", "--quiet", "--fail-on-method-failure"])
    return command


def validate_archive(plan, trial, out, returncode):
    # Imported only during a scheduler allocation, never for --validate-only.
    import numpy as np
    from paper.experiments.qpt_quiroga_direct_comparison import (
        _array_sha256, aggregate_sidecars)

    require(returncode in (0, 2), f"Infrastructure/runner exit {returncode}")
    metadata = json.loads((out / "pair.json").read_text(encoding="utf-8"))
    aggregate_sidecars([metadata])
    comparison, config = metadata["comparison"], metadata["configuration"]
    require(metadata["provenance"]["git"]["commit"] == COMMIT, "Archive source mismatch")
    require(metadata["provenance"]["git"]["worktree_dirty"] is False, "Dirty archive source")
    require(comparison["trial_id"] == trial["trial_id"]
            and comparison["cohort_id"] == trial["cohort_id"], "Trial identity mismatch")
    require(comparison["split"] == "calibration"
            and comparison["selection_provenance"] is None, "Unexpected selection identity")
    expected_config = {
        "n_steps": 2000, "frames_steps": 8000, "metrics_frequency": 100,
        "frames_metrics_frequency": 400, "rank": 1, "batch_size": 32, "tau": 2,
        "channel_seed": trial["channel"], "initialization_seed": trial["initialization_seed"],
        "sampling_seed": trial["sampling_seed"], "subset_seed": trial["subset_seed"],
        "subset_size": 0, "observation_mode": "noiseless", "noise_std": 0,
        "noise_seed": 0, "shot_seed": 0, "shots": None,
        "eta_scale": trial["printed"]["eta_scale"],
        "printed_tp_weight_per_row_alpha": trial["printed"]["alpha"],
        "frames_initial_tp_weight_per_row_alpha": trial["frames"]["alpha0"],
        "frames_smoothing_exponent": 0.25, "spectral_norm_method": "dense",
        "spectral_norm_tolerance": 1e-10, "dense_max_process_dimension": 1024,
        "rho_schedule": {"type": "PowerSchedule", "scale": 2.0, "offset": 4.0,
                         "exponent": 0.6, "cap": 1.0},
        "step_size_schedule": {"type": "PowerSchedule", "scale": 2.0, "offset": 2.0,
                               "exponent": 1.0, "cap": 1.0}}
    for name, value in expected_config.items():
        require(config[name] == value, f"Configuration mismatch: {name}")
    statuses = {method: metadata["methods"][method]["status"] for method in METHODS}
    require(set(statuses.values()) <= {"complete", "failed"}, "Unrecognized method status")
    has_failure = "failed" in statuses.values()
    require((returncode == 2) == has_failure, "Runner exit and method status disagree")
    nonfinite_failed_arrays = []
    with np.load(out / "pair.npz", allow_pickle=False) as archive:
        require(json.loads(archive["metadata_json"].item()) == metadata,
                "Embedded NPZ metadata differs from JSON")
        for name, shape, dtype, hash_key in (
            ("active_rows", (128,), "int64", "active_rows_sha256"),
            ("active_observations", (128,), "float64", "active_observations_sha256"),
            ("initial_factor", (16, 1), "complex128", "initial_factor_sha256"),
            ("truth_factor", (16, 1), "complex128", "truth_factor_sha256")):
            array = archive[name]
            require(array.shape == shape and str(array.dtype) == dtype,
                    f"Unexpected shape/dtype: {name}")
            require(_array_sha256(array) == metadata[hash_key], f"Hash mismatch: {name}")
        require(np.array_equal(archive["active_rows"], np.arange(128)), "Not full n2 rows")
        for name in archive.files:
            array = archive[name]
            if array.dtype.kind in "fci" and not np.isfinite(array).all():
                failed_prefixes = tuple(m.replace("-", "_") + "_" for m in METHODS
                                        if statuses[m] == "failed")
                require(failed_prefixes and name.startswith(failed_prefixes)
                        and not name.endswith("final_factor"),
                        f"Nonfinite array outside a failed trajectory: {name}")
                nonfinite_failed_arrays.append(name)
        for method in METHODS:
            prefix = method.replace("-", "_")
            endpoint = metadata["methods"][method]
            diagnostics = comparison["endpoint_diagnostics"][method]
            factor = archive[prefix + "_final_factor"]
            require(factor.shape == (16, 1) and str(factor.dtype) == "complex128",
                    f"Unexpected final factor shape/dtype: {method}")
            require(_array_sha256(factor) == diagnostics["final_factor_sha256"],
                    f"Endpoint factor hash mismatch: {method}")
            steps = archive[prefix + "_checkpoint_steps"]
            accesses = archive[prefix + "_cumulative_data_row_accesses"]
            require(steps.ndim == 1 and steps.size and steps[0] == 0
                    and np.all(np.diff(steps) > 0), f"Invalid checkpoint axis: {method}")
            require(int(steps[-1]) == endpoint["completed_steps"], "Endpoint step mismatch")
            require(accesses.shape == steps.shape
                    and int(accesses[-1]) == endpoint["cumulative_data_row_accesses"],
                    "Endpoint accounting mismatch")
            if statuses[method] == "complete":
                stride = 100 if method == "printed-adafgd" else 400
                require(np.array_equal(steps, np.arange(21) * stride), "Missing checkpoints")
                require(np.array_equal(accesses, np.arange(21) * 12800),
                        "Logical-row checkpoints disagree")
            else:
                require(endpoint["failure_reason"] and not comparison["method_gate"][method]["passed"],
                        "Failure reason/gate was not preserved")
            for metric in PHYSICAL_METRICS:
                array = archive[prefix + "_" + metric]
                require(array.shape == steps.shape, f"Metric length mismatch: {method}/{metric}")
                value = endpoint[metric]
                require((value is None and not np.isfinite(array[-1]))
                        or (value is not None and math.isclose(float(array[-1]), value,
                            rel_tol=1e-12, abs_tol=1e-15)), f"Endpoint metric mismatch: {metric}")
    return {"method_statuses": statuses, "method_gate": comparison["method_gate"],
            "data_instance_sha256": comparison["data_instance_sha256"],
            "paired_instance_sha256": comparison["paired_instance_sha256"],
            "nonfinite_arrays_retained_only_for_failed_methods": nonfinite_failed_arrays,
            "validation": "metadata coherence, candidate identity, embedded metadata, shapes/dtypes, array hashes, finite successful trajectories and endpoint/row accounting"}


def run_channel(args, plan, inventory, plan_bytes):
    require(os.environ.get("SLURM_JOB_ID"), "Experiments require a Slurm allocation")
    require(int(os.environ["SLURM_ARRAY_TASK_ID"]) == args.channel, "Array/channel mismatch")
    require(args.channel in plan["channels"], "Unexpected calibration channel")
    require(args.deadline_epoch is not None and args.deadline_epoch > time.time(),
            "A future allocation deadline is required")
    repo = Path(plan["source"]["repo"])
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                   text=True).strip() == COMMIT, "Checkout commit mismatch")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=repo, text=True).strip(), "Checkout is dirty")
    sys.path.insert(0, str(repo))
    out = args.out
    require(out is not None and out.is_dir(), "Launcher must create a fresh output directory")
    require(not any(out.iterdir()), "Refusing to reuse nonempty channel output")
    trials = [dict(t, execution_status="pending_not_run") for t in inventory
              if t["channel"] == args.channel]
    record = {
        "schema_version": 1, "campaign_id": plan["campaign_id"],
        "kind": "calibration_execution_inventory_not_selection_manifest",
        "channel": args.channel, "started_utc": utc_now(), "status": "running",
        "git_commit": COMMIT, "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "wrapper_sha256": file_hash(__file__), "hostname": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
        "deadline_epoch": args.deadline_epoch, "expected_trials": 44, "trials": trials}
    with (out / "calibration_plan.json").open("xb") as stream:
        stream.write(plan_bytes)
    write_json(out / "inventory.json", record)
    final_exit = 0
    for trial in trials:
        remaining = args.deadline_epoch - time.time()
        if remaining <= plan["resources"]["minimum_seconds_before_new_trial"]:
            record["status"] = "incomplete_time_guard"
            final_exit = 75
            break
        trial_out = out / trial["trial_id"]
        trial_out.mkdir()
        command = command_for(plan, trial, trial_out)
        trial.update(execution_status="running", started_utc=utc_now(),
                     output_directory=str(trial_out), command=command)
        write_json(out / "inventory.json", record, replace=True)
        result = {"trial_id": trial["trial_id"], "started_utc": trial["started_utc"],
                  "command": command, "runner_exit_code": None}
        try:
            with (trial_out / "stdout.log").open("xb") as stdout, \
                    (trial_out / "stderr.log").open("xb") as stderr:
                completed = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr,
                                           timeout=max(1, remaining - 10), check=False)
            result["runner_exit_code"] = completed.returncode
            result["archive_validation"] = validate_archive(plan, trial, trial_out,
                                                             completed.returncode)
            result["status"] = "method_failure_archived" if completed.returncode == 2 else "complete"
            if completed.returncode == 2:
                final_exit = 2
        except Exception as error:
            result.update(status="infrastructure_failure_or_invalid_archive",
                          error=f"{type(error).__name__}: {error}")
            final_exit = 3
        result["finished_utc"] = utc_now()
        result["artifacts"] = {
            name: {"sha256": file_hash(trial_out / name), "bytes": (trial_out / name).stat().st_size}
            for name in ("pair.json", "pair.npz", "stdout.log", "stderr.log")
            if (trial_out / name).is_file()}
        write_json(trial_out / "execution_status.json", result)
        trial.update(execution_status=result["status"], finished_utc=result["finished_utc"],
                     runner_exit_code=result["runner_exit_code"],
                     status_record_sha256=file_hash(trial_out / "execution_status.json"))
        print(json.dumps({"trial_id": trial["trial_id"], "status": result["status"],
                          "runner_exit_code": result["runner_exit_code"]}), flush=True)
        write_json(out / "inventory.json", record, replace=True)
        if final_exit == 3:
            record["status"] = "incomplete_infrastructure_or_invalid_archive"
            break
    else:
        record["status"] = "all_trials_archived_with_method_failures" if final_exit == 2 else "all_trials_archived"
    record.update(finished_utc=utc_now(), scheduler_exit_code=final_exit,
                  trial_status_counts=dict(collections.Counter(t["execution_status"] for t in trials)))
    write_json(out / "inventory.json", record, replace=True)
    print(json.dumps({"status": record["status"], "scheduler_exit_code": final_exit,
                      "trial_status_counts": record["trial_status_counts"]}), flush=True)
    return final_exit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    inventory = make_inventory(plan)
    if args.validate_only:
        print(json.dumps({"status": "inventory_valid", "total_trials": len(inventory),
                          "counts": plan["expected_counts"],
                          "plan_sha256": hashlib.sha256(plan_bytes).hexdigest()}, sort_keys=True))
        return 0
    return run_channel(args, plan, inventory, plan_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
