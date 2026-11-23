#!/usr/bin/env python3
"""Execute a frozen validation plan; never select or retune configurations.

Inventory validation and plan freezing use only the standard library.
Experiment and archive-validation imports are restricted to Slurm execution.
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
CAMPAIGN = "qpt_quiroga_n2_full_validation_20261005_2136"
REPO = "/gpfs/workdir/silvetian/frankwolfe-quiroga-" + COMMIT
OUTPUT_ROOT = "/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_" + COMMIT + "_20261005_02"
STAGED = "validation-stage-20261005-2136"
METHODS = ("printed-adafgd", "frames")
METRICS = ("measurement_loss_sum", "measurement_loss_mean", "rmse",
           "tp_violation", "normalized_tp_violation", "process_fidelity")
SHARED = {
    "n_qubits": 2, "active_rows": 128, "rank": 1, "subset_size": 0,
    "observation_mode": "noiseless", "noise_std": 0, "noise_seed": 0,
    "shot_seed": 0, "row_epochs": 10000, "checkpoint_epochs": 100,
    "batch_size": 32, "tau": 2, "frames_penalty_exponent": 0.25,
    "spectral_norm_method": "dense", "dense_max_process_dimension": 1024,
    "spectral_norm_tolerance": 1e-10,
    "rho_schedule": {"scale": 2, "offset": 4, "exponent": 0.6},
    "step_schedule": {"scale": 2, "offset": 2, "exponent": 1},
    "gate": {"name": "noiseless_recovery_v1", "fidelity_min": 0.99,
             "rmse_max": 1e-6, "normalized_tp_max": 0.001}}
RESOURCES = {
    "account": "fwllm", "partition": "cpu_short", "nodes": 1, "tasks": 1,
    "cpus_per_task": 2, "memory": "1G", "walltime_seconds": 1200,
    "array": "0-7%1", "minimum_seconds_before_new_trial": 120}
SEEDS = {
    "initialization": "200000+10*channel_id+initialization_index",
    "sampling": "300000+100*channel_id+10*initialization_index+sampling_index",
    "subset": "400000+channel_id"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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


def selected_configs(manifest):
    """Read winners already frozen by the calibration selector, not rank them."""
    require(manifest["schema_version"] == 1
            and manifest["kind"] == "qpt_method_native_selection_manifest"
            and manifest["regime_id"] == "n2_full_128"
            and manifest["selection_rule_id"] == "gate_first_common_physical_metrics_v1",
            "Unsupported selection manifest")
    require(manifest["source_commit"] == COMMIT, "Selection source differs")
    selected = manifest["selected_candidates"]
    q, f = selected["printed-adafgd"], selected["frames"]
    require(manifest["printed_config_id"] == q["id"]
            and manifest["frames_config_id"] == f["id"], "Manifest selected IDs disagree")
    calibration_shared = dict(manifest["shared_calibration_configuration"])
    require(calibration_shared["row_epochs"] == 2000, "Unexpected calibration horizon")
    calibration_shared["row_epochs"] = 10000
    require(calibration_shared == SHARED, "Selection/validation shared settings differ")
    values = {"printed-adafgd": {"config_id": q["id"], "alpha": q["alpha"],
                                "eta_scale": q["eta_scale"]},
              "frames": {"config_id": f["id"], "alpha0": f["alpha0"]}}
    for method, candidate in values.items():
        require(isinstance(candidate["config_id"], str) and candidate["config_id"],
                "Selected candidate needs a nonempty ID")
        for key, value in candidate.items():
            if key != "config_id":
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value > 0,
                        "Selected hyperparameters must be finite and positive")
    return values


def frozen_plan(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1, "kind": "frozen_confirmatory_validation_plan",
        "campaign_id": CAMPAIGN, "frozen_utc": now(),
        "scientific_role": "Confirm the independently selected calibration configurations on separate validation channels; never rerank or retune on these outcomes.",
        "source": {"commit": COMMIT, "repo": REPO, "require_clean": True},
        "output_root": OUTPUT_ROOT, "staged_files_directory": STAGED,
        "selection": {"manifest_filename": "selection_manifest.json",
                      "manifest_sha256": file_hash(manifest_path),
                      "selected": selected_configs(manifest)},
        "shared": SHARED, "resources": RESOURCES,
        "environment": {"module": "anaconda3/2023.09-0/none-none",
                        "conda_environment": "qpt", "threads": 2},
        "channels": list(range(100, 108)), "initialization_indices": [0],
        "sampling_indices": [0, 1, 2], "seed_formulas": SEEDS,
        "array_channel_mapping": "channel_id=100+SLURM_ARRAY_TASK_ID",
        "expected_counts": {"paired_trials": 24, "paired_trials_per_channel": 3,
                            "unique_printed_outcomes": 8, "unique_frames_outcomes": 24,
                            "printed_executions": 24, "frames_executions": 24},
        "matched_budget": {"nominal_row_accesses_per_method": 1280000,
                           "printed_steps": 10000, "frames_steps": 40000,
                           "checkpoints_including_initial": 101},
        "failure_policy": {"valid_runner_exits": [0, 2],
                           "exit2": "Continue only after both artifacts validate and at least one method is failed; retain this outcome in the gate denominator.",
                           "invalid_archive_or_infrastructure": "Preserve logs, partial artifacts and entire pending inventory, then stop the channel with exit3.",
                           "time_guard": "Do not begin a new trial with120seconds or less remaining; retain pending inventory and exit75.",
                           "no_overwrite": True},
        "analysis_policy": {"no_selection_or_reranking": True,
                            "printed_deduplication": "Verify matching non-timing endpoints and final-factor hashes across sampling streams; count once per channel/initialization.",
                            "primary_gate_estimand": "Mean sampling pass rate within initialization, then equal-weight mean across channels; failed runs remain in denominators.",
                            "paired_scores": "Physical metrics only; native penalized objectives are not paired scores."}}


def make_inventory(plan, manifest_path):
    require(plan["schema_version"] == 1 and plan["kind"] == "frozen_confirmatory_validation_plan",
            "Not a frozen validation plan")
    require(plan["campaign_id"] == CAMPAIGN, "Unexpected validation campaign")
    require(plan["source"] == {"commit": COMMIT, "repo": REPO, "require_clean": True},
            "Unexpected source pin")
    require(plan["output_root"] == OUTPUT_ROOT and plan["staged_files_directory"] == STAGED,
            "Unexpected validation destinations")
    require(plan["shared"] == SHARED and plan["resources"] == RESOURCES,
            "Validation protocol/resources differ from frozen design")
    require(plan["expected_counts"] == {"paired_trials": 24, "paired_trials_per_channel": 3,
            "unique_printed_outcomes": 8, "unique_frames_outcomes": 24,
            "printed_executions": 24, "frames_executions": 24}, "Declared validation counts differ")
    require(plan["matched_budget"] == {"nominal_row_accesses_per_method": 1280000,
            "printed_steps": 10000, "frames_steps": 40000, "checkpoints_including_initial": 101},
            "Declared matched budget differs")
    require(plan["environment"] == {"module": "anaconda3/2023.09-0/none-none",
            "conda_environment": "qpt", "threads": 2}, "Environment differs")
    require(plan["array_channel_mapping"] == "channel_id=100+SLURM_ARRAY_TASK_ID"
            and plan["selection"]["manifest_filename"] == "selection_manifest.json",
            "Array/manifest identity differs")
    require(plan["failure_policy"]["valid_runner_exits"] == [0, 2]
            and plan["failure_policy"]["no_overwrite"] is True
            and plan["analysis_policy"]["no_selection_or_reranking"] is True, "Safety policy differs")
    require(plan["channels"] == list(range(100, 108))
            and plan["initialization_indices"] == [0] and plan["sampling_indices"] == [0, 1, 2]
            and plan["seed_formulas"] == SEEDS, "Validation sample inventory changed")
    require(file_hash(manifest_path) == plan["selection"]["manifest_sha256"],
            "Selection manifest hash differs from frozen plan")
    selected = selected_configs(json.loads(manifest_path.read_text(encoding="utf-8")))
    require(selected == plan["selection"]["selected"], "Frozen parameters differ from selection")
    q, f = selected["printed-adafgd"], selected["frames"]
    cohort = "n2-full-validation-" + q["config_id"] + "__" + f["config_id"] + "-c7a8c97-" + plan["selection"]["manifest_sha256"][:12]
    inventory = [{"channel": c, "array_index": c - 100, "initialization_index": 0,
                  "sampling_index": s, "initialization_seed": 200000 + 10 * c,
                  "sampling_seed": 300000 + 100 * c + s, "subset_seed": 400000 + c,
                  "printed_config_id": q["config_id"], "frames_config_id": f["config_id"],
                  "printed": q, "frames": f, "cohort_id": cohort,
                  "trial_id": f"qpt-val-n2f-c{c}-i0-s{s}-20261005-2136"}
                 for c in plan["channels"] for s in plan["sampling_indices"]]
    require(len(inventory) == len({r["trial_id"] for r in inventory}) == 24,
            "Validation requires24 unique paired trials")
    return inventory


def command_for(plan, trial, out):
    p, selection = plan["shared"], plan["selection"]
    values = {
        "n-qubits": p["n_qubits"], "row-epochs": p["row_epochs"],
        "checkpoint-epochs": p["checkpoint_epochs"],
        "printed-tp-weight-per-row": trial["printed"]["alpha"],
        "frames-initial-tp-weight-per-row": trial["frames"]["alpha0"],
        "frames-penalty-exponent": p["frames_penalty_exponent"],
        "eta-scale": trial["printed"]["eta_scale"], "rank": p["rank"], "tau": p["tau"],
        "batch-size": p["batch_size"], "subset-size": p["subset_size"],
        "subset-seed": trial["subset_seed"], "channel-seed": trial["channel"],
        "initialization-seed": trial["initialization_seed"], "sampling-seed": trial["sampling_seed"],
        "observation-mode": p["observation_mode"], "noise-std": p["noise_std"],
        "noise-seed": p["noise_seed"], "shot-seed": p["shot_seed"],
        "spectral-norm-method": p["spectral_norm_method"],
        "dense-max-process-dimension": p["dense_max_process_dimension"],
        "spectral-norm-tolerance": p["spectral_norm_tolerance"],
        "gate-name": p["gate"]["name"], "gate-fidelity-min": p["gate"]["fidelity_min"],
        "gate-rmse-max": p["gate"]["rmse_max"], "gate-normalized-tp-max": p["gate"]["normalized_tp_max"],
        "selection-manifest-sha256": selection["manifest_sha256"],
        "printed-config-id": trial["printed_config_id"], "frames-config-id": trial["frames_config_id"],
        "cohort-id": trial["cohort_id"], "trial-id": trial["trial_id"], "split": "validation",
        "expected-commit": COMMIT, "save": str(out / "pair.npz")}
    for name in ("rho", "step"):
        for key, value in p[name + "_schedule"].items():
            values[f"{name}-{key}"] = value
    return [sys.executable, "-m", "paper.experiments.qpt_quiroga_direct_comparison",
            *(arg for flag, value in values.items() for arg in ("--" + flag, str(value))),
            "--require-clean", "--quiet", "--fail-on-method-failure"]


def validate_archive(plan, trial, out, returncode):
    import numpy as np
    from paper.experiments.qpt_quiroga_direct_comparison import _array_sha256, aggregate_sidecars

    require(returncode in (0, 2), f"Infrastructure/runner exit{returncode}")
    metadata = json.loads((out / "pair.json").read_text(encoding="utf-8"))
    aggregate_sidecars([metadata])
    comparison, config = metadata["comparison"], metadata["configuration"]
    require(metadata["provenance"]["git"]["commit"] == COMMIT
            and metadata["provenance"]["git"]["worktree_dirty"] is False, "Source provenance mismatch")
    require(comparison["trial_id"] == trial["trial_id"] and comparison["cohort_id"] == trial["cohort_id"],
            "Validation trial identity mismatch")
    require(comparison["split"] == "validation" and comparison["selection_provenance"] == {
        "selection_manifest_sha256": plan["selection"]["manifest_sha256"],
        "printed_config_id": trial["printed_config_id"], "frames_config_id": trial["frames_config_id"]},
        "Selection manifest/config IDs differ from frozen validation")
    expected = {
        "n_steps": 10000, "frames_steps": 40000, "metrics_frequency": 100,
        "frames_metrics_frequency": 400, "rank": 1, "batch_size": 32, "tau": 2,
        "channel_seed": trial["channel"], "initialization_seed": trial["initialization_seed"],
        "sampling_seed": trial["sampling_seed"], "subset_seed": trial["subset_seed"],
        "subset_size": 0, "observation_mode": "noiseless", "noise_std": 0,
        "noise_seed": 0, "shot_seed": 0, "shots": None,
        "eta_scale": trial["printed"]["eta_scale"],
        "printed_tp_weight_per_row_alpha": trial["printed"]["alpha"],
        "frames_initial_tp_weight_per_row_alpha": trial["frames"]["alpha0"],
        "frames_smoothing_exponent": 0.25, "spectral_norm_method": "dense",
        "spectral_norm_tolerance": 1e-10, "dense_max_process_dimension": 1024}
    for name in ("rho", "step"):
        expected["rho_schedule" if name == "rho" else "step_size_schedule"] = dict(
            plan["shared"][name + "_schedule"], type="PowerSchedule", cap=1.0)
    for key, value in expected.items():
        require(config[key] == value, "Unexpected configuration field:" + key)
    statuses = {method: metadata["methods"][method]["status"] for method in METHODS}
    require(set(statuses.values()) <= {"complete", "failed"}, "Unrecognized method status")
    require((returncode == 2) == ("failed" in statuses.values()), "Exit disagrees with method status")
    nonfinite = []
    with np.load(out / "pair.npz", allow_pickle=False) as archive:
        require(json.loads(archive["metadata_json"].item()) == metadata, "Embedded metadata differs")
        for key, shape, dtype in (("active_rows", (128,), "int64"),
                                 ("active_observations", (128,), "float64"),
                                 ("initial_factor", (16, 1), "complex128"),
                                 ("truth_factor", (16, 1), "complex128")):
            a = archive[key]
            require(a.shape == shape and str(a.dtype) == dtype, "Shape/dtype mismatch:" + key)
            require(_array_sha256(a) == metadata[key + "_sha256"], "Array hash mismatch:" + key)
        require(np.array_equal(archive["active_rows"], np.arange(128)), "Not full n2 rows")
        for key in archive.files:
            a = archive[key]
            if a.dtype.kind in "fci" and not np.isfinite(a).all():
                failed = tuple(m.replace("-", "_") + "_" for m in METHODS if statuses[m] == "failed")
                require(failed and key.startswith(failed) and not key.endswith("final_factor"),
                        "Nonfinite value outside failed trace:" + key)
                nonfinite.append(key)
        for method in METHODS:
            prefix, endpoint = method.replace("-", "_"), metadata["methods"][method]
            factor = archive[prefix + "_final_factor"]
            require(factor.shape == (16, 1) and str(factor.dtype) == "complex128", "Invalid final factor")
            require(_array_sha256(factor) == comparison["endpoint_diagnostics"][method]["final_factor_sha256"],
                    "Final factor hash mismatch")
            steps, work = archive[prefix + "_checkpoint_steps"], archive[prefix + "_cumulative_data_row_accesses"]
            require(steps.ndim == 1 and steps.size and steps[0] == 0 and np.all(np.diff(steps) > 0),
                    "Invalid checkpoint axis")
            require(int(steps[-1]) == endpoint["completed_steps"] and work.shape == steps.shape
                    and int(work[-1]) == endpoint["cumulative_data_row_accesses"], "Endpoint accounting mismatch")
            if statuses[method] == "complete":
                stride = 100 if method == "printed-adafgd" else 400
                require(np.array_equal(steps, np.arange(101) * stride)
                        and np.array_equal(work, np.arange(101) * 12800), "Complete trial missed checkpoints/budget")
            else:
                require(endpoint["failure_reason"] and not comparison["method_gate"][method]["passed"],
                        "Failed method lacks reason or fails gate accounting")
            for metric in METRICS:
                a, value = archive[prefix + "_" + metric], endpoint[metric]
                require(a.shape == steps.shape and ((value is None and not np.isfinite(a[-1]))
                        or (value is not None and math.isclose(float(a[-1]), value, rel_tol=1e-12, abs_tol=1e-15))),
                        "Endpoint/trace metric mismatch:" + metric)
    return {"method_statuses": statuses, "method_gate": comparison["method_gate"],
            "nonfinite_arrays_retained_only_for_failed_methods": nonfinite,
            "paired_instance_sha256": comparison["paired_instance_sha256"],
            "data_instance_sha256": comparison["data_instance_sha256"]}


def run_channel(args, plan, inventory, plan_bytes, manifest_bytes):
    require(os.environ.get("SLURM_JOB_ID"), "Experiments require a Slurm allocation")
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    require(args.array_index == index and 0 <= index < 8, "Array index mismatch")
    require(args.deadline_epoch is not None and args.deadline_epoch > time.time(), "Missing future deadline")
    repo = Path(plan["source"]["repo"])
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == COMMIT,
            "Checkout commit mismatch")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=repo, text=True).strip(), "Checkout is dirty")
    sys.path.insert(0, str(repo))
    out = args.out
    require(out is not None and out.is_dir() and not any(out.iterdir()), "Refusing nonempty/missing output")
    trials = [dict(t, execution_status="pending_not_run") for t in inventory if t["array_index"] == index]
    record = {"schema_version": 1, "kind": "validation_execution_inventory",
              "campaign_id": CAMPAIGN, "channel": 100 + index, "status": "running",
              "started_utc": now(), "git_commit": COMMIT, "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
              "selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "wrapper_sha256": file_hash(__file__), "hostname": socket.gethostname(),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
              "slurm_array_task_id": index, "deadline_epoch": args.deadline_epoch, "expected_trials": 3, "trials": trials}
    for name, content in (("validation_plan.json", plan_bytes), ("selection_manifest.json", manifest_bytes)):
        with (out / name).open("xb") as stream:
            stream.write(content)
    write_json(out / "inventory.json", record)
    exitcode = 0
    for trial in trials:
        remaining = args.deadline_epoch - time.time()
        if remaining <= RESOURCES["minimum_seconds_before_new_trial"]:
            record["status"], exitcode = "incomplete_time_guard", 75
            break
        trial_out = out / trial["trial_id"]
        trial_out.mkdir()
        command = command_for(plan, trial, trial_out)
        trial.update(execution_status="running", started_utc=now(), output_directory=str(trial_out), command=command)
        write_json(out / "inventory.json", record, replace=True)
        result = {"trial_id": trial["trial_id"], "started_utc": trial["started_utc"], "command": command,
                  "runner_exit_code": None}
        try:
            with (trial_out / "stdout.log").open("xb") as stdout, (trial_out / "stderr.log").open("xb") as stderr:
                completed = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr,
                                           timeout=max(1, remaining - 10), check=False)
            result["runner_exit_code"] = completed.returncode
            result["archive_validation"] = validate_archive(plan, trial, trial_out, completed.returncode)
            result["status"] = "method_failure_archived" if completed.returncode == 2 else "complete"
            if completed.returncode == 2:
                exitcode = 2
        except Exception as error:
            result.update(status="infrastructure_failure_or_invalid_archive", error=f"{type(error).__name__}: {error}")
            exitcode = 3
        result["finished_utc"] = now()
        result["artifacts"] = {name: {"sha256": file_hash(trial_out / name), "bytes": (trial_out / name).stat().st_size}
                               for name in ("pair.json", "pair.npz", "stdout.log", "stderr.log") if (trial_out / name).is_file()}
        write_json(trial_out / "execution_status.json", result)
        trial.update(execution_status=result["status"], finished_utc=result["finished_utc"],
                     runner_exit_code=result["runner_exit_code"], status_record_sha256=file_hash(trial_out / "execution_status.json"))
        print(json.dumps({"trial_id": trial["trial_id"], "status": result["status"], "runner_exit_code": result["runner_exit_code"]}), flush=True)
        write_json(out / "inventory.json", record, replace=True)
        if exitcode == 3:
            record["status"] = "incomplete_infrastructure_or_invalid_archive"
            break
    else:
        record["status"] = "all_trials_archived_with_method_failures" if exitcode == 2 else "all_trials_archived"
    record.update(finished_utc=now(), scheduler_exit_code=exitcode,
                  trial_status_counts=dict(collections.Counter(t["execution_status"] for t in trials)))
    write_json(out / "inventory.json", record, replace=True)
    print(json.dumps({"status": record["status"], "scheduler_exit_code": exitcode,
                      "trial_status_counts": record["trial_status_counts"]}), flush=True)
    return exitcode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    args = parser.parse_args()
    if args.freeze_plan:
        plan = frozen_plan(args.manifest)
        inventory = make_inventory(plan, args.manifest)
        write_json(args.plan, plan)
        print(json.dumps({"status": "plan_frozen", "plan_sha256": file_hash(args.plan), "paired_trials": len(inventory)}))
        return 0
    plan_bytes, manifest_bytes = args.plan.read_bytes(), args.manifest.read_bytes()
    plan = json.loads(plan_bytes)
    inventory = make_inventory(plan, args.manifest)
    if args.validate_only:
        print(json.dumps({"status": "frozen_inventory_valid", "paired_trials": len(inventory),
                          "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                          "selection_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                          "selected": plan["selection"]["selected"]}, sort_keys=True))
        return 0
    return run_channel(args, plan, inventory, plan_bytes, manifest_bytes)


if __name__ == "__main__":
    raise SystemExit(main())
