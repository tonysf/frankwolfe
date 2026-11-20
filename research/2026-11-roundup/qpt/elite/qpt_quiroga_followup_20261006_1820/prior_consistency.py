"""Read-only consistency checks against the fully audited original pilot."""
import json
from pathlib import Path

import run_followup as runner
from work_accounting import check_work


def load_prior(path, base):
    report = json.loads(path.read_text(encoding="utf-8"))
    runner.require(report["kind"] == "qpt_diagnostic_result_audit"
                   and report["audit_passed"] is True
                   and report["all_results_present"] is True
                   and report["all_inventory_checksums_verified"] is True
                   and report["audited_trials"] == report["expected_trials"] == 60
                   and not report["missing_trial_ids"]
                   and not report["trial_ids_without_inventory_checksum"],
                   "The prior pilot must be fully audited with all inventory checksums")
    runner.require(report["provenance"]["source_commit"] == base.COMMIT
                   and report["provenance"]["runner_sha256"] == base.file_hash(Path(base.__file__)),
                   "Prior pilot source/base-harness provenance differs")
    return report


def matching_case(report, trial, default_step):
    candidates = []
    for case in report["cases"]:
        prior = case["trial"]
        if (all(prior[key] == trial[key] for key in ("channel", "method", "alpha", "eta_scale", "tau"))
                and prior.get("step_schedule", default_step) == trial.get("step_schedule", default_step)):
            candidates.append(case)
    runner.require(len(candidates) == 1, "Expected exactly one matching prior pilot case")
    return candidates[0]


def compare_arrays(current, prior, method, np, metrics):
    """Compare shared checkpoint physics, excluding timing and native objectives."""
    prefix = method.replace("-", "_")
    new_steps = current[prefix + "_checkpoint_steps"]
    old_steps = prior[prefix + "_checkpoint_steps"]
    common, new_indices, old_indices = np.intersect1d(new_steps, old_steps, return_indices=True)
    runner.require(common.size > 0 and common[0] == 0, "No shared initial checkpoint")
    if method in {"printed-adafgd", "frames"}:
        previous_horizon = 10000 if method == "printed-adafgd" else 40000
        runner.require(int(old_steps[-1]) == previous_horizon and previous_horizon in common,
                       "Extended run does not include the original completed horizon")
    elif min(int(new_steps[-1]), int(old_steps[-1])) >= 100:
        runner.require(common.size >= 2, "Missing noninitial shared Armijo checkpoint")
    deviations = {}
    for metric in metrics:
        new_values = current[prefix + "_" + metric][new_indices]
        old_values = prior[prefix + "_" + metric][old_indices]
        runner.require(np.all(np.isfinite(new_values)) and np.all(np.isfinite(old_values)),
                       "Undefined shared checkpoint prevents trajectory comparison: " + metric)
        runner.require(np.allclose(new_values, old_values, atol=1e-12, rtol=1e-10),
                       "Prior checkpoint trajectory differs: " + metric)
        deviations[metric] = float(np.max(np.abs(new_values - old_values)))
    return {"matching_checkpoint_count": int(common.size), "matching_checkpoint_steps": common.tolist(),
            "compared_metrics": list(metrics), "absolute_tolerance": 1e-12, "relative_tolerance": 1e-10,
            "max_absolute_deviation": deviations,
            "excluded_fields": ["optimizer_seconds", "penalized_objective", "terminal_status", "terminal_work_charge"],
            "qualification": "Only common accepted-step checkpoints are compared. Armijo's new terminal stationarity check adds a gradient row charge and may stop between old checkpoints; its status is not transferred to old artifacts."}


def check_prior(report, trial, identity, current_arrays, plan, base, np):
    case = matching_case(report, trial, plan["shared"]["step"])
    runner.require(case["shared_array_hashes"] == identity,
                   "Follow-up data, initial factor or truth differs from the original pilot")
    json_path, npz_path = Path(case["json_path"]), Path(case["npz_path"])
    runner.require(base.file_hash(json_path) == case["json_sha256"]
                   and base.file_hash(npz_path) == case["npz_sha256"], "Prior archive changed after audit")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    runner.require(metadata["diagnostic"]["trial"] == case["trial"], "Prior archive trial differs from audit")
    with np.load(npz_path, allow_pickle=False) as archive:
        runner.require(json.loads(str(archive["metadata_json"].item())) == metadata,
                       "Prior NPZ/JSON metadata differ")
        arrays = {key: archive[key] for key in archive.files}
    for key in identity:
        runner.require(np.array_equal(current_arrays[key], arrays[key]),
                       "Follow-up/prior shared arrays differ: " + key)
    comparison = compare_arrays(current_arrays, arrays, trial["method"], np, base.PHYSICAL_METRICS)
    return {"prior_trial_id": case["trial_id"], "prior_terminal_status": case["status"],
            "prior_json_sha256": case["json_sha256"], "prior_npz_sha256": case["npz_sha256"],
            "shared_data_initial_factor_truth_exact_match": True, **comparison}
