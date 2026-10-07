#!/usr/bin/env python3
"""Audit initializer diagnostic factors and provenance without rerunning optimizers.

Adapted from 2026-10-06 audit_followup.py, SHA256
ce3427bc9a0e604c92a73e6e3541c47f06c7ad87d5af25844b584192916f9716.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import run_diagnostic as runner
import paired_reference
import support as base
from work_accounting import check_work


def array_hash(array, np):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--prior-audit", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--inventory", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    require = runner.require
    require(not args.output.exists(), "Refusing to overwrite report")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    trials = runner.validate_plan(plan)
    expected = {trial["id"]: trial for trial in trials}
    prior_report = paired_reference.load_prior(args.prior_audit, plan)
    source_hashes = runner.source_hashes(args.repo, plan["source"]["commit"], base)
    here = Path(__file__).resolve().parent
    provenance = {"campaign_id": plan["campaign_id"], "plan_sha256": base.file_hash(args.plan),
                  "runner_sha256": base.file_hash(here / "run_diagnostic.py"),
                  "plan_builder_sha256": base.file_hash(here / "freeze_plan.py"),
                  "source_commit": plan["source"]["commit"], "source_sha256": source_hashes}
    sys.path.insert(0, str(args.repo))
    import numpy as np
    from paper.experiments.qpt_quiroga_campaign import evaluate_common_metrics, _summed_objective_and_half_gradient
    from paper.experiments.qpt_quiroga_sensing import QuirogaSensingData, QuirogaSensingDesign
    cases, shared_identity = {}, {}
    for path in sorted(args.results_dir.rglob("result.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        trial_id = raw["diagnostic"]["trial_id"]
        require(trial_id in expected and trial_id not in cases, "Unknown/duplicate trial")
        trial = expected[trial_id]
        method, prefix = trial["method"], trial["method"].replace("-", "_")
        metadata = runner.validate_saved(path.with_suffix(".npz"), path, plan, trial, provenance, base, np)
        d, ep = metadata["diagnostic"], metadata["methods"][method]
        require(int(d["slurm_array_task_id"]) == trial["channel"], "Wrong Slurm channel")
        require(metadata["observation_mode"] == "noiseless" and metadata["active_row_count"] == 128
                and metadata["data_metadata"]["channel_seed"] == trial["channel"], "Data identity differs")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        identity = {}
        for key, shape in (("active_rows", (128,)), ("active_observations", (128,)),
                           ("truth_factor", (16, 1)), ("initial_factor", (16, 1))):
            require(arrays[key].shape == shape and np.all(np.isfinite(arrays[key])), "Invalid shared array")
            identity[key] = array_hash(arrays[key], np)
            require(identity[key] == metadata[key + "_sha256"], "Shared array checksum differs")
        channel = trial["channel"]
        if channel in shared_identity:
            require(identity == shared_identity[channel], "Methods do not share data/truth/initial factor")
        shared_identity[channel] = identity
        prior_check = paired_reference.check_prior(prior_report, trial, arrays, metadata, np)
        require(np.isclose(float(np.vdot(arrays["initial_factor"], arrays["initial_factor"]).real),
                           4.0, rtol=1e-13, atol=1e-13), "Initial factor trace differs")
        data = QuirogaSensingData(QuirogaSensingDesign(2), arrays["truth_factor"], "noiseless")
        require(np.allclose(data.all_observations(), arrays["active_observations"], atol=1e-14, rtol=1e-12),
                "Targets differ from archived truth")
        factor = arrays[prefix + "_final_factor"]
        trace = type("ArchivedTrace", (), {"status": ep["status"], "final_factor": factor})()
        independent, errors = base.independent_endpoint(data, trace, arrays["active_observations"], np,
                                                       evaluate_common_metrics)
        for key, value in independent.items():
            archived = d["full_design_endpoint"][key]
            require((value is None and archived is None)
                    or (value is not None and archived is not None
                        and np.isclose(value, archived, atol=1e-13, rtol=1e-10)),
                    "Independent endpoint differs: " + key)
        require(set(errors) == set(d["endpoint_metric_errors"]), "Undefined metric evidence differs")
        require(base.gate_for(ep["status"], independent, plan["shared"]["gate"])["pass"] == d["gate"]["pass"],
                "Independent gate differs")
        stopping_audit = None
        if ep["status"] == "converged":
            _, half = _summed_objective_and_half_gradient(data, factor, arrays["active_rows"],
                                                         arrays["active_observations"], 128 * trial["alpha"])
            final_norm = 2 * float(np.linalg.norm(half, ord="fro"))
            stop = ep["termination"]
            _, initial_half = _summed_objective_and_half_gradient(
                data, arrays["initial_factor"], arrays["active_rows"], arrays["active_observations"],
                128 * trial["alpha"])
            initial_norm = 2 * float(np.linalg.norm(initial_half, ord="fro"))
            require(np.isclose(initial_norm, stop["initial_full_gradient_norm"], rtol=1e-10, atol=1e-12),
                    "Independent initial gradient norm differs")
            require(np.isclose(final_norm, stop["full_gradient_norm"], rtol=1e-8, atol=1e-13),
                    "Independent stationarity norm differs")
            # Small cross-BLAS summation differences can appear at machine precision.
            require(final_norm <= stop["gradient_threshold"] + 1e-13, "Retained point fails stationarity")
            stopping_audit = {"independent_full_gradient_norm": final_norm,
                              "independent_initial_full_gradient_norm": initial_norm,
                              "archived_termination": stop, "comparison_atol": 1e-13}
        steps = arrays[prefix + "_checkpoint_steps"]
        rows = arrays[prefix + "_cumulative_data_row_accesses"]
        work_check = check_work(
            arrays, prefix, method, ep["status"], ep["failure_reason"], np)
        first = None
        for index, step in enumerate(steps):
            point = {key: float(arrays[prefix + "_" + key][index]) for key in base.PHYSICAL_METRICS}
            if base.gate_for("complete", point, plan["shared"]["gate"])["pass"]:
                first = {"step": int(step), "charged_rows": int(rows[index]), "metrics": point}
                break
        budget = d["budget"]
        require(budget["actual_optimizer_row_accesses"] == int(rows[-1])
                and budget["actual_optimizer_row_epochs"] == int(rows[-1]) / 128
                and budget["primary_equal_work_comparison"] is False, "Budget accounting differs")
        cases[trial_id] = {
            "trial_id": trial_id, "trial": trial, "method": method, "channel": channel,
            "status": ep["status"], "failure_reason": ep["failure_reason"],
            "completed_steps": int(steps[-1]), "actual_optimizer_row_accesses": int(rows[-1]),
            "endpoint_physical": d["full_design_endpoint"], "gate": d["gate"],
            "first_recorded_physical_gate": first, "stopping_audit": stopping_audit,
            "baseline": prior_check,
            "work_accounting_audit": work_check,
            "shared_array_hashes": identity, "final_factor_sha256": array_hash(factor, np),
            "json_path": str(path.resolve()), "json_sha256": base.file_hash(path),
            "npz_sha256": base.file_hash(path.with_suffix(".npz")),
        }
    require(cases, "No results found")
    missing = sorted(set(expected) - set(cases))
    require(args.allow_partial or not missing, "Campaign incomplete; use --allow-partial")
    inventories, verified, channels = [], set(), set()
    for path in sorted(set(args.inventory or args.results_dir.rglob("inventory.json"))):
        inv = json.loads(path.read_text(encoding="utf-8"))
        require(inv["kind"] == "qpt_gell_mann_inventory" and inv["schema_version"] == 1, "Wrong inventory kind")
        for key, value in provenance.items():
            require(inv[key] == value, "Inventory provenance differs")
        channel = inv["channel"]
        require(channel in (0, 1) and channel not in channels, "Unexpected/duplicate channel inventory")
        channels.add(channel)
        require(args.allow_partial or inv["state"] == "complete", "Inventory incomplete")
        ids = [record["trial"]["id"] for record in inv["trials"]]
        require(len(ids) == len(set(ids)) == 4 and set(ids) == {
            key for key, trial in expected.items() if trial["channel"] == channel}, "Inventory cases differ")
        for record in inv["trials"]:
            trial_id = record["trial"]["id"]
            require(record["trial"] == expected[trial_id], "Inventory trial differs")
            if trial_id not in cases:
                continue
            case = cases[trial_id]
            require(record["json_sha256"] == case["json_sha256"] and record["npz_sha256"] == case["npz_sha256"],
                    "Inventory checksum differs")
            require(record["method_status"] == case["status"] and record["gate_pass"] == case["gate"]["pass"]
                    and record["actual_optimizer_row_accesses"] == case["actual_optimizer_row_accesses"],
                    "Inventory status/gate/work differs")
            require(record["status"] == ("archived_method_failure" if case["status"] == "failed" else "complete"),
                    "Inventory failure classification differs")
            verified.add(trial_id)
        inventories.append({"path": str(path.resolve()), "sha256": base.file_hash(path),
                            "channel": channel, "state": inv["state"]})
    require(args.allow_partial or (channels == {0, 1} and verified == set(expected)),
            "Both complete inventory checksum sets are required")
    report = {
        "schema_version": 1, "kind": "qpt_gell_mann_audit", "audit_passed": True, "created_utc": base.now(),
        "expected_trials": 8, "audited_trials": len(cases), "missing_trial_ids": missing,
        "partial_mode": args.allow_partial, "provenance": provenance,
        "inventories": inventories, "trial_ids_without_inventory_checksum": sorted(set(cases) - verified),
        "cases": [cases[key] for key in sorted(cases)],
        "method_counts": {method: {
            "status_counts": dict(Counter(c["status"] for c in cases.values() if c["method"] == method)),
            "official_endpoint_gate_passes": sum(c["gate"]["pass"] for c in cases.values() if c["method"] == method),
        } for method in base.METHODS},
        "qualification": "Exploratory calibration only, not selection or unseen validation. Convergence is an optimizer stop, not a recovery certificate. Earlier-checkpoint physical gates are retrospective diagnostics and never relabel failed terminal status. Armijo row charges are not nominal-budget-matched to the primary methods.",
        "audit_script_sha256": base.file_hash(__file__),
        "prior_audit": {"path": str(args.prior_audit.resolve()), "sha256": base.file_hash(args.prior_audit)},
        "paired_reference_script_sha256": base.file_hash(here / "paired_reference.py"),
        "work_accounting_script_sha256": base.file_hash(here / "work_accounting.py"),
    }
    require(runner.source_hashes(args.repo, plan["source"]["commit"], base) == source_hashes, "Source changed during audit")
    base.write_json(args.output, report)
    print(json.dumps({"audit_passed": True, "audited_trials": len(cases), "missing_trials": len(missing),
                      "method_counts": report["method_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
