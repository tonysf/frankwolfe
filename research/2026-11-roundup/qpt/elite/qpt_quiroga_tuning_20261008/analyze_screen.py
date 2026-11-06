#!/usr/bin/env python3
"""Select only from a complete pinned screen audit; never run an optimizer.

Both method families are ranked separately using the rule frozen in the screen
plan. This script selects no failed, nonfinite, incomplete, or gate-failing
configuration. Confirmation plans are deliberately outside its scope.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
METHODS = ("printed-adafgd", "exact-linesearch-fgd")
KEYS = ("method", "alpha", "eta_scale", "n_steps", "metrics_frequency")
GATE = {"fidelity_min": .99, "rmse_max": 1e-6, "normalized_tp_max": .001}
RULE = ("Rank each method separately by worst-channel max(RMSE/1e-6, "
        "normalizedTP/1e-3, max(0,1-fidelity)/.01); failed/nonfinite or "
        "missing-channel groups are ineligible. Select only a configuration "
        "passing both terminal gates. Break ties by smaller total charged "
        "rows, then alpha, then eta_scale.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pinned(path, digest):
    require(re.fullmatch(r"[0-9a-f]{64}", digest), "Invalid SHA256")
    require(file_hash(path) == digest, "Checksum mismatch: " + str(path))
    return json.loads(Path(path).read_text())


def validate_audit(audit_path, audit_sha256, plan_path):
    audit = read_pinned(audit_path, audit_sha256)
    require(audit["kind"] == "qpt_quiroga_tuning_audit" and audit["schema_version"] == 1
            and audit["audit_passed"] is True and audit["partial_mode"] is False
            and audit["missing_trial_ids"] == [] and audit["trial_ids_without_inventory_checksum"] == [],
            "The screen audit is not complete and verified")
    provenance = audit["provenance"]
    require(provenance["phase"] == "calibration", "Not a calibration screen")
    plan = read_pinned(plan_path, provenance["plan_sha256"])
    require(plan["phase"] == "calibration" and plan["gate"] == GATE
            and plan["source"]["commit"] == provenance["source_commit"]
            and plan["source"]["sha256"] == provenance["source_sha256"]
            and plan["campaign_id"] == provenance["campaign_id"], "Audit/plan identity differs")
    expected = {trial["id"]: trial for trial in plan["trials"]}
    require(len(expected) == len(plan["trials"]) == audit["expected_trials"] == audit["audited_trials"] == len(audit["cases"]),
            "Wrong number of screen cases")
    require({t["channel"] for t in expected.values()} == {0, 1}, "Screen channels must be exactly 0 and 1")
    seen = set()
    for case in audit["cases"]:
        key = case["trial_id"]
        require(key in expected and key not in seen and case["trial"] == expected[key], "Unknown/duplicate/changed trial")
        require(case["channel"] == case["trial"]["channel"] and case["method"] == case["trial"]["method"]
                and case["method"] in METHODS, "Case method/channel differs")
        seen.add(key)
        native_path = Path(case["json_path"])
        native = read_pinned(native_path, case["json_sha256"])
        require(file_hash(native_path.with_suffix(".npz")) == case["npz_sha256"], "Native NPZ checksum differs")
        require(native["tuning"]["trial"] == case["trial"] and native["tuning"]["gate"] == case["gate"]
                and native["methods"][case["method"]]["status"] == case["status"], "Native/audit endpoint differs")
        for name, value in case["endpoint_physical"].items():
            archived = native["tuning"]["full_design_endpoint"][name]
            require((value is None and archived is None) or (finite_number(value) and finite_number(archived)
                    and math.isclose(value, archived, rel_tol=1e-10, abs_tol=1e-13)), "Native/audited metric differs: " + name)
        rows = case["work_accounting"]["verified_rows"]
        require(type(rows) is int and rows >= 0
                and native["methods"][case["method"]]["cumulative_data_row_accesses"] == rows, "Charged work differs")
    require(seen == set(expected), "Missing cases")
    inventory_channels, inventory_cases = set(), set()
    cases_by_id = {case["trial_id"]: case for case in audit["cases"]}
    for item in audit["inventories"]:
        inventory = read_pinned(item["path"], item["sha256"])
        channel = inventory["channel"]
        require(channel in {0, 1} and channel not in inventory_channels
                and inventory["state"] == item["state"] == "complete", "Incomplete or duplicate channel inventory")
        inventory_channels.add(channel)
        for key, value in provenance.items():
            require(inventory[key] == value, "Inventory provenance differs")
        for record in inventory["trials"]:
            key = record["trial"]["id"]
            require(key in expected and key not in inventory_cases and record["trial"] == expected[key]
                    and record["trial"]["channel"] == channel, "Inventory trial identity differs")
            case = cases_by_id[key]
            require(record["json_sha256"] == case["json_sha256"] and record["npz_sha256"] == case["npz_sha256"]
                    and record["method_status"] == case["status"] and record["gate_pass"] == case["gate"]["pass"],
                    "Inventory archive identity/status differs")
            inventory_cases.add(key)
    require(inventory_channels == {0, 1} and inventory_cases == seen, "Inventory coverage differs")
    return audit, plan


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def group_results(cases):
    grouped = {}
    for case in cases:
        key = tuple(case["trial"][k] for k in KEYS)
        grouped.setdefault(key, []).append(case)
    groups = []
    for key, members in grouped.items():
        channels = [case["channel"] for case in members]
        require(len(channels) == len(set(channels)), "A setting repeats a channel/initializer")
        finite, allowed_status, deficits, endpoints = True, True, [], []
        for case in sorted(members, key=lambda c: c["channel"]):
            physical = case["endpoint_physical"]
            fid, rmse, tp = (physical[name] for name in ("process_fidelity", "rmse", "normalized_tp_violation"))
            valid = all(finite_number(value) for value in (fid, rmse, tp)) and rmse >= 0 and tp >= 0
            status_ok = case["status"] in {"complete", "converged"}
            deficit = max(rmse / 1e-6, tp / .001, max(0., 1. - fid) / .01) if valid else None
            passed = valid and status_ok and fid >= .99 and rmse <= 1e-6 and tp <= .001
            require(case["gate"]["thresholds"] == GATE and case["gate"]["pass"] == passed, "Gate is not the unchanged physical gate")
            finite &= valid
            allowed_status &= status_ok
            if deficit is not None:
                deficits.append(deficit)
            endpoints.append({"channel": case["channel"], "trial_id": case["trial_id"],
                              "initialization_seed": case["trial"]["initialization_seed"],
                              "status": case["status"], "failure_reason": case["failure_reason"],
                              "gate_pass": passed, "gate_deficit": deficit, "physical": physical,
                              "completed_steps": case["completed_steps"],
                              "charged_rows": case["work_accounting"]["verified_rows"],
                              "json_path": case["json_path"], "json_sha256": case["json_sha256"], "npz_sha256": case["npz_sha256"]})
        full = set(channels) == {0, 1}
        eligible = full and finite and allowed_status
        groups.append({"configuration": dict(zip(KEYS, key)), "eligible": eligible,
                       "ineligible_reasons": [reason for condition, reason in ((full, "missing_channel"), (finite, "nonfinite_or_invalid_physical_endpoint"), (allowed_status, "failed_or_unknown_terminal_status")) if not condition],
                       "all_endpoint_gates_pass": eligible and all(case["gate_pass"] for case in endpoints),
                       "worst_channel_deficit": max(deficits) if finite and deficits else None,
                       "total_charged_rows": sum(case["charged_rows"] for case in endpoints), "cases": endpoints})
    def order(group):
        cfg = group["configuration"]
        return (not group["eligible"], group["worst_channel_deficit"] if group["worst_channel_deficit"] is not None else math.inf,
                group["total_charged_rows"], cfg["alpha"], cfg["eta_scale"], cfg["n_steps"], cfg["metrics_frequency"])
    by_method, selected = {}, {}
    for method in METHODS:
        ranked = sorted((group for group in groups if group["configuration"]["method"] == method), key=order)
        for rank, group in enumerate(ranked, 1):
            group["rank_within_method"] = rank
        by_method[method] = ranked
        passing = [group for group in ranked if group["all_endpoint_gates_pass"]]
        selected[method] = passing[0]["configuration"] if passing else None
    return by_method, selected


def write_new(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "audit_screen.json")
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "stage/tuning_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "selection_screen.json")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Refusing selection overwrite")
    audit, plan = validate_audit(args.audit, args.audit_sha256, args.plan)
    grouped, selected = group_results(audit["cases"])
    result = {"schema_version": 1, "kind": "qpt_quiroga_screen_selection", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "audit_path": str(args.audit.resolve()), "audit_sha256": args.audit_sha256,
              "source_plan": {"path": str(args.plan.resolve()), "sha256": file_hash(args.plan)},
              "provenance": audit["provenance"], "gate": GATE, "selection_rule": RULE,
              "frozen_plan_selection_rule": plan.get("selection_rule"),
              "methods": {method: {"selected_configuration": selected[method], "ranked_settings": grouped[method]} for method in METHODS},
              "analysis_script_sha256": file_hash(__file__),
              "qualification": "Screen channels 0 and 1 only; this does not constitute confirmation or unseen validation. Prior archived checkpoint pairs are not imported or silently pooled. Exact-line search remains a modification, not the printed Quiroga step. No next-stage plan was created."}
    require(file_hash(args.audit) == args.audit_sha256, "Audit changed during analysis")
    write_new(args.output, result)
    print("method                      alpha       eta  worst deficit  rows       gates/status")
    for method in METHODS:
        for group in grouped[method]:
            cfg = group["configuration"]
            deficit = "undefined" if group["worst_channel_deficit"] is None else f'{group["worst_channel_deficit"]:.5g}'
            label = "PASS" if group["all_endpoint_gates_pass"] else ("fail gate" if group["eligible"] else "INELIGIBLE")
            print(f'{method:27} {cfg["alpha"]:9.3g} {cfg["eta_scale"]:5g} {deficit:>13} {group["total_charged_rows"]:10d} {label}')
        print("Selected " + method + ": " + json.dumps(selected[method], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
