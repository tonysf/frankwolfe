#!/usr/bin/env python3
"""Run one immutable tuning-plan channel in a Slurm allocation."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import time

import common as c


def execute(plan, trial, out, provenance):
    import numpy as np
    from paper.experiments.qpt_quiroga_campaign import run_campaign, save_campaign, evaluate_common_metrics
    from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
    c.check_module_origins(c.REPO)
    destination = out / trial["id"]
    destination.mkdir()
    data = generate_quiroga_sensing_data(2, channel_seed=trial["channel"], observation_mode="noiseless")
    metadata, payload, traces = run_campaign(data, **c.configuration(trial))
    trace = traces[trial["method"]]
    physical, errors = c.base.independent_endpoint(data, trace, data.all_observations(), np, evaluate_common_metrics)
    metadata["tuning"] = {**provenance, "trial": trial, "requested_configuration": c.configuration(trial),
                          "full_design_endpoint": physical, "endpoint_metric_errors": errors,
                          "gate": c.base.gate_for(trace.status, physical, plan["gate"])}
    npz, js = destination / "result.npz", destination / "result.json"
    save_campaign(npz, js, metadata, payload)
    from audit_results import validate_archive
    validate_archive(js, plan, trial, provenance, np)
    return {"status": "archived_method_failure" if trace.status == "failed" else "complete",
            "method_status": trace.status, "failure_reason": trace.failure_reason or None,
            "gate_pass": metadata["tuning"]["gate"]["pass"],
            "json": str(js.relative_to(out)), "json_sha256": c.base.file_hash(js),
            "npz": str(npz.relative_to(out)), "npz_sha256": c.base.file_hash(npz), "completed_utc": c.base.now()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--calibration-audit", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    trials = c.validate_plan(plan)
    c.check_selection(plan, args.calibration_audit)
    provenance = c.provenance(plan, args.plan, args.plan_sha256, c.REPO)
    if args.validate_only:
        print(json.dumps({"valid": True, "trials": len(trials), **provenance}))
        return 0
    c.require(os.environ.get("SLURM_JOB_ID") and os.environ.get("SLURM_JOB_NODELIST"), "Scientific execution requires Slurm")
    channels = sorted({t["channel"] for t in trials})
    c.require(args.array_index in range(len(channels)) and args.out is not None and args.deadline_epoch is not None
              and math.isfinite(args.deadline_epoch), "Missing/invalid controls")
    c.require(int(os.environ["SLURM_ARRAY_TASK_ID"]) == args.array_index, "Wrong array task")
    channel = channels[args.array_index]
    args.out.mkdir(exist_ok=False)
    provenance.update(hostname=socket.gethostname(), slurm_job_id=os.environ["SLURM_JOB_ID"],
                      slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"), slurm_array_task_id=args.array_index)
    inventory = {"schema_version": 1, "kind": "qpt_quiroga_tuning_inventory", **provenance,
                 "channel": channel, "state": "prepared", "created_utc": c.base.now(),
                 "trials": [{"trial": t, "status": "pending"} for t in trials if t["channel"] == channel]}
    inventory_path = args.out / "inventory.json"
    c.write_json(inventory_path, inventory)
    try:
        for record in inventory["trials"]:
            if args.deadline_epoch - time.time() <= plan["resources"]["minimum_seconds_before_new_trial"]:
                inventory["state"] = "time_guard_pending"
                c.write_json(inventory_path, inventory, True)
                return 75
            inventory["state"] = "running"
            record.update(status="running", started_utc=c.base.now())
            c.write_json(inventory_path, inventory, True)
            print("START " + record["trial"]["id"], flush=True)
            record.update(execute(plan, record["trial"], args.out, provenance))
            c.write_json(inventory_path, inventory, True)
            print("END " + record["trial"]["id"] + " " + record["status"], flush=True)
        c.require(c.source_hashes(c.REPO, plan["source"]["commit"]) == provenance["source_sha256"], "Source changed")
        inventory.update(state="complete", completed_utc=c.base.now())
        c.write_json(inventory_path, inventory, True)
        return 2 if any(t["method_status"] == "failed" for t in inventory["trials"]) else 0
    except Exception as error:
        inventory.update(state="infrastructure_or_invalid_archive", error=f"{type(error).__name__}: {error}", stopped_utc=c.base.now())
        c.write_json(inventory_path, inventory, True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
