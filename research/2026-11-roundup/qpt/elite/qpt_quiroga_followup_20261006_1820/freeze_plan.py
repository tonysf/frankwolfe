#!/usr/bin/env python3
"""Generate the explicit follow-up plan once; never modify an existing plan."""
import argparse
import datetime
import json
from pathlib import Path
import re

BASE_SHA = "13bcfa92d5d4876cc68f4d04fa4b0869d611c67d42ec52cc56c067925c3884b6"
OLD_COMMIT = "c7a8c9789b7814ede1c95e53eef13bfe1dd58c63"


def build_plan(commit, frozen_utc):
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == OLD_COMMIT:
        raise ValueError("A new full source commit containing the reviewed Armijo stop is required")
    cases = []
    for channel in (0, 1):
        for alpha, label in ((5e-6, "5em6"), (5e-4, "5em4")):
            cases.append({"id": f"c{channel}-q-a{label}-e1-100kep", "channel": channel,
                          "method": "printed-adafgd", "alpha": alpha, "eta_scale": 1.0,
                          "tau": 2.0, "row_epochs": 100000, "checkpoint_epochs": 1000,
                          "armijo_gradient_atol": 0.0, "armijo_gradient_rtol": 0.0})
        cases.append({"id": f"c{channel}-f-fast-average-100kep", "channel": channel,
                      "method": "frames", "alpha": 5e-5, "eta_scale": 0.01, "tau": 2.0,
                      "row_epochs": 100000, "checkpoint_epochs": 1000,
                      "step_schedule": {"scale": 4.0, "offset": 4.0, "exponent": 1.0, "cap": 1.0},
                      "armijo_gradient_atol": 0.0, "armijo_gradient_rtol": 0.0})
        for alpha, label in ((5e-5, "5em5"), (5e-4, "5em4"), (0.5, "5em1")):
            cases.append({"id": f"c{channel}-armijo-a{label}-gradstop", "channel": channel,
                          "method": "armijo-fgd", "alpha": alpha, "eta_scale": 0.01,
                          "tau": 2.0, "row_epochs": 10000, "checkpoint_epochs": 100,
                          "armijo_gradient_atol": 1e-12, "armijo_gradient_rtol": 1e-10})
    return {
        "schema_version": 1, "kind": "qpt_calibration_horizon_stopping_followup",
        "campaign_id": "qpt_followup_20261006_1820", "frozen_utc": frozen_utc,
        "role": "Exploratory calibration horizon/stopping diagnostic; not selection robustness or unseen validation.",
        "rationale": "Extend stable printed-rule cases at eta scale1 and two distinct low fixed penalties; extend the faster FRAMES averaging schedule tested on both calibration channels. Check the isolated opt-in Armijo stationarity stop instead of mistaking machine-precision line-search stagnation for numerical divergence. The stopping check is not a physical recovery certificate.",
        "preservation": "Earlier calibration, manifest, validation and resolution pilot remain unchanged. This plan does not relabel their failures or consume heldout channels.",
        "source": {"commit": commit, "repo": "/gpfs/workdir/silvetian/frankwolfe-quiroga-" + commit},
        "base_runner_sha256": BASE_SHA,
        "shared": {"n_qubits": 2, "rank": 1, "active_rows": 128, "batch_size": 32,
                   "frames_penalty_exponent": 0.25, "observation_mode": "noiseless",
                   "initialization": "trace_normalized_choi_uniform",
                   "rho": {"scale": 2, "offset": 4, "exponent": 0.6, "cap": 1},
                   "step": {"scale": 2, "offset": 2, "exponent": 1, "cap": 1},
                   "gate": {"fidelity_min": 0.99, "rmse_max": 1e-6, "normalized_tp_max": 0.001}},
        "channels": [0, 1], "trials": cases,
        "seed_rule": {"initialization": "200000+10*channel", "sampling": "300000+100*channel"},
        "expected_counts": {"trials": 12, "per_channel": 6, "printed": 4, "frames": 2, "armijo": 6},
        "resources": {"account": "fwllm", "partition": "cpu_short", "cpus_per_task": 2,
                      "memory": "1G", "walltime_seconds": 1200, "array": "0-1%1",
                      "minimum_seconds_before_new_trial": 120},
        "budget_qualification": "Printed100000 updates and FRAMES400000 batch32 updates charge12800000 nominal optimizer rows when complete; spectral work and checkpoints are excluded. Armijo has10000 maximum iterations with actual line-search and terminal-gradient row charges, not equal work. Converged Armijo endpoints must independently pass the physical gate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = build_plan(args.commit, datetime.datetime.now(datetime.timezone.utc).isoformat())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
