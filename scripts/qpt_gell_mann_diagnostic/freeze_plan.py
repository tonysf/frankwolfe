"""Freeze the eight initializer-sensitivity cases; never overwrite a plan."""
import argparse
import datetime
import json
from pathlib import Path
import re

PRIOR_COMMIT = "0f7857e94f6b16ac07733be248f68aa6904d287b"
PRIOR_AUDIT_SHA = "a091664e575338ff7a1947eae0d195ea349f64ca8881566dc15d712da2b120de"


def build_plan(commit, frozen_utc):
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit == PRIOR_COMMIT:
        raise ValueError("A new full commit containing the reviewed initializer is required")
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
        cases.append({"id": f"c{channel}-armijo-a5em4-gradstop", "channel": channel,
                      "method": "armijo-fgd", "alpha": 5e-4, "eta_scale": 0.01,
                      "tau": 2.0, "row_epochs": 10000, "checkpoint_epochs": 100,
                      "armijo_gradient_atol": 1e-12, "armijo_gradient_rtol": 1e-10})
    return {
        "schema_version": 1, "kind": "qpt_gell_mann_initialization_diagnostic",
        "campaign_id": "qpt_gell_mann_init_20261007", "frozen_utc": frozen_utc,
        "role": "Calibration initializer-sensitivity diagnostic; not paper-exact reproduction, selection, or unseen validation.",
        "preservation": "Earlier archives and failures are unchanged. Baselines are compared from archives, not rerun. Channels 100..107 and 1000..1019 are untouched.",
        "source": {"commit": commit, "repo": "/gpfs/workdir/silvetian/frankwolfe-quiroga-" + commit},
        "prior": {"commit": PRIOR_COMMIT, "audit_sha256": PRIOR_AUDIT_SHA,
                  "qualification": "Only the normalized coefficient basis changes versus the matched eight archived follow-up cases. Trajectories need not match."},
        "shared": {"n_qubits": 2, "rank": 1, "active_rows": 128, "batch_size": 32,
                   "frames_penalty_exponent": 0.25, "observation_mode": "noiseless",
                   "initialization": "trace_normalized_uniform_gell_mann_coefficients",
                   "initialization_basis": "gell-mann",
                   "rho": {"scale": 2, "offset": 4, "exponent": 0.6, "cap": 1},
                   "step": {"scale": 2, "offset": 2, "exponent": 1, "cap": 1},
                   "gate": {"fidelity_min": 0.99, "rmse_max": 1e-6, "normalized_tp_max": 0.001}},
        "channels": [0, 1], "trials": cases,
        "seed_rule": {"initialization": "200000+10*channel", "sampling": "300000+100*channel"},
        "expected_counts": {"trials": 8, "per_channel": 4, "printed": 4, "frames": 2, "armijo": 2},
        "resources": {"account": "fwllm", "partition": "cpu_short", "cpus_per_task": 2,
                      "memory": "1G", "walltime_seconds": 1200, "array": "0-1%1",
                      "minimum_seconds_before_new_trial": 360},
        "budget_qualification": "Printed 100000 updates and FRAMES 400000 batch32 updates charge 12800000 nominal optimizer rows when complete; spectral work and checkpoints are excluded. Armijo has 10000 maximum iterations with actual line-search and terminal-gradient row charges, not equal work. Converged Armijo endpoints must independently pass the physical gate.",
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
