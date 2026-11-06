"""Validation, selection, and aggregation for the n=8 baseline campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


DATA_SHA256 = "c05b02bdbc1b8955a89ae0cff2da77a986c76f96d2971860f3d2f7d0b925e439"
TUNING_SCALES = (0.1, 0.3, 1.0, 3.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        return meta, {
            "checkpoint_steps": archive["checkpoint_steps"].copy(),
            "fidelity": archive["process_fidelity_proxy"].copy(),
            "loss": archive["measurement_loss"].copy(),
            "tp": archive["tp_violation"].copy(),
            "objective": archive["smoothed_objective"].copy(),
            "gap": archive["smoothed_gap"].copy(),
            "factor": archive["final_factor"].copy(),
        }


def validate_inputs(args):
    record = json.loads(args.source_record.read_text())
    expected = record["source_sha256"]
    actual = {
        relative: sha256(args.source / relative) for relative in sorted(expected)
    }
    if actual != expected:
        bad = [name for name in expected if actual.get(name) != expected[name]]
        raise RuntimeError(f"frozen source checksum mismatch: {bad}")
    actual_data = sha256(args.data)
    if actual_data != DATA_SHA256:
        raise RuntimeError(f"data checksum mismatch: {actual_data}")
    report = {
        "status": "validated",
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source": str(args.source),
        "source_record": str(args.source_record),
        "source_file_count": len(actual),
        "data": str(args.data),
        "data_sha256": actual_data,
        "baseline_script": str(args.baseline),
        "baseline_script_sha256": sha256(args.baseline),
    }
    args.save.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def select(args):
    rows = []
    initial_hashes = set()
    batch_hashes = set()
    for scale in TUNING_SCALES:
        label = str(scale).replace(".", "p")
        path = args.tuning_dir / f"eta_{label}" / "final.npz"
        meta, values = metadata(path)
        if meta.get("optimizer") != "projected_momentum_sgd":
            raise RuntimeError(f"wrong optimizer metadata in {path}")
        if meta["n_steps"] != args.steps or meta["sampling_seed"] != 0:
            raise RuntimeError(f"wrong budget or tuning seed in {path}")
        if not all(np.all(np.isfinite(value)) for key, value in values.items() if key != "fidelity"):
            raise RuntimeError(f"nonfinite tuning output in {path}")
        if not np.all(np.isfinite(values["fidelity"])):
            raise RuntimeError(f"nonfinite tuning fidelity in {path}")
        if np.linalg.norm(values["factor"], ord=2) > meta["tau"] + 1e-9:
            raise RuntimeError(f"infeasible tuning factor in {path}")
        initial_hashes.add(meta["initial_factor_sha256"])
        batch_hashes.add(meta["batch_symbols_sha256"])
        rows.append({
            "eta_scale": scale,
            "path": str(path),
            "result_sha256": sha256(path),
            "final_fidelity": float(values["fidelity"][-1]),
            "final_measurement_loss": float(values["loss"][-1]),
            "final_tp_violation": float(values["tp"][-1]),
            "final_smoothed_objective": float(values["objective"][-1]),
            "final_fw_gap": float(values["gap"][-1]),
            "optimizer_seconds": meta["optimizer_seconds"],
        })
    if len(initial_hashes) != 1 or len(batch_hashes) != 1:
        raise RuntimeError("tuning candidates did not use identical U0 and minibatches")
    # Predeclared rule: maximize terminal synthetic-truth fidelity on seed 0;
    # deterministic tie-breaks favor lower TP violation, loss, then eta.
    chosen = min(
        rows,
        key=lambda row: (
            -row["final_fidelity"],
            row["final_tp_violation"],
            row["final_measurement_loss"],
            row["eta_scale"],
        ),
    )
    report = {
        "status": "selected",
        "selected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_seed": 0,
        "tuning_steps": args.steps,
        "candidates": list(TUNING_SCALES),
        "criterion": "maximum terminal fidelity; ties: TP, loss, eta_scale",
        "selected_eta_scale": chosen["eta_scale"],
        "initial_factor_sha256": next(iter(initial_hashes)),
        "batch_symbols_sha256": next(iter(batch_hashes)),
        "runs": rows,
    }
    args.save.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def summarize(args):
    selection = json.loads(args.selection.read_text())
    eta = float(selection["selected_eta_scale"])
    rows = []
    by_key = {}
    initial_hashes = set()
    for method in ("frames", "momentum_sgd"):
        for seed in (1, 2, 3):
            run_dir = args.results_dir / f"{method}_seed{seed}"
            path = run_dir / "final.npz"
            verification_path = run_dir / "verification.json"
            meta, values = metadata(path)
            verification = json.loads(verification_path.read_text())
            if meta["n_steps"] != 200000 or meta["sampling_seed"] != seed:
                raise RuntimeError(f"wrong budget or seed in {path}")
            if method == "momentum_sgd":
                if meta.get("optimizer") != "projected_momentum_sgd":
                    raise RuntimeError(f"wrong optimizer in {path}")
                if not np.isclose(meta["eta_scale"], eta):
                    raise RuntimeError(f"wrong selected eta in {path}")
            if verification.get("status") != "verified":
                raise RuntimeError(f"missing verification for {path}")
            initial_hashes.add(meta["initial_factor_sha256"])
            row = {
                "method": method,
                "sampling_seed": seed,
                "result": str(path),
                "result_sha256": sha256(path),
                "batch_symbols_sha256": meta["batch_symbols_sha256"],
                "final_fidelity": float(values["fidelity"][-1]),
                "final_measurement_loss": float(values["loss"][-1]),
                "final_tp_violation": float(values["tp"][-1]),
                "final_fw_gap": float(values["gap"][-1]),
                "optimizer_seconds": meta["optimizer_seconds"],
            }
            rows.append(row)
            by_key[(method, seed)] = row
    if len(initial_hashes) != 1:
        raise RuntimeError("comparison runs did not use identical U0")
    for seed in (1, 2, 3):
        if by_key[("frames", seed)]["batch_symbols_sha256"] != by_key[("momentum_sgd", seed)]["batch_symbols_sha256"]:
            raise RuntimeError(f"paired minibatch mismatch for seed {seed}")
    aggregates = {}
    for method in ("frames", "momentum_sgd"):
        method_rows = [row for row in rows if row["method"] == method]
        aggregates[method] = {
            key: {
                "mean": float(np.mean([row[key] for row in method_rows])),
                "std_ddof1": float(np.std([row[key] for row in method_rows], ddof=1)),
            }
            for key in ("final_fidelity", "final_measurement_loss", "final_tp_violation", "optimizer_seconds")
        }
    report = {
        "status": "complete",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "held_out_sampling_seeds": [1, 2, 3],
        "steps_per_run": 200000,
        "selected_eta_scale": eta,
        "initial_factor_sha256": next(iter(initial_hashes)),
        "paired_minibatches_verified": True,
        "runs": rows,
        "aggregates": aggregates,
    }
    args.save.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate-inputs")
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--source-record", type=Path, required=True)
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--save", type=Path, required=True)
    check.set_defaults(func=validate_inputs)
    choose = sub.add_parser("select")
    choose.add_argument("--tuning-dir", type=Path, required=True)
    choose.add_argument("--steps", type=int, default=20000)
    choose.add_argument("--save", type=Path, required=True)
    choose.set_defaults(func=select)
    final = sub.add_parser("summarize")
    final.add_argument("--selection", type=Path, required=True)
    final.add_argument("--results-dir", type=Path, required=True)
    final.add_argument("--save", type=Path, required=True)
    final.set_defaults(func=summarize)
    args = parser.parse_args()
    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.save.exists():
        raise FileExistsError(args.save)
    args.func(args)


if __name__ == "__main__":
    main()
