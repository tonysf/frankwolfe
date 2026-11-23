"""Select a stable fixed-step MiFGD candidate from the seed-0 calibration."""

import argparse
import json
from pathlib import Path

import numpy as np


ETAS = (1e-6, 1e-5, 1e-4, 1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    rows, initial_hashes, batch_hashes = [], set(), set()
    for eta in ETAS:
        label = f"{eta:.0e}".replace("-", "m")
        path = args.root / f"eta_{label}" / "final.npz"
        if not path.exists():
            rows.append({"eta": eta, "status": "failed_or_nonfinite"})
            continue
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata_json"].item()))
            fidelity = float(archive["process_fidelity_proxy"][-1])
            loss = float(archive["measurement_loss"][-1])
            tp = float(archive["tp_violation"][-1])
            factor_norm = float(np.linalg.norm(archive["final_factor"]))
        if meta.get("optimizer") != "mifgd_fixed_tp":
            raise RuntimeError(f"wrong optimizer metadata in {path}")
        if meta["sampling_seed"] != 0 or meta["n_steps"] != 20000:
            raise RuntimeError(f"wrong tuning seed or budget in {path}")
        initial_hashes.add(meta["initial_factor_sha256"])
        batch_hashes.add(meta["batch_symbols_sha256"])
        rows.append({
            "eta": eta,
            "status": "finite" if np.all(np.isfinite([fidelity, loss, tp, factor_norm])) else "nonfinite",
            "final_fidelity": fidelity,
            "final_measurement_loss": loss,
            "final_tp_violation": tp,
            "final_factor_norm": factor_norm,
            "optimizer_seconds": meta["optimizer_seconds"],
            "path": str(path),
        })
    finite = [row for row in rows if row["status"] == "finite"]
    if not finite or len(initial_hashes) != 1 or len(batch_hashes) != 1:
        raise RuntimeError("no comparable finite tuning candidates")
    admissible = [row for row in finite if row["final_tp_violation"] <= 1.0]
    chosen = min(
        admissible,
        key=lambda row: (-row["final_fidelity"], row["final_tp_violation"], row["eta"]),
    ) if admissible else None
    report = {
        "status": "selected" if chosen else "no_admissible_candidate",
        "selection_seed": 0,
        "steps": 20000,
        "mu": 0.75,
        "tp_weight": 1.0,
        "objective": "measurement_loss + 1.0 * 0.5*TP_residual_squared",
        "candidates": list(ETAS),
        "admissibility": "finite and terminal TP violation <= 1.0",
        "selection": "maximum terminal fidelity; ties by TP then eta",
        "selected_eta": None if chosen is None else chosen["eta"],
        "best_tp_candidate": min(finite, key=lambda row: row["final_tp_violation"])["eta"],
        "initial_factor_sha256": next(iter(initial_hashes)),
        "batch_symbols_sha256": next(iter(batch_hashes)),
        "runs": rows,
    }
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

