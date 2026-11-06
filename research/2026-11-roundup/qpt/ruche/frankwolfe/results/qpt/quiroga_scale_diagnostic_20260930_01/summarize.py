"""Summarize bounded n=2/n=3 Quiroga scale diagnostics."""
import json
from pathlib import Path
import numpy as np

root = Path("results")
rows = []
for path in sorted(root.glob("*/s*_w*/final.npz")):
    case = path.parents[1].name
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata_json"].item()))
        fidelity = archive["process_fidelity_proxy"]
        tp = archive["tp_violation"]
        loss = archive["measurement_loss"]
        eta = archive["checkpoint_adaptive_step"]
    rows.append({
        "case": case,
        "label": path.parent.name,
        "n_qubits": meta["n_qubits"],
        "initialization": meta["initialization"],
        "initial_factor_norm": meta["initial_factor_norm"],
        "eta_scale": meta["eta_scale"],
        "tp_weight": meta["tp_weight"],
        "initial_fidelity": float(fidelity[0]),
        "final_fidelity": float(fidelity[-1]),
        "fidelity_change": float(fidelity[-1] - fidelity[0]),
        "initial_tp": float(tp[0]),
        "final_tp": float(tp[-1]),
        "initial_loss": float(loss[0]),
        "final_loss": float(loss[-1]),
        "final_eta": float(eta[-1]),
        "finite": bool(np.all(np.isfinite(fidelity)) and np.all(np.isfinite(tp)) and np.all(np.isfinite(loss))),
        "result": str(path),
    })
reports = {}
for case in sorted({row["case"] for row in rows}):
    subset = [row for row in rows if row["case"] == case and row["finite"]]
    improving = [row for row in subset if row["fidelity_change"] > 0]
    admissible = [row for row in improving if row["final_tp"] <= 1.0]
    reports[case] = {
        "completed": len(subset),
        "best_fidelity": max(subset, key=lambda r: r["final_fidelity"]) if subset else None,
        "best_tp": min(subset, key=lambda r: r["final_tp"]) if subset else None,
        "best_improving": max(improving, key=lambda r: r["final_fidelity"]) if improving else None,
        "best_admissible": max(admissible, key=lambda r: r["final_fidelity"]) if admissible else None,
    }
report = {
    "status": "complete",
    "purpose": "separate dimension/noise/initialization effects within QPT_BFW sensing; not a reproduction of Quiroga sensing",
    "steps": 1000,
    "paper_row_counts": {"n2": 128, "n3": 1024},
    "reports": reports,
    "rows": rows,
}
Path("summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": report["status"], "reports": reports}, indent=2, allow_nan=False))
