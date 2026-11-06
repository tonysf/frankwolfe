"""Read only small scalar members of completed QPT result archives."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("base", type=Path)
parser.add_argument("save", type=Path)
args = parser.parse_args()
folders = [
    "on_demand_profile_20260912_04", "on_demand_a100_20260912_02",
    "on_demand_recovery_n8_20260912_01", "on_demand_n10_batches_20260912_01",
    "on_demand_weaker_tp_20260912_01",
]
fields = ["checkpoint_steps", "process_fidelity_proxy", "measurement_loss",
          "tp_violation", "smoothed_gap", "checkpoint_smoothing_parameters"]
records = []
for folder in folders:
    for path in sorted((args.base / folder).glob("*_benchmark.json")):
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            continue
        row = next(r for r in report["runs"] if r["purpose"] == "timing" and r["status"] == "success")
        with np.load(row["structured_result_path"], allow_pickle=False) as archive:
            trace = {field: archive[field].tolist() for field in fields}
        records.append(dict(report=str(path), folder=folder, name=path.name,
                            n=report["n_qubits"], configuration=report["configuration"],
                            source_metadata=report["source_metadata"],
                            device=row["device_kind"], hostname=row["hostname"], trace=trace))
args.save.parent.mkdir(parents=True, exist_ok=True)
with args.save.open("x") as handle:
    json.dump(records, handle, indent=2, allow_nan=False)
print(f"Saved {len(records)} completed scalar trajectories to {args.save}")
