"""Bounded n=8, sigma=.05 comparison; execute only in a Slurm allocation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
plan = json.loads((root / "plan.json").read_text())
case = plan["cases"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
output = root / case["name"]
output.mkdir()
source = Path(plan["source"])
data = Path(plan["data"])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

hashes = {name: sha(source / name) for name in plan["source_sha256"]}
assert hashes == plan["source_sha256"], "Validated source changed"
with np.load(data, allow_pickle=False) as z:
    data_meta = json.loads(str(z["metadata_json"].item()))
    assert int(z["n_qubits"]) == 8 and data_meta["noise_std"] == .05
    assert data_meta["channel_seed"] == data_meta["noise_seed"] == 0

common = [sys.executable, "-m", "paper.experiments.quantum_process_tomography_structured_jax",
    "--data", str(data), "--device", "gpu", "--precision", "64",
    "--measurement-backend", "product-state", "--rank", "1", "--tau", "20",
    "--batch-size", str(case["batch_size"]), "--chunk-steps", "10", "--prefetch",
    "--metrics-every", "1000", "--metric-mode", "sampled", "--metric-samples", "512",
    "--metric-batch-size", "512", "--metric-seed", "12345",
    "--initialization-seed", "0", "--sampling-seed", "0",
    "--rho-scale", "2", "--rho-offset", "4", "--rho-exponent", "0.6",
    "--smoothing-scale", "10000000", "--smoothing-offset", "1", "--smoothing-exponent", "0.25",
    "--step-scale", "10", "--step-offset", "10", "--step-exponent", "1"]
command = common + ["--steps", str(case["total_steps"]), "--fidelity-target", "0.99",
    "--restart-path", str(output / "latest_restart.npz"), "--save", str(output / "final.npz")]
parent_hash = None
expected_initial_hash = plan["fresh_initial_factor_sha256"]
if case.get("resume"):
    parent = Path(case["resume"])
    parent_hash = sha(parent)
    with np.load(parent, allow_pickle=False) as z:
        parent_meta = json.loads(str(z["metadata_json"].item()))
        assert parent_meta["completed_steps"] == case["start_step"]
        assert parent_meta["configuration"]["batch_size"] == case["batch_size"]
        expected_initial_hash = hashlib.sha256(np.ascontiguousarray(z["factor"]).tobytes()).hexdigest()
    command += ["--resume", str(parent)]

provenance = dict(case=case, command=command, data_sha256=sha(data), data_metadata=data_meta,
    source_sha256=hashes, parent_sha256=parent_hash, expected_initial_factor_sha256=expected_initial_hash,
    started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    slurm_job_id=os.environ["SLURM_JOB_ID"], slurm_array_job_id=os.environ["SLURM_ARRAY_JOB_ID"],
    slurm_array_task_id=os.environ["SLURM_ARRAY_TASK_ID"],
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), xla_flags=os.environ.get("XLA_FLAGS"))
(output / "provenance.json").write_text(json.dumps(provenance, indent=2))
if case.get("smoke_steps"):
    smoke = common + ["--steps", str(case["smoke_steps"]), "--save", str(output / "smoke.npz")]
    print("SMOKE", json.dumps(smoke), flush=True)
    subprocess.run(smoke, cwd=source, check=True)
    with np.load(output / "smoke.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        assert meta["initial_factor_sha256"] == plan["fresh_initial_factor_sha256"]
        assert np.all(np.isfinite(z["final_factor"]))

print("PRODUCTION", json.dumps(command), flush=True)
subprocess.run(command, cwd=source, check=True)
with np.load(output / "final.npz", allow_pickle=False) as z:
    meta = json.loads(str(z["metadata_json"].item()))
    assert meta["initial_factor_sha256"] == expected_initial_hash
    assert meta["noise_std"] == .05 and meta["n_qubits"] == 8
    summary = dict(metadata=meta, trajectory=[dict(step=int(s), fidelity=float(f),
        tp_violation=float(t), measurement_loss=float(l)) for s, f, t, l in zip(
            z["checkpoint_steps"], z["process_fidelity_proxy"], z["tp_violation"], z["measurement_loss"])])
(output / "summary.json").write_text(json.dumps(summary, indent=2))
subprocess.run([sys.executable, str(root / "verify_saved_factors.py"), "--data", str(data),
    "--result", str(output / "final.npz"), "--save", str(output / "verification.json")], check=True)
if case.get("resume"):
    assert sha(case["resume"]) == parent_hash, "Parent restart changed"
print("COMPLETE", case["name"], json.dumps(summary["trajectory"][-1]), flush=True)
