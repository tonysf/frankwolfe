"""Validate and run the one authorized n=8 QPT continuation in Slurm."""
from __future__ import print_function

import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("This driver must run inside a Slurm allocation")

import numpy as np

OUT = Path(os.environ["QPT_OUTPUT"])
CONFIG = json.loads((OUT / "proposal.json").read_text())
SOURCE = Path(CONFIG["source"])
PARENT = Path(CONFIG["parent_result"])
RESUME = Path(CONFIG["resume"])
DATA = Path(CONFIG["dataset"])
FINAL = OUT / "final.npz"
RESTART = OUT / "latest_restart.npz"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_result(path):
    names = {
        "step": "checkpoint_steps",
        "fidelity": "process_fidelity_proxy",
        "measurement_loss": "measurement_loss",
        "tp_violation": "tp_violation",
        "smoothed_gap": "smoothed_gap",
        "smoothed_objective": "smoothed_objective",
        "smoothing_parameter": "checkpoint_smoothing_parameters",
    }
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {key: archive[name].copy() for key, name in names.items()}
        final_factor = archive["final_factor"].copy()
        gradient_estimate = archive["gradient_estimate"].copy()
        schedules = {name: archive[name].copy() for name in
                     ("step_sizes", "smoothing_parameters", "momentum_weights")}
    length = len(arrays["step"])
    assert length and all(len(value) == length for value in arrays.values())
    assert all(np.all(np.isfinite(value)) for value in arrays.values())
    assert np.all(np.diff(arrays["step"]) > 0)
    trajectory = [
        {key: int(value[i]) if key == "step" else float(value[i])
         for key, value in arrays.items()}
        for i in range(length)
    ]
    return metadata, trajectory, final_factor, gradient_estimate, schedules


status = {
    "status": "validating",
    "started_at_utc": now(),
    "job_id": os.environ["SLURM_JOB_ID"],
    "hostname": socket.gethostname(),
}
write_json(OUT / "status.json", status)

try:
    selection = json.loads(Path(CONFIG["source_selection"]).read_text())
    assert selection["status"] == "validated" and selection["selected"] == "fast"
    assert selection["source"] == str(SOURCE)
    source_hashes = {name: sha(SOURCE / name) for name in selection["source_sha256"]}
    assert source_hashes == selection["source_sha256"], "Frozen source changed"
    assert sha(DATA) == CONFIG["dataset_archive_sha256"], "Dataset archive changed"
    assert sha(RESUME) == CONFIG["resume_archive_sha256"], "Parent restart changed"

    with np.load(RESUME, allow_pickle=False) as archive:
        assert set(archive.files) == {"factor", "gradient_estimate", "metadata_json"}
        restart_factor = archive["factor"].copy()
        restart_estimate = archive["gradient_estimate"].copy()
        restart_meta = json.loads(str(archive["metadata_json"].item()))
    assert restart_meta["format"] == "qpt_structured_restart"
    assert restart_meta["schema_version"] == 1
    assert restart_meta["completed_steps"] == CONFIG["start_step"] == 148000
    assert restart_meta["configuration"]["data_sha256"] == CONFIG["dataset_internal_digest"]
    assert restart_meta["configuration"] == {
        "data_sha256": CONFIG["dataset_internal_digest"],
        "rank": 1,
        "tau": 20.0,
        "batch_size": 65536,
        "precision": "64",
        "measurement_backend": "product-state",
    }
    assert restart_factor.shape == restart_estimate.shape == (65536, 1)
    assert restart_factor.dtype == restart_estimate.dtype == np.dtype("complex128")
    assert np.all(np.isfinite(restart_factor)) and np.all(np.isfinite(restart_estimate))
    resumed_factor_sha256 = array_sha(restart_factor)

    parent_meta, parent_trajectory, parent_factor, _, _ = load_result(PARENT)
    assert parent_meta["n_steps"] == 148000 and parent_meta["start_step"] == 0
    assert array_sha(parent_factor) == resumed_factor_sha256
    assert parent_trajectory[-1]["step"] == 148000

    command = [
        sys.executable, "-m", "paper.experiments.quantum_process_tomography_structured_jax",
        "--data", str(DATA), "--device", "gpu", "--precision", "64",
        "--measurement-backend", "product-state", "--rank", "1", "--tau", "20",
        "--batch-size", "65536", "--chunk-steps", "1", "--metrics-every", "1000",
        "--metric-mode", "sampled", "--metric-samples", "512",
        "--metric-batch-size", "512", "--metric-seed", "12345",
        "--initialization-seed", "0", "--sampling-seed", "0",
        "--rho-scale", "0.6", "--rho-offset", "4", "--rho-exponent", "0.5",
        "--smoothing-scale", "10000000.0", "--smoothing-offset", "1",
        "--smoothing-exponent", "0.25", "--step-scale", "0.5",
        "--step-offset", "10", "--step-exponent", "0.75", "--prefetch",
        "--steps", "200000", "--restart-path", str(RESTART), "--save", str(FINAL),
        "--resume", str(RESUME),
    ]
    provenance = {
        "question": CONFIG["question"],
        "type": "exact_continuation",
        "parent_case": CONFIG["parent_case"],
        "command": command,
        "source": str(SOURCE),
        "source_sha256": source_hashes,
        "source_selection": CONFIG["source_selection"],
        "dataset": str(DATA),
        "dataset_archive_sha256": CONFIG["dataset_archive_sha256"],
        "dataset_internal_digest": CONFIG["dataset_internal_digest"],
        "resume": str(RESUME),
        "resume_archive_sha256": CONFIG["resume_archive_sha256"],
        "resumed_factor_sha256": resumed_factor_sha256,
        "parent_initial_factor_sha256": CONFIG["parent_initial_factor_sha256"],
        "start_step": 148000,
        "total_steps": 200000,
        "additional_steps": 52000,
        "environment": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "hostname": socket.gethostname(),
        "started_at_utc": now(),
        "driver_sha256": sha(Path(__file__)),
    }
    write_json(OUT / "provenance.json", provenance)
    status["status"] = "running"
    status["production_started_at_utc"] = now()
    write_json(OUT / "status.json", status)
    with (OUT / "optimizer.log").open("x") as log:
        subprocess.run(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)

    meta, trajectory, factor, estimate, schedules = load_result(FINAL)
    assert meta["n_steps"] == meta["requested_n_steps"] == 200000
    assert meta["start_step"] == 148000 and meta["segment_steps"] == 52000
    assert meta["resume_from"] == str(RESUME)
    assert meta["batch_size"] == 65536 and meta["precision"] == "64"
    assert meta["measurement_backend"] == "product-state" and meta["prefetch"] is True
    assert meta["fidelity_target"] is None and not meta["fidelity_target_reached"]
    assert meta["initial_factor_sha256"] == resumed_factor_sha256
    assert meta["sampled_measurements"] == 52000 * 65536
    assert trajectory[0]["step"] == 148000 and trajectory[-1]["step"] == 200000
    for key in ("fidelity", "measurement_loss", "tp_violation"):
        assert np.isclose(trajectory[0][key], parent_trajectory[-1][key], rtol=1e-12, atol=1e-14), key
    k = np.arange(148000, 200000, dtype=float)
    expected = {
        "step_sizes": np.minimum(1.0, 0.5 / (k + 10.0) ** 0.75),
        "smoothing_parameters": 10000000.0 / (k + 1.0) ** 0.25,
        "momentum_weights": np.minimum(1.0, 0.6 / (k + 4.0) ** 0.5),
    }
    for name, values in expected.items():
        assert schedules[name].shape == values.shape
        assert np.allclose(schedules[name], values, rtol=1e-13, atol=0), name
    assert factor.shape == estimate.shape == (65536, 1)
    assert np.all(np.isfinite(factor)) and np.all(np.isfinite(estimate))

    final_ten = [row for row in trajectory if 191000 <= row["step"] <= 200000]
    assert [row["step"] for row in final_ten] == list(range(191000, 200001, 1000))
    crossings = [row["step"] for row in trajectory if row["fidelity"] >= 0.99]
    summary = {
        "metadata": meta,
        "trajectory": trajectory,
        "schedule_verification": "All 52,000 applied scalar values match the frozen exponents, scales, offsets and caps.",
        "continuity_verification": "The resumed step-148000 factor hash and initial fidelity/loss/TP match the parent endpoint.",
        "resumed_factor_sha256": resumed_factor_sha256,
        "parent_endpoint": parent_trajectory[-1],
        "endpoint": trajectory[-1],
        "first_saved_crossing_0p99": min(crossings) if crossings else None,
        "all_final_checkpoints_191k_200k_ge_0p99": all(row["fidelity"] >= 0.99 for row in final_ten),
        "final_checkpoints_191k_200k": final_ten,
    }
    write_json(OUT / "summary.json", summary)

    verification_script = Path(CONFIG["verification_script"])
    with (OUT / "verification.log").open("x") as log:
        subprocess.run([sys.executable, str(verification_script), "--data", str(DATA),
                        "--result", str(FINAL), "--save", str(OUT / "verification.json")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    verification = json.loads((OUT / "verification.json").read_text())
    assert verification["status"] == "verified" and verification["step"] == 200000
    assert sha(DATA) == CONFIG["dataset_archive_sha256"]

    checksums = {path.name: sha(path) for path in sorted(OUT.iterdir())
                 if path.is_file() and path.name not in ("checksums.json", "status.json")}
    write_json(OUT / "checksums.json", checksums)
    status.update({
        "status": "completed",
        "completed_at_utc": now(),
        "endpoint": trajectory[-1],
        "first_saved_crossing_0p99": summary["first_saved_crossing_0p99"],
        "all_final_checkpoints_191k_200k_ge_0p99": summary["all_final_checkpoints_191k_200k_ge_0p99"],
        "optimizer_seconds": meta["optimizer_seconds"],
        "runner_seconds": meta["wall_seconds"],
    })
    write_json(OUT / "status.json", status)
    print("COMPLETE", json.dumps(status, sort_keys=True), flush=True)
except Exception:
    status.update({"status": "failed", "failed_at_utc": now(), "error": traceback.format_exc()})
    write_json(OUT / "status.json", status)
    raise
