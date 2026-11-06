"""Independent NumPy verification of saved QPT factors; run in an allocation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--result", type=Path, required=True)
parser.add_argument("--save", type=Path, required=True)
args = parser.parse_args()
if args.save.exists():
    raise FileExistsError(args.save)
with np.load(args.data, allow_pickle=False) as z:
    truth = z["truth_factor"].copy()
    basis = z["local_basis"].copy()
    n = int(z["n_qubits"])
    data_meta = json.loads(str(z["metadata_json"].item()))
with np.load(args.result, allow_pickle=False) as z:
    factor = z["final_factor"].copy()
    meta = json.loads(str(z["metadata_json"].item()))
    step = int(z["checkpoint_steps"][-1])
    saved_fidelity = float(z["process_fidelity_proxy"][-1])
    saved_tp = float(z["tp_violation"][-1])
d = 2**n
assert n in (4, 5, 8, 10) and factor.shape == truth.shape == (d*d, 1)
assert factor.dtype == truth.dtype == np.dtype("complex128")
assert np.all(np.isfinite(factor)) and np.all(np.isfinite(truth))
assert meta["noise_std"] == data_meta["noise_std"]
truth_norm2 = float(np.vdot(truth, truth).real)
factor_norm2 = float(np.vdot(factor, factor).real)
assert truth_norm2 > 0 and factor_norm2 > 0
fidelity = float(abs(np.vdot(truth, factor))**2 / (truth_norm2*factor_norm2))

def kraus(coefficients):
    # Independent contraction implementation; no QPT operator or JAX import.
    tensor = coefficients[:, 0].reshape((4,)*n)
    transform = basis.reshape(4, 4).T
    for axis in range(n):
        tensor = np.moveaxis(np.tensordot(transform, tensor, axes=(1, axis)), 0, axis)
    tensor = tensor.reshape((2, 2)*n)
    return tensor.transpose(tuple(range(0, 2*n, 2))+tuple(range(1, 2*n, 2))).reshape(d, d)

actual_matrix, truth_matrix = kraus(factor), kraus(truth)
matrix_fidelity = float(abs(np.vdot(truth_matrix, actual_matrix))**2 /
                        (np.vdot(truth_matrix, truth_matrix).real*np.vdot(actual_matrix, actual_matrix).real))
tp = float(np.linalg.norm(actual_matrix.conj().T @ actual_matrix - np.eye(d)))
truth_tp = float(np.linalg.norm(truth_matrix.conj().T @ truth_matrix - np.eye(d)))
assert np.isclose(truth_norm2, d, rtol=1e-10, atol=1e-10)
assert np.isclose(factor_norm2, np.vdot(actual_matrix, actual_matrix).real, rtol=1e-10, atol=1e-10)
assert truth_tp < 1e-8
assert np.isclose(fidelity, matrix_fidelity, rtol=1e-11, atol=1e-12)
assert np.isclose(fidelity, saved_fidelity, rtol=1e-10, atol=1e-12)
assert np.isclose(tp, saved_tp, rtol=1e-8, atol=1e-8)
assert np.sqrt(factor_norm2) <= meta["tau"] + 1e-9
assert meta["n_steps"] == step
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()
report = dict(status="verified", checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    n_qubits=n, step=step, noise_std=data_meta["noise_std"], fidelity_numpy=fidelity,
    fidelity_matrix_numpy=matrix_fidelity, saved_fidelity=saved_fidelity,
    tp_violation_numpy=tp, saved_tp_violation=saved_tp, truth_tp_violation=truth_tp,
    factor_norm2=factor_norm2, truth_norm2=truth_norm2, target_reached=fidelity >= .99,
    primary_target_met=data_meta["noise_std"] == .05 and fidelity >= .99,
    data=str(args.data), result=str(args.result), data_sha256=sha(args.data), result_sha256=sha(args.result),
    script_sha256=sha(Path(__file__)), numpy_version=np.__version__, host=socket.gethostname(),
    slurm_job_id=os.environ.get("SLURM_JOB_ID"), result_optimizer_seconds=meta["optimizer_seconds"],
    initialization_seed=meta["initialization_seed"], sampling_seed=meta["sampling_seed"],
    initial_factor_sha256=meta["initial_factor_sha256"], verification="complete saved factors; NumPy only; no optimization")
with args.save.open("x") as handle:
    json.dump(report, handle, indent=2, allow_nan=False)
print(json.dumps(report, indent=2, allow_nan=False))

