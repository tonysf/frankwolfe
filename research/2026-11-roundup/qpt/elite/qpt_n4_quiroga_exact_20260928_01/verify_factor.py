"""Independent NumPy verification of a serialized rank-one n=4 factor."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.qpt_structured_operators import product_measurement_vectors, rank_one_measurement_vectors
from subset import fixed_subset

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--result", type=Path, required=True)
p.add_argument("--save", type=Path, required=True)
a = p.parse_args()
with np.load(a.data, allow_pickle=False) as z:
    truth = z["truth_factor"].copy(); basis = z["local_basis"].copy(); n = int(z["n_qubits"])
with np.load(a.result, allow_pickle=False) as z:
    factor = z["final_factor"].copy(); meta = json.loads(str(z["metadata_json"].item()))
    saved_f = float(z["process_fidelity_proxy"][-1]); saved_tp = float(z["tp_violation"][-1])
    step = int(z["checkpoint_steps"][-1])
d = 2**n
assert n == 4 and factor.shape == truth.shape == (d*d, 1) and np.all(np.isfinite(factor))

def kraus(x):
    tensor=x[:,0].reshape((4,)*n); transform=basis.reshape(4,4).T
    for axis in range(n):
        tensor=np.moveaxis(np.tensordot(transform,tensor,axes=(1,axis)),0,axis)
    tensor=tensor.reshape((2,2)*n)
    return tensor.transpose(tuple(range(0,2*n,2))+tuple(range(1,2*n,2))).reshape(d,d)

actual, target = kraus(factor), kraus(truth)
fidelity=float(abs(np.vdot(target,actual))**2/(np.vdot(target,target).real*np.vdot(actual,actual).real))
tp=float(np.linalg.norm(actual.conj().T@actual-np.eye(d)))
assert np.isclose(fidelity,saved_f,rtol=1e-10,atol=1e-12)
assert np.isclose(tp,saved_tp,rtol=1e-8,atol=1e-8)
assert meta["n_steps"] == step
data = StructuredQPTData.load_npz(a.data)
_, symbols, observations, _ = fixed_subset(data, int(meta["subset_size"]), int(meta["subset_seed"]))
bank = rank_one_measurement_vectors(data.local_measurements)
vectors = product_measurement_vectors(symbols, bank)
amplitudes = vectors.conj() @ factor
predicted = np.sum(np.abs(amplitudes) ** 2, axis=1)
fixed_subset_measurement_loss_sum = float(0.5 * np.sum((predicted - observations) ** 2))
if meta.get("measurement_loss_normalization") == "sum_half_squared_as_published":
    with np.load(a.result, allow_pickle=False) as z:
        assert np.isclose(fixed_subset_measurement_loss_sum, float(z["measurement_loss"][-1]), rtol=1e-9, atol=1e-8)
report=dict(status="verified", optimizer=meta["optimizer"], subset_seed=meta["subset_seed"],
            step=step, fidelity=fidelity, tp_violation=tp, factor_norm=float(np.linalg.norm(factor)),
            fixed_subset_measurement_loss_sum=fixed_subset_measurement_loss_sum,
            measurement_loss_convention="0.5*sum_over_fixed_subset(residual^2)")
a.save.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps(report,indent=2))
