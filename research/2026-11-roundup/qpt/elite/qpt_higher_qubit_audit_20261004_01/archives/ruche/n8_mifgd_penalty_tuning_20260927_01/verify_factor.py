"""Independent NumPy verification of an unconstrained FGD/MiFGD factor."""
import argparse, json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--result", type=Path, required=True); p.add_argument("--save", type=Path, required=True); a = p.parse_args()
with np.load(a.data, allow_pickle=False) as z:
    truth=z["truth_factor"].copy(); basis=z["local_basis"].copy(); n=int(z["n_qubits"])
with np.load(a.result, allow_pickle=False) as z:
    factor=z["final_factor"].copy(); meta=json.loads(str(z["metadata_json"].item())); saved_f=float(z["process_fidelity_proxy"][-1]); saved_tp=float(z["tp_violation"][-1]); step=int(z["checkpoint_steps"][-1])
d=2**n
assert factor.shape == truth.shape == (d*d,1) and np.all(np.isfinite(factor))
def kraus(x):
    tensor=x[:,0].reshape((4,)*n); transform=basis.reshape(4,4).T
    for axis in range(n): tensor=np.moveaxis(np.tensordot(transform,tensor,axes=(1,axis)),0,axis)
    tensor=tensor.reshape((2,2)*n)
    return tensor.transpose(tuple(range(0,2*n,2))+tuple(range(1,2*n,2))).reshape(d,d)
actual, target = kraus(factor), kraus(truth)
fidelity=float(abs(np.vdot(target,actual))**2/(np.vdot(target,target).real*np.vdot(actual,actual).real))
tp=float(np.linalg.norm(actual.conj().T@actual-np.eye(d)))
assert np.isclose(fidelity,saved_f,rtol=1e-10,atol=1e-12) and np.isclose(tp,saved_tp,rtol=1e-8,atol=1e-8)
assert meta["n_steps"] == step and meta["optimizer"] in {"fgd_fixed_tp","mifgd_fixed_tp"}
report=dict(status="verified", optimizer=meta["optimizer"], sampling_seed=meta["sampling_seed"], step=step, fidelity=fidelity, tp_violation=tp, factor_norm=float(np.linalg.norm(factor)), projection="none")
a.save.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); print(json.dumps(report,indent=2))
