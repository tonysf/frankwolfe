"""Independent high-accuracy POT first-subproblem check, no outer trajectory."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json, time, warnings, hashlib
import numpy as np
import ot
ROOT = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
instance = ROOT/'experiments/scot_cpu_20261014/reference/prepared/scot1047'
a, b = np.load(instance/'a.npy'), np.load(instance/'b.npy')
rows=[]
for oracle in ('full','pair'):
    own = np.load(out/(oracle+'_plain')/'plan.npy')
    gradient = np.load(out/(oracle+'_plain')/'gradient.npy')
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        independent, log = ot.bregman.sinkhorn_log(a,b,gradient,.001,
            numItermax=10000, stopThr=1e-12, log=True, warn=True)
    rows.append(dict(oracle=oracle, pot_version=ot.__version__, stopThr=1e-12,
        numItermax=10000, niter=int(log['niter']),
        warnings=[str(w.message) for w in caught], seconds=time.perf_counter()-start,
        marginal_l1=float(np.abs(independent.sum(1)-a).sum()+np.abs(independent.sum(0)-b).sum()),
        plan_l1_difference=float(np.abs(independent-own).sum()),
        plan_max_difference=float(np.abs(independent-own).max()),
        explanation='Independent POT solve is tighter than production stopping criteria; this tests the same subproblem, not parity with historical POT finite-inner settings.'))
receipt=dict(results=rows, source_hashes={name:hashlib.sha256((out/name).read_bytes()).hexdigest()
    for name in ('preflight.py','preflight_plain.py','pot_parity.py')})
(out/'pot_parity.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(receipt),flush=True)
