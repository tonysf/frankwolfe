"""Bounded kernel-range diagnostic on fixed saved rounded validation plans.

These rounded diagnostic states are not the optimizer's raw states. Pair
draws use separate fresh diagnostic seeds, not production RNG replay.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, hashlib, importlib.util
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('old_entropy_run',ROOT/'extensions/scot_entropy_md_20261015/run.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
problem,_,_,_=module.load_problem(ROOT/'experiments/scot_cpu_20261014/reference/prepared/scot1047')
logref=(problem.a[:,None]*problem.b[None,:]).log()
states=(('full_gradient',3,None),('pair_seed811',3,100811),('pair_seed812',2,100812),('pair_seed813',3,100813))
rows=[]
with torch.no_grad():
    for name,iteration,seed in states:
        filename=ROOT/'experiments/scot_pot_step_validation_20261015/runs'/name/'checkpoints'/f'rounded_{iteration:06d}.npy'
        plan=torch.from_numpy(np.load(filename))
        if seed is None:
            gradient=problem.gradient(plan)
        else:
            rng=torch.Generator().manual_seed(seed)
            gradient=module.pair_gradient(problem,module.sample_pairs(problem,plan,1024,generator=rng))
        original=logref-1000*gradient
        row_shifted=original-original.max(dim=1,keepdim=True).values
        both_shifted=row_shifted-row_shifted.max(dim=0,keepdim=True).values
        def ranges(value):
            return dict(minimum=float(value.min()),maximum=float(value.max()),
                below_log_min_normal=int((value<np.log(np.finfo(np.float64).tiny)).sum()),
                below_log_min_subnormal=int((value<np.log(np.nextafter(np.float64(0),np.float64(1)))).sum()),
                exp_zero_entries=int((value.exp()==0).sum()),minimum_exp=float(value.exp().min()))
        rows.append(dict(name=name,saved_rounded_iteration=iteration,diagnostic_seed=seed,
            plan_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
            gradient_min=float(gradient.min()),gradient_max=float(gradient.max()),
            raw=ranges(original),row_shifted=ranges(row_shifted),row_then_column_shifted=ranges(both_shifted)))
receipt=dict(scope=__doc__,results=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
target=Path(__file__).resolve().parent/'kernel_ranges.json'
target.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps(receipt),flush=True)
