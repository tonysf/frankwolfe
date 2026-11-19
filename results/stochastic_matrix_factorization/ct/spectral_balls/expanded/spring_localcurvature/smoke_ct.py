"""Bounded real-data verification of the corrected steps, outside evaluation."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.sched_setaffinity(0, {6})

import hashlib
import json
from pathlib import Path

import numpy as np

from ct_problem import CTProblem
from local_curvature import make_solver
from methods import op_norm


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'convincing_4x_recovery'
CONFIG = dict(batch=64, refresh_period=16, power_iterations=5, step_coefficient=.5)
answer = dict(status='running', purpose='Implementation smoke only; not parameter selection or campaign evaluation',
              source_sha256=hashlib.sha256((ROOT/'local_curvature.py').read_bytes()).hexdigest(),
              optimizer=CONFIG, cases=[])


def save():
    path = ROOT/'ct_smoke.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


for n, iterations in ((360,28),(1080,8)):
    problem = CTProblem(OLD/'data'/'slice03611.npz',512,n)
    try:
        with np.load(OLD/'calibration'/'slice03611_seed201.npz') as saved:
            U,V,caps = saved['U'],saved['V'],saved['caps']
        solver = make_solver(problem,U,V,caps,seed=100203,config=CONFIG)
        loss0 = problem.image_loss(U@V.T,problem.train_ids)
        item = dict(n_train=n, seed=203, slice_id='slice03611', requested_iterations=iterations,
                    trace=[dict(iteration=0,loss=loss0,relative_loss=1.)])
        answer['cases'].append(item)
        save()
        for k in range(iterations):
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                step = solver.step()
            loss = problem.image_loss(solver.U@solver.V.T,problem.train_ids)
            assert np.isfinite(loss) and np.min(solver.U)>=0 and np.min(solver.V)>=0
            assert op_norm(solver.U)<=caps[0]*(1+1e-10) and op_norm(solver.V)<=caps[1]*(1+1e-10)
            assert step['curvature_operator_calls']==12 and step['curvature_forward_views']==768
            assert step['oracle_calls']==step['gradient_calls']+12
            item['trace'].append(dict(iteration=k+1,loss=loss,relative_loss=loss/loss0,step=step))
            save()
            if (k+1)%4==0:
                print(json.dumps(dict(n_train=n,iteration=k+1,relative_loss=loss/loss0)),flush=True)
        item.update(status='passed',endpoint_ratio=loss/loss0,
                    max_ratio=max(row['relative_loss'] for row in item['trace']))
        save()
    finally:
        problem.close()
answer['status']='passed'
save()
print('CT smoke passed',flush=True)
