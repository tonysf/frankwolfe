"""Small ASTRA checks for the paper baselines; never used to select settings."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.sched_setaffinity(0,{6})

import hashlib
import json
from pathlib import Path

import numpy as np

from ct_problem import CTProblem
from methods import op_norm
import palm_paper
import prox_paper


ROOT=Path(__file__).resolve().parent
SOURCES=('palm_paper.py','prox_paper.py','curvature_reference.py','methods.py','ct_problem.py',
         'spectral_reference_baseline.py','spectral_reference_core.py')
answer=dict(status='running',purpose='Bounded implementation smoke only; not calibration or evaluation',
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES},cases=[])


def save():
    target=ROOT/'method_smoke.json'
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
    temp.replace(target)


for method,n,steps in (('palm',360,2),('palm',1080,1),('prox_sarah',360,6),('prox_sarah',1080,4)):
    problem=CTProblem(ROOT/'data/slice03611.npz',512,n)
    try:
        with np.load(ROOT/'calibration/slice03611_seed201.npz') as calibration:
            U,V,caps=calibration['U'],calibration['V'],calibration['caps']
            scale=float(np.sum(calibration['curvature']))
        if method=='palm':
            solver=palm_paper.make_solver(problem,U,V,caps,seed=100203,config=palm_paper.parameters(n))
        else:
            solver=prox_paper.make_solver(problem,U,V,caps,seed=100203,config={'loss_scale':scale})
        initial=problem.image_loss(U@V.T,problem.train_ids)
        case=dict(method=method,n_train=n,slice_id='slice03611',seed=203,requested_iterations=steps,
                  optimizer=solver.config,trace=[dict(iteration=0,loss=initial,relative_loss=1.)])
        answer['cases'].append(case)
        save()
        for k in range(steps):
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                step=solver.step()
            loss=problem.image_loss(solver.U@solver.V.T,problem.train_ids)
            assert np.isfinite(loss) and np.min(solver.U)>=0 and np.min(solver.V)>=0
            assert op_norm(solver.U)<=caps[0]*(1+1e-10) and op_norm(solver.V)<=caps[1]*(1+1e-10)
            assert step['forward_views']==step['backward_views']
            assert step['oracle_calls']==step['gradient_calls']+step['curvature_operator_calls']
            if method=='palm':
                assert step['forward_views']==14*n and step['oracle_calls']==14
            else:
                assert step['curvature_operator_calls']==0
                assert step['forward_views']==(n if k==0 else 2*solver.batch)
            case['trace'].append(dict(iteration=k+1,loss=loss,relative_loss=loss/initial,step=step))
            save()
            print(json.dumps(dict(method=method,n_train=n,iteration=k+1,relative_loss=loss/initial)),flush=True)
        case.update(status='passed',endpoint_ratio=loss/initial,max_ratio=max(row['relative_loss'] for row in case['trace']))
        save()
    finally:
        problem.close()
answer['status']='passed'
save()
print('Both paper baselines passed the real CT smoke.',flush=True)
