import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.sched_setaffinity(0,{6})
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'convincing_4x_recovery'
sys.path.insert(0,str(OLD))
from ct_problem import CTProblem
from methods import SpringSARAH,op_norm
from run import method_config
name='evaluation_slice03611_360_spring_sarah_seed203_a2p3_q16'
reference=json.loads((OLD/'runs'/name/'trace.json').read_text())
metadata=json.loads((OLD/'runs'/name/'metadata.json').read_text())
class Diagnostic(SpringSARAH):
    def _update(self):
        U,V=self.U,self.V
        query_u=U,V
        eu,ru=self._estimate(query_u,self.previous_u,self.est_u,0)
        nu=self.project(U-self.eta_u*eu,self.ru)
        query_v=nu,V
        ev,rv=self._estimate(query_v,self.previous_v,self.est_v,1)
        nv=self.project(V-self.eta_v*ev,self.rv)
        self.U,self.V=nu,nv
        self.previous_u,self.previous_v=query_u,query_v
        self.est_u,self.est_v=eu,ev
        details={'refresh_u':ru,'refresh_v':rv}
        if self.label=='original' and 10<=self.k<=21:
            for block,old,new,other,est,eta in (
                ('U',U,nu,V,eu,self.eta_u),('V',V,nv,nu,ev,self.eta_v)):
                delta=new-old
                image=old@other.T if block=='U' else other@old.T
                delta_image=delta@other.T if block=='U' else other@delta.T
                r=self.problem.forward(image,self.train_ids).astype(np.float64)-self.problem.y[self.train_ids]
                pd=self.problem.forward(delta_image,self.train_ids).astype(np.float64)
                sq=float(np.sum(delta*delta))
                lin=float(np.sum(r*pd)/r.size)
                quad=float(np.sum(pd*pd)/r.size)
                details[block]=dict(
                    eta_curvature_on_actual_step=eta*quad/sq if sq else None,
                    true_directional_gradient=lin,
                    estimator_directional_gradient=float(np.sum(est*delta)),
                    predicted_block_loss_change=lin+.5*quad,
                    projected_step_residual=float(np.sum(est*delta)+sq/eta),
                    old_norm=float(np.linalg.norm(old)),new_norm=float(np.linalg.norm(new)),
                    new_opnorm=op_norm(new))
        return details

answer={'reference':name,'cases':[],'purpose':'Short diagnostic replay, not a new selected evaluation'}
for label,rate,q in [('original',2.3,16),('smaller_same_refresh',1.6,16),('development_screen_winner',1.6,32),('full_gradient_original_step',2.3,1)]:
    problem=CTProblem(OLD/'data'/'slice03611.npz',512,360)
    try:
        with np.load(OLD/'calibration'/'slice03611_seed201.npz') as saved:
            U,V=saved['U'],saved['V']; caps=saved['caps']; curvature=saved['curvature']
        cfg=dict(metadata['config'],rate=rate,refresh_period=q)
        solver=Diagnostic(problem,U,V,caps,seed=cfg['seed']+100000,config=method_config(cfg,curvature))
        solver.label=label
        trace=[dict(iteration=0,loss=problem.image_loss(U@V.T,problem.train_ids))]
        for k in range(28):
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                details=solver.step()
            loss=problem.image_loss(solver.U@solver.V.T,problem.train_ids)
            row=dict(iteration=k+1,loss=loss,relative_loss=loss/trace[0]['loss'],
                     u_norm=float(np.linalg.norm(solver.U)),v_norm=float(np.linalg.norm(solver.V)),
                     u_opnorm=op_norm(solver.U),v_opnorm=op_norm(solver.V),details=details)
            if label=='original' and loss!=reference[k+1]['raw_loss']:
                raise AssertionError((k+1,loss,reference[k+1]['raw_loss']))
            trace.append(row)
        item=dict(label=label,rate=rate,refresh_period=q,trace=trace,
                  endpoint_ratio=trace[-1]['relative_loss'],
                  max_ratio=max(x['loss']/trace[0]['loss'] for x in trace))
        answer['cases'].append(item)
        (ROOT/'replay.json').write_text(json.dumps(answer,indent=2)+'\n')
        print(label,'endpoint',item['endpoint_ratio'],'peak',item['max_ratio'],flush=True)
    finally:
        problem.close()
answer['status']='complete'
(ROOT/'replay.json').write_text(json.dumps(answer,indent=2)+'\n')
