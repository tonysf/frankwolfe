"""Allocated single-A100 float32 checks against frozen-array float64 evaluation."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import torch
import numpy as np
from gw_core import GWProblem, kl_prox, assignment_lmo, round_transport, marginal_residuals, rademacher
from smoothing_fw import MomentumConfig, initialize_state, step
from capture_environment import capture


def main():
    p=argparse.ArgumentParser(); p.add_argument('--instance',required=True); p.add_argument('--out',required=True)
    args=p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    env=capture('cuda',1)
    z=np.load(args.instance)
    pr=GWProblem(*(torch.tensor(z[key],device='cuda',dtype=torch.float32) for key in ('A','B','a','b')))
    ref=GWProblem(*(torch.tensor(z[key],dtype=torch.float64) for key in ('A','B','a','b')))
    n=len(pr.a); rows=[]
    def validate(label,P):
        P64=P.cpu().double()
        loss32=pr.loss(P).item(); loss64=ref.loss(P64).item()
        grad_error=(pr.gradient(P).cpu().double()-ref.gradient(P64)).abs().max().item()
        R=round_transport(P,pr.a,pr.b)
        residual=marginal_residuals(R,pr.a,pr.b)
        row={'label':label,'raw_loss_float32':loss32,'raw_loss_float64':loss64,'loss_absolute_error':abs(loss32-loss64),'gradient_max_absolute_error':grad_error,'raw':marginal_residuals(P,pr.a,pr.b),'rounded':residual,'rounded_loss_float64':ref.loss(R.cpu().double()).item(),'small_negative_float32_loss':loss32<0 and loss64>=-1e-12}
        assert row['loss_absolute_error']<3e-6,row
        assert grad_error<1e-5,row
        assert residual['both_l1']<2e-6,row
        assert torch.isfinite(P).all() and P.min()>=0,row
        rows.append(row)
    P0=torch.outer(pr.a,pr.b)
    validate('product',P0)
    off=P0.clone(); off[:,0]+=off[:,1]; off[:,1]=0
    validate('row_feasible_wrong_columns',off)
    for oracle in ('exact','sketch'):
        cfg=MomentumConfig(alpha=.75,beta=.5,delta=.25,gamma0=1,rho0=1,mu0=1/n,oracle=oracle,batch=32)
        state=initialize_state(pr,cfg,seed=0)
        for _ in range(16):
            step(pr,state)
            assert (state.P.sum(1)-pr.a).abs().sum()<2e-6
        validate('momentum_'+oracle,state.P)
    for method in ('fw','kl','stochastic-kl'):
        P=P0.clone(); logP=P.log(); gen=torch.Generator(device='cuda').manual_seed(0)
        inner=[]
        for _ in range(5):
            G=pr.sketch_gradient(P,rademacher(n,32,like=P,generator=gen)) if method=='stochastic-kl' else pr.gradient(P)
            if method=='fw':
                S=assignment_lmo(G,pr.a,pr.b); D=S-P
                ell=(G*D).sum().item(); perm=S.argmax(1)
                q=(pr.A-pr.B[perm][:,perm]).square().mean().item()-.5*(G*P).sum().item()-ell
                gamma=min(1.,max(0.,-ell/(2*q))) if q>0 else float(ell+q<0)
                P=P+gamma*D
            else:
                P,logP,info=kl_prox(logP,G,pr.a,pr.b,eta=10 if method=='kl' else 1,tol=1e-6,max_iter=1000)
                inner.append(info)
                assert info['converged'],info
        validate(method,P)
        rows[-1]['inner_solve_info']=inner
    (out/'validation.json').write_text(json.dumps({'environment':env,'instance':args.instance,'status':'passed','rows':rows,'precision_note':'Float64 promotes canonical stored float32 costs and actual returned plans; losses are not clamped.'},indent=2)+'\n')
    print(f'{len(rows)} A100 float32 checkpoints validated against float64')

if __name__=='__main__': main()
