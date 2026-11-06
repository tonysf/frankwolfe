import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,itertools,json
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,'/home/tony/gw_benchmark_experiment_20260927/extensions/shareseq_25000_20260930')
from sharded_gw import RowShardedGW
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
gen=torch.Generator().manual_seed(420)
A=torch.cdist(torch.randn(4,3,generator=gen),torch.randn(4,3,generator=gen));A=(A+A.T)/2
B=torch.cdist(torch.randn(5,2,generator=gen),torch.randn(5,2,generator=gen));B=(B+B.T)/2
a=torch.rand(4,generator=gen);a/=a.sum();b=torch.rand(5,generator=gen);b/=b.sum()
pr=RowShardedGW(A,B,a,b,exact_block_rows=2)
checks={}
for plan in ('positive','signed'):
 P=torch.rand(4,5,generator=gen) if plan=='positive' else torch.randn(4,5,generator=gen)
 if plan=='positive': P*=a[:,None]/P.sum(1)[:,None]
 P.requires_grad_()
 C=(A[:,:,None,None]-B[None,None,:,:]).square()
 f=torch.einsum('ikjl,ij,kl->',C,P,P)
 G=torch.autograd.grad(f,P)[0]
 allsigns=torch.tensor(list(itertools.product([-1.,1.],repeat=5))).T
 Z=allsigns[:,[3,15,17]]
 U=P@Z; V=P@(B@Z)
 r=P.sum(1);c=P.sum(0)
 fs=r@(A.square()@r)+c@(B.square()@c)-2*(U*(A@V)).sum()/Z.shape[1]
 gs=torch.autograd.grad(fs,P)[0]
 checks[plan]={'loss_abs_error':float(abs(pr.loss(P)-f)), 'gradient_max_abs_error':float((pr.gradient(P)-G).abs().max()),'sketch_autograd_max_abs_error':float((pr.sketch_gradient(P,Z)-gs).abs().max()),'exhaustive_rademacher_gradient_max_abs_error':float((pr.sketch_gradient(P,allsigns)-G).abs().max()),'exhaustive_rademacher_loss_abs_error':float(abs(pr.sketch_loss(P,allsigns)-f))}
root=Path('/home/tony/gw_benchmark_experiment_20260927')
arr=np.load(root/'data/bmmc1000.npz')
A,B,a,b=[torch.tensor(arr[k],dtype=torch.float64) for k in ('A','B','a','b')]
pr=RowShardedGW(A,B,a,b,exact_block_rows=256)
P0=torch.outer(a,b);G0=pr.gradient(P0)
P1=.9*P0; perm=torch.randperm(len(a),generator=gen);P1[torch.arange(len(a)),perm]+=.1*a
rowcenter=lambda x: x-x.mean(1,keepdim=True)
doublecenter=lambda x: rowcenter(x)-rowcenter(x).mean(0,keepdim=True)
stats={}
for name,P in [('product',P0),('randomized',P1)]:
 G=pr.gradient(P);Gc=rowcenter(G);Gt=doublecenter(G)
 denom=(G*P).sum()-sum(a*G.min(1).values)
 rows=[]
 for rep in range(16):
  Z=pr.rademacher(128,generator=gen)
  S=pr.sketch_gradient(P,Z)
  SC=S-pr.sketch_gradient(P0,Z)+G0
  d={}
  for label,estimate in [('raw',S),('control_variate',SC)]:
   E=estimate-G; Ec=rowcenter(E); Et=doublecenter(E)
   chosen=estimate.argmin(1); lmo=(a*G[torch.arange(len(a)),chosen]).sum()
   d[label]={'error_relative_full':float(E.norm()/G.norm()),'error_relative_rowcenter':float(Ec.norm()/Gc.norm()),'error_relative_doublecenter':float(Et.norm()/Gt.norm()),'error_energy_rowconstant_fraction':float(1-Ec.square().sum()/E.square().sum()) if E.norm()>0 else 0,'error_energy_rowcenter_columnconstant_fraction':float(1-Et.square().sum()/Ec.square().sum()) if Ec.norm()>0 else 0,'lmo_true_gap_fraction':float(((G*P).sum()-lmo)/denom),'lmo_exact_argmin_fraction':float((chosen==G.argmin(1)).double().mean()),'lmo_unique_columns':int(chosen.unique().numel())}
  rows.append(d)
 stats[name]={'shape':list(P.shape),'initial_loss':float(pr.loss(P)),'gradient_norm':float(G.norm()),'rowcenter_gradient_norm':float(Gc.norm()),'doublecenter_gradient_norm':float(Gt.norm()),'exact_row_lmo_gap':float(denom),'exact_lmo_unique_columns':int(G.argmin(1).unique().numel()),'replicates':16,'metrics':{label:{k:float(np.mean([r[label][k] for r in rows])) for k in rows[0][label]} for label in rows[0]}}
report={'tiny_checks':checks,'bmmc1000':stats}
Path('/tmp/audit_frames_oracle/result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
