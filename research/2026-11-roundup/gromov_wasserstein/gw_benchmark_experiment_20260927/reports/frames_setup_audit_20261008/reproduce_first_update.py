import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
import sys,json,torch,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/tony/gw_benchmark_experiment_20260927/extensions/shareseq_25000_20260930')
from sharded_gw import RowShardedGW
torch.set_num_threads(1)
f=np.load('/home/tony/gw_benchmark_experiment_20260927/data/bmmc1000.npz')
A,B,a,b=[torch.tensor(f[k],dtype=torch.float64) for k in ('A','B','a','b')]
p=RowShardedGW(A,B,a,b,exact_block_rows=256); P=torch.outer(a,b)
G=p.gradient(P)
rows=[]
for seed in (810,811,812):
 Z=p.rademacher(128,generator=torch.Generator().manual_seed(seed));S=p.sketch_gradient(P,Z)
 for name,d in [('exact',G),('sketch',S)]:
  Q=p.fw_step(P,d,.1,.1/p.m);R=p.round_transport(Q)
  rows.append({'seed':seed,'oracle':name,'chosen_columns':len(d.argmin(1).unique()),'raw_l1_columns':p.marginal_residuals(Q)['col_l1'],'rounded_distance_from_product_l1':float((R-P).abs().sum()),'initial_loss':float(p.loss(P)),'rounded_loss':float(p.loss(R))})
Path('/tmp/audit_frames_oracle/first_update.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
