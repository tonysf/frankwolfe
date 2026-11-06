"""Allocated GPU/Gloo validation of theorem-matched MSBPG, including empty shards."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/'shareseq_25000_20260930'))
sys.path.insert(0, str(HERE.parent/'published_stochastic_20261002'))
from gw_core import kl_prox as reference_prox
from sharded_gw import RowShardedGW, row_partition
from stable_prox import StableCertifiedKLProx
from msbpg import constants_from_problem, msbpg_step, normalized_displacement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    if args.device=='cuda':
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('GPU validation must run in a Slurm allocation.')
        device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        if torch.cuda.device_count()!=int(os.environ['LOCAL_WORLD_SIZE']):
            raise RuntimeError('Use one process per visible allocated GPU.')
    else:
        device=torch.device('cpu')
    dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(minutes=5))
    rank,world=dist.get_rank(),dist.get_world_size()
    rows=[]
    started=time.perf_counter()
    def check(name,condition,**metrics):
        failure=torch.tensor(int(not condition),device=device)
        dist.all_reduce(failure,op=dist.ReduceOp.MAX)
        rows.append(dict(name=name,passed=failure.item()==0,**metrics))
        if failure.item():
            raise AssertionError(name)
    try:
        for dtype in (torch.float64,torch.float32):
            for n in (2,3,17):
                label=f'{dtype}/n{n}'
                lo,hi=row_partition(n,world,rank)
                rng=np.random.default_rng(713+n)
                x=rng.normal(size=(n,3))
                A=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
                A=(A/A.max()).astype(np.float32)
                B=A[::-1,::-1].copy()
                A=torch.tensor(A,dtype=dtype,device=device)
                B=torch.tensor(B,dtype=dtype,device=device)
                a=torch.full((n,),1/n,dtype=dtype,device=device)
                problem=RowShardedGW(A[lo:hi].clone(),B,a,a.clone(),exact_block_rows=3)
                constants=constants_from_problem(problem,block_rows=2)
                W=(A.double()[:,None,:,None]-B.double()[None,:,None,:]).square().reshape(n*n,n*n)
                expected_bound=2*W.sum(1).max().item()
                check(label+'/ambient_hessian_bound',constants['hessian_bound']>=expected_bound,
                      hessian_bound=constants['hessian_bound'],exact_row_bound=expected_bound)
                check(label+'/kernel_constants',constants['mu']==1 and constants['kernel_scale']==a.max().item()
                      and constants['kappa']==2*constants['hessian_bound']**2)
                solver=StableCertifiedKLProx(problem,gap_tol=1e-7,certificate_block_rows=2)
                fullP=torch.outer(a,a)
                logP=fullP[lo:hi].double().log()
                full_logP=fullP.double().log()
                momentum=None
                full_momentum=None
                for k in range(3):
                    fullG=torch.tensor(rng.normal(size=(n,n)),dtype=dtype,device=device)
                    beta=.6/math.sqrt(k) if k else None
                    full_momentum=fullG if k==0 else (1-beta)*full_momentum+beta*fullG
                    previous=logP
                    Q,logP,momentum,info=msbpg_step(logP,fullG[lo:hi],momentum,iteration=k,
                        constants=constants,prox=solver,marginal_residuals=solver.marginal_residuals,
                        inner_tol=1e-7,gap_tol=1e-7,inner_max=10000,check_every=1,
                        coupling_fraction=.8,momentum_prefactor=.6,require=problem._require)
                    alpha=.8/(2*math.sqrt(2)*constants['hessian_bound'])*.6/math.sqrt(k+1)
                    check(label+f'/k{k}/index_and_coupling',info['beta']==beta
                          and abs(info['alpha']/alpha-1)<1e-14
                          and info['eta']==info['alpha']/constants['kernel_scale'])
                    check(label+f'/k{k}/momentum',bool(torch.equal(momentum,full_momentum[lo:hi])))
                    expected,full_logP,ref_info=reference_prox(full_logP,full_momentum.double(),
                        a.double(),a.double(),info['eta'],tol=1e-13,max_iter=10000,check_every=1)
                    check(label+f'/k{k}/dense_reference_converged',ref_info['converged'])
                    check(label+f'/k{k}/independent_solution',bool(torch.allclose(Q.double(),expected[lo:hi],atol=2e-8,rtol=2e-6)))
                    check(label+f'/k{k}/both_criteria',info['marginal_l1']<=1e-7
                          and info['normalized_gap_upper_bound']<=1e-7,
                          marginal_l1=info['marginal_l1'],normalized_gap_upper_bound=info['normalized_gap_upper_bound'])
                    check(label+f'/k{k}/retained_log64',logP.dtype==torch.float64)
                    statistic=normalized_displacement(previous,logP,info['alpha'],block_rows=2,global_sum=problem._reduce)
                    check(label+f'/k{k}/normalized_displacement',math.isfinite(statistic) and statistic>=0,
                          normalized_displacement_squared=statistic)
                bad=momentum.clone()
                if rank==0:
                    bad[0,0]=float('nan')
                rejected=False
                try:
                    msbpg_step(logP,fullG[lo:hi],bad,iteration=3,constants=constants,prox=solver,
                        marginal_residuals=solver.marginal_residuals,inner_tol=1e-7,inner_max=100,
                        coupling_fraction=.8,momentum_prefactor=.6,require=problem._require)
                except (ValueError,FloatingPointError):
                    rejected=True
                check(label+'/one_rank_nonfinite_momentum_rejected',rejected)
        payload=dict(status='passed',checks=rows,check_count=len(rows),world_size=world,
            device=str(device),torch=torch.__version__,cuda=torch.version.cuda,
            job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.perf_counter()-started,
            scope='Small MSBPG schedule, momentum, row-shard and certificate checks; not performance results.')
        if rank==0:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(status='passed',checks=len(rows),world_size=world)))
    except BaseException as exc:
        if rank==0:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(dict(status='failed',checks=rows,error=repr(exc)),indent=2)+'\n')
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
