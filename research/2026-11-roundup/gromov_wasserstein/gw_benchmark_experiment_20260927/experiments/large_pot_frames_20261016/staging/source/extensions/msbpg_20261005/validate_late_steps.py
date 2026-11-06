"""Actual-instance minimum-step numerical gate, not a late-trajectory benchmark.

At each standard initialization, form one q128 stochastic gradient and use it
as a representative incoming momentum and fresh gradient. Apply precisely the
k=T-1 schedule at the smallest candidate prefactors. This tests small-step
certificate cancellation and FP32 quantization on real-size arrays; it does
not claim to reproduce the optimizer's state after T-1 updates.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_msbpg import (Clock, initialization, load_problem, memory, setup, sha256,
                       source_hashes, verified_input, write_json)
from msbpg import constants_from_problem, msbpg_step, normalized_displacement
from stable_prox import StableCertifiedKLProx


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--horizon',type=int,default=2000)
    p.add_argument('--coupling-fraction',type=float,default=.125)
    p.add_argument('--momentum-prefactor',type=float,default=.1)
    p.add_argument('--certificate-block-rows',type=int,default=256)
    p.add_argument('--exact-block-rows',type=int,default=256)
    return p


@torch.no_grad()
def run(args,device):
    if args.horizon<2:
        raise ValueError('The late-step gate requires a horizon of at least two.')
    clock=Clock(device)
    clock.start()
    inputs=verified_input(Path(args.instance),device)
    problem=load_problem(args.instance,torch.float32,device,args.exact_block_rows)
    constants=constants_from_problem(problem,block_rows=args.certificate_block_rows)
    prox=StableCertifiedKLProx(problem,gap_tol=1e-7,certificate_block_rows=args.certificate_block_rows)
    preparation=clock.stop()
    results=[]
    for initialization_name in ('product','randomized'):
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        clock.start()
        P=initialization(problem,initialization_name,713)
        logP=torch.empty(P.shape,dtype=torch.float64,device=device)
        for start in range(0,problem.local_rows,args.certificate_block_rows):
            block=slice(start,start+args.certificate_block_rows)
            logP[block]=P[block].double().log()
        generator=torch.Generator(device=device).manual_seed(714)
        gradient=problem.sketch_gradient(P,problem.rademacher(128,generator=generator))
        Q,logQ,momentum,info=msbpg_step(logP,gradient,gradient,iteration=args.horizon-1,
            constants=constants,prox=prox,marginal_residuals=prox.marginal_residuals,
            inner_tol=1e-7,gap_tol=1e-7,inner_max=10000,check_every=10,
            coupling_fraction=args.coupling_fraction,momentum_prefactor=args.momentum_prefactor,
            require=problem._require)
        solver_seconds=clock.stop()
        clock.start()
        changed=torch.zeros((),dtype=torch.int64,device=device)
        physical_squared=torch.zeros((),dtype=torch.float64,device=device)
        for start in range(0,problem.local_rows,args.certificate_block_rows):
            block=slice(start,start+args.certificate_block_rows)
            difference=Q[block].double()-P[block].double()
            changed.add_((Q[block]!=P[block]).sum())
            physical_squared.add_((difference/info['alpha']).square().sum())
        record=dict(initialization=initialization_name,iteration=args.horizon-1,status='passed',
            certificate=info,solver_seconds=solver_seconds,
            retained_log_normalized_displacement_squared=normalized_displacement(logP,logQ,info['alpha'],
                block_rows=args.certificate_block_rows,global_sum=problem._reduce),
            stored_plan_normalized_displacement_squared=problem._reduce(physical_squared).item(),
            stored_plan_changed_entries=problem._reduce(changed).item(),
            global_plan_entries=problem.n*problem.m,
            probe_calls=1,sketch_probes=128,
            incoming_momentum='Same representative q128 gradient as fresh gradient; this is a numerical gate, not a trajectory',
            diagnostic_seconds=clock.stop())
        records=[None]*dist.get_world_size()
        dist.all_gather_object(records,dict(rank=dist.get_rank(),memory=memory(device)))
        record['ranks']=records
        results.append(record)
        del P,Q,logP,logQ,momentum,gradient
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
    payload=dict(status='passed',method_key='published-msbpg',horizon=args.horizon,
        iteration=args.horizon-1,coupling_fraction=args.coupling_fraction,
        momentum_prefactor=args.momentum_prefactor,theorem_constants=constants,
        dimensions=dict(source=problem.n,target=problem.m),checks=results,
        preparation_seconds=preparation,world_size=dist.get_world_size(),
        array_files_sha256=inputs['hashes'],instance_manifest_sha256=inputs['manifest_sha256'],
        source_file_sha256=source_hashes(),validation_driver_sha256=sha256(Path(__file__)),
        device=str(device),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        scope='Minimum theoretical step at standard initializations with representative momentum; no claim to reproduce a late optimizer trajectory or establish exact-proximal convergence.')
    if dist.get_rank()==0:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        write_json(args.out,payload)
        print(f'Passed late-step numerical gate for {problem.n} x {problem.m} on {dist.get_world_size()} ranks.',flush=True)
    return payload


def main():
    args=parser().parse_args()
    rendezvous=None
    try:
        if args.device=='cpu' and 'RANK' not in os.environ:
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
            torch.use_deterministic_algorithms(True)
            rendezvous=tempfile.TemporaryDirectory(prefix='msbpg_late_step_')
            dist.init_process_group('gloo',init_method='file://'+str(Path(rendezvous.name)/'store'),rank=0,world_size=1)
            device=torch.device('cpu')
        else:
            device=setup(args.device)
        run(args,device)
    except BaseException as exc:
        rank=dist.get_rank() if dist.is_initialized() else int(os.environ.get('RANK','0'))
        if rank==0:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.out,dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if rendezvous is not None:
            rendezvous.cleanup()


if __name__=='__main__':
    main()
