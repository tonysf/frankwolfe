"""Separate instrumented update replay; never supplies primary wall-time rankings."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
import gw_core
import smoothing_fw
import run_benchmark as runner


@contextmanager
def profile_hooks(device, samples):
    """Synchronize individual kernels only in this explicitly separate replay."""
    def timed(function, label):
        def wrapped(*args, **kwargs):
            runner.sync(device); start=time.perf_counter()
            value=function(*args, **kwargs)
            runner.sync(device); samples[label].append(time.perf_counter()-start)
            return value
        return wrapped
    originals=(gw_core.GWProblem.sketch_gradient,smoothing_fw.rademacher,runner.rademacher,
               smoothing_fw.fw_row_update,runner.kl_prox)
    try:
        gw_core.GWProblem.sketch_gradient=timed(originals[0],'gradient_kernel')
        smoothing_fw.rademacher=timed(originals[1],'probe_generation')
        runner.rademacher=timed(originals[2],'probe_generation')
        smoothing_fw.fw_row_update=timed(originals[3],'row_fw_update')
        runner.kl_prox=timed(originals[4],'transport_kl_prox')
        yield
    finally:
        (gw_core.GWProblem.sketch_gradient,smoothing_fw.rademacher,runner.rademacher,
         smoothing_fw.fw_row_update,runner.kl_prox)=originals


def replay(problem, method, parameters, seed, warmup=20, measured=200, instrument=True):
    device=problem.A.device
    parameters=dict(parameters)
    config=smoothing_fw.MomentumConfig(alpha=.75,beta=.5,delta=.25,
        gamma0=parameters.get('gamma0',1.),rho0=.3,mu0=parameters.get('mu_scale',10.)/len(problem.b),
        oracle='sketch',batch=128)
    P,_=runner.make_initial_plan(problem,'product',100)
    if method=='smoothing-fw':
        state=smoothing_fw.initialize_state(problem,config,seed=seed,P0=P)
    else:
        generator=torch.Generator(device=device).manual_seed(seed)
        momentum=runner._kl_momentum_gradient(problem,P,None,generator,.3,128)
        logP=P.log()
    samples={k:[] for k in ('gradient_kernel','probe_generation','row_fw_update','transport_kl_prox')}
    totals=[]; failures=0
    from contextlib import nullcontext
    with profile_hooks(device,samples) if instrument else nullcontext():
        for k in range(warmup+measured):
            if k==warmup:
                for v in samples.values():v.clear()
            runner.sync(device);started=time.perf_counter()
            if method=='smoothing-fw':
                smoothing_fw.step(problem,state);P=state.P
            else:
                if k:
                    momentum=runner._kl_momentum_gradient(problem,P,momentum,generator,.3/(k+1)**.5,128)
                P,logP,info=runner.kl_prox(logP,momentum,problem.a,problem.b,
                    eta=parameters['eta']/(k+1)**parameters.get('eta_power',0.),
                    tol=parameters['inner_tol'],max_iter=1000)
                failures+=int(not info['converged'])
            runner._assert_plan(P);runner.sync(device)
            if k>=warmup:totals.append(time.perf_counter()-started)
    components={name:dict(calls=len(v),seconds=sum(v),median_call_seconds=statistics.median(v) if v else None)
                for name,v in samples.items()}
    return P,dict(warmup_updates=warmup,measured_updates=measured,measured_total_seconds=sum(totals),
        median_update_seconds=statistics.median(totals),components=components,
        other_update_seconds=sum(totals)-sum(x['seconds'] for x in components.values()),
        inner_stopping_failures_including_warmup=failures)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--instance',required=True)
    p.add_argument('--instance-sha256',required=True);p.add_argument('--method',choices=['smoothing-fw','stochastic-kl-momentum'],required=True)
    p.add_argument('--parameters',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=400)
    args=p.parse_args();out=Path(args.out)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU profiling must run in a Slurm allocation.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if out.exists():raise FileExistsError(out)
    instance=Path(args.instance)
    if runner.sha256_file(instance)!=args.instance_sha256:raise ValueError('Input binding mismatch.')
    device=runner.resolve_device('cuda');torch.cuda.set_device(device);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(instance,allow_pickle=False) as data:
        problem=gw_core.GWProblem(*(torch.as_tensor(data[k],dtype=torch.float32,device=device) for k in ('A','B','a','b')))
    _,result=replay(problem,args.method,json.loads(args.parameters),args.seed)
    result.update(config=vars(args),environment=runner.environment(device),source_commit=runner._command(['git','rev-parse','HEAD']),
        source_dirty=bool(runner._command(['git','status','--porcelain'])),
        source_file_sha256={name:runner.sha256_file(Path(__file__).parent/name) for name in ('profile_components.py','gw_core.py','smoothing_fw.py','run_benchmark.py')},
        interpretation='Fixed 220-update companion from product start: discard first20 timing observations, retain200. Extra synchronization changes throughput; components are descriptive and do not replace primary timing. Momentum mixing, penalties, checks and Python overhead remain in other_update_seconds.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
