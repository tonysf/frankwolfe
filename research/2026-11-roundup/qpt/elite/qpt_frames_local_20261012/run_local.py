"""Local-only FRAMES calibration wrapper around the unchanged campaign runner."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
sys.path.insert(0, str(REPO))
import numpy as np
import scipy
from paper.experiments.qpt_quiroga_campaign import run_campaign, save_campaign, evaluate_common_metrics
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.quantum_process_tomography import PowerSchedule

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')

def source():
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    names=subprocess.check_output(['git','-C',str(REPO),'ls-files','paper','frank_wolfe'],text=True).splitlines()
    return {name:digest(REPO/name) for name in names if name.endswith('.py')}

def check_config(c):
    assert type(c['channel']) is int and 0<=c['channel']<100, 'Calibration only; held-out and historical seeds forbidden.'
    for key in ('steps','metrics_every','batch_size'):
        assert type(c[key]) is int and c[key]>0
    assert c['steps']%c['metrics_every']==0
    for key in ('alpha0','penalty_exponent','step_scale','step_offset','step_exponent','rho_scale','rho_offset','rho_exponent'):
        assert np.isfinite(c[key]) and c[key]>0
    assert c['tau']==2. and c['batch_size']==32, 'Fixed n2 radius and batch for first calibration.'

def execute(plan_path, trial_id):
    plan=json.loads(plan_path.read_text())
    c=next(t for t in plan['trials'] if t['id']==trial_id)
    check_config(c)
    assert plan['source_commit']==COMMIT and plan['scope']=='local_n2_full_noiseless_frames_calibration'
    assert plan['gate']=={'fidelity_min':.99,'rmse_max':1e-6,'normalized_tp_max':.001}
    assert plan['phase'] in ('pilot','screen','long','confirmation','sampling_confirmation')
    assert c['channel'] in ((0,1) if plan['phase'] in ('pilot','screen','long') else tuple(range(2,8)))
    if plan.get('harness_sha256'):
        assert all(digest(ROOT/k)==v for k,v in plan['harness_sha256'].items()), 'Frozen harness changed.'
    src=source()
    assert src==plan['source_sha256']
    out=ROOT/plan['phase']/c['id']
    out.mkdir(parents=True,exist_ok=False)
    provenance={'trial':c,'plan_path':str(plan_path),'plan_sha256':digest(plan_path),
                'source_commit':COMMIT,'source_sha256':src,'runner_sha256':digest(__file__),
                'host':socket.gethostname(),'platform':platform.platform(),'python':platform.python_version(),
                'numpy':np.__version__,'scipy':scipy.__version__,'pid':os.getpid(),
                'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                'cpu_affinity':sorted(os.sched_getaffinity(0)),
                'started_utc':now(),'execution':'local_cpu','split':'calibration'}
    write(out/'started.json',provenance)
    start=time.perf_counter()
    try:
        data=generate_quiroga_sensing_data(2,channel_seed=c['channel'],observation_mode='noiseless')
        beta0=1/(2*c['alpha0'])
        cfg=dict(methods=['frames'],rank=1,initialization_basis='gell-mann',
                 initialization_seed=200000+10*c['channel'],sampling_seed=300000+100*c['channel']+c.get('sampling_replicate',0),
                 n_steps=c['steps'],frames_steps=c['steps'],metrics_frequency=c['metrics_every'],
                 frames_metrics_frequency=c['metrics_every'],eta_scale=1.,tp_weight=1.,
                 tau=c['tau'],batch_size=c['batch_size'],frames_beta0=beta0,
                 frames_smoothing_schedule=PowerSchedule(beta0,1.,c['penalty_exponent']),
                 step_size_schedule=PowerSchedule(c['step_scale'],c['step_offset'],c['step_exponent'],cap=1.),
                 rho_schedule=PowerSchedule(c['rho_scale'],c['rho_offset'],c['rho_exponent'],cap=1.),show_progress=False)
        if plan.get('backend')=='jax_dense_cpu64_scan':
            from paper.experiments.quantum_process_tomography import make_factor_initial_point,unpack_factor
            from paper.experiments.qpt_quiroga_initialization import gell_mann_coefficients_to_choi
            from jax_backend import run
            factor0=gell_mann_coefficients_to_choi(unpack_factor(make_factor_initial_point(
                data,rank=1,seed=cfg['initialization_seed']),16,1))
            metadata,payload,trace=run(data,factor0,c)
        else:
            metadata,payload,traces=run_campaign(data,**cfg)
            trace=traces['frames']
        ep=evaluate_common_metrics(data,trace.final_factor,np.arange(128,dtype=np.int64),data.all_observations(),0.)
        gate=trace.status=='complete' and ep['process_fidelity']>=.99 and ep['rmse']<=1e-6 and ep['normalized_tp_violation']<=.001
        ratio=max(ep['rmse']/1e-6,ep['normalized_tp_violation']/.001,max(0,1-ep['process_fidelity'])/.01)
        # Validate every returned scalar against the native archive schema.
        st=payload['frames_checkpoint_steps']
        assert np.array_equal(payload['frames_cumulative_data_row_accesses'],st*c['batch_size'])
        if trace.status=='complete':
            assert int(st[-1])==c['steps']
            assert np.allclose(payload['frames_measurement_loss_mean'],.5*payload['frames_rmse']**2,rtol=1e-13,atol=0)
            assert np.isclose(ep['rmse'],payload['frames_rmse'][-1],rtol=1e-12,atol=1e-15)
        passes=(payload['frames_rmse']<=1e-6)&(payload['frames_normalized_tp_violation']<=.001)&(payload['frames_process_fidelity']>=.99)
        first=int(st[np.flatnonzero(passes)[0]]) if passes.any() else None
        persistent=np.logical_and.accumulate(passes[::-1])[::-1]
        persistent_first=int(st[np.flatnonzero(persistent)[0]]) if persistent.any() else None
        metadata['local_calibration']={**provenance,'independent_penalty':True,'endpoint_gate':bool(gate),
            'gate_deficit':float(ratio),'first_recorded_gate_step':first,'persistent_recorded_gate_step':persistent_first,
            'qualification':'FRAMES-only calibration; no Quiroga retuning, no held-out use; tp_weight=1 is unused legacy argument.'}
        save_campaign(out/'result.npz',out/'result.json',metadata,payload)
        result={'trial':c,'status':trace.status,'gate_pass':bool(gate),'deficit':float(ratio),
                'rmse':float(ep['rmse']),'normalized_tp':float(ep['normalized_tp_violation']),
                'fidelity':float(ep['process_fidelity']),'first_recorded_gate_step':first,
                'persistent_recorded_gate_step':persistent_first,
                'optimizer_seconds':float(trace.optimizer_seconds[-1]),'total_seconds':time.perf_counter()-start,
                'json_sha256':digest(out/'result.json'),'npz_sha256':digest(out/'result.npz'),
                'finished_utc':now(),'failure_reason':trace.failure_reason}
        write(out/'summary.json',result)
        print(json.dumps(result),flush=True)
        return result
    except Exception as error:
        write(out/'failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'finished_utc':now(),
                                 'total_seconds':time.perf_counter()-start})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--trial',required=True)
    a=p.parse_args()
    execute(a.plan,a.trial)
