"""Separate, predeclared two-phase FRAMES polish; native updates stay unchanged."""
import argparse
from dataclasses import dataclass
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/tony/qpt_frames_n4_tuning_20261012')
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
CONFIG_ID = 'penalty64-two-phase'
FIXED_PARAMETERS = dict(alpha0=5e-5/64, penalty_exponent=.25, rho_scale=4., rho_offset=8., rho_exponent=2/3,
    step_scale=4**.75, step_offset=4., step_exponent=.75, switch_update=48000,
    post_step_scale=4., post_step_offset=4., post_step_exponent=1., batch_size=32, tau=4.,
    n_qubits=4, steps=400000, metrics_every=4000, charged_row_budget=12800000, config_id=CONFIG_ID)
PLAN_SCHEDULE = dict(clock='global zero-based optimizer update', switch_update=48000,
    before='(4/(t+4))**0.75', after='4/(t+4)', momentum_reset=False,
    penalty_clock_reset=False, tp_projection=False, resume=False)
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
sys.path.insert(0, str(REPO))
import numpy as np
import scipy
from paper.experiments.qpt_quiroga_campaign import run_campaign, evaluate_common_metrics, save_campaign
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.qpt_quiroga_initialization import gell_mann_coefficients_to_choi
from paper.experiments.quantum_process_tomography import PowerSchedule, make_factor_initial_point, unpack_factor


def require(test, message):
    if not test: raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def source():
    require(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT, 'Source commit changed.')
    require(not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip(), 'Source worktree is dirty.')
    names=subprocess.check_output(['git','-C',str(REPO),'ls-files','paper','frank_wolfe'],text=True).splitlines()
    return {name:digest(REPO/name) for name in names if name.endswith('.py')}


@dataclass(frozen=True)
class TwoPhaseStep:
    """Choose a branch, never reset its absolute zero-based optimizer clock."""
    switch_update: int = 48000
    step_scale: float = 4**.75
    step_offset: float = 4.
    step_exponent: float = .75
    post_step_scale: float = 4.
    post_step_offset: float = 4.
    post_step_exponent: float = 1.

    def __post_init__(self):
        require(type(self.switch_update) is int and self.switch_update > 0, 'Invalid switch update.')
        for key, value in vars(self).items():
            require(not isinstance(value,bool) and np.isfinite(value) and value>0, 'Invalid schedule parameter: '+key)

    def __call__(self, iteration):
        require(isinstance(iteration,(int,np.integer)) and not isinstance(iteration,bool) and iteration>=0, 'Invalid update index.')
        if iteration < self.switch_update:
            return min(1., self.step_scale/(iteration+self.step_offset)**self.step_exponent)
        return min(1., self.post_step_scale/(iteration+self.post_step_offset)**self.post_step_exponent)

    @classmethod
    def from_config(cls, config):
        return cls(**{key:config[key] for key in cls.__dataclass_fields__})

    def descriptor(self):
        return dict(type='two_phase_global_clock',switch_update=self.switch_update,
                    pre=dict(scale=self.step_scale,offset=self.step_offset,exponent=self.step_exponent,cap=1.),
                    post=dict(scale=self.post_step_scale,offset=self.post_step_offset,exponent=self.post_step_exponent,cap=1.),
                    clock='absolute_zero_based_optimizer_update',momentum_reset=False)


def runtime_guard():
    require(socket.gethostname().split('.')[0]=='elite', 'Local Elite execution only.')
    require(not any(os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_STEP_ID','PBS_JOBID')), 'Cluster execution forbidden.')
    affinity=set(os.sched_getaffinity(0))
    require(affinity and affinity<={0,1}, 'CPU affinity must be a subset of0,1.')
    require(all(os.environ.get(k)=='1' for k in THREAD_ENV), 'BLAS/OMP threads must equal1.')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10, 'Low priority nice>=10 required.')


def path_guard(path):
    path=Path(path).resolve()
    require(path.parent==ROOT and path.name.startswith('plan_') and path.suffix=='.json', 'Plan must belong to the new campaign root.')
    return path


def check_parent(plan):
    expected={str(PARENT/name) for name in ('completion_manifest.json','audit_long.json','selection_long.json')}
    require(set(plan['parent_pins'])==expected, 'Incomplete parent provenance.')
    require(all(digest(path)==sha for path,sha in plan['parent_pins'].items()), 'Parent evidence changed.')
    manifest=json.loads((PARENT/'completion_manifest.json').read_text())
    selection=json.loads((PARENT/'selection_long.json').read_text())
    audit=json.loads((PARENT/'audit_long.json').read_text())
    require(manifest['status']=='complete' and manifest['source_commit']==COMMIT, 'Parent round is not complete.')
    require(manifest['ruche_used'] is False and manifest['final_heldout_used'] is False, 'Parent scope changed.')
    require(manifest['recovery_winner_found'] is False and manifest['selected_config_ids']==[], 'Parent has an eligible winner.')
    require(selection['selected_config_ids']==[] and selection['confirmation_status']=='skipped_no_joint_gate_eligible_configuration',
            'Parent no-winner conclusion does not hold.')
    require(selection['audit_sha256']==digest(PARENT/'audit_long.json'), 'Parent selection audit changed.')
    for name in ('audit_long.json','selection_long.json'):
        require(manifest['evidence_sha256'][name]==digest(PARENT/name), 'Parent manifest does not bind evidence.')
    require(audit['phase_complete_and_valid'] is True and all(r['valid'] for r in audit['trials']), 'Parent long audit incomplete.')
    require(all(not r['both_pass'] for r in selection['ranking_inputs']), 'Parent eligible configuration was omitted.')
    for row in audit['trials']:
        for filename,key in [('result.json','json_sha256'),('result.npz','npz_sha256'),
                             ('summary.json','summary_sha256'),('started.json','started_sha256')]:
            require(digest(PARENT/'long'/row['id']/filename)==row[key], 'Parent long archive changed.')


def validate_plan(plan, *, check_files=True):
    require(plan['scope']=='local_frames_n4_polish' and plan['gate']==GATE, 'Wrong scope/gate.')
    require(plan['phase'] in ('calibration','confirmation') and plan['source_commit']==COMMIT, 'Wrong phase/source.')
    require(plan['reserved_final_channels']==list(range(1000,1020)), 'Reserved final channels changed.')
    require(bool(plan.get('motivation')) and plan.get('schedule')==PLAN_SCHEDULE, 'Missing rationale or changed predeclared schedule.')
    channels,initrep,stream=([0,1],0,0) if plan['phase']=='calibration' else ([2,3],1,1)
    require(len(plan['trials'])==2 and {c['channel'] for c in plan['trials']}==set(channels), 'Incorrect paired cohort.')
    for c in plan['trials']:
        require(all(c.get(k)==v for k,v in FIXED_PARAMETERS.items()), 'Predeclared optimizer profile changed.')
        for k in ('channel','initialization_replicate','sampling_replicate','initialization_seed','sampling_seed',
                  'n_qubits','steps','metrics_every','charged_row_budget','batch_size','switch_update'):
            require(type(c[k]) is int, 'Expected an integer: '+k)
        require(c['initialization_replicate']==initrep and c['sampling_replicate']==stream, 'Wrong paired seeds.')
        require(c['initialization_seed']==200000+10*c['channel']+initrep, 'Initialization seed mismatch.')
        require(c['sampling_seed']==300000+100*c['channel']+stream, 'Sampling seed mismatch.')
        require(c['id']==f"{CONFIG_ID}-c{c['channel']}-i{initrep}-s{stream}", 'Unexpected trial ID.')
    require(plan['total_charged_row_budget']==25600000, 'Phase row budget changed.')
    r=plan['resources']
    require(type(r['max_workers']) is int and 1<=r['max_workers']<=2 and r['cpu_affinity']==[0,1], 'Invalid CPU allocation.')
    require(type(r['nice']) is int and 10<=r['nice']<=19, 'Invalid nice priority.')
    require(type(r['trial_wall_seconds']) is int and type(r['batch_wall_seconds']) is int
            and 0<r['trial_wall_seconds']<=r['batch_wall_seconds']<=7200, 'Invalid bounded timeouts.')
    if check_files:
        require(plan['source_sha256']==source(), 'Source manifest changed.')
        check_parent(plan)
        require(set(plan['harness_sha256'])=={'run_polish.py','run_batch.py','prepare_polish.py'}, 'Incomplete harness manifest.')
        require(all(digest(ROOT/name)==sha for name,sha in plan['harness_sha256'].items()), 'Frozen harness changed.')


def confirmation_guard(plan):
    if plan['phase']=='calibration': return
    selection_path=ROOT/'selection_calibration.json'; selection=json.loads(selection_path.read_text())
    require(Path(plan['selection']['path']).resolve()==selection_path and plan['selection']['sha256']==digest(selection_path),
            'Confirmation selection pin mismatch.')
    require(selection['selected_config_ids']==[CONFIG_ID], 'No unique frozen polish winner.')
    require(selection['confirmation_status']=='planned' and plan['selected_config_ids']==[CONFIG_ID], 'Confirmation was not selected.')
    require(selection['audit_sha256']==digest(ROOT/'audit_calibration.json'), 'Calibration audit changed.')
    require(selection['calibration_plan_sha256']==digest(ROOT/'plan_calibration.json'), 'Calibration plan changed.')
    previous=json.loads((ROOT/'plan_calibration.json').read_text()); validate_plan(previous)
    audit=json.loads((ROOT/'audit_calibration.json').read_text())
    require(audit['phase_complete_and_valid'] is True and audit['plan_sha256']==digest(ROOT/'plan_calibration.json'),
            'Calibration audit incomplete or stale.')
    require(audit['batch_inventory_sha256']==digest(ROOT/'inventory_calibration.json'), 'Calibration inventory changed.')
    require(audit['expected_trials']==audit['validated_trials']==2 and len(audit['trials'])==2
            and {r['channel'] for r in audit['trials']}=={0,1}, 'Calibration pair incomplete.')
    for row in audit['trials']:
        require(row['valid'] and row['status']=='complete' and row['gate_pass'], 'Both calibration channels must pass.')
        require(row['prefix_check']['verified'] is True, 'Calibration prefix must be independently verified.')
        for name,key in [('result.json','json_sha256'),('result.npz','npz_sha256'),('summary.json','summary_sha256'),('started.json','started_sha256')]:
            require(digest(ROOT/'calibration'/row['id']/name)==row[key], 'Calibration archive changed.')


def initial_factor(data, seed):
    return gell_mann_coefficients_to_choi(unpack_factor(make_factor_initial_point(data,rank=1,seed=seed),data.process_dimension,1))


def run_native(data, factor0, c):
    require(all(c.get(k)==v for k,v in FIXED_PARAMETERS.items()), 'Polish profile changed.')
    require(data.n_qubits==4 and data.m==8192 and data.observation_mode=='noiseless'
            and data.metadata['channel_seed']==c['channel'], 'Wrong physical data.')
    require(np.array_equal(factor0,initial_factor(data,c['initialization_seed'])), 'Initial factor mismatches seed.')
    preserved=factor0.copy(); beta0=1/(2*c['alpha0']); step=TwoPhaseStep.from_config(c)
    metadata,payload,traces=run_campaign(data,methods=['frames'],rank=1,initialization_basis='gell-mann',
        initialization_seed=c['initialization_seed'],sampling_seed=c['sampling_seed'],n_steps=c['steps'],frames_steps=c['steps'],
        metrics_frequency=c['metrics_every'],frames_metrics_frequency=c['metrics_every'],eta_scale=1.,tp_weight=1.,tau=4.,batch_size=32,
        frames_beta0=beta0,frames_smoothing_schedule=PowerSchedule(beta0,1.,c['penalty_exponent']),
        rho_schedule=PowerSchedule(c['rho_scale'],c['rho_offset'],c['rho_exponent'],cap=1.),step_size_schedule=step,show_progress=False)
    require(np.array_equal(payload['initial_factor'],preserved) and np.array_equal(factor0,preserved), 'Initial factor changed.')
    metadata['polish_configuration']=dict(c)
    metadata['polish_backend']=dict(name='structured_numpy',precision='complex128',dense_sensing_bank=False,resume=False,
        repository_runner='paper.experiments.qpt_quiroga_campaign.run_campaign',observed_batch_indices_archived=False,
        step_schedule=step.descriptor(),momentum_reset=False,sampling='PCG64 with replacement; no realized batch archive',
        failure_archival='native runner retains initial factor on FRAMES exception')
    return metadata,payload,traces['frames']


def execute(plan_path, trial_id):
    runtime_guard(); plan_path=path_guard(plan_path); plan=json.loads(plan_path.read_text())
    validate_plan(plan); confirmation_guard(plan)
    matches=[c for c in plan['trials'] if c['id']==trial_id]; require(len(matches)==1, 'Trial not declared.')
    c=matches[0]; out=ROOT/plan['phase']/trial_id; out.mkdir(parents=True,exist_ok=False)
    pins={} if plan['phase']=='calibration' else {name:digest(ROOT/name) for name in
          ('selection_calibration.json','audit_calibration.json','plan_calibration.json','inventory_calibration.json')}
    started=dict(trial=c,plan_path=str(plan_path),plan_sha256=digest(plan_path),source_commit=COMMIT,source_sha256=plan['source_sha256'],
        parent_pins=plan['parent_pins'],harness_sha256=plan['harness_sha256'],runner_sha256=digest(__file__),execution_prerequisites_sha256=pins,
        host=socket.gethostname(),platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),threads={k:os.environ.get(k) for k in THREAD_ENV},
        pid=os.getpid(),started_utc=now(),execution='local_cpu',split=plan['phase'],charged_row_budget=c['charged_row_budget'])
    write(out/'started.json',started); began=time.perf_counter()
    try:
        data=generate_quiroga_sensing_data(4,channel_seed=c['channel'],observation_mode='noiseless')
        metadata,payload,trace=run_native(data,initial_factor(data,c['initialization_seed']),c); targets=data.all_observations()
        m=evaluate_common_metrics(data,trace.final_factor,np.arange(data.m,dtype=np.int64),targets,0.)
        gate=bool(trace.status=='complete' and m['rmse']<=1e-6 and m['normalized_tp_violation']<=.001 and m['process_fidelity']>=.99)
        deficit=max(m['rmse']/1e-6,m['normalized_tp_violation']/.001,max(0.,1-m['process_fidelity'])/.01)
        st=payload['frames_checkpoint_steps']; charged=payload['frames_cumulative_data_row_accesses']
        require(np.array_equal(charged,st*32), 'Charged-row accounting mismatch.')
        if trace.status=='complete':require(np.array_equal(st,np.arange(0,400001,4000)), 'Checkpoint grid mismatch.')
        require(np.isclose(m['rmse'],payload['frames_rmse'][-1],rtol=1e-9,atol=1e-15), 'Endpoint RMSE mismatch.')
        passing=(payload['frames_rmse']<=1e-6)&(payload['frames_normalized_tp_violation']<=.001)&(payload['frames_process_fidelity']>=.99)
        persistent=np.logical_and.accumulate(passing[::-1])[::-1]
        first=int(st[np.flatnonzero(passing)[0]]) if passing.any() else None
        stable=int(st[np.flatnonzero(persistent)[0]]) if persistent.any() else None
        metadata['polish_provenance']=dict(started,endpoint_gate=gate,gate_deficit=float(deficit),
            first_recorded_gate_step=first,persistent_recorded_gate_step=stable)
        save_campaign(out/'result.npz',out/'result.json',metadata,payload)
        result=dict(trial=c,status=trace.status,failure_reason=trace.failure_reason,gate_pass=gate,deficit=float(deficit),
            rmse=float(m['rmse']),normalized_tp=float(m['normalized_tp_violation']),fidelity=float(m['process_fidelity']),
            relative_rmse=float(m['rmse']*math.sqrt(data.m)/np.linalg.norm(targets)),outcome_scaled_rmse=float(32*m['rmse']),
            completed_steps=int(st[-1]),charged_rows=int(charged[-1]),charged_row_budget=c['charged_row_budget'],
            first_recorded_gate_step=first,persistent_recorded_gate_step=stable,optimizer_seconds=float(trace.optimizer_seconds[-1]),
            total_seconds=time.perf_counter()-began,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            json_sha256=digest(out/'result.json'),npz_sha256=digest(out/'result.npz'),finished_utc=now())
        write(out/'summary.json',result); print(json.dumps(result),flush=True); return result
    except Exception as error:
        write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),total_seconds=time.perf_counter()-began,finished_utc=now()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--trial',required=True)
    args=parser.parse_args();result=execute(args.plan,args.trial)
    raise SystemExit(0 if result['status']=='complete' else 2)
