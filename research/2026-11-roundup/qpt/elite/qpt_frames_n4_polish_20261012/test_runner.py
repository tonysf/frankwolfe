"""Polish schedule, provenance/safety guards and small unchanged-runner checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_polish as runner
import run_batch as batch


def fixture(phase='calibration'):
    channels,i,s=([0,1],0,0) if phase=='calibration' else ([2,3],1,1)
    trials=[dict(runner.FIXED_PARAMETERS,id=f'{runner.CONFIG_ID}-c{c}-i{i}-s{s}',channel=c,
                 initialization_replicate=i,initialization_seed=200000+10*c+i,
                 sampling_replicate=s,sampling_seed=300000+100*c+s) for c in channels]
    return dict(scope='local_frames_n4_polish',phase=phase,source_commit=runner.COMMIT,gate=runner.GATE,
                reserved_final_channels=list(range(1000,1020)),motivation='Separate conditional round',
                schedule=deepcopy(runner.PLAN_SCHEDULE),trials=trials,
                total_charged_row_budget=25600000,
                resources=dict(max_workers=2,cpu_affinity=[0,1],nice=10,trial_wall_seconds=1200,batch_wall_seconds=3600))


def test_boundary_uses_global_clock_and_exact_prior_power_arithmetic():
    step=runner.TwoPhaseStep();prior=runner.PowerSchedule(4**.75,4.,.75,cap=1.)
    assert step(0)==1. and step(47999)==prior(47999)
    assert step(48000)==4./48004. and step(48001)==4./48005.
    assert step(48000)<prior(48000) and step(48000)!=1.
    assert step.descriptor()['momentum_reset'] is False
    assert step.descriptor()['clock']=='absolute_zero_based_optimizer_update'


@pytest.mark.parametrize('phase',['calibration','confirmation'])
def test_exact_declared_profiles_validate(phase):runner.validate_plan(fixture(phase),check_files=False)


@pytest.mark.parametrize('key,value',[('switch_update',47999),('post_step_scale',5.),('post_step_exponent',.75),
    ('alpha0',5e-5),('penalty_exponent',0.),('step_scale',4.),('rho_exponent',.6),('steps',800000),
    ('metrics_every',1000),('sampling_seed',42),('initialization_seed',42),('channel',1000),('id','../outside')])
def test_no_silent_retuning_or_seed_changes(key,value):
    p=fixture();p['trials'][0][key]=value
    with pytest.raises(ValueError):runner.validate_plan(p,check_files=False)


def test_missing_cohort_and_unbounded_resource_rejected():
    p=fixture();p['trials'].pop()
    with pytest.raises(ValueError):runner.validate_plan(p,check_files=False)
    p=fixture();p['resources']['max_workers']=3
    with pytest.raises(ValueError):runner.validate_plan(p,check_files=False)
    with pytest.raises(ValueError):runner.path_guard('/tmp/plan_calibration.json')


@pytest.mark.parametrize('key',['momentum_reset','penalty_clock_reset','tp_projection','resume'])
def test_declared_no_reset_and_no_projection(key):
    p=fixture();p['schedule'][key]=True
    with pytest.raises(ValueError):runner.validate_plan(p,check_files=False)


def test_confirmation_cannot_run_without_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    with pytest.raises(FileNotFoundError):runner.confirmation_guard(fixture('confirmation'))


def test_runtime_affinity_thread_and_host_guards(monkeypatch):
    monkeypatch.setattr(runner.socket,'gethostname',lambda:'elite')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda p:{0,1})
    monkeypatch.setattr(runner.os,'getpriority',lambda *a:10)
    for key in runner.THREAD_ENV:monkeypatch.setenv(key,'1')
    for key in ('SLURM_JOB_ID','SLURM_STEP_ID','PBS_JOBID'):monkeypatch.delenv(key,raising=False)
    runner.runtime_guard()
    monkeypatch.setenv('OMP_NUM_THREADS','2')
    with pytest.raises(ValueError):runner.runtime_guard()
    monkeypatch.setenv('OMP_NUM_THREADS','1');monkeypatch.setattr(runner.socket,'gethostname',lambda:'ruche02')
    with pytest.raises(ValueError):runner.runtime_guard()


def test_exclusive_outputs_owned_inventory_and_method_failure(tmp_path):
    p=tmp_path/'inventory.json';runner.write(p,dict(batch_id='ours',status='running'))
    with pytest.raises(FileExistsError):runner.write(p,{})
    with pytest.raises(ValueError):batch.update_inventory(p,dict(batch_id='other'))
    batch.update_inventory(p,dict(batch_id='ours',status='complete'))
    assert json.loads(p.read_text())['status']=='complete'
    assert batch.worker_outcome(2,{'status':'failed'})=='method_failure'
    assert batch.worker_outcome(1,{'status':'failed'})=='infrastructure_failure'
    assert batch.worker_outcome(0,{})=='infrastructure_failure'


def test_short_two_phase_callable_matches_native_direct_schedule(monkeypatch):
    """Main-agent test only:30 updates with synthetic switch20; no400k test."""
    from paper.experiments.qpt_quiroga_sensing import QuirogaSensingData
    def forbidden(*a,**kw):raise AssertionError('Dense bank is forbidden')
    monkeypatch.setattr(QuirogaSensingData,'to_dense_qpt_data',forbidden)
    data=runner.generate_quiroga_sensing_data(4,channel_seed=0,observation_mode='noiseless')
    common=dict(methods=['frames'],rank=1,initialization_basis='gell-mann',initialization_seed=200000,
        sampling_seed=300000,n_steps=30,frames_steps=30,metrics_frequency=5,frames_metrics_frequency=5,
        eta_scale=1.,tp_weight=1.,tau=4.,batch_size=32,frames_beta0=640000.,
        frames_smoothing_schedule=runner.PowerSchedule(640000.,1.,.25),
        rho_schedule=runner.PowerSchedule(4.,8.,2/3,cap=1.),show_progress=False)
    schedule=runner.TwoPhaseStep(switch_update=20)
    _,actual,ta=runner.run_campaign(data,step_size_schedule=schedule,**common)
    explicit=lambda t:min(1.,4**.75/(t+4)**.75) if t<20 else min(1.,4./(t+4))
    _,reference,tr=runner.run_campaign(data,step_size_schedule=explicit,**common)
    np.testing.assert_array_equal(ta['frames'].final_factor,tr['frames'].final_factor)
    for key in ('frames_diagnostic_step_size','frames_diagnostic_momentum_weight','frames_diagnostic_smoothing_parameter',
                'frames_rmse','frames_normalized_tp_violation','frames_process_fidelity'):
        np.testing.assert_array_equal(actual[key],reference[key])
    prefix_kwargs={**common,'n_steps':20,'frames_steps':20}
    _,prefix,tp=runner.run_campaign(data,step_size_schedule=runner.PowerSchedule(4**.75,4.,.75,cap=1.),**prefix_kwargs)
    for key in ('frames_rmse','frames_normalized_tp_violation','frames_process_fidelity'):
        np.testing.assert_array_equal(actual[key][:5],prefix[key])
    assert actual['frames_diagnostic_step_size'][20]==4./24.
    assert actual['frames_diagnostic_momentum_weight'][20]==min(1.,4./28.**(2/3))
