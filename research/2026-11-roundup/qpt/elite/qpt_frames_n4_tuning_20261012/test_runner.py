"""Validation, accounting, safety and tiny native-runner parity tests."""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_tuning as runner
import run_batch as batch


def fixture(phase='screen'):
    configs=[dict(id='base',alpha0=5e-5,penalty_exponent=.25,step_scale=4.,step_offset=4.,step_exponent=1.,
                  rho_scale=4.,rho_offset=8.,rho_exponent=2/3,batch_size=32,tau=4.)]
    rule=dict(channels=[0,1],initialization_replicates=[0],sampling_replicates=[0],config_ids=['base'],
              charged_row_budget=3200,checkpoint_charged_rows=320)
    spec=dict(configs=configs,phases={p:deepcopy(rule) for p in ('pilot','screen','long','confirmation')})
    trials=[]
    for cfg,ch,i,s in itertools.product(configs,rule['channels'],[0],[0]):
        c={k:v for k,v in cfg.items() if k!='id'}
        trials.append(dict(c,id=f"{cfg['id']}-c{ch}-i{i}-s{s}",config_id=cfg['id'],channel=ch,n_qubits=4,
                           initialization_replicate=i,initialization_seed=200000+10*ch+i,
                           sampling_replicate=s,sampling_seed=300000+100*ch+s,steps=100,metrics_every=10,charged_row_budget=3200))
    plan=dict(scope='local_frames_n4_tuning',phase=phase,gate=runner.GATE,source_commit=runner.COMMIT,
              reserved_final_channels=list(range(1000,1020)),trials=trials,total_charged_row_budget=6400,
              resources=dict(max_workers=2,cpu_affinity=[0,1],nice=10,batch_wall_seconds=600,trial_wall_seconds=300))
    if phase in ('long','confirmation'): plan['selected_config_ids']=['base']
    return plan,spec


@pytest.mark.parametrize('phase',['pilot','screen','long','confirmation'])
def test_declared_full_phase(phase):
    plan,spec=fixture(phase); runner.validate_plan(plan,spec=spec,check_files=False)


@pytest.mark.parametrize('field,value',[('alpha0',1e-3),('penalty_exponent',0.),('tau',2.),('batch_size',64),
    ('initialization_seed',3),('sampling_seed',3),('steps',99),('metrics_every',20),('charged_row_budget',42),
    ('n_qubits',3),('channel',1000),('id','../escape'),('rho_exponent',.6)])
def test_mutated_trial_rejected(field,value):
    p,s=fixture();p['trials'][0][field]=value
    with pytest.raises(ValueError): runner.validate_plan(p,spec=s,check_files=False)


def test_missing_duplicate_undeclared_trial_and_total_rejected():
    for change in ('missing','duplicate','total'):
        p,s=fixture()
        if change=='missing': p['trials'].pop()
        elif change=='duplicate': p['trials'].append(deepcopy(p['trials'][0]))
        else: p['total_charged_row_budget']+=1
        with pytest.raises(ValueError): runner.validate_plan(p,spec=s,check_files=False)


def test_row_budget_changes_steps_with_batch_without_changing_data_budget():
    p,s=fixture();s['configs'][0]['batch_size']=64
    for c in p['trials']: c.update(batch_size=64,steps=50,metrics_every=5)
    runner.validate_plan(p,spec=s,check_files=False)


def test_zero_exponent_and_unknown_grid_knob_rejected():
    p,s=fixture();s['configs'][0]['penalty_exponent']=0.
    with pytest.raises(ValueError): runner.validate_plan(p,spec=s,check_files=False)
    p,s=fixture();s['configs'][0]['new_algorithm']='forbidden'
    with pytest.raises(ValueError): runner.validate_plan(p,spec=s,check_files=False)


def test_later_phase_requires_frozen_selection():
    p,s=fixture('long');p.pop('selected_config_ids')
    with pytest.raises(ValueError,match='selection'): runner.validate_plan(p,spec=s,check_files=False)


def test_selection_contract_ids_and_spec_are_bound():
    p,_=fixture('long');p['campaign_spec_sha256']='spec'
    chosen=dict(selected_config_ids=['base'],spec_sha256='spec')
    runner.selection_contract(p,chosen)
    for key,value in [('selected_config_ids',['other']),('spec_sha256','other')]:
        bad=dict(chosen);bad[key]=value
        with pytest.raises(ValueError,match='Selection'):runner.selection_contract(p,bad)


def test_screen_requires_a_resource_decision(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    p,s=fixture('screen')
    with pytest.raises(FileNotFoundError):runner.execution_prerequisites(p,s)
    runner.write(tmp_path/'resource_decision.json',dict(allowed=False,quality_metrics_used_for_resource_decision=False))
    with pytest.raises(ValueError,match='resource gate'):runner.execution_prerequisites(p,s)


def test_resource_limits_and_external_plan_rejected():
    p,s=fixture();p['resources']['max_workers']=3
    with pytest.raises(ValueError,match='two workers'): runner.validate_plan(p,spec=s,check_files=False)
    with pytest.raises(ValueError,match='campaign root'): runner.plan_path_in_root('/tmp/plan_screen.json')


def test_runtime_guard_rejects_remote_affinity_threads_and_priority(monkeypatch):
    monkeypatch.setattr(runner.socket,'gethostname',lambda:'elite')
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:{0,1})
    monkeypatch.setattr(runner.os,'getpriority',lambda *a:10)
    for key in runner.THREAD_ENV:monkeypatch.setenv(key,'1')
    for key in ('SLURM_JOB_ID','SLURM_STEP_ID','PBS_JOBID'):monkeypatch.delenv(key,raising=False)
    runner.runtime_guard()
    monkeypatch.setattr(runner.socket,'gethostname',lambda:'ruche02')
    with pytest.raises(ValueError,match='Elite'):runner.runtime_guard()
    monkeypatch.setattr(runner.socket,'gethostname',lambda:'elite')
    monkeypatch.setenv('OMP_NUM_THREADS','2')
    with pytest.raises(ValueError,match='thread'):runner.runtime_guard()
    monkeypatch.setenv('OMP_NUM_THREADS','1');monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:{0,1,2})
    with pytest.raises(ValueError,match='affinity'):runner.runtime_guard()
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:{0,1});monkeypatch.setattr(runner.os,'getpriority',lambda *a:0)
    with pytest.raises(ValueError,match='priority'):runner.runtime_guard()


def test_exclusive_outputs_and_inventory_ownership(tmp_path):
    p=tmp_path/'record.json';runner.write(p,{'batch_id':'owner','status':'running'})
    with pytest.raises(FileExistsError):runner.write(p,{'bad':1})
    with pytest.raises(ValueError,match='ownership'):batch.update_inventory(p,{'batch_id':'other'})
    batch.update_inventory(p,{'batch_id':'owner','status':'complete'})
    assert json.loads(p.read_text())['status']=='complete'


@pytest.mark.parametrize('code,summary,wanted',[(0,{'status':'complete'},'finished'),(2,{'status':'failed'},'method_failure'),
                                               (0,{},'infrastructure_failure'),(1,{},'infrastructure_failure')])
def test_failure_inventory_classification(code,summary,wanted):
    assert batch.worker_outcome(code,summary)==wanted


def test_native_runner_parity_and_no_dense_bank(monkeypatch):
    """Only the main agent runs this bounded 100-update verification."""
    from paper.experiments.qpt_quiroga_sensing import QuirogaSensingData
    def forbidden(*a,**kw):raise AssertionError('No dense sensing bank permitted')
    monkeypatch.setattr(QuirogaSensingData,'to_dense_qpt_data',forbidden)
    p,_=fixture();c=p['trials'][0]
    data=runner.generate_quiroga_sensing_data(4,channel_seed=c['channel'],observation_mode='noiseless')
    factor0=runner.initial_factor(data,c['initialization_seed'])
    metadata,payload,trace=runner.run_native(data,factor0,c)
    reference,archive,traces=runner.run_campaign(data,methods=['frames'],rank=1,initialization_basis='gell-mann',
        initialization_seed=c['initialization_seed'],sampling_seed=c['sampling_seed'],n_steps=100,frames_steps=100,
        metrics_frequency=10,frames_metrics_frequency=10,eta_scale=1.,tp_weight=1.,tau=4.,batch_size=32,
        frames_beta0=10000.,frames_smoothing_schedule=runner.PowerSchedule(10000.,1.,.25),
        rho_schedule=runner.PowerSchedule(4.,8.,2/3,cap=1.),step_size_schedule=runner.PowerSchedule(4.,4.,1.,cap=1.),show_progress=False)
    assert trace.status=='complete'
    np.testing.assert_array_equal(trace.final_factor,traces['frames'].final_factor)
    np.testing.assert_array_equal(payload['initial_factor'],factor0)
    np.testing.assert_array_equal(payload['frames_cumulative_data_row_accesses'],payload['frames_checkpoint_steps']*32)
    assert metadata['tuning_configuration']==c and metadata['tuning_backend']['dense_sensing_bank'] is False
    assert metadata['configuration']['methods']==['frames']
    assert metadata['configuration']['frames_penalty_mode']=='independent_schedule'
