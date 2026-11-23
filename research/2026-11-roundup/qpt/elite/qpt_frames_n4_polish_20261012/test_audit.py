"""Independent polish audit tests; synthetic physics only, no optimization."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_polish as audit


def declared(phase='confirmation'):
    coords = [(0, 0), (1, 0)] if phase == 'calibration' else [(2, 1), (3, 1)]
    trials = [dict(audit.FROZEN, id=f"penalty64-two-phase-c{ch}-i{r}-s{r}", n_qubits=4, channel=ch,
                   initialization_replicate=r, sampling_replicate=r, initialization_seed=200000+10*ch+r,
                   sampling_seed=300000+100*ch+r, steps=400000, metrics_every=4000, charged_row_budget=12800000)
              for ch, r in coords]
    plan = dict(scope='local_frames_n4_polish', phase=phase, gate=audit.GATE, source_commit=audit.COMMIT,
                reserved_final_channels=list(range(1000,1020)), trials=trials, total_charged_row_budget=25600000,
                parent_pins={}, source_sha256={}, harness_sha256={'run_polish.py':'test-runner'},
                schedule=dict(clock='global zero-based optimizer update', switch_update=48000,
                    before='(4/(t+4))**0.75', after='4/(t+4)', momentum_reset=False,
                    penalty_clock_reset=False, tp_projection=False, resume=False),
                resources=dict(max_workers=2, cpu_affinity=[0,1], nice=10, trial_wall_seconds=1200,batch_wall_seconds=2400))
    if phase == 'confirmation':
        plan.update(selected_config_ids=['penalty64-two-phase'], selection={'path':'fixture','sha256':'fixture'})
    return plan, None


def write_json(path,value):
    path.write_text(json.dumps(value,sort_keys=True))


def archive_fixture(tmp_path):
    """A physically coherent synthetic trace, not claimed to be optimizer output."""
    plan, spec = declared()
    c = plan['trials'][0]
    path = tmp_path/'plan_confirmation.json'; write_json(path, plan)
    folder = tmp_path/'confirmation'/c['id']; folder.mkdir(parents=True)
    for name in ('selection_calibration.json', 'audit_calibration.json', 'plan_calibration.json', 'inventory_calibration.json'):
        write_json(tmp_path/name, {'synthetic_fixture_only': True})
    initial = audit.physical.independent_initializer(c['initialization_seed'], 16)
    truth = audit.physical.independent_truth(c['channel'], 16)
    targets = audit.physical.born_values(truth, 16)
    steps = np.arange(0, c['steps']+1, c['metrics_every'])
    schedules = audit.schedule_histories(c)
    beta = schedules['smoothing_parameter'][np.minimum(steps, c['steps']-1)]
    states = [(1-t/c['steps'])*initial+(t/c['steps'])*truth for t in steps]
    metrics = [audit.physical.independent_metrics(x, truth, targets, 1/(2*b), 16) for x, b in zip(states, beta)]
    arrays = dict(active_rows=np.arange(8192), active_observations=targets, initial_factor=initial,
                  truth_factor=truth, frames_final_factor=truth, frames_checkpoint_steps=steps,
                  frames_optimizer_seconds=np.arange(len(steps), dtype=float)*.1,
                  frames_cumulative_data_row_accesses=steps*c['batch_size'])
    arrays.update({'frames_'+k: np.array([m[k] for m in metrics]) for k in audit.COMMON})
    arrays.update({'frames_diagnostic_'+k: v for k, v in schedules.items()})
    arrays.update(frames_diagnostic_checkpoint_smoothing_parameter=beta,
                  frames_diagnostic_checkpoint_tp_weight_per_row_alpha=1/(2*beta),
                  frames_diagnostic_checkpoint_tp_weight_lambda=8192/(2*beta),
                  frames_diagnostic_estimated_gap=np.zeros(c['steps']),
                  frames_diagnostic_exact_smoothed_gap=np.zeros(len(steps)))
    started = dict(trial=c, plan_sha256=audit.digest(path), source_commit=plan['source_commit'],
        source_sha256=plan['source_sha256'], harness_sha256=plan['harness_sha256'],
        parent_pins=plan['parent_pins'], runner_sha256='test-runner', campaign_spec_sha256='test-spec',
        host='elite', execution='local_cpu', cpu_affinity=[0, 1], nice=10,
        threads={k:'1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        split='confirmation', charged_row_budget=c['charged_row_budget'], execution_prerequisites_sha256={
            name: audit.digest(tmp_path/name) for name in ('selection_calibration.json', 'audit_calibration.json', 'plan_calibration.json', 'inventory_calibration.json')})
    final = metrics[-1]; passed, deficit = audit.physical.gate_and_deficit(final)
    passing = [(m['rmse'] <= 1e-6 and m['normalized_tp_violation'] <= .001 and m['process_fidelity'] >= .99) for m in metrics]
    first, persistent = audit.physical.gate_crossings(steps, passing)
    conf = dict(methods=['frames'], rank=1, initialization_basis='gell-mann',
        initialization_seed=c['initialization_seed'], sampling_seed=c['sampling_seed'],
        frames_steps=c['steps'], frames_metrics_frequency=c['metrics_every'], batch_size=c['batch_size'],
        tau=4., frames_penalty_mode='independent_schedule', frames_beta0=1/(2*c['alpha0']),
        frames_initial_tp_weight_per_row_alpha=c['alpha0'], frames_initial_tp_weight_lambda=8192*c['alpha0'])
    for key, scale, offset, exponent, cap in (
        ('frames_smoothing_schedule', 1/(2*c['alpha0']), 1., c['penalty_exponent'], None),
        ('rho_schedule', c['rho_scale'], c['rho_offset'], c['rho_exponent'], 1.),
        ('step_size_schedule', c['step_scale'], c['step_offset'], c['step_exponent'], 1.)):
        conf[key] = dict(type='PowerSchedule', scale=scale, offset=offset, exponent=exponent, cap=cap)
    conf['step_size_schedule'] = dict(type='callable', module='run_polish', qualname='TwoPhaseStep')
    metadata = dict(format='qpt_quiroga_campaign', polish_configuration=c, configuration=conf,
        design=dict(n_qubits=4, d=16, rows=8192, probes=256, outcomes=32, povm_a=1/37, povm_b=1/37),
        observation_mode='noiseless', active_row_count=8192, initialization=dict(basis='gell-mann'),
        data_metadata=dict(channel_seed=c['channel']), source_sha256={},
        polish_backend=dict(name='structured_numpy', precision='complex128', dense_sensing_bank=False,
            resume=False, observed_batch_indices_archived=False,
            repository_runner='paper.experiments.qpt_quiroga_campaign.run_campaign', step_schedule=audit.step_descriptor(c), momentum_reset=False),
        polish_provenance=dict(started, endpoint_gate=passed, gate_deficit=deficit,
            first_recorded_gate_step=first, persistent_recorded_gate_step=persistent),
        methods=dict(frames=dict(final, status='complete', failure_reason=None, completed_steps=c['steps'],
                               cumulative_data_row_accesses=c['charged_row_budget'])))
    metadata.update({k+'_sha256': audit.array_digest(arrays[k]) for k in ('active_rows', 'active_observations', 'initial_factor', 'truth_factor')})
    summary = dict(trial=c, status='complete', gate_pass=passed, deficit=deficit,
        rmse=final['rmse'], normalized_tp=final['normalized_tp_violation'], fidelity=final['process_fidelity'],
        relative_rmse=final['relative_rmse'], outcome_scaled_rmse=32*final['rmse'], optimizer_seconds=float(arrays['frames_optimizer_seconds'][-1]),
        total_seconds=10., peak_rss_mib=120., completed_steps=c['steps'], charged_rows=c['charged_row_budget'],
        charged_row_budget=c['charged_row_budget'], first_recorded_gate_step=first, persistent_recorded_gate_step=persistent)
    write_json(folder/'started.json', started)
    persist(folder, metadata, arrays, summary)
    return plan, path, c, folder, metadata, arrays, summary


def persist(folder, metadata, arrays, summary):
    write_json(folder/'result.json', metadata)
    np.savez(folder/'result.npz', **arrays, metadata_json=json.dumps(metadata, sort_keys=True))
    summary.update(json_sha256=audit.digest(folder/'result.json'), npz_sha256=audit.digest(folder/'result.npz'))
    write_json(folder/'summary.json', summary)


@pytest.mark.parametrize('phase', ['calibration', 'confirmation'])
def test_fixed_plan_cohorts(phase):
    audit.check_plan(declared(phase)[0])


@pytest.mark.parametrize('key,value', [('switch_update',47999), ('post_step_offset',4.-48000),
    ('post_step_scale',8.), ('post_step_exponent',.75), ('step_scale',4.), ('alpha0',5e-5),
    ('penalty_exponent',0.), ('batch_size',128), ('tau',2.), ('steps',352000),
    ('charged_row_budget',11264000), ('sampling_seed',4), ('initialization_seed',4)])
def test_schedule_seed_and_budget_tampering_rejected(key,value):
    p,_=declared();p['trials'][0][key]=value
    with pytest.raises(ValueError):audit.check_plan(p)


def test_phase_inventory_and_no_reset_projection():
    for key in ('momentum_reset','penalty_clock_reset','tp_projection','resume'):
        p,_=declared();p['schedule'][key]=True
        with pytest.raises(ValueError):audit.check_plan(p)
    p,_=declared();p['trials'].pop()
    with pytest.raises(ValueError):audit.check_plan(p)


def test_piecewise_schedule_absolute_clock_and_switch_boundary():
    c=declared()[0]['trials'][0];h=audit.schedule_histories(c)
    for t in (0,1,47998,47999,48000,48001,399999):
        expected=min(1.,4**.75/(t+4)**.75) if t<48000 else 4/(t+4)
        np.testing.assert_allclose(h['step_size'][t],expected,rtol=2e-15)
        np.testing.assert_allclose(h['smoothing_parameter'][t],640000/(t+1)**.25,rtol=2e-15)
        np.testing.assert_allclose(h['momentum_weight'][t],min(1.,4/(t+8)**(2/3)),rtol=2e-15)
    assert h['step_size'][48000]<h['step_size'][47999]/10
    np.testing.assert_allclose(h['tp_weight_lambda'],8192*h['tp_weight_per_row_alpha'])


def test_complete_synthetic_archive_and_tampering(tmp_path):
    p,path,c,folder,m,z,summary=archive_fixture(tmp_path)
    row=audit.check_trial(tmp_path,path,p,c)
    assert row['valid'] and row['gate_pass'] and row['charged_rows']==12800000
    assert not row['prefix_check']['required']
    for key,index in [('step_size',47999),('step_size',48000),('momentum_weight',48000),
                      ('smoothing_parameter',48000),('checkpoint_smoothing_parameter',-1)]:
        field='frames_diagnostic_'+key;old=z[field][index];z[field][index]*=1.01
        persist(folder,m,z,summary)
        with pytest.raises(ValueError):audit.check_trial(tmp_path,path,p,c)
        z[field][index]=old
    m['polish_backend']['step_schedule']['momentum_reset']=True
    persist(folder,m,z,summary)
    with pytest.raises(ValueError,match='descriptor'):audit.check_trial(tmp_path,path,p,c)


def prefix_fixture():
    c=declared()[0]['trials'][0];h=audit.schedule_histories(c)
    z={k:np.zeros((1,)) for k in ('initial_factor','truth_factor','active_rows','active_observations')}
    z['frames_checkpoint_steps']=np.arange(0,400001,4000)
    for key in audit.COMMON:z['frames_'+key]=np.arange(101,dtype=float)
    z.update({'frames_diagnostic_'+k:v for k,v in h.items()})
    prior={k:v.copy() for k,v in z.items()}
    prior['frames_checkpoint_steps']=prior['frames_checkpoint_steps'][:13]
    for key in audit.COMMON:prior['frames_'+key]=prior['frames_'+key][:13]
    for key in h:prior['frames_diagnostic_'+key]=prior['frames_diagnostic_'+key][:48000]
    prior['frames_penalized_objective'][-1]=999
    return z,prior


def test_prefix_exact_metrics_and_schedules_excludes_terminal_objective():
    z,prior=prefix_fixture();result=audit.compare_prefix(z,prior)
    assert result['verified'] and result['compared_checkpoint_count']==13
    assert all(v==0 for v in result['max_abs_metric_differences'].values())
    for key in ('frames_rmse','frames_diagnostic_step_size','initial_factor'):
        old=prior[key][0];prior[key][0]+=1e-12
        with pytest.raises(ValueError):audit.compare_prefix(z,prior)
        prior[key][0]=old


def test_failed_native_run_never_claims_prefix():
    p,_=declared('calibration')
    result=audit.check_prefix({},p['trials'][0],'failed')
    assert result['required'] and not result['verified']


def test_independent_identity_born_and_tp():
    probes,effects=audit.physical.physical_design(16)
    np.testing.assert_allclose(effects.sum(axis=0),np.eye(16),atol=1e-15)
    assert effects.shape==(32,16,16) and np.linalg.eigvalsh(effects).min()>-1e-14
    identity=np.eye(16,dtype=complex).T.reshape(256,1)
    y=audit.physical.born_values(identity,16)
    np.testing.assert_allclose(y.reshape(256,32).sum(axis=1),1.,atol=1e-14)
    m=audit.physical.independent_metrics(identity,identity,y,5e-5/64,16)
    assert m['rmse']==m['tp_violation']==0 and m['process_fidelity']==1


def test_confirmation_requires_two_passing_verified_calibration_cases(monkeypatch,tmp_path):
    plan,_=declared();selection_path=tmp_path/'selection_calibration.json'
    plan['selection']={'path':str(selection_path),'sha256':selection_path.name}
    rows=[dict(id=f'c{c}',channel=c,valid=True,status='complete',gate_pass=True,prefix_check={'verified':True},
               started_sha256='started.json',summary_sha256='summary.json',json_sha256='result.json',npz_sha256='result.npz') for c in (0,1)]
    selection=dict(audit_sha256='audit_calibration.json',calibration_plan_sha256='plan_calibration.json',
                   selected_config_ids=['penalty64-two-phase'],confirmation_status='planned')
    report=dict(phase_complete_and_valid=True,auditor_sha256='audit_polish.py',plan_sha256='plan_calibration.json',
                batch_inventory_sha256='inventory_calibration.json',trials=rows)
    monkeypatch.setattr(audit,'digest',lambda p:Path(p).name)
    monkeypatch.setattr(audit,'read',lambda p:selection if Path(p).name=='selection_calibration.json' else report)
    assert audit.check_confirmation_selection(plan,tmp_path)
    for key,value in [('gate_pass',False),('status','failed'),('prefix_check',{'verified':False})]:
        old=rows[1][key];rows[1][key]=value
        with pytest.raises(ValueError,match='eligibility'):audit.check_confirmation_selection(plan,tmp_path)
        rows[1][key]=old


