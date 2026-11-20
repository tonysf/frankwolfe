"""Joint campaign cohort, development selection, and operator-accounting checks."""

from copy import deepcopy
from collections import Counter
import json

import pytest

from paper.experiments.stochastic_ct import joint_lbfgsb_campaign as campaign


def cohort_inputs():
    slices = [f'slice{i:05d}' for i in range(14)]
    development = ['slice99998', 'slice99999']
    cohort = dict(evaluation_slices=slices, development_slices=development,
                  evaluation_slice_groups={sid: str(i % 6) for i, sid in enumerate(slices)})
    inputs = {sid: dict(data_file=f'/data/{sid}.npz', data_sha256='data',
        calibration_file=f'/cal/{sid}.npz', calibration_sha256='cal',
        calibration_seed=201 if sid in slices else 110) for sid in slices + development}
    return cohort, inputs


def row(iteration, calls, kind='accepted_update'):
    r = dict(iteration=iteration, observation_type=kind, forward_calls=calls,
        backward_calls=calls, forward_views=360*calls, backward_views=360*calls,
        gradient_calls=calls, oracle_calls=calls, normal_operator_calls=0,
        rhs_backward_calls=0, rhs_backward_views=0, completed_blocks=0, outer_sweeps=0,
        raw_loss=1/(iteration+1), feasible_loss=1/(iteration+1), validation_loss=1.,
        d_distance=0., factor_u_sha256=f'U{iteration}', factor_v_sha256=f'V{iteration}')
    r['last_step'] = dict(r)
    return r


def trace():
    return [row(0, 0, 'initial'), row(1, 2), row(1, 3, 'budget_stop')]


def test_evaluation_is_deterministic_and_balances_each_view_count():
    cohort, inputs = cohort_inputs()
    plan = campaign.make_plan(cohort, inputs, {'joint_lbfgsb': dict(campaign.FIXED_PARAMETERS, memory=10)})
    assert len(plan['runs']) == 28
    assert len({c['run_id'] for c in plan['runs']}) == 28
    assert Counter(plan['assignments'].values()) == {cpu: 7 for cpu in campaign.CPUS}
    for cpu in campaign.CPUS:
        counts = Counter(c['n_train'] for c in plan['runs'] if plan['assignments'][c['run_id']] == cpu)
        assert sorted(counts.values()) == [3, 4]
    assert all(c['calibration_seed'] == 201 and c['view_budget'] == 69120 for c in plan['runs'])
    assert all('seed' not in c and 'caps' not in c and 'inner_maxiter' not in c['parameters'] for c in plan['runs'])


def test_memory_selection_uses_only_development_raw_loss_and_breaks_ties():
    cohort, inputs = cohort_inputs()
    plan = campaign.make_development_plan(cohort, inputs)
    assert len(plan['runs']) == 12
    assert {c['parameters']['memory'] for c in plan['runs']} == {5, 10, 20}
    assert {c['slice_id'] for c in plan['runs']}.isdisjoint(cohort['evaluation_slices'])
    cases = [dict(status='completed', run_id=c['run_id'], n_train=c['n_train'],
        relative_loss=.2 if c['parameters']['memory'] in (5, 10) else .3,
        validation_loss=0 if c['parameters']['memory'] == 20 else 10) for c in plan['runs']]
    assert campaign.select_parameters(cases, plan)['selected_memory'] == 5
    with pytest.raises(ValueError, match='all development'):
        campaign.select_parameters(cases[:-1], plan)
    cases[0]['status'] = 'failed'
    with pytest.raises(ValueError, match='failed'):
        campaign.select_parameters(cases, plan)


def test_unaccepted_trial_work_is_retained_without_fabricated_iterate():
    cfg = dict(method='joint_lbfgsb', n_train=360, view_budget=1080)
    assert campaign.validate_trace(trace(), cfg)
    altered = trace()
    altered[-1]['factor_u_sha256'] = 'trial'
    with pytest.raises(ValueError, match='changed factors'):
        campaign.validate_trace(altered, cfg)
    with pytest.raises(ValueError, match='exceeded'):
        campaign.validate_trace(trace(), dict(cfg, view_budget=720))


def test_joint_updates_cannot_be_mislabeled_as_blocks_or_omit_work():
    cfg = dict(method='joint_lbfgsb', n_train=360, view_budget=1080)
    altered = trace()
    altered[1]['oracle_calls'] = 1
    with pytest.raises(ValueError):
        campaign.validate_trace(altered, cfg)
    altered = trace()
    for r in altered[1:]:
        r['completed_blocks'] = r['last_step']['completed_blocks'] = 1
    with pytest.raises(ValueError, match='not alternating'):
        campaign.validate_trace(altered, cfg)


def fixture(root, termination='budget_stop', reason=None):
    cfg = dict(run_id='fixture', method='joint_lbfgsb', slice_id='slice00001', n_train=360,
        calibration_seed=201, view_budget=1080, data_sha256='data', calibration_sha256='cal',
        parameters=dict(campaign.FIXED_PARAMETERS, memory=10))
    entry = dict(initial_factor_sha256={'U': 'U0', 'V': 'V0'}, historical_reference_caps=[1, 2], initial_factor_frobenius_norm=1.)
    protocol = dict(numerical_sources={'ct_problem.py': 'source'}, inputs={cfg['slice_id']: entry})
    campaign.write(root / 'protocol.json', protocol)
    meta = dict(config=cfg, cpu=0, source_sha256=protocol['numerical_sources'], geometry='nonnegative',
        numpy='2.5.4', scipy='1.18.1', spectral_caps_are_constraints=False, deterministic=True,
        data_sha256='data', calibration_sha256='cal', initial_u_sha256='U0', initial_v_sha256='V0',
        reference_spectral_caps=[1, 2], protocol_sha256=campaign.digest(root/'protocol.json'),
        optimizer=cfg['parameters'], train_view_ids=list(range(0,3600,10)), validation_view_ids=list(range(9,3600,10)))
    values = trace()
    if termination == 'converged':
        values[-1] = row(1, 2, 'converged')
        values[-1]['last_step'].update(stop_reason=reason, native_task=[4, 401 if reason == 'kkt' else 402], solver_loss=.5, kkt_inf_norm=.001, kkt_target=.002)
    result = dict(status='completed', termination=termination, metadata=meta, initial=values[0],
        final=values[-1], iteration=1, observations=len(values))
    folder = root/'runs'/cfg['run_id']
    for name, value in (('metadata.json',meta),('trace.json',values),('result.json',result)):
        campaign.write(folder/name,value)
    return cfg, dict(groups={cfg['slice_id']:'group'}), protocol, values


def test_validated_result_preserves_shared_initialization_and_budget(tmp_path):
    cfg, plan, protocol, values = fixture(tmp_path)
    result = campaign.validate_result(cfg, 0, plan, protocol, tmp_path)
    assert result['endpoint'] == values[-1]
    assert result['forward_view_slack'] == result['backward_view_slack'] == 0
    assert result['initial_factor_sha256'] == {'U':'U0','V':'V0'}


@pytest.mark.parametrize('reason',['kkt','relative_function_decrease'])
def test_native_convergence_is_explicit_and_certified(tmp_path, reason):
    cfg, plan, protocol, values = fixture(tmp_path,'converged',reason)
    result = campaign.validate_result(cfg, 0, plan, protocol, tmp_path)
    assert result['termination'] == 'converged'
    values[-1]['last_step']['native_task'] = [4,999]
    folder = tmp_path/'runs'/cfg['run_id']
    saved=campaign.read(folder/'result.json'); saved['final']=values[-1]
    campaign.write(folder/'trace.json',values); campaign.write(folder/'result.json',saved)
    with pytest.raises(ValueError,match='stopping code'):
        campaign.validate_result(cfg,0,plan,protocol,tmp_path)


def test_tampered_initialization_and_wrong_domain_are_rejected(tmp_path):
    cfg,plan,protocol,_=fixture(tmp_path)
    folder=tmp_path/'runs'/cfg['run_id']; meta=campaign.read(folder/'metadata.json')
    meta['spectral_caps_are_constraints']=True; campaign.write(folder/'metadata.json',meta)
    with pytest.raises(ValueError,match='domain'):
        campaign.validate_result(cfg,0,plan,protocol,tmp_path)
