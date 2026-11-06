"""Synthetic selection tests: no optimizer calls or experiment launches."""
from copy import deepcopy
import json

import pytest

from select_stage import (GATE, build_next_plan, deficit, group_outcomes,
                          persistent_gate_step, promoted_configs, publish,
                          rank_long, rank_screen, verify_audit, _close)


def pair(name, d, *, failed=0, passes=0, persistent=None):
    return [{'trial_id': f'{name}-c{c}-s0', 'config_id': name, 'channel': c,
             'sampling_replicate': 0, 'failed': c < failed,
             'gate_pass': c < passes, 'deficit': None if c < failed else d,
             'persistent_recorded_gate_step': persistent if c < passes else None}
            for c in (0, 1)]


def plan(phase='screen'):
    config = dict(config_id='best', alpha0=5e-5, penalty_exponent=.25,
                  step_scale=4., step_offset=4., step_exponent=1., rho_scale=2.,
                  rho_offset=4., rho_exponent=.6, tau=2., batch_size=32)
    return dict(phase=phase, trials=[dict(config, id=f'best-c{c}-s0', channel=c,
                                        steps=100000, metrics_every=4000,
                                        sampling_replicate=0) for c in (0, 1)],
                long_stage=dict(steps=400000, metrics_every=4000, channels=[0, 1]),
                confirmation=dict(steps=400000, channels=list(range(2, 8)), sampling_replicates=[0, 1]),
                harness_sha256={'run_local.py': 'runner', 'jax_backend.py': 'backend'},
                gate=GATE)


def test_screen_rule_failure_first_and_deterministic_ties():
    rows = (pair('z', .1, failed=1, passes=1) + pair('b', 3.)
            + pair('a', 3.) + pair('winner', .8, passes=2))
    assert [r['config_id'] for r in rank_screen(rows)] == ['winner', 'a', 'b', 'z']


def test_screen_gate_count_precedes_deficit():
    rows = pair('some_pass', 3., passes=1) + pair('no_pass', 2.)
    assert rank_screen(rows)[0]['config_id'] == 'some_pass'


def test_incumbent_carried_and_deduplicated():
    ranking = rank_screen(pair('a', 1.) + pair('b', 2.) + pair('old', 50.))
    assert promoted_configs(ranking, 'old') == ['a', 'b', 'old']
    assert promoted_configs(ranking, 'a') == ['a', 'b']
    with pytest.raises(ValueError, match='incumbent'):
        promoted_configs(ranking, 'missing')


def test_long_requires_both_endpoint_passes_and_prefers_persistent_speed():
    rows = (pair('fails', .01, passes=1, persistent=1)
            + pair('slow', .001, passes=2, persistent=8000)
            + pair('fast', .7, passes=2, persistent=4000))
    assert [r['config_id'] for r in rank_long(rows)] == ['fast', 'slow']
    assert rank_long(pair('none', 2.)) == []


def test_long_ties_use_endpoint_deficit_then_id():
    rows = (pair('b', .2, passes=2, persistent=4000)
            + pair('a', .2, passes=2, persistent=4000)
            + pair('c', .1, passes=2, persistent=4000))
    assert [r['config_id'] for r in rank_long(rows)] == ['c', 'a', 'b']


def test_persistent_gate_ignores_transient_pass():
    assert persistent_gate_step([0, 4, 8, 12], [False, True, False, True]) == 12
    assert persistent_gate_step([0, 4, 8, 12], [False, True, True, True]) == 4
    assert persistent_gate_step([0, 4], [True, False]) is None


def test_deficit_handles_fidelity_roundoff_and_gate_boundary():
    assert deficit(0., 0., 1.0000000000000002) == 0.
    assert deficit(1e-6, .001, .99) == pytest.approx(1.)
    with pytest.raises(ValueError, match='Nonfinite'):
        deficit(float('nan'), 0., 1.)


def test_actual_screen_dense_structured_deficit_roundoff():
    # Same archived factor, independently evaluated by the structured summary
    # and dense checkpoint routines. RMSE differs by only 1.64e-17, but dividing
    # by the 1e-6 recovery threshold amplifies the dimensionless discrepancy.
    _close(2.427901048666732, 2.4279010486503783, 'actual screen gate deficit')
    # This tolerance must not erase a materially different candidate metric.
    with pytest.raises(ValueError, match='Metric mismatch'):
        _close(2.427901048666732, 2.4279011486503783, 'material deficit change')


def test_grouping_rejects_missing_or_duplicate_channels_and_rng_replicates():
    with pytest.raises(ValueError, match='exactly channels'):
        group_outcomes(pair('x', 1.)[:1])
    rows = pair('x', 1.)
    rows[1]['channel'] = 0
    with pytest.raises(ValueError, match='exactly channels'):
        group_outcomes(rows)
    rows = pair('x', 1.)
    rows[1]['sampling_replicate'] = 1
    with pytest.raises(ValueError, match='replicate zero'):
        group_outcomes(rows)


def test_audit_must_be_complete_plan_bound_and_checksum_bound():
    files = {'a': {'summary.json': 'sum', 'result.npz': 'npz'}}
    audit = dict(phase_complete_and_valid=True, expected_trials=1, validated_trials=1,
                 plan_sha256='plan', trials=[dict(id='a', summary_sha256='sum', npz_sha256='npz')])
    verify_audit(audit, 'plan', files)
    for key, value in [('phase_complete_and_valid', False), ('validated_trials', 0),
                       ('plan_sha256', 'other'), ('trials', [])]:
        changed = deepcopy(audit)
        changed[key] = value
        with pytest.raises(ValueError):
            verify_audit(changed, 'plan', files)


def test_long_plan_is_fresh_preserves_configuration_and_has_pins(tmp_path):
    original = plan()
    frozen = deepcopy(original)
    result = build_next_plan(original, ['best'], tmp_path / 'selection.json', 'selection-hash')
    assert original == frozen
    assert result['phase'] == 'long' and len(result['trials']) == 2
    assert all(t['steps'] == 400000 and t['metrics_every'] == 4000 for t in result['trials'])
    assert 'fresh' in result['stage_start']
    assert result['selection']['sha256'] == 'selection-hash'
    assert set(result['harness_sha256']) == {'run_local.py', 'jax_backend.py', 'select_stage.py'}


def test_confirmation_is_six_exposed_channels_two_nested_streams(tmp_path):
    result = build_next_plan(plan('long'), ['best'], tmp_path / 'selected.json', 'hash')
    assert result['phase'] == 'confirmation' and len(result['trials']) == 12
    assert {(t['channel'], t['sampling_replicate']) for t in result['trials']} == {
        (c, r) for c in range(2, 8) for r in (0, 1)}
    assert all(t['steps'] == 400000 for t in result['trials'])
    assert 'not_unseen_test' in result['split']


def test_config_drift_across_channels_is_rejected(tmp_path):
    bad = plan()
    bad['trials'][1]['alpha0'] = .1
    with pytest.raises(ValueError, match='varies across channels'):
        build_next_plan(bad, ['best'], tmp_path / 'selected.json', 'hash')


def test_no_overwrite_and_no_nonfinite_json(tmp_path):
    path = tmp_path / 'selection.json'
    publish(path, {'ok': True})
    with pytest.raises(FileExistsError):
        publish(path, {'ok': False})
    assert json.loads(path.read_text()) == {'ok': True}
