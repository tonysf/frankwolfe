"""Predeclared selection is outcome-driven only at declared calibration stages."""
import pytest
from make_plans import specification, trial, select_screen, select_long


def rows(name, deficit, fidelity, passed=False, crossing=None, batch=32, status='complete'):
    return [dict(config_id=name, channel=c, status=status, deficit=deficit,
                 metrics={'process_fidelity': fidelity}, gate_pass=passed,
                 persistent_recorded_gate_step=crossing, charged_rows=12800000,
                 completed_steps=12800000//batch) for c in (0, 1)]


def test_grid_covers_knobs_and_selected_interactions():
    s = specification(); configs = s['configs']
    assert len(configs) == 12 and len({c['id'] for c in configs}) == 12
    for key in ('alpha0', 'penalty_exponent', 'step_scale', 'step_exponent', 'rho_exponent', 'batch_size'):
        assert len({c[key] for c in configs}) > 1
    assert min(c['penalty_exponent'] for c in configs) > 0


@pytest.mark.parametrize('phase', ['pilot', 'screen', 'long', 'confirmation'])
def test_all_trials_match_declared_row_budget(phase):
    spec = specification(); p = spec['phases'][phase]
    for c in spec['configs']:
        t = trial(c, p, p['channels'][0], p['initialization_replicates'][0], p['sampling_replicates'][0])
        assert t['steps']*t['batch_size'] == p['charged_row_budget']
        assert t['metrics_every']*t['batch_size'] == p['checkpoint_charged_rows']
        assert t['steps'] % t['metrics_every'] == 0
        assert t['channel'] not in range(1000, 1020)


def test_deficit_and_fidelity_nominees_are_both_kept():
    chosen, _ = select_screen(rows('a', 2, .98)+rows('b', 3, .999)+rows('incumbent', 1, 1.))
    assert chosen == ['a', 'b']


def test_same_nominee_deduplicates_and_fills():
    chosen, _ = select_screen(rows('a', 2, .999)+rows('b', 3, .99)+rows('c', 4, .98))
    assert chosen == ['a', 'b']


def test_failed_execution_is_not_nominated():
    chosen, _ = select_screen(rows('bad', .1, 1., status='failed')+rows('b', 3, .99)+rows('c', 4, .98))
    assert chosen == ['b', 'c']


def test_selection_uses_worst_channel_not_best():
    r = rows('a', 2, .9)+rows('b', 3, .8)+rows('c', 4, .7)
    r[1]['deficit'] = 100; r[1]['metrics']['process_fidelity'] = .1
    assert select_screen(r)[0] == ['b', 'c']


def test_no_gate_eligible_long_winner():
    assert select_long(rows('a', 2, .999)+rows('incumbent', 3, .9))[0] == []


def test_long_selection_uses_charged_rows_not_updates():
    r = rows('a', .2, .999, True, 50000, 32)+rows('b', .1, .999, True, 20000, 128)
    assert select_long(r)[0] == ['a']


def test_long_requires_both_channels():
    r = rows('a', .2, .999, True, 50000)
    r[1]['gate_pass'] = False; r[1]['persistent_recorded_gate_step'] = None
    assert select_long(r)[0] == []


def test_deterministic_tie_break():
    assert select_screen(rows('z', 2, .9)+rows('a', 2, .9))[0] == ['a', 'z']
