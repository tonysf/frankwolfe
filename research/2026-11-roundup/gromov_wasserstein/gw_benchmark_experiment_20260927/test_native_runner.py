"""Tiny CPU native-runner accounting checks, not performance comparisons."""
import csv
import json

import numpy as np
import pytest

from make_instances import random_metric, write_instance
from run_native_sampled import parser, run


def _args(tmp_path, *, dtype='float64', budget='.03', max_iter='1000000', zero=False):
    rng = np.random.default_rng(1001)
    A = np.zeros((16, 16), dtype=np.float32) if zero else random_metric(16, rng)
    B = np.zeros_like(A) if zero else random_metric(16, rng)
    instance = tmp_path / 'toy16.npz'
    write_instance(instance, A, B, dict(family='native_runner_test_only', seed=1001))
    return parser().parse_args([
        '--instance', str(instance), '--out', str(tmp_path / 'run'), '--budget', budget,
        '--batch', '8', '--epsilon', '.1', '--seed', '100', '--init-seed', '200',
        '--dtype', dtype, '--max-iter', max_iter])


@pytest.mark.parametrize('dtype,budget', [('float64', '.03'), ('float32', '.1')])
def test_native_single_trajectory_outputs_and_instrumentation(tmp_path, dtype, budget):
    result = run(_args(tmp_path, dtype=dtype, budget=budget))
    assert result['status'] in ('completed', 'completed_with_warnings'), result['error']
    report = result['instrumentation']
    assert report['completed_iterations'] >= 1
    assert report['callback_calls'] == report['completed_iterations']
    assert result['counters']['inner_iterations'] is None
    assert result['counters']['inner_stopping_failures'] is None
    assert result['stochastic_seed'] == 100 and result['initialization_seed'] == 200
    assert report['time_budget_reached'] or report['native_early_stop'] or report['max_iter_exhausted']
    out = tmp_path / 'run'
    with np.load(out / 'final.npz') as plans:
        tolerance = 1e-6 if dtype == 'float32' else 1e-12
        np.testing.assert_allclose(plans['P_rounded'].sum(0), 1 / 16, atol=tolerance, rtol=0)
        np.testing.assert_allclose(plans['P_rounded'].sum(1), 1 / 16, atol=tolerance, rtol=0)
        assert plans['P_rounded'].min() >= 0
    archived = json.loads((out / 'native_source' / 'manifest.json').read_text())
    assert archived['pot_version'] == result['pot_version']
    assert (out / 'native_source' / 'native_function.py').exists()
    assert (out / 'native_source' / 'instrumented_function.py').exists()
    assert (out / 'source_snapshot' / 'run_native_sampled.py').exists()
    saved = json.loads((out / 'metadata.json').read_text())
    assert saved['final_float64_validation']['dtype'] == 'float64'
    assert saved['timing']['serialization_seconds'] > 0
    assert saved['timing']['solver_seconds'] == report['solver_seconds']
    with (out / 'trace.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['kind'] == 'checkpoint':
            assert float(row['actual_solver_seconds']) >= float(row['target_solver_seconds'])
            assert float(row['time_to_this_feasible_output']) >= float(row['actual_solver_seconds'])


def test_native_iteration_limit_preserves_early_final_output_and_missing_windows(tmp_path):
    result = run(_args(tmp_path, budget='.1', max_iter='1'))
    assert result['status'] in ('completed', 'completed_with_warnings'), result['error']
    report = result['instrumentation']
    assert report['completed_iterations'] == 1
    assert report['max_iter_exhausted'] is True
    with (tmp_path / 'run' / 'trace.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if report['solver_seconds'] < .1:
        assert any(row['kind'] == 'missed' and float(row['target_solver_seconds']) == .1 for row in rows)
        final = [row for row in rows if row['kind'] == 'final'][-1]
        assert float(final['actual_solver_seconds']) == report['solver_seconds']
    assert result['final_float64_validation']['rounded']['both_l1'] < 1e-12


def test_native_zero_cost_skips_are_counted_without_false_updates(tmp_path):
    result = run(_args(tmp_path, zero=True))
    assert result['status'] == 'completed', result['error']
    assert result['instrumentation']['time_budget_reached']
    assert result['counters']['completed_iterations'] >= 1
    assert result['counters']['completed_updates'] == 0
    assert result['final_float64_validation']['rounded_loss'] == 0
