"""Small accounting checks; these are not comparative performance benchmarks."""
import json
from pathlib import Path

import numpy as np
import pytest

from make_instances import random_metric, write_instance
from run_benchmark import CheckpointGrid, METHODS, parser, run


def test_checkpoint_crossing_preserves_actual_time_and_marks_missed_windows():
    grid = CheckpointGrid([1, 3, 10, 30])
    assert grid.consume(.8) == (None, [])
    assert grid.consume(4.2) == (1., [3.])
    assert grid.consume(4.3) == (None, [])
    assert grid.consume(33.) == (10., [30.])
    assert grid.remaining() == []


@pytest.mark.parametrize('method', METHODS)
def test_small_budget_run_preserves_outputs_counts_and_reproducibility_metadata(tmp_path, method):
    rng = np.random.default_rng(1000)
    A = random_metric(16, rng)
    permutation = rng.permutation(16)
    instance = tmp_path / 'toy16.npz'
    write_instance(instance, A, A[np.ix_(permutation, permutation)],
                   dict(family='test_only', seed=1000, noise=0))
    out = tmp_path / method
    args = parser().parse_args([
        '--instance', str(instance), '--method', method, '--out', str(out),
        '--budget', '.03', '--checkpoints', '.01,.03', '--seed', '100',
        '--init-seed', '200', '--inner-tol', '1e-9', '--inner-max', '1000',
        '--eta', '.1', '--batch', '8'])
    if method == 'stochastic-kl-momentum':
        args.batch, args.rho0 = 128, .3
    result = run(args)
    assert result['status'] in ('completed', 'completed_with_inner_failures'), result['error']
    counts = result['counters']
    updates = counts['completed_updates']
    assert updates >= 1
    assert counts['oracle_evaluations'] == updates
    assert counts['attempted_updates'] == updates
    assert result['timing']['solver_seconds'] >= .03
    assert result['timing']['serialization_seconds'] > 0
    assert result['timing']['final_float64_validation_seconds'] > 0
    assert result['initialization_seed'] == 200
    assert result['stochastic_seed'] == 100
    if method.startswith('smoothing-'):
        assert counts['initialization_oracle_calls'] == 1
        assert counts['sketch_probes'] == 8 * updates
    if method in ('full-kl', 'stochastic-kl', 'pg'):
        assert counts['inner_iterations_total'] >= updates
    if method == 'stochastic-kl':
        assert counts['stochastic_gradient_calls'] == updates
        assert counts['sketch_probes'] == 8 * updates
    if method in ('full-kl', 'fw', 'pg'):
        assert counts['exact_gradient_calls'] == updates
    saved = json.loads((out / 'metadata.json').read_text())
    assert saved['counters'] == counts
    assert saved['array_sha256']['A'] == result['instance_manifest']['A_sha256']
    with np.load(out / 'final.npz') as plans:
        assert np.min(plans['P_rounded']) >= 0
        np.testing.assert_allclose(plans['P_rounded'].sum(0), 1 / 16, rtol=0, atol=1e-12)
        np.testing.assert_allclose(plans['P_rounded'].sum(1), 1 / 16, rtol=0, atol=1e-12)
    assert (out / 'source_snapshot' / 'run_benchmark.py').exists()
    import csv
    with (out / 'trace.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    checkpoints = [row for row in rows if row['kind'] == 'checkpoint']
    assert checkpoints
    assert len({row['iteration'] for row in checkpoints}) == len(checkpoints)
    for row in checkpoints:
        assert float(row['actual_solver_seconds']) >= float(row['target_solver_seconds'])
        assert float(row['time_to_this_feasible_output']) >= float(row['actual_solver_seconds'])
        assert float(row['rounded_both_l1']) < 1e-12
    assert result['final_float64_validation']['rounded']['both_l1'] < 1e-12


def _failure_run_args(tmp_path, method):
    rng = np.random.default_rng(1002)
    A = random_metric(16, rng)
    B = random_metric(16, rng)
    instance = tmp_path / 'toy_failure.npz'
    write_instance(instance, A, B, dict(family='test_only_failure_accounting'))
    return parser().parse_args([
        '--instance', str(instance), '--method', method, '--out', str(tmp_path / 'run'),
        '--budget', '.03', '--seed', '100', '--init-seed', '200', '--gap-max-n', '0',
        '--inner-max', '1', '--inner-tol', '1e-30', '--eta', '10'])


def test_inner_failures_are_aggregated_even_between_checkpoints(tmp_path):
    result = run(_failure_run_args(tmp_path, 'full-kl'))
    counts = result['counters']
    assert result['status'] == 'completed_with_inner_failures'
    assert counts['inner_iterations_total'] == counts['completed_updates']
    assert counts['inner_stopping_failures'] > 0
    events = json.loads((tmp_path / 'run' / 'failures.json').read_text())
    assert sum(event['type'] == 'inner_tolerance_not_attained' for event in events) == counts['inner_stopping_failures']


def test_nonfinite_gradient_rejects_run_and_preserves_failure_evidence(tmp_path, monkeypatch):
    import torch
    import run_benchmark
    args = _failure_run_args(tmp_path, 'fw')
    monkeypatch.setattr(run_benchmark.GWProblem, 'gradient',
                        lambda self, P: torch.full_like(P, float('nan')))
    result = run(args)
    assert result['status'] == 'failed'
    assert 'Nonfinite gradient rejected' in result['error']
    assert result['counters']['attempted_updates'] == 1
    assert result['counters']['completed_updates'] == 0
    assert result['counters']['oracle_evaluations'] == 1
    assert result['final_float64_validation'] is None
    saved = json.loads((tmp_path / 'run' / 'metadata.json').read_text())
    assert saved['status'] == 'failed'
    assert json.loads((tmp_path / 'run' / 'failures.json').read_text())[-1]['type'] == 'run_rejected'


def test_unsuffixed_cuda_resolves_to_logical_visible_device_zero():
    from run_benchmark import resolve_device
    assert str(resolve_device('cuda')) == 'cuda:0'
    assert str(resolve_device('cuda:1')) == 'cuda:1'
    assert str(resolve_device('cpu')) == 'cpu'
