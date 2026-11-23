"""Check diagnostic point alignment and that logging does not advance the solver."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_benchmark
from make_instances import random_metric, write_instance
from smoothing_fw import initialize_state, step


@pytest.mark.parametrize('method,oracle', [
    ('smoothing-fw', 'sketch'), ('smoothing-fw', 'exact'),
    ('smoothing-row-projection', 'sketch')])
@pytest.mark.parametrize('initialization', ['product', 'randomized'])
def test_actual_momentum_is_compared_with_gradient_at_its_own_input(
        tmp_path, monkeypatch, method, oracle, initialization):
    rng = np.random.default_rng(3901)
    A, B = random_metric(4, rng), random_metric(4, rng)
    instance = tmp_path / 'toy.npz'
    write_instance(instance, A, B, dict(family='diagnostic_logging_test'))
    args = run_benchmark.parser().parse_args([
        '--instance', str(instance), '--out', str(tmp_path / 'run'),
        '--method', method, '--oracle', oracle, '--initialization', initialization,
        '--budget', '4', '--checkpoints', '1,3,4', '--seed', '17', '--init-seed', '99',
        '--rho0', '.3', '--batch', '8', '--eta', '.005', '--eta-power', '.25',
        '--gap-max-n', '0'])
    clock, observed, initialized = [0.], [], []
    monkeypatch.setattr(run_benchmark, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    def capture_initial(problem, config, **kwargs):
        state = initialize_state(problem, config, **kwargs)
        initialized.append((problem, config, state.P.clone(), state.d.clone(),
                            state.generator.get_state().clone()))
        return state
    def capture_step(problem, state):
        info = step(problem, state)
        observed.append((info.P_before.clone(), info.d_used.clone(), state.P.clone(),
                         state.generator.get_state().clone()))
        clock[0] += 1.
        return info
    monkeypatch.setattr(run_benchmark, 'initialize_state', capture_initial)
    monkeypatch.setattr(run_benchmark, 'step', capture_step)
    result = run_benchmark.run(args)
    assert result['status'] == 'completed', result['error']
    assert result['counters']['completed_updates'] == 4
    assert result['counters']['oracle_evaluations'] == 4
    assert result['counters']['initialization_oracle_calls'] == 1
    assert result['counters']['sketch_probes'] == (32 if oracle == 'sketch' else 0)
    problem, config, P0, d0, rng0 = initialized[0]
    # Verify every recorded random state and endpoint against a solver-only run.
    replay = initialize_state(problem, config, seed=args.seed, P0=P0)
    assert torch.equal(replay.generator.get_state(), rng0)
    for before, d, after, state_rng in observed:
        info = step(problem, replay)
        assert torch.equal(info.P_before, before)
        assert torch.equal(info.d_used, d)
        assert torch.equal(replay.P, after)
        assert torch.equal(replay.generator.get_state(), state_rng)
    with np.load(tmp_path / 'run/final.npz') as z:
        assert np.array_equal(z['P_raw'], replay.P.numpy())
    with (tmp_path / 'run/trace.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['kind'] != 'missed']
    assert [int(row['iteration']) for row in rows] == [0, 1, 3, 4]
    assert result['diagnostic_counts']['full_gradient_calls'] == 7
    distinct_error_found = False
    for row in rows:
        K = int(row['iteration'])
        prior, d, current = (P0, d0, P0) if K == 0 else observed[K-1][:3]
        x, y = prior.clone().requires_grad_(True), current.clone().requires_grad_(True)
        Gx, = torch.autograd.grad(problem.loss(x), x)
        Gy, = torch.autograd.grad(problem.loss(y), y)
        expected = (d-Gx).norm().item()
        stale = (d-Gy).norm().item()
        assert int(row['momentum_point_iteration']) == max(0, K-1)
        assert int(row['stored_momentum_iteration']) == max(0, K-1)
        assert int(row['tracking_error_point_iteration']) == K
        assert row['momentum_error_matches_current_output'] == str(K == 0)
        assert float(row['momentum_error_frobenius']) == pytest.approx(expected, abs=2e-14)
        assert float(row['stored_momentum_current_output_tracking_error_frobenius']) == pytest.approx(stale, abs=2e-14)
        distinct_error_found |= abs(expected-stale) > 1e-10
    assert distinct_error_found  # the test can detect an accidental point swap
