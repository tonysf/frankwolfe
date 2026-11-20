"""Seeded recurrence, initialization reuse, accounting and point-aligned logging."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_benchmark
from gw_core import GWProblem, kl_prox, rademacher
from make_instances import random_metric, write_instance
from smoothing_fw import MomentumConfig, initialize_state


def _instance(tmp_path):
    rng = np.random.default_rng(450)
    A, B = random_metric(4, rng), random_metric(4, rng)
    path = tmp_path / 'tiny.npz'
    write_instance(path, A, B, dict(family='momentum_ablation_test', seed=450))
    mass = torch.full((4,), .25, dtype=torch.float64)
    problem = GWProblem(torch.from_numpy(A).double(), torch.from_numpy(B).double(), mass, mass.clone())
    return path, problem


def _args(tmp_path, path, **overrides):
    args = run_benchmark.parser().parse_args([
        '--instance', str(path), '--out', str(tmp_path / 'run'),
        '--method', 'stochastic-kl-momentum', '--budget', '4', '--checkpoints', '1,3,4',
        '--seed', '17', '--init-seed', '9901', '--batch', '128', '--rho0', '.3',
        '--eta', '.05', '--eta-power', '.25', '--inner-tol', '1e-12', '--inner-max', '5000'])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize('initialization', ['product', 'randomized'])
@pytest.mark.parametrize('rho0', [.3, 1.])
def test_four_updates_match_independent_autograd_recurrence_and_point_diagnostics(tmp_path, monkeypatch, initialization, rho0):
    path, problem = _instance(tmp_path)
    args = _args(tmp_path, path, initialization=initialization, rho0=rho0)
    P0, _ = run_benchmark.make_initial_plan(problem, initialization, args.init_seed)
    P, logP = P0.clone(), P0.log()
    reference_generator = torch.Generator().manual_seed(args.seed)
    expected_inputs, expected_momenta, expected_rng_states = [], [], []
    d = None
    for k in range(4):
        probes = rademacher(4, 128, like=P, generator=reference_generator)
        x = P.clone().requires_grad_(True)
        sampled = problem.sketch_loss(x, probes)
        fresh, = torch.autograd.grad(sampled, x)
        rho = rho0 / (k + 1)**.5
        d = fresh if d is None else (1-rho)*d + rho*fresh
        expected_inputs.append(P.clone())
        expected_momenta.append(d.clone())
        expected_rng_states.append(reference_generator.get_state().clone())
        P, logP, info = kl_prox(logP, d, problem.a, problem.b,
                               eta=args.eta/(k+1)**args.eta_power,
                               tol=args.inner_tol, max_iter=args.inner_max)
        assert info['converged']
    # Only advance an artificial clock for an actual completed KL update. This
    # fixes the horizon at four updates without a wall-time-sensitive test.
    clock = [0.]
    monkeypatch.setattr(run_benchmark, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    observed_inputs, observed_momenta, observed_rng_states, prox_gradients = [], [], [], []
    actual_estimate = run_benchmark._kl_momentum_gradient
    def capture_estimate(problem, P, previous, generator, rho, batch):
        value = actual_estimate(problem, P, previous, generator, rho, batch)
        observed_inputs.append(P.clone())
        observed_momenta.append(value.clone())
        observed_rng_states.append(generator.get_state().clone())
        return value
    def capture_prox(logP, G, *arguments, **keywords):
        prox_gradients.append(G.clone())
        result = kl_prox(logP, G, *arguments, **keywords)
        clock[0] += 1.
        return result
    monkeypatch.setattr(run_benchmark, '_kl_momentum_gradient', capture_estimate)
    monkeypatch.setattr(run_benchmark, 'kl_prox', capture_prox)
    result = run_benchmark.run(args)
    assert result['status'] == 'completed', result['error']
    assert len(observed_momenta) == len(prox_gradients) == 4  # no extra initialization draw
    smooth = initialize_state(problem, MomentumConfig(alpha=.75, beta=.5, delta=.25,
        gamma0=1., rho0=rho0, mu0=1/4, oracle='sketch', batch=128), seed=args.seed, P0=P0)
    torch.testing.assert_close(observed_momenta[0], smooth.d, rtol=0, atol=0)
    assert torch.equal(observed_rng_states[0], smooth.generator.get_state())
    for k in range(4):
        torch.testing.assert_close(observed_inputs[k], expected_inputs[k], rtol=0, atol=2e-14)
        torch.testing.assert_close(observed_momenta[k], expected_momenta[k], rtol=2e-13, atol=2e-14)
        torch.testing.assert_close(prox_gradients[k], observed_momenta[k], rtol=0, atol=0)
        assert torch.equal(observed_rng_states[k], expected_rng_states[k])
    with np.load(tmp_path / 'run/final.npz') as z:
        torch.testing.assert_close(torch.from_numpy(z['P_raw']), P, rtol=0, atol=2e-14)
    counts = result['counters']
    assert counts['attempted_updates'] == counts['completed_updates'] == 4
    assert counts['oracle_evaluations'] == counts['stochastic_gradient_calls'] == 4
    assert counts['sketch_probes'] == 512 and counts['initialization_oracle_calls'] == 1
    assert counts['momentum_initializations'] == 1 and counts['momentum_ema_updates'] == 3
    assert counts['kl_projection_calls'] == 4
    assert counts['exact_gradient_calls'] == counts['row_lmo_calls'] == counts['row_projection_calls'] == 0
    assert result['smoothing_config'] is None and result['mu0'] is None and result['m_mu0'] is None
    assert result['momentum_config']['rho0'] == rho0
    assert result['momentum_config']['beta'] == .5
    with (tmp_path / 'run/trace.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    observed_rows = [row for row in rows if row['kind'] != 'missed']
    assert [int(row['iteration']) for row in observed_rows] == [0,1,3,4]
    assert result['diagnostic_counts']['full_gradient_calls'] == 2*len(observed_rows)
    for row in observed_rows:
        iteration = int(row['iteration'])
        point = max(0, iteration-1)
        assert int(row['momentum_point_iteration']) == point
        assert row['momentum_error_matches_current_output'] == str(iteration == 0)
        expected_error = (observed_momenta[point] - problem.gradient(observed_inputs[point])).norm().item()
        assert float(row['momentum_error_frobenius']) == pytest.approx(expected_error, abs=1e-14)
        assert 'true_smoothed_gap' not in row and 'mu' not in row
        if iteration:
            assert float(row['last_update_rho']) == pytest.approx(rho0/iteration**.5)
            assert float(row['last_update_effective_momentum_weight']) == pytest.approx(1. if iteration == 1 else rho0/iteration**.5)


@pytest.mark.parametrize('overrides,message', [
    ({'rho0': None}, 'explicit selected smoothing rho0'),
    ({'rho0': float('nan')}, 'explicit selected smoothing rho0'),
    ({'rho0': 0.}, 'explicit selected smoothing rho0'),
    ({'oracle': 'exact'}, 'frozen sketch oracle'),
    ({'batch': 32}, 'frozen sketch oracle'),
])
def test_new_method_rejects_unmatched_configuration(tmp_path, overrides, message):
    path, _ = _instance(tmp_path)
    with pytest.raises(ValueError, match=message):
        run_benchmark.run(_args(tmp_path, path, **overrides))


def test_nonfinite_initial_oracle_preserves_counts_and_failure(tmp_path, monkeypatch):
    path, _ = _instance(tmp_path)
    monkeypatch.setattr(run_benchmark.GWProblem, 'sketch_gradient',
                        lambda self, P, Z: torch.full_like(P, float('nan')))
    result = run_benchmark.run(_args(tmp_path, path))
    assert result['status'] == 'failed'
    assert 'Nonfinite gradient' in result['error']
    assert result['counters']['oracle_evaluations'] == 1
    assert result['counters']['sketch_probes'] == 128
    assert result['counters']['initialization_oracle_calls'] == 1
    assert result['counters']['completed_updates'] == 0
    assert result['final_float64_validation'] is None
    with np.load(tmp_path / 'run/final.npz') as z:
        assert 'P_rounded' not in z
