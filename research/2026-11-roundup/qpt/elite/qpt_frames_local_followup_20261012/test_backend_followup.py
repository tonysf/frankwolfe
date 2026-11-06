"""Small local parity/physics checks; no dense n=3/4 measurement banks."""
import json
import math
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/tony/frankwolfe')

import numpy as np
import pytest

import backend_followup as backend
from paper.experiments.qpt_quiroga_campaign import COMMON_METRICS, run_frames
from paper.experiments.qpt_quiroga_frames import run_quiroga_sensing_stochastic_frames
from paper.experiments.qpt_quiroga_sensing import (
    QuirogaSensingData, generate_quiroga_sensing_data, trace_preserving_residual,
    trace_preserving_loss_and_gradient,
)
from paper.experiments.quantum_process_tomography import pack_factor


def fixture(n, channel=0, init_replicate=1, sampling_replicate=1, steps=100):
    config = dict(backend.FROZEN_PARAMETERS, n_qubits=n, channel=channel,
                  initialization_replicate=init_replicate,
                  initialization_seed=200000 + 10 * channel + init_replicate,
                  sampling_replicate=sampling_replicate,
                  sampling_seed=300000 + 100 * channel + sampling_replicate,
                  steps=steps, metrics_every=max(1, steps // 10), tau=math.sqrt(2**n))
    data = generate_quiroga_sensing_data(n, channel_seed=channel, observation_mode='noiseless')
    return data, backend.expected_initial_factor(data, config['initialization_seed']), config


def independent_born(data, factor):
    d = data.d
    states = [np.eye(d, dtype=np.complex128)[k] for k in range(d)]
    for phase in (1., 1j):
        for j in range(d):
            for k in range(j + 1, d):
                state = np.zeros(d, dtype=np.complex128)
                state[j], state[k] = 1 / math.sqrt(2), phase / math.sqrt(2)
                states.append(state)
    states = np.asarray(states)
    eye = np.eye(d, dtype=np.complex128)
    a, b = data.design.povm_a, data.design.povm_b
    elements = [a * np.outer(eye[0], eye[0])]
    for phase in (1., 1j):
        for k in range(1, d):
            element = b * eye.copy()
            element[0, k] += b * phase
            element[k, 0] += b * np.conj(phase)
            elements.append(element)
    elements.append(eye - sum(elements))
    kraus = factor[:, 0].reshape(d, d).T
    outputs = states @ kraus.T
    return np.einsum('pi,kij,pj->pk', outputs.conj(), np.asarray(elements), outputs).real.reshape(-1)


@pytest.mark.parametrize('channel,init_replicate', [(0, 1), (1, 2)])
@pytest.mark.parametrize('steps', [100, 1000])
def test_n2_new_initializer_delegation_parity(channel, init_replicate, steps):
    data, factor0, config = fixture(2, channel, init_replicate, 1, steps)
    started = perf_counter()
    metadata, payload, actual = backend.run(data, factor0, config)
    seconds = perf_counter() - started
    assert actual.status == 'complete', actual.failure_reason
    reference = run_frames(
        data, factor0, np.arange(data.m, dtype=np.int64), data.all_observations(),
        n_steps=steps, metrics_frequency=config['metrics_every'], rank=1,
        tau=config['tau'], batch_size=32, sampling_seed=config['sampling_seed'],
        tp_weight=1., show_progress=False, **backend.schedules(config),
    )
    np.testing.assert_allclose(actual.final_factor, reference.final_factor, rtol=1e-8, atol=1e-10)
    for name in COMMON_METRICS:
        np.testing.assert_allclose(getattr(actual, name), getattr(reference, name), rtol=1e-8, atol=1e-11)
    expected_batches = np.random.default_rng(config['sampling_seed']).integers(0, data.m, size=(steps, 32))
    np.testing.assert_array_equal(payload['frames_diagnostic_batch_indices'], expected_batches)
    np.testing.assert_array_equal(payload['initial_factor'], factor0)
    assert metadata['followup_configuration'] == config
    assert metadata['followup_backend']['frozen_adapter_sha256'] == backend.FROZEN_N2_ADAPTER_SHA256
    assert metadata['followup_backend']['observed_batch_indices_archived']
    assert np.all(np.isnan(actual.optimizer_seconds[1:-1]))
    print(json.dumps({'test': 'n2_new_initializer_parity', 'channel': channel,
                      'initialization_replicate': init_replicate, 'steps': steps,
                      'adapter_seconds': seconds,
                      'max_factor_difference': float(np.max(np.abs(actual.final_factor-reference.final_factor)))}))


@pytest.mark.parametrize('n', [3, 4])
@pytest.mark.parametrize('channel', [0, 1])
def test_higher_dimension_structured_parity_and_direct_physics(n, channel, monkeypatch):
    data, factor0, config = fixture(n, channel, 1, 1, 100)
    def forbidden_dense(*args, **kwargs):
        raise AssertionError('Higher-dimensional dense bridge is forbidden in this test.')
    monkeypatch.setattr(QuirogaSensingData, 'to_dense_qpt_data', forbidden_dense)
    started = perf_counter()
    metadata, payload, actual = backend.run(data, factor0, config)
    seconds = perf_counter() - started
    assert actual.status == 'complete', actual.failure_reason
    reference = run_quiroga_sensing_stochastic_frames(
        data, n_steps=config['steps'], rank=1, tau=config['tau'], batch_size=32,
        sampling_seed=config['sampling_seed'], x0=pack_factor(factor0),
        metrics_frequency=config['metrics_every'], show_progress=False, **backend.schedules(config),
    )
    np.testing.assert_array_equal(actual.final_factor, reference.final_factor)
    for actual_name, reference_name in [('measurement_loss_mean', 'measurement_loss'),
                                        ('tp_violation', 'tp_violation'),
                                        ('process_fidelity', 'process_fidelity')]:
        np.testing.assert_array_equal(getattr(actual, actual_name), getattr(reference, reference_name))
    for diagnostic, reference_name in [('exact_smoothed_gap', 'exact_smoothed_gap'),
                                        ('estimated_gap', 'estimated_gaps'),
                                        ('momentum_weight', 'momentum_weights'),
                                        ('smoothing_parameter', 'smoothing_parameters'),
                                        ('step_size', 'step_sizes'),
                                        ('checkpoint_smoothing_parameter', 'checkpoint_smoothing_parameters')]:
        np.testing.assert_array_equal(actual.diagnostics[diagnostic], getattr(reference, reference_name))
    np.testing.assert_array_equal(payload['frames_cumulative_data_row_accesses'], actual.checkpoint_steps * 32)
    assert metadata['configuration']['initialization_seed'] == config['initialization_seed']
    assert metadata['configuration']['sampling_seed'] == config['sampling_seed']
    assert metadata['followup_configuration'] == config
    assert not metadata['followup_backend']['observed_batch_indices_archived']
    assert 'frames_diagnostic_batch_indices' not in payload
    assert np.all(np.isfinite(actual.optimizer_seconds))
    # Independent direct Born probabilities, Kraus TP and normalized overlap.
    true_born = independent_born(data, data.truth_factor)
    np.testing.assert_allclose(true_born, payload['active_observations'], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(true_born.reshape(data.d**2, 2*data.d).sum(axis=1), 1., atol=1e-14)
    predicted = independent_born(data, actual.final_factor)
    rmse = np.sqrt(np.mean((predicted - true_born)**2))
    kraus = actual.final_factor[:, 0].reshape(data.d, data.d).T
    tp = np.linalg.norm(kraus.conj().T @ kraus - np.eye(data.d)) / np.sqrt(data.d)
    fidelity = abs(np.vdot(data.truth_factor, actual.final_factor))**2 / (
        np.vdot(data.truth_factor, data.truth_factor).real * np.vdot(actual.final_factor, actual.final_factor).real)
    np.testing.assert_allclose(actual.rmse[-1], rmse, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.normalized_tp_violation[-1], tp, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual.process_fidelity[-1], fidelity, rtol=1e-12, atol=1e-14)
    # Bounded finite-difference checks without any full sensing matrix bank.
    rng = np.random.default_rng(123 + n + channel)
    direction = rng.normal(size=factor0.shape) + 1j*rng.normal(size=factor0.shape)
    direction /= np.linalg.norm(direction)
    rows = rng.integers(0, data.m, size=13)
    _, gradient = data.design.loss_and_gradient(factor0, rows, true_born[rows], reduction='mean')
    epsilon = 1e-6
    loss_plus = .5*np.mean((independent_born(data, factor0+epsilon*direction)[rows] - true_born[rows])**2)
    loss_minus = .5*np.mean((independent_born(data, factor0-epsilon*direction)[rows] - true_born[rows])**2)
    np.testing.assert_allclose((loss_plus-loss_minus)/(2*epsilon), np.vdot(gradient,direction).real,
                               rtol=1e-5, atol=1e-10)
    _, tp_gradient = trace_preserving_loss_and_gradient(factor0)
    tp_plus = .5*np.linalg.norm(trace_preserving_residual(factor0+epsilon*direction))**2
    tp_minus = .5*np.linalg.norm(trace_preserving_residual(factor0-epsilon*direction))**2
    np.testing.assert_allclose((tp_plus-tp_minus)/(2*epsilon), np.vdot(tp_gradient,direction).real,
                               rtol=1e-6, atol=1e-8)
    json.dumps(metadata, allow_nan=False)
    print(json.dumps({'test': 'higher_dimension_structured_and_direct_physics',
                      'n_qubits': n, 'channel': channel, 'steps': 100,
                      'adapter_seconds': seconds, 'rmse': rmse, 'normalized_tp': tp,
                      'max_factor_difference': float(np.max(np.abs(actual.final_factor-reference.final_factor)))}))


def test_initialization_seed_mismatch_rejected():
    data, factor0, config = fixture(2)
    config['initialization_seed'] += 1
    with pytest.raises(ValueError, match='initialization seed'):
        backend.run(data, factor0, config)


def test_n4_radius_must_scale_with_dimension():
    data, factor0, config = fixture(4)
    config['tau'] = 2.
    with pytest.raises(ValueError, match='tau=sqrt'):
        backend.run(data, factor0, config)


def test_frozen_parameters_reject_retuning():
    data, factor0, config = fixture(2)
    config['rho_exponent'] = .6
    with pytest.raises(ValueError, match='Frozen setting changed'):
        backend.run(data, factor0, config)


def test_frozen_adapter_hash_checked(monkeypatch):
    monkeypatch.setattr(backend, 'FROZEN_N2_ADAPTER_SHA256', '0' * 64)
    with pytest.raises(RuntimeError, match='hash mismatch'):
        backend._load_frozen_n2_adapter()
