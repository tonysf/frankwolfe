"""Parity of local archival adapter against the unchanged structured runner."""
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/tony/frankwolfe')
import numpy as np
import pytest

from jax_backend import run, schedules, sampling_seed
from paper.experiments.qpt_quiroga_campaign import (
    COMMON_METRICS, evaluate_common_metrics, run_frames, save_campaign,
)
from paper.experiments.qpt_quiroga_initialization import gell_mann_coefficients_to_choi
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.quantum_process_tomography import make_factor_initial_point, unpack_factor


def fixture(channel=0, steps=100, alternate=False):
    config = dict(channel=channel, steps=steps, metrics_every=steps // 10,
                  alpha0=5e-5, penalty_exponent=.25, step_scale=4., step_offset=4.,
                  step_exponent=1., rho_scale=2., rho_offset=4., rho_exponent=.6,
                  batch_size=32, tau=2., sampling_replicate=0)
    if alternate:
        config.update(alpha0=5e-4, penalty_exponent=.5,
                      step_scale=2., step_offset=8., step_exponent=.8,
                      rho_scale=4., rho_offset=8., rho_exponent=2/3)
    data = generate_quiroga_sensing_data(2, channel_seed=channel, observation_mode='noiseless')
    factor0 = gell_mann_coefficients_to_choi(unpack_factor(
        make_factor_initial_point(data, rank=1, seed=200000 + 10 * channel), 16, 1))
    return data, factor0, config


@pytest.mark.parametrize('channel', [0, 1])
@pytest.mark.parametrize('steps', [100, 1000])
@pytest.mark.parametrize('alternate', [False, True])
def test_parity(channel, steps, alternate):
    data, factor0, config = fixture(channel, steps, alternate)
    initial_copy = factor0.copy()
    started = perf_counter()
    metadata, payload, actual = run(data, factor0, config)
    jax_wall = perf_counter() - started
    assert actual.status == 'complete', actual.failure_reason
    schedule = schedules(config)
    started = perf_counter()
    expected = run_frames(
        data, factor0, np.arange(data.m, dtype=np.int64), data.all_observations(),
        n_steps=steps, metrics_frequency=config['metrics_every'], rank=1,
        tau=config['tau'], batch_size=config['batch_size'], sampling_seed=sampling_seed(config),
        tp_weight=1., show_progress=False, **schedule,
    )
    numpy_wall = perf_counter() - started
    print(json.dumps({'channel': channel, 'steps': steps, 'alternate': alternate,
                      'jax_adapter_wall_seconds': jax_wall,
                      'jax_scan_including_compile_seconds': actual.optimizer_seconds[-1],
                      'structured_numpy_wall_seconds': numpy_wall,
                      'max_factor_difference': float(np.max(np.abs(actual.final_factor - expected.final_factor)))}))
    np.testing.assert_allclose(actual.final_factor, expected.final_factor, rtol=1e-8, atol=1e-10)
    for name in COMMON_METRICS:
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=1e-8, atol=1e-11, err_msg=name)
    for name in ('exact_smoothed_gap', 'estimated_gap', 'checkpoint_smoothing_parameter',
                 'momentum_weight', 'smoothing_parameter', 'step_size'):
        np.testing.assert_allclose(actual.diagnostics[name], expected.diagnostics[name],
                                   rtol=1e-8, atol=1e-11, err_msg=name)
    indices = np.random.default_rng(sampling_seed(config)).integers(
        0, data.m, size=(steps, config['batch_size']), dtype=np.int64)
    np.testing.assert_array_equal(payload['frames_diagnostic_batch_indices'], indices)
    np.testing.assert_array_equal(payload['frames_checkpoint_steps'], expected.checkpoint_steps)
    np.testing.assert_array_equal(payload['frames_cumulative_data_row_accesses'],
                                  expected.checkpoint_steps * config['batch_size'])
    np.testing.assert_array_equal(payload['initial_factor'], initial_copy)
    np.testing.assert_array_equal(factor0, initial_copy)
    np.testing.assert_array_equal(payload['truth_factor'], data.truth_factor)
    assert metadata['jax']['precision'] == '64' and metadata['jax']['device_platform'] == 'cpu'
    assert metadata['jax']['warmup'] is False and metadata['resume_supported'] is False
    assert np.all(np.isnan(actual.optimizer_seconds[1:-1]))
    metrics = evaluate_common_metrics(data, actual.final_factor, np.arange(data.m),
                                      data.all_observations(), data.m / (2 * actual.diagnostics['checkpoint_smoothing_parameter'][-1]))
    for name in COMMON_METRICS:
        np.testing.assert_allclose(getattr(actual, name)[-1], metrics[name], rtol=1e-10, atol=1e-13)
    json.dumps(metadata, allow_nan=False)


def test_archive_roundtrip(tmp_path):
    data, factor0, config = fixture(0, 100, False)
    metadata, payload, trace = run(data, factor0, config)
    assert trace.status == 'complete'
    npz, meta = tmp_path / 'result.npz', tmp_path / 'result.json'
    save_campaign(npz, meta, metadata, payload)
    with np.load(npz, allow_pickle=False) as archive:
        assert json.loads(str(archive['metadata_json'])) == json.loads(meta.read_text())
        for name, values in payload.items():
            np.testing.assert_array_equal(archive[name], values)
    with pytest.raises(FileExistsError):
        save_campaign(npz, meta, metadata, payload)


def test_failure_is_archived_without_inventing_progress(monkeypatch):
    import jax_backend
    data, factor0, config = fixture()
    def fail(*args, **kwargs):
        raise FloatingPointError('injected test failure')
    monkeypatch.setattr(jax_backend, 'run_qpt_stochastic_frames_jax', fail)
    metadata, payload, trace = run(data, factor0, config)
    assert trace.status == 'failed' and 'injected test failure' in trace.failure_reason
    np.testing.assert_array_equal(trace.final_factor, factor0)
    np.testing.assert_array_equal(payload['frames_checkpoint_steps'], [0])
    assert 'failure_traceback' in metadata
    json.dumps(metadata, allow_nan=False)
