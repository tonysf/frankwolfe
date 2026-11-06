"""Local n=2 calibration adapter for the unchanged dense CPU JAX runner.

This is an archive/measurement adapter, not a new FRAMES implementation.
The repository runner owns every optimization update.  Full-design metrics
are post-hoc; scan timing includes compilation and has no intermediate times.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

REPO = Path('/home/tony/frankwolfe')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from paper.experiments.qpt_quiroga_campaign import (
    COMMON_METRICS, _make_trace, evaluate_common_metrics,
)
from paper.experiments.qpt_quiroga_initialization import initialization_descriptor
from paper.experiments.qpt_quiroga_sensing import _array_sha256
from paper.experiments.quantum_process_tomography import PowerSchedule, pack_factor
from paper.experiments.quantum_process_tomography_jax import run_qpt_stochastic_frames_jax


def schedules(config):
    """The independently chosen FRAMES continuation and open-loop schedules."""
    beta0 = 1.0 / (2.0 * config['alpha0'])
    return dict(
        beta0=beta0,
        smoothing_schedule=PowerSchedule(beta0, 1.0, config['penalty_exponent']),
        rho_schedule=PowerSchedule(config['rho_scale'], config['rho_offset'],
                                   config['rho_exponent'], cap=1.0),
        step_size_schedule=PowerSchedule(config['step_scale'], config['step_offset'],
                                         config['step_exponent'], cap=1.0),
    )


def sampling_seed(config):
    return int(config.get('sampling_seed',
                          300000 + 100 * config['channel'] + config.get('sampling_replicate', 0)))


def _validate(data, factor0, config):
    if data.design.n_qubits != 2 or data.m != 128 or data.observation_mode != 'noiseless':
        raise ValueError('This adapter is restricted to full, noiseless n=2 calibration.')
    if factor0.shape != (16, 1) or not np.all(np.isfinite(factor0)):
        raise ValueError('Explicit rank-one Gell-Mann-initialized Choi factor required.')
    if data.truth_factor is None or data.truth_factor.shape != (16, 1):
        raise ValueError('Rank-one truth is required for calibration fidelity.')
    if data.metadata.get('channel_seed') != config['channel']:
        raise ValueError('Configuration channel does not match supplied data.')
    for key in ('steps', 'metrics_every', 'batch_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ('alpha0', 'penalty_exponent', 'step_scale', 'step_offset',
                'step_exponent', 'rho_scale', 'rho_offset', 'rho_exponent', 'tau'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if np.linalg.norm(factor0, ord=2) > config['tau'] + 1e-10:
        raise ValueError('Initial factor is outside the operator-norm ball.')


def run(data, factor0, config):
    """Return ``(metadata, payload, MethodTrace)`` for one independent trial.

    ``factor0`` is an explicit Choi-coordinate factor already synthesized from
    the declared Gell-Mann initialization by the caller.  No reinitialization,
    projection, altered sampling, warm-up optimization or resume occurs here.
    """
    factor0 = np.asarray(factor0, dtype=np.complex128).copy()
    _validate(data, factor0, config)
    schedule = schedules(config)
    rows = np.arange(data.m, dtype=np.int64)
    targets = np.asarray(data.all_observations(), dtype=np.float64)
    metadata = {
        'format': 'qpt_frames_local_dense_jax_calibration_v1',
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'execution': 'local_cpu_calibration_dense_bridge',
        'algorithm': 'unchanged stochastic FRAMES repository JAX implementation',
        'repository_runner': 'paper.experiments.quantum_process_tomography_jax.run_qpt_stochastic_frames_jax',
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in (
                'paper/experiments/qpt_quiroga_campaign.py',
                'paper/experiments/qpt_quiroga_sensing.py',
                'paper/experiments/qpt_quiroga_initialization.py',
                'paper/experiments/quantum_process_tomography.py',
                'paper/experiments/quantum_process_tomography_jax.py',
            )
        },
        'configuration': dict(config),
        'initialization': {
            **initialization_descriptor('gell-mann'),
            'factor_source': 'explicit caller-provided Choi-coordinate factor; no internal random initialization',
        },
        'design': data.design.describe(),
        'data_metadata': data.metadata,
        'observation_mode': data.observation_mode,
        'active_row_count': int(rows.size),
        'active_rows_sha256': _array_sha256(rows),
        'active_observations_sha256': _array_sha256(targets),
        'initial_factor_sha256': _array_sha256(factor0),
        'truth_factor_sha256': _array_sha256(data.truth_factor),
        'sampling_seed': sampling_seed(config),
        'sampling': 'PCG64 NumPy Generator.integers with replacement; full saved batch-index plan',
        'objective': 'mean measurement half-squared loss + ||TP residual||_F^2/(2 beta_k)',
        'archived_penalized_objective': 'summed equivalent: m*mean loss + m/(2 beta_k)*||TP residual||_F^2',
        'frames_initial_tp_weight_per_row_alpha': float(config['alpha0']),
        'frames_initial_tp_weight_lambda': float(rows.size * config['alpha0']),
        'independent_penalty': True,
        'method_domains': {'frames': {'constraint': 'operator_norm_ball', 'tau': float(config['tau'])}},
        'checkpoint_beta_convention': 'beta[min(completed_updates, final_update_index)]',
        'checkpoint_metrics': 'exact over all active rows, computed post hoc by unchanged repository runner',
        'timing': {
            'warmup': False,
            'scan_time_includes_jit_compilation': True,
            'compile_time_separately_observed': False,
            'interior_optimizer_times': 'unavailable, stored as NaN; never interpolated',
            'scan_time_excludes': 'data bridge, host schedule/batch generation, transfers and post-hoc metrics',
        },
        'work_counter': 'nominal sampled optimizer rows only; excludes full checkpoint metric passes, TP, LMO, compilation and setup',
        'resume_supported': False,
        'failure_archival': 'on runner exception only initial factor/metrics available; no unobserved progress invented',
        'environment': {
            'python': platform.python_version(), 'numpy': np.__version__,
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'threads': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        },
    }
    adapter_start = perf_counter()
    try:
        bridge_start = perf_counter()
        dense = data.to_dense_qpt_data(rows=rows, observations=targets)
        metadata['dense_bridge_seconds'] = perf_counter() - bridge_start
        runner_start = perf_counter()
        result = run_qpt_stochastic_frames_jax(
            dense, n_steps=config['steps'], rank=1, tau=float(config['tau']),
            batch_size=config['batch_size'], sampling_seed=sampling_seed(config),
            x0=pack_factor(factor0), metrics_frequency=config['metrics_every'],
            device='cpu', precision='64', execution_mode='scan', warmup=False,
            show_progress=False, **schedule,
        )
        metadata['runner_wall_seconds'] = perf_counter() - runner_start
        metadata['jax'] = {
            name: getattr(result, name) for name in (
                'backend', 'device', 'device_platform', 'device_kind', 'jax_version',
                'jaxlib_version', 'precision', 'mode', 'compile_seconds', 'total_seconds', 'warmup',
            )
        }
        checkpoint_beta = result.checkpoint_smoothing_parameters
        metrics = []
        for index, beta in enumerate(checkpoint_beta):
            loss = float(result.measurement_loss[index])
            tp = float(result.tp_violation[index])
            metrics.append({
                'measurement_loss_sum': rows.size * loss,
                'measurement_loss_mean': loss,
                'rmse': math.sqrt(2.0 * loss),
                'tp_violation': tp,
                'normalized_tp_violation': tp / math.sqrt(data.d),
                'penalized_objective': rows.size * (loss + 0.5 * tp**2 / beta),
                'process_fidelity': float(result.process_fidelity_proxy[index]),
            })
        if not all(np.isfinite(value) for metric in metrics for value in metric.values()):
            raise FloatingPointError('Nonfinite checkpoint metric returned by JAX runner.')
        diagnostics = {
            'exact_smoothed_gap': result.exact_smoothed_gap,
            'checkpoint_smoothing_parameter': checkpoint_beta,
            'checkpoint_tp_weight_per_row_alpha': 1.0 / (2.0 * checkpoint_beta),
            'checkpoint_tp_weight_lambda': rows.size / (2.0 * checkpoint_beta),
            'estimated_gap': result.estimated_gaps,
            'momentum_weight': result.momentum_weights,
            'smoothing_parameter': result.smoothing_parameters,
            'tp_weight_per_row_alpha': 1.0 / (2.0 * result.smoothing_parameters),
            'tp_weight_lambda': rows.size / (2.0 * result.smoothing_parameters),
            'step_size': result.step_sizes,
            'batch_indices': result.batch_indices,
            'final_gradient_estimate': result.final_gradient_estimate,
        }
        trace = _make_trace(
            'frames', 'complete', '', result.final_factor, result.checkpoint_steps,
            result.optimizer_seconds, result.checkpoint_steps * config['batch_size'],
            metrics, diagnostics,
        )
        metadata['metric_checkpoint_row_count'] = int(rows.size * len(result.checkpoint_steps))
        metadata['metric_checkpoint_row_count_qualification'] = 'one full loss/gradient metric evaluation per checkpoint; not a total flop count'
    except Exception as error:
        metadata['failure_traceback'] = traceback.format_exc()
        metric = evaluate_common_metrics(data, factor0, rows, targets, rows.size * config['alpha0'])
        trace = _make_trace('frames', 'failed', f'{type(error).__name__}: {error}',
                            factor0, [0], [0.0], [0], [metric])
    metadata['adapter_total_seconds'] = perf_counter() - adapter_start
    metadata['endpoints'] = {'frames': trace.endpoint()}
    payload = {
        'active_rows': rows, 'active_observations': targets,
        'initial_factor': factor0, 'truth_factor': data.truth_factor.copy(),
    }
    for name in ('final_factor', 'checkpoint_steps', 'optimizer_seconds',
                 'cumulative_data_row_accesses', *COMMON_METRICS):
        payload[f'frames_{name}'] = np.asarray(getattr(trace, name))
    for name, values in trace.diagnostics.items():
        payload[f'frames_diagnostic_{name}'] = np.asarray(values)
    return metadata, payload, trace
