"""Frozen-setting local FRAMES follow-up using existing optimizers only.

n=2 delegates by verified file hash to the frozen dense-JAX adapter.
n=3/4 use the repository's unchanged matrix-free NumPy campaign runner.
The supplied initial factor is verified against the explicitly declared seed.
No dense higher-dimensional sensing banks, new updates or resumption are used.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

REPO = Path('/home/tony/frankwolfe')
FROZEN_N2_ADAPTER = Path('/home/tony/qpt_frames_local_20261012/jax_backend.py')
FROZEN_N2_ADAPTER_SHA256 = 'a96b2931d3be190c221a5025e722363d6fc0e494b1a508da9f6e3b3cbb30f44e'
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from paper.experiments.qpt_quiroga_campaign import run_campaign
from paper.experiments.qpt_quiroga_initialization import gell_mann_coefficients_to_choi
from paper.experiments.quantum_process_tomography import (
    PowerSchedule, make_factor_initial_point, unpack_factor,
)

FROZEN_PARAMETERS = {
    'alpha0': 5e-5, 'penalty_exponent': .25,
    'rho_scale': 4., 'rho_offset': 8., 'rho_exponent': 2. / 3.,
    'step_scale': 4., 'step_offset': 4., 'step_exponent': 1.,
    'batch_size': 32,
}


def schedules(config):
    beta0 = 1. / (2. * config['alpha0'])
    return {
        'beta0': beta0,
        'smoothing_schedule': PowerSchedule(beta0, 1., config['penalty_exponent']),
        'rho_schedule': PowerSchedule(config['rho_scale'], config['rho_offset'],
                                      config['rho_exponent'], cap=1.),
        'step_size_schedule': PowerSchedule(config['step_scale'], config['step_offset'],
                                            config['step_exponent'], cap=1.),
    }


def expected_initial_factor(data, initialization_seed):
    coefficients = unpack_factor(
        make_factor_initial_point(data, rank=1, seed=initialization_seed),
        data.process_dimension, 1,
    )
    return gell_mann_coefficients_to_choi(coefficients)


def _validate(data, factor0, config):
    for key in ('n_qubits', 'channel', 'initialization_replicate', 'initialization_seed',
                'sampling_replicate', 'sampling_seed', 'steps', 'metrics_every', 'batch_size'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be an explicit nonnegative integer.')
    if config['n_qubits'] not in (2, 3, 4):
        raise ValueError('Follow-up is limited to n=2,3,4.')
    if any(config[key] <= 0 for key in ('steps', 'metrics_every', 'batch_size')):
        raise ValueError('Steps, metric cadence and batch size must be positive.')
    if data.design.n_qubits != config['n_qubits'] or data.observation_mode != 'noiseless':
        raise ValueError('Supplied data must match the declared noiseless dimension.')
    if data.metadata.get('channel_seed') != config['channel']:
        raise ValueError('Supplied data must match the declared channel seed.')
    if data.m != 2 * data.d**3 or data.truth_factor is None or data.truth_factor.shape != (data.process_dimension, 1):
        raise ValueError('Full Quiroga design and rank-one truth are required.')
    if factor0.shape != (data.process_dimension, 1) or not np.all(np.isfinite(factor0)):
        raise ValueError('Explicit finite rank-one factor required.')
    if not np.isfinite(config['tau']) or config['tau'] != math.sqrt(data.d):
        raise ValueError('The frozen dimension rule requires tau=sqrt(d).')
    for key, expected in FROZEN_PARAMETERS.items():
        if config[key] != expected:
            raise ValueError(f'Frozen setting changed: {key} must equal {expected!r}.')
    expected = expected_initial_factor(data, config['initialization_seed'])
    if not np.array_equal(factor0, expected):
        raise ValueError('Explicit factor0 does not match the declared Gell-Mann initialization seed.')


def _load_frozen_n2_adapter():
    actual_hash = hashlib.sha256(FROZEN_N2_ADAPTER.read_bytes()).hexdigest()
    if actual_hash != FROZEN_N2_ADAPTER_SHA256:
        raise RuntimeError('Frozen n=2 adapter hash mismatch; refusing delegation.')
    spec = importlib.util.spec_from_file_location('qpt_frozen_n2_adapter_20261012', FROZEN_N2_ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(data, factor0, config):
    """Return native metadata/payload and FRAMES trace, plus follow-up fields.

    ``metadata['configuration']`` is left native to its backend.  The complete
    follow-up trial is in ``followup_configuration`` in both cases.  For NumPy
    only the seed is archived: its actual RNG batches were not returned by the
    campaign runner, and no reconstructed arrays are mislabelled as observed.
    """
    factor0 = np.asarray(factor0, dtype=np.complex128).copy()
    _validate(data, factor0, config)
    if config['n_qubits'] == 2:
        metadata, payload, trace = _load_frozen_n2_adapter().run(data, factor0, config)
        backend = {
            'name': 'frozen_n2_dense_jax',
            'native_runner': 'paper.experiments.quantum_process_tomography_jax.run_qpt_stochastic_frames_jax',
            'frozen_adapter_path': str(FROZEN_N2_ADAPTER),
            'frozen_adapter_sha256': FROZEN_N2_ADAPTER_SHA256,
            'observed_batch_indices_archived': True,
            'sampling_qualification': 'Frozen JAX runner returns the complete host-generated PCG64 batch plan, archived in frames_diagnostic_batch_indices.',
            'timing_qualification': 'Whole-scan timing includes JIT compilation; interior optimizer times unavailable (NaN); post-hoc metrics excluded.',
        }
    else:
        schedule = schedules(config)
        metadata, payload, traces = run_campaign(
            data, methods=['frames'], rank=1,
            initialization_basis='gell-mann', initialization_seed=config['initialization_seed'],
            sampling_seed=config['sampling_seed'], n_steps=config['steps'],
            frames_steps=config['steps'], metrics_frequency=config['metrics_every'],
            frames_metrics_frequency=config['metrics_every'], tau=config['tau'],
            batch_size=config['batch_size'], eta_scale=1., tp_weight=1.,
            frames_beta0=schedule['beta0'], frames_smoothing_schedule=schedule['smoothing_schedule'],
            rho_schedule=schedule['rho_schedule'], step_size_schedule=schedule['step_size_schedule'],
            show_progress=False,
        )
        trace = traces['frames']
        backend = {
            'name': 'native_quiroga_structured_numpy',
            'native_runner': 'paper.experiments.qpt_quiroga_campaign.run_campaign -> run_quiroga_sensing_stochastic_frames',
            'observed_batch_indices_archived': False,
            'sampling_qualification': 'Native NumPy objective draws Generator(PCG64(sampling_seed)).integers(0,m,batch_size) per update. Explicit seed archived; actual batch arrays and terminal RNG state are not returned or archived. No reconstructed expected stream is presented as observed.',
            'timing_qualification': 'Native checkpoint optimizer times measured; post-hoc full-design metrics excluded; no JAX compilation in this branch.',
        }
    if not np.array_equal(payload['initial_factor'], factor0):
        raise RuntimeError('Native runner initial factor differs from supplied factor0.')
    metadata['followup_configuration'] = dict(config)
    metadata['followup_backend'] = {
        **backend,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initial_factor_verified_before_and_after': True,
        'explicit_initialization_seed': int(config['initialization_seed']),
        'explicit_sampling_seed': int(config['sampling_seed']),
        'higher_dimension_dense_banks': False,
        'tp': 'exact factorized residual and Jacobian-adjoint penalty in structured branch; exact dense equivalent in frozen n=2 branch',
        'tau_rule': 'sqrt(d)',
        'alpha_rule': 'fixed per-row alpha0; equivalent summed lambda0=m*alpha0',
        'independent_frames_penalty': True,
        'resume_supported': False,
    }
    metadata['endpoints'] = {'frames': trace.endpoint()}
    return metadata, payload, trace
