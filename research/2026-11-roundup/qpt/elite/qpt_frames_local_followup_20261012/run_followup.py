"""Frozen local FRAMES robustness and dimension-transfer experiments; no remote use."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/tony/qpt_frames_local_20261012')
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
sys.path.insert(0, str(REPO))
import numpy as np
import scipy
from paper.experiments.qpt_quiroga_campaign import evaluate_common_metrics, save_campaign
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.qpt_quiroga_initialization import gell_mann_coefficients_to_choi
from paper.experiments.quantum_process_tomography import make_factor_initial_point, unpack_factor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source():
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
    names = subprocess.check_output(['git', '-C', str(REPO), 'ls-files', 'paper', 'frank_wolfe'], text=True).splitlines()
    return {name: digest(REPO / name) for name in names if name.endswith('.py')}


def validate_plan(plan):
    assert plan['scope'] == 'local_frames_frozen_followup' and plan['gate'] == GATE
    assert plan['phase'] in ('pilot', 'robustness', 'tail', 'transfer')
    assert plan['source_commit'] == COMMIT and plan['source_sha256'] == source()
    assert len({t['id'] for t in plan['trials']}) == len(plan['trials']) and plan['trials']
    allowed_parents = {str(PARENT / n) for n in ('selection_long.json', 'audit_confirmation.json',
                                               'completion_manifest.json', 'jax_backend.py')}
    assert set(plan['parent_pins']) == allowed_parents
    assert all(digest(p) == h for p, h in plan['parent_pins'].items())
    assert set(plan['harness_sha256']) == {'run_followup.py', 'run_batch.py', 'make_plans.py', 'backend_followup.py'}
    assert all(Path(p).name == p and digest(ROOT / p) == h for p, h in plan['harness_sha256'].items())
    assert plan['reserved_final_channels'] == list(range(1000, 1020))
    if plan['phase'] == 'transfer':
        decision = plan['resource_decision']
        assert Path(decision['path']).resolve() == ROOT / 'resource_decision.json'
        assert digest(decision['path']) == decision['sha256']
        dimensions = json.loads(Path(decision['path']).read_text())['dimensions']
        expected_pairs = {(int(n), ch) for n, row in dimensions.items() if row['selected_steps'] for ch in (0, 1)}
        assert {(c['n_qubits'], c['channel']) for c in plan['trials']} == expected_pairs
        assert all(c['steps'] == dimensions[str(c['n_qubits'])]['selected_steps'] for c in plan['trials'])
    elif plan['phase'] == 'pilot':
        assert len(plan['trials']) == 2 and {c['n_qubits'] for c in plan['trials']} == {3, 4}
    elif plan['phase'] == 'robustness':
        assert len(plan['trials']) == 12
        assert {(c['channel'], c['sampling_replicate']) for c in plan['trials']} == {(c, s) for c in range(2, 8) for s in (10, 11)}
    else:
        assert len(plan['trials']) == 1
    for c in plan['trials']:
        assert Path(c['id']).name == c['id']
        assert c['n_qubits'] in (2, 3, 4) and c['channel'] in range(8)
        assert c['initialization_replicate'] in range(10) and c['sampling_replicate'] in range(100)
        assert c['initialization_seed'] == 200000 + 10 * c['channel'] + c['initialization_replicate']
        assert c['sampling_seed'] == 300000 + 100 * c['channel'] + c['sampling_replicate']
        assert c['alpha0'] == 5e-5 and c['penalty_exponent'] == .25
        assert (c['rho_scale'], c['rho_offset'], c['rho_exponent']) == (4., 8., 2 / 3)
        assert (c['step_scale'], c['step_offset'], c['step_exponent']) == (4., 4., 1.)
        assert c['tau'] == math.sqrt(2 ** c['n_qubits']) and c['batch_size'] == 32
        assert type(c['steps']) is int and 0 < c['steps'] <= 800000
        assert type(c['metrics_every']) is int and c['steps'] % c['metrics_every'] == 0
        if plan['phase'] == 'pilot':
            assert c['n_qubits'] in (3, 4) and (c['channel'], c['initialization_replicate'], c['sampling_replicate'], c['steps'], c['metrics_every']) == (0, 0, 0, 2000, 1000)
        elif plan['phase'] == 'robustness':
            assert c['n_qubits'] == 2 and c['channel'] in range(2, 8)
            assert c['initialization_replicate'] == 1 and c['sampling_replicate'] in (10, 11)
            assert (c['steps'], c['metrics_every']) == (400000, 4000)
        elif plan['phase'] == 'tail':
            assert (c['n_qubits'], c['channel'], c['initialization_replicate'], c['sampling_replicate'], c['steps'], c['metrics_every']) == (2, 0, 0, 0, 800000, 4000)
        else:
            assert c['n_qubits'] in (3, 4) and c['channel'] in (0, 1)
            assert c['initialization_replicate'] == c['sampling_replicate'] == 0
            assert c['steps'] in (100000, 400000) and c['metrics_every'] == 4000


def execute(plan_path, trial_id):
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    c = next(t for t in plan['trials'] if t['id'] == trial_id)
    out = ROOT / plan['phase'] / trial_id
    out.mkdir(parents=True, exist_ok=False)
    started = {'trial': c, 'plan_path': str(plan_path.resolve()), 'plan_sha256': digest(plan_path),
               'source_commit': COMMIT, 'source_sha256': plan['source_sha256'],
               'harness_sha256': plan['harness_sha256'], 'parent_pins': plan['parent_pins'],
               'runner_sha256': digest(__file__), 'host': socket.gethostname(),
               'platform': platform.platform(), 'python': platform.python_version(),
               'numpy': np.__version__, 'scipy': scipy.__version__,
               'cpu_affinity': sorted(os.sched_getaffinity(0)), 'pid': os.getpid(),
               'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
               'started_utc': now(), 'execution': 'local_cpu', 'split': plan['phase']}
    write(out / 'started.json', started)
    start = time.perf_counter()
    try:
        data = generate_quiroga_sensing_data(c['n_qubits'], channel_seed=c['channel'], observation_mode='noiseless')
        factor0 = gell_mann_coefficients_to_choi(unpack_factor(make_factor_initial_point(
            data, rank=1, seed=c['initialization_seed']), data.process_dimension, 1))
        from backend_followup import run
        metadata, payload, trace = run(data, factor0, c)
        targets = data.all_observations()
        metrics = evaluate_common_metrics(data, trace.final_factor, np.arange(data.m, dtype=np.int64), targets, 0.)
        gate = bool(trace.status == 'complete' and metrics['process_fidelity'] >= .99
                    and metrics['rmse'] <= 1e-6 and metrics['normalized_tp_violation'] <= .001)
        deficit = max(metrics['rmse'] / 1e-6, metrics['normalized_tp_violation'] / .001,
                      max(0., 1. - metrics['process_fidelity']) / .01)
        st = payload['frames_checkpoint_steps']
        assert np.array_equal(payload['frames_cumulative_data_row_accesses'], st * c['batch_size'])
        if trace.status == 'complete':
            assert np.array_equal(st, np.arange(0, c['steps'] + 1, c['metrics_every']))
        assert np.isclose(metrics['rmse'], payload['frames_rmse'][-1], rtol=1e-9, atol=1e-15)
        passes = ((payload['frames_rmse'] <= 1e-6)
                  & (payload['frames_normalized_tp_violation'] <= .001)
                  & (payload['frames_process_fidelity'] >= .99))
        first = int(st[np.flatnonzero(passes)[0]]) if passes.any() else None
        persistent = np.logical_and.accumulate(passes[::-1])[::-1]
        persistent_step = int(st[np.flatnonzero(persistent)[0]]) if persistent.any() else None
        metadata['followup_provenance'] = {**started, 'endpoint_gate': gate,
                                          'persistent_recorded_gate_step': persistent_step}
        save_campaign(out / 'result.npz', out / 'result.json', metadata, payload)
        summary = {
            'trial': c, 'status': trace.status, 'failure_reason': trace.failure_reason,
            'gate_pass': gate, 'deficit': float(deficit), 'rmse': float(metrics['rmse']),
            'normalized_tp': float(metrics['normalized_tp_violation']), 'fidelity': float(metrics['process_fidelity']),
            'relative_rmse': float(metrics['rmse'] * math.sqrt(data.m) / np.linalg.norm(targets)),
            'outcome_scaled_rmse': float(2 * data.d * metrics['rmse']),
            'first_recorded_gate_step': first, 'persistent_recorded_gate_step': persistent_step,
            'optimizer_seconds': float(trace.optimizer_seconds[-1]),
            'total_seconds': time.perf_counter() - start,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
            'json_sha256': digest(out / 'result.json'), 'npz_sha256': digest(out / 'result.npz'),
            'finished_utc': now(),
        }
        write(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
        return summary
    except Exception as error:
        write(out / 'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                                    'total_seconds': time.perf_counter() - start, 'finished_utc': now()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--trial', required=True)
    args = parser.parse_args()
    execute(args.plan, args.trial)
