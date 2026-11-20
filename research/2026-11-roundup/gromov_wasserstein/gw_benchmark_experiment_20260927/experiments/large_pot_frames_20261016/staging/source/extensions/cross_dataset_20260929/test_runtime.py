"""Fresh-process integration checks for the frozen cross-dataset runner.

Tiny CPU runs test output/provenance contracts, not timing performance. Importing
runner modules only inside subprocesses avoids collisions with original study
modules when the whole repository's pytest suite is collected together.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime'
REPO = HERE.parents[1]
PYTHON = REPO / '.venv/bin/python'
METHODS = ('smoothing-fw', 'stochastic-kl-momentum')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_process(arguments, *, timeout=90):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(RUNTIME), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return subprocess.run([str(PYTHON), *map(str, arguments)], cwd=RUNTIME,
                          env=env, capture_output=True, text=True, timeout=timeout)


def runner_args(instance, correspondence, digest, method, out):
    return [RUNTIME / 'run_benchmark.py', '--instance', instance,
            '--correspondence-file', correspondence, '--correspondence-sha256', digest,
            '--method', method, '--out', out, '--budget', '.005', '--checkpoints', '.002,.005',
            '--device', 'cpu', '--dtype', 'float32', '--threads', '1', '--seed', '724',
            '--init-seed', '921', '--initialization', 'product', '--oracle', 'sketch',
            '--batch', '128', '--rho0', '.3', '--gamma0', '.3', '--mu-scale', '10',
            '--eta', '.1', '--inner-tol', '1e-6', '--inner-max', '100', '--gap-max-n', '0']


@pytest.fixture(scope='module')
def tiny_runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cross_runtime')
    n = 7
    X = np.random.default_rng(814).normal(size=(n, 3))
    A = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    A = (A / A.max()).astype(np.float32)
    permutation = np.array([3, 6, 1, 5, 0, 2, 4])
    B = A[np.ix_(permutation, permutation)]
    instance = directory / 'paired.npz'
    np.savez(instance, A=A, B=B, a=np.full(n, 1/n), b=np.full(n, 1/n))
    instance.with_suffix('.json').write_text(json.dumps({
        'A_sha256': hashlib.sha256(A.tobytes()).hexdigest(),
        'B_sha256': hashlib.sha256(B.tobytes()).hexdigest(),
        'family': 'small_integration_fixture', 'n': n, 'm': n}))
    correspondence = directory / 'paired.correspondence.npz'
    forward = np.argsort(permutation)
    np.savez(correspondence, forward=forward, inverse=permutation)
    digest = sha256(correspondence)
    records = {}
    for method in METHODS:
        output = directory / method
        completed = run_process(runner_args(instance, correspondence, digest, method, output))
        assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr
        metadata = json.loads((output / 'metadata.json').read_text())
        with (output / 'trace.csv').open(newline='') as stream:
            trace = list(csv.DictReader(stream))
        with np.load(output / 'final.npz', allow_pickle=False) as archive:
            plans = {key: archive[key].copy() for key in ('P_raw', 'P_rounded')}
        records[method] = {'output': output, 'metadata': metadata, 'trace': trace, 'plans': plans}
    return {'directory': directory, 'instance': instance, 'correspondence': correspondence,
            'digest': digest, 'n': n, 'A': A, 'B': B, 'records': records}


def loss_from_definition(A, B, P):
    """Independent four-index definition; tests full objective and off-feasibility."""
    discrepancy = (A.astype(np.float64)[:, None, :, None]
                   - B.astype(np.float64)[None, :, None, :]) ** 2
    return float(np.einsum('ijkl,ij,kl->', discrepancy, P.astype(np.float64), P.astype(np.float64)))


@pytest.mark.parametrize('method', METHODS)
def test_saved_output_float64_validation_and_correspondence_match(tiny_runs, method):
    run = tiny_runs['records'][method]
    metadata, plans = run['metadata'], run['plans']
    assert metadata['status'] in ('completed', 'completed_with_inner_failures')
    assert metadata['error'] is None
    assert metadata['counters']['completed_updates'] >= 1
    assert metadata['failure_accounting_complete'] is True
    validation = metadata['final_float64_validation']
    assert validation['dtype'] == 'float64' and validation['loss_calls'] == 2
    for key, field in (('P_raw', 'raw'), ('P_rounded', 'rounded')):
        P = plans[key]
        assert P.dtype == np.float32 and P.shape == (tiny_runs['n'], tiny_runs['n'])
        assert np.isfinite(P).all() and np.all(P >= 0)
        expected = loss_from_definition(tiny_runs['A'], tiny_runs['B'], P)
        assert validation[field + '_loss'] == pytest.approx(expected, rel=1e-11, abs=1e-12)
        residual = np.abs(P.astype(np.float64).sum(1) - 1 / tiny_runs['n']).sum()
        residual += np.abs(P.astype(np.float64).sum(0) - 1 / tiny_runs['n']).sum()
        assert validation[field]['both_l1'] == pytest.approx(residual, abs=1e-13)
    assert validation['rounded']['both_l1'] <= 1.2e-5

    # A second fresh process recomputes the diagnostic from exactly the stored
    # rounded output. The runner's imports cannot reuse another test's modules.
    verification = run_process(['-c', '''
import json, sys, numpy as np, torch
from measurements import load_correspondence, correspondence_metrics
with np.load(sys.argv[1], allow_pickle=False) as archive:
    plan = torch.as_tensor(archive['P_rounded'])
labels = load_correspondence(sys.argv[2], sys.argv[3], len(plan), len(plan), 'cpu')
found = correspondence_metrics(plan, labels, chunk=2)
print(json.dumps(found, sort_keys=True))
''', run['output'] / 'final.npz', tiny_runs['correspondence'], tiny_runs['digest']])
    assert verification.returncode == 0, verification.stderr
    recomputed = json.loads(verification.stdout)
    evaluated = [row for row in run['trace'] if row['kind'] != 'missed']
    final = evaluated[-1]
    assert int(final['iteration']) == metadata['counters']['completed_updates']
    for key, value in recomputed.items():
        assert float(final[key]) == pytest.approx(value, abs=1e-13)
    initial = evaluated[0]
    assert initial['kind'] == 'initial' and int(initial['iteration']) == 0
    assert float(initial['correspondence_top1_accuracy']) == pytest.approx(1/tiny_runs['n'])
    assert float(initial['correspondence_rank_fraction']) == .5
    assert float(initial['correspondence_true_pair_mass']) == pytest.approx(1/tiny_runs['n'])


@pytest.mark.parametrize('method', METHODS)
def test_labels_source_binding_and_diagnostic_accounting(tiny_runs, method):
    run = tiny_runs['records'][method]
    metadata, counters = run['metadata'], run['metadata']['counters']
    assert metadata['correspondence_sha256'] == tiny_runs['digest']
    assert metadata['config']['correspondence_sha256'] == tiny_runs['digest']
    assert metadata['instance_sha256'] == sha256(tiny_runs['instance'])
    assert 'withheld from solver updates and parameter selection' in metadata['correspondence_policy']
    for name in ('run_benchmark.py', 'gw_core.py', 'smoothing_fw.py', 'measurements.py'):
        snapshot = run['output'] / 'source_snapshot' / name
        assert metadata['source_file_sha256'][name] == sha256(snapshot) == sha256(RUNTIME / name)
    evaluated = [row for row in run['trace'] if row['kind'] != 'missed']
    diagnostic_counts = metadata['diagnostic_counts']
    assert diagnostic_counts['loss_calls'] == 2 * len(evaluated)
    assert diagnostic_counts['full_gradient_calls'] >= len(evaluated)
    assert counters['exact_gradient_calls'] == 0
    assert counters['stochastic_gradient_calls'] == counters['completed_updates']
    assert counters['sketch_probes'] == 128 * counters['stochastic_gradient_calls']
    assert counters['initialization_oracle_calls'] == 1
    times = metadata['timing']
    measured = ('preparation_seconds', 'solver_seconds', 'repair_seconds',
                'exact_diagnostic_seconds', 'final_float64_validation_seconds', 'serialization_seconds')
    assert all(np.isfinite(times[key]) and times[key] >= 0 for key in measured)
    assert times['exact_diagnostic_seconds'] > 0
    assert times['final_float64_validation_seconds'] > 0
    assert sum(times[key] for key in measured) + times['unclassified_python_overhead_seconds'] == pytest.approx(
        times['total_elapsed_seconds_before_final_metadata_write'], abs=1e-10)
    final = evaluated[-1]
    assert float(final['actual_solver_seconds']) == times['solver_seconds']
    assert float(final['diagnostic_seconds_cumulative']) == times['exact_diagnostic_seconds']
    assert float(final['time_to_this_feasible_output']) == pytest.approx(
        times['solver_seconds'] + float(final['repair_seconds_current']), abs=1e-12)
    # Full diagnostics and float64 validation are measured separately from the
    # solver clock; they do not appear in the feasible-output solver+repair axis.
    assert times['total_elapsed_seconds_before_final_metadata_write'] >= sum(times[key] for key in measured)


def test_matched_initial_plans_are_identical(tiny_runs):
    metadata = [tiny_runs['records'][method]['metadata'] for method in METHODS]
    assert metadata[0]['initialization_plan_sha256'] == metadata[1]['initialization_plan_sha256']
    assert metadata[0]['stochastic_seed'] == metadata[1]['stochastic_seed']
    assert metadata[0]['initialization_seed'] == metadata[1]['initialization_seed']


def test_bad_correspondence_hash_cannot_produce_a_valid_output(tiny_runs):
    output = tiny_runs['directory'] / 'rejected_bad_correspondence'
    completed = run_process(runner_args(tiny_runs['instance'], tiny_runs['correspondence'],
                                       '0' * 64, METHODS[0], output))
    assert completed.returncode != 0
    assert 'Correspondence SHA256 mismatch' in completed.stderr
    assert not (output / 'final.npz').exists()
    if (output / 'metadata.json').exists():
        assert json.loads((output / 'metadata.json').read_text())['status'] == 'failed'
