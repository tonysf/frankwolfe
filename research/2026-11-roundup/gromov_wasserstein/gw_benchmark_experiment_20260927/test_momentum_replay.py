"""Small CPU fidelity tests, not production trajectory or timing measurements."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from make_instances import random_metric, write_instance
from replay_momentum_diagnostics import input_bindings
from run_benchmark import parser, run


ROOT = Path(__file__).resolve().parent
UTILITY = ROOT / 'replay_momentum_diagnostics.py'


@pytest.fixture(scope='module')
def original(tmp_path_factory):
    base = tmp_path_factory.mktemp('replay_original')
    rng = np.random.default_rng(190)
    instance = base / 'toy8.npz'
    write_instance(instance, random_metric(8, rng), random_metric(8, rng), dict(family='test_only'))
    path = base / 'run'
    result = run(parser().parse_args([
        '--instance', str(instance), '--method', 'smoothing-fw', '--out', str(path),
        '--budget', '.01', '--checkpoints', '.003,.01', '--seed', '17', '--init-seed', '119',
        '--gamma0', '.3', '--rho0', '.3', '--batch', '8']))
    assert result['status'] == 'completed'
    assert result['counters']['completed_updates'] > 2
    return path, instance


def replay(path, instance, out):
    result = subprocess.run([sys.executable, str(UTILITY), '--run-dir', str(path),
                             '--instance', str(instance), '--out', str(out), '--device', 'cpu'],
                            text=True, capture_output=True, timeout=60)
    report = json.loads((out / 'verification.json').read_text())
    return result, report


def read_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_replay_matches_original_endpoint_checkpoints_rng_and_prior_input_indices(original, tmp_path):
    path, instance = original
    before = input_bindings(path, instance)
    bytecode_before = list((path / 'source_snapshot').rglob('*.pyc'))
    result, report = replay(path, instance, tmp_path / 'replay')
    assert result.returncode == 0, report
    assert report['verified']
    assert report['checks']['P_raw_bitwise'] and report['checks']['P_rounded_bitwise']
    assert report['checks']['diagnostics_preserved_oracle_rng']
    assert input_bindings(path, instance) == before == report['bindings']
    assert list((path / 'source_snapshot').rglob('*.pyc')) == bytecode_before
    assert report['replay_counters'] == report['original_counters']
    assert report['timing']['controller_elapsed_seconds'] > 0
    rows = read_rows(tmp_path / 'replay' / 'diagnostics.csv')
    original_rows = read_rows(path / 'trace.csv')
    assert len(rows) == report['checkpoint_count']
    diagnostic_counts = report['replay_diagnostic_counts']
    assert diagnostic_counts['loss_calls'] == 3 * len(rows)
    assert diagnostic_counts['full_gradient_calls'] == 3 * len(rows) - 1
    assert diagnostic_counts['assignment_lmo_calls'] == len(rows)
    assert diagnostic_counts['rounding_calls'] == len(rows)
    assert diagnostic_counts['oracle_draws'] == 0
    for row in rows:
        k = int(row['output_iteration'])
        assert int(row['momentum_point_iteration']) == max(0, k - 1)
        assert int(row['stored_momentum_iteration']) == max(0, k - 1)
        assert int(row['tracking_error_point_iteration']) == k
        assert row['momentum_error_matches_current_output'] == str(k == 0)
        assert row['original_actual_solver_seconds'] == original_rows[int(row['original_trace_row'])]['actual_solver_seconds']
        assert int(row['replay_diagnostic_oracle_draws']) == 0
        assert int(row['original_solver_oracle_evaluations']) == max(k, 1)
    assert rows[0]['momentum_error_frobenius'] == rows[0]['stored_momentum_current_output_tracking_error_frobenius']
    assert not (tmp_path / 'replay' / 'candidate_diagnostics.csv').exists()


@pytest.mark.parametrize('method,oracle,initialization', [
    ('smoothing-fw', 'exact', 'randomized'),
    ('smoothing-row-projection', 'sketch', 'randomized'),
])
def test_replay_resolved_schedules_initialization_and_exact_momentum_control(original, tmp_path, method, oracle, initialization):
    _, instance = original
    path = tmp_path / 'original'
    metadata = run(parser().parse_args([
        '--instance', str(instance), '--method', method, '--out', str(path),
        '--budget', '.01', '--checkpoints', '.003,.01', '--seed', '17', '--init-seed', '119',
        '--gamma0', '.3', '--rho0', '.3', '--batch', '8', '--eta', '.01', '--eta-power', '.25',
        '--oracle', oracle, '--initialization', initialization]))
    assert metadata['status'] == 'completed'
    result, report = replay(path, instance, tmp_path / 'replay')
    assert result.returncode == 0, report
    assert report['initialization_plan_sha256'] == metadata['initialization_plan_sha256']
    rows = read_rows(tmp_path / 'replay' / 'diagnostics.csv')
    if oracle == 'exact':
        assert float(rows[0]['momentum_error_frobenius']) == 0
        assert any(float(row['momentum_error_frobenius']) > 0 for row in rows[1:])


@pytest.mark.parametrize('corruption', ['raw', 'rounded', 'checkpoint', 'missing_gap', 'source'])
def test_replay_fails_closed_on_plan_metric_or_source_mismatch(original, tmp_path, corruption):
    path, instance = original
    changed = tmp_path / 'changed'
    shutil.copytree(path, changed)
    if corruption in ('raw', 'rounded'):
        with np.load(changed / 'final.npz') as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        key = 'P_raw' if corruption == 'raw' else 'P_rounded'
        arrays[key][0, 0] = np.nextafter(arrays[key][0, 0], np.inf)
        np.savez(changed / 'final.npz', **arrays)
    elif corruption in ('checkpoint', 'missing_gap'):
        rows = read_rows(changed / 'trace.csv')
        if corruption == 'checkpoint':
            rows[0]['raw_loss'] = str(float(rows[0]['raw_loss']) + .1)
        else:
            rows[0]['original_fw_gap'] = ''
        with (changed / 'trace.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with (changed / 'source_snapshot' / 'gw_core.py').open('a') as stream:
            stream.write('\n# deliberate test-only source hash mismatch\n')
    result, report = replay(changed, instance, tmp_path / 'replay')
    assert result.returncode == 1 and not report['verified']
    assert not (tmp_path / 'replay' / 'diagnostics.csv').exists()
    expected = {'raw': 'Final bitwise plan mismatch: P_raw', 'rounded': 'Final bitwise plan mismatch: P_rounded',
                'checkpoint': 'Checkpoint scalar mismatch', 'missing_gap': 'Original FW gap coverage mismatch',
                'source': 'Original source hash mismatch'}
    assert expected[corruption] in report['error']
    assert report['checks']['original_files_unchanged']
    if corruption in ('raw', 'rounded'):
        comparison = report['endpoint_comparisons']['P_' + corruption]
        assert not comparison['bitwise_match']
        assert comparison['expected_array_sha256'] != comparison['replay_array_sha256']
        assert comparison['max_abs_difference'] > 0


def test_failed_original_with_missing_outputs_is_bound_but_never_reconstructed(original, tmp_path):
    path, instance = original
    changed = tmp_path / 'failed'
    shutil.copytree(path, changed)
    meta = json.loads((changed / 'metadata.json').read_text())
    meta.update(status='failed', error='test-only failed original')
    (changed / 'metadata.json').write_text(json.dumps(meta))
    (changed / 'trace.csv').unlink()
    (changed / 'final.npz').unlink()
    result, report = replay(changed, instance, tmp_path / 'replay')
    assert result.returncode == 1
    assert report['failure_kind'] == 'unsupported_original_failure'
    assert report['bindings']['trace_sha256'] is None and report['bindings']['final_sha256'] is None
    assert report['candidate_checkpoint_count'] == 0
    assert 'replay_environment' not in report


def test_legacy_pilot_snapshot_uses_old_initializer_without_live_fallback(original, tmp_path):
    # Optional preserved archive integration: this executes only an eight-point CPU
    # test problem, never the historical 512-point GPU trajectory or held-out data.
    archive = ROOT / 'ruche_results/1961459/metric512_s1001/smoothing-fw/source_snapshot'
    if not archive.exists():
        pytest.skip('Preserved pilot source snapshot not available.')
    _, instance = original
    code = tmp_path / 'pilot_code'
    shutil.copytree(archive, code)
    (code / 'configs').mkdir()
    shutil.copy2(code / 'theory_config.json', code / 'configs/theory_config.json')
    path = tmp_path / 'pilot_original'
    result = subprocess.run([sys.executable, '-B', str(code / 'run_benchmark.py'),
                             '--instance', str(instance), '--method', 'smoothing-fw', '--out', str(path),
                             '--budget', '.01', '--checkpoints', '.003,.01', '--seed', '17', '--init-seed', '119',
                             '--batch', '8', '--device', 'cpu'], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    before = input_bindings(path, instance)
    result, report = replay(path, instance, tmp_path / 'replay')
    assert result.returncode == 0, report
    assert report['checks']['initialization_hash'].startswith('unavailable_in_legacy_original')
    assert before == input_bindings(path, instance)
    assert report['bindings']['source_file_sha256']['smoothing_fw.py'] == hashlib.sha256((archive / 'smoothing_fw.py').read_bytes()).hexdigest()


def test_archived_source_is_compiled_directly_despite_unverified_cached_bytecode(original, tmp_path):
    import importlib.util
    import importlib._bootstrap_external
    path, instance = original
    copied = tmp_path / 'cached_original'
    shutil.copytree(path, copied)
    source = copied / 'source_snapshot' / 'gw_core.py'
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(exist_ok=True)
    # A timestamp-valid cache with different code must never override bound .py.
    code = compile("raise RuntimeError('unverified cached bytecode was executed')", str(source), 'exec')
    cache.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(
        code, int(source.stat().st_mtime), source.stat().st_size))
    before = cache.read_bytes()
    result, report = replay(copied, instance, tmp_path / 'replay')
    assert result.returncode == 0, report
    assert cache.read_bytes() == before
    assert report['checks']['P_raw_bitwise'] and report['checks']['P_rounded_bitwise']
