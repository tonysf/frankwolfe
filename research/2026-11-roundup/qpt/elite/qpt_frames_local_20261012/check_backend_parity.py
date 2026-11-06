"""Read-only historical/current 400k FRAMES comparison; write one new audit."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/tony/frankwolfe')
OLD = Path('/home/tony/qpt_gell_mann_init_20261007_1750/completed_channels')
OUT = ROOT / 'backend_parity.json'
TOLERANCES = {
    'shared_floating_arrays': {'rtol': 0., 'atol': 1e-14},
    'final_factor': {'rtol': 1e-8, 'atol': 1e-10},
    'metrics_and_gaps': {'rtol': 1e-8, 'atol': 1e-11},
    'schedules': {'rtol': 1e-14, 'atol': 0.},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    valid_shape = actual.shape == expected.shape
    delta = np.abs(actual - expected) if valid_shape else None
    return {
        'pass': bool(valid_shape and np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
                     and np.allclose(actual, expected, **tolerance)),
        'exact_equal': bool(valid_shape and np.array_equal(actual, expected)),
        'max_abs_difference': float(np.max(delta)) if valid_shape else None,
        'max_relative_difference_with_tiny_denominator_floor': (
            float(np.max(delta / np.maximum(np.abs(expected), np.finfo(float).tiny))) if valid_shape else None),
        'shape_new': list(actual.shape), 'shape_old': list(expected.shape),
        'tolerance': tolerance,
    }


def git_source(commit, name):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', f'{commit}:{name}'])


def gate(archive):
    return ((archive['frames_rmse'] <= 1e-6)
            & (archive['frames_normalized_tp_violation'] <= 1e-3)
            & (archive['frames_process_fidelity'] >= .99))


def audit_channel(channel):
    old_path = OLD / f'channel-{channel}' / f'c{channel}-f-fast-average-100kep' / 'result.npz'
    new_path = ROOT / 'long' / f'penalty-a1-p1-c{channel}-s0' / 'result.npz'
    with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        om = json.loads(old_path.with_suffix('.json').read_text())
        nm = json.loads(new_path.with_suffix('.json').read_text())
        checks = {
            'old_embedded_metadata_matches_json': json.loads(str(old['metadata_json'])) == om,
            'new_embedded_metadata_matches_json': json.loads(str(new['metadata_json'])) == nm,
            'active_rows_exact': np.array_equal(old['active_rows'], new['active_rows']),
            'checkpoint_steps_exact': np.array_equal(old['frames_checkpoint_steps'], new['frames_checkpoint_steps']),
            'charged_rows_exact': np.array_equal(old['frames_cumulative_data_row_accesses'], new['frames_cumulative_data_row_accesses']),
            'gate_decisions_at_all_stored_checkpoints_exact': np.array_equal(gate(new), gate(old)),
            'both_horizons_400000': int(new['frames_checkpoint_steps'][-1]) == int(old['frames_checkpoint_steps'][-1]) == 400000,
            'old_status_complete': om['methods']['frames']['status'] == 'complete',
            'new_status_complete': nm['endpoints']['frames']['status'] == 'complete',
        }
        shared = {key: compare(new[key], old[key], TOLERANCES['shared_floating_arrays'])
                  for key in ['active_observations', 'initial_factor', 'truth_factor']}
        factor = compare(new['frames_final_factor'], old['frames_final_factor'], TOLERANCES['final_factor'])
        metrics = {
            key: compare(new['frames_' + key], old['frames_' + key], TOLERANCES['metrics_and_gaps'])
            for key in ['measurement_loss_mean', 'measurement_loss_sum', 'rmse', 'tp_violation',
                        'normalized_tp_violation', 'penalized_objective', 'process_fidelity',
                        'diagnostic_exact_smoothed_gap', 'diagnostic_estimated_gap']
        }
        schedules = {
            key: compare(new['frames_diagnostic_' + key], old['frames_diagnostic_' + key], TOLERANCES['schedules'])
            for key in ['checkpoint_smoothing_parameter', 'checkpoint_tp_weight_per_row_alpha',
                        'checkpoint_tp_weight_lambda', 'smoothing_parameter', 'momentum_weight',
                        'step_size', 'tp_weight_per_row_alpha', 'tp_weight_lambda']
        }
        for label, archive in [('old', old), ('new', new)]:
            expected_beta = archive['frames_diagnostic_smoothing_parameter'][
                np.minimum(archive['frames_checkpoint_steps'], 399999)]
            checks[label + '_checkpoint_beta_last_applied_at_terminal'] = np.array_equal(
                expected_beta, archive['frames_diagnostic_checkpoint_smoothing_parameter'])
        oc, nc = om['configuration'], nm['configuration']
        checks.update({
            'sampling_seed_matches': oc['sampling_seed'] == nm['sampling_seed'] == 300000 + 100 * channel,
            'initialization_seed_matches': oc['initialization_seed'] == 200000 + 10 * channel,
            'initialization_basis_declared_gell_mann': oc['initialization_basis'] == 'gell-mann',
            'penalty_schedule_configuration_matches': (
                oc['frames_beta0'] == 1. / (2. * nc['alpha0'])
                and oc['frames_smoothing_schedule']['exponent'] == nc['penalty_exponent']
                and oc['frames_smoothing_schedule']['offset'] == 1.),
            'radius_and_batch_match': oc['tau'] == nc['tau'] == 2. and oc['batch_size'] == nc['batch_size'] == 32,
        })
        batches = new['frames_diagnostic_batch_indices']
        rng = np.random.default_rng(oc['sampling_seed'])
        checks['new_full_sampled_plan_matches_declared_pcg64_stream'] = all(
            np.array_equal(batches[start:start + 10000], rng.integers(
                0, 128, size=(min(10000, 400000 - start), 32), dtype=np.int64))
            for start in range(0, 400000, 10000))
        del batches
        old_commit = om['diagnostic']['source_commit']
        new_commit = nm['local_calibration']['source_commit']
        source_checks = {}
        for name in ['frank_wolfe/algorithms/frames.py', 'paper/experiments/qpt_generate_data.py',
                     'paper/experiments/qpt_quiroga_sensing.py', 'paper/experiments/qpt_quiroga_initialization.py',
                     'paper/experiments/qpt_quiroga_frames.py', 'paper/experiments/quantum_process_tomography.py',
                     'paper/experiments/quantum_process_tomography_jax.py']:
            old_hash = hashlib.sha256(git_source(old_commit, name)).hexdigest()
            new_hash = hashlib.sha256(git_source(new_commit, name)).hexdigest()
            archived_old = om['diagnostic']['source_sha256'].get(name)
            archived_new = nm['local_calibration']['source_sha256'][name]
            source_checks[name] = {
                'old_git_sha256': old_hash, 'new_git_sha256': new_hash,
                'old_archived_sha256': archived_old, 'new_archived_sha256': archived_new,
                'unchanged': old_hash == new_hash,
                'old_archive_matches_git_when_recorded': archived_old is None or archived_old == old_hash,
                'new_archive_matches_git': archived_new == new_hash,
            }
        campaign_name = 'paper/experiments/qpt_quiroga_campaign.py'
        function_hashes = {}
        for commit in (old_commit, new_commit):
            source = git_source(commit, campaign_name).decode()
            function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'run_frames')
            function_hashes[commit] = hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest()
        checks['campaign_run_frames_function_unchanged'] = len(set(function_hashes.values())) == 1
        checks['relevant_source_files_unchanged_and_archived_hashes_valid'] = all(
            item['unchanged'] and item['old_archive_matches_git_when_recorded'] and item['new_archive_matches_git']
            for item in source_checks.values())
        checks = {name: bool(value) for name, value in checks.items()}
        passed = (all(checks.values()) and factor['pass'] and
                  all(result['pass'] for group in (shared, metrics, schedules) for result in group.values()))
        return {
            'channel': channel, 'status': 'pass' if passed else 'fail',
            'old_archive': {'path': str(old_path), 'npz_sha256': digest(old_path),
                            'json_sha256': digest(old_path.with_suffix('.json')), 'source_commit': old_commit,
                            'numpy_version': om['numpy_version']},
            'new_archive': {'path': str(new_path), 'npz_sha256': digest(new_path),
                            'json_sha256': digest(new_path.with_suffix('.json')), 'source_commit': new_commit,
                            'numpy_version': nm['environment']['numpy'], 'adapter_sha256': nm['adapter_sha256']},
            'checks': checks,
            'shared_floating_arrays': shared,
            'all_shared_arrays_bitwise_identical': checks['active_rows_exact'] and all(item['exact_equal'] for item in shared.values()),
            'final_factor': factor, 'metric_and_gap_differences': metrics, 'schedule_differences': schedules,
            'relevant_source_checks': source_checks,
            'campaign_run_frames_function_sha256': function_hashes,
            'checkpoint_count': int(len(new['frames_checkpoint_steps'])),
            'endpoint_gate_new': bool(gate(new)[-1]), 'endpoint_gate_old': bool(gate(old)[-1]),
            'endpoints': {label: {name: float(archive['frames_' + name][-1])
                                  for name in ['rmse', 'normalized_tp_violation', 'process_fidelity']}
                          for label, archive in [('old', old), ('new', new)]},
        }


if __name__ == '__main__':
    if OUT.exists():
        raise FileExistsError(OUT)
    channels = [audit_channel(channel) for channel in (0, 1)]
    report = {
        'schema': 'qpt_frames_backend_parity_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'pass' if all(item['status'] == 'pass' for item in channels) else 'fail',
        'comparison_kind': 'cross_environment_numerical_reproduction_of_incumbent_400000_update_runs',
        'qualification': (
            'Not a bitwise-identical-input backend isolation: Ruche NumPy 2.5.3 and Elite NumPy 2.4.6 '
            'regenerate the same seeded physical setup with floating-point differences below 1e-14. '
            'Exact checks are retained separately. No new optimization or timing-speedup claim in this audit.'),
        'tolerances': TOLERANCES,
        'checker_sha256': digest(__file__),
        'channels': channels,
        'worst_final_factor_abs_difference': max(item['final_factor']['max_abs_difference'] for item in channels),
        'worst_rmse_abs_difference_all_checkpoints': max(item['metric_and_gap_differences']['rmse']['max_abs_difference'] for item in channels),
        'worst_normalized_tp_abs_difference_all_checkpoints': max(item['metric_and_gap_differences']['normalized_tp_violation']['max_abs_difference'] for item in channels),
        'all_shared_arrays_bitwise_identical': all(item['all_shared_arrays_bitwise_identical'] for item in channels),
    }
    with OUT.open('x') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ['status', 'worst_final_factor_abs_difference',
                                                 'worst_rmse_abs_difference_all_checkpoints',
                                                 'worst_normalized_tp_abs_difference_all_checkpoints',
                                                 'all_shared_arrays_bitwise_identical']}))
    print(str(OUT))
