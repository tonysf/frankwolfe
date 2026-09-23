"""Validate downloaded completed cases from scalar evidence; never runs QPT."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inspection', type=Path)
    parser.add_argument('cases', nargs='+')
    args = parser.parse_args()
    inspection = read(args.inspection)
    assert inspection['accounting']['returncode'] == 0
    accounting = {}
    for line in inspection['accounting']['stdout'].splitlines():
        fields = line.split('|')
        accounting[fields[1]] = dict(zip(
            ['raw_job_id', 'job_id', 'state', 'exit_code', 'elapsed', 'node', 'start', 'end'], fields))
    plan = read(ROOT / 'plan.json')
    deployed = read(ROOT / 'deployment_sha256.json')
    assert sha(ROOT / 'plan.json') == deployed['plan.json']
    manifest = read(ROOT / 'manifest.json')
    cases = {case['name']: case for case in plan['cases']}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = {}
    for name in args.cases:
        case = cases[name]
        directory = ROOT / name
        checksums = read(directory / 'checksums.json')
        for filename, expected in checksums.items():
            assert Path(filename).name == filename
            assert sha(directory / filename) == expected, (name, filename)
        status = read(directory / 'status.json')
        summary = read(directory / 'summary.json')
        provenance = read(directory / 'provenance.json')
        verification = read(directory / 'verification.json')
        baseline = read(directory / 'baseline_traces.json')
        metadata = summary['metadata']
        job_id = '1925109' if case['group'] == 'small' else '1925110_' + str(case['array_task'])
        scheduler = accounting[job_id]
        assert scheduler['state'] == 'COMPLETED' and scheduler['exit_code'] == '0:0'
        assert status['status'] == 'completed' and status['name'] == name
        assert str(status['job_id']) == scheduler['raw_job_id']
        assert verification['status'] == 'verified'
        assert verification['script_sha256'] == deployed['verify_saved_factors.py']
        assert verification['result_sha256'] == checksums['final.npz']
        assert provenance['driver_sha256'] == deployed['run_campaign.py']
        assert provenance['case'] == case
        assert provenance['source_sha256'] == plan['source_sha256']
        assert provenance['base_commit'] == plan['base_commit']
        for item in (provenance, verification):
            assert item['data_sha256'] == case['data_sha256']
        assert metadata['initial_factor_sha256'] == verification['initial_factor_sha256'] == case['initial_factor_sha256']
        assert metadata['start_step'] == 0 and metadata['resume_from'] is None
        assert metadata['fidelity_target'] is None
        assert metadata['n_steps'] == metadata['requested_n_steps'] == verification['step'] == case['steps']
        for key in ('batch_size', 'n_qubits', 'tau', 'rank', 'measurement_backend', 'chunk_steps', 'metric_batch_size', 'prefetch'):
            assert metadata[key] == case[key], (name, key)
        for key in ('precision', 'metric_mode', 'metric_samples', 'metric_seed', 'initialization_seed', 'sampling_seed'):
            assert metadata[key] == plan['shared'][key], (name, key)
        assert metadata['noise_std'] == metadata['data_metadata']['noise_std'] == case['noise_std']
        assert metadata['data_metadata']['channel_seed'] == metadata['data_metadata']['noise_seed'] == 0
        command = provenance['command']
        assert '--resume-from' not in command and '--fidelity-target' not in command
        expected_flags = {
            'step-exponent': .75, 'smoothing-exponent': .25, 'rho-exponent': .5,
            'step-scale': case['step_scale'], 'step-offset': case['step_offset'],
            'smoothing-scale': case['smoothing_scale'], 'smoothing-offset': 1,
            'rho-scale': 2, 'rho-offset': 4, 'steps': case['steps']}
        for flag, expected in expected_flags.items():
            assert float(command[command.index('--' + flag) + 1]) == expected
        assert summary['schedule_verification'] == 'All applied scalar values match requested exponents, scales, offsets and caps.'
        trajectory = summary['trajectory']
        assert trajectory[0]['step'] == 0 and trajectory[-1]['step'] == case['steps']
        assert all(a['step'] < b['step'] for a, b in zip(trajectory, trajectory[1:]))
        assert all(math.isfinite(value) for point in trajectory for value in point.values())
        final = trajectory[-1]
        assert status['final'] == final
        close(final['fidelity'], verification['fidelity_numpy'])
        close(final['fidelity'], verification['fidelity_matrix_numpy'])
        close(final['tp_violation'], verification['tp_violation_numpy'])
        old_points = {}
        previous_segment = None
        for segment in baseline['segments']:
            old_metadata = segment['metadata']
            if previous_segment is None:
                assert old_metadata['start_step'] == 0
                assert old_metadata['initial_factor_sha256'] == case['initial_factor_sha256']
            else:
                # The old report may concatenate genuine checkpoint resumes.
                # Their initial-factor hash describes the resumed factor, not U0.
                old_end = previous_segment['trajectory'][-1]
                old_start = segment['trajectory'][0]
                assert old_metadata['start_step'] == old_start['step'] == old_end['step']
                assert old_metadata['resume_from'] == str(Path(previous_segment['path']).with_name('latest_restart.npz'))
                for key in ('fidelity', 'measurement_loss', 'tp_violation'):
                    close(old_start[key], old_end[key])
            for point in segment['trajectory']:
                old_points[point['step']] = point
            previous_segment = segment
        old = old_points[case['steps']]
        close(old_points[0]['fidelity'], trajectory[0]['fidelity'])
        close(old_points[0]['tp_violation'], trajectory[0]['tp_violation'])
        if case['group'] == 'small':
            configuration = baseline['configuration']
            assert configuration['step_exponent'] == 1 and configuration['rho_exponent'] == .6
            assert configuration['smoothing_exponent'] == .25
            for key in ('batch_size', 'steps', 'tau', 'rank', 'smoothing_scale', 'step_scale', 'step_offset'):
                assert configuration[key] == case[key]
            assert baseline['segments'][0]['metadata']['batch_symbols_sha256'] == metadata['batch_symbols_sha256']
        record = dict(case=name, status='verified_and_backed_up', checked_at_utc=now,
                      scheduler_state='COMPLETED', scheduler_exit='0:0', job_id=job_id,
                      raw_job_id=scheduler['raw_job_id'], steps=case['steps'],
                      batch_size=case['batch_size'], sigma=case['noise_std'], new=final, baseline=old,
                      fidelity_numpy=verification['fidelity_numpy'], tp_numpy=verification['tp_violation_numpy'],
                      fidelity_change_percentage_points=100 * (final['fidelity'] - old['fidelity']),
                      best_new_checkpoint=max(trajectory, key=lambda x: x['fidelity']),
                      verified_file_count=len(checksums), factor_sha256=checksums['final.npz'],
                      restart_sha256=checksums['latest_restart.npz'],
                      optimizer_seconds=metadata['optimizer_seconds'], runner_seconds=metadata['wall_seconds'])
        records[name] = record
    # Persist only after every requested case passes validation.
    for name, record in records.items():
        save(ROOT / name / 'local_validation.json', record)
        print('{}: F {:.6f}% -> {:.6f}%; TP {:.8g} -> {:.8g}'.format(
            name, 100 * record['baseline']['fidelity'], 100 * record['new']['fidelity'],
            record['baseline']['tp_violation'], record['new']['tp_violation']))
    manifest.setdefault('completed_cases', {}).update(records)
    manifest['completed_count'] = len(manifest['completed_cases'])
    manifest['status'] = 'results_verified' if manifest['completed_count'] == len(cases) else 'partially_completed'
    manifest['latest_inspection_file'] = args.inspection.name
    manifest['last_checked_at_utc'] = now
    manifest['last_scheduler_check'] = dict(checked_at_utc=inspection['checked_at_utc'], **accounting)
    save(ROOT / 'manifest.json', manifest)
    print('Verified total: {}/{}'.format(manifest['completed_count'], len(cases)))


if __name__ == '__main__':
    main()
