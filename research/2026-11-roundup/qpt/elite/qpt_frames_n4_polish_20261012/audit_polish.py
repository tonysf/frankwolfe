"""Independent, no-optimization audit of native n=4 two-phase FRAMES polish archives.

Physical checks reuse an immutable independently implemented Born/TP helper.
No project/optimizer modules are imported and no dense sensing bank is built.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
HELPER_PATH = Path('/home/tony/qpt_frames_local_followup_20261012/audit_followup.py')
HELPER_SHA256 = '8dcf5e7fab5573b8231671b1230f79baae0af0b4a8bb214cb74f725e0838b7e3'
PARENT = Path('/home/tony/qpt_frames_n4_tuning_20261012')
PARENT_NAMES = ('completion_manifest.json', 'audit_long.json', 'selection_long.json')
FROZEN = dict(config_id='penalty64-two-phase', alpha0=5e-5/64, penalty_exponent=.25,
              step_scale=4**.75, step_offset=4., step_exponent=.75,
              switch_update=48000, post_step_scale=4., post_step_offset=4., post_step_exponent=1.,
              rho_scale=4., rho_offset=8., rho_exponent=2/3, batch_size=32, tau=4.)

GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
PARAMETERS = ('alpha0', 'penalty_exponent', 'step_scale', 'step_offset', 'step_exponent',
              'rho_scale', 'rho_offset', 'rho_exponent', 'batch_size')
COMMON = ('measurement_loss_sum', 'measurement_loss_mean', 'rmse', 'tp_violation',
          'normalized_tp_violation', 'penalized_objective', 'process_fidelity')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


require(digest(HELPER_PATH) == HELPER_SHA256, 'Independent physical helper hash mismatch')
_spec = importlib.util.spec_from_file_location('qpt_frozen_independent_physics_helper', HELPER_PATH)
physical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physical)
close = physical.close
array_digest = physical.array_digest


def check_trial_configuration(c):
    for key in ('n_qubits', 'channel', 'initialization_replicate', 'sampling_replicate', 'initialization_seed',
                'sampling_seed', 'steps', 'metrics_every', 'charged_row_budget', 'switch_update', 'batch_size'):
        require(type(c[key]) is int, 'Integer field required: '+key)
    require(all(c[k] == v for k, v in FROZEN.items()), 'Frozen polish configuration changed')
    require(c['n_qubits'] == 4 and type(c['channel']) is int and c['channel'] in (0, 1, 2, 3), 'Wrong dimension/channel')
    require(type(c['initialization_replicate']) is int and c['initialization_replicate'] in (0, 1), 'Initialization replicate changed')
    require(type(c['sampling_replicate']) is int and c['sampling_replicate'] in (0, 1), 'Sampling replicate changed')
    require(c['initialization_seed'] == 200000+10*c['channel']+c['initialization_replicate'], 'Initialization seed changed')
    require(c['sampling_seed'] == 300000+100*c['channel']+c['sampling_replicate'], 'Sampling seed changed')
    require(c['steps'] == 400000 and c['metrics_every'] == 4000 and c['charged_row_budget'] == 12800000, 'Fresh-start budget changed')
    require(c['id'] == f"{c['config_id']}-c{c['channel']}-i{c['initialization_replicate']}-s{c['sampling_replicate']}", 'Trial ID changed')


def step_descriptor(c):
    return dict(type='two_phase_global_clock', switch_update=c['switch_update'],
                pre=dict(scale=c['step_scale'], offset=c['step_offset'], exponent=c['step_exponent'], cap=1.),
                post=dict(scale=c['post_step_scale'], offset=c['post_step_offset'], exponent=c['post_step_exponent'], cap=1.),
                clock='absolute_zero_based_optimizer_update', momentum_reset=False)


def schedule_histories(c):
    iteration = np.arange(c['steps'], dtype=np.float64)
    beta = 1. / (2. * c['alpha0']) / (iteration + 1.)**c['penalty_exponent']
    return {
        'smoothing_parameter': beta,
        'tp_weight_per_row_alpha': 1. / (2. * beta),
        'tp_weight_lambda': 8192. / (2. * beta),
        'step_size': np.minimum(1., np.where(iteration < c['switch_update'],
            c['step_scale'] / (iteration+c['step_offset'])**c['step_exponent'],
            c['post_step_scale'] / (iteration+c['post_step_offset'])**c['post_step_exponent'])),
        'momentum_weight': np.minimum(1., c['rho_scale'] / (iteration + c['rho_offset'])**c['rho_exponent']),
    }


def check_sources(plan, root):
    require(plan['source_commit'] == COMMIT, 'Unexpected repository revision')
    require(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT, 'Checkout revision changed')
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip(), 'Checkout is dirty')
    tracked = subprocess.check_output(['git', '-C', str(REPO), 'ls-tree', '-r', '--name-only', COMMIT,
                                       '--', 'paper', 'frank_wolfe'], text=True).splitlines()
    require(set(plan['source_sha256']) == {name for name in tracked if name.endswith('.py')}, 'Incomplete source manifest')
    for name, sha in plan['source_sha256'].items():
        content = subprocess.check_output(['git', '-C', str(REPO), 'show', COMMIT + ':' + name])
        require(hashlib.sha256(content).hexdigest() == sha and digest(REPO / name) == sha, 'Repository source mismatch: ' + name)
    require(set(plan['parent_pins']) == {str(PARENT/name) for name in PARENT_NAMES}, 'Wrong prior campaign evidence')
    for path, sha in plan['parent_pins'].items():
        require(digest(path) == sha, 'Immutable prior campaign file changed')
    manifest = read(PARENT/'completion_manifest.json')
    require(manifest['status'] == 'complete' and manifest['source_commit'] == COMMIT
            and manifest['selected_config_ids'] == [], 'Prior round incomplete/eligible')
    for name, sha in manifest['evidence_sha256'].items():
        require(Path(name).name == name and digest(PARENT/name) == sha, 'Prior sealed evidence changed')
    require(read(PARENT/'audit_long.json')['phase_complete_and_valid'], 'Prior long audit incomplete')
    selection = read(PARENT/'selection_long.json')
    require(selection['selected_config_ids'] == [] and selection['confirmation_status'] == 'skipped_no_joint_gate_eligible_configuration', 'Conditional polish not authorized by prior outcome')
    require(selection['audit_sha256'] == digest(PARENT/'audit_long.json'), 'Prior selection audit changed')
    require(set(plan['harness_sha256']) == {'run_polish.py', 'run_batch.py', 'prepare_polish.py'}, 'Incomplete harness manifest')
    for name, sha in plan['harness_sha256'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'Unsafe harness path')
        require(digest(root / name) == sha, 'Frozen harness changed: ' + name)
    return len(plan['source_sha256'])


def check_trial(root, plan_path, plan, c):
    folder = root / plan['phase'] / c['id']
    missing = [name for name in ('started.json', 'result.json', 'result.npz', 'summary.json') if not (folder / name).is_file()]
    if missing:
        return {'id': c['id'], 'config_id': c['config_id'], 'valid': False, 'archive_complete': False,
                'missing': missing, 'failure_record_present': (folder / 'failure.json').exists()}
    require(not (folder / 'failure.json').exists(), 'Ambiguous completed archive plus infrastructure failure record')
    started, meta, summary = [read(folder / name) for name in ('started.json', 'result.json', 'summary.json')]
    with np.load(folder / 'result.npz', allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    require(json.loads(str(z['metadata_json'])) == meta, 'NPZ/JSON metadata mismatch')
    for kind in ('json', 'npz'):
        require(summary[kind + '_sha256'] == digest(folder / ('result.' + kind)), 'Result hash mismatch')
    require(meta['format'] == 'qpt_quiroga_campaign', 'Only native structured NumPy archives accepted')
    require(summary['trial'] == started['trial'] == meta['polish_configuration'] == c, 'Trial configuration mismatch')
    provenance = meta['polish_provenance']
    for key in started:
        require(provenance[key] == started[key], 'Started/archive provenance mismatch: ' + key)
    require(started['plan_sha256'] == digest(plan_path), 'Plan hash mismatch')
    for key in ('source_commit', 'source_sha256', 'harness_sha256', 'parent_pins'):
        require(started[key] == plan[key], 'Started/plan mismatch: ' + key)
    require(started['runner_sha256'] == plan['harness_sha256']['run_polish.py'], 'Runner hash mismatch')
    require(started['host'].split('.')[0] == 'elite' and started['execution'] == 'local_cpu', 'Execution was not local Elite CPU')
    require(started['cpu_affinity'] and set(started['cpu_affinity']) <= {0, 1} and started['nice'] >= 10, 'Local CPU limits violated')
    require(started['threads'] == {key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'Thread limits violated')
    require(started['split'] == plan['phase'] and started['charged_row_budget'] == c['charged_row_budget'], 'Started phase/budget mismatch')
    prereqs = [] if plan['phase'] == 'calibration' else [
        'selection_calibration.json', 'audit_calibration.json', 'plan_calibration.json', 'inventory_calibration.json']
    require(started['execution_prerequisites_sha256'] == {name: digest(root/name) for name in prereqs}, 'Execution prerequisites changed')
    design = meta['design']
    for key, expected in {'n_qubits': 4, 'd': 16, 'rows': 8192, 'probes': 256, 'outcomes': 32}.items():
        require(design[key] == expected, 'Design metadata mismatch: ' + key)
    close(design['povm_a'], 1/37, 'POVM a', rtol=0, atol=1e-15)
    close(design['povm_b'], 1/37, 'POVM b', rtol=0, atol=1e-15)
    require(meta['observation_mode'] == 'noiseless' and meta['active_row_count'] == 8192, 'Data regime changed')
    require(meta['initialization']['basis'] == 'gell-mann', 'Initialization basis changed')
    require(meta['data_metadata']['channel_seed'] == c['channel'], 'Channel seed metadata changed')
    backend = meta['polish_backend']
    require(backend['name'] == 'structured_numpy' and backend['precision'] == 'complex128', 'Unexpected backend')
    require(backend['dense_sensing_bank'] is False and backend['resume'] is False, 'Dense bank/resumption not authorized')
    require(backend['observed_batch_indices_archived'] is False, 'Native sampling provenance overstated')
    require(backend['repository_runner'] == 'paper.experiments.qpt_quiroga_campaign.run_campaign', 'Native runner changed')
    require('frames_diagnostic_batch_indices' not in z, 'Unexpected realized batch archive in native runner')
    conf = meta['configuration']
    for key, expected in {
        'methods': ['frames'], 'rank': 1, 'initialization_basis': 'gell-mann',
        'initialization_seed': c['initialization_seed'], 'sampling_seed': c['sampling_seed'],
        'frames_steps': c['steps'], 'frames_metrics_frequency': c['metrics_every'],
        'batch_size': c['batch_size'], 'tau': 4., 'frames_penalty_mode': 'independent_schedule',
    }.items():
        require(conf[key] == expected, 'Native configuration mismatch: ' + key)
    close(conf['frames_beta0'], 1/(2*c['alpha0']), 'Initial beta')
    close(conf['frames_initial_tp_weight_per_row_alpha'], c['alpha0'], 'Mean-loss alpha metadata')
    close(conf['frames_initial_tp_weight_lambda'], 8192*c['alpha0'], 'Summed-loss lambda metadata')
    for key, scale, offset, exponent, cap in (
        ('frames_smoothing_schedule', 1/(2*c['alpha0']), 1., c['penalty_exponent'], None),
        ('rho_schedule', c['rho_scale'], c['rho_offset'], c['rho_exponent'], 1.)):
        require(conf[key] == dict(type='PowerSchedule', scale=scale, offset=offset, exponent=exponent, cap=cap), 'Native schedule metadata mismatch: ' + key)
    require(conf['step_size_schedule']['type'] == 'callable' and conf['step_size_schedule']['qualname'] == 'TwoPhaseStep'
            and conf['step_size_schedule']['module'] in ('run_polish', '__main__'), 'Native two-phase callable metadata changed')
    require(backend['step_schedule'] == step_descriptor(c), 'Two-phase step descriptor changed')
    require(backend['momentum_reset'] is False, 'Momentum reset is not authorized')
    for name, sha in meta['source_sha256'].items():
        path = name if '/' in name else 'paper/experiments/' + name
        require(plan['source_sha256'][path] == sha, 'Native source differs from pinned source')
    for name in ('active_rows', 'active_observations', 'initial_factor', 'truth_factor'):
        require(meta[name + '_sha256'] == array_digest(z[name], shape_tag=True), 'Native array hash mismatch: ' + name)
    require(np.array_equal(z['active_rows'], np.arange(8192)), 'Incomplete/noncanonical sensing rows')
    require(z['active_observations'].shape == (8192,), 'Wrong observation shape')
    for name in ('initial_factor', 'truth_factor', 'frames_final_factor'):
        require(z[name].shape == (256, 1) and np.all(np.isfinite(z[name])), 'Wrong/nonfinite factor: ' + name)
    close(z['truth_factor'], physical.independent_truth(c['channel'], 16), 'Independent truth')
    close(z['initial_factor'], physical.independent_initializer(c['initialization_seed'], 16), 'Independent initializer')
    close(z['active_observations'], physical.born_values(z['truth_factor'], 16), 'Independent Born observations')
    endpoint = meta['methods']['frames']
    status = endpoint['status']
    require(status in ('complete', 'failed') and summary['status'] == status, 'Conflicting method status')
    steps, times = z['frames_checkpoint_steps'], z['frames_optimizer_seconds']
    require(steps.ndim == 1 and np.issubdtype(steps.dtype, np.integer) and len(steps) > 0
            and steps[0] == 0 and np.all(np.diff(steps) > 0), 'Invalid checkpoints')
    require(np.array_equal(z['frames_cumulative_data_row_accesses'], steps*c['batch_size']), 'Wrong sampled row charges')
    require(times.shape == steps.shape and np.all(np.isfinite(times)) and np.all(times >= 0)
            and np.all(np.diff(times) >= 0), 'Invalid native checkpoint timings')
    for name in COMMON:
        require(z['frames_' + name].shape == steps.shape and np.all(np.isfinite(z['frames_' + name])), 'Invalid metric history: ' + name)
    close(z['frames_measurement_loss_mean'], .5*z['frames_rmse']**2, 'Mean-loss history')
    close(z['frames_measurement_loss_sum'], 8192*z['frames_measurement_loss_mean'], 'Sum-loss history')
    close(z['frames_normalized_tp_violation'], z['frames_tp_violation']/4, 'Normalized TP history')
    if status == 'complete':
        require(np.array_equal(steps, np.arange(0, c['steps'] + 1, c['metrics_every'])), 'Incomplete checkpoint grid')
        histories = schedule_histories(c)
        for name, values in histories.items():
            close(z['frames_diagnostic_' + name], values, 'Schedule ' + name)
        checkpoint_beta = histories['smoothing_parameter'][np.minimum(steps, c['steps'] - 1)]
        for name, values in {'checkpoint_smoothing_parameter': checkpoint_beta,
                             'checkpoint_tp_weight_per_row_alpha': 1/(2*checkpoint_beta),
                             'checkpoint_tp_weight_lambda': 8192/(2*checkpoint_beta)}.items():
            close(z['frames_diagnostic_' + name], values, 'Checkpoint ' + name)
        close(z['frames_penalized_objective'], z['frames_measurement_loss_sum']
              + 8192/(2*checkpoint_beta)*z['frames_tp_violation']**2, 'Penalized objective history')
        for name, length in [('estimated_gap', c['steps']), ('exact_smoothed_gap', len(steps))]:
            require(z['frames_diagnostic_' + name].shape == (length,)
                    and np.all(np.isfinite(z['frames_diagnostic_' + name])), 'Invalid gap history')
        require(np.linalg.norm(z['frames_final_factor'], 2) <= 4. + 3e-12, 'Final point outside radius')
        alpha_end = 1/(2*checkpoint_beta[-1])
    else:
        require(bool(endpoint['failure_reason']) and np.array_equal(steps, [0]), 'Invalid native failed-run archive')
        require(np.array_equal(z['frames_final_factor'], z['initial_factor']), 'Failed native archive invented progress')
        alpha_end = c['alpha0']
    metric = physical.independent_metrics(z['frames_final_factor'], z['truth_factor'], z['active_observations'], alpha_end, 16)
    initial = physical.independent_metrics(z['initial_factor'], z['truth_factor'], z['active_observations'], c['alpha0'], 16)
    for name in COMMON:
        close(z['frames_' + name][-1], metric[name], 'Independent endpoint ' + name)
        close(endpoint[name], metric[name], 'JSON endpoint ' + name)
        close(z['frames_' + name][0], initial[name], 'Independent initial ' + name)
    passed, deficit = physical.gate_and_deficit(metric, status)
    require(summary['gate_pass'] == provenance['endpoint_gate'] == passed, 'Physical gate mismatch')
    close(summary['deficit'], deficit, 'Summary deficit')
    close(provenance['gate_deficit'], deficit, 'Provenance deficit')
    for name, key in [('rmse', 'rmse'), ('normalized_tp', 'normalized_tp_violation'),
                      ('fidelity', 'process_fidelity'), ('relative_rmse', 'relative_rmse')]:
        close(summary[name], metric[key], 'Summary ' + name)
    close(summary['outcome_scaled_rmse'], 32*metric['rmse'], 'Outcome-scaled RMSE')
    close(summary['optimizer_seconds'], times[-1], 'Summary optimizer time')
    require(np.isfinite(summary['total_seconds']) and summary['total_seconds'] >= times[-1]
            and np.isfinite(summary['peak_rss_mib']) and summary['peak_rss_mib'] > 0, 'Invalid resource measurements')
    completed, charged = int(steps[-1]), int(steps[-1]*c['batch_size'])
    require(endpoint['completed_steps'] == summary['completed_steps'] == completed, 'Completed step mismatch')
    require(endpoint['cumulative_data_row_accesses'] == summary['charged_rows'] == charged, 'Endpoint work mismatch')
    require(summary['charged_row_budget'] == c['charged_row_budget'] and charged <= c['charged_row_budget'], 'Budget mismatch')
    passing = ((z['frames_rmse'] <= 1e-6) & (z['frames_normalized_tp_violation'] <= .001)
               & (z['frames_process_fidelity'] >= .99))
    first, persistent = physical.gate_crossings(steps, passing)
    for key, expected in [('first_recorded_gate_step', first), ('persistent_recorded_gate_step', persistent)]:
        require(summary[key] == provenance[key] == expected, 'Gate crossing mismatch: ' + key)
    prefix = check_prefix(z, c, status) if plan['phase'] == 'calibration' else {'required': False, 'reason': 'different channel/initialization cohort'}
    return {
        'prefix_check': prefix,
        'id': c['id'], 'config_id': c['config_id'], 'n_qubits': 4, 'channel': c['channel'],
        'initialization_replicate': c['initialization_replicate'], 'sampling_replicate': c['sampling_replicate'],
        'archive_complete': True, 'valid': True, 'status': status, 'gate_pass': passed,
        'deficit': deficit, 'metrics': metric, 'completed_steps': completed, 'charged_rows': charged,
        'charged_row_budget': c['charged_row_budget'], 'optimizer_seconds': float(times[-1]),
        'first_recorded_gate_step': first, 'persistent_recorded_gate_step': persistent,
        'json_sha256': digest(folder/'result.json'), 'npz_sha256': digest(folder/'result.npz'),
        'summary_sha256': digest(folder/'summary.json'), 'started_sha256': digest(folder/'started.json'),
        'paired_arrays_sha256': {name: array_digest(z[name]) for name in ('active_rows', 'active_observations', 'initial_factor', 'truth_factor')},
        'sampling_verification': 'Declared seed and unchanged native runner verified; realized batches not archived.',
    }


def compare_prefix(current, previous, cutoff=48000):
    """Compare physical metrics and applied schedules, never terminal beta labels."""
    for key in ('initial_factor', 'truth_factor', 'active_rows', 'active_observations'):
        require(np.array_equal(current[key], previous[key]), 'Prefix paired data/init mismatch: '+key)
    steps = previous['frames_checkpoint_steps']
    require(np.array_equal(steps, np.arange(0, cutoff+1, 4000)), 'Prior screen checkpoint grid changed')
    newsteps = current['frames_checkpoint_steps']
    indices = np.searchsorted(newsteps, steps)
    require(np.all(indices < len(newsteps)) and np.array_equal(newsteps[indices], steps), 'Polish prefix checkpoints missing')
    differences = {}
    for name in COMMON:
        if name == 'penalized_objective':
            continue
        x, y = current['frames_'+name][indices], previous['frames_'+name]
        require(np.array_equal(x, y) and np.all(np.isfinite(x)), 'Physical metric prefix differs: '+name)
        differences[name] = float(np.max(abs(x-y)))
    for name in ('step_size', 'momentum_weight', 'smoothing_parameter', 'tp_weight_per_row_alpha', 'tp_weight_lambda'):
        key = 'frames_diagnostic_'+name
        require(previous[key].shape == (cutoff,) and np.array_equal(current[key][:cutoff], previous[key]), 'Applied schedule prefix differs: '+name)
    return dict(required=True, verified=True, cutoff_steps=cutoff, compared_checkpoint_count=len(steps),
                physical_metrics_match=True, max_abs_metric_differences=differences,
                schedule_prefix_match=True, paired_data_and_initial_factor_identical=True,
                qualification='Physical checkpoints and applied schedules only; intermediate factors/realized batches not archived. Penalized objective excluded: prior terminal versus new interior beta label differs.')


def check_prefix(current, c, status):
    if status != 'complete':
        return dict(required=True, verified=False, cutoff_steps=48000,
                    reason='Native failure archives the initial factor only; a 48k prefix cannot be claimed.')
    manifest = read(PARENT/'completion_manifest.json')
    require(manifest['evidence_sha256']['audit_screen.json'] == digest(PARENT/'audit_screen.json'), 'Prefix audit unsealed/changed')
    report = read(PARENT/'audit_screen.json')
    require(report['phase_complete_and_valid'], 'Prefix source audit incomplete')
    candidates = [r for r in report['trials'] if r['config_id'] == 'penalty64-stepslow' and r['channel'] == c['channel']]
    require(len(candidates) == 1, 'Prefix source trial missing/duplicate')
    row = candidates[0]; folder = PARENT/'screen'/row['id']
    require(row['valid'] and row['status'] == 'complete', 'Prefix source method incomplete')
    for name, key in [('started.json', 'started_sha256'), ('summary.json', 'summary_sha256'),
                      ('result.json', 'json_sha256'), ('result.npz', 'npz_sha256')]:
        require(digest(folder/name) == row[key], 'Previously audited prefix artifact changed')
    old = read(folder/'started.json')['trial']
    for key in (*PARAMETERS, 'tau', 'channel', 'initialization_seed', 'sampling_seed', 'n_qubits',
                'initialization_replicate', 'sampling_replicate', 'metrics_every'):
        require(old[key] == c[key], 'Prior prefix configuration differs: '+key)
    require(old['steps'] == 48000 and old['charged_row_budget'] == 1536000, 'Prior screen horizon changed')
    with np.load(folder/'result.npz', allow_pickle=False) as archive:
        result = compare_prefix(current, archive)
    result.update(archived_parent_npz_sha256=row['npz_sha256'], archived_parent_npz_path=str(folder/'result.npz'),
                  archived_parent_audit_sha256=digest(PARENT/'audit_screen.json'))
    return result


def check_plan(plan):
    require(plan['scope'] == 'local_frames_n4_polish' and plan['phase'] in ('calibration', 'confirmation'), 'Wrong polish scope/phase')
    require(plan['source_commit'] == COMMIT and plan['gate'] == GATE, 'Source/gate changed')
    require(plan['reserved_final_channels'] == list(range(1000, 1020)), 'Final held-out split changed')
    trials = plan['trials']
    require(len(trials) == 2 and len({c['id'] for c in trials}) == 2, 'Incomplete/duplicate trial inventory')
    for c in trials:
        check_trial_configuration(c)
    axes = [(c['channel'], c['initialization_replicate'], c['sampling_replicate']) for c in trials]
    expected = [(0, 0, 0), (1, 0, 0)] if plan['phase'] == 'calibration' else [(2, 1, 1), (3, 1, 1)]
    require(sorted(axes) == expected, 'Phase cohort mismatch')
    require(plan['total_charged_row_budget'] == 25600000, 'Full fresh-start work not charged')
    require(plan['schedule'] == dict(clock='global zero-based optimizer update', switch_update=48000,
        before='(4/(t+4))**0.75', after='4/(t+4)', momentum_reset=False,
        penalty_clock_reset=False, tp_projection=False, resume=False), 'Piecewise schedule changed/reset/projected')
    r = plan['resources']
    require(r['max_workers'] <= 2 and set(r['cpu_affinity']) <= {0, 1} and r['nice'] >= 10
            and r['trial_wall_seconds'] <= 1200 and r['batch_wall_seconds'] <= 2400, 'Resource allocation expanded')
    if plan['phase'] == 'confirmation':
        require(plan['selected_config_ids'] == [FROZEN['config_id']] and 'selection' in plan, 'Frozen confirmation selection missing')


def check_confirmation_selection(plan, root):
    if plan['phase'] == 'calibration':
        return False
    path = root/'selection_calibration.json'
    require(plan['selection'] == {'path': str(path), 'sha256': digest(path)}, 'Confirmation selection changed')
    selection = read(path); report = read(root/'audit_calibration.json')
    require(selection['audit_sha256'] == digest(root/'audit_calibration.json')
            and selection['calibration_plan_sha256'] == digest(root/'plan_calibration.json'), 'Selection evidence changed')
    require(selection['selected_config_ids'] == [FROZEN['config_id']] and selection['confirmation_status'] == 'planned', 'Calibration not eligible')
    require(report['phase_complete_and_valid'] and report['auditor_sha256'] == digest(__file__)
            and report['plan_sha256'] == digest(root/'plan_calibration.json')
            and report['batch_inventory_sha256'] == digest(root/'inventory_calibration.json'), 'Calibration audit provenance changed')
    require(len(report['trials']) == 2 and {r['channel'] for r in report['trials']} == {0, 1}, 'Calibration cohort incomplete')
    for r in report['trials']:
        require(r['valid'] and r['status'] == 'complete' and r['gate_pass'] and r['prefix_check']['verified'], 'Calibration eligibility/prefix gate failed')
        for name, key in [('started.json', 'started_sha256'), ('summary.json', 'summary_sha256'),
                          ('result.json', 'json_sha256'), ('result.npz', 'npz_sha256')]:
            require(digest(root/'calibration'/r['id']/name) == r[key], 'Calibration artifact changed')
    return True


def audit(plan_path, root=None):
    plan_path = Path(plan_path).resolve(); root = plan_path.parent if root is None else Path(root)
    plan = read(plan_path); check_plan(plan)
    source_count = check_sources(plan, root)
    selection_verified = check_confirmation_selection(plan, root)
    results = []
    for c in plan['trials']:
        try:
            row = check_trial(root, plan_path, plan, c)
        except Exception as error:
            row = dict(id=c['id'], config_id=c['config_id'], valid=False, archive_complete=False,
                       audit_error=type(error).__name__+': '+str(error))
        results.append(row)
    inventory_path = root/f"inventory_{plan['phase']}.json"
    inventory_valid, inventory_error = False, None
    try:
        inventory = read(inventory_path)
        require(inventory['phase'] == plan['phase'] and inventory['plan_sha256'] == digest(plan_path), 'Inventory plan/phase mismatch')
        require(set(inventory['trials']) == {c['id'] for c in plan['trials']} and inventory['total_charged_row_budget'] == 25600000, 'Inventory count/work mismatch')
        for r in results:
            require(r['valid'], 'Invalid/incomplete archived trial')
            state = inventory['trials'][r['id']]
            expected = ('finished', 0) if r['status'] == 'complete' else ('method_failure', 2)
            require((state['state'], state['exit_code']) == expected, 'Inventory method outcome differs')
            require(state['charged_rows'] == r['charged_rows'] and state['charged_row_budget'] == 12800000
                    and state['gate_pass'] == r['gate_pass'], 'Inventory work/gate differs')
        expected = 'complete' if all(r['status'] == 'complete' for r in results) else 'finished_with_failures'
        require(inventory['status'] == expected, 'Incomplete inventory terminal state')
        inventory_valid = True
    except Exception as error:
        inventory_error = type(error).__name__+': '+str(error)
    return dict(schema='n4_frames_polish_independent_audit_v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_path=str(plan_path), plan_sha256=digest(plan_path), auditor_sha256=digest(__file__),
        physical_helper=dict(path=str(HELPER_PATH), sha256=HELPER_SHA256), source_commit=COMMIT, source_file_count=source_count,
        expected_trials=len(plan['trials']), validated_trials=sum(r['valid'] for r in results),
        passing_trials=sum(r.get('gate_pass', False) for r in results),
        complete_method_trials=sum(r.get('status') == 'complete' for r in results),
        failed_method_trials=sum(r.get('status') == 'failed' for r in results),
        prefix_checks_verified=sum(r.get('prefix_check', {}).get('verified', False) for r in results),
        confirmation_selection_verified=selection_verified,
        phase_complete_and_valid=inventory_valid and all(r['valid'] for r in results),
        batch_inventory_valid=inventory_valid, batch_inventory_error=inventory_error,
        batch_inventory_sha256=digest(inventory_path) if inventory_path.exists() else None, trials=results,
        qualification='Independent physical/source/step-switch audit. Calibration prefixes compare archived physical metrics and applied schedules, not unarchived intermediate factors or realized batches. Failed native traces may omit partial work. Separate adaptive follow-up, not untouched final evaluation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); report = audit(args.plan)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')
    raise SystemExit(0 if report['phase_complete_and_valid'] else 2)



