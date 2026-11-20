"""Independent, no-optimization audit of native n=4 FRAMES tuning archives.

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
PARENT = Path('/home/tony/qpt_frames_local_followup_20261012')
PARENT_PINS = {
    str(PARENT / 'completion_manifest.json'): '5ef64bf0f5932fb5a5751e83090f4ea8de2fef7986315cd8c8d4941869b1c973',
    str(PARENT / 'audit_transfer.json'): '99701941f77f2b5fc8090fc73671d225f4ff60e8cf4d034a34e3ad6c48fcbe87',
}
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


def check_trial_configuration(c, configs):
    require(type(c['n_qubits']) is int and c['n_qubits'] == 4, 'Only n=4 is in scope')
    require(type(c['channel']) is int and 0 <= c['channel'] < 100, 'Forbidden channel')
    for key, limit in [('initialization_replicate', 10), ('sampling_replicate', 100)]:
        require(type(c[key]) is int and 0 <= c[key] < limit, 'Invalid ' + key)
    require(c['initialization_seed'] == 200000 + 10 * c['channel'] + c['initialization_replicate'], 'Initialization seed mapping changed')
    require(c['sampling_seed'] == 300000 + 100 * c['channel'] + c['sampling_replicate'], 'Sampling seed mapping changed')
    for key in ('steps', 'metrics_every', 'batch_size', 'charged_row_budget'):
        require(type(c[key]) is int and c[key] > 0, 'Invalid ' + key)
    require(c['steps'] % c['metrics_every'] == 0, 'Metric cadence does not divide horizon')
    require(c['charged_row_budget'] == c['steps'] * c['batch_size'], 'Charged row budget mismatch')
    require(c['tau'] == 4., 'n=4 radius must be 4')
    require(c['config_id'] in configs, 'Configuration absent from frozen spec')
    for key in PARAMETERS:
        require(c[key] == configs[c['config_id']][key], 'Frozen candidate changed: ' + key)
    for key in PARAMETERS[:-1]:
        require(np.isfinite(c[key]) and c[key] > 0, 'Invalid/increasing-penalty schedule: ' + key)
    require(Path(c['id']).name == c['id'] and c['id'] not in ('.', '..'), 'Unsafe trial id')


def schedule_histories(c):
    iteration = np.arange(c['steps'], dtype=np.float64)
    beta = 1. / (2. * c['alpha0']) / (iteration + 1.)**c['penalty_exponent']
    return {
        'smoothing_parameter': beta,
        'tp_weight_per_row_alpha': 1. / (2. * beta),
        'tp_weight_lambda': 8192. / (2. * beta),
        'step_size': np.minimum(1., c['step_scale'] / (iteration + c['step_offset'])**c['step_exponent']),
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
    require(plan['parent_pins'] == PARENT_PINS, 'Prior campaign pins changed')
    for path, sha in PARENT_PINS.items():
        require(digest(path) == sha, 'Immutable prior campaign file changed')
    require(read(PARENT / 'audit_transfer.json')['phase_complete_and_valid'], 'Prior transfer audit was incomplete')
    require(set(plan['harness_sha256']) == {'run_tuning.py', 'run_batch.py', 'make_plans.py'}, 'Incomplete harness manifest')
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
    require(summary['trial'] == started['trial'] == meta['tuning_configuration'] == c, 'Trial configuration mismatch')
    provenance = meta['tuning_provenance']
    for key in started:
        require(provenance[key] == started[key], 'Started/archive provenance mismatch: ' + key)
    require(started['plan_sha256'] == digest(plan_path), 'Plan hash mismatch')
    for key in ('source_commit', 'source_sha256', 'harness_sha256', 'parent_pins', 'campaign_spec_sha256'):
        require(started[key] == plan[key], 'Started/plan mismatch: ' + key)
    require(started['runner_sha256'] == plan['harness_sha256']['run_tuning.py'], 'Runner hash mismatch')
    require(started['host'].split('.')[0] == 'elite' and started['execution'] == 'local_cpu', 'Execution was not local Elite CPU')
    require(started['cpu_affinity'] and set(started['cpu_affinity']) <= {0, 1} and started['nice'] >= 10, 'Local CPU limits violated')
    require(started['threads'] == {key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'Thread limits violated')
    require(started['split'] == plan['phase'] and started['charged_row_budget'] == c['charged_row_budget'], 'Started phase/budget mismatch')
    prereqs = [] if plan['phase'] == 'pilot' else ['resource_decision.json', 'audit_pilot.json']
    if plan['phase'] in ('long', 'confirmation'):
        previous = 'screen' if plan['phase'] == 'long' else 'long'
        prereqs += [f'selection_{previous}.json', f'audit_{previous}.json']
    require(started['execution_prerequisites_sha256'] == {name: digest(root/name) for name in prereqs}, 'Execution prerequisites changed')
    design = meta['design']
    for key, expected in {'n_qubits': 4, 'd': 16, 'rows': 8192, 'probes': 256, 'outcomes': 32}.items():
        require(design[key] == expected, 'Design metadata mismatch: ' + key)
    close(design['povm_a'], 1/37, 'POVM a', rtol=0, atol=1e-15)
    close(design['povm_b'], 1/37, 'POVM b', rtol=0, atol=1e-15)
    require(meta['observation_mode'] == 'noiseless' and meta['active_row_count'] == 8192, 'Data regime changed')
    require(meta['initialization']['basis'] == 'gell-mann', 'Initialization basis changed')
    require(meta['data_metadata']['channel_seed'] == c['channel'], 'Channel seed metadata changed')
    backend = meta['tuning_backend']
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
        ('rho_schedule', c['rho_scale'], c['rho_offset'], c['rho_exponent'], 1.),
        ('step_size_schedule', c['step_scale'], c['step_offset'], c['step_exponent'], 1.)):
        require(conf[key] == dict(type='PowerSchedule', scale=scale, offset=offset, exponent=exponent, cap=cap), 'Native schedule metadata mismatch: ' + key)
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
    return {
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


def check_plan(plan, spec):
    """Validate the complete declared cohort, not merely permissible trials."""
    require(plan['scope'] == 'local_frames_n4_tuning', 'Wrong experiment scope')
    require(plan['phase'] in ('pilot', 'screen', 'long', 'confirmation'), 'Unknown phase')
    require(plan['gate'] == GATE and plan['source_commit'] == COMMIT, 'Gate/source changed')
    require(plan['reserved_final_channels'] == list(range(1000, 1020)), 'Reserved final cohort changed')
    require(plan['parent_pins'] == PARENT_PINS, 'Prior campaign pins changed')
    require(spec['gate'] == GATE and spec['reserved_final_channels'] == list(range(1000, 1020)), 'Spec gate/split changed')
    require(spec['physics_audit_helper'] == {'path': str(HELPER_PATH), 'sha256': HELPER_SHA256}, 'Physical helper pin changed')
    configs = {c['id']: c for c in spec['configs']}
    require(len(configs) == len(spec['configs']) and configs, 'Duplicate/empty candidate grid')
    for c in configs.values():
        require(set(c) == {'id', 'tau', *PARAMETERS}, 'Unknown/missing candidate knob')
    phase = plan['phase']; rule = spec['phases'][phase]
    allowed = rule['config_ids']; selected = plan.get('selected_config_ids', allowed)
    require(len(set(allowed)) == len(allowed) and set(allowed) <= set(configs), 'Invalid allowed candidates')
    require(len(set(selected)) == len(selected) and set(selected) <= set(allowed), 'Invalid selected candidates')
    if phase in ('pilot', 'screen'):
        require(selected == allowed, 'Pilot/screen inventory silently pruned')
    else:
        require('selected_config_ids' in plan and len(selected) == (2 if phase == 'long' else 1)
                and 'selection' in plan, 'Missing complete frozen selection')
    declared_axes = {'pilot': ([0], [0], [0]), 'screen': ([0, 1], [0], [0]),
                     'long': ([0, 1], [0], [0]), 'confirmation': ([2, 3], [1], [1])}
    require(tuple(rule[k] for k in ('channels', 'initialization_replicates', 'sampling_replicates')) == declared_axes[phase], 'Phase cohort changed')
    budget, cadence = rule['charged_row_budget'], rule['checkpoint_charged_rows']
    require(type(budget) is int and type(cadence) is int and 0 < cadence <= budget and budget % cadence == 0, 'Invalid row budget/cadence')
    trials = plan['trials']
    require(trials and len({c['id'] for c in trials}) == len(trials), 'Empty/duplicate trial inventory')
    observed = []
    for c in trials:
        check_trial_configuration(c, configs)
        b = c['batch_size']
        require(budget % b == cadence % b == 0 and c['steps'] == budget//b and c['metrics_every'] == cadence//b
                and c['charged_row_budget'] == budget, 'Budget/clock normalization mismatch')
        require(c['id'] == f"{c['config_id']}-c{c['channel']}-i{c['initialization_replicate']}-s{c['sampling_replicate']}", 'Trial ID mismatch')
        observed.append((c['config_id'], c['channel'], c['initialization_replicate'], c['sampling_replicate']))
    expected = list(itertools.product(selected, *declared_axes[phase]))
    require(sorted(observed) == sorted(expected), 'Incomplete or undeclared Cartesian inventory')
    require(plan['total_charged_row_budget'] == len(expected)*budget, 'Total charged budget mismatch')
    return configs


def ranking_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['config_id'], []).append(row)
    ranked = []
    for cid, group in groups.items():
        require(len(group) == 2 and {r['channel'] for r in group} == {0, 1}, 'Ranking cohort incomplete')
        if any(r['status'] != 'complete' for r in group):
            continue
        passed = all(r['gate_pass'] for r in group)
        ranked.append({'config_id': cid, 'worst_deficit': max(r['deficit'] for r in group),
                       'worst_fidelity': min(r['metrics']['process_fidelity'] for r in group),
                       'both_pass': passed, 'worst_persistent_gate_rows': max(
                           r['persistent_recorded_gate_step']*(r['charged_rows']//r['completed_steps'])
                           for r in group) if passed else None})
    return ranked


def selected_from_rows(rows, phase):
    groups = ranking_groups(rows)
    if phase == 'screen':
        groups = [g for g in groups if g['config_id'] != 'incumbent']
        require(len(groups) >= 2, 'Insufficient complete screen candidates')
        deficit = sorted(groups, key=lambda g: (g['worst_deficit'], g['config_id']))
        fidelity = sorted(groups, key=lambda g: (-g['worst_fidelity'], g['worst_deficit'], g['config_id']))
        selected = list(dict.fromkeys((deficit[0]['config_id'], fidelity[0]['config_id'])))
        selected += [g['config_id'] for g in deficit if g['config_id'] not in selected][:2-len(selected)]
    else:
        eligible = sorted((g for g in groups if g['both_pass']), key=lambda g: (
            g['worst_persistent_gate_rows'], g['worst_deficit'], g['config_id']))
        selected = [eligible[0]['config_id']] if eligible else []
    return selected, groups


def verified_prior_audit(root, phase):
    report = read(root/f'audit_{phase}.json')
    require(report['phase_complete_and_valid'] and all(r['valid'] for r in report['trials']), 'Prior audit incomplete')
    require(report['plan_sha256'] == digest(root/f'plan_{phase}.json')
            and report['batch_inventory_sha256'] == digest(root/f'inventory_{phase}.json'), 'Prior plan/inventory changed')
    require(report['auditor_sha256'] == digest(__file__), 'Prior auditor changed')
    require(report['campaign_spec_sha256'] == digest(root/'campaign_spec.json'), 'Prior campaign spec changed')
    for r in report['trials']:
        for name, key in [('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'),
                          ('summary.json', 'summary_sha256'), ('started.json', 'started_sha256')]:
            require(digest(root/phase/r['id']/name) == r[key], 'Previously audited artifact changed')
    return report


def check_selection_chain(plan, spec, root):
    if plan['phase'] not in ('long', 'confirmation'):
        return 0
    previous = 'screen' if plan['phase'] == 'long' else 'long'
    path = root/f'selection_{previous}.json'
    require(plan['selection'] == {'path': str(path), 'sha256': digest(path)}, 'Selection pin mismatch')
    selection = read(path); prior = verified_prior_audit(root, previous)
    require(selection['audit_sha256'] == digest(root/f'audit_{previous}.json')
            and selection['spec_sha256'] == digest(root/'campaign_spec.json'), 'Selection evidence changed')
    rows = prior['trials']
    levels = 1
    if previous == 'long':
        levels += check_selection_chain(read(root/'plan_long.json'), spec, root)
        oldpath = PARENT/'audit_transfer.json'
        require(selection['archived_control_audit_sha256'] == PARENT_PINS[str(oldpath)], 'Archived control audit changed')
        old = read(oldpath)
        require(old['phase_complete_and_valid'] and old['source_commit'] == COMMIT, 'Archived control provenance changed')
        configs = {c['id']: c for c in spec['configs']}
        archived = []
        for r in old['trials']:
            if r['n_qubits'] != 4:
                continue
            for name, key in [('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'),
                              ('summary.json', 'summary_sha256'), ('started.json', 'started_sha256')]:
                require(digest(PARENT/'transfer'/r['id']/name) == r[key], 'Archived incumbent changed')
            start = read(PARENT/'transfer'/r['id']/'started.json')
            require(start['source_commit'] == COMMIT and start['source_sha256'] == plan['source_sha256'], 'Archived control source mismatch')
            c = start['trial']; batch = configs['incumbent']['batch_size']
            require(all(c[k] == configs['incumbent'][k] for k in (*PARAMETERS, 'tau')), 'Archived control knobs differ')
            require(c['n_qubits'] == 4 and c['channel'] == r['channel'] and c['initialization_replicate'] == c['sampling_replicate'] == 0
                    and c['initialization_seed'] == 200000+10*r['channel'] and c['sampling_seed'] == 300000+100*r['channel'], 'Archived control seeds differ')
            rule = spec['phases']['long']
            require(c['steps']*batch == r['charged_rows'] == rule['charged_row_budget']
                    and c['metrics_every']*batch == rule['checkpoint_charged_rows'], 'Archived control row horizon/cadence differs')
            archived.append({**r, 'config_id': 'incumbent'})
        require(len(archived) == 2 and {r['channel'] for r in archived} == {0, 1}, 'Archived control cohort incomplete')
        rows = rows+archived
        require(selection['confirmation_status'] == 'planned' and selection['archived_control_extra_compute_this_round'] == 0, 'Invalid confirmation state')
    selected, groups = selected_from_rows(rows, previous)
    require(selection['rule'] == spec['selection_rules'][previous], 'Selection rule changed')
    require(selection['ranking_inputs'] == groups, 'Selection ranking inputs changed')
    require(selection['selected_config_ids'] == plan['selected_config_ids'] == selected, 'Selection differs from predeclared ranking')
    return levels


def check_resource_decision(plan, spec, root):
    if plan['phase'] == 'pilot':
        return False
    decision = read(root/'resource_decision.json')
    require(decision['allowed'] is True and decision['quality_metrics_used_for_resource_decision'] is False, 'Resource decision missing/conditioned on quality')
    require(decision['campaign_spec_sha256'] == digest(root/'campaign_spec.json')
            and decision['pilot_audit_sha256'] == digest(root/'audit_pilot.json'), 'Resource evidence changed')
    pilot = verified_prior_audit(root, 'pilot')
    estimates = {r['id']: r for r in decision['estimates']}
    require(len(estimates) == len(decision['estimates']) and set(estimates) == {r['id'] for r in pilot['trials']}, 'Resource inventory incomplete')
    rule = spec['pilot_resource_gate']
    for row in pilot['trials']:
        summary = read(root/'pilot'/row['id']/'summary.json')
        projected = summary['optimizer_seconds']*spec['phases']['long']['charged_row_budget']/row['charged_rows']+summary['total_seconds']
        close(estimates[row['id']]['projected_long_seconds'], projected, 'Pilot resource projection')
        require(estimates[row['id']]['peak_rss_mib'] == summary['peak_rss_mib'], 'Pilot RSS changed')
        require(summary['status'] == 'complete' and projected < rule['max_projected_long_seconds']
                and summary['peak_rss_mib'] < rule['max_peak_rss_mib'], 'Pilot resource bound exceeded')
    return True


def audit(plan_path, root=None):
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent if root is None else Path(root)
    plan, spec = read(plan_path), read(root/'campaign_spec.json')
    require(plan['campaign_spec_sha256'] == digest(root/'campaign_spec.json'), 'Campaign spec hash mismatch')
    check_plan(plan, spec)
    source_count = check_sources(plan, root)
    selection_levels = check_selection_chain(plan, spec, root)
    resource_verified = check_resource_decision(plan, spec, root)
    results, data_pairs, init_pairs = [], {}, {}
    for c in plan['trials']:
        try:
            result = check_trial(root, plan_path, plan, c)
            if result['valid']:
                arrays = result['paired_arrays_sha256']
                data_key = c['channel']
                data = {key: arrays[key] for key in ('active_rows', 'active_observations', 'truth_factor')}
                require(data_pairs.setdefault(data_key, data) == data, 'Same-channel physical data mismatch')
                key = (c['channel'], c['initialization_replicate'])
                require(init_pairs.setdefault(key, arrays['initial_factor']) == arrays['initial_factor'], 'Paired initialization mismatch')
        except Exception as error:
            result = {'id': c['id'], 'config_id': c['config_id'], 'valid': False, 'archive_complete': False,
                      'audit_error': type(error).__name__ + ': ' + str(error)}
        results.append(result)
    inventory_path = root / f"inventory_{plan['phase']}.json"
    inventory_valid, inventory_error = False, None
    try:
        inventory = read(inventory_path)
        require(inventory['phase'] == plan['phase'] and inventory['plan_sha256'] == digest(plan_path), 'Inventory plan/phase mismatch')
        require(inventory['status'] in ('complete', 'finished_with_failures') and set(inventory['trials']) == {c['id'] for c in plan['trials']}, 'Incomplete batch inventory')
        require(inventory['total_charged_row_budget'] == plan['total_charged_row_budget'], 'Inventory total budget mismatch')
        by_id = {r['id']: r for r in results}
        for trial in plan['trials']:
            record, result = inventory['trials'][trial['id']], by_id[trial['id']]
            require(result['valid'], 'Batch has unaudited/incomplete archive')
            expected = ('finished', 0) if result['status'] == 'complete' else ('method_failure', 2)
            require((record['state'], record['exit_code']) == expected, 'Batch/trace outcome mismatch')
            require(record['charged_row_budget'] == trial['charged_row_budget'] and record['charged_rows'] == result['charged_rows'], 'Inventory work mismatch')
            require(record['gate_pass'] == result['gate_pass'], 'Inventory gate mismatch')
        all_complete = all(r['status'] == 'complete' for r in results)
        require((inventory['status'] == 'complete') == all_complete, 'Batch terminal status mismatch')
        inventory_valid = True
    except Exception as error:
        inventory_error = type(error).__name__ + ': ' + str(error)
    return {
        'schema': 'n4_frames_tuning_independent_audit_v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'plan_path': str(plan_path), 'plan_sha256': digest(plan_path), 'campaign_spec_sha256': digest(root/'campaign_spec.json'),
        'auditor_sha256': digest(__file__), 'physical_helper': {'path': str(HELPER_PATH), 'sha256': HELPER_SHA256},
        'source_commit': COMMIT, 'source_file_count': source_count, 'expected_trials': len(plan['trials']),
        'selection_chain_levels_verified': selection_levels,
        'pilot_resource_decision_verified': resource_verified,
        'complete_method_trials': sum(row.get('status') == 'complete' for row in results),
        'failed_method_trials': sum(row.get('status') == 'failed' for row in results),
        'validated_trials': sum(row['valid'] for row in results), 'passing_trials': sum(row.get('gate_pass', False) for row in results),
        'phase_complete_and_valid': inventory_valid and all(row['valid'] for row in results),
        'batch_inventory_valid': inventory_valid, 'batch_inventory_error': inventory_error,
        'batch_inventory_sha256': digest(inventory_path) if inventory_path.exists() else None,
        'distinct_channels': len(data_pairs), 'distinct_initialization_groups': len(init_pairs), 'trials': results,
        'qualification': 'Independent n=4 physical/provenance audit; no optimization, no dense sensing bank, no final held-out use. Nominal row counts exclude TP/metrics. Sampling seeds verified, not unarchived realized batches.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.plan)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')
    raise SystemExit(0 if report['phase_complete_and_valid'] else 2)
