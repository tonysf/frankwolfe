"""Freeze audited FRAMES-only promotions; never overwrite or use held-out data.

Selection reads archived metrics, never launches an optimizer.  The audit is
mandatory and its plan and trial-file inventory must match the current files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
CONFIG_KEYS = ('config_id', 'alpha0', 'penalty_exponent', 'step_scale',
               'step_offset', 'step_exponent', 'rho_scale', 'rho_offset',
               'rho_exponent', 'tau', 'batch_size')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def publish(path, value):
    """Exclusive creation protects every previous selection and frozen plan."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def deficit(rmse, normalized_tp, fidelity):
    values = (float(rmse), float(normalized_tp), float(fidelity))
    check(all(math.isfinite(v) for v in values), 'Nonfinite gate metric.')
    check(values[0] >= 0 and values[1] >= 0, 'Negative residual metric.')
    return max(values[0] / GATE['rmse_max'],
               values[1] / GATE['normalized_tp_max'],
               max(0., 1. - values[2]) / (1. - GATE['fidelity_min']))


def persistent_gate_step(steps, passes):
    steps, passes = np.asarray(steps), np.asarray(passes, dtype=bool)
    check(steps.ndim == passes.ndim == 1 and steps.size == passes.size and steps.size,
          'Invalid checkpoint arrays.')
    persistent = np.logical_and.accumulate(passes[::-1])[::-1]
    return int(steps[np.flatnonzero(persistent)[0]]) if persistent.any() else None


def group_outcomes(outcomes):
    """Summarize the two paired channel endpoints, retaining failures."""
    grouped = {}
    for row in outcomes:
        grouped.setdefault(row['config_id'], []).append(row)
    result = []
    for config_id, rows in grouped.items():
        check(sorted(r['channel'] for r in rows) == [0, 1],
              f'{config_id}: exactly channels 0 and 1 are required.')
        check(all(r.get('sampling_replicate', 0) == 0 for r in rows),
              'Selection stages require sampling replicate zero.')
        failed = sum(r['failed'] for r in rows)
        passes = sum(r['gate_pass'] for r in rows)
        finite = [r['deficit'] for r in rows if r['deficit'] is not None]
        persistent = [r['persistent_recorded_gate_step'] for r in rows]
        result.append({
            'config_id': config_id, 'failed_trials': failed,
            'gate_passes': passes, 'channels': [0, 1],
            'worst_endpoint_deficit': max(finite) if finite else None,
            'mean_log_deficit': (sum(math.log(max(float(d), np.finfo(float).tiny))
                                     for d in finite) / len(finite)) if finite else None,
            'worst_persistent_recorded_gate_step': (
                max(persistent) if passes == 2 and all(v is not None for v in persistent)
                else None),
            'trial_ids': sorted(r['trial_id'] for r in rows),
        })
    return result


def _finite_or_inf(value):
    return math.inf if value is None else value


def rank_screen(outcomes):
    return sorted(group_outcomes(outcomes), key=lambda r: (
        r['failed_trials'], -r['gate_passes'],
        _finite_or_inf(r['worst_endpoint_deficit']),
        _finite_or_inf(r['mean_log_deficit']), r['config_id']))


def promoted_configs(ranking, incumbent):
    check(incumbent in {r['config_id'] for r in ranking}, 'Missing incumbent.')
    return list(dict.fromkeys([r['config_id'] for r in ranking[:2]] + [incumbent]))


def rank_long(outcomes):
    eligible = [r for r in group_outcomes(outcomes)
                if not r['failed_trials'] and r['gate_passes'] == 2
                and r['worst_persistent_recorded_gate_step'] is not None]
    return sorted(eligible, key=lambda r: (
        r['worst_persistent_recorded_gate_step'], r['worst_endpoint_deficit'],
        r['config_id']))


def _close(a, b, label):
    # Summary endpoints use the structured common metric routine; checkpoint
    # metrics use the independently implemented dense bridge.  The same factor
    # differs by roundoff, which is amplified in the dimensionless deficit.
    check(math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=1e-15),
          f'Metric mismatch: {label}.')


def load_trial(root, phase, trial, plan, plan_hash):
    directory = root / phase / trial['id']
    started_path = directory / 'started.json'
    check(started_path.is_file(), f'Missing trial start: {trial["id"]}.')
    started = read(started_path)
    check(started['trial'] == trial and started['plan_sha256'] == plan_hash,
          f'Trial start/configuration mismatch: {trial["id"]}.')
    check(started['source_commit'] == plan['source_commit']
          and started['source_sha256'] == plan['source_sha256'], 'Trial source mismatch.')
    check(started['runner_sha256'] == plan['harness_sha256']['run_local.py'],
          'Trial runner pin mismatch.')
    files = {'started.json': digest(started_path)}
    summary_path, failure_path = directory / 'summary.json', directory / 'failure.json'
    row = {'trial_id': trial['id'], 'config_id': trial['config_id'],
           'channel': trial['channel'], 'sampling_replicate': trial.get('sampling_replicate', 0)}
    if not summary_path.is_file():
        check(failure_path.is_file(), f'Unrecorded or unfinished trial: {trial["id"]}.')
        failure = read(failure_path)
        check(bool(failure.get('error')) and bool(failure.get('traceback'))
              and bool(failure.get('finished_utc')), 'Incomplete failure evidence.')
        files['failure.json'] = digest(failure_path)
        for name in ('result.json', 'result.npz'):
            if (directory / name).exists():
                files[name] = digest(directory / name)
        return {**row, 'failed': True, 'gate_pass': False, 'deficit': None,
                'persistent_recorded_gate_step': None, 'status': 'exception',
                'failure_reason': failure['error']}, files
    check(not failure_path.exists(), 'Ambiguous summary plus exception failure file.')
    summary = read(summary_path)
    check(summary['trial'] == trial, 'Summary trial differs from frozen plan.')
    files['summary.json'] = digest(summary_path)
    for name, field in (('result.json', 'json_sha256'), ('result.npz', 'npz_sha256')):
        check((directory / name).is_file(), f'Missing required archive: {name}.')
        files[name] = digest(directory / name)
        check(files[name] == summary[field], f'Archive checksum mismatch: {name}.')
    metadata = read(directory / 'result.json')
    provenance = metadata['local_calibration']
    check(provenance['trial'] == trial and provenance['plan_sha256'] == plan_hash,
          'Archive trial/plan pin mismatch.')
    check(provenance['source_commit'] == plan['source_commit']
          and provenance['source_sha256'] == plan['source_sha256'], 'Archive source mismatch.')
    endpoints = metadata.get('methods', metadata.get('endpoints', {}))
    check(endpoints['frames']['status'] == summary['status'], 'Archive status mismatch.')
    failed = summary['status'] != 'complete'
    if failed:
        check(bool(summary.get('failure_reason')), 'Failure status lacks a reason.')
    with np.load(directory / 'result.npz', allow_pickle=False) as arrays:
        steps = arrays['frames_checkpoint_steps']
        rmse = arrays['frames_rmse']
        tp = arrays['frames_normalized_tp_violation']
        fidelity = arrays['frames_process_fidelity']
        check(all(v.ndim == 1 and v.size == steps.size for v in (rmse, tp, fidelity)),
              'Checkpoint metric shape mismatch.')
        check(steps.size > 0 and steps[0] == 0 and np.all(np.diff(steps) > 0),
              'Invalid checkpoint steps.')
        check(all(np.all(np.isfinite(v)) for v in (rmse, tp, fidelity)),
              'Nonfinite checkpoint metrics.')
        check(np.array_equal(arrays['frames_cumulative_data_row_accesses'],
                             steps * trial['batch_size']), 'Logical row accounting mismatch.')
        if not failed:
            check(np.array_equal(steps, np.arange(0, trial['steps'] + 1, trial['metrics_every'])),
                  'Complete trajectory does not cover the frozen checkpoint grid.')
        _close(summary['rmse'], rmse[-1], 'endpoint RMSE')
        _close(summary['normalized_tp'], tp[-1], 'endpoint normalized TP')
        _close(summary['fidelity'], fidelity[-1], 'endpoint fidelity')
        d = deficit(rmse[-1], tp[-1], fidelity[-1])
        _close(summary['deficit'], d, 'gate deficit')
        passes = ((rmse <= GATE['rmse_max']) & (tp <= GATE['normalized_tp_max'])
                  & (fidelity >= GATE['fidelity_min']))
        gate = bool(not failed and passes[-1])
        first = int(steps[np.flatnonzero(passes)[0]]) if passes.any() else None
        persistent = persistent_gate_step(steps, passes)
    check(summary['gate_pass'] == gate == provenance['endpoint_gate'], 'Endpoint gate mismatch.')
    check(summary['first_recorded_gate_step'] == first, 'First gate checkpoint mismatch.')
    check(summary['persistent_recorded_gate_step'] == persistent
          == provenance['persistent_recorded_gate_step'], 'Persistent gate checkpoint mismatch.')
    return {**row, 'failed': failed, 'gate_pass': gate, 'deficit': d,
            'persistent_recorded_gate_step': persistent if gate else None,
            'status': summary['status'], 'failure_reason': summary.get('failure_reason')}, files


def verify_audit(audit, plan_hash, trial_files):
    """Bind the independent audit to every current trial file and the plan."""
    check(audit.get('phase_complete_and_valid') is True,
          'Audit is not complete and verified.')
    check(audit.get('plan_sha256') == plan_hash, 'Audit plan checksum mismatch.')
    check(audit.get('expected_trials') == audit.get('validated_trials') == len(trial_files),
          'Audit trial count mismatch.')
    rows = audit.get('trials', [])
    check(len(rows) == len(trial_files)
          and len({r['id'] for r in rows}) == len(rows), 'Duplicate or missing audit trial.')
    field_names = {'started.json': 'started_sha256', 'summary.json': 'summary_sha256',
                   'result.json': 'json_sha256', 'result.npz': 'npz_sha256',
                   'failure.json': 'failure_sha256'}
    audit_files = {row['id']: {name: row[field] for name, field in field_names.items()
                              if field in row} for row in rows}
    check(audit_files == trial_files,
          'Audit trial inventory differs from the current archived files.')


def build_next_plan(plan, selected, selection_path, selection_hash):
    next_plan = deepcopy(plan)
    phase = 'long' if plan['phase'] == 'screen' else 'confirmation'
    configs = {}
    for trial in plan['trials']:
        config = {key: trial[key] for key in CONFIG_KEYS}
        if trial['config_id'] in configs:
            check(configs[trial['config_id']] == config, 'Configuration varies across channels.')
        configs[trial['config_id']] = config
    settings = plan['long_stage'] if phase == 'long' else plan['confirmation']
    channels = settings['channels']
    replicates = [0] if phase == 'long' else settings['sampling_replicates']
    check(channels == ([0, 1] if phase == 'long' else list(range(2, 8))),
          'Forbidden stage channels.')
    check(replicates == ([0] if phase == 'long' else [0, 1]), 'Unexpected sampling replicates.')
    trials = []
    for config_id in selected:
        for channel in channels:
            for replicate in replicates:
                trials.append({**configs[config_id], 'id': f'{config_id}-c{channel}-s{replicate}',
                               'channel': channel, 'sampling_replicate': replicate,
                               'steps': settings['steps'],
                               'metrics_every': plan['long_stage']['metrics_every']})
    next_plan.update(phase=phase, trials=trials, created_utc=now(),
                     stage_start='fresh initial factor and stochastic momentum; no resumption',
                     selected_config_ids=selected,
                     selection={'path': str(selection_path.resolve()), 'sha256': selection_hash,
                                'parent_phase': plan['phase']})
    next_plan['harness_sha256']['select_stage.py'] = digest(__file__)
    if phase == 'confirmation':
        next_plan['split'] = 'previously_exposed_channel_confirmation_not_unseen_test'
    return next_plan


def select_stage(plan_path, audit_path, selection_path, output_plan_path):
    plan_path, audit_path = Path(plan_path).resolve(), Path(audit_path).resolve()
    selection_path, output_plan_path = Path(selection_path), Path(output_plan_path)
    check(not selection_path.exists() and not output_plan_path.exists(),
          'Refusing to overwrite an existing selection or plan.')
    plan, audit = read(plan_path), read(audit_path)
    check(plan.get('phase') in ('screen', 'long'), 'Only screen and long stages can select.')
    check(plan.get('scope') == 'local_n2_full_noiseless_frames_calibration', 'Wrong scope.')
    check(plan.get('gate') == GATE, 'Recovery gate changed.')
    check(plan.get('final_heldout', {}).get('channels') == list(range(1000, 1020)),
          'Missing reserved final held-out cohort.')
    check(plan['trials'] and len({t['id'] for t in plan['trials']}) == len(plan['trials']),
          'Empty or duplicate trial IDs.')
    # The entire source and harness are checked, not merely the commit label.
    from run_local import source, COMMIT
    check(plan['source_commit'] == COMMIT and plan['source_sha256'] == source(),
          'Current source differs from the frozen stage.')
    for name, expected in plan['harness_sha256'].items():
        check(digest(ROOT / name) == expected, f'Frozen harness changed: {name}.')
    plan_hash = digest(plan_path)
    outcomes, trial_files = [], {}
    for trial in plan['trials']:
        outcome, files = load_trial(plan_path.parent, plan['phase'], trial, plan, plan_hash)
        outcomes.append(outcome)
        trial_files[trial['id']] = files
    verify_audit(audit, plan_hash, trial_files)
    ranking = rank_screen(outcomes)
    eligible = rank_long(outcomes) if plan['phase'] == 'long' else []
    selected = (promoted_configs(ranking, plan['incumbent_config_id'])
                if plan['phase'] == 'screen' else ([eligible[0]['config_id']] if eligible else []))
    selection = {
        'schema_version': 1, 'created_utc': now(), 'phase': plan['phase'],
        'status': 'selected' if selected else 'no_selection',
        'qualification': 'FRAMES-only calibration; no final held-out data used.',
        'input_plan': {'path': str(plan_path), 'sha256': plan_hash},
        'input_audit': {'path': str(audit_path), 'sha256': digest(audit_path)},
        'trial_files_sha256': trial_files, 'source_commit': plan['source_commit'],
        'source_sha256': plan['source_sha256'],
        'harness_sha256': {**plan['harness_sha256'], 'select_stage.py': digest(__file__)},
        'screen_order_ranking': ranking, 'long_eligible_ranking': eligible,
        'selected_config_ids': selected, 'trial_outcomes': outcomes,
        'next_plan_path': str(output_plan_path.resolve()) if selected else None,
        'selection_rule': (plan['screen_selection_rule'] if plan['phase'] == 'screen'
                           else plan['long_stage']['selection']),
        'reason': (None if selected else 'No full-horizon configuration passes both channel gates; '
                   'confirmation is not opened. Any expanded tuning requires a new declared round.'),
    }
    # Validate next-stage construction before publishing either output.
    next_plan = build_next_plan(plan, selected, selection_path, 'pending') if selected else None
    publish(selection_path, selection)
    if next_plan is not None:
        next_plan['selection']['sha256'] = digest(selection_path)
        publish(output_plan_path, next_plan)
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-plan', type=Path, required=True)
    args = parser.parse_args()
    result = select_stage(args.plan, args.audit, args.selection, args.output_plan)
    print(json.dumps({'status': result['status'], 'selected_config_ids': result['selected_config_ids'],
                      'selection': str(args.selection), 'next_plan': result['next_plan_path']}))
