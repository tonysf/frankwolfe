"""Four-page, audit-bound n=4 FRAMES report. No optimizer or LaTeX calls.

Run only after finalization. All files are read from the declared plans, not
globbed from partially populated result directories. Outputs are exclusive.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/tony/qpt_frames_local_followup_20261012')
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
STEM = 'QPT_FRAMES_N4_TUNING_20261012'
PDF, EVIDENCE = ROOT/(STEM+'.pdf'), ROOT/(STEM+'_EVIDENCE.json')
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
KNOBS = ('alpha0', 'penalty_exponent', 'step_scale', 'step_offset', 'step_exponent',
         'rho_scale', 'rho_offset', 'rho_exponent', 'batch_size', 'tau')
ARCHIVES = (('started.json', 'started_sha256'), ('summary.json', 'summary_sha256'),
            ('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'))
INPUTS = {}
NAVY, BLUE, ORANGE, MUTED, GREEN = '#183047', '#1479ad', '#c47124', '#596b7b', '#2c8356'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path, sha=None):
    path = Path(path).resolve()
    value = digest(path)
    require(sha is None or value == sha, 'Changed evidence: '+str(path))
    require(str(path) not in INPUTS or INPUTS[str(path)] == value,
            'Input changed while report was built: '+str(path))
    INPUTS[str(path)] = value
    return value


def read(path, sha=None):
    pin(path, sha)
    return json.loads(Path(path).read_text())


def traces(path):
    pin(path)
    with np.load(path, allow_pickle=False) as z:
        names = ('checkpoint_steps', 'cumulative_data_row_accesses', 'measurement_loss_mean',
                 'rmse', 'normalized_tp_violation', 'process_fidelity')
        out = {key: z['frames_'+key].copy() for key in names}
        target_norm = np.linalg.norm(z['active_observations'])
        require(target_norm > 0, 'Zero measurement normalization')
        out['relative_rmse'] = out['rmse']*np.sqrt(len(z['active_observations']))/target_norm
        out['infidelity'] = 1-out['process_fidelity']
    return out


def archive_row(directory, audited_row, trial):
    require(audited_row['valid'] and audited_row['archive_complete'], 'Invalid result archive')
    for name, key in ARCHIVES:
        pin(directory/name, audited_row[key])
    summary = read(directory/'summary.json')
    require(summary['trial'] == trial, 'Trial/summary mismatch')
    require(summary['status'] == audited_row['status']
            and summary['gate_pass'] == audited_row['gate_pass'], 'Outcome mismatch')
    return {**audited_row, 'trial': trial, 'summary': summary,
            'trace': traces(directory/'result.npz')}


def load_stage(phase, spec, manifest):
    pp, ap, ip = [ROOT/f'{kind}_{phase}.json' for kind in ('plan', 'audit', 'inventory')]
    for path in (pp, ap, ip):
        require(path.name in manifest['evidence_sha256'], 'Unsealed phase evidence: '+path.name)
    plan, audit, inventory = read(pp), read(ap), read(ip)
    require(plan['scope'] == 'local_frames_n4_tuning' and plan['phase'] == phase, 'Wrong campaign scope')
    require(plan['source_commit'] == audit['source_commit'] == COMMIT, 'Wrong source revision')
    require(plan['gate'] == GATE and plan['reserved_final_channels'] == list(range(1000, 1020)),
            'Gate or final-channel reservation changed')
    require(plan['campaign_spec_sha256'] == audit['campaign_spec_sha256'] == digest(ROOT/'campaign_spec.json'),
            'Spec pin mismatch')
    require(audit['phase_complete_and_valid'] and audit['batch_inventory_valid'], 'Incomplete phase audit')
    require(audit['plan_sha256'] == digest(pp) and audit['batch_inventory_sha256'] == digest(ip),
            'Audit plan/inventory pin mismatch')
    require(audit['auditor_sha256'] == digest(ROOT/'audit_tuning.py'), 'Auditor changed after audit')
    expected_ids = [t['id'] for t in plan['trials']]
    require(len(set(expected_ids)) == len(expected_ids), 'Duplicate trial IDs')
    require([r['id'] for r in audit['trials']] == expected_ids, 'Incomplete audit trial inventory')
    require(audit['expected_trials'] == audit['validated_trials'] == len(expected_ids), 'Audit counts mismatch')
    require(set(inventory['trials']) == set(expected_ids), 'Incomplete execution inventory')
    any_failure = any(r['status'] == 'failed' for r in audit['trials'])
    require(inventory['status'] == ('finished_with_failures' if any_failure else 'complete'),
            'Execution inventory status disagrees with audited outcomes')
    for row in audit['trials']:
        state = inventory['trials'][row['id']]
        wanted = ('method_failure', 2) if row['status'] == 'failed' else ('finished', 0)
        require((state['state'], state['exit_code']) == wanted,
                'Missing/infrastructure failure or inconsistent native outcome')
    axes = spec['phases'][phase]
    for trial in plan['trials']:
        require(trial['n_qubits'] == 4 and trial['channel'] in axes['channels'], 'Wrong report cohort')
        require(trial['initialization_replicate'] in axes['initialization_replicates']
                and trial['sampling_replicate'] in axes['sampling_replicates'], 'Wrong seed replicate')
    for path, sha in plan['parent_pins'].items():
        pin(path, sha)
    for name, sha in plan['harness_sha256'].items():
        pin(ROOT/name, sha)
    for name, sha in plan['source_sha256'].items():
        pin(REPO/name, sha)
    rows = [archive_row(ROOT/phase/t['id'], r, t) for t, r in zip(plan['trials'], audit['trials'])]
    require(audit['passing_trials'] == sum(r['gate_pass'] for r in rows), 'Gate count mismatch')
    return {'phase': phase, 'plan': plan, 'audit': audit, 'rows': rows}


def load_control(spec, current_plan):
    old_path = PARENT/'audit_transfer.json'
    old = read(old_path, current_plan['parent_pins'][str(old_path)])
    old_manifest_path = PARENT/'completion_manifest.json'
    old_manifest = read(old_manifest_path, current_plan['parent_pins'][str(old_manifest_path)])
    require(old_manifest['status'] == 'complete' and old['phase_complete_and_valid'], 'Prior campaign incomplete')
    old_plan = read(PARENT/'plan_transfer.json', old['plan_sha256'])
    pin(PARENT/'inventory_transfer.json', old['batch_inventory_sha256'])
    require(old_plan['source_commit'] == COMMIT and old_plan['gate'] == GATE, 'Archived control source/gate differs')
    baseline = next(c for c in spec['configs'] if c['id'] == 'incumbent')
    trials = {t['id']: t for t in old_plan['trials']}
    rows = []
    for audited in old['trials']:
        if audited['n_qubits'] != 4:
            continue
        trial = trials[audited['id']]
        require(all(trial[k] == baseline[k] for k in KNOBS), 'Archived control parameters differ')
        require(trial['channel'] in (0, 1) and trial['initialization_replicate'] == trial['sampling_replicate'] == 0,
                'Archived control seeds differ')
        require(trial['initialization_seed'] == 200000+10*trial['channel']
                and trial['sampling_seed'] == 300000+100*trial['channel'], 'Archived control seed mapping differs')
        require(trial['steps']*trial['batch_size'] == spec['phases']['long']['charged_row_budget']
                and trial['metrics_every']*trial['batch_size'] == spec['phases']['long']['checkpoint_charged_rows'],
                'Archived control budget/checkpoints differ')
        rows.append(archive_row(PARENT/'transfer'/trial['id'], {**audited, 'config_id': 'incumbent'}, trial))
    require(len(rows) == 2 and {r['channel'] for r in rows} == {0, 1}, 'Archived n4 control cohort incomplete')
    return rows


def groups(rows):
    result = []
    by_config = {}
    for row in rows:
        by_config.setdefault(row['config_id'], []).append(row)
    for cid, pair in by_config.items():
        require(len(pair) == 2 and {r['channel'] for r in pair} == {0, 1}, 'Incomplete ranking pair')
        if any(r['status'] != 'complete' for r in pair):
            continue
        passed = all(r['gate_pass'] for r in pair)
        result.append({'config_id': cid, 'worst_deficit': max(r['deficit'] for r in pair),
                       'worst_fidelity': min(r['metrics']['process_fidelity'] for r in pair),
                       'both_pass': passed, 'worst_persistent_gate_rows': max(
                           r['persistent_recorded_gate_step']*r['trial']['batch_size'] for r in pair) if passed else None})
    return result


def verify_selections(stages, control, spec, screen, long):
    for phase, selection in [('screen', screen), ('long', long)]:
        require(selection['audit_sha256'] == digest(ROOT/f'audit_{phase}.json')
                and selection['spec_sha256'] == digest(ROOT/'campaign_spec.json'), 'Selection evidence changed')
        require(selection['rule'] == spec['selection_rules'][phase], 'Selection rule changed')
    sg = [g for g in groups(stages['screen']['rows']) if g['config_id'] != 'incumbent']
    require(len(sg) >= 2, 'Too few complete screen candidates')
    deficit = sorted(sg, key=lambda g: (g['worst_deficit'], g['config_id']))
    fidelity = sorted(sg, key=lambda g: (-g['worst_fidelity'], g['worst_deficit'], g['config_id']))
    nominees = list(dict.fromkeys([deficit[0]['config_id'], fidelity[0]['config_id']]))
    nominees += [g['config_id'] for g in deficit if g['config_id'] not in nominees][:2-len(nominees)]
    require(screen['ranking_inputs'] == sg and screen['selected_config_ids'] == nominees,
            'Screen selection differs from declared ranking')
    require(stages['long']['plan']['selected_config_ids'] == nominees, 'Long nominees differ')
    lg = groups(stages['long']['rows']+control)
    eligible = sorted((g for g in lg if g['both_pass']), key=lambda g: (
        g['worst_persistent_gate_rows'], g['worst_deficit'], g['config_id']))
    winner = [eligible[0]['config_id']] if eligible else []
    require(long['ranking_inputs'] == lg and long['selected_config_ids'] == winner,
            'Long selection differs from declared ranking')
    require(long['archived_control_audit_sha256'] == digest(PARENT/'audit_transfer.json')
            and long['archived_control_extra_compute_this_round'] == 0, 'Archived-control accounting changed')
    for phase, previous in [('long', 'screen'), ('confirmation', 'long')]:
        if phase not in stages:
            continue
        path = ROOT/f'selection_{previous}.json'
        require(stages[phase]['plan']['selection'] == {'path': str(path), 'sha256': digest(path)},
                'Plan/selection pin mismatch')
    return winner


def counts(stage):
    rows = stage['rows']
    return {'planned_trials': len(rows), 'valid_archives': len(rows),
            'numerically_complete': sum(r['status'] == 'complete' for r in rows),
            'numerical_failures': sum(r['status'] != 'complete' for r in rows),
            'gate_passes': sum(r['gate_pass'] for r in rows),
            'updates': sum(r['completed_steps'] for r in rows),
            'charged_rows': sum(r['charged_rows'] for r in rows),
            'optimizer_seconds_sum': sum(r['optimizer_seconds'] for r in rows),
            'total_seconds_sum': sum(r['summary']['total_seconds'] for r in rows)}


def paragraph(fig, x, y, value, width=138, size=10, leading=.025):
    for line in textwrap.wrap(value, width, break_long_words=False, break_on_hyphens=False):
        fig.text(x, y, line, fontsize=size, color=NAVY, va='top')
        y -= leading
    return y


def page(title, subtitle, number):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
    fig.text(.05, .935, title, fontsize=23, weight='bold', color=NAVY)
    fig.text(.05, .89, subtitle, fontsize=10.5, color=MUTED)
    fig.text(.05, .026, '12 October 2026 | Local CPU only | n=4, noiseless rank-one channels', fontsize=8.5, color=MUTED)
    fig.text(.95, .026, f'{number}/4', fontsize=8.5, color=MUTED, ha='right')
    return fig


def table(fig, box, labels, rows, widths=None, size=9):
    ax = fig.add_axes(box); ax.axis('off')
    tab = ax.table(cellText=rows, colLabels=labels, colWidths=widths, cellLoc='center', bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(size)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('white'); cell.PAD = .035
        cell.set_facecolor(NAVY if r == 0 else ('#eaf2f7' if r % 2 else '#f6f8fa'))
        cell.set_text_props(color='white' if r == 0 else NAVY, weight='bold' if r == 0 else 'normal')
    return tab


def outcome(row):
    return 'NUM. FAIL' if row['status'] != 'complete' else ('PASS' if row['gate_pass'] else 'FAIL')


def metric(row, key):
    v = 1-row['metrics']['process_fidelity'] if key == 'infidelity' else row['metrics'][key]
    return f'{v:.2e}'


def schedule(c, prefix):
    exponent = c[prefix+'_exponent']
    if c[prefix+'_scale'] == c[prefix+'_offset']**exponent:
        return f"({c[prefix+'_offset']:g}/(t+{c[prefix+'_offset']:g}))^{exponent:g}" if exponent != 1 else f"{c[prefix+'_scale']:g}/(t+{c[prefix+'_offset']:g})"
    exp = '' if exponent == 1 else ('^(2/3)' if exponent == 2/3 else f'^{exponent:g}')
    return f"{c[prefix+'_scale']:.3g}/(t+{c[prefix+'_offset']:g}){exp}"


def headline(winner, stages):
    if not winner:
        return 'No joint-gate recovery winner: confirmation explicitly skipped.'
    passed = sum(r['gate_pass'] for r in stages['confirmation']['rows'])
    return f'{winner[0]} passes both calibration channels; confirmation passes {passed}/2 new channel/seed cases.'


def first_page(spec, stages, screen, winner):
    fig = page('FRAMES at four qubits: bounded local tuning',
               'Separate n=4 calibration after n=2-tuned parameters failed the n=4 transfer gate', 1)
    paragraph(fig, .05, .83, headline(winner, stages), size=13, width=113, leading=.03)
    paragraph(fig, .05, .757, 'Full physical design: d=16, 256 probes, 32 outcomes, 8,192 noiseless Born probabilities; '
              'rank one, Gell-Mann initialization, factor radius tau=4. Gate: RMSE <=1e-6, TPnorm <=1e-3, F >=0.99.', size=10.2)
    rows = [[c['id'], f"{c['alpha0']:.7g}", f"{c['penalty_exponent']:g}", schedule(c, 'step'), schedule(c, 'rho'),
             c['batch_size'], 'YES' if c['id'] in screen['selected_config_ids'] else '—'] for c in spec['configs']]
    table(fig, [.05, .242, .9, .435], ['Candidate', 'alpha0', 'p', 'gamma(t), capped at 1', 'rho(t), capped at 1', 'Batch', 'Long nominee'],
          rows, widths=[.225,.12,.045,.205,.205,.06,.14], size=9)
    paragraph(fig, .05, .205, 'alpha(t)=alpha0(t+1)^p grows in every arm; beta(t)=1/[2alpha(t)]. '
              'The measurement gradient is in mean-loss units; summed-equivalent TP weight is lambda=8192 alpha, not batch × alpha. '
              't counts optimizer updates, starting at zero.', size=9.5, width=151, leading=.025)
    paragraph(fig, .05, .127, 'Screen: 12 settings × channels 0/1, 1.536 million sampled rows each. Long: two nominated settings × the same channels, '
              '12.8 million rows each, compared with the archived same-seed incumbent. This partial grid does not establish globally optimal tuning.',
              size=9.5, width=151, leading=.025)
    return fig


def second_page(spec, stages, screen):
    fig = page('All screen outcomes, paired by channel',
               'Each cell lists channel 0 then channel 1 | Same initializer/stream within channel | All 12 configurations retained', 2)
    by_key = {(r['config_id'], r['channel']): r for r in stages['screen']['rows']}
    rows = []
    for config in spec['configs']:
        cid = config['id']; pair = [by_key[cid, c] for c in (0, 1)]
        label = cid + (' *' if cid in screen['selected_config_ids'] else '')
        rows.append([label]+['\n'.join(metric(r, key) for r in pair) for key in
                            ('rmse', 'normalized_tp_violation', 'infidelity', 'relative_rmse')]
                    +['\n'.join(outcome(r) for r in pair)])
    table(fig, [.05, .19, .9, .65], ['Configuration (* nominated)', 'RMSE\nc0 / c1', 'TP norm.\nc0 / c1',
          'Infidelity (1−F)\nc0 / c1', 'Relative RMSE\nc0 / c1', 'Joint gate\nc0 / c1'], rows,
          widths=[.25,.155,.155,.155,.165,.12], size=9)
    paragraph(fig, .05, .157, 'Relative RMSE = ||A(UU^H)−y||₂ / ||y||₂. Nomination uses one smallest worst-channel joint deficit '
              'and one largest worst-channel fidelity; duplicates are filled by deficit order. Truth fidelity is used only for calibration selection.',
              size=9.5, width=149, leading=.025)
    paragraph(fig, .05, .082, 'Joint deficit D=max(RMSE/1e-6, TPnorm/1e-3, max(0,1−F)/0.01). '
              'Numerical failures remain visible and cannot be nominated. Penalized objectives with different alpha are never ranked.',
              size=9, width=155, leading=.023)
    return fig


def third_page(stages, control):
    fig = page('Long-budget trajectories versus archived n=4 control',
               'Two matched calibration channels | Same physical data, initialization and declared sampling seed | Fresh starts', 3)
    ids = stages['long']['plan']['selected_config_ids']
    palette = {ids[0]: BLUE, ids[1]: ORANGE, 'incumbent': '#424b52'}
    all_rows = stages['long']['rows']+control
    metrics = [('measurement_loss_mean', 'Mean measurement loss', 5e-13),
               ('normalized_tp_violation', 'Normalized TP residual', .001),
               ('infidelity', 'Infidelity (1−F)', .01)]
    for channel in (0, 1):
        for j, (key, label, threshold) in enumerate(metrics):
            ax = fig.add_axes([.066+j*.313, .55-channel*.276, .262, .214])
            for row in all_rows:
                if row['channel'] != channel:
                    continue
                cid = row['config_id']; trace = row['trace']; x = trace['cumulative_data_row_accesses']/1e6
                mask = x > 0
                ax.plot(x[mask], np.maximum(trace[key][mask], 1e-16), color=palette[cid],
                        linestyle='--' if cid == 'incumbent' else '-', linewidth=1.4, label=cid)
            ax.axhline(threshold, color=GREEN, linestyle=':', linewidth=1)
            ax.set(xscale='log', yscale='log', xlabel='Charged rows (millions)', ylabel=label)
            ax.set_title(f'Channel {channel}', fontsize=9, loc='left', pad=3)
            ax.grid(True, which='major', color='#d9e3eb', linewidth=.55)
            ax.tick_params(labelsize=8); ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
            ax.spines[['top', 'right']].set_visible(False)
            if channel == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.857), ncol=3, fontsize=9, frameon=False)
    endpoints = []
    for row in all_rows:
        step = row['persistent_recorded_gate_step']
        crossing = '—' if step is None else f"{step*row['trial']['batch_size']/1e6:.3f} M"
        endpoints.append([row['config_id'], row['channel'], metric(row, 'rmse'), metric(row, 'normalized_tp_violation'),
                          metric(row, 'relative_rmse'), metric(row, 'infidelity'), outcome(row), crossing])
    table(fig, [.05, .071, .9, .132], ['Configuration', 'Chan.', 'RMSE', 'TP norm.', 'Relative RMSE', '1−F', 'Gate', 'Persistent gate rows'],
          endpoints, widths=[.235,.05,.115,.115,.13,.115,.08,.16], size=7.5)
    fig.text(.05, .044, 'Dotted: physical gate. Zero-row points omitted; plotted values floored at 1e-16. Crossings certify recorded checkpoints only.',
             fontsize=8, color=MUTED)
    return fig


def fourth_page(spec, stages, control, winner, tests):
    fig = page('Confirmation, total work and remaining limits',
               'Eligibility requires both calibration channels to pass all three unchanged physical thresholds', 4)
    if winner:
        paragraph(fig, .05, .83, headline(winner, stages), size=12.5, width=121)
        rows = [[r['channel'], r['trial']['initialization_replicate'], r['trial']['sampling_replicate'],
                 metric(r, 'rmse'), metric(r, 'normalized_tp_violation'), metric(r, 'relative_rmse'),
                 metric(r, 'infidelity'), outcome(r)] for r in stages['confirmation']['rows']]
        table(fig, [.05,.66,.9,.098], ['Channel', 'Init.', 'Stream', 'RMSE', 'TP norm.', 'Relative RMSE', '1−F', 'Gate'],
              rows, widths=[.075,.055,.065,.16,.16,.18,.17,.135], size=9)
        paragraph(fig, .05, .635, 'Channels 2/3 use initialization replicate 1 and sampling replicate 1. '
                  'No retuning is allowed from confirmation; two cases do not establish a population success rate or final held-out performance.',
                  size=9.7, width=145, leading=.025)
    else:
        paragraph(fig, .05, .83, 'CONFIRMATION SKIPPED: no configuration passed both calibration channels.', size=13, width=120)
        paragraph(fig, .05, .763, 'The two long nominees and the archived control were assessed by the same joint gate. '
                  'No recovery winner is declared, and channels 2/3 were not used to rescue or retune the selection. '
                  'Failed gates are completed scientific outcomes, not missing experiments.', size=10.3, width=131, leading=.029)
        paragraph(fig, .05, .656, 'This bounded grid can show progress or failure within its declared work budget; '
                  'it cannot prove that FRAMES is incapable of solving this physical setup or that one penalty mechanism caused the failure.',
                  size=9.7, width=145, leading=.025)
    stage_rows = []
    totals = {k: 0 for k in ('planned_trials', 'numerically_complete', 'numerical_failures', 'gate_passes', 'updates', 'charged_rows')}
    for phase, stage in stages.items():
        c = counts(stage)
        for key in totals:
            totals[key] += c[key]
        stage_rows.append([phase, c['planned_trials'], c['numerically_complete'], c['numerical_failures'], c['gate_passes'],
                           f"{c['charged_rows']/1e6:.3f}", f"{c['updates']:,}"])
    if not winner:
        stage_rows.append(['confirmation', 'SKIPPED', '—', '—', '—', '0', '0'])
    lower_bound = totals['numerical_failures'] > 0
    stage_rows.append(['TOTAL RECORDED' if lower_bound else 'TOTAL NEW', totals['planned_trials'], totals['numerically_complete'], totals['numerical_failures'],
                       totals['gate_passes'], f"{totals['charged_rows']/1e6:.3f}", f"{totals['updates']:,}"])
    table(fig, [.05,.376,.9,.17], ['Stage', 'Trials', 'Complete', 'Num. failures', 'Gate passes',
          'Recorded rows (M)' if lower_bound else 'New rows (M)', 'Recorded updates' if lower_bound else 'New updates'],
          stage_rows, widths=[.19,.105,.115,.14,.14,.15,.16], size=8.9)
    work_note = ('Recorded rows/updates are LOWER BOUNDS: failed native runs may omit partial work; its size is unknown. '
                 'Pilots and recorded repeated prefixes are counted.' if lower_bound else
                 'All new starts, resource pilots and repeated prefixes are included above. '
                 'Pilot gate outcomes are short diagnostics, not recovery tests.')
    paragraph(fig, .05, .348, f"Archived control: {sum(r['charged_rows'] for r in control)/1e6:.1f} million rows from the previous round, "
              'zero additional compute here. '+work_note, size=9.4, width=148, leading=.024)
    paragraph(fig, .05, .267, 'Interpretation limits: larger batches mean fewer updates at matched row budgets and different terminal alpha; '
              'the sampling contrast is not an isolated variance test. Row counts exclude TP, LMO and metric work, so they are not runtime/FLOP equivalence. '
              'Seeds and native source are verified; realized sampling batches were not archived.', size=9.4, width=148, leading=.024)
    paragraph(fig, .05, .163, 'Only noiseless, rank-one channels are tested. Channels 1000–1019 remain reserved. '
              'The earlier n=2 result is not evidence of tuned n=4 recovery. Published Quiroga remains unresolved: '
              'this is FRAMES calibration, not a completed best-for-best comparison or a speed claim.', size=9.4, width=148, leading=.024)
    log_last = next((line.strip() for line in reversed(tests.get('log', '').splitlines()) if 'passed' in line), 'passed (see evidence JSON)')
    fig.text(.05, .073, 'Validation: '+log_last[:145]+'. Source '+COMMIT[:12]+'. All input/output SHA-256 hashes are in the companion evidence JSON.',
             fontsize=8, color=MUTED)
    return fig


def load_inputs():
    manifest = read(ROOT/'completion_manifest.json')
    require(manifest['status'] == 'complete' and manifest['source_commit'] == COMMIT, 'Campaign is not finalized')
    require(manifest['ruche_used'] is False and manifest['final_heldout_used'] is False, 'Scope changed')
    require(manifest['script_sha256']['build_report.py'] == digest(__file__), 'Report builder changed after finalization')
    required = {'campaign_spec.json', 'selection_screen.json', 'selection_long.json', 'resource_decision.json', 'validation_tests.json'}
    require(required <= set(manifest['evidence_sha256']), 'Required campaign evidence is not sealed')
    for name, sha in manifest['evidence_sha256'].items():
        require(Path(name).name == name, 'Unsafe manifest evidence filename')
        pin(ROOT/name, sha)
    for name, sha in manifest['script_sha256'].items():
        require(Path(name).name == name, 'Unsafe manifest script filename')
        pin(ROOT/name, sha)
    spec = read(ROOT/'campaign_spec.json')
    require(spec['gate'] == GATE and spec['reserved_final_channels'] == list(range(1000, 1020)), 'Spec gate/cohort changed')
    stages = {phase: load_stage(phase, spec, manifest) for phase in ('pilot', 'screen', 'long')}
    screen, long = read(ROOT/'selection_screen.json'), read(ROOT/'selection_long.json')
    require(long['confirmation_status'] in ('planned', 'skipped_no_joint_gate_eligible_configuration'), 'Unknown confirmation state')
    if long['selected_config_ids']:
        require(long['confirmation_status'] == 'planned', 'Winner/confirmation state mismatch')
        stages['confirmation'] = load_stage('confirmation', spec, manifest)
        require(stages['confirmation']['plan']['selected_config_ids'] == long['selected_config_ids'], 'Confirmation winner changed')
    else:
        require(long['confirmation_status'] == 'skipped_no_joint_gate_eligible_configuration', 'Missing explicit skip')
        require(all(not (ROOT/f'{kind}_confirmation.json').exists() for kind in ('plan','audit','inventory')),
                'Skipped confirmation nevertheless has stage artifacts')
    control = load_control(spec, stages['long']['plan'])
    winner = verify_selections(stages, control, spec, screen, long)
    require(manifest['selected_config_ids'] == winner and manifest['recovery_winner_found'] == bool(winner),
            'Manifest winner differs from independent ranking')
    require(set(manifest['stages']) == set(stages), 'Manifest has missing or undeclared stages')
    accounting = {phase: counts(stage) for phase, stage in stages.items()}
    for phase, actual in accounting.items():
        for key in ('planned_trials', 'valid_archives', 'gate_passes', 'numerical_failures', 'updates', 'charged_rows'):
            require(manifest['stages'][phase][key] == actual[key], 'Manifest accounting mismatch: '+phase+'/'+key)
    require(manifest['total_updates_including_pilots'] == sum(c['updates'] for c in accounting.values())
            and manifest['total_charged_rows_including_pilots'] == sum(c['charged_rows'] for c in accounting.values()),
            'Total work accounting mismatch')
    lower_bound = any(c['numerical_failures'] for c in accounting.values())
    require(manifest['recorded_work_is_lower_bound'] == lower_bound
            and manifest['unrecorded_partial_work_unknown'] == lower_bound, 'Unknown failed-run work is not qualified')
    require(manifest['archived_control_added_compute_this_round'] == 0
            and manifest['repeated_prefixes_charged'] is True, 'Repeated-prefix/control accounting changed')
    if winner:
        require(manifest['confirmation_status'] == 'complete'
                and manifest['all_confirmation_gates_pass'] == all(r['gate_pass'] for r in stages['confirmation']['rows']),
                'Manifest confirmation claim differs from results')
    else:
        require(manifest['confirmation_status'] == long['confirmation_status']
                and manifest['all_confirmation_gates_pass'] is None, 'Skipped confirmation claim differs')
    resource = read(ROOT/'resource_decision.json')
    require(resource['allowed'] and resource['quality_metrics_used_for_resource_decision'] is False
            and resource['pilot_audit_sha256'] == digest(ROOT/'audit_pilot.json')
            and resource['campaign_spec_sha256'] == digest(ROOT/'campaign_spec.json'), 'Resource decision not pinned/valid')
    tests = read(ROOT/'validation_tests.json')
    require(tests['exit_code'] == 0, 'Validation failed')
    pin(ROOT/'validation_tests.log', tests['log_sha256'])
    for path, sha in tests['test_sha256'].items():
        pin(path, sha)
    return manifest, spec, stages, control, screen, long, winner, tests, resource


def main():
    for path in (PDF, EVIDENCE):
        require(not path.exists(), 'Refusing overwrite: '+str(path))
    manifest, spec, stages, control, screen, long, winner, tests, resource = load_inputs()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'text.color': NAVY})
    figures = [first_page(spec, stages, screen, winner), second_page(spec, stages, screen),
               third_page(stages, control), fourth_page(spec, stages, control, winner, tests)]
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.infodict().update(Title='FRAMES n=4 local tuning — 12 October 2026', Author='QPT project')
        for figure in figures:
            pdf.savefig(figure)
    for figure in figures:
        plt.close(figure)
    content = buffer.getvalue()
    evidence = {'schema': 'frames_n4_tuning_report_v1', 'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'builder_sha256': digest(__file__), 'input_sha256': INPUTS, 'pages': 4,
                'outputs_sha256': {str(PDF): hashlib.sha256(content).hexdigest()},
                'counts': {phase: counts(stage) for phase, stage in stages.items()},
                'selection_screen': screen, 'selection_long': long, 'resource_decision': resource,
                'confirmation_status': long['confirmation_status'], 'winner': winner,
                'archived_control_new_compute': 0, 'final_heldout_used': False, 'ruche_used': False,
                'recorded_work_is_lower_bound': manifest['recorded_work_is_lower_bound'],
                'unrecorded_partial_work_unknown': manifest['unrecorded_partial_work_unknown'],
                'best_for_best_complete': False, 'validation_exit_code': tests['exit_code'],
                'qualification': 'Bounded local n4 FRAMES calibration; all failures retained; no objective ranking, exhaustive optimality, causal penalty claim, or published-Quiroga superiority claim.'}
    rendered = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+'\n'
    # Detect mutable inputs before committing either output. Hashing is read-only.
    for path, sha in tuple(INPUTS.items()):
        require(digest(path) == sha, 'Input changed during rendering: '+path)
    with PDF.open('xb') as stream:
        stream.write(content)
    with EVIDENCE.open('x') as stream:
        stream.write(rendered)
    print(json.dumps({'pdf': str(PDF), 'pages': 4, 'evidence': str(EVIDENCE), 'pdf_sha256': digest(PDF)}))


if __name__ == '__main__':
    main()
