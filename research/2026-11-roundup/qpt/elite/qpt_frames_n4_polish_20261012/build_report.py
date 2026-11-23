"""Three-page audit-bound report of the separately declared n=4 polish.

No optimization, projection, rescaling, LaTeX, remote access, or old-file edits.
The finalized input inventory is mandatory; report outputs are exclusive.
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
PARENT = Path('/home/tony/qpt_frames_n4_tuning_20261012')
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
STEM = 'QPT_FRAMES_N4_POLISH_20261012'
PDF, EVIDENCE = ROOT/(STEM+'.pdf'), ROOT/(STEM+'_EVIDENCE.json')
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
CANDIDATE, CONTROL = 'penalty64-two-phase', 'penalty64-stepslow'
FILES = (('started.json', 'started_sha256'), ('summary.json', 'summary_sha256'),
         ('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'))
INPUTS = {}
NAVY, BLUE, ORANGE, MUTED, GREEN = '#183047', '#1479ad', '#b77432', '#596b7b', '#2c8356'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path, expected=None):
    path = Path(path).resolve(); value = digest(path)
    require(expected is None or value == expected, 'Changed input: '+str(path))
    require(str(path) not in INPUTS or INPUTS[str(path)] == value, 'Input changed while reading: '+str(path))
    INPUTS[str(path)] = value
    return value


def read(path, expected=None):
    pin(path, expected)
    return json.loads(Path(path).read_text())


def traces(path):
    pin(path)
    with np.load(path, allow_pickle=False) as z:
        keys = ('checkpoint_steps', 'cumulative_data_row_accesses', 'measurement_loss_mean',
                'rmse', 'normalized_tp_violation', 'process_fidelity')
        trace = {k: z['frames_'+k].copy() for k in keys}
        norm = np.linalg.norm(z['active_observations'])
        require(norm > 0, 'Invalid relative-error normalization')
        trace['relative_rmse'] = trace['rmse']*np.sqrt(len(z['active_observations']))/norm
        trace['infidelity'] = 1-trace['process_fidelity']
    return trace


def archive_row(directory, row, trial):
    require(row['valid'] and row['archive_complete'], 'Incomplete audited archive')
    for name, key in FILES:
        pin(directory/name, row[key])
    summary = read(directory/'summary.json')
    require(summary['trial'] == trial and summary['status'] == row['status']
            and summary['gate_pass'] == row['gate_pass'], 'Summary/audit mismatch')
    return {**row, 'trial': trial, 'summary': summary, 'trace': traces(directory/'result.npz')}


def load_stage(phase, manifest):
    paths = {kind: ROOT/f'{kind}_{phase}.json' for kind in ('plan','audit','inventory')}
    require(all(p.name in manifest['evidence_sha256'] for p in paths.values()), 'Unsealed stage evidence')
    plan,audit,inventory = [read(paths[k]) for k in ('plan','audit','inventory')]
    require(plan['scope']=='local_frames_n4_polish' and plan['phase']==phase, 'Wrong campaign scope')
    require(plan['source_commit']==audit['source_commit']==COMMIT and plan['gate']==GATE, 'Source or gate changed')
    require(plan['reserved_final_channels']==list(range(1000,1020)), 'Final held-out reservation changed')
    require(audit['phase_complete_and_valid'] and audit['batch_inventory_valid'], 'Incomplete independent audit')
    require(audit['plan_sha256']==digest(paths['plan']) and audit['batch_inventory_sha256']==digest(paths['inventory']),
            'Audit plan/inventory pin mismatch')
    require(audit['auditor_sha256']==digest(ROOT/'audit_polish.py'), 'Auditor changed after audit')
    pin(audit['physical_helper']['path'],audit['physical_helper']['sha256'])
    ids=[t['id'] for t in plan['trials']]
    require(len(ids)==len(set(ids))==2 and [r['id'] for r in audit['trials']]==ids, 'Incomplete or duplicate audit inventory')
    require(audit['expected_trials']==audit['validated_trials']==2 and set(inventory['trials'])==set(ids),
            'Incorrect audit/execution counts')
    channels=(0,1) if phase=='calibration' else (2,3)
    replicate=0 if phase=='calibration' else 1
    require({t['channel'] for t in plan['trials']}==set(channels), 'Incorrect channel cohort')
    frozen={'config_id':CANDIDATE,'n_qubits':4,'alpha0':5e-5/64,'penalty_exponent':.25,
            'rho_scale':4.,'rho_offset':8.,'rho_exponent':2/3,
            'step_scale':4**.75,'step_offset':4.,'step_exponent':.75,
            'switch_update':48000,'post_step_scale':4.,'post_step_offset':4.,'post_step_exponent':1.,
            'batch_size':32,'tau':4.,'steps':400000,'metrics_every':4000,'charged_row_budget':12800000,
            'initialization_replicate':replicate,'sampling_replicate':replicate}
    for trial in plan['trials']:
        require(all(trial[k]==v for k,v in frozen.items()), 'Frozen one-arm configuration changed')
        require(trial['initialization_seed']==200000+10*trial['channel']+replicate
                and trial['sampling_seed']==300000+100*trial['channel']+replicate, 'Seed mapping changed')
        require(trial['id']==f"{CANDIDATE}-c{trial['channel']}-i{replicate}-s{replicate}", 'Trial ID changed')
    expected_schedule={'clock':'global zero-based optimizer update','switch_update':48000,
                       'before':'(4/(t+4))**0.75','after':'4/(t+4)','momentum_reset':False,
                       'penalty_clock_reset':False,'tp_projection':False,'resume':False}
    require(plan['schedule']==expected_schedule and plan['total_charged_row_budget']==25600000,
            'Two-phase schedule or matched budget changed')
    failures=any(r['status']=='failed' for r in audit['trials'])
    require(inventory['status']==('finished_with_failures' if failures else 'complete'), 'Execution inventory status mismatch')
    for row in audit['trials']:
        require(row['status'] in ('complete','failed'), 'Unknown native outcome')
        state=inventory['trials'][row['id']]
        expected=('method_failure',2) if row['status']=='failed' else ('finished',0)
        require((state['state'],state['exit_code'])==expected, 'Unresolved infrastructure failure')
    require(set(plan['parent_pins'])=={str(PARENT/name) for name in ('completion_manifest.json','audit_long.json','selection_long.json')},
            'Incorrect prior-campaign pins')
    for path,sha in plan['parent_pins'].items():pin(path,sha)
    require(set(plan['harness_sha256'])=={'run_polish.py','run_batch.py','prepare_polish.py'}, 'Incomplete harness pins')
    for name,sha in plan['harness_sha256'].items():pin(ROOT/name,sha)
    for name,sha in plan['source_sha256'].items():pin(REPO/name,sha)
    rows=[archive_row(ROOT/phase/t['id'],r,t) for t,r in zip(plan['trials'],audit['trials'])]
    require(audit['passing_trials']==sum(r['gate_pass'] for r in rows), 'Gate count changed')
    return {'phase':phase,'plan':plan,'audit':audit,'rows':rows}


def load_control(calibration):
    current=calibration['plan']
    old_manifest=read(PARENT/'completion_manifest.json',current['parent_pins'][str(PARENT/'completion_manifest.json')])
    old_audit=read(PARENT/'audit_long.json',current['parent_pins'][str(PARENT/'audit_long.json')])
    old_selection=read(PARENT/'selection_long.json',current['parent_pins'][str(PARENT/'selection_long.json')])
    require(old_manifest['status']=='complete' and old_manifest['source_commit']==COMMIT
            and old_manifest['selected_config_ids']==[] and old_audit['phase_complete_and_valid'], 'Parent round not completed without a winner')
    require(old_selection['selected_config_ids']==[]
            and old_selection['confirmation_status']=='skipped_no_joint_gate_eligible_configuration'
            and old_selection['audit_sha256']==digest(PARENT/'audit_long.json'), 'Polish outcome condition not satisfied')
    for name,sha in old_manifest['evidence_sha256'].items():
        require(Path(name).name==name,'Unsafe parent evidence path');pin(PARENT/name,sha)
    old_plan=read(PARENT/'plan_long.json',old_audit['plan_sha256'])
    pin(PARENT/'inventory_long.json',old_audit['batch_inventory_sha256'])
    require(old_plan['source_commit']==COMMIT and old_plan['source_sha256']==current['source_sha256']
            and old_plan['gate']==GATE, 'Prior control source/physical gate mismatch')
    trials={t['id']:t for t in old_plan['trials']}
    current_trials={t['channel']:t for t in current['trials']}
    shared=('n_qubits','channel','initialization_replicate','initialization_seed','sampling_replicate','sampling_seed',
            'alpha0','penalty_exponent','rho_scale','rho_offset','rho_exponent','step_scale','step_offset','step_exponent',
            'batch_size','tau','steps','metrics_every','charged_row_budget')
    control=[]
    for row in old_audit['trials']:
        if row['config_id']!=CONTROL:continue
        trial=trials[row['id']]
        require(trial['channel'] in (0,1), 'Unexpected control channel')
        require(all(trial[k]==current_trials[trial['channel']][k] for k in shared), 'Prior/new matched controls differ')
        require(row['status']=='complete' and row['gate_pass'] is False, 'Prior best did not complete with failed joint gate')
        metric=row['metrics']
        require(metric['process_fidelity']>=.99 and metric['normalized_tp_violation']<=.001
                and 1e-6<metric['rmse']<1e-5, 'Claimed prior-best gate pattern differs')
        control.append(archive_row(PARENT/'long'/trial['id'],row,trial))
    require(len(control)==2 and {r['channel'] for r in control}=={0,1}, 'Missing paired prior-best control')
    by_channel={r['channel']:r for r in control}
    prefix_audit=read(PARENT/'audit_screen.json',old_manifest['evidence_sha256']['audit_screen.json'])
    require(prefix_audit['phase_complete_and_valid'], 'Archived screen-prefix audit incomplete')
    prefix_plan=read(PARENT/'plan_screen.json',prefix_audit['plan_sha256'])
    pin(PARENT/'inventory_screen.json',prefix_audit['batch_inventory_sha256'])
    prefix_trials={t['id']:t for t in prefix_plan['trials']}
    for row in calibration['rows']:
        prefix=row['prefix_check']
        require(prefix['required'] is True and prefix['cutoff_steps']==48000, 'Wrong prefix audit scope')
        if row['status']=='failed':
            require(prefix.get('verified') is False and bool(prefix.get('reason')), 'Failed-run prefix was not qualified')
            continue
        old=by_channel[row['channel']]
        prior_prefix=[r for r in prefix_audit['trials'] if r['config_id']==CONTROL and r['channel']==row['channel']]
        require(len(prior_prefix)==1 and prior_prefix[0]['valid'] and prior_prefix[0]['status']=='complete',
                'Archived screen prefix missing/incomplete')
        prior_prefix=prior_prefix[0]
        prefix_trial=prefix_trials[prior_prefix['id']]
        require(all(prefix_trial[k]==row['trial'][k] for k in shared if k not in ('steps','charged_row_budget'))
                and prefix_trial['steps']==48000 and prefix_trial['charged_row_budget']==1536000,
                'Archived screen prefix setup/budget differs')
        prefix_dir=PARENT/'screen'/prior_prefix['id'];old_path=prefix_dir/'result.npz'
        for name,key in FILES:pin(prefix_dir/name,prior_prefix[key])
        require(prefix['physical_metrics_match'] is True and prefix['schedule_prefix_match'] is True
                and prefix['paired_data_and_initial_factor_identical'] is True and prefix['verified'] is True, 'Prefix or pairing mismatch')
        require(prefix['archived_parent_npz_path']==str(old_path)
                and prefix['archived_parent_npz_sha256']==prior_prefix['npz_sha256']
                and prefix['archived_parent_audit_sha256']==digest(PARENT/'audit_screen.json'), 'Prefix audit parent changed')
        require(prefix['compared_checkpoint_count']==13, 'Incomplete 0..48k prefix checkpoint comparison')
        # The auditor uses the short screen archive. Also bind the displayed
        # long-budget control to exactly those same physical prefix checkpoints.
        for key in ('checkpoint_steps','cumulative_data_row_accesses','measurement_loss_mean','rmse',
                    'normalized_tp_violation','process_fidelity'):
            require(np.array_equal(row['trace'][key][:13],old['trace'][key][:13]),
                    'Displayed long control has a different physical prefix: '+key)
    return control


def load_inputs():
    manifest=read(ROOT/'completion_manifest.json')
    require(manifest['status']=='complete' and manifest['source_commit']==COMMIT, 'Polish is not finalized')
    require(manifest['ruche_used'] is False and manifest['final_heldout_used'] is False, 'Execution scope changed')
    require(manifest['script_sha256']['build_report.py']==digest(__file__), 'Report builder changed after sealing')
    require({'selection_calibration.json','validation_tests.json','validation_tests.log'}<=set(manifest['evidence_sha256']),
            'Required selection/validation evidence not sealed')
    for name,sha in manifest['evidence_sha256'].items():
        require(Path(name).name==name,'Unsafe evidence filename');pin(ROOT/name,sha)
    for name,sha in manifest['script_sha256'].items():
        require(Path(name).name==name,'Unsafe script filename');pin(ROOT/name,sha)
    stages={'calibration':load_stage('calibration',manifest)}
    control=load_control(stages['calibration'])
    selection=read(ROOT/'selection_calibration.json')
    require(selection['audit_sha256']==digest(ROOT/'audit_calibration.json')
            and selection['calibration_plan_sha256']==digest(ROOT/'plan_calibration.json'), 'Selection evidence changed')
    eligible=all(r['status']=='complete' and r['gate_pass'] for r in stages['calibration']['rows'])
    chosen=[CANDIDATE] if eligible else []
    require(selection['selected_config_ids']==chosen, 'Eligibility does not match both full-precision gates')
    if eligible:
        require(selection['confirmation_status']=='planned','Eligible winner confirmation not planned')
        stages['confirmation']=load_stage('confirmation',manifest)
        plan=stages['confirmation']['plan']
        require(plan['selected_config_ids']==chosen and plan['selection']=={
            'path':str(ROOT/'selection_calibration.json'),'sha256':digest(ROOT/'selection_calibration.json')}, 'Confirmation winner pin changed')
        for row in stages['confirmation']['rows']:
            require(row['prefix_check']['required'] is False,'Confirmation cannot claim same-channel prefix agreement')
    else:
        require(selection['confirmation_status']=='skipped_no_joint_gate_eligible_configuration','Missing explicit confirmation skip')
        require(all(not (ROOT/f'{kind}_confirmation.json').exists() for kind in ('plan','audit','inventory')),
                'Skipped confirmation nevertheless has artifacts')
    require(manifest['selected_config_ids']==chosen and manifest['recovery_winner_found']==eligible, 'Manifest eligibility differs')
    require(set(manifest['stages'])==set(stages),'Manifest stage inventory differs')
    accounting={phase:counts(stage) for phase,stage in stages.items()}
    for phase,actual in accounting.items():
        for key in ('planned_trials','valid_archives','gate_passes','numerical_failures','updates','charged_rows'):
            require(manifest['stages'][phase][key]==actual[key], 'Manifest work/outcome count differs: '+phase+'/'+key)
    require(manifest['total_updates_including_pilots']==sum(c['updates'] for c in accounting.values())
            and manifest['total_charged_rows_including_pilots']==sum(c['charged_rows'] for c in accounting.values()), 'Manifest total work differs')
    failed=any(c['numerical_failures'] for c in accounting.values())
    require(manifest['recorded_work_is_lower_bound']==failed
            and manifest['unrecorded_partial_work_unknown']==failed,'Failed native work not qualified')
    if eligible:
        require(manifest['confirmation_status']=='complete'
                and manifest['all_confirmation_gates_pass']==all(r['gate_pass'] for r in stages['confirmation']['rows']), 'Confirmation conclusion differs')
    else:
        require(manifest['confirmation_status']==selection['confirmation_status']
                and manifest['all_confirmation_gates_pass'] is None,'Skipped confirmation conclusion differs')
    tests=read(ROOT/'validation_tests.json')
    require(tests['exit_code']==0,'Validation failed')
    pin(ROOT/'validation_tests.log',tests['log_sha256'])
    for path,sha in tests['test_sha256'].items():pin(path,sha)
    return manifest,stages,control,selection,tests


def counts(stage):
    rows = stage['rows']
    return {'planned_trials': len(rows), 'valid_archives': len(rows),
            'numerically_complete': sum(r['status'] == 'complete' for r in rows),
            'numerical_failures': sum(r['status'] == 'failed' for r in rows),
            'gate_passes': sum(r['gate_pass'] for r in rows),
            'updates': sum(r['completed_steps'] for r in rows),
            'charged_rows': sum(r['charged_rows'] for r in rows),
            'optimizer_seconds_sum': sum(r['optimizer_seconds'] for r in rows)}


def paragraph(fig, x, y, value, width=142, size=10, leading=.026):
    for line in textwrap.wrap(value, width, break_long_words=False, break_on_hyphens=False):
        fig.text(x, y, line, fontsize=size, color=NAVY, va='top')
        y -= leading
    return y


def page(title, subtitle, number):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
    fig.text(.05, .935, title, fontsize=23, weight='bold', color=NAVY)
    fig.text(.05, .89, subtitle, fontsize=10.5, color=MUTED)
    fig.text(.05, .026, '12 October 2026 | Local CPU only | Separately declared one-arm n=4 follow-up', fontsize=8.5, color=MUTED)
    fig.text(.95, .026, f'{number}/3', fontsize=8.5, color=MUTED, ha='right')
    return fig


def table(fig, box, labels, rows, widths=None, size=9):
    ax = fig.add_axes(box); ax.axis('off')
    tab = ax.table(cellText=rows, colLabels=labels, colWidths=widths, cellLoc='center', bbox=[0,0,1,1])
    tab.auto_set_font_size(False); tab.set_fontsize(size)
    for (r,c), cell in tab.get_celld().items():
        cell.set_edgecolor('white'); cell.PAD = .035
        cell.set_facecolor(NAVY if r == 0 else ('#eaf2f7' if r % 2 else '#f6f8fa'))
        cell.set_text_props(color='white' if r == 0 else NAVY, weight='bold' if r == 0 else 'normal')
    return tab


def outcome(row):
    return 'NUM. FAIL' if row['status'] != 'complete' else ('PASS' if row['gate_pass'] else 'FAIL')


def metric(row, key):
    value = 1-row['metrics']['process_fidelity'] if key == 'infidelity' else row['metrics'][key]
    return f'{value:.2e}'


def headline(stages):
    calibration = counts(stages['calibration'])
    if calibration['gate_passes'] != 2:
        return f"Calibration: {calibration['gate_passes']}/2 joint-gate passes. No eligible recovery winner; confirmation skipped."
    confirmation = counts(stages['confirmation'])
    return f"Calibration: 2/2 joint-gate passes. Frozen confirmation: {confirmation['gate_passes']}/2 new channel/seed cases pass."


def first_page(stages, control):
    fig = page('FRAMES n=4: two-phase precision polishing',
               'Adaptive calibration follow-up, motivated by the completed 12-arm grid—not part of that original frozen grid', 1)
    paragraph(fig, .05, .83, headline(stages), size=13, width=117, leading=.03)
    paragraph(fig, .05, .757, 'One new arm, declared before its runs: gamma(t)=(4/(t+4))^0.75 for t<48,000, '
              'then gamma(t)=4/(t+4) for t>=48,000. The iteration clock and momentum state continue; no restart at the switch, '
              'no post-hoc rescaling and no TP projection.', size=10.5, width=135, leading=.027)
    paragraph(fig, .05, .657, 'Unchanged: alpha0=5e-5/64; alpha(t)=alpha0(t+1)^0.25; '
              'rho(t)=4/(t+8)^(2/3); batch=32; tau=4. Each trajectory starts afresh for 400,000 updates '
              '(12.8 million sampled rows), including its repeated 48k prefix.', size=10.1, width=143, leading=.027)
    rows = []
    for name, source in [('Prior best, single phase', control), ('New, two phase', stages['calibration']['rows'])]:
        for row in source:
            rows.append([name, row['channel'], metric(row,'rmse'), metric(row,'normalized_tp_violation'),
                         metric(row,'relative_rmse'), metric(row,'infidelity'), outcome(row)])
    table(fig, [.05,.367,.9,.19], ['Schedule', 'Channel', 'RMSE', 'TP norm.', 'Relative RMSE', 'Infidelity (1−F)', 'Joint gate'],
          rows, widths=[.23,.065,.14,.14,.15,.165,.11], size=9.4)
    prefix_rows = []
    for row in stages['calibration']['rows']:
        check = row['prefix_check']
        if row['status']=='failed':
            prefix_rows.append([row['channel'],'—','NOT VERIFIED','native failure','—'])
        else:
            prefix_rows.append([row['channel'], check['compared_checkpoint_count'],
                                'MATCH' if check['physical_metrics_match'] else 'MISMATCH',
                                'MATCH' if check['schedule_prefix_match'] else 'MISMATCH',
                                f"{check['max_abs_metric_differences']['rmse']:.2e}"])
    table(fig, [.05,.218,.9,.091], ['48k prefix: channel', 'Checkpoints compared', 'Physical metrics', 'Applied schedules', 'Max |RMSE difference|'],
          prefix_rows, widths=[.20,.20,.19,.19,.22], size=9)
    paragraph(fig, .05, .182, 'Unchanged joint gate: full-design RMSE <=1e-6, normalized TP residual <=1e-3, process fidelity >=0.99. '
              'Relative RMSE = ||A(UU^H)−y||₂/||y||₂. Gate decisions use full precision, not rounded table values.', size=9.5, width=149, leading=.025)
    paragraph(fig, .05, .105, 'Motivation: the prior best had very high fidelity and passed TP, but missed the RMSE threshold. '
              'Its small trace deficit and late fluctuations suggested testing smaller steps. This is an empirical hypothesis, '
              'not an established causal explanation or convergence theorem.', size=9.5, width=149, leading=.025)
    return fig


def second_page(stages, control):
    fig = page('Paired trajectories at identical sampled-row budgets',
               'Same physical data, initializer and declared sampling seed within each calibration channel; only gamma changes after 48k', 2)
    rows = [('Two-phase polish', BLUE, '-', row) for row in stages['calibration']['rows']]
    rows += [('Prior single-phase best', ORANGE, '--', row) for row in control]
    metrics = [('measurement_loss_mean','Mean measurement loss',5e-13),
               ('normalized_tp_violation','Normalized TP residual',.001),
               ('infidelity','Infidelity (1−F)',.01)]
    for channel in (0,1):
        for j,(key,label,threshold) in enumerate(metrics):
            ax = fig.add_axes([.065+j*.313,.535-channel*.315,.262,.252])
            for name,color,linestyle,row in rows:
                if row['channel'] != channel:
                    continue
                trace = row['trace']; x = trace['cumulative_data_row_accesses']/1e6; mask=x>0
                ax.plot(x[mask],np.maximum(trace[key][mask],1e-16),color=color,linestyle=linestyle,
                        linewidth=1.5,label=name)
            ax.axhline(threshold,color=GREEN,linestyle=':',linewidth=1)
            ax.axvline(1.536,color=MUTED,linestyle=':',linewidth=.8)
            ax.set(xscale='log',yscale='log',xlabel='Charged rows (millions)',ylabel=label)
            ax.set_title(f'Channel {channel}',fontsize=9,loc='left',pad=3)
            ax.grid(True,which='major',color='#d9e3eb',linewidth=.55)
            ax.tick_params(labelsize=8);ax.xaxis.label.set_size(8.5);ax.yaxis.label.set_size(8.5)
            ax.spines[['top','right']].set_visible(False)
            if channel==0 and j==0:
                handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.859),ncol=2,fontsize=10,frameon=False)
    paragraph(fig,.05,.132,'Vertical dotted line: fixed switch after 48,000 updates (1.536 million rows). '
              'Horizontal dotted lines: physical gates. Mean measurement loss excludes the TP penalty. '
              'Penalized objectives are not used for ranking or comparison.',size=9.7,width=146,leading=.025)
    paragraph(fig,.05,.077,'Zero-row initial points are omitted on logarithmic axes; plotted values are floored at 1e-16. '
              'Recorded checkpoints do not certify every intermediate iterate. Row counts exclude TP, LMO and metric work.',
              size=9,width=158,leading=.023)
    return fig


def third_page(stages, control, tests):
    fig = page('Confirmation, accounting and remaining uncertainty',
               'No new tuning from confirmation; final held-out channels 1000–1019 remain unopened',3)
    paragraph(fig,.05,.829,headline(stages),size=12.6,width=121,leading=.03)
    if 'confirmation' in stages:
        rows = []
        for row in stages['confirmation']['rows']:
            step=row['persistent_recorded_gate_step']
            crossing='—' if step is None else f"{step*32/1e6:.3f} M"
            rows.append([row['channel'],metric(row,'rmse'),metric(row,'normalized_tp_violation'),
                         metric(row,'relative_rmse'),metric(row,'infidelity'),outcome(row),crossing])
        table(fig,[.05,.656,.9,.103],['Channel','RMSE','TP norm.','Relative RMSE','1−F','Gate','Persistent gate rows'],
              rows,widths=[.07,.155,.155,.18,.16,.09,.19],size=9.2)
        paragraph(fig,.05,.628,'Confirmation uses channels 2/3, initialization replicate 1 and sampling replicate 1. '
                  'Both seeds change together, so this is a joint generalization check, not an isolated initialization or sampling experiment. '
                  'Two cases do not establish a population success rate.',size=9.8,width=141,leading=.025)
    else:
        paragraph(fig,.05,.744,'CONFIRMATION EXPLICITLY SKIPPED: this arm did not pass all three endpoint gates on both calibration channels. '
                  'No recovery winner is declared; channels 2/3 were not opened to rescue this calibration. '
                  'The completed negative result and the original 12-arm report are preserved.',size=10.5,width=133,leading=.029)
    rows=[];total_rows=0;total_updates=0;failures=0
    for phase,stage in stages.items():
        c=counts(stage);total_rows+=c['charged_rows'];total_updates+=c['updates'];failures+=c['numerical_failures']
        rows.append([phase,c['planned_trials'],c['numerically_complete'],c['numerical_failures'],c['gate_passes'],
                     f"{c['charged_rows']/1e6:.3f}",f"{c['updates']:,}"])
    if 'confirmation' not in stages:
        rows.append(['confirmation','SKIPPED','—','—','—','0','0'])
    rows.append(['TOTAL RECORDED' if failures else 'TOTAL NEW',sum(len(s['rows']) for s in stages.values()),
                 sum(counts(s)['numerically_complete'] for s in stages.values()),failures,
                 sum(counts(s)['gate_passes'] for s in stages.values()),f'{total_rows/1e6:.3f}',f'{total_updates:,}'])
    table(fig,[.05,.397,.9,.128],['Stage','Trials','Complete','Num. failures','Gate passes','Recorded rows (M)','Recorded updates'],
          rows,widths=[.20,.085,.105,.14,.13,.17,.17],size=9)
    work=('Recorded work is a LOWER BOUND: failed native runs may omit partial work, whose size is unknown. '
          if failures else 'All new starts and repeated prefixes are charged. ')
    paragraph(fig,.05,.367,work+f"The prior matched control used {sum(r['charged_rows'] for r in control)/1e6:g} million rows "
              'in the earlier round; reusing its archive adds zero optimizer work here. This polish is additional tuning cost, '
              'not a replacement for the original search budget.',size=9.7,width=144,leading=.025)
    paragraph(fig,.05,.276,'The original twelve-arm grid was completed before this separate, adaptively motivated test. '
              'Only one new schedule was evaluated here. Success would support this frozen recipe on the tested cases, '
              'not an exhaustive optimum, a proof of nonlinear factor-method convergence, or a noise/higher-rank claim.',
              size=9.7,width=144,leading=.025)
    paragraph(fig,.05,.181,'This remains a noiseless rank-one n=4 experiment (d=16; 8,192 physical Born probabilities; tau=4). '
              'Sampling seeds and unchanged native source are pinned; realized batches are not archived. '
              'Published Quiroga remains unresolved: no completed best-for-best or runtime-superiority claim.',
              size=9.7,width=144,leading=.025)
    log=next((line.strip() for line in reversed(tests.get('log','').splitlines()) if 'passed' in line),'passed; see evidence JSON')
    fig.text(.05,.079,'Validation: '+log[:125]+'. Source '+COMMIT[:12]+'. Input/output SHA-256 hashes accompany this PDF.',fontsize=8,color=MUTED)
    return fig


def main():
    for path in (PDF,EVIDENCE):
        require(not path.exists(),'Refusing overwrite: '+str(path))
    manifest,stages,control,selection,tests=load_inputs()
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'text.color':NAVY})
    figures=[first_page(stages,control),second_page(stages,control),third_page(stages,control,tests)]
    buffer=io.BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.infodict().update(Title='FRAMES n=4 two-phase polish — 12 October 2026',Author='QPT project')
        for fig in figures:pdf.savefig(fig)
    for fig in figures:plt.close(fig)
    data=buffer.getvalue()
    evidence={'schema':'frames_n4_polish_report_v1','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'builder_sha256':digest(__file__),'input_sha256':INPUTS,'pages':3,
              'outputs_sha256':{str(PDF):hashlib.sha256(data).hexdigest()},
              'counts':{phase:counts(stage) for phase,stage in stages.items()},'selection':selection,
              'prefix_checks':{str(r['channel']):r['prefix_check'] for r in stages['calibration']['rows']},
              'recorded_work_is_lower_bound':manifest['recorded_work_is_lower_bound'],
              'unrecorded_partial_work_unknown':manifest['unrecorded_partial_work_unknown'],
              'ruche_used':False,'final_heldout_used':False,'best_for_best_complete':False,
              'qualification':'Separately declared, adaptively motivated one-arm FRAMES calibration. All old results preserved; no noise, higher-rank, exhaustive optimum, causal-mechanism or Quiroga-superiority claim.'}
    rendered=json.dumps(evidence,indent=2,sort_keys=True,allow_nan=False)+'\n'
    for path,sha in tuple(INPUTS.items()):
        require(digest(path)==sha,'Input changed while rendering: '+path)
    with PDF.open('xb') as stream:stream.write(data)
    with EVIDENCE.open('x') as stream:stream.write(rendered)
    print(json.dumps({'pdf':str(PDF),'evidence':str(EVIDENCE),'pages':3,'pdf_sha256':digest(PDF)}))


if __name__=='__main__':
    main()
