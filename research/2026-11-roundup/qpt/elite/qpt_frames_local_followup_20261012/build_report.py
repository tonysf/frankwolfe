"""Three-page, audit-bound frozen-setting follow-up report; no optimizer calls."""
import datetime
import hashlib
import json
import re
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/tony/qpt_frames_local_20261012')
STEM = 'QPT_FRAMES_LOCAL_FOLLOWUP_20261012'
PDF, EVIDENCE = ROOT / (STEM + '.pdf'), ROOT / (STEM + '_EVIDENCE.json')
PNGS = [ROOT / (STEM + '_LOSS_TP.png'), ROOT / (STEM + '_TRANSFER.png')]
INPUTS = {}
NAVY, BLUE, ORANGE, MUTED = '#183047', '#177dac', '#bc6a18', '#596b7b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    INPUTS[str(path)] = digest(path)
    return json.loads(Path(path).read_text())


def require(test, text):
    if not test:
        raise ValueError(text)


def curves(path):
    INPUTS[str(path)] = digest(path)
    with np.load(path, allow_pickle=False) as a:
        keys = ['checkpoint_steps', 'cumulative_data_row_accesses', 'measurement_loss_mean',
                'rmse', 'normalized_tp_violation', 'process_fidelity']
        out = {k: a['frames_' + k].copy() for k in keys}
        targets = a['active_observations']
        out['relative_rmse'] = out['rmse'] * np.sqrt(len(targets)) / np.linalg.norm(targets)
    return out


def load(phase):
    pp, ap = ROOT / f'plan_{phase}.json', ROOT / f'audit_{phase}.json'
    if not pp.exists():
        return dict(phase=phase, plan=None, audit=None, rows=[], curves={})
    p = read(pp)
    require(p['scope'] == 'local_frames_frozen_followup', 'Wrong report scope')
    require(p['reserved_final_channels'] == list(range(1000, 1020)), 'Held-out reservation changed')
    require(all(t['channel'] in range(8) for t in p['trials']), 'Unexpected channel in plan')
    for path, sha in p['parent_pins'].items():
        require(digest(path) == sha, 'Frozen parent changed')
        INPUTS[path] = sha
    a = read(ap) if ap.exists() else None
    if a:
        require(a['plan_sha256'] == digest(pp), 'Audit plan hash mismatch')
        require([r['id'] for r in a['trials']] == [t['id'] for t in p['trials']], 'Audit inventory mismatch')
    rows, traces = [], {}
    by_id = {r['id']: r for r in a['trials']} if a else {}
    for trial in p['trials']:
        r = by_id.get(trial['id'], {'id': trial['id'], 'valid': False})
        rows.append({**r, 'trial': trial})
        if not r.get('valid'):
            continue
        folder = ROOT / phase / trial['id']
        for filename, field in [('result.npz', 'npz_sha256'), ('result.json', 'json_sha256'),
                                ('started.json', 'started_sha256'), ('summary.json', 'summary_sha256')]:
            require(digest(folder / filename) == r[field], 'File differs from independent audit')
            INPUTS[str(folder / filename)] = r[field]
        traces[trial['id']] = curves(folder / 'result.npz')
    return dict(phase=phase, plan=p, audit=a, rows=rows, curves=traces)


def counts(stage):
    r = stage['rows']
    return dict(planned=len(r), audited=sum(x.get('valid', False) for x in r),
                completed=sum(x.get('status') == 'complete' for x in r),
                passes=sum(x.get('gate_pass', False) for x in r),
                failures=sum(x.get('status') == 'failed' or x.get('failure_record_present', False) for x in r))


def text(fig, x, y, value, width=125, size=10, leading=.026):
    for line in textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False):
        fig.text(x, y, line, fontsize=size, color=NAVY, va='top')
        y -= leading
    return y


def page(title, subtitle, index):
    f = plt.figure(figsize=(13.333, 7.5), facecolor='white')
    f.text(.055, .93, title, fontsize=24, color=NAVY, weight='bold')
    f.text(.055, .883, subtitle, fontsize=10.5, color=MUTED)
    f.text(.055, .031, '12 October 2026 • Local CPU only • Frozen settings; no Ruche', fontsize=9, color=MUTED)
    f.text(.95, .031, f'{index}/3', ha='right', fontsize=9, color=MUTED)
    return f


def table(fig, bbox, labels, rows, size=10, widths=None):
    ax = fig.add_axes(bbox); ax.axis('off')
    tab = ax.table(cellText=rows or [['—'] * len(labels)], colLabels=labels, colWidths=widths,
                   cellLoc='center', bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(size)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('white'); cell.PAD = .04
        cell.set_facecolor(NAVY if r == 0 else ('#eaf2f7' if r % 2 else '#f6f8fa'))
        cell.set_text_props(color='white' if r == 0 else NAVY, weight='bold' if r == 0 else 'normal')


def metric(row, key):
    v = row.get('metrics', {}).get(key)
    return '—' if v is None else f'{v:.2e}'


def outcome(row):
    return ('PASS' if row['gate_pass'] else 'FAIL') if row.get('valid') else 'unaudited'


def first_page(stages):
    r = stages['robustness']; c = counts(r)
    f = page('Frozen FRAMES: initialization and sampling robustness',
             'No new parameter search • Noiseless rank-one n=2 • Previously exposed channels 2–7', 1)
    text(f, .055, .814, f"{c['passes']}/{c['audited']} audited trajectories pass; {c['completed']} completed, "
         f"{c['failures']} numerical failures; {c['planned'] - c['audited']} planned trajectories unaudited.", size=15)
    rows = [[x['trial']['channel'], x['trial']['initialization_replicate'], x['trial']['sampling_replicate'],
             metric(x, 'rmse'), metric(x, 'normalized_tp_violation'), metric(x, 'relative_rmse'),
             '—' if not x.get('metrics') else f"{1-x['metrics']['process_fidelity']:.2e}", outcome(x)] for x in r['rows']]
    table(f, [.055, .285, .895, .46], ['Channel', 'Init.', 'Stream', 'RMSE', 'TP norm.', 'Relative error', 'Infidelity (1−F)', 'Gate'],
          rows, size=9.5, widths=[.08, .06, .08, .16, .16, .17, .18, .11])
    text(f, .055, .247, 'Six channels × one new initializer × two new sampling streams = twelve trajectories, not twelve independent channels. '
         'Initialization and sampling both change: this tests joint robustness, not an isolated causal initialization effect.', size=10.5)
    text(f, .055, .165, 'Frozen controls: alpha0=5e-5; alpha(t)=alpha0(t+1)^0.25; beta(t)=1/[2alpha(t)]; '
         'gamma=min(1,4/(t+4)); rho=min(1,4/(t+8)^(2/3)); batch32; tau=sqrt(d). '
         'Every robustness trajectory restarts for 400,000 updates; all work is counted.', size=10)
    text(f, .055, .091, 'Unchanged joint gate: fidelity >=0.99, full-design RMSE <=1e-6 and TPnorm <=1e-3. '
         'Outcomes do not trigger retuning. Final held-out channels 1000–1019 remain excluded.', size=9.1, width=150, leading=.023)
    return f


def style(ax, ylabel):
    ax.set(xscale='log', yscale='log', xlabel='Charged data-row accesses (millions)', ylabel=ylabel)
    ax.grid(True, which='major', color='#d9e3eb', linewidth=.7)
    ax.tick_params(labelsize=9); ax.spines[['top', 'right']].set_visible(False)


def second_page(stages, old):
    f = page('Measurement loss, TP error and the longer channel-0 tail',
             'Measurement loss excludes the TP penalty • Charged rows exclude TP, LMO, metrics and compilation work', 2)
    tail = stages['tail']; tail_row = next((r for r in tail['rows'] if r.get('valid')), None)
    for j, (key, label, threshold) in enumerate([('measurement_loss_mean', 'Measurement mean loss', 5e-13),
                                               ('normalized_tp_violation', 'Normalized TP residual', 1e-3)]):
        ax = f.add_axes([.075 + .478*j, .425, .397, .37])
        for i, tr in enumerate(stages['robustness']['curves'].values()):
            mask = tr['cumulative_data_row_accesses'] > 0
            ax.plot(tr['cumulative_data_row_accesses'][mask]/1e6, tr[key][mask], color=BLUE, alpha=.38,
                    linewidth=1, label='New initializer / streams' if i == 0 else None)
        for tr in tail['curves'].values():
            mask = tr['cumulative_data_row_accesses'] > 0
            ax.plot(tr['cumulative_data_row_accesses'][mask]/1e6, tr[key][mask], color=ORANGE, linewidth=1.7, label='c0 original seeds: 800k')
        if old:
            mask = old['cumulative_data_row_accesses'] > 0
            ax.plot(old['cumulative_data_row_accesses'][mask]/1e6, old[key][mask], 'k:', linewidth=1.1, label='Archived c0 400k prefix')
        ax.axhline(threshold, color='#2d8354', linestyle='--', linewidth=1, label='Gate threshold')
        ax.axvline(12.8, color=MUTED, linestyle=':', linewidth=.8)
        style(ax, label); ax.legend(fontsize=8, loc='best')
    if tail_row:
        tr = tail['curves'][tail_row['id']]
        passing = (tr['rmse'] <= 1e-6) & (tr['normalized_tp_violation'] <= .001) & (tr['process_fidelity'] >= .99)
        mask = tr['checkpoint_steps'] > 400000
        failed_steps = tr['checkpoint_steps'][mask & ~passing]
        text(f, .055, .33, f"Tail endpoint: {outcome(tail_row)}; RMSE {metric(tail_row, 'rmse')}, "
             f"TPnorm {metric(tail_row, 'normalized_tp_violation')}. Beyond 400k: "
             f"{int(np.sum(passing[mask]))}/{int(mask.sum())} recorded checkpoints pass. "
             f"Last failing checkpoint after 400k: {int(failed_steps[-1]) if len(failed_steps) else 'none'}.", size=12)
        detail = tail_row.get('tail_stability')
        prefix = (f"Independent 400k prefix audit: {'MATCH' if detail['prefix_metrics_match'] else 'MISMATCH'}; "
                  f"maximum RMSE difference {detail['prefix_max_abs_metric_differences']['rmse']:.2e}. "
                  'Stored physical metrics, schedules and sampled batches are compared; intermediate factors are not archived.') if detail else 'Independent tail-prefix audit details are pending.'
        text(f, .055, .215, prefix + ' The longer run starts afresh, so its duplicated 400k prefix is charged. '
             'Checks every 4,000 updates support only observed checkpoint stability, not continuous-time or asymptotic convergence.', size=10.5)
    else:
        text(f, .055, .33, 'The 800k tail is not yet independently audited. No tail-stability conclusion is available.', size=13)
    text(f, .055, .105, 'Channel 0 first passed at 308k but later crossed back above the gate; its 400k endpoint had a narrow RMSE margin. '
         'This follow-up checks that same seed trajectory beyond the old endpoint; it is not another independent channel. '
         'The zero-row initial point is omitted on logarithmic axes.', size=9, width=150, leading=.023)
    return f


def third_page(stages, decision, tests):
    f = page('Frozen-parameter dimension transfer: exploratory only',
             'n=3/4 use unchanged structured NumPy FRAMES; n=2 uses the pinned JAX CPU adapter', 3)
    records = [(name, x) for name in ('pilot', 'transfer') for x in stages[name]['rows']]
    rows = []
    for phase, x in records:
        t = x['trial']; epochs = x.get('charged_rows', 0)/(2*8**t['n_qubits'])
        rows.append([phase, t['n_qubits'], t['channel'], f"{t['steps']:,}", f'{epochs:g}' if x.get('valid') else '—',
                     metric(x, 'rmse'), metric(x, 'relative_rmse'), metric(x, 'normalized_tp_violation'),
                     '—' if not x.get('metrics') else f"{1-x['metrics']['process_fidelity']:.2e}", outcome(x)])
    table(f, [.055, .59, .895, .225], ['Stage', 'n', 'Chan.', 'Updates', 'Row epochs', 'RMSE', 'Rel. error', 'TP norm.', '1−F', 'Gate'],
          rows, size=9.1, widths=[.095,.04,.055,.12,.115,.135,.13,.125,.115,.07])
    all_counts = [counts(s) for s in stages.values()]
    text(f, .055, .556, f"All follow-up stages: {sum(c['completed'] for c in all_counts)}/{sum(c['planned'] for c in all_counts)} completed; "
         f"{sum(c['failures'] for c in all_counts)} numerical failures; "
         f"{sum(c['planned']-c['audited'] for c in all_counts)} unaudited. "
         'Pilot gate outcomes are short-budget diagnostics, not recovery tests.', width=158, size=9)
    ax = f.add_axes([.078, .183, .405, .31])
    for phase in ('robustness', 'transfer'):
        for i, row in enumerate(stages[phase]['rows']):
            tr = stages[phase]['curves'].get(row['id'])
            if tr is None: continue
            n = row['trial']['n_qubits']; mask = tr['cumulative_data_row_accesses'] > 0
            ax.plot(tr['cumulative_data_row_accesses'][mask]/1e6, tr['relative_rmse'][mask],
                    color={2:BLUE,3:'#2c8857',4:'#985caa'}[n], alpha=.28 if n==2 else .9,
                    linestyle='-' if row['trial']['channel'] % 2 == 0 else '--',
                    label=('n2 robustness' if i==0 else None) if n==2 else f"n{n}, c{row['trial']['channel']}")
    style(ax, 'Relative measurement error'); ax.legend(fontsize=8, loc='best')
    summary = []
    for n in (3, 4):
        item = (decision or {}).get('dimensions', {}).get(str(n))
        summary.append(f"n{n}: " + ('resource decision pending' if item is None else
                       ('omitted by resource rule' if item['selected_steps']==0 else f"{item['selected_steps']:,}-update budget")))
    text(f, .54, .493, '; '.join(summary) + '. Budget decisions use pilot runtime/memory only, never numerical success.', width=62, size=10)
    text(f, .54, .376, 'Relative error = ||A(UU^H)-y||₂ / ||y||₂. The same absolute RMSE threshold '
         'does not mean the same relative accuracy across dimensions. Alpha0 is preserved per mean-loss row; '
         'summed lambda, POVM constants, tau and row epochs change with dimension.', width=62, size=10)
    n4_rows = [x for x in stages['transfer']['rows'] if x['trial']['n_qubits'] == 4]
    n4_failed_all = n4_rows and all(x.get('valid') and not x.get('gate_pass') and
        x['metrics']['process_fidelity'] < .99 and x['metrics']['rmse'] > 1e-6 and
        x['metrics']['normalized_tp_violation'] > .001 for x in n4_rows)
    conclusion = ('Both n4 cases failed data fit, fidelity and TP, not just TP. Next: separately declare n4 penalty/sampling tuning. '
                  if n4_failed_all else 'Transferred settings are not tuned at n=3/4; these tests do not establish optimal performance. ')
    text(f, .54, .207, conclusion + 'Quiroga remains unresolved; no head-to-head or speed claim.', width=62, size=10)
    validation = 'Regression results are in the evidence JSON.'
    if tests:
        require(tests['exit_code'] == 0, 'Regression suite failed')
        match = re.search(r'(\d+) passed, (\d+) subtests passed', tests['log'])
        require(match, 'Unrecognized regression result')
        validation = f'Regression suite: {match[1]} tests and {match[2]} subtests passed.'
    text(f, .055, .09, 'Noiseless rank-one tests only; no noisy or higher-rank conclusions. ' + validation +
         ' Source 4ce1108a0692 and all input hashes are in the companion evidence JSON.', size=9, width=150, leading=.023)
    return f


def main():
    for p in [PDF, EVIDENCE, *PNGS]: require(not p.exists(), f'Refusing overwrite: {p}')
    manifest = read(ROOT / 'completion_manifest.json')
    require(manifest['status'] == 'complete', 'Campaign is not finalized')
    require(not manifest['ruche_used'] and not manifest['final_heldout_used'], 'Campaign scope changed')
    require(manifest['script_sha256'][Path(__file__).name] == digest(__file__), 'Report builder changed after finalization')
    for name, sha in manifest['evidence_sha256'].items():
        require(digest(ROOT / name) == sha, 'Finalized evidence changed')
    stages = {p: load(p) for p in ('pilot', 'robustness', 'tail', 'transfer')}
    require(all(stage['audit'] and stage['audit']['phase_complete_and_valid'] for stage in stages.values()),
            'Final report requires all phase audits')
    decision = read(ROOT/'resource_decision.json') if (ROOT/'resource_decision.json').exists() else None
    tests = read(ROOT/'validation_tests.json') if (ROOT/'validation_tests.json').exists() else None
    if stages['transfer']['plan']:
        require(digest(ROOT/'resource_decision.json') == stages['transfer']['plan']['resource_decision']['sha256'], 'Resource-decision pin mismatch')
    old = None
    if stages['tail']['curves']:
        audit = read(PARENT/'audit_long.json'); row = next(r for r in audit['trials'] if r['id']=='rho-standard-c0-s0')
        path = PARENT/'long'/'rho-standard-c0-s0'/'result.npz'
        require(digest(path)==row['npz_sha256'], 'Old prefix archive changed'); old = curves(path)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'text.color':NAVY,'axes.labelsize':10})
    figures = [first_page(stages), second_page(stages,old), third_page(stages,decision,tests)]
    with PDF.open('xb') as stream, PdfPages(stream) as pdf:
        pdf.infodict().update(Title='Frozen FRAMES local follow-up — 12 October 2026', Author='QPT project')
        for fig in figures: pdf.savefig(fig)
    for fig,path in zip(figures[1:],PNGS):
        with path.open('xb') as stream: fig.savefig(stream,format='png',dpi=165)
    for fig in figures: plt.close(fig)
    tests = read(ROOT/'validation_tests.json') if (ROOT/'validation_tests.json').exists() else None
    evidence = dict(schema='frames_frozen_followup_report_v1',date='2026-10-12',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    builder_sha256=digest(__file__),input_sha256=INPUTS,counts={p:counts(s) for p,s in stages.items()},
                    audits={p:s['audit'] for p,s in stages.items()},resource_decision=decision,validation_tests=tests,
                    outputs_sha256={str(p):digest(p) for p in [PDF,*PNGS]},heldout_used=False,
                    qualification='Frozen-setting follow-up; no retuning, no Ruche, no causal initializer isolation, no published-Quiroga or timing superiority claim.')
    with EVIDENCE.open('x') as stream: json.dump(evidence,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
    print(json.dumps({'pdf':str(PDF),'pages':3,'plots':[str(p) for p in PNGS],'evidence':str(EVIDENCE)}))


if __name__ == '__main__': main()
