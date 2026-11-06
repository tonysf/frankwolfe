"""Build an exclusive, four-page PDF from frozen local FRAMES audit artifacts.

This does not run experiments, retune, access Ruche, or inspect the held-out
cohort.  Confirmation archives are read only after their frozen selection is
verified.  Round-one artifacts only; later calibration rounds are not pooled.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parent
DATE = '12 October 2026'
PDF = ROOT / 'QPT_FRAMES_LOCAL_CALIBRATION_20261012_v2.pdf'
LOSS_PNG = ROOT / 'QPT_FRAMES_LOCAL_LOSS_20261012_v2.png'
TP_PNG = ROOT / 'QPT_FRAMES_LOCAL_TP_20261012_v2.png'
EVIDENCE = ROOT / 'QPT_FRAMES_LOCAL_CALIBRATION_20261012_v2_EVIDENCE.json'
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
NAVY, BLUE, PURPLE, MUTED = '#172a43', '#176f9e', '#9458a2', '#5a6879'
GREEN, RED, PALE = '#237650', '#af4242', '#edf3f8'
INPUTS = {}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_input(path):
    path = Path(path)
    INPUTS[str(path)] = digest(path)
    return json.loads(path.read_text())


def load_selection(phase, plan_path, audit_path):
    path = ROOT / f'selection_{phase}.json'
    selection = read_input(path)
    check(selection['input_plan']['sha256'] == digest(plan_path), 'Selection plan pin mismatch.')
    check(selection['input_audit']['sha256'] == digest(audit_path), 'Selection audit pin mismatch.')
    return selection


def load_stage(phase, *, required=True, parent_selection=None):
    plan_path, audit_path = ROOT / f'plan_{phase}.json', ROOT / f'audit_{phase}.json'
    if not plan_path.exists():
        check(not required, f'Missing frozen {phase} plan.')
        return None
    plan = read_input(plan_path)
    check(plan['phase'] == phase and plan['gate'] == GATE, 'Stage or gate mismatch.')
    check(plan['scope'] == 'local_n2_full_noiseless_frames_calibration', 'Wrong scientific scope.')
    check(plan['final_heldout']['channels'] == list(range(1000, 1020)), 'Reserved cohort changed.')
    check(plan['trials'] and len({t['id'] for t in plan['trials']}) == len(plan['trials']),
          'Empty or repeated trial inventory.')
    allowed_channels = set(range(2, 8)) if phase == 'confirmation' else {0, 1}
    check(all(t['channel'] in allowed_channels for t in plan['trials']), 'Unexpected channel cohort.')
    if parent_selection is not None:
        check(plan['selection']['sha256'] == digest(parent_selection), 'Parent selection pin mismatch.')
        parent = read_input(parent_selection)
        check(parent['status'] == 'selected', 'Stage opened without a selected parent configuration.')
        check(set(t['config_id'] for t in plan['trials']) == set(parent['selected_config_ids']),
              'Stage configurations differ from frozen selection.')
    if not audit_path.exists():
        check(not required, f'Missing independent {phase} audit.')
        return {'phase': phase, 'plan': plan, 'audit': None, 'trials': [], 'curves': {},
                'selection': None, 'state': 'frozen; audit not available'}
    audit = read_input(audit_path)
    check(audit['plan_sha256'] == digest(plan_path), 'Audit plan checksum mismatch.')
    check(audit['expected_trials'] == len(plan['trials']), 'Audit expected count mismatch.')
    check([r['id'] for r in audit['trials']] == [t['id'] for t in plan['trials']],
          'Audit ordered inventory differs from frozen plan.')
    if required:
        check(audit['phase_complete_and_valid'] is True, f'{phase} audit incomplete or invalid.')
    by_id = {t['id']: t for t in plan['trials']}
    curves = {}
    for row in audit['trials']:
        if not row.get('valid'):
            continue
        trial = by_id[row['id']]
        folder = ROOT / phase / row['id']
        check(Path(row['id']).name == row['id'], 'Unsafe trial ID.')
        for name, field in (('started.json', 'started_sha256'), ('summary.json', 'summary_sha256'),
                            ('result.json', 'json_sha256'), ('result.npz', 'npz_sha256')):
            path = folder / name
            check(path.is_file() and digest(path) == row[field], f'Archive changed since audit: {path}.')
            INPUTS[str(path)] = row[field]
        summary = json.loads((folder / 'summary.json').read_text())
        check(summary['trial'] == trial, 'Archived configuration changed.')
        check(summary['status'] == row['status'] and summary['gate_pass'] == row['gate_pass'],
              'Audit and summary outcomes differ.')
        if phase == 'long':
            with np.load(folder / 'result.npz', allow_pickle=False) as arrays:
                curves[row['id']] = {
                    'config_id': trial['config_id'], 'channel': trial['channel'],
                    'charged_rows': arrays['frames_cumulative_data_row_accesses'].copy(),
                    'mean_loss': arrays['frames_measurement_loss_mean'].copy(),
                    'normalized_tp': arrays['frames_normalized_tp_violation'].copy(),
                }
    selection = load_selection(phase, plan_path, audit_path) if phase in ('screen', 'long') else None
    return {'phase': phase, 'plan': plan, 'audit': audit, 'trials': audit['trials'],
            'curves': curves, 'selection': selection,
            'state': 'complete and audited' if audit['phase_complete_and_valid'] else 'partial audit'}


def sci(value, digits=2):
    return '—' if value is None else f'{float(value):.{digits}e}'


def stage_counts(stage):
    if stage is None:
        return {'expected': 0, 'validated': 0, 'completed': 0, 'failures': 0, 'passes': 0,
                'pending': 0, 'channels': 0}
    rows = stage['trials']
    valid = [r for r in rows if r.get('valid')]
    expected = len(stage['plan']['trials'])
    return {'expected': expected, 'validated': len(valid),
            'completed': sum(r.get('status') == 'complete' for r in valid),
            'failures': sum(r.get('status') != 'complete' for r in valid),
            'passes': sum(bool(r.get('gate_pass')) for r in valid),
            'pending': expected - len(valid), 'channels': len({r['channel'] for r in valid})}


def paragraph(fig, x, y, text, *, width=90, size=11, color=NAVY, leading=.027):
    lines = []
    for part in text.split('\n'):
        lines.extend(textwrap.wrap(part, width=width, break_long_words=False,
                                   break_on_hyphens=False) if part else [''])
    for line in lines:
        fig.text(x, y, line, fontsize=size, color=color, va='top')
        y -= leading
    return y


def header(title, subtitle, number):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
    fig.text(.055, .93, title, fontsize=24, weight='bold', color=NAVY)
    fig.text(.055, .883, subtitle, fontsize=10.5, color=MUTED)
    fig.add_artist(plt.Line2D([.055, .95], [.859, .859], transform=fig.transFigure,
                             color='#cad6e1', linewidth=1))
    fig.text(.055, .032, f'FRAMES local calibration • {DATE} • Round one only • No Ruche',
             fontsize=8.5, color=MUTED)
    fig.text(.95, .032, f'{number}/4', ha='right', fontsize=8.5, color=MUTED)
    return fig


def table(fig, bbox, columns, rows, *, widths=None, size=10, highlight_rows=()):
    ax = fig.add_axes(bbox)
    ax.axis('off')
    tab = ax.table(cellText=rows, colLabels=columns, colWidths=widths,
                   cellLoc='center', colLoc='center', bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(size)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(1)
        cell.PAD = .055
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#e4f2ea' if r - 1 in highlight_rows else (PALE if r % 2 else '#f7f9fb'))
            cell.set_text_props(color=NAVY)
    return tab


def cfg_lookup(stage):
    return {t['config_id']: t for t in stage['plan']['trials']}


def setting_text(cfg):
    return (f"alpha0={cfg['alpha0']:g}; p={cfg['penalty_exponent']:g}; "
            f"gamma=min(1,{cfg['step_scale']:g}/(t+{cfg['step_offset']:g})^{cfg['step_exponent']:g}); "
            f"rho=min(1,{cfg['rho_scale']:g}/(t+{cfg['rho_offset']:g})^{cfg['rho_exponent']:.6g}).")


def page_overview(screen, long, confirmation):
    fig = header('Independent FRAMES tuning',
                 'Corrected Quiroga-style physical design • noiseless two-qubit rank-one recovery', 1)
    selected = long['selection']['selected_config_ids']
    confirm_counts = stage_counts(confirmation)
    if selected and confirmation and confirmation['audit'] and confirmation['audit']['phase_complete_and_valid']:
        title = (f"Frozen setting: {confirm_counts['passes']}/{confirm_counts['expected']} "
                 'confirmation trajectories pass the joint gate')
    elif selected:
        title = 'A setting passes both calibration channels; confirmation is not yet fully audited'
    else:
        title = 'No setting passed both full-horizon calibration channels'
    paragraph(fig, .055, .819, title, width=102, size=17, leading=.034)
    paragraph(fig, .055, .744,
              'Independent tuning changes FRAMES controls only. Quiroga’s published adaptive update '
              'has not been reproduced successfully; the exact-line-search FGD result remains a '
              'modified diagnostic, not the published competitor.', width=136, size=10.5, leading=.026)
    rows = []
    for label, stage in [('Screen: 9 configurations × 2 channels', screen),
                         ('Long stage: frozen promotions', long),
                         ('Confirmation: 6 channels × 2 streams', confirmation)]:
        counts = stage_counts(stage)
        rows.append([label, f"{counts['validated']}/{counts['expected']}" if stage else 'not opened',
                     str(counts['completed']), str(counts['passes']), str(counts['failures']),
                     str(counts['pending']) if stage else '—'])
    table(fig, [.055, .49, .895, .185],
          ['Declared stage', 'Audited / planned', 'Completed', 'Gate pass', 'Failures', 'Unaudited'],
          rows, widths=[.37, .18, .12, .11, .11, .11], size=10.5)
    audited = [r for stage in (screen, long, confirmation) if stage
               for r in stage['trials'] if r.get('valid')]
    updates = sum(r['completed_steps'] for r in audited)
    charged = sum(r['charged_rows'] for r in audited)
    fig.text(.055, .473,
             f'Audited work: {updates / 1e6:g} million updates; {charged / 1e6:g} million charged rows. '
             'Fresh-start prefixes included; the 2,000-step timing pilot is excluded.',
             fontsize=8.8, color=MUTED)
    fig.text(.055, .44, 'Shared physical problem and unchanged gate', size=13, weight='bold', color=NAVY)
    paragraph(fig, .055, .401,
              'n=2, d=4; Haar-unitary truth; rank-one factor. All 128 rows: 16 global probes '
              'and an 8-outcome POVM with a=b=0.1. Paired normalized Gell-Mann initialization; '
              'batch32 and operator-norm radius tau=2.', width=73, size=10.5, leading=.028)
    paragraph(fig, .055, .257,
              'Joint gate: fidelity >=0.99; full-design RMSE <=1e-6; '
              'normalized TP residual <=1e-3. Every component must pass. '
              'Numerical failure is never a success.', width=73, size=10.5, leading=.028)
    fig.text(.545, .44, 'Method-specific penalty: independently tuned', size=13, weight='bold', color=NAVY)
    fig.text(.545, .391, r'$Q_{F,t}(U)=\frac{1}{2m}\|A(UU^H)-y\|_2^2+\alpha_F(t)H(U)$',
             size=12.5, color=NAVY)
    fig.text(.545, .332, r'$\alpha_F(t)=\alpha_0(t+1)^p,\quad \beta_t=[2\alpha_F(t)]^{-1},\quad p>0$',
             size=12.5, color=NAVY)
    paragraph(fig, .545, .278,
              'H(U) is the squared, unnormalized TP residual. The initial alpha0 and complete '
              'schedule are fixed before a run; the penalty then increases. '
              'No shared TP coefficient is imposed on Quiroga/FGD.',
              width=64, size=10.5, leading=.028)
    if selected:
        cfg = cfg_lookup(long)[selected[0]]
        paragraph(fig, .055, .124, f"Selected configuration: {selected[0]}. {setting_text(cfg)}",
                  width=140, size=10, leading=.025)
    else:
        paragraph(fig, .055, .124,
                  'The predeclared rule blocks confirmation. Further tuning requires a new declared '
                  'round; these failures remain in the record.', width=138, size=10, leading=.025)
    return fig


def gamma_label(cfg):
    return f"{cfg['step_scale']:g}/(t+{cfg['step_offset']:g})"


def rho_label(cfg):
    exponent = '2/3' if math.isclose(cfg['rho_exponent'], 2/3) else f"{cfg['rho_exponent']:g}"
    return f"{cfg['rho_scale']:g}/(t+{cfg['rho_offset']:g})^{exponent}"


def page_screen(screen):
    fig = header('What was tuned, and what advanced',
                 'Predeclared nine-arm screen • channels 0 and 1 • 100,000 updates per trajectory', 2)
    configs = cfg_lookup(screen)
    selected = screen['selection']['selected_config_ids']
    ranking = screen['selection']['screen_order_ranking']
    rows = []
    for rank, row in enumerate(ranking, 1):
        cfg = configs[row['config_id']]
        rows.append([rank, row['config_id'], sci(cfg['alpha0'], 0), f"{cfg['penalty_exponent']:g}",
                     gamma_label(cfg), rho_label(cfg), f"{row['gate_passes']}/2",
                     'failed' if row['worst_endpoint_deficit'] is None else f"{row['worst_endpoint_deficit']:.3f}",
                     'yes' if row['config_id'] in selected else '—'])
    table(fig, [.055, .365, .895, .43],
          ['Rank', 'Configuration ID', 'alpha0', 'p', 'gamma(t)', 'rho(t)', 'Pass', 'Worst D', 'Advance'],
          rows, widths=[.055, .195, .085, .065, .145, .185, .07, .105, .095], size=10,
          highlight_rows=[i for i, row in enumerate(ranking) if row['config_id'] in selected])
    paragraph(fig, .055, .32,
              'gamma and rho are capped at 1. Six arms vary initial penalty and positive growth '
              'exponent; three arms change step size or momentum one at a time around the incumbent. '
              'The complete candidate grid was frozen before observing its outcomes.',
              width=139, size=10.5, leading=.027)
    paragraph(fig, .055, .222,
              'Selection order: fewest numerical failures; most endpoint joint-gate passes; smallest '
              'worst-channel deficit D; smallest mean log D; configuration ID. Top two plus the '
              'incumbent advance, deduplicated. Every long trajectory starts afresh, including momentum.',
              width=139, size=10.5, leading=.027)
    fig.text(.055, .105,
             r'$D=\max\{\mathrm{RMSE}/10^{-6},\ \mathrm{TP}_{\mathrm{norm}}/10^{-3},\ '
             r'\max(0,1-F)/0.01\}$', fontsize=12, color=NAVY)
    fig.text(.055, .069,
             '100,000 updates × 32 sampled rows = 3.2 million charged rows per trial. '
             'Prefix failures are not proofs of nonconvergence.', fontsize=9.5, color=MUTED)
    return fig


def draw_curve(ax, long, metric, *, compact=False):
    ids = sorted({curve['config_id'] for curve in long['curves'].values()})
    palette = {name: color for name, color in zip(ids, [BLUE, PURPLE, '#b27b19'])}
    for trial_id, curve in sorted(long['curves'].items()):
        x = curve['charged_rows'] / 1e6
        values = curve[metric]
        mask = (x > 0) & np.isfinite(values) & (values > 0)
        ax.plot(x[mask], values[mask], color=palette[curve['config_id']], linewidth=1.65,
                 linestyle='-' if curve['channel'] == 0 else '--',
                 label=f"{curve['config_id']} · c{curve['channel']}")
    threshold = .5e-12 if metric == 'mean_loss' else .001
    ax.axhline(threshold, color=GREEN, linewidth=1.25, linestyle=':',
               label='RMSE gate equivalent' if metric == 'mean_loss' else 'TP gate')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Charged data-row accesses (millions)', fontsize=9.5 if compact else 11)
    ax.set_ylabel('Measurement mean loss' if metric == 'mean_loss' else 'Normalized TP residual',
                  fontsize=10 if compact else 11)
    ax.grid(True, which='major', color='#d9e1e8', linewidth=.7)
    ax.tick_params(labelsize=9 if compact else 10)
    ax.legend(loc='best', fontsize=7.8 if compact else 9, frameon=True, framealpha=.93)
    ax.spines[['top', 'right']].set_visible(False)


def page_long(long):
    fig = header('Full-horizon measurement loss and TP error',
                 'Channels 0 and 1 • 400,000 updates per trajectory • identical physical data and initial factors', 3)
    draw_curve(fig.add_axes([.078, .447, .40, .357]), long, 'mean_loss', compact=True)
    draw_curve(fig.add_axes([.55, .447, .40, .357]), long, 'normalized_tp', compact=True)
    configs = {t['id']: t for t in long['plan']['trials']}
    rows = []
    for row in long['trials']:
        cfg = configs[row['id']]
        metrics = row.get('metrics', {})
        persistent = row.get('persistent_recorded_gate_step')
        rows.append([cfg['config_id'], row.get('channel', cfg['channel']),
                     sci(metrics.get('rmse')), sci(metrics.get('normalized_tp_violation')),
                     '—' if not metrics else sci(1 - metrics['process_fidelity']),
                     'PASS' if row.get('gate_pass') else ('FAIL' if row.get('valid') else 'unaudited'),
                     '—' if persistent is None else f'{persistent:,}'])
    table(fig, [.055, .137, .895, .216],
          ['Configuration', 'Channel', 'RMSE', 'TP normalized', 'Infidelity (1−F)', 'Joint gate', 'Persistent gate step'],
          rows, widths=[.19, .07, .145, .16, .17, .10, .165], size=10)
    paragraph(fig, .055, .10,
              'Mean loss = RMSE²/2, without TP penalties. Zero-row initial points are omitted on log x axes. '
              'Charged rows exclude TP, LMO, metrics and compilation; they are not wall time. '
              'Persistent means all later recorded checkpoints pass (4,000-update spacing). '
              'At 400,000, this certifies only the final point—not a stable tail.',
              width=165, size=8.7, leading=.020)
    return fig


def parity_text(parity):
    if parity is None:
        return 'Old-incumbent 400k backend parity: pending; no archived parity report is available.'
    status = str(parity.get('status', 'unspecified'))
    if status == 'pass' and 'worst_final_factor_abs_difference' in parity:
        return ('Old-incumbent 400k parity: PASS numerically, not byte-identical across NumPy versions. '
                f"Max factor difference {parity['worst_final_factor_abs_difference']:.2e}; "
                f"max checkpoint RMSE difference {parity['worst_rmse_abs_difference_all_checkpoints']:.2e}. "
                'The original 1/2 gate outcome is reproduced.')
    return (f'Old-incumbent 400k backend parity: recorded status = {status}. '
            'Exact differences and provenance are retained in the evidence JSON.')


def page_confirmation(confirmation, long, screen, parity, validation=None):
    fig = header('Frozen-setting confirmation and limits',
                 'Channels 2–7 were exposed in earlier QPT work: confirmation, not an unseen final test', 4)
    if confirmation is not None:
        by_id = {r['id']: r for r in confirmation['trials']}
        rows = []
        for trial in confirmation['plan']['trials']:
            row = by_id.get(trial['id'], {})
            metrics = row.get('metrics', {})
            persistent = row.get('persistent_recorded_gate_step')
            rows.append([trial['channel'], trial['sampling_replicate'],
                         sci(metrics.get('rmse')), sci(metrics.get('normalized_tp_violation')),
                         '—' if not metrics else sci(1 - metrics['process_fidelity']),
                         ('PASS' if row['gate_pass'] else 'FAIL') if row.get('valid') else 'pending',
                         '—' if persistent is None else f'{persistent // 1000}k'])
        table(fig, [.055, .302, .638, .497],
              ['Channel', 'Stream', 'RMSE', 'TP norm.', 'Infidelity (1−F)', 'Gate', 'Persist. step'], rows,
              widths=[.09, .08, .17, .17, .20, .10, .19], size=8.9)
        counts = stage_counts(confirmation)
        paragraph(fig, .055, .269,
                  f"{counts['validated']}/{counts['expected']} audited trajectories; "
                  f"{counts['passes']} gate passes; {counts['failures']} numerical failures. "
                  'Two streams per channel, one initialization. Sampling streams are not independent channels.',
                  width=108, size=9.4, leading=.023)
    else:
        paragraph(fig, .055, .785,
                  'Confirmation has not been opened. No eligible all-channel full-horizon setting '
                  'was selected, or the frozen confirmation plan is not yet available. '
                  'No confirmation result is inferred from the calibration screen.',
                  width=84, size=13, leading=.038)
        paragraph(fig, .055, .581,
                  'Next action: declare a separate calibration round if the long stage has no winner. '
                  'Keep this round intact; do not pool newly tuned candidates into the frozen selection. '
                  'If a winner exists, freeze its full configuration before opening confirmation.',
                  width=92, size=11.5, leading=.033)
    fig.text(.726, .807, 'What remains to establish', size=12, weight='bold', color=NAVY)
    paragraph(fig, .726, .767,
              'This is noiseless n=2, rank-one recovery. No noisy, higher-rank or higher-qubit '
              'performance follows from these runs.\n\n'
              'The JAX CPU dense sensing bridge accelerates local calibration; it is not a '
              'large-qubit memory/scaling result. Scan timing includes compilation and is '
              'not a controlled head-to-head timing benchmark.\n\n'
              'Quiroga’s printed adaptive update remains unresolved. Freeze both independently '
              'tuned competitors before final paired tests.',
              width=34, size=10, leading=.026)
    if validation is not None:
        check(validation['exit_code'] == 0, 'Regression suite did not pass.')
        match = re.search(r'(\d+) passed, (\d+) subtests passed', validation['log'])
        check(match is not None, 'Unrecognized regression summary.')
        fig.text(.055, .214, f'Validation: {match[1]} tests and {match[2]} subtests passed; '
                 'independent archive audits passed for every planned scientific trajectory.',
                 fontsize=9.2, color=MUTED)
    paragraph(fig, .055, .188, parity_text(parity), width=143, size=9.2, leading=.023)
    source = screen['plan']['source_commit']
    selection = long['selection']
    reserved = 'Final held-out channels 1000–1019 are excluded by every loaded plan; no held-out files are read.'
    paragraph(fig, .055, .123,
              f'Source {source[:12]}: {len(screen["plan"]["source_sha256"])} Python files pinned. '
              f'Screen plan {digest(ROOT / "plan_screen.json")[:12]}; '
              f'long selection {digest(ROOT / "selection_long.json")[:12]}. '
              'Full source, harness, input-archive and report hashes accompany this PDF.',
              width=150, size=8.6, leading=.021)
    paragraph(fig, .055, .073, reserved + ' Earlier repositories and results are preserved.',
              width=153, size=8.6, leading=.021)
    return fig


def export_curve(long, metric, destination):
    fig, ax = plt.subplots(figsize=(9.5, 5.7), facecolor='white')
    draw_curve(ax, long, metric)
    ax.set_title('FRAMES local full-horizon calibration · n=2, rank1, noiseless',
                 color=NAVY, pad=14, fontsize=12)
    fig.text(.1, .014, 'Charged rows are not wall time. Mean loss excludes TP penalties. '
                      'Calibration channels only; no Ruche.', fontsize=8.4, color=MUTED)
    fig.tight_layout(rect=[0, .04, 1, 1])
    with destination.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    for path in (PDF, LOSS_PNG, TP_PNG, EVIDENCE):
        check(not path.exists(), f'Refusing to overwrite existing artifact: {path}.')
    screen = load_stage('screen')
    long = load_stage('long', parent_selection=ROOT / 'selection_screen.json')
    confirmation = load_stage('confirmation', required=False,
                               parent_selection=ROOT / 'selection_long.json')
    if confirmation is not None:
        check(len(long['selection']['selected_config_ids']) == 1, 'Confirmation requires one frozen winner.')
        check(len(confirmation['plan']['trials']) == 12, 'Confirmation cohort must have exactly 12 trials.')
        check({(t['channel'], t['sampling_replicate']) for t in confirmation['plan']['trials']}
              == {(c, r) for c in range(2, 8) for r in (0, 1)}, 'Confirmation pairing changed.')
    parity_path = ROOT / 'backend_parity.json'
    parity = read_input(parity_path) if parity_path.exists() else None
    validation_path = ROOT / 'validation_tests.json'
    validation = read_input(validation_path) if validation_path.exists() else None
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.labelcolor': NAVY, 'text.color': NAVY})
    with PDF.open('xb') as stream:
        with PdfPages(stream) as pdf:
            metadata = pdf.infodict()
            metadata.update(Title='Independent FRAMES local calibration — 12 October 2026',
                            Author='QPT project', Subject='Audited round-one results; not final unseen comparison',
                            Keywords='QPT FRAMES independent calibration noiseless n2 rank1')
            for fig in (page_overview(screen, long, confirmation), page_screen(screen),
                        page_long(long), page_confirmation(confirmation, long, screen, parity, validation)):
                pdf.savefig(fig)
                plt.close(fig)
    export_curve(long, 'mean_loss', LOSS_PNG)
    export_curve(long, 'normalized_tp', TP_PNG)
    evidence = {
        'schema': 'qpt_frames_local_report_v1', 'report_date': '2026-10-12',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Round-one local calibration only; no Ruche and no final held-out data access.',
        'report_builder_sha256': digest(__file__), 'source_commit': screen['plan']['source_commit'],
        'source_sha256': screen['plan']['source_sha256'], 'gate': GATE,
        'input_sha256': INPUTS,
        'counts': {stage['phase']: stage_counts(stage) for stage in (screen, long, confirmation) if stage},
        'screen_selection': screen['selection'], 'long_selection': long['selection'],
        'confirmation_audit': None if confirmation is None else confirmation['audit'],
        'backend_parity': parity,
        'validation_tests': validation,
        'timing_pilot_qualification': ('The preliminary 2,000-step timing pilot is excluded from '
                                      'scientific audit counts: its pre-freeze runner snapshot was not retained.'),
        'outputs_sha256': {str(path): digest(path) for path in (PDF, LOSS_PNG, TP_PNG)},
        'limitations': [
            'Previously exposed calibration/confirmation channels; final channels1000..1019 not read.',
            'Two confirmation sampling streams nested within each of six channels; one initialization each.',
            'Fixed initial FRAMES penalty with increasing schedule; not shared with fixed FGD penalty.',
            'Printed Quiroga reproduction unresolved; modified exact-line-search FGD is diagnostic.',
            'Dense JAX CPU bridge is not a higher-qubit scalability demonstration.',
            'Compile-inclusive scan times cannot support fair time-to-gate comparisons.',
            'Charged rows exclude TP, LMO, checkpoint and compilation work.',
        ],
    }
    with EVIDENCE.open('x') as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'pdf': str(PDF), 'pages': 4, 'pdf_sha256': digest(PDF),
                      'plots': [str(LOSS_PNG), str(TP_PNG)], 'evidence': str(EVIDENCE)}))


if __name__ == '__main__':
    main()
