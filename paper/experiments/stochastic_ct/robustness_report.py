"""Dense-trace evaluation and compact reporting for the CT robustness campaign.

Numerical evidence is immutable. Descriptive repetitions are aggregated within
slice, then acquisition group. Bootstrap intervals resample groups, not seeds.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from statistics import median
import tempfile
from types import SimpleNamespace

import numpy as np

try:
    from . import nonnegative_plots as plotting
except ImportError:
    import nonnegative_plots as plotting

CAP = 69120
BUNDLE = Path('/home/tony/frankwolfe/paper/experiments/stochastic_ct')
LABELS = {
    'sframes_joint': ('SFRAMES (joint tune)', '#333333'),
    'sframes_loss': ('SFRAMES (loss tune)', '#0072B2'),
    'sframes_full': ('SFRAMES (full batch)', '#E69F00'),
    'frames': ('Deterministic FraMES', '#CC79A7'),
    'prox_sarah': ('ProxSARAH-v3', '#009E73'),
    'spring_sarah': ('SPRING-SARAH', '#D55E00'),
    'joint_lbfgsb': ('Joint L-BFGS-B', '#666600'),
}
METRICS = ('raw_loss', 'feasible_loss', 'validation_loss',
           'reference_relative_error', 'relative_d_distance')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped(rows, groups):
    by_slice = defaultdict(list)
    for sid, value in rows:
        if not np.isfinite(value):
            raise ValueError('Nonfinite summary metric')
        by_slice[sid].append(float(value))
    slices = {sid: median(values) for sid, values in by_slice.items()}
    grouped_values = defaultdict(list)
    for sid, value in slices.items():
        grouped_values[groups[sid]].append(value)
    group_values = {group: median(values) for group, values in grouped_values.items()}
    return dict(median_across_groups=median(group_values.values()),
                by_group=group_values, by_slice=slices)


def weighted_attainment(rows, groups):
    by_slice = defaultdict(list)
    for sid, value in rows:
        by_slice[sid].append(float(value))
    by_group = defaultdict(list)
    for sid, values in by_slice.items():
        by_group[groups[sid]].append(float(np.mean(values)))
    values = {group: float(np.mean(x)) for group, x in by_group.items()}
    return dict(group_balanced_fraction=float(np.mean(list(values.values()))), by_group=values)


def paired_groups(left, right, groups, *, bootstrap_seed=20261122):
    """Paired relative differences; the interval is descriptive with six groups."""
    common = sorted(set(left) & set(right))
    if not common or set(left) != set(right) or any(right[g] <= 0 for g in common):
        raise ValueError('Paired relative comparison requires positive common groups')
    per_group = defaultdict(list)
    for sid in common:
        per_group[groups[sid]].append(100 * (left[sid] / right[sid] - 1))
    names = sorted(per_group)
    values = np.array([median(per_group[g]) for g in names])
    rng = np.random.default_rng(bootstrap_seed)
    resampled = np.median(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return dict(by_group=dict(zip(names, values.tolist())),
                median_percent_difference=float(np.median(values)),
                bootstrap_95_percentile=np.quantile(resampled, [.025, .975]).tolist(),
                groups=len(names), groups_improved=int(np.sum(values < 0)),
                resampling_unit='acquisition group', bootstrap_replicates=2000,
                interpretation='Descriptive paired group bootstrap; only six groups, not a population guarantee.')


def load_configs(root):
    configs = {}
    for path in sorted((root / 'plans').glob('*.json')):
        plan = read(path)
        for cfg in plan.get('runs', []):
            key = cfg['run_id']
            if key in configs and configs[key] != cfg:
                raise ValueError('Conflicting immutable run configuration: ' + key)
            configs[key] = cfg
    return configs


def memberships(cfg):
    values = cfg.get('memberships')
    if values is not None:
        return values
    return [dict(study=study, variant=cfg['variant'])
            for study in cfg.get('studies', [cfg.get('study', cfg.get('phase'))])]


def read_run(root, cfg):
    folder = root / 'runs' / cfg['run_id']
    if (folder / 'candidate_failure.json').exists():
        failure = read(folder / 'candidate_failure.json')
        if cfg['phase'] != 'development' or failure['config'] != cfg:
            raise ValueError('Invalid retained development failure')
        return dict(config=cfg, status='numerical_failure', error=failure['error'], trace=[], directory=str(folder))
    if not (folder / 'result.json').exists():
        return None
    result, meta = read(folder / 'result.json'), read(folder / 'metadata.json')
    if meta['config'] != cfg:
        raise ValueError('Result configuration changed: ' + cfg['run_id'])
    trace = read(folder / 'trace.json')
    if result['status'] not in ('completed', 'numerical_failure'):
        raise ValueError('Unknown/operational result status: ' + result['status'])
    if result['status'] != 'completed':
        return dict(config=cfg, status=result['status'], error=result.get('error'), trace=trace,
                    metadata=meta, directory=str(folder))
    if not trace or trace[0]['iteration'] != 0:
        raise ValueError('Missing initialization')
    last_k, last_views = 0, 0
    for row in trace:
        k = row['iteration']
        views = max(row['gradient_forward_views'], row['gradient_backward_views'])
        if k < last_k or k > last_k + 1 or views < last_views or views > cfg['view_budget']:
            raise ValueError('Sparse/nonmonotone/over-budget trace: ' + cfg['run_id'])
        last_k, last_views = k, views
    norm = meta['initial_factor_frobenius_norm']
    rows = []
    for row in trace:
        rows.append(dict(row, relative_d_distance=row['d_distance']/norm,
            comparison_views=max(row['gradient_forward_views'], row['gradient_backward_views']),
            comparison_oracle_calls=row['gradient_calls']))
    return dict(config=cfg, status='completed', trace=rows, metadata=meta,
                initial_norm=norm, directory=str(folder))


def summarize(root, stage=None):
    root = Path(root)
    if (root/'staged_report.json').exists():
        frozen_report = read(root/'staged_report.json')
        if digest(root/'report_summary.json') != frozen_report['summary_sha256']:
            raise ValueError('Rendered report summary changed before publication retry')
        return read(root/'report_summary.json')
    cohort = read(root / 'cohort.json')
    groups = dict(cohort['evaluation_slice_groups'])
    groups.update(cohort.get('confirmation_slice_groups', {}))
    groups.update({sid: 'development_' + sid for sid in cohort['development_slices']})
    configs = load_configs(root)
    input_catalog = read(root/'inputs.json') if (root/'inputs.json').exists() else {}
    reference_norms = {}
    studies = defaultdict(list)
    completed, reused, failed, pending = [], [], [], []
    for cfg in configs.values():
        run = read_run(root, cfg)
        if run is None:
            pending.append(cfg['run_id'])
            continue
        if run['status'] != 'completed':
            failed.append(dict(run_id=cfg['run_id'], status=run['status'], error=run.get('error')))
            continue
        reference_key = f"{cfg['slice_id']}:{cfg['rank']}:standard"
        if reference_key not in reference_norms:
            if cfg.get('initialization', 'standard') == 'standard':
                reference_norms[reference_key] = run['initial_norm']
            else:
                with np.load(input_catalog[reference_key]['calibration_file'], allow_pickle=False) as saved:
                    reference_norms[reference_key] = float(np.hypot(np.linalg.norm(saved['U']), np.linalg.norm(saved['V'])))
        for row in run['trace']:
            row['fixed_reference_relative_d_distance'] = row['d_distance']/reference_norms[reference_key]
        completed.append(cfg['run_id'])
        if run['metadata'].get('producer') == 'historical_reuse':
            reused.append(cfg['run_id'])
        for member in memberships(cfg):
            studies[member['study']].append((member, run))
    result = dict(schema_version=1, updated_utc=datetime.now(timezone.utc).isoformat(),
        stage=stage, planned_physical_runs=len(configs), completed_physical_runs=len(completed),
        reused_physical_runs=len(reused), newly_executed_completed_runs=len(completed)-len(reused),
        failed_runs=failed, pending_runs=pending, budget_per_direction=CAP,
        aggregation='Median over seeds, slices within acquisition group, then acquisition groups.',
        optimizer_timing=False, studies={})
    for study, members in studies.items():
        cells = defaultdict(list)
        for member, run in members:
            cfg = run['config']
            key = (member['variant'], cfg['n_train'], cfg['rank'], cfg.get('radius_scale', 2),
                   cfg.get('initialization', 'standard'), json.dumps(cfg['parameters'], sort_keys=True))
            cells[key].append(run)
        records = []
        for key, runs in sorted(cells.items()):
            variant, n, rank, radius, initialization, parameter_key = key
            metrics = {metric: grouped([(r['config']['slice_id'], r['trace'][-1][metric])
                        for r in runs], groups) for metric in METRICS}
            for metric in ('d_distance', 'fixed_reference_relative_d_distance'):
                metrics[metric] = grouped([(r['config']['slice_id'], r['trace'][-1][metric]) for r in runs], groups)
            stationarity = {}
            for metric in ('smoothed_fw_gap', 'mean_smoothed_fw_gap', 'stationarity_beta'):
                if all(metric in r['trace'][-1] for r in runs):
                    stationarity[metric] = grouped([(r['config']['slice_id'], r['trace'][-1][metric]) for r in runs], groups)
            budgets = {}
            for budget in (8640, 17280, 34560, 69120):
                if any(r['config']['view_budget'] < budget for r in runs):
                    continue
                observations = [(r, [row for row in r['trace'] if row['comparison_views'] <= budget][-1]) for r in runs]
                budgets[str(budget)] = dict(
                    relative_evaluated_loss=grouped([(r['config']['slice_id'], row['feasible_loss']/r['trace'][0]['feasible_loss']) for r,row in observations], groups),
                    relative_raw_feasibility=grouped([(r['config']['slice_id'], row['fixed_reference_relative_d_distance']) for r,row in observations], groups),
                    joint_attainment={str(epsilon): weighted_attainment([(r['config']['slice_id'],
                        row['feasible_loss'] <= .30*r['trace'][0]['feasible_loss'] and row['fixed_reference_relative_d_distance'] <= epsilon)
                        for r,row in observations], groups) for epsilon in (.001, .005, .01, .02)})
            inclusive = []
            for r in runs:
                states = [row for row in r['trace'] if max(row['forward_views'], row['backward_views']) <= CAP]
                inclusive.append((r['config']['slice_id'], states[-1]['feasible_loss']))
            records.append(dict(variant=variant, n_train=n, rank=rank, radius_scale=radius,
                initialization=initialization, parameters=json.loads(parameter_key), physical_runs=len(runs),
                reused_runs=sum(r['metadata'].get('producer') == 'historical_reuse' for r in runs),
                run_ids=[r['config']['run_id'] for r in runs], metrics=metrics,
                stationarity_metrics=stationarity, budget_summaries=budgets,
                evaluated_loss_at_total_operator_cap=grouped(inclusive, groups)))
        result['studies'][study] = records
    write(root / 'report_summary.json', result)
    return result


def main_cells(summary):
    records = summary['studies'].get('main', [])
    cells = {(c['n_train'], c['variant']): c for c in records}
    if len(cells) != len(records):
        raise ValueError('Main comparison has duplicate settings or multiple parameter choices')
    if set(cells) != {(n, name) for n in (120, 360, 1080, 2160) for name in LABELS}:
        raise ValueError('The main comparison is incomplete or has duplicate settings')
    return cells


def quality_table(summary):
    cells = main_cells(summary)
    lines = [r'\begin{table}[!htbp]', r'\centering',
        r'\caption{Endpoint evaluated loss (multiplied by $10^3$) / raw relative nonnegativity violation (percent), at 69,120 gradient views per direction. Summaries take medians over repetitions, slices within groups, and groups. Bold marks the best value in each metric and view count. L-BFGS-B targets $D$ alone; other methods target $C\cap D$. Full raw, held-out, and reference-image metrics appear in the reproducibility supplement.}',
        r'\label{tab:ct-results}', r'\small', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Method & $N=120$ & $N=360$ & $N=1{,}080$ & $N=2{,}160$ \\', r'\midrule']
    for name, (label, _) in LABELS.items():
        values = []
        for n in (120, 360, 1080, 2160):
            numbers = []
            for metric, scale in [('feasible_loss', 1000), ('relative_d_distance', 100)]:
                value = cells[n, name]['metrics'][metric]['median_across_groups']
                best = min(cells[n, other]['metrics'][metric]['median_across_groups'] for other in LABELS)
                number = f'{scale*value:.2f}'
                if np.isclose(value, best, rtol=1e-10, atol=1e-14):
                    number = r'\boldsymbol{' + number + '}'
                numbers.append(number)
            values.append('$' + ' / '.join(numbers) + '$')
        lines.append(label.replace('SPRING-SARAH', 'SPRING--SARAH') + ' & ' + ' & '.join(values) + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def full_appendix(summary, selection):
    lines = ['# Expanded CT evaluation results', '',
        'All settings were frozen before evaluation. These are algorithm comparisons on the stated domains; they are not claims of optimal CT reconstruction.', '',
        'Endpoint loss and raw feasibility are separate. Spectral-constrained factors are clipped to nonnegativity and radially restored to their balls for evaluated metrics. L-BFGS-B factors are evaluated without spectral repair. Gradient work includes all L-BFGS-B trials; curvature and diagnostic work remain separately recorded.', '',
        'The fresh confirmation set contains six previously uninspected slices within familiar acquisition groups, not six new independent specimen groups. Bootstrap intervals below resample acquisition groups and remain descriptive with six groups.', '',
        f"Of {summary['completed_physical_runs']} completed physical runs, {summary.get('reused_physical_runs', 0)} are authenticated historical trajectories and {summary.get('newly_executed_completed_runs', summary['completed_physical_runs'])} were newly executed. Reused trajectories retain their original recorded dependency versions; unrecorded historical versions remain explicitly unknown. They are not described as having run in the new environment.", '',
        '## Frozen selected parameters', '', '```json', json.dumps(selection['global_selection'], indent=2), '```', '',
        '## Complete endpoint summaries', '']
    for study, cells in summary['studies'].items():
        lines += ['### ' + study.replace('_', ' '), '',
            '| Variant | Views | Rank | Radius | Start | Parameters | Runs | Raw loss | Evaluated loss | Held-out loss | Reference error | Raw violation (%) |',
            '| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for c in cells:
            m = c['metrics']
            values = [m[k]['median_across_groups'] for k in METRICS]
            lines.append('| ' + ' | '.join([c['variant'], str(c['n_train']), str(c['rank']), str(c['radius_scale']),
                c['initialization'], '`' + json.dumps(c['parameters'], sort_keys=True) + '`', str(c['physical_runs']),
                *(f'{v:.7g}' for v in values[:-1]), f'{100*values[-1]:.5g}']) + ' |')
        lines.append('')
    lines += ['## Longer-run stationarity diagnostics', '',
        'These use the true full gradient and each smoothing algorithm\'s own beta schedule. They are not a common stationarity score for the proximal baselines. The cumulative mean includes initialization once; terminal duplicate records do not add a gap observation.', '',
        '| Variant | Views | Final true smoothed gap | Cumulative mean gap | Final beta | Raw distance |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for cells in summary['studies'].values():
        for c in cells:
            stat = c.get('stationarity_metrics', {})
            if stat:
                values = [stat[k]['median_across_groups'] for k in ('smoothed_fw_gap','mean_smoothed_fw_gap','stationarity_beta')]
                lines.append('| ' + ' | '.join([c['variant'],str(c['n_train']),*(f'{x:.7g}' for x in values),
                    f"{c['metrics']['d_distance']['median_across_groups']:.7g}"]) + ' |')
    lines += ['', '## Initialization feasibility and budget sensitivity', '',
        'Initialization-study feasibility uses the fixed standard-reference factor norm as well as each run\'s original normalization. Thus changing the start does not change the denominator in the cross-start comparison. Full raw distances, fixed-reference distances, and threshold attainment at 8,640/17,280/34,560/69,120 views are stored in report_summary.json. The full tuning grid is evaluated only at the screening budget; longer-budget sensitivity concerns shortlisted settings.', '',
        '| Variant | Views | Start | Raw distance | Fixed-reference violation (%) |',
        '| --- | ---: | --- | ---: | ---: |']
    for c in summary['studies'].get('initialization', []):
        lines.append('| ' + ' | '.join([c['variant'],str(c['n_train']),c['initialization'],
            f"{c['metrics']['d_distance']['median_across_groups']:.7g}",
            f"{100*c['metrics']['fixed_reference_relative_d_distance']['median_across_groups']:.5g}"]) + ' |')
    lines += ['## Failures', '', json.dumps(summary['failed_runs'], indent=2), '',
        'Development failures are retained and excluded from selection with their reasons. An evaluation or operational failure halts publication. Full dense traces and manifests are in the accompanying compressed result archive.', '']
    return '\n'.join(lines)


def ablation_results(summary):
    """Compact measured outcomes; never compare unmatched cohorts or settings."""
    studies = summary['studies']
    names = ('batch', 'averaging_control', 'fresh_confirmation', 'radius')
    if not any(studies.get(name) for name in names) and summary.get('stage') == 'fixture':
        return ''

    def cell(study, variant, n=1080, radius=2):
        selected = [c for c in studies.get(study, []) if c['variant'] == variant
            and c['n_train'] == n and c['rank'] == 64 and c['radius_scale'] == radius
            and c['initialization'] == 'standard']
        if len(selected) != 1:
            raise ValueError('Incomplete or ambiguous ablation evidence: ' + study + '/' + variant)
        if len(selected[0]['metrics']['feasible_loss']['by_slice']) != 6:
            raise ValueError('Ablation cohorts must contain six slices')
        return selected[0]

    def matched(left, right, *, same_parameters=False, same_repetitions=True, vary_radius=False):
        keys = ('n_train', 'rank', 'initialization') + (() if vary_radius else ('radius_scale',))
        if any(left[key] != right[key] for key in keys):
            raise ValueError('Ablation settings do not match')
        for metric in ('feasible_loss', 'fixed_reference_relative_d_distance'):
            for unit in ('by_slice', 'by_group'):
                if set(left['metrics'][metric][unit]) != set(right['metrics'][metric][unit]):
                    raise ValueError('Ablation cohorts do not match')
        if same_parameters and left['parameters'] != right['parameters']:
            raise ValueError('Paired averaging/radius parameters do not match')
        if same_repetitions and left['physical_runs'] != right['physical_runs']:
            raise ValueError('Ablation repetition counts do not match')

    def metric(c, key):
        value = c['metrics'][key]['median_across_groups']
        if not np.isfinite(value) or value < 0:
            raise ValueError('Invalid ablation endpoint metric')
        return value

    def ratio(left, right):
        denominator = metric(right, 'feasible_loss')
        if denominator == 0:
            raise ValueError('Ablation loss ratio has a zero denominator')
        return metric(left, 'feasible_loss')/denominator

    def violation(c):
        return 100*metric(c, 'fixed_reference_relative_d_distance')

    def numbers(values):
        return ','.join(f'{value:.3g}' for value in values)

    batches = [cell('batch', variant) for variant in
               ('sframes_b16','sframes_b64','sframes_b256','sframes_full')]
    base = batches[1]
    for item in batches:
        matched(item, base, same_repetitions=item['variant'] != 'sframes_full')
    no_average = cell('averaging_control', 'sframes_noavg_paired')
    matched(no_average, base, same_parameters=True)
    fresh = [(cell('fresh_confirmation','sframes_joint',n),
              cell('fresh_confirmation','prox_sarah',n)) for n in (360,1080)]
    for sf, prox in fresh:
        matched(sf, prox)
    radii = [cell('radius','sframes_joint',radius=radius) for radius in (1,2,4)]
    for item in radii:
        matched(item, radii[1], same_parameters=True, vary_radius=True)
    return (r'At equal view budgets, separately tuned batches \(16,64,256,N\) give '
        r'evaluated-loss ratios \(' + numbers(ratio(c,base) for c in batches) + r'\) versus batch 64. '
        r'In the matched averaging control, removing averaging gives loss ratio \('
        + numbers([ratio(no_average,base)]) + r'\), with raw violation \('
        + numbers([violation(base)]) + r'\%\to ' + numbers([violation(no_average)]) + r'\%\). '
        r'On six fresh slices within the same acquisition groups, SFRAMES/ProxSARAH loss ratios are \('
        + numbers(ratio(sf,prox) for sf,prox in fresh) + r'\) at \(N=360,1080\), with SFRAMES violations \('
        + numbers(violation(sf) for sf,_ in fresh) + r'\%\). '
        r'Radius multipliers \(1,2,4\) give SFRAMES loss ratios \('
        + numbers(ratio(c,radii[1]) for c in radii) + r'\) versus radius 2 and violations \('
        + numbers(violation(c) for c in radii) + r'\%\). '
        r'These descriptive ratios compare the group-median evaluated losses; the batch and radius panels use \(N=1080\).' + '\n\n')


def paper_body(source, summary, selection):
    # Preserve the definitions and requested explicit estimator explanation.
    problem = re.search(r'\\paragraph\{Reconstruction problem\.\}.*?(?=\\paragraph\{Stochastic gradient estimator\.\})', source, re.S).group()
    problem = re.sub(r'These three additional baselines minimize the same loss\s+over \\(D\\) alone, so they solve a less constrained application problem\.',
        r'Joint L-BFGS-B minimizes the same loss over \\(D\\) alone.', problem)
    # Avoid regex escaping ambiguities in the predecessor sentence.
    problem = problem.replace('These three additional baselines minimize the same loss\nover \\(D\\) alone, so they solve a less constrained application problem.',
        'Joint L-BFGS-B minimizes the same loss over \\(D\\) alone.')
    estimator = re.search(r'\\paragraph\{Stochastic gradient estimator\.\}.*?(?=\\paragraph\{Evaluation\.\})', source, re.S).group()
    cells = main_cells(summary)
    differences = []
    error_differences = []
    for n in (120, 360, 1080, 2160):
        sf = cells[n, 'sframes_joint']['metrics']
        det = cells[n, 'frames']['metrics']
        lb = cells[n, 'joint_lbfgsb']['metrics']
        differences.append(100*(sf['feasible_loss']['median_across_groups']/det['feasible_loss']['median_across_groups']-1))
        error_differences.append(100*(sf['reference_relative_error']['median_across_groups']/lb['reference_relative_error']['median_across_groups']-1))
    chosen = selection['global_selection']
    joint = chosen['sframes_b64']['joint']['parameters']
    loss = chosen['sframes_b64']['loss']['parameters']
    det = chosen['frames']['joint']['parameters']
    prox = chosen['prox_sarah']['loss']['parameters']['curvature_scale']
    memory = chosen['joint_lbfgsb']['loss']['parameters']['memory']
    text = r'''\section{Numerical experiments}
\label{sec:ct-experiments}
We study the benefit of stochastic updates, parameter transfer, and sensitivity
to the feasible domain on low-rank reconstruction from real low-dose CT data.
The main comparison includes SFRAMES, deterministic FraMES, SPRING--SARAH~\cite{spring},
ProxSARAH~\cite{proxsarah}, and nonnegative joint L-BFGS-B.

''' + problem + estimator + r'''\paragraph{Evaluation.}
We use 14 slices from six acquisition groups with
\(N\in\{120,360,1080,2160\}\) nested training views and 360 fixed held-out views.
Five sampling repetitions share the same initial factors in each case;
deterministic algorithms run once. A separate confirmation set contains six
fresh slices from these groups. At a fixed problem size, we test batch sizes
\(16,64,256,N\) and remove gradient averaging in a paired ablation.
Further predefined panels vary rank \(32,64,128\), spectral radius, and
initial factors; a stress panel uses ten sampling repetitions.

We compare the last native state within 69,120 gradient views in each direction.
All L-BFGS-B line-search trials count. Curvature-estimation and diagnostic work
are separately recorded; these are not equal-total-work or runtime comparisons.
Every accepted iterate is recorded. Raw loss is evaluated at the current factors;
evaluated loss uses orthant clipping followed by radial spectral repair for the
bounded methods, and the actual nonnegative factors for L-BFGS-B.
Held-out loss and relative Frobenius error against the high-dose reconstruction
use this evaluated image. Raw nonnegativity violation is
\[
 d_D(U,V)=\bigl(\|U-U_+\|_F^2+\|V-V_+\|_F^2\bigr)^{1/2},
\]
normalized by the initial factor-pair norm. Endpoint summaries take successive
medians over repetitions, slices within groups, and groups. Trajectories show
group-balanced means and one descriptive standard deviation.

\paragraph{Results.}
''' + ('Across the four view counts, joint-tuned SFRAMES changes the group-balanced\n'
       'evaluated loss relative to deterministic FraMES by \\(' + r'\), \('.join(f'{v:+.1f}\\%' for v in differences)
       + '\\), respectively; negative values favor SFRAMES.\n'
       'Its reference-image error relative to L-BFGS-B changes by \\(' + r'\), \('.join(f'{v:+.1f}\\%' for v in error_differences)
       + '\\). These are ratios of the tabulated group summaries, not paired percentage changes.\n') + r'''
Table~\ref{tab:ct-results} reports the loss--feasibility trade-off.
Figures~\ref{fig:ct-trajectories-360} and~\ref{fig:ct-trajectories-1080}
retain trajectories against iterations, gradient-oracle calls, and processed
views. The reproducibility supplement gives all view counts, fresh-confirmation
results, joint parameter surfaces, failures, and the targeted ablations.
Longer diagnostic runs record the true full-gradient smoothed FW gap and its
cumulative average for SFRAMES and FraMES, alongside raw feasibility; this
smoothed gap is not used to rank exactly feasible proximal methods.

''' + ablation_results(summary) + quality_table(summary) + r'''

\paragraph{Implementation and parameter selection.}
All main runs reuse the same training-only SVD-based factorization of a
sensitivity-normalized backprojection of 120 common training views, with its
amplitude fitted to those views. Set \(R_U=2\|U_0\|_{\rm op}\) and
\(R_V=2\|V_0\|_{\rm op}\). Initialization ablations retain these reference radii;
radius ablations use multipliers \(1,2,4\) of the reference factor norms.
SFRAMES uses \(\gamma_k=\gamma_0(k+1)^{-3/4}\) and
\(\beta_k=\beta_0(k+1)^{-1/4}\), with
\(\beta_0=c_\beta/\max(L_U,L_V)\) from initial block-curvature estimates.
Deterministic FraMES uses exact current gradients without averaging and
exponents \(2/3\) and \(1/3\), respectively.
''' + (f'The joint and loss-only SFRAMES selections are \\((c_\\beta,\\gamma_0)=({joint["beta_scale"]:g},{joint["gamma0"]:g})\\) and '
       f'\\(({loss["beta_scale"]:g},{loss["gamma0"]:g})\\); deterministic FraMES uses '
       f'\\(({det["beta_scale"]:g},{det["gamma0"]:g})\\).\n') + r'''
Each family is screened on a joint grid, then shortlisted candidates are
confirmed on two development slices at both original view counts with three
stochastic repetitions. Joint selection prioritizes attaining evaluated loss
below 30\% of initialization and raw relative violation below 1\%; loss-only
selection minimizes evaluated loss. Parameters are then frozen across the
main view counts. Threshold sensitivity and separate condition retuning are
reported in the supplement.

SPRING--SARAH uses 64-view differences, independent full refreshes with
probability \(1/16\), and the practical current-curvature step \(1/(2\widetilde L)\).
ProxSARAH uses the paper's v3 relaxation \(0.99\), full snapshots, and
\(m=\lfloor\sqrt N\rfloor\), \(b=\lfloor\sqrt N/(2/(3\cdot0.99^2))\rfloor\).
''' + (f'Its step is \\(0.400802/\\widehat L\\), with \\(\\widehat L={prox:g}(L_U+L_V)\\). '
       f'Joint L-BFGS-B uses memory {memory}, selected on development data.\n') + r'''
No evaluation slice enters numerical parameter selection.
The \CTReproducibility{} specifies complete grids, data splits, uncertainty
summaries, numerical projection tolerances, and all auxiliary results.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{trajectories_360.pdf}
\caption{Mean trajectories and $\pm1$ descriptive standard deviation for 360 training views, weighting acquisition groups equally. Work axes use the latest recorded state at or below each budget, without interpolating losses. Curves stop at the shortest observed support of their fixed cohort. Iterations count native accepted updates; their costs differ.}
\label{fig:ct-trajectories-360}
\end{figure}
\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{trajectories_1080.pdf}
\caption{The same comparison at 1,080 training views. Proximal baselines and L-BFGS-B coincide at zero orthant distance. Full-batch SFRAMES retains stochastic schedules and averaging; deterministic FraMES uses its own schedules and current exact gradients.}
\label{fig:ct-trajectories-1080}
\end{figure}
'''
    if text.count(r'\begin{figure}') != 2 or text.count(r'\begin{table}') != 1:
        raise ValueError('Report structure changed')
    return text


def publish(root):
    """Build and validate a complete report before changing either published copy."""
    root = Path(root).resolve()
    if (root/'staged_report.json').exists():
        try:
            from .robustness_export import export
        except ImportError:
            from robustness_export import export
        export(root, staged=read(root/'staged_report.json'))
        return read(root/'report_update.json')
    if (root/'report_update.json').exists():
        audit = read(root/'report_update.json')
        if audit.get('status') == 'completed_published':
            try:
                from .robustness_export import export
            except ImportError:
                from robustness_export import export
            export(root)
            return audit
    completion = read(root/'completion.json')
    if completion.get('status') != 'completed':
        raise ValueError('Cannot publish an incomplete experiment')
    summary = summarize(root, 'completed')
    if summary['pending_runs']:
        raise ValueError('Publication requires every planned run')
    configs = load_configs(root)
    if any(configs[c['run_id']]['phase'] != 'development' for c in summary['failed_runs']):
        raise ValueError('Evaluation failures require review')
    publication = read(root/'publication_config.json')
    predecessor = root/'predecessor_bundle'
    for name, expected in publication['predecessor_files'].items():
        if digest(predecessor/name) != expected:
            raise ValueError('Frozen predecessor document changed: ' + name)
    selection = read(root/'selection.json')
    cohort = read(root/'cohort.json')
    groups = dict(cohort['evaluation_slice_groups'])
    main = {}
    for cfg in configs.values():
        labels = [m['variant'] for m in memberships(cfg) if m['study'] == 'main']
        if not labels:
            continue
        run = read_run(root, cfg)
        if run is None or run['status'] != 'completed':
            raise ValueError('Missing successful main trajectory')
        for variant in labels:
            key = cfg['slice_id'], cfg['seed'], cfg['n_train'], variant
            if key in main:
                raise ValueError('Duplicated plotted case')
            main[key] = dict(trace=run['trace'], initial_norm=run['initial_norm'])
    cells = main_cells(summary)
    paired = {}
    for n in (120, 360, 1080, 2160):
        for reference in ('frames', 'prox_sarah', 'joint_lbfgsb'):
            paired[f'{n}:sframes_joint:{reference}'] = {
                metric: paired_groups(cells[n,'sframes_joint']['metrics'][metric]['by_slice'],
                                      cells[n,reference]['metrics'][metric]['by_slice'], groups)
                for metric in ('feasible_loss', 'reference_relative_error')}
    summary['paired_main_comparisons'] = paired
    write(root/'report_summary.json', summary)
    stage = Path(tempfile.mkdtemp(prefix='report_stage_', dir=root))
    report = stage/'report'
    report.mkdir()
    for name in ('sframes_ct_section.tex', 'ct_reproducibility_link.tex'):
        shutil.copyfile(predecessor/'report'/name, report/name)
    statistics = {}
    for n in (360, 1080):
        plotting.plot_size(SimpleNamespace(VARIANTS=LABELS, legend_columns=4), main,
            cohort['evaluation_slices'], groups, n, CAP, CAP//4, report)
        statistics[f'statistics_{n}.json'] = digest(report/f'statistics_{n}.json')
    source = (predecessor/'report/ct_experiments.tex').read_text()
    (report/'ct_experiments.tex').write_text(paper_body(source, summary, selection))
    appendix = (predecessor/'CT_REPRODUCIBILITY.md').read_text()
    appendix += ('\n\n## Expanded mechanism and robustness evaluation\n\n'
        'The current paper uses the separately frozen expanded campaign. Earlier comparisons above remain historical evidence. '
        'See [complete expanded results](ROBUSTNESS_RESULTS.md) and [the protocol](ROBUSTNESS_EVALUATION.md) '
        'for all settings, fresh confirmation, sensitivity studies, failures, and group-level comparisons.\n')
    (stage/'CT_REPRODUCIBILITY.md').write_text(appendix)
    (stage/'ROBUSTNESS_RESULTS.md').write_text(full_appendix(summary, selection) +
        '\n## Paired group comparisons\n\n```json\n' + json.dumps(paired, indent=2) + '\n```\n')
    shutil.copyfile(root/'frozen_sources/ROBUSTNESS_EVALUATION.md', stage/'ROBUSTNESS_EVALUATION.md')
    with (stage/'build.log').open('w') as stream:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','sframes_ct_section.tex'],
            cwd=report, stdout=stream, stderr=subprocess.STDOUT, check=True)
    log = (report/'sframes_ct_section.log').read_text()
    if 'Overfull \\hbox' in log or 'undefined references' in log:
        raise ValueError('Report overflow or unresolved references; stage retained')
    info = subprocess.check_output(['pdfinfo', str(report/'sframes_ct_section.pdf')], text=True)
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    if pages > 6:
        raise ValueError('Expanded paper unexpectedly exceeds six pages; stage retained')
    audit = dict(status='completed_published', generated_utc=datetime.now(timezone.utc).isoformat(),
        stage=str(stage), pages=pages, figures=2, tables=1,
        physical_runs=summary['completed_physical_runs'],
        reused_physical_runs=summary['reused_physical_runs'],
        newly_executed_completed_runs=summary['newly_executed_completed_runs'],
        evaluation_runs=sum(c['phase']=='evaluation' for c in configs.values()),
        development_runs=sum(c['phase']=='development' for c in configs.values()),
        failed_development_runs=len(summary['failed_runs']),
        statistics=statistics, protocol_sha256=digest(root/'protocol.json'),
        summary_sha256=digest(root/'report_summary.json'),
        supplement_sha256={name: digest(stage/name) for name in
            ('CT_REPRODUCIBILITY.md', 'ROBUSTNESS_RESULTS.md', 'ROBUSTNESS_EVALUATION.md')},
        published_sha256={name:digest(report/name) for name in
            ('ct_experiments.tex','sframes_ct_section.tex','ct_reproducibility_link.tex',
             'trajectories_360.pdf','trajectories_360.png','trajectories_1080.pdf','trajectories_1080.png','sframes_ct_section.pdf')})
    # The exporter first verifies the exact Git base and stages an isolated copy,
    # so a newer user edit cannot be overwritten by a long-running experiment.
    write(root/'staged_report.json', audit)
    try:
        from .robustness_export import export
    except ImportError:
        from robustness_export import export
    export(root, staged=audit)
    return read(root/'report_update.json')
