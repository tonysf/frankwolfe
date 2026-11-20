"""Audit long stochastic GW runs and summarize exact observed checkpoints.

Only small JSON/source receipts are read. Missing or failed seeds remain visible;
three-seed summaries never substitute an earlier iterate or a different seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

METHODS = ('smoothing-fw', 'published-smd', 'published-msbpg')
LABELS = {
    'smoothing-fw': 'FraMES (stochastic gradient momentum)',
    'published-smd': 'Entropic SMD (fresh stochastic gradient, no momentum)',
    'published-msbpg': 'MSBPG (theorem parameter choices)',
}
COLORS = {'smoothing-fw': '#1766ad', 'published-smd': '#c65e12',
          'published-msbpg': '#278447'}
STATES = ('completed', 'missing', 'failed', 'invalid', 'started_without_result')

_spec = importlib.util.spec_from_file_location(
    '_long_horizon_report_contract', Path(__file__).with_name('campaign.py'))
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=nonfinite)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), 'Invalid finite number: ' + name)
    require(minimum is None or value >= minimum, 'Out-of-range ' + name)
    return float(value)


def quantiles(values):
    """Median and linearly interpolated quartiles across seeds."""
    values = sorted(finite(value, 'summary value') for value in values)
    require(bool(values), 'Empty summary')

    def q(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (position - lower) * (values[upper] - values[lower])

    return dict(n=len(values), median=statistics.median(values), q25=q(.25), q75=q(.75))


def checkpoint_iterations(config):
    return sorted({0, config['final_iterations'], *config['final_checkpoint_iterations']})


def extract_result(metadata, config, method, start, seed):
    """Extract current iterates only, after the controller's provenance audit.

    A current diagnostic and a final diagnostic may share the final iteration.
    Only the final diagnostic receives the independently validated FP64 loss.
    At every earlier iteration we use the current diagnostic at that exact index.
    """
    horizon = config['final_iterations']
    actual = metadata['config']
    require(metadata.get('status') == 'completed', 'Runner did not complete')
    require(metadata.get('method_key') == method and actual.get('method') == method,
            'Runner method mismatch')
    for key, expected in dict(iterations=horizon, initialization=start, seed=seed).items():
        require(actual.get(key) == expected, 'Runner config mismatch: ' + key)
    require(metadata['counters']['completed_updates'] == horizon,
            'Incomplete update count')
    validation = metadata['final_float64_validation']
    require(validation.get('dtype') == 'float64' and validation.get('exact_objective') is True,
            'Missing independent final FP64 objective')
    final_loss = finite(validation['plans']['last_rounded']['loss'], 'final rounded loss')
    solver_seconds = finite(metadata['timing']['solver_seconds'], 'solver seconds', 0)
    expected_precision = actual['dtype']
    require(expected_precision in ('float32', 'float64'), 'Unknown diagnostic precision')
    points = {}
    previous_iteration = -1
    for row in metadata['trace']:
        kind, iteration = row['kind'], row['iteration']
        require(isinstance(iteration, int) and not isinstance(iteration, bool)
                and previous_iteration <= iteration <= horizon,
                'Nonmonotone or out-of-range trace iteration')
        previous_iteration = iteration
        if kind not in ('initial', 'current_diagnostic', 'last'):
            continue
        require(kind != 'initial' or iteration == 0, 'Initial diagnostic at a nonzero iteration')
        require(kind != 'last' or iteration == horizon, 'Final diagnostic before final horizon')
        elapsed = finite(row['actual_solver_seconds'], 'diagnostic solver seconds', 0)
        repair = finite(row['repair_seconds_current'], 'diagnostic rounding seconds', 0)
        feasible = finite(row['time_to_this_feasible_output'], 'feasible output seconds', 0)
        require(elapsed <= solver_seconds + 1e-8 and
                math.isclose(feasible, elapsed + repair, rel_tol=1e-9, abs_tol=1e-8),
                'Inconsistent diagnostic timing')
        loss = finite(row['rounded_loss'], 'diagnostic rounded loss')
        expected_kind = 'initial' if iteration == 0 else 'last' if iteration == horizon else 'current_diagnostic'
        if kind != expected_kind:
            continue
        require(iteration not in points, 'Duplicate current iterate at iteration ' + str(iteration))
        points[iteration] = dict(iteration=iteration, kind=kind,
                                rounded_loss=final_loss if kind == 'last' else loss,
                                precision='float64' if kind == 'last' else expected_precision,
                                actual_solver_seconds=elapsed,
                                time_to_this_feasible_output=feasible)
    missing = sorted(set(checkpoint_iterations(config)) - set(points))
    require(not missing, 'Missing prescribed current-iterate checkpoints: ' + str(missing))
    counters = metadata['counters']
    result = dict(last_rounded_loss=final_loss,
                  solver_seconds=solver_seconds,
                  last_feasible_output_seconds=points[horizon]['time_to_this_feasible_output'],
                  completed_updates=horizon,
                  stochastic_gradient_calls=counters['stochastic_gradient_calls'],
                  sketch_probes=counters['sketch_probes'],
                  inner_iterations=counters['inner_iterations_total'],
                  certificate_checks=counters['certificate_checks'],
                  trajectory=[points[k] for k in sorted(points)])
    if method == 'published-msbpg':
        result['sampled_normalized_displacement_squared'] = finite(
            metadata['selected_update_statistic']['normalized_displacement_squared'],
            'MSBPG sampled normalized displacement', 0)
    return result


def summarize(config, records):
    """Only complete configured seed populations contribute a median or IQR."""
    seeds = config['final_seeds']
    summaries, trajectories, progress = [], [], []
    checkpoints = checkpoint_iterations(config)
    reference = config.get('progress_reference_iteration', 2000)
    horizon = config['final_iterations']
    fields = ('last_rounded_loss', 'solver_seconds', 'last_feasible_output_seconds',
              'stochastic_gradient_calls', 'sketch_probes', 'inner_iterations', 'certificate_checks')
    for dataset in config['instances']:
        for start in config['initializations']:
            for method in METHODS:
                population = [row for row in records if (row['dataset'], row['initialization'], row['method'])
                              == (dataset, start, method)]
                valid = [row for row in population if row['status'] == 'completed']
                complete = len(population) == len(seeds) and sorted(row['seed'] for row in valid) == sorted(seeds)
                identity = dict(dataset=dataset, initialization=start, method=method,
                                method_label=LABELS[method], expected_seeds=seeds,
                                completed_seeds=sorted(row['seed'] for row in valid),
                                status='completed' if complete else 'incomplete')
                stats = {field: quantiles([row[field] for row in valid]) for field in fields} if complete else None
                if complete and method == 'published-msbpg':
                    stats['sampled_normalized_displacement_squared'] = quantiles(
                        [row['sampled_normalized_displacement_squared'] for row in valid])
                summaries.append(dict(identity, statistics=stats))
                point_maps = {row['seed']: {point['iteration']: point for point in row['trajectory']} for row in valid}
                for iteration in checkpoints:
                    observed = [point_maps[seed][iteration] for seed in seeds if seed in point_maps and iteration in point_maps[seed]]
                    common = complete and len(observed) == len(seeds)
                    precision = sorted({point['precision'] for point in observed})
                    require(len(precision) <= 1, 'Mixed precision at a common checkpoint')
                    record = dict(identity, iteration=iteration,
                                  status='completed' if common else 'incomplete',
                                  precision=precision[0] if precision else None,
                                  n=len(observed), median=None, q25=None, q75=None)
                    if common:
                        record.update(quantiles([point['rounded_loss'] for point in observed]))
                    trajectories.append(record)
                before = next((row for row in trajectories if (row['dataset'], row['initialization'], row['method'], row['iteration'])
                               == (dataset, start, method, reference)), None)
                comparison = dict(identity, reference_iteration=reference, final_iteration=horizon,
                                  status='completed' if complete and before is not None else 'incomplete',
                                  reference_precision=before['precision'] if before else None,
                                  final_precision='float64', reference_loss=None, final_loss=None,
                                  paired_loss_drop=None, paired_relative_loss_drop_percent=None)
                if comparison['status'] == 'completed':
                    initial = [point_maps[seed][reference]['rounded_loss'] for seed in seeds]
                    final = [point_maps[seed][horizon]['rounded_loss'] for seed in seeds]
                    comparison.update(reference_loss=quantiles(initial), final_loss=quantiles(final),
                                      paired_loss_drop=quantiles([a-b for a,b in zip(initial, final)]))
                    if all(value > 0 for value in initial):
                        comparison['paired_relative_loss_drop_percent'] = quantiles(
                            [100*(a-b)/a for a,b in zip(initial, final)])
                progress.append(comparison)
    return summaries, trajectories, progress


def audit(config_path, source, prepared, out):
    config_path, out = Path(config_path), Path(out)
    config = _contract.load_config(config_path)
    records = []
    for dataset in config['instances']:
        for method in METHODS:
            candidate = config['selected_candidates'][dataset][method]
            for start in config['initializations']:
                for seed in config['final_seeds']:
                    folder = _contract.run_directory(out, dataset, method, start, seed)
                    metadata_path = folder/'output'/'metadata.json'
                    row = dict(dataset=dataset, method=method, method_label=LABELS[method],
                               candidate_id=candidate['id'], initialization=start, seed=seed,
                               status='missing', metadata_file=str(metadata_path))
                    try:
                        status_path = folder/'status.json'
                        status = read(status_path) if status_path.exists() else None
                        if status and status.get('status') == 'failed':
                            row.update(status='failed', reason=status.get('error', 'Controller rejected run'),
                                       failure_file=str(status_path))
                        elif metadata_path.exists():
                            validated = _contract.validate_run(config, source, prepared, out, dataset, method, start, seed)
                            row.update(extract_result(validated['metadata'], config, method, start, seed),
                                       status='completed', metadata_sha256=sha256(metadata_path))
                        else:
                            failures = sorted((folder/'output').glob('failure*.json'))
                            failures += sorted((folder/'output').glob('rank*/inner_failure.json'))
                            if failures:
                                row.update(status='failed', failure_files=[str(path) for path in failures],
                                           reason='; '.join(str(read(path).get('error', 'Runner failure')) for path in failures))
                            elif status or (folder/'request.json').exists() or (folder/'output'/'progress.json').exists():
                                row.update(status='started_without_result')
                    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                        row.update(status='invalid', reason=str(exc))
                    records.append(row)
    summaries, trajectories, progress = summarize(config, records)
    counts = {state: sum(row['status'] == state for row in records) for state in STATES}
    return dict(schema_version=1, status='completed' if counts['completed'] == len(records) else 'incomplete',
                config_file=str(config_path), config_sha256=sha256(config_path),
                source_directory=str(source), prepared_directory=str(prepared),
                methods=LABELS, instances=config['instances'], final_iterations=config['final_iterations'],
                checkpoints=checkpoint_iterations(config), final_seeds=config['final_seeds'],
                initializations=config['initializations'], task_counts=counts,
                runs=records, final_summaries=summaries, trajectory_summaries=trajectories,
                progress_from_reference=progress,
                objective='Unregularized squared GW loss of the current rounded feasible plan; no entropy term or factor of one half.',
                summary_policy='Median and linear-interpolation IQR across all configured final seeds. Missing, failed or invalid seeds prevent a complete median.',
                trajectory_policy='Exact observed iteration indices only. No nearest-checkpoint substitution, extrapolation, seed-trace interpolation, or best-so-far selection. The final endpoint uses the FP64 last-rounded validation.',
                precision_policy='Intermediate diagnostics use the optimizer dtype; final endpoint loss uses independent FP64 evaluation. Small differences across these precisions require caution.',
                timing_scope='Observed solver times at the same update count, not time to equal accuracy. Diagnostic evaluation and final FP64 validation are outside solver time.',
                theorem_scope='MSBPG uses the theorem parameter prescription with numerical proximal solves at the configured tolerance; its sampled displacement statistic is separate from feasible GW loss.')


def figures(report, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import ScalarFormatter
    except ImportError:
        return dict(status='unavailable', reason='matplotlib is not installed', files=[])
    datasets = list(report['instances'])
    starts = report['initializations']
    fig, axes = plt.subplots(len(datasets), len(starts), figsize=(12.8, 3.15*len(datasets)+1.9),
                             squeeze=False, sharex=True, sharey='row')
    fig.subplots_adjust(left=.095, right=.98, bottom=.17, top=.80, hspace=.4, wspace=.15)
    fig.suptitle('Median feasible GW loss over iterations', fontsize=17, y=.98)
    fig.text(.5, .94, 'All three final seeds required at each saved checkpoint · Lower loss is better',
             ha='center', fontsize=10)
    handles = [Line2D([], [], marker='o', linestyle='none', color=COLORS[method],
                      label=LABELS[method]) for method in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .915),
               ncol=1, fontsize=9, frameon=False)
    for i, dataset in enumerate(datasets):
        for j, start in enumerate(starts):
            ax = axes[i, j]
            incomplete = False
            for method in METHODS:
                population = [row for row in report['trajectory_summaries']
                              if (row['dataset'], row['initialization'], row['method']) == (dataset, start, method)]
                points = [row for row in population if row['status'] == 'completed']
                incomplete |= len(points) != len(population)
                ax.errorbar([row['iteration'] for row in points], [row['median'] for row in points],
                            yerr=[[row['median']-row['q25'] for row in points],
                                  [row['q75']-row['median'] for row in points]],
                            linestyle='none', marker='o', markersize=4.5,
                            capsize=2.5, color=COLORS[method])
            n = report['instances'][dataset]['n']
            ax.set_title(f'{dataset} ({n:,} cells) | {start} start', fontsize=10)
            if incomplete:
                ax.text(.98, .98, 'Incomplete seed population', transform=ax.transAxes,
                        ha='right', va='top', fontsize=8, color='#9b2424')
            ax.grid(alpha=.2)
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(labelsize=8)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0:
                ax.set_ylabel('Feasible GW loss', fontsize=10)
            if i == len(datasets)-1:
                ax.set_xlabel('Completed outer updates', fontsize=10)
    fig.text(.095, .09, 'Markers: observed medians. Bars: interquartile range across three seeds. No unsaved losses are estimated.', fontsize=9)
    fig.text(.095, .06, 'Intermediate diagnostics use optimizer precision; final losses use FP64. Each dataset has its own vertical scale.', fontsize=9)
    folder = Path(out)/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ('png', 'pdf'):
        path = folder/f'median_loss_iterations.{suffix}'
        fig.savefig(path, dpi=170, facecolor='white')
        files.append(str(path))
    plt.close(fig)
    return dict(status='generated', files=files)


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_report(report, out, no_plots=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report['figures'] = dict(status='disabled', files=[]) if no_plots else figures(report, out)
    (out/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    write_csv(out/'report.csv', report['runs'],
              ['dataset', 'method', 'candidate_id', 'initialization', 'seed', 'status',
               'last_rounded_loss', 'completed_updates', 'stochastic_gradient_calls',
               'sketch_probes', 'inner_iterations', 'certificate_checks', 'solver_seconds',
               'last_feasible_output_seconds', 'sampled_normalized_displacement_squared',
               'metadata_file', 'metadata_sha256', 'reason'])
    write_csv(out/'median_loss_iterations.csv', report['trajectory_summaries'],
              ['dataset', 'method', 'initialization', 'iteration', 'status', 'precision',
               'n', 'median', 'q25', 'q75'])
    observations = [dict(dataset=row['dataset'], method=row['method'], initialization=row['initialization'],
                         seed=row['seed'], **point)
                    for row in report['runs'] if row['status'] == 'completed' for point in row['trajectory']]
    write_csv(out/'observations.csv', observations,
              ['dataset', 'method', 'initialization', 'seed', 'iteration', 'kind',
               'precision', 'rounded_loss', 'actual_solver_seconds', 'time_to_this_feasible_output'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'source', 'prepared', 'out'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    report = audit(args.config, args.source, args.prepared, args.out)
    write_report(report, args.out, args.no_plots)
    print(json.dumps(dict(status=report['status'], task_counts=report['task_counts'], report=str(args.out/'report.json'))))


if __name__ == '__main__':
    main()
