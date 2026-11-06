"""Audit the fixed-horizon FraMES probe-count experiment without selecting iterates.

Only receipts and metadata are read. Plots use observed checkpoints; time axes
place each common-iteration median at the median recorded elapsed time.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parent
_contract = _load('_batch_report_contract', ROOT/'campaign.py')
require, sha256 = _contract.require, _contract.sha256
STATES = ('completed', 'missing', 'failed', 'invalid', 'started_without_result')
COLORS = {128: '#1766ad', 1024: '#cf661b', 2048: '#278447'}
POINT_FIELDS = ('raw_loss', 'rounded_loss', 'loss_reduction_percent',
                'raw_col_l1', 'raw_col_l2', 'rounded_col_l1', 'rounded_both_l1',
                'repair_l1', 'actual_solver_seconds', 'repair_seconds_current',
                'time_to_this_feasible_output')
RUN_FIELDS = ('initial_rounded_loss', 'last_raw_loss', 'last_rounded_loss',
              'loss_reduction_percent', 'last_raw_col_l1', 'last_rounded_both_l1',
              'solver_seconds', 'last_feasible_output_seconds', 'total_elapsed_seconds',
              'completed_updates', 'stochastic_gradient_calls', 'sketch_probes',
              'updates_per_solver_second', 'probes_per_solver_second',
              'optimization_peak_gpu_allocated_bytes_per_rank_max',
              'optimization_peak_gpu_reserved_bytes_per_rank_max',
              'fp64_peak_gpu_allocated_bytes_per_rank_max')


def read(path):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=nonfinite)


def finite(value, name, minimum=None):
    value = _contract.finite(value, name)
    require(minimum is None or value >= minimum, 'Out-of-range ' + name)
    return value


def quantiles(values):
    """Same across-seed median and linear-interpolation IQR as the long-run report."""
    values = sorted(finite(value, 'summary value') for value in values)
    require(bool(values), 'Empty summary')
    def q(fraction):
        position = (len(values)-1)*fraction
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (position-lower)*(values[upper]-values[lower])
    return dict(n=len(values), median=statistics.median(values), q25=q(.25), q75=q(.75))


def checkpoints(config, dataset):
    instance = config['instances'][dataset]
    return sorted({0, instance['final_iterations'], *instance['final_checkpoint_iterations']})


def extract_result(metadata, config, dataset, q, seed):
    """Use exact prescribed iterates and independently evaluated FP64 endpoints."""
    instance = config['instances'][dataset]
    horizon = instance['final_iterations']
    actual = metadata['config']
    require(metadata.get('status') == 'completed', 'Runner did not complete')
    require(metadata.get('method_key') == 'smoothing-fw', 'Method mismatch')
    for key, expected in dict(method='smoothing-fw', iterations=horizon, batch=q,
                              initialization='product', seed=seed).items():
        require(actual.get(key) == expected, 'Runner config mismatch: ' + key)
    counts = metadata['counters']
    require(counts['completed_updates'] == counts['stochastic_gradient_calls'] == horizon
            and counts['sketch_probes'] == q*horizon, 'Update or probe budget mismatch')
    validation = metadata['final_float64_validation']
    require(validation.get('dtype') == 'float64' and validation.get('exact_objective') is True,
            'Missing independent final FP64 objective')
    plans = validation['plans']
    final_loss = finite(plans['last_rounded']['loss'], 'final rounded loss')
    final_raw_loss = finite(plans['last_raw']['loss'], 'final raw loss')
    solver_seconds = finite(metadata['timing']['solver_seconds'], 'solver seconds', 0)
    require(solver_seconds > 0, 'Solver time must be positive')
    precision = actual['dtype']
    require(precision in ('float32', 'float64'), 'Unknown diagnostic precision')
    points = {}
    previous_iteration, previous_elapsed = -1, -1.
    for row in metadata['trace']:
        kind, iteration = row['kind'], row['iteration']
        require(isinstance(iteration, int) and not isinstance(iteration, bool)
                and previous_iteration <= iteration <= horizon,
                'Nonmonotone or out-of-range trace iteration')
        previous_iteration = iteration
        if kind not in ('initial', 'current_diagnostic', 'last'):
            continue
        require(kind != 'initial' or iteration == 0, 'Initial diagnostic at nonzero iteration')
        require(kind != 'last' or iteration == horizon, 'Final diagnostic before horizon')
        elapsed = finite(row['actual_solver_seconds'], 'diagnostic solver seconds', 0)
        repair = finite(row['repair_seconds_current'], 'rounding seconds', 0)
        feasible = finite(row['time_to_this_feasible_output'], 'feasible output seconds', 0)
        require(previous_elapsed <= elapsed <= solver_seconds + 1e-8,
                'Nonmonotone or excessive diagnostic solver time')
        require(abs(feasible-elapsed-repair) <= 1e-8*max(1., feasible),
                'Inconsistent diagnostic feasible output time')
        previous_elapsed = elapsed
        expected_kind = 'initial' if iteration == 0 else 'last' if iteration == horizon else 'current_diagnostic'
        if kind != expected_kind:
            continue
        require(iteration not in points, 'Duplicate current iterate at iteration ' + str(iteration))
        raw = plans['last_raw']['marginal_residuals'] if kind == 'last' else row['raw_residuals']
        rounded = plans['last_rounded']['marginal_residuals'] if kind == 'last' else row['rounded_residuals']
        points[iteration] = dict(
            iteration=iteration, kind=kind, precision='float64' if kind == 'last' else precision,
            repair_precision=precision,
            raw_loss=final_raw_loss if kind == 'last' else finite(row['raw_loss'], 'raw loss'),
            rounded_loss=final_loss if kind == 'last' else finite(row['rounded_loss'], 'rounded loss'),
            raw_col_l1=finite(raw['col_l1'], 'raw column L1', 0),
            raw_col_l2=finite(raw['col_l2'], 'raw column L2', 0),
            rounded_col_l1=finite(rounded['col_l1'], 'rounded column L1', 0),
            rounded_both_l1=finite(rounded['both_l1'], 'rounded marginal L1', 0),
            repair_l1=finite(row['repair_l1'], 'rounding displacement L1', 0),
            actual_solver_seconds=elapsed, repair_seconds_current=repair,
            time_to_this_feasible_output=feasible)
    missing = sorted(set(checkpoints(config, dataset)) - set(points))
    require(not missing, 'Missing prescribed current-iterate checkpoints: ' + str(missing))
    require(set(points) == set(checkpoints(config, dataset)), 'Unexpected current-iterate checkpoint')
    tolerance = config['final_marginal_l1_tolerance']
    require(points[horizon]['rounded_both_l1'] <= tolerance, 'Final rounded plan is infeasible')
    require(points[0]['rounded_both_l1'] <= tolerance, 'Initial rounded plan is infeasible')
    initial = finite(points[0]['rounded_loss'], 'initial feasible loss', 0)
    require(initial > 0, 'Relative loss reduction needs positive initial loss')
    for point in points.values():
        point['loss_reduction_percent'] = 100*(initial-point['rounded_loss'])/initial
    memory = [dict(rank=rank['rank'], stages=rank['memory']) for rank in metadata['ranks']]

    def peak(stage, key):
        return max(finite(rank['stages'][stage][key], 'GPU peak memory', 0) for rank in memory)

    return dict(initial_rounded_loss=initial, initial_precision=precision,
                last_raw_loss=final_raw_loss, last_rounded_loss=final_loss,
                loss_reduction_percent=points[horizon]['loss_reduction_percent'],
                last_raw_col_l1=points[horizon]['raw_col_l1'],
                last_rounded_both_l1=points[horizon]['rounded_both_l1'],
                solver_seconds=solver_seconds,
                last_feasible_output_seconds=points[horizon]['time_to_this_feasible_output'],
                total_elapsed_seconds=finite(metadata['timing']['total_elapsed_seconds'], 'total elapsed seconds', 0),
                completed_updates=horizon, stochastic_gradient_calls=counts['stochastic_gradient_calls'],
                sketch_probes=counts['sketch_probes'], updates_per_solver_second=horizon/solver_seconds,
                probes_per_solver_second=counts['sketch_probes']/solver_seconds,
                optimization_peak_gpu_allocated_bytes_per_rank_max=peak('optimization_and_diagnostics', 'cuda_peak_allocated_bytes'),
                optimization_peak_gpu_reserved_bytes_per_rank_max=peak('optimization_and_diagnostics', 'cuda_peak_reserved_bytes'),
                fp64_peak_gpu_allocated_bytes_per_rank_max=peak('float64_validation', 'cuda_peak_allocated_bytes'),
                memory_by_rank=memory, timing=metadata['timing'],
                trajectory=[points[k] for k in sorted(points)])


def summarize(config, records):
    seeds = config['development_seeds']
    summaries, trajectories = [], []
    for dataset in config['instances']:
        for q in config['batches']:
            population = [row for row in records if (row['dataset'], row['batch']) == (dataset, q)]
            valid = [row for row in population if row['status'] == 'completed']
            complete = len(population) == len(seeds) and sorted(row['seed'] for row in valid) == sorted(seeds)
            identity = dict(dataset=dataset, batch=q, initialization='product',
                            expected_seeds=seeds, completed_seeds=sorted(row['seed'] for row in valid),
                            status='completed' if complete else 'incomplete', n=len(valid))
            summaries.append(dict(identity, statistics={key: quantiles([row[key] for row in valid])
                                                         for key in RUN_FIELDS} if complete else None))
            for iteration in checkpoints(config, dataset):
                observed = [next(point for point in row['trajectory'] if point['iteration'] == iteration)
                            for row in valid]
                trajectories.append(dict(identity, iteration=iteration,
                    precision='float64' if iteration == config['instances'][dataset]['final_iterations'] else config['runner']['dtype'],
                    statistics={key: quantiles([point[key] for point in observed])
                                for key in POINT_FIELDS} if complete else None))
    return summaries, trajectories


def audit(config_path, source, prepared, out):
    config_path, out = Path(config_path), Path(out)
    config = _contract.load_config(config_path)
    records = []
    for dataset in config['instances']:
        for q in config['batches']:
            for seed in config['development_seeds']:
                folder = _contract.run_directory(out, dataset, q, seed)
                metadata_path = folder/'output/metadata.json'
                row = dict(dataset=dataset, batch=q, seed=seed, initialization='product',
                           status='missing', metadata_file=str(metadata_path))
                try:
                    status_path = folder/'status.json'
                    status = read(status_path) if status_path.exists() else None
                    if status and status.get('status') == 'failed':
                        row.update(status='failed', reason=status.get('error', 'Controller rejected run'))
                    elif metadata_path.exists():
                        validated = _contract.validate_run(config, source, prepared, out, dataset, q, seed)
                        row.update(extract_result(validated['metadata'], config, dataset, q, seed),
                                   status='completed', metadata_sha256=sha256(metadata_path))
                    else:
                        failures = sorted((folder/'output').glob('failure*.json'))
                        if failures:
                            row.update(status='failed', failure_files=[str(path) for path in failures],
                                       reason='; '.join(str(read(path).get('error', 'Runner failure')) for path in failures))
                        elif status or (folder/'request.json').exists() or (folder/'output/progress.json').exists():
                            row.update(status='started_without_result')
                except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                    row.update(status='invalid', reason=str(exc))
                records.append(row)
    summaries, trajectories = summarize(config, records)
    counts = {state: sum(row['status'] == state for row in records) for state in STATES}
    return dict(schema_version=1, status='completed' if counts['completed'] == len(records) else 'incomplete',
                expected_runs=len(records), task_counts=counts, config_file=str(config_path),
                config_sha256=sha256(config_path), source_directory=str(source), prepared_directory=str(prepared),
                instances=config['instances'], batches=config['batches'], development_seeds=config['development_seeds'],
                method='FraMES (stochastic gradient momentum)', initialization='product',
                runs=records, final_summaries=summaries, trajectory_summaries=trajectories,
                objective='Unregularized squared GW loss of the current rounded feasible plan; no entropy term or factor of one half.',
                summary_policy='Every configured development seed is required for each median/IQR. Percent reductions are computed per seed relative to its initial feasible loss, then summarized.',
                trajectory_policy='Exact observed iterations only; no seed-trace interpolation, nearest-checkpoint substitution, extrapolation, or best-checkpoint selection.',
                precision_policy='Initial and intermediate losses use optimizer precision; final raw/rounded losses and residuals use independent FP64 validation. Repair L1 remains optimizer precision.',
                timing_scope='Common-iteration medians are plotted at median recorded solver time or solver plus current rounding time. These are not median losses at a common time or times to equal accuracy. No time interpolation. Exact diagnostics, other rounding calls, and final FP64 validation are excluded from these time axes; full timing is retained per run.',
                memory_scope='Measured peaks per GPU rank and stage; optimization includes diagnostics. Stage peaks are not added. FP64 validation occurs after optimizer state release.',
                scientific_scope='Development experiment varying only probe count within each dataset. All theoretical exponents, prefactors, initializations, and horizons remain fixed. This is not a held-out final comparison.')


def figures(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    folder = Path(out)/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for dataset, instance in report['instances'].items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True)
        fig.subplots_adjust(left=.075, right=.985, top=.78, bottom=.24, wspace=.15)
        fig.suptitle(f'FraMES probe-count experiment: {dataset} ({instance["n"]:,} points)', y=.98)
        fig.text(.5, .925, 'Product initialization · median and IQR across all three development seeds · lower loss is better', ha='center', fontsize=9)
        for q in report['batches']:
            points = [row for row in report['trajectory_summaries']
                      if row['dataset'] == dataset and row['batch'] == q and row['status'] == 'completed']
            stats = [row['statistics'] for row in points]
            losses = [row['rounded_loss'] for row in stats]
            for ax, xfield in zip(axes, ('iteration', 'actual_solver_seconds', 'time_to_this_feasible_output')):
                x = [row['iteration'] for row in points] if xfield == 'iteration' else [row[xfield]['median'] for row in stats]
                ax.errorbar(x, [row['median'] for row in losses],
                            yerr=[[row['median']-row['q25'] for row in losses],
                                  [row['q75']-row['median'] for row in losses]],
                            marker='o', markersize=3.8, linestyle='none', capsize=2,
                            color=COLORS.get(q), label=f'q = {q:,}')
        for ax, label in zip(axes, ('Completed updates', 'Recorded solver seconds', 'Solver + current rounding seconds')):
            ax.set_xlabel(label, fontsize=10)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.tick_params(labelsize=8)
        axes[0].set_ylabel('Feasible squared GW loss')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .895), ncol=3, frameon=False)
        groups = [row for row in report['final_summaries'] if row['dataset'] == dataset]
        incomplete = [f'q={row["batch"]}: {row["n"]}/3 valid seeds' for row in groups if row['status'] != 'completed']
        if incomplete:
            fig.text(.075, .185, 'Incomplete groups omitted: ' + '; '.join(incomplete), fontsize=9, color='#9b2424')
        fig.text(.075, .13, 'Markers are saved checkpoints; no interpolated losses. Time panels place each update-count median at its median recorded time.', fontsize=8)
        fig.text(.075, .085, 'Final losses: independent FP64 evaluation. Earlier losses: optimizer precision. Time axes exclude exact diagnostics and FP64 validation.', fontsize=8)
        for suffix in ('png', 'pdf'):
            path = folder/f'{dataset}_loss_iterations_time.{suffix}'
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
    identity = ['dataset', 'batch', 'initialization', 'seed']
    write_csv(out/'runs.csv', report['runs'], identity + ['status'] + list(RUN_FIELDS) + ['metadata_file', 'metadata_sha256', 'reason'])
    observations = [dict(**{key: row[key] for key in identity}, **point)
                    for row in report['runs'] if row['status'] == 'completed' for point in row['trajectory']]
    write_csv(out/'observations.csv', observations,
              identity + ['iteration', 'kind', 'precision', 'repair_precision'] + list(POINT_FIELDS))
    for filename, source in [('final_summaries.csv', report['final_summaries']),
                             ('trajectory_summaries.csv', report['trajectory_summaries'])]:
        rows = [dict(dataset=row['dataset'], batch=row['batch'], status=row['status'],
                     iteration=row.get('iteration'), precision=row.get('precision'), metric=key,
                     **dict(n=row['n'], **{k: v for k, v in stats.items() if k != 'n'}))
                for row in source for key, stats in (row['statistics'] or {'unavailable': dict(median=None, q25=None, q75=None)}).items()]
        write_csv(out/filename, rows, ['dataset', 'batch', 'status', 'n', 'iteration', 'precision', 'metric', 'median', 'q25', 'q75'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'source', 'prepared', 'out'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--report-dir', type=Path,
                        help='Separate report destination; defaults to the results root.')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    report = audit(args.config, args.source, args.prepared, args.out)
    destination = args.report_dir or args.out
    write_report(report, destination, args.no_plots)
    print(json.dumps(dict(status=report['status'], task_counts=report['task_counts'], report=str(destination/'report.json'))))
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    sys.exit(main())
