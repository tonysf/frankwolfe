"""Summarize recorded scaling outputs; never execute an optimizer or objective."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import statistics

# Set before the optional plotting imports in an allocated CPU report stage.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'

from campaign import HERE, POLICY, digest, frozen_selection, load_task, task_specs, write

NAMES = {
    'smoothing-fw': 'Smoothed row FW + momentum (sketch gradient)',
    'stochastic-kl-momentum': 'Transport KL-proximal gradient + momentum (sketch gradient)',
}
COLORS = {'smoothing-fw': '#0072B2', 'stochastic-kl-momentum': '#AA3377'}
WARNINGS = [
    'One BMMC dataset: optimizer seeds are repetitions, not independent biological datasets.',
    'The two cell subsets are nested and costs are separately normalized by size; raw loss levels cannot establish improvement across sizes.',
    'Each method and size is independently tuned on the prescribed limited grid with equal development budgets; this does not establish globally optimal hyperparameters.',
    'Equal solver-time budgets are not equal gradient-call or probe budgets. The work plots show actual observations without interpolation or a fixed-work claim.',
    'Checkpoint losses are full, unsampled float32 objective evaluations. Final endpoint losses are separate float64 evaluations of the saved rounded outputs.',
    'Solver plus current output repair time excludes preparation, prior diagnostics and serialization. End-to-end recorded process time includes these costs and is reported separately.',
    'Inner stopping failures remain visible even when the final rounded output passes validation.',
    'Missing and infrastructure-failed runs are excluded from numerical aggregates and never replaced by zero or treated as successful runs.',
]


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def stats(values, expected):
    values = sorted(v for v in (number(x) for x in values) if v is not None)
    def quantile(fraction):
        position = (len(values) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    return dict(count=len(values), expected_count=expected, coverage_complete=len(values) == expected,
                median=statistics.median(values) if values else None,
                q25=quantile(.25) if values else None, q75=quantile(.75) if values else None)


def source_binding(path):
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def read_trace(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        result = {}
        for key, value in row.items():
            if value == '':
                result[key] = None
            elif value in ('True', 'False'):
                result[key] = value == 'True'
            else:
                result[key] = number(value) if number(value) is not None else value
        parsed.append(result)
    return parsed


def endpoint_metrics(record):
    validation = record.get('final_float64_validation') or {}
    values = {'rounded_loss_float64': validation.get('rounded_loss'),
              'raw_loss_float64': validation.get('raw_loss'),
              'launcher_process_seconds': record.get('instrumented_process_seconds')}
    for section in ('timing', 'counters', 'memory'):
        for key, value in record.get(section, {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[section + '.' + key] = value
    for prefix in ('raw', 'rounded'):
        for key, value in validation.get(prefix, {}).items():
            values[prefix + '.' + key] = value
    usable = [row for row in record.get('trace', []) if row.get('kind') != 'missed'
              and number(row.get('rounded_loss')) is not None]
    final = max(usable, key=lambda x: number(x.get('iteration')) or 0) if usable else {}
    values['solver_plus_final_repair_seconds'] = final.get('time_to_this_feasible_output')
    values['final_repair_seconds'] = final.get('repair_seconds_current')
    return values


def expected_cells(config):
    return list(itertools.product(config['sizes'], config['initializations'], config['final_seeds']))


def collect(config, root):
    selection_path = root / 'selection.json'
    selection, selection_error = None, None
    selection_binding = source_binding(selection_path) if selection_path.is_file() else None
    try:
        selection = frozen_selection(config, root)
        if selection['config_sha256'] != digest(HERE / 'campaign.json'):
            raise ValueError('Frozen selection configuration does not match this campaign.')
        if selection['controller_sha256'] != digest(HERE / 'campaign.py'):
            raise ValueError('Frozen selection controller does not match this campaign.')
        for n in config['sizes']:
            for method in config['methods']:
                chosen = selection['selected'][str(n)][method]
                if chosen['parameters'] != config['candidates'][method][chosen['candidate_id']]:
                    raise ValueError('Frozen selected parameters do not match candidate grid.')
    except Exception as exc:
        selection_error = f'{type(exc).__name__}: {exc}'
    tasks, records, trace_issues = [], [], []
    for task, (n, start, seed) in enumerate(expected_cells(config)):
        folder = root / 'final' / ('task_' + str(task).zfill(3))
        receipt = dict(task=task, n=n, initialization=start, seed=seed,
                       expected_run_count=len(config['methods']), files=[])
        for name in ('manifest.json', 'controller.json', '_campaign/shard_00.json', 'validation.json'):
            path = folder / name
            if path.is_file():
                receipt['files'].append(source_binding(path))
        if not folder.exists():
            receipt.update(status='missing', error='No final task output directory.')
        else:
            try:
                if selection_error:
                    raise ValueError('Cannot validate final outputs: ' + selection_error)
                controller = json.loads((folder / 'controller.json').read_text())
                if controller['selection_sha256'] != selection_binding['sha256']:
                    raise ValueError('Task is not bound to the frozen selection.')
                if controller['controller_sha256'] != digest(HERE / 'campaign.py'):
                    raise ValueError('Task controller source binding mismatch.')
                task_records = load_task(config, root, 'final', task)
                prescribed = task_specs(config, 'final', task, selection)
                if len(task_records) != len(prescribed):
                    raise ValueError('Unexpected number of runs in final task.')
                for record, specification in zip(task_records, prescribed):
                    for key, value in specification.items():
                        if record.get(key) != value:
                            raise ValueError('Final manifest differs from selected specification: ' + key)
                    metadata_path = Path(record['metadata'])
                    metadata = json.loads(metadata_path.read_text())
                    record['source_commit'] = metadata.get('source_commit')
                    record['source_file_sha256'] = metadata.get('source_file_sha256')
                    record['legacy_runner_momentum_config'] = metadata.get('momentum_config')
                    record['parameter_selection_policy'] = POLICY
                    record['error'] = metadata.get('error')
                    record['task'] = task
                    record['source_bindings'] = [source_binding(metadata_path)]
                    for name in ('trace.csv', 'failures.json'):
                        path = metadata_path.parent / name
                        if path.is_file():
                            record['source_bindings'].append(source_binding(path))
                    record['trace'] = []
                    trace_path = metadata_path.parent / 'trace.csv'
                    try:
                        record['trace'] = read_trace(trace_path)
                    except Exception as exc:
                        trace_issues.append(dict(run_id=record['run_id'], error=f'{type(exc).__name__}: {exc}'))
                    failures_path = metadata_path.parent / 'failures.json'
                    record['failure_events'] = json.loads(failures_path.read_text()) if failures_path.is_file() else None
                    record['classification'] = 'valid' if record['eligible'] else 'scientific_failure'
                    record['endpoint_metrics'] = endpoint_metrics(record) if record['eligible'] else {}
                records.extend(task_records)
                receipt.update(status='completed', valid=sum(r['eligible'] for r in task_records),
                               scientific_failures=sum(not r['eligible'] for r in task_records))
            except Exception as exc:
                receipt.update(status='infrastructure_or_provenance_failure', error=f'{type(exc).__name__}: {exc}')
        tasks.append(receipt)
    return tasks, records, dict(binding=selection_binding, error=selection_error,
                               valid=selection_error is None, content=selection), trace_issues


def aggregate(config, records):
    groups, curves = [], []
    count = len(config['final_seeds'])
    required_metrics = ['rounded_loss_float64', 'raw_loss_float64', 'timing.solver_seconds',
        'solver_plus_final_repair_seconds', 'launcher_process_seconds',
        'counters.completed_updates', 'counters.sketch_probes', 'raw.both_l1', 'raw.col_l1',
        'raw.col_l2', 'counters.inner_stopping_failures', 'counters.inner_iterations_total',
        'memory.cuda_peak_allocated_bytes', 'memory.cuda_peak_reserved_bytes', 'memory.cpu_process_peak_rss_kib']
    trace_keys = ['rounded_loss', 'raw_loss', 'actual_solver_seconds', 'time_to_this_feasible_output',
                  'iteration', 'cumulative_sketch_probes', 'cumulative_oracle_evaluations',
                  'both_l1', 'col_l1', 'col_l2', 'cumulative_inner_stopping_failures']
    for n, start, method in itertools.product(config['sizes'], config['initializations'], config['methods']):
        subset = [r for r in records if (r['n'], r['initialization'], r['method']) == (n, start, method)]
        valid = [r for r in subset if r['eligible']]
        keys = sorted(set(required_metrics).union(*(r['endpoint_metrics'] for r in valid)))
        group = dict(n=n, initialization=start, method=method, method_name=NAMES[method],
            expected_count=count, valid_count=len(valid), scientific_failure_count=len(subset)-len(valid),
            unavailable_count=count-len(subset), coverage_complete=len(valid) == count,
            seeds=[r['seed'] for r in valid], metrics={k: stats([r['endpoint_metrics'].get(k) for r in valid], count) for k in keys})
        groups.append(group)
        points = []
        for kind, target in (('initial', None), ('checkpoint', 30.), ('checkpoint', 100.),
                             ('checkpoint', float(config['budgets']['final'])), ('final', None)):
            rows, present = [], []
            for record in valid:
                matches = [row for row in record['trace'] if row.get('kind') == kind
                    and row.get('target_solver_seconds') == target and number(row.get('rounded_loss')) is not None]
                if len(matches) == 1:
                    rows.append(matches[0]); present.append(record['seed'])
            metrics = {key: stats([row.get(key) for row in rows], count) for key in trace_keys}
            points.append(dict(kind=kind, target_solver_seconds=target,
                seeds=present, count=len(rows), expected_count=count, coverage_complete=len(rows) == count,
                metrics=metrics))
        curves.append(dict(n=n, initialization=start, method=method, method_name=NAMES[method], points=points))
    return groups, curves


def plot(config, curves, out):
    if not any(p['count'] for c in curves for p in c['points']):
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    files = []
    for start in config['initializations']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
        for column, n in enumerate(config['sizes']):
            selected = [c for c in curves if c['n'] == n and c['initialization'] == start]
            for row, xkey in enumerate(('time_to_this_feasible_output', 'cumulative_sketch_probes')):
                axis = axes[row, column]
                for curve in selected:
                    for point in curve['points']:
                        x, y = point['metrics'][xkey], point['metrics']['rounded_loss']
                        if x['median'] is None or y['median'] is None or x['median'] <= 0:
                            continue
                        color = COLORS[curve['method']]
                        axis.errorbar(x['median'], y['median'],
                            xerr=[[x['median']-x['q25']], [x['q75']-x['median']]],
                            yerr=[[y['median']-y['q25']], [y['q75']-y['median']]],
                            fmt='o', linestyle='none', color=color, capsize=2,
                            markerfacecolor=color if point['coverage_complete'] else 'none')
                axis.set_title(f'BMMC: {n:,} cells per view')
                axis.set_xlabel('Actual solver + current output repair time (s)' if row == 0 else 'Cumulative Rademacher probes')
                axis.set_ylabel('Full GW loss at rounded output (float32)')
                axis.set_xscale('log'); axis.grid(alpha=.2)
                if not any(p['count'] for c in selected for p in c['points']):
                    axis.text(.5, .5, 'No validated recorded checkpoints', transform=axis.transAxes, ha='center')
        handles = [Line2D([], [], marker='o', linestyle='none', color=COLORS[m], label=NAMES[m]) for m in config['methods']]
        fig.legend(handles=handles, loc='lower center', fontsize=9, ncol=1)
        fig.suptitle(f'Independently tuned stochastic methods: {start} start\nObserved checkpoints; medians and IQRs over up to five fresh optimizer seeds', fontsize=12)
        fig.text(.5, .12, 'No interpolated curves. Open markers: fewer than five seeds. Same dataset; nested subsets.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .16, 1, .92))
        for extension in ('png', 'pdf'):
            name = f'{start}_quality.{extension}'
            fig.savefig(out / name, dpi=160); files.append(name)
        plt.close(fig)
    return files


def fmt(metric):
    if metric['median'] is None:
        return '— (0/5)'
    return f"{metric['median']:.6g} [{metric['q25']:.6g}, {metric['q75']:.6g}] ({metric['count']}/{metric['expected_count']})"


def markdown(report):
    counts = report['counts']
    lines = ['# BMMC scaling campaign', '',
        f"Expected **40 final runs in 20 paired tasks**. Valid outputs: **{counts['valid']}**; scientific failures: **{counts['scientific_failure']}**; missing runs: **{counts['missing']}**; infrastructure/provenance failures: **{counts['infrastructure_or_provenance_failure']}**.", '',
        'Execution is complete.' if report['execution_complete'] else '**Execution is incomplete. Do not interpret partial coverage as a completed comparison.**', '',
        'Compare **Smoothed row FW + momentum (sketch gradient)** with **Transport KL-proximal gradient + momentum (sketch gradient)**. Each final pair shares its instance, starting plan, stochastic seed, q=128 estimator, momentum schedule and GPU allocation. Both receive a 300-second solver budget.', '',
        '## Parameter selection', '', POLICY, '',
        'The authoritative policy is this campaign configuration and frozen selection. Legacy `momentum_config.parameter_policy` text in the frozen runner describing inherited Stage C settings is superseded; it is retained in JSON for provenance.', '',
        'Selection rule: ' + report['configuration']['selection_rule'], '',
        'Selection provenance: ' + ('validated' if report['selection']['valid'] else 'unavailable or invalid: ' + report['selection']['error']), '',
        '| Cells/view | Method | Selected candidate | Parameters |', '|---:|---|---|---|']
    if report['selection']['valid']:
        for n, methods in report['selection']['content']['selected'].items():
            for method, choice in methods.items():
                lines.append(f"| {n} | {NAMES[method]} | {choice['candidate_id']} | {json.dumps(choice['parameters'], sort_keys=True)} |")
    lines += ['', '## Final saved outputs', '',
        'Entries are median [Q1, Q3] with observed/prescribed seed counts; IQR uses linear quantiles. Only validated final outputs contribute. The float64 losses evaluate the saved rounded couplings, without optimizing or re-rounding them.', '',
        '| Cells/view | Start | Method | Float64 rounded GW loss | Actual solver seconds | Solver + final repair seconds | Recorded process seconds |',
        '|---:|---|---|---:|---:|---:|---:|']
    for group in report['endpoints']:
        keys = ('rounded_loss_float64', 'timing.solver_seconds', 'solver_plus_final_repair_seconds', 'launcher_process_seconds')
        lines.append('| ' + ' | '.join([str(group['n']), group['initialization'], NAMES[group['method']]] + [fmt(group['metrics'][k]) for k in keys]) + ' |')
    lines += ['', '| Cells/view | Start | Method | Outer updates | Probes | Raw marginal L1 residual | Inner stopping failures | Peak allocated GPU bytes |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for group in report['endpoints']:
        keys = ('counters.completed_updates', 'counters.sketch_probes', 'raw.both_l1', 'counters.inner_stopping_failures', 'memory.cuda_peak_allocated_bytes')
        lines.append('| ' + ' | '.join([str(group['n']), group['initialization'], NAMES[group['method']]] + [fmt(group['metrics'][k]) for k in keys]) + ' |')
    lines += ['', 'Detailed timing components, CPU peak RSS, GPU reserved memory, raw row/column residuals, inner iteration counts, seed identities, environments and all recorded trace rows are retained in [report.json](report.json). Launcher process time includes interpreter startup and the final metadata write; the runner-only timing components have narrower scopes. Memory peaks include preparation and float64 validation.', '',
        '## Coverage and failures', '', '| Task | Size/start/seed | Status | Detail |', '|---:|---|---|---|']
    for task in report['tasks']:
        detail = task.get('error', f"{task.get('valid', 0)} valid; {task.get('scientific_failures', 0)} scientific failures")
        lines.append(f"| {task['task']} | {task['n']}/{task['initialization']}/{task['seed']} | {task['status']} | {detail.replace('|', '/')} |")
    failures = [r for r in report['runs'] if not r['eligible']]
    for record in failures:
        lines += ['', f"- Scientific failure `{record['run_id']}`: {record.get('error') or record['status']}; metadata and failure events retained in JSON."]
    if report['trace_issues']:
        lines += ['', f"Trace read issues: {len(report['trace_issues'])}; endpoints can remain valid while checkpoint coverage is incomplete. See JSON for each error."]
    lines += ['', '## Recorded checkpoints', '',
        'Markers are actual checkpoint observations aggregated by prescribed checkpoint, with medians and IQRs of the recorded coordinates. They are not interpolated to a common time or probe budget. Open markers indicate incomplete five-seed coverage. Traces from rejected final outputs remain in JSON but are excluded from these figures. Final float64 endpoint loss appears in the tables; checkpoint plots use the full float32 loss logged by the solver.', '']
    if report['figures']:
        lines += [f'![{name}]({name})' if name.endswith('.png') else f'- [{name}]({name})' for name in report['figures']]
    else:
        lines += ['No validated checkpoint observations are available; no numerical plot was fabricated.']
    lines += ['', '## Interpretation limits', ''] + ['- ' + warning for warning in WARNINGS]
    lines += ['', '## Provenance', '',
        'The JSON binds the campaign configuration, controller, report source, frozen selection, per-task receipts/manifests/shard records, metadata, traces and failure records by SHA-256. Recorded runner source hashes, instance/array hashes and initial-plan hashes are preserved. Numerical outputs and original records are not modified.', '',
        f"Runtime commit: `{report['configuration']['runtime_commit']}`. Selection SHA-256: `{(report['selection']['binding'] or {}).get('sha256', 'unavailable')}`.", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Absolute campaign output root; reports are written below reports/.')
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error('--out must be an absolute campaign output root')
    config = json.loads((HERE / 'campaign.json').read_text())
    if len(expected_cells(config)) != 20 or len(config['methods']) != 2 or len(config['final_seeds']) != 5:
        raise ValueError('Reporter expects the prescribed 20 tasks / 40 final runs / five seeds.')
    tasks, records, selection, trace_issues = collect(config, args.out)
    endpoints, curves = aggregate(config, records)
    counts = Counter(r['classification'] for r in records)
    for task in tasks:
        if task['status'] != 'completed':
            counts[task['status']] += task['expected_run_count']
    counts = {key: counts[key] for key in ('valid', 'scientific_failure', 'missing', 'infrastructure_or_provenance_failure')}
    if sum(counts.values()) != 40:
        raise ValueError('Final run accounting does not sum to 40.')
    out = args.out / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    figures = plot(config, curves, out)
    report = dict(schema_version=1, generated_utc=datetime.now(timezone.utc).isoformat(), output_root=str(args.out),
        job_id=os.environ.get('SLURM_JOB_ID'), expected_final_runs=40, expected_final_tasks=20,
        execution_complete=all(t['status'] == 'completed' for t in tasks), all_final_outputs_valid=counts['valid'] == 40,
        counts=counts, configuration=config, parameter_selection_policy=POLICY, selection=selection,
        source_bindings=[source_binding(HERE / name) for name in ('campaign.json', 'campaign.py', 'report.py')],
        tasks=tasks, runs=records, endpoints=endpoints, checkpoint_aggregates=curves,
        trace_issues=trace_issues, warnings=WARNINGS, figures=figures)
    write(out / 'report.json', report)
    (out / 'report.md').write_text(markdown(report))
    print(json.dumps(dict(report=str(out / 'report.md'), counts=counts,
                         execution_complete=report['execution_complete'], figures=len(figures))))


if __name__ == '__main__':
    main()
