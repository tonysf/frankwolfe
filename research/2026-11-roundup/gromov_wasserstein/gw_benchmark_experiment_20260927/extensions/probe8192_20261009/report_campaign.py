"""Audit all three q8192 runs and plot dense objective/direction diagnostics."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load('_probe_report_contract', ROOT/'campaign.py')
prior = load('_probe_report_previous', ROOT.parent/'batch_size_20261008/report_campaign.py')
prior._contract = contract
original_extract = prior.extract_result
require, read, sha256, finite, quantiles = prior.require, prior.read, prior.sha256, prior.finite, prior.quantiles
DIRECTION_FIELDS = ('raw_col_l1', 'raw_row_l1', 'penalty_value', 'raw_structure_l1',
    'row_centered_momentum_norm', 'row_centered_full_gradient_norm', 'row_centered_penalty_norm',
    'row_centered_momentum_error_norm', 'penalty_to_momentum_ratio',
    'row_centered_momentum_relative_error', 'lmo_agreement_with_without_penalty',
    'lmo_agreement_momentum_full_gradient_without_penalty', 'lmo_agreement_momentum_full_gradient_with_penalty',
    'selected_column_count_with_penalty', 'selected_column_count_without_penalty',
    'largest_source_mass_with_penalty', 'largest_source_mass_without_penalty',
    'actual_solver_seconds', 'actual_end_to_end_seconds', 'gamma', 'rho_effective', 'mu')


def extract_result(metadata, config, dataset, q, seed):
    result = original_extract(metadata, config, dataset, q, seed)
    trace = {(row['iteration'], row['kind']): row for row in metadata['trace']}
    for point in result['trajectory']:
        point['actual_end_to_end_seconds'] = trace[(point['iteration'], point['kind'])]['actual_end_to_end_seconds']
    result['direction_diagnostic_seconds'] = metadata['timing']['direction_diagnostic_seconds']
    result['direction_trajectory'] = []
    for row in metadata['direction_trace']:
        point = {key: row[key] for key in DIRECTION_FIELDS if key in row}
        point.update(iteration=row['iteration'], raw_col_l1=row['residuals']['col_l1'],
                     raw_row_l1=row['residuals']['row_l1'])
        for label in ('with_penalty', 'without_penalty'):
            for key in ('selected_column_count', 'largest_source_mass'):
                point[key+'_'+label] = row['lmo_'+label][key]
        result['direction_trajectory'].append(point)
    return result


prior.extract_result = extract_result
prior.POINT_FIELDS += ('actual_end_to_end_seconds',)
prior.RUN_FIELDS += ('direction_diagnostic_seconds',)


def baseline(config):
    if 'baseline' not in config:
        return dict(status='not_configured', runs=[], trajectory_summaries=[])
    path = Path(config['_config_path']).parent/config['baseline']['file']
    require(sha256(path) == config['baseline']['sha256'], 'Prior report hash mismatch.')
    report = read(path)
    require(report['status'] == 'completed' and report['task_counts']['completed'] == report['expected_runs'] == 18,
            'Prior experiment was not fully validated.')
    name = contract.DATASET
    old, new = report['instances'][name], config['instances'][name]
    for key in ('n', 'world_size', 'array_files_sha256', 'instance_manifest_sha256', 'prefactors', 'final_iterations'):
        require(old[key] == new[key], 'Prior experiment differs on '+key)
    require(report['development_seeds'] == config['development_seeds'] and report['initialization'] == 'product',
            'Prior seeds/initialization differ.')
    rows = [row for row in report['runs'] if row['dataset'] == name and row['batch'] == 2048]
    require(sorted(row['seed'] for row in rows) == config['development_seeds'], 'Prior q2048 population is incomplete.')
    for row in rows:
        require(row['status'] == 'completed' and row['completed_updates'] == new['final_iterations']
                and row['sketch_probes'] == 2048*new['final_iterations'], 'Invalid prior q2048 work budget.')
        require(contract.base._hash(row['metadata_sha256']), 'Prior run lacks metadata hash.')
        point = row['trajectory'][-1]
        require(point['iteration'] == new['final_iterations'] and point['precision'] == 'float64'
                and point['kind'] == 'last' and point['rounded_loss'] == row['last_rounded_loss'],
                'Prior endpoint is not independently validated last iterate.')
        require(row['last_rounded_both_l1'] <= config['final_marginal_l1_tolerance'], 'Prior endpoint infeasible.')
    summaries = [row for row in report['trajectory_summaries'] if row['dataset'] == name and row['batch'] == 2048]
    return dict(status='validated', report_file=str(path), report_sha256=sha256(path),
                historical_source_directory=report['source_directory'], historical_config_sha256=report['config_sha256'],
                runs=rows, trajectory_summaries=summaries,
                label='q = 2,048 (previous experiment; sparse checkpoints)',
                scope='Same frozen inputs, theoretical schedules, prefactors, seeds, initialization and horizon; earlier source and diagnostic grid. Its recorded solver times exclude diagnostics; checkpoint elapsed run time is unavailable.')


def audit(config_path, source, prepared, out):
    config = contract.load_config(config_path)
    report = prior.audit(config_path, source, prepared, out)
    valid = [row for row in report['runs'] if row['status'] == 'completed']
    complete = len(valid) == 3
    expected = [0, *config['instances'][contract.DATASET]['final_checkpoint_iterations'][:-1]]
    summaries = []
    for k in expected:
        points = [next(point for point in row['direction_trajectory'] if point['iteration'] == k) for row in valid]
        stats = {}
        for field in DIRECTION_FIELDS:
            values = [point[field] for point in points if point[field] is not None]
            stats[field] = dict(n=len(values), undefined_count=len(points)-len(values),
                statistics=quantiles(values) if complete and len(values) == 3 else None)
        summaries.append(dict(iteration=k, status='completed' if complete else 'incomplete', n=len(points), statistics=stats))
    report['direction_summaries'] = summaries
    report['baseline'] = baseline(config)
    report['scientific_scope'] = 'q8192 with unchanged synthetic FraMES prefactors and schedules; three development seeds. Added read-only diagnostics must pass exact trajectory-parity gates. This is a last-iterate diagnostic study, not a theorem-output claim.'
    report['direction_scope'] = 'Preupdate P_k and actual d_k after its gradient/momentum update, before the FW step. No d_T exists or is imputed. Momentum discrepancy relative to the full gradient combines sampling error and EMA lag; it is not a pure variance estimate. Norms and analytical full gradients use FP64.'
    report['timing_scope'] = 'Solver wall time includes optimizer work and excludes all loss/direction diagnostics, rounding, snapshots, logging and FP64 validation. Observed elapsed run time at each new checkpoint includes preparation and prior instrumentation since run() entry; it excludes interpreter/process-group startup and snapshot/FP64 validation occurring later. Old q2048 has no checkpoint elapsed run times. Time plots place each common-iteration median at median recorded solver time; no common-time interpolation.'
    return report


def figures(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    folder = Path(out)/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    files = []

    def save(fig, stem):
        for suffix in ('png', 'pdf'):
            path = folder/(stem+'.'+suffix)
            fig.savefig(path, dpi=180, facecolor='white')
            files.append(str(path))
        plt.close(fig)

    groups = [(report['trajectory_summaries'], '#8b359b', 'q = 8,192 (new dense diagnostics)')]
    if report['baseline']['status'] == 'validated':
        groups.insert(0, (report['baseline']['trajectory_summaries'], '#278447', report['baseline']['label']))
    for xfield, xlabel, stem in [('iteration', 'Completed updates', 'loss_iterations'),
                                ('actual_solver_seconds', 'Solver wall-clock time (seconds)', 'loss_solver_walltime')]:
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        fig.subplots_adjust(left=.11, right=.98, bottom=.24, top=.86)
        for rows, color, label in groups:
            rows = [row for row in rows if row['status'] == 'completed']
            x = [row['iteration'] if xfield == 'iteration' else row['statistics'][xfield]['median'] for row in rows]
            y = [row['statistics']['rounded_loss'] for row in rows]
            ax.plot(x, [p['median'] for p in y], marker='o', markersize=2.6, linewidth=1.3, color=color, label=label)
            ax.fill_between(x, [p['q25'] for p in y], [p['q75'] for p in y], color=color, alpha=.16)
        ax.set(title='FraMES: clean 512-point synthetic problem', xlabel=xlabel, ylabel='Feasible squared GW loss')
        ax.grid(alpha=.2)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.legend(fontsize=9)
        fig.text(.11, .12, 'Median and IQR across all three development seeds. Markers: recorded checkpoints; lines are visual guides.', fontsize=8)
        fig.text(.11, .075, 'Final loss: independent FP64 evaluation. Earlier loss: FP32. Solver times exclude instrumentation.', fontsize=8)
        if report['status'] != 'completed':
            fig.text(.11, .17, f'New experiment incomplete: {report["task_counts"]["completed"]}/3 validated seeds; new median omitted.', fontsize=9, color='#9b2424')
        save(fig, stem)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.subplots_adjust(left=.07, right=.985, bottom=.16, top=.88, hspace=.45, wspace=.3)
    fields = [('raw_col_l1', 'Raw column-marginal L1 error'),
              ('penalty_to_momentum_ratio', 'Row-centered penalty / momentum norm'),
              ('row_centered_momentum_relative_error', 'Momentum discrepancy / full-gradient norm'),
              ('lmo_agreement_with_without_penalty', 'LMO agreement with/without penalty'),
              ('largest_source_mass_with_penalty', 'Largest target-column share of LMO mass'),
              ('raw_structure_l1', 'Distance from independent raw coupling (L1)')]
    for ax, (field, label) in zip(axes.flat, fields):
        points = [row for row in report['direction_summaries'] if row['statistics'][field]['statistics'] is not None]
        x = [row['iteration'] for row in points]
        values = [row['statistics'][field]['statistics'] for row in points]
        ax.plot(x, [v['median'] for v in values], color='#8b359b', linewidth=1.1)
        ax.fill_between(x, [v['q25'] for v in values], [v['q75'] for v in values], color='#8b359b', alpha=.17)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Preupdate iteration k', fontsize=9)
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=8)
    fig.suptitle('q = 8,192: actual momentum and feasibility-penalty diagnostics', fontsize=13)
    fig.text(.07, .065, 'Median and IQR over three seeds; undefined ratios are omitted. Momentum discrepancy includes sampling error AND EMA lag.', fontsize=9)
    fig.text(.07, .025, 'Direction diagnostics observe d_k before its FW update; there is no terminal d_T. Analytical full gradients and norms use FP64.', fontsize=9)
    save(fig, 'direction_diagnostics')
    return dict(status='generated', files=files)


def write_report(report, out, no_plots=False):
    out = Path(out)
    prior.write_report(report, out, no_plots=True)
    report['figures'] = dict(status='disabled', files=[]) if no_plots else figures(report, out)
    (out/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    observations = [dict(seed=row['seed'], **point) for row in report['runs'] if row['status'] == 'completed'
                    for point in row['direction_trajectory']]
    prior.write_csv(out/'direction_observations.csv', observations, ['seed', 'iteration', *DIRECTION_FIELDS])
    rows = [dict(iteration=row['iteration'], status=row['status'], metric=field,
                 n=value['n'], undefined_count=value['undefined_count'],
                 **{key: (value['statistics'] or {}).get(key) for key in ('median', 'q25', 'q75')})
            for row in report['direction_summaries'] for field, value in row['statistics'].items()]
    prior.write_csv(out/'direction_summaries.csv', rows,
                    ['iteration', 'status', 'metric', 'n', 'undefined_count', 'median', 'q25', 'q75'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'source', 'prepared', 'out'):
        parser.add_argument('--'+flag, required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    report = audit(args.config, args.source, args.prepared, args.out)
    write_report(report, args.report_dir, args.no_plots)
    print(json.dumps(dict(status=report['status'], task_counts=report['task_counts'], report=str(args.report_dir/'report.json'))))
    return 0 if report['status'] == 'completed' else 2


if __name__ == '__main__':
    sys.exit(main())
