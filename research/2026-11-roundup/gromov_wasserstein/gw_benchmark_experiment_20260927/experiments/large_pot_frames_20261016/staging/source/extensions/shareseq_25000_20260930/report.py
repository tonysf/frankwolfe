"""Audit and report all six fixed-configuration SHARE-seq final runs.

Sparse checkpoints are current outputs. Nothing is interpolated, selected by
loss, or dropped from the required final population. Run inside the scheduled
campaign after completion; only small JSONs and figures are processed here.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
from pathlib import Path
import textwrap

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ('smoothing-fw', 'stochastic-kl-momentum')
VALID = ('completed', 'completed_with_inner_failures')
METRICS = ('correspondence_top1_accuracy', 'correspondence_rank_fraction',
           'correspondence_true_pair_mass')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open() as handle:
        return json.load(handle, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError('Nonfinite JSON token: ' + token)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, minimum=None):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value), 'Invalid/nonfinite ' + label)
    if minimum is not None:
        require(value >= minimum, 'Below permitted minimum: ' + label)
    return float(value)


def integer(value, label, minimum=0):
    require(not isinstance(value, bool) and isinstance(value, int) and value >= minimum,
            'Invalid integer: ' + label)
    return value


def frozen_dimension(config, instance):
    """Bind every dimension-dependent label and check to the frozen instance."""
    n = integer(instance.get('n'), 'frozen instance n', 2)
    m = integer(instance.get('m'), 'frozen instance m', 2)
    expected = integer(config.get('expected_n'), 'configured expected_n', 2)
    require(n == m == expected, 'Frozen instance dimensions disagree with configured expected_n.')
    identifier = config.get('instance_id')
    require(isinstance(identifier, str) and identifier and instance.get('id') == identifier,
            'Frozen instance identifier disagrees with configured instance_id.')
    return n


def method_names(source=None):
    # Read the runner's exact presentation names without importing torch or
    # creating a distributed process group in this lightweight report process.
    tree = ast.parse((Path(source) if source is not None else ROOT).joinpath('run_sharded.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == 'METHOD_NAMES' for target in node.targets):
            result = ast.literal_eval(node.value)
            require(set(result) == set(METHODS), 'Unexpected runner method names.')
            return result
    raise ValueError('Runner METHOD_NAMES assignment is missing.')


def summary(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
            'Summary requires a nonempty finite sample.')
    q25, median, q75 = np.quantile(values, [.25, .5, .75], method='linear')
    return dict(n=len(values), median=float(median), q25=float(q25), q75=float(q75),
                minimum=float(values.min()), maximum=float(values.max()))


def check_trace(trace, expected_targets, final_updates, final_solver):
    require(isinstance(trace, list) and trace, 'Trace is missing.')
    observed, missed = {}, {}
    previous_iteration, previous_time = -1, -1.
    for row in trace:
        kind = row.get('kind')
        require(kind in ('initial', 'checkpoint', 'final', 'missed'), 'Unknown trace kind.')
        iteration = integer(row.get('iteration'), 'trace iteration')
        elapsed = finite(row.get('actual_solver_seconds'), 'trace solver seconds', 0)
        require(previous_iteration <= iteration <= final_updates and previous_time <= elapsed
                <= final_solver + 1e-8, 'Trace iterations/time are inconsistent.')
        previous_iteration, previous_time = iteration, elapsed
        target = row.get('target_solver_seconds')
        if kind == 'initial':
            require(iteration == 0 and target is None and 'initial' not in observed,
                    'Initial trace point must occur once at iteration zero.')
            key = 'initial'
        elif kind in ('checkpoint', 'missed'):
            require(target in expected_targets and elapsed >= target,
                    'Unexpected or prematurely recorded checkpoint.')
            key = str(float(target))
            require(key not in observed and key not in missed, 'Duplicate checkpoint target.')
        else:
            require(target is None and iteration == final_updates and 'final' not in observed,
                    'Invalid additional final trace point.')
            key = 'final'
        if kind == 'missed':
            require(isinstance(row.get('reason'), str) and row['reason'],
                    'Missing checkpoint reason absent.')
            missed[key] = row
            continue
        for field in ('raw_loss', 'rounded_loss', 'repair_seconds_current',
                      'time_to_this_feasible_output', 'rounded_both_l1'):
            finite(row.get(field), 'trace ' + field,
                   None if field in ('raw_loss', 'rounded_loss') else 0)
        require(math.isclose(row['time_to_this_feasible_output'],
                             elapsed + row['repair_seconds_current'], rel_tol=1e-10, abs_tol=1e-8),
                'Feasible-output clock disagrees with solver plus current repair.')
        require(row['rounded_both_l1'] <= 100 * np.finfo(np.float32).eps,
                'Checkpoint rounded plan is infeasible.')
        for metric in METRICS:
            value = finite(row.get(metric), metric, 0)
            require(value <= 1 + 1e-8, 'Correspondence metric is out of range.')
        observed[key] = row
    require('initial' in observed, 'Initial observation is missing.')
    require(set(observed).union(missed).issuperset(str(float(x)) for x in expected_targets),
            'A prescribed checkpoint is neither observed nor marked missed.')
    finals = [row for row in observed.values() if row['iteration'] == final_updates]
    require(finals, 'Final completed iterate has no exact checkpoint evaluation.')
    return observed, missed, finals[-1]


def audit(campaign):
    campaign = Path(campaign)
    evidence = {key: read(campaign / (key + '.json')) for key in
                ('frozen_campaign', 'correctness', 'pilot_gate', 'completion')}
    frozen, complete = evidence['frozen_campaign'], evidence['completion']
    config, instance = frozen['config'], frozen['instance_manifest']
    require(complete.get('status') == 'completed', 'Campaign is incomplete.')
    require(evidence['correctness'].get('status') == 'passed', 'Correctness gate did not pass.')
    require(evidence['pilot_gate'].get('status') == 'passed', 'Memory pilot gate did not pass.')
    require(config['methods'] == list(METHODS), 'Unexpected comparison methods.')
    seeds = config['final']['seeds']
    require(seeds == [500, 501, 502], 'Expected the three frozen final seeds 500–502.')
    n = frozen_dimension(config, instance)
    require(config['hardware']['gpus'] == 4, 'Expected four GPUs.')
    require(config['final']['budget_seconds'] == 300
            and config['final']['checkpoints'] == [10, 30, 100, 300], 'Final budget/grid changed.')
    require(config['batch'] == 128 and config['rho0'] == .3 and config['momentum_exponent'] == .5
            and config['fw'] == dict(gamma0=.1, mu_scale=.1, step_exponent=.75, smoothing_exponent=.25)
            and config['kl'] == dict(eta=.1, inner_tol=.0001, inner_max=1000, check_every=10)
            and config['initialization'] == 'product a*b^T' and config['initialization_seed'] == 100,
            'Frozen transferred configurations changed.')
    expected = {(method, seed) for method in METHODS for seed in seeds}
    finals = [item for item in complete.get('results', []) if item.get('phase') == 'final']
    keys = [(item.get('method'), item.get('seed')) for item in finals]
    require(len(finals) == 6 and set(keys) == expected and len(set(keys)) == 6,
            'Completion must contain exactly all six expected final results.')
    names, runs = method_names(frozen.get('source')), []
    source_reference = None
    for item in sorted(finals, key=lambda item: (item['method'], item['seed'])):
        method, seed = item['method'], item['seed']
        basename = f'final_{seed}_{method}'
        require(Path(item['path']).name == basename, 'Completion path does not identify expected run.')
        path = campaign / basename / 'metadata.json'
        run = read(path)
        require(run.get('status') in VALID and item.get('status') == run['status'],
                'Invalid final run status: ' + basename)
        rcfg, env = run['config'], run['environment']
        require(run['method'] == names[method] and rcfg['method'] == method and rcfg['seed'] == seed,
                'Run identity mismatch: ' + basename)
        expected_config = dict(initialization='product', init_seed=100, batch=128, rho0=.3,
                               gamma0=.1, mu_scale=.1, eta=.1, inner_tol=.0001, inner_max=1000,
                               budget=300, max_updates=0, device='cuda', save_plans=True)
        require(all(rcfg.get(key) == value for key, value in expected_config.items()),
                'Run configuration differs from frozen comparison: ' + basename)
        require(sorted(float(x) for x in rcfg['checkpoints'].split(',')) == [10, 30, 100, 300],
                'Run checkpoint grid changed.')
        require(env.get('world_size') == 4 and env.get('backend') == 'nccl'
                and env.get('slurm_job_id') == frozen['slurm_job_id'], 'Run allocation mismatch.')
        require(env.get('deterministic_algorithms') is True and env.get('tf32') is False,
                'Deterministic/precision policy changed.')
        require(run.get('array_files_sha256') == instance['array_files_sha256']
                and run.get('instance_manifest') == instance, 'Run data provenance changed.')
        require(rcfg.get('correspondence_sha256') == frozen['correspondence_sha256'],
                'Correspondence binding changed.')
        require(run.get('source_file_sha256') and all(len(value) == 64 for value in run['source_file_sha256'].values()),
                'Missing runtime source hashes.')
        if source_reference is None:
            source_reference = run['source_file_sha256']
        require(run['source_file_sha256'] == source_reference, 'Source hashes differ between final runs.')
        counters, timing, validation = run['counters'], run['timing'], run['final_float64_validation']
        updates = integer(counters.get('completed_updates'), 'completed updates', 1)
        solver = finite(timing.get('solver_seconds'), 'solver wall time', 300)
        require(math.isclose(finite(timing.get('solver_gpu_seconds'), 'solver GPU seconds', 0),
                             4 * solver, rel_tol=1e-10), 'GPU-seconds accounting mismatch.')
        for key, value in timing.items():
            finite(value, 'timing/' + key, 0)
        failures = integer(counters.get('inner_stopping_failures'), 'inner stopping failures')
        integer(counters.get('inner_iterations_total'), 'inner iterations')
        require((run['status'] == 'completed_with_inner_failures') == (failures > 0),
                'Inner failure status/count mismatch.')
        require(validation.get('dtype') == 'float64' and validation.get('exact_objective') is True,
                'Exact float64 validation is missing.')
        loss = finite(validation.get('rounded_loss'), 'float64 rounded loss')
        finite(validation.get('raw_loss'), 'float64 raw loss')
        residual = finite(validation.get('rounded', {}).get('both_l1'), 'rounded marginal residual', 0)
        require(residual <= 100 * np.finfo(np.float32).eps, 'Final rounded coupling is infeasible.')
        require(item.get('updates') == updates and item.get('loss') == loss
                and item.get('timing') == timing, 'Completion and run metadata disagree.')
        observations, missed, last = check_trace(run['trace'], config['final']['checkpoints'], updates, solver)
        ranks = run['ranks']
        require(len(ranks) == 4 and sorted(rank['rank'] for rank in ranks) == list(range(4)),
                'Exactly four unique rank records are required.')
        memory = []
        for rank in sorted(ranks, key=lambda rank: rank['rank']):
            require('A100' in rank.get('gpu', ''), 'Unexpected GPU model.')
            phases = rank.get('memory', {})
            require(set(phases) == {'preparation', 'solver_and_diagnostics', 'float64_validation'},
                    'Incomplete phase memory measurements.')
            reserved = []
            for phase, measure in phases.items():
                reserve = finite(measure.get('cuda_peak_reserved_bytes'), 'reserved bytes', 0)
                allocated = finite(measure.get('cuda_peak_allocated_bytes'), 'allocated bytes', 0)
                capacity = finite(measure.get('cuda_total_memory_bytes'), 'device memory bytes', 1)
                require(allocated <= reserve <= capacity, 'Invalid GPU memory measurements.')
                finite(measure.get('cpu_process_peak_rss_kib'), 'host RSS', 0)
                reserved.append(reserve)
            require(set(rank.get('saved_plans', {})) == {'P_raw', 'P_rounded'},
                    'Both raw and rounded final shards must be saved.')
            for record in rank['saved_plans'].values():
                require(record.get('shape') == [n//4 + int(rank['rank'] < n%4), n] and record.get('file')
                        and len(record.get('file_sha256', '')) == 64, 'Invalid saved shard binding.')
            memory.append(dict(rank=rank['rank'], gpu=rank['gpu'], phases=phases,
                               peak_reserved_bytes=max(reserved), peak_reserved_gib=max(reserved) / 1024**3))
        runs.append(dict(method_id=method, method=names[method], seed=seed, status=run['status'],
            metadata_file=str(path.relative_to(campaign)), metadata_sha256=sha256(path),
            source_revision=run.get('source_revision'), source_file_sha256=run['source_file_sha256'],
            environment=env, counters=counters, timing=timing, solver_gpu_hours=4 * solver / 3600,
            final_float64_validation=validation, final_correspondence={key:last[key] for key in METRICS},
            final_observed_smoothed_gap=last.get('true_smoothed_gap'), rank_memory=memory,
            observed_checkpoints=observations, missed_checkpoints=missed, trace=run['trace']))
    return evidence, runs, names


def aggregate(evidence, runs, names):
    config = evidence['frozen_campaign']['config']
    by_method = {}
    for method in METHODS:
        selected = [run for run in runs if run['method_id'] == method]
        values = dict(
            final_float64_rounded_loss=[run['final_float64_validation']['rounded_loss'] for run in selected],
            completed_outer_updates=[run['counters']['completed_updates'] for run in selected],
            solver_wall_seconds=[run['timing']['solver_seconds'] for run in selected],
            solver_gpu_hours=[run['solver_gpu_hours'] for run in selected],
            total_run_seconds=[run['timing']['total_elapsed_seconds'] for run in selected],
            final_current_repair_seconds=[next(reversed(run['observed_checkpoints'].values()))['repair_seconds_current']
                                         for run in selected],
            inner_stopping_failures=[run['counters']['inner_stopping_failures'] for run in selected],
            inner_iterations_total=[run['counters']['inner_iterations_total'] for run in selected])
        values.update({key:[run['final_correspondence'][key] for run in selected] for key in METRICS})
        checkpoints = {}
        for key in ['initial'] + [str(float(x)) for x in config['final']['checkpoints']]:
            available = [run['observed_checkpoints'][key] for run in selected if key in run['observed_checkpoints']]
            # Keep missingness explicit. No partial-seed median masquerades as
            # the prescribed three-seed comparison.
            point = dict(observed_seeds=[run['seed'] for run in selected if key in run['observed_checkpoints']],
                         missing_seeds=[run['seed'] for run in selected if key not in run['observed_checkpoints']],
                         n=len(available), target_solver_seconds=None if key == 'initial' else float(key),
                         summary=None)
            if len(available) == 3:
                point['summary'] = {field:summary([row[field] for row in available]) for field in
                    ('rounded_loss', 'actual_solver_seconds', 'time_to_this_feasible_output', 'iteration')}
            checkpoints[key] = point
        by_method[method] = dict(method=names[method], n=3,
            summaries={key:summary(value) for key, value in values.items()},
            inner_stopping_failures_total=sum(values['inner_stopping_failures']),
            runs_with_inner_stopping_failures=sum(value > 0 for value in values['inner_stopping_failures']),
            peak_reserved_gib_by_rank={str(rank):summary([run['rank_memory'][rank]['peak_reserved_gib']
                                               for run in selected]) for rank in range(4)},
            maximum_reserved_gib_any_rank=max(rank['peak_reserved_gib'] for run in selected for rank in run['rank_memory']),
            checkpoints=checkpoints)
    pairs = []
    for seed in config['final']['seeds']:
        pair = {run['method_id']:run for run in runs if run['seed'] == seed}
        fw = pair['smoothing-fw']['final_float64_validation']['rounded_loss']
        kl = pair['stochastic-kl-momentum']['final_float64_validation']['rounded_loss']
        pairs.append(dict(seed=seed, fw_loss=fw, kl_loss=kl, kl_minus_fw=kl-fw,
                          lower_loss_method=METHODS[0] if fw < kl else METHODS[1] if kl < fw else 'tie'))
    workload = finite(evidence['completion'].get('allocated_workload_seconds'), 'campaign workload seconds', 0)
    return dict(methods=by_method, paired_final_losses=pairs,
                paired_kl_minus_fw=summary([pair['kl_minus_fw'] for pair in pairs]),
                final_solver_gpu_hours_total=sum(run['solver_gpu_hours'] for run in runs),
                campaign_workload_seconds=workload, campaign_workload_gpu_hours=4*workload/3600,
                full_slurm_allocation_gpu_hours=None,
                full_slurm_allocation_accounting='Not measured here; obtain full allocation elapsed time from sacct.')


def figures(campaign, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette = dict(zip(METHODS, ('#2166ac', '#b2182b')))
    n = frozen_dimension(report['config'], report['provenance']['instance_manifest'])
    output = []
    for filename, field, xlabel in (
        ('current_rounded_loss_vs_feasible_time', 'time_to_this_feasible_output',
         'Solver wall time + repair of this output (seconds)'),
        ('current_rounded_loss_vs_outer_updates', 'iteration', 'Completed outer updates')):
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in METHODS:
            selected = [run for run in report['runs'] if run['method_id'] == method]
            for run in selected:
                observations = [row for row in run['observed_checkpoints'].values() if row['kind'] != 'final']
                ax.scatter([row[field] for row in observations], [row['rounded_loss'] for row in observations],
                           s=12, alpha=.22, color=palette[method], zorder=2)
            stats = [point['summary'] for point in report['aggregate']['methods'][method]['checkpoints'].values()
                     if point['summary'] is not None]
            if stats:
                x = np.asarray([point[field]['median'] for point in stats])
                y = np.asarray([point['rounded_loss']['median'] for point in stats])
                xerr = np.asarray([[point[field]['median']-point[field]['q25'] for point in stats],
                                   [point[field]['q75']-point[field]['median'] for point in stats]])
                yerr = np.asarray([[point['rounded_loss']['median']-point['rounded_loss']['q25'] for point in stats],
                                   [point['rounded_loss']['q75']-point['rounded_loss']['median'] for point in stats]])
                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o-', markersize=4, linewidth=1.4,
                            capsize=3, color=palette[method],
                            label=textwrap.fill(report['method_names'][method], 65), zorder=3)
        ax.set(xlabel=xlabel, ylabel='Current rounded GW loss (full objective)',
               title=f'SHARE-seq: {n:,} paired cells, four A100 GPUs\nFixed transferred constants; three paired stochastic seeds')
        ax.grid(alpha=.2)
        ax.legend(loc='best', fontsize=8)
        fig.text(.5, .015, 'Points: recorded checkpoints. Bars: median and IQR across all three seeds.\n'
                 'Lines only connect observed summaries; faint dots show individual seeds.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, 1))
        for suffix in ('png', 'pdf'):
            path = campaign / (filename + '.' + suffix)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            output.append(dict(file=path.name, sha256=sha256(path)))
        plt.close(fig)
    return output


def format_stat(value, digits=6):
    return f"{value['median']:.{digits}g} [{value['q25']:.{digits}g}, {value['q75']:.{digits}g}]"


def markdown(report):
    aggregate, names = report['aggregate'], report['method_names']
    n = frozen_dimension(report['config'], report['provenance']['instance_manifest'])
    uniform = 1/n
    plan_gb = 6*2*n*n*np.dtype(np.float32).itemsize/1e9
    wins = {method:sum(pair['lower_loss_method'] == method for pair in aggregate['paired_final_losses'])
            for method in METHODS}
    lines = [f'# SHARE-seq {n:,}-cell exploratory comparison', '',
        'All six prescribed final runs passed the report audit: three paired stochastic seeds, '
        'product initialization, 300 solver seconds per method, and the same four A100 GPUs. '
        'Hyperparameters were transferred unchanged from the selected SNARE-10,000 configurations; '
        'there was no SHARE-seq tuning or selection.', '',
        f"- **{names[METHODS[0]]}**: gamma0 = 0.1, mu_scale = 0.1; fixed step exponent 3/4 and smoothing exponent 1/4.",
        f"- **{names[METHODS[1]]}**: eta = 0.1, Sinkhorn tolerance = 1e-4, maximum 1,000 inner iterations, checked every 10.",
        '- Both methods use 128 Rademacher probes and momentum rho(k) = 0.3/(k+1)^(1/2); '
        'the first sketch gradient initializes momentum and is reused at update zero.', '',
        'The table entries are median [25th percentile, 75th percentile], using linear quantiles over three seeds. '
        'Final losses are the full unregularized GW objective evaluated in float64 on the saved rounded output values, '
        'without re-rounding or choosing a best iterate.', '',
        '| Method | Final rounded loss, float64 | Outer updates | Solver wall seconds | Solver GPU-hours (4× wall time) |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        m = aggregate['methods'][method]['summaries']
        lines.append(f"| {names[method]} | {format_stat(m['final_float64_rounded_loss'],9)} | "
                     f"{format_stat(m['completed_outer_updates'])} | {format_stat(m['solver_wall_seconds'])} | "
                     f"{format_stat(m['solver_gpu_hours'])} |")
    lines += ['', f"FW has lower final loss in {wins[METHODS[0]]}/3 paired seeds; KL has lower final loss in "
              f"{wins[METHODS[1]]}/3. Paired KL-minus-FW loss: {format_stat(aggregate['paired_kl_minus_fw'],9)}. "
              'Negative differences favor KL. Three seeds on one fixed dataset do not establish a general or asymptotic ranking.', '',
        '| Method | Largest reserved memory on any GPU (GiB) | Inner stopping misses (total) | Runs with misses |',
        '|---|---:|---:|---:|']
    for method in METHODS:
        m = aggregate['methods'][method]
        lines.append(f"| {names[method]} | {m['maximum_reserved_gib_any_rank']:.3f} | "
                     f"{m['inner_stopping_failures_total']} | {m['runs_with_inner_stopping_failures']}/3 |")
    lines += ['', 'Memory includes preparation, solver/diagnostics and final float64 validation. '
              'Per-rank, per-phase measurements and their three-run summaries are preserved in report.json. '
              'A Sinkhorn stopping miss means the requested inner tolerance was not reached within the cap; '
              'the corresponding completed run remains visible and its rounded final feasibility is checked.', '',
        '| Method | Symmetric exact-cell top-1 accuracy | Normalized true-match rank | True-pair mass |',
        '|---|---:|---:|---:|']
    for method in METHODS:
        m = aggregate['methods'][method]['summaries']
        lines.append(f"| {names[method]} | {format_stat(m[METRICS[0]])} | {format_stat(m[METRICS[1]])} | "
                     f"{format_stat(m[METRICS[2]])} |")
    lines += ['', 'Top-1 gives uniform credit to exact ties and averages both correspondence directions. '
              f'Its uniform-plan baseline is 1/{n:,} = {uniform:.9g}. Normalized true-match rank counts strictly larger '
              f'weights plus half the tied wrong weights, divided by {n-1:,}; its uniform baseline is 0.5 and lower is better. '
              f'True-pair mass also has uniform baseline {uniform:.9g}. These metrics evaluate exact cell pairing, '
              'not cell-type matching. Objective ranking alone does not demonstrate biological alignment.', '',
        '![Current rounded loss against solver plus current repair time](current_rounded_loss_vs_feasible_time.png)', '',
        '![Current rounded loss against completed outer updates](current_rounded_loss_vs_outer_updates.png)', '',
        'The plots use the recorded full-objective diagnostics of the current rounded outputs, calculated in the '
        'solver’s float32 arithmetic. Final table values use the separate float64 validation. Checkpoints occur '
        'at initialization and the first completed update reaching 10, 30, 100 and 300 solver seconds. '
        'The time axis includes solver work and repair of the particular output, excluding diagnostic overhead '
        'and repairs of earlier outputs. Communication needed by the algorithm is part of solver time. '
        'Lines connect observed summaries only; there is no interpolated target attainment, best-so-far curve '
        'or unobserved continuous trajectory.', '']
    for method in METHODS:
        missed = [(run['seed'], key, row['reason']) for run in report['runs'] if run['method_id'] == method
                  for key, row in run['missed_checkpoints'].items()]
        if missed:
            lines.append(f"Missing recorded checkpoints for {names[method]}: {missed}. "
                         'A checkpoint with fewer than three observed seeds has no median/IQR summary.')
    lines += [f"Final solver workload totals {aggregate['final_solver_gpu_hours_total']:.6g} GPU-hours. "
              f"The recorded campaign workload elapsed time is {aggregate['campaign_workload_seconds']:.6g} seconds, "
              f"or {aggregate['campaign_workload_gpu_hours']:.6g} GPU-hours at four GPUs. "
              'That workload clock includes correctness, pilots and final execution after its start; it is not '
              'the full Slurm allocation accounting. Full allocation GPU-hours remain unreported until sacct is collected.', '',
        f"Slurm job: `{report['provenance']['slurm_job_id']}`; world size: 4; "
        f"frozen source manifest SHA-256: `{report['provenance']['source_manifest_sha256']}`. "
        'report.json records exact data-array hashes, runtime hashes, per-run metadata hashes, allocation environment, '
        'trace observations, counters and final feasibility evidence. Saved coupling hashes are taken from runtime '
        f'metadata; this lightweight report does not reread the approximately {plan_gb:.3g} GB of saved plans.', '',
        'FW smoothed-gap diagnostics, when recorded, refer to the current checkpoint and its stated smoothing '
        'parameter. They are not a theorem-matched full-trajectory output statistic or an asymptotic claim.', '']
    return '\n'.join(lines)


def generate(campaign):
    campaign = Path(campaign).resolve()
    evidence, runs, names = audit(campaign)
    frozen = evidence['frozen_campaign']
    n = frozen_dimension(frozen['config'], frozen['instance_manifest'])
    report = dict(schema_version=1, status='completed', expected_final_runs=6, valid_final_runs=len(runs),
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope=frozen['config']['scope'], method_names=names, config=frozen['config'],
        quantile_convention='numpy.quantile(method=linear), three fixed seeds; IQR is not a confidence interval',
        provenance=dict(slurm_job_id=frozen['slurm_job_id'], world_size=4,
            source_manifest_sha256=frozen['source_manifest_sha256'],
            array_files_sha256=frozen['instance_manifest']['array_files_sha256'],
            correspondence_sha256=frozen['correspondence_sha256'],
            instance_manifest=frozen['instance_manifest'], report_source_sha256=sha256(__file__),
            input_json_sha256={key+'.json':sha256(campaign/(key+'.json')) for key in evidence}),
        uniform_baselines=dict(top1_accuracy=1/n, normalized_rank=.5, true_pair_mass=1/n),
        correctness=evidence['correctness'], pilot_gate=evidence['pilot_gate'],
        aggregate=aggregate(evidence, runs, names), runs=runs)
    # The audit finishes before any report artifact is written. Publication of
    # report.json is last, after both requested figures and Markdown complete.
    report['figures'] = figures(campaign, report)
    (campaign / 'report.md').write_text(markdown(report))
    report['markdown_sha256'] = sha256(campaign / 'report.md')
    write(campaign / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.campaign)
    print(json.dumps(dict(status=result['status'], valid_final_runs=result['valid_final_runs'],
                          report=str(args.campaign / 'report.json')), sort_keys=True))


if __name__ == '__main__':
    main()
