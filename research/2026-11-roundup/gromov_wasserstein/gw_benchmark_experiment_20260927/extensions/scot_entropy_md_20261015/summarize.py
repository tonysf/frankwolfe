"""Independent audit and plots for the corrected SCOT entropy-MD experiment.

Both curves optimize F_epsilon = f + epsilon sum(P (log(P)-1)).  The plotted
current rounded plans are diagnostics, distinct from the independently sampled
theorem output.  Finite proximal tolerances are not an exact-prox theorem.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
ARRAY_NAMES = ('A.npy', 'B.npy', 'a.npy', 'b.npy')
PLAN_KEYS = ('last_raw', 'last_rounded', 'selected_raw', 'selected_rounded')
STRICT_FEASIBILITY = 100 * sys.float_info.epsilon
INNER_TOL = 1e-7
MARGINAL_ACCUMULATION = ('FP64 sums over each contiguous row and each contiguous copied column; '
                         'both axes use the fast-axis NumPy reduction; no plan repair or tolerance change')
LABELS = {'full': 'Full GW gradient + exact entropy gradient',
          'pair': 'Pair-sampled GW gradient + exact entropy gradient'}
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right, label, tolerance=1e-11):
    require(finite(left) and finite(right) and abs(left-right) <= tolerance,
            f'{label}: inconsistent values {left!r}, {right!r}.')


def safe_file(base, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Expected a safe relative artifact path.')
    path = Path(base) / relative
    require(path.is_file(), 'Missing artifact: ' + str(path))
    return path


def check_hashes(base, hashes, label):
    require(isinstance(hashes, dict) and bool(hashes), 'Missing ' + label + ' hashes.')
    for name, expected in hashes.items():
        require(sha256(safe_file(base, name)) == expected,
                'Changed ' + label + ': ' + name)


def summarize_values(values):
    require(bool(values) and all(finite(v) for v in values), 'Invalid numeric summary.')
    return dict(n=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values))


def validate_config(config):
    require(type(config['iterations']) is int and config['iterations'] > 0,
            'The frozen horizon must be a positive integer.')
    require(config['epsilon'] == .001 and config['pairs_per_call'] == 1024,
            'The SCOT regularization or pair budget changed.')
    require(config['inner_tol'] == config['inner_gap_tol'] == INNER_TOL,
            'The frozen inner stopping criteria changed.')
    checkpoints = config['checkpoint_iterations']
    require(checkpoints == sorted(set(checkpoints)) and checkpoints[0] == 0
            and checkpoints[-1] == config['iterations']
            and all(type(k) is int and 0 <= k <= config['iterations'] for k in checkpoints),
            'Invalid frozen checkpoint list.')
    expected = [('full_gradient', 'full', 811), ('pair_seed811', 'pair', 811),
                ('pair_seed812', 'pair', 812), ('pair_seed813', 'pair', 813)]
    actual = [(run['id'], run['oracle'], run['seed']) for run in config['runs']]
    require(actual == expected, 'The fixed full-gradient/control seed design changed.')
    for run in config['runs']:
        require(run['iterations'] == config['iterations'] and run['output_seed'] == run['seed'] + 20000,
                'Per-run horizon or independent output seed changed.')
    check_hashes(ROOT, config['source_file_sha256'], 'frozen source')


def load_arrays(config, np):
    instance = Path(config['instance'])
    require(set(config['array_files_sha256']) == set(ARRAY_NAMES), 'Incomplete array hash inventory.')
    check_hashes(instance, config['array_files_sha256'], 'input array')
    if 'instance_manifest_sha256' in config:
        require(sha256(instance/'manifest.json') == config['instance_manifest_sha256'], 'Changed instance manifest.')
    A, B, a, b = arrays = [np.load(instance/name, allow_pickle=False) for name in ARRAY_NAMES]
    require(all(x.dtype == np.float64 and np.isfinite(x).all() for x in arrays),
            'The input arrays must all be finite FP64.')
    require(A.shape == (len(a), len(a)) and B.shape == (len(b), len(b))
            and a.ndim == b.ndim == 1 and (a > 0).all() and (b > 0).all(),
            'Invalid distance or marginal dimensions.')
    require(np.array_equal(A, A.T) and np.array_equal(B, B.T)
            and (A >= 0).all() and (B >= 0).all(), 'Invalid distance matrices.')
    close(float(a.sum()), 1., 'Source marginal mass', 1e-14)
    close(float(b.sum()), 1., 'Target marginal mass', 1e-14)
    return arrays


def fp64_marginals(plan, np):
    """Sum both axes with the same contiguous FP64 reduction.

    NumPy's strided axis-zero reduction may accumulate serially.  For the
    1047-square product coupling that error alone exceeds the 100-eps marginal
    threshold.  Copying the transpose gives column sums the same fast-axis
    reduction as row sums.  Only the diagnostic accumulation order changes;
    the stored plan is never adjusted.  Tests cross-check math.fsum on both axes.
    """
    rows = np.ascontiguousarray(plan).sum(axis=1, dtype=np.float64)
    cols = np.ascontiguousarray(plan.T).sum(axis=1, dtype=np.float64)
    return rows, cols


def evaluate_plan(path, arrays, epsilon, np, ot, *, rounded):
    """Independent NumPy contraction plus POT using the ACTUAL marginals."""
    A, B, a, b = arrays
    P = np.load(path, allow_pickle=False)
    require(P.dtype == np.float64 and P.shape == (len(a), len(b))
            and np.isfinite(P).all() and (P >= 0).all(), 'Invalid saved FP64 coupling: ' + str(path))
    rows, cols = fp64_marginals(P, np)
    loss = float(rows @ ((A*A) @ rows) + cols @ ((B*B) @ cols)
                 - 2*np.sum((A @ P @ B.T)*P))
    constant, hA, hB = ot.gromov.init_matrix(A, B, rows, cols, 'square_loss')
    library_loss = float(ot.gromov.gwloss(constant, hA, hB, P))
    close(loss, library_loss, 'Independent NumPy/POT full objective')
    positive = P > 0
    entropy = float(np.sum(P[positive] * (np.log(P[positive]) - 1)))
    row_error, col_error = float(np.abs(rows-a).sum()), float(np.abs(cols-b).sum())
    residuals = dict(row_l1=row_error, col_l1=col_error, both_l1=row_error+col_error,
                     mass_error=float(abs(math.fsum(rows)-math.fsum(a))), min_entry=float(P.min()))
    tolerance = STRICT_FEASIBILITY if rounded else INNER_TOL
    require(residuals['both_l1'] <= tolerance, 'Saved plan violates marginal feasibility: ' + str(path))
    require(all(finite(x) for x in (loss, library_loss, entropy)), 'Nonfinite independent objective.')
    return dict(loss=loss, entropy=entropy, regularized_loss=loss+epsilon*entropy,
                pot_actual_marginals_loss=library_loss,
                independent_loss_absolute_difference=abs(loss-library_loss),
                residuals=residuals, marginal_l1_tolerance=tolerance,
                marginal_accumulation=MARGINAL_ACCUMULATION,
                file_sha256=sha256(path))


def inner_certificates(metadata, config, cmax):
    rows = metadata['inner_trace']
    require(len(rows) == config['iterations'] and metadata['counters']['inner_stopping_failures'] == 0,
            'Missing or failed proximal solves.')
    maxima = dict(marginal_l1=0., normalized_gap_upper_bound=0., primal_witness_marginal_l1=0.)
    inner_iterations = 0
    for k, row in enumerate(rows):
        require(row['iteration'] == k and row['converged'] is True
                and row['proximal_optimality_certified'] is True and row['certificate_valid'] is True,
                'An inner step was not accepted.')
        close(row['eta'], 1/((4*cmax+2*config['epsilon'])*math.sqrt(k+1)), 'Theoretical step size', 1e-14)
        require(row['inner_tolerance'] == row['normalized_gap_tolerance'] == INNER_TOL
                and 0 < row['inner_iterations'] <= metadata['config']['inner_max'],
                'Changed per-step stopping criteria.')
        fields = ('primal_objective', 'dual_objective', 'primal_dual_gap', 'stable_primal_dual_gap',
                  'gap_roundoff_tolerance', 'normalized_primal_dual_gap', 'normalized_gap_upper_bound',
                  'marginal_l1', 'primal_witness_marginal_l1')
        require(all(finite(row[key]) for key in fields), 'Nonfinite proximal certificate.')
        primal, dual = row['primal_objective'], row['dual_objective']
        gap, stable, guard = (row[key] for key in ('primal_dual_gap', 'stable_primal_dual_gap', 'gap_roundoff_tolerance'))
        scale = max(1., abs(primal), abs(dual))
        upper = (max(gap, stable)+guard)/scale
        require(0 <= row['marginal_l1'] <= INNER_TOL
                and 0 <= row['primal_witness_marginal_l1'] <= 1e-12
                and 0 <= guard/scale <= INNER_TOL
                and min(gap, stable) >= -guard and abs(gap-stable) <= guard
                and upper <= INNER_TOL, 'A proximal certificate fails the guarded criteria.')
        close(primal-dual, gap, 'Primal-dual identity', 1e-12*scale)
        close(gap/scale, row['normalized_primal_dual_gap'], 'Normalized gap', 1e-13)
        close(upper, row['normalized_gap_upper_bound'], 'Guarded normalized gap', 1e-13)
        for name in maxima:
            maxima[name] = max(maxima[name], row[name])
        inner_iterations += row['inner_iterations']
    return dict(accepted_solves=len(rows), inner_iterations=inner_iterations, maxima=maxima,
                scope='Numerical acceptance criteria, not an exact-proximal outer convergence guarantee.')


def validate_output(metadata, config):
    selected = metadata['selected_output']
    index, horizon = selected['index'], config['iterations']
    require(type(index) is int and 0 <= index < horizon
            and selected['output_seed'] == metadata['config']['output_seed'],
            'Invalid independently selected pre-update output.')
    probability = 1/math.sqrt(index+1)/sum(1/math.sqrt(k+1) for k in range(horizon))
    close(selected['probability'], probability, 'Independent output probability', 1e-14)
    return dict(index=index, output_seed=selected['output_seed'], probability=probability,
                scope='Independently sampled pre-update numerical output; the theorem stationarity quantity is not evaluated.')


def validate_logged_values(observation, evaluated, prefix):
    for field in ('loss', 'entropy', 'regularized_loss'):
        close(observation[prefix+'_'+field], evaluated[field], prefix+' '+field)
    residuals = observation[prefix+'_residuals']
    require(all(finite(residuals[name]) for name in ('row_l1', 'col_l1', 'both_l1', 'mass_error', 'min_entry')),
            'Invalid recorded plan residuals.')
    for field in ('row_l1', 'col_l1', 'both_l1', 'mass_error', 'min_entry'):
        close(residuals[field], evaluated['residuals'][field], 'Recorded residual '+field, 1e-12)
    require(0 <= residuals['both_l1'] <= evaluated['marginal_l1_tolerance']
            and residuals['min_entry'] >= 0, 'Recorded coupling fails feasibility.')


def audit_run(out, spec, config, arrays, np, ot):
    directory = Path(out)/'runs'/spec['id']
    metadata_path = directory/'metadata.json'
    metadata = read(metadata_path)
    require(metadata['status'] == 'completed', 'Run is not completed: '+spec['id'])
    run_config = metadata['config']
    for key in ('oracle', 'seed', 'output_seed', 'iterations'):
        require(run_config[key] == spec[key], 'Run differs from frozen design: '+key)
    require(run_config['epsilon'] == config['epsilon']
            and run_config['pairs'] == config['pairs_per_call']
            and run_config['inner_tol'] == run_config['inner_gap_tol'] == INNER_TOL
            and Path(run_config['instance']).resolve() == Path(config['instance']).resolve(),
            'Run numerical settings or dataset changed.')
    require(metadata['array_files_sha256'] == config['array_files_sha256'], 'Run input hashes differ.')
    if 'instance_manifest_sha256' in config:
        require(metadata['manifest_sha256'] == config['instance_manifest_sha256'], 'Run manifest hash differs.')
    source = metadata['source_file_sha256']
    check_hashes(ROOT, source, 'run source')
    require(all(config['source_file_sha256'].get(name) == digest for name, digest in source.items()),
            'Run source inventory differs from the frozen campaign.')
    A, B, a, b = arrays
    require(metadata['dimensions'] == dict(n=len(a), m=len(b)), 'Run dimensions differ from input arrays.')
    environment = metadata['environment']
    require(environment['dtype'] == 'float64' and environment['device'] == 'cpu'
            and environment['distributed'] is False
            and environment['torch_threads'] == environment['torch_interop_threads'] == 1
            and environment['deterministic_algorithms'] is True
            and environment['thread_environment'] == THREAD_ENV,
            'The frozen FP64 CPU execution contract changed.')
    if 'cpu' in spec:
        require(environment['cpu_affinity'] == [spec['cpu']], 'CPU affinity differs from the frozen worker core.')
    cmax = max(abs(float(A.min())-float(B.max())), abs(float(A.max())-float(B.min())))**2
    schedule = metadata['schedule']
    for field, expected in dict(cmax=cmax, epsilon=config['epsilon'],
                               effective_cmax=cmax+config['epsilon']/2,
                               relative_smoothness=2*cmax+config['epsilon'],
                               prefactor_fraction=1., eta0=1/(4*cmax+2*config['epsilon']),
                               eta_last=1/((4*cmax+2*config['epsilon'])*math.sqrt(config['iterations']))).items():
        close(schedule[field], expected, 'Recorded schedule '+field, 1e-14)
    require(schedule['inner_tol'] == schedule['inner_gap_tol'] == INNER_TOL
            and schedule['inner_max'] == run_config['inner_max']
            and schedule['inner_check_every'] == run_config['inner_check_every'] == 10,
            'Recorded schedule inner settings differ.')
    counters = metadata['counters']
    horizon = config['iterations']
    require(counters['completed_updates'] == horizon, 'Completed update count differs from horizon.')
    full = spec['oracle'] == 'full'
    require(counters['full_gradient_calls'] == (horizon if full else 0)
            and counters['stochastic_gradient_calls'] == (0 if full else horizon)
            and counters['pair_samples'] == (0 if full else horizon*config['pairs_per_call']),
            'Gradient oracle work counters are inconsistent.')
    certificates = inner_certificates(metadata, config, cmax)
    output = validate_output(metadata, config)
    trace = metadata['trace']
    require([row['iteration'] for row in trace] == config['checkpoint_iterations'],
            'Missing, duplicated, or unexpected checkpoints.')
    cache = {}

    def plan(name, expected, *, rounded):
        path = safe_file(directory, name)
        require(sha256(path) == expected, 'Changed saved plan: '+str(path))
        key = (str(path), rounded)
        if key not in cache:
            cache[key] = evaluate_plan(path, arrays, config['epsilon'], np, ot, rounded=rounded)
        return cache[key]

    curve = []
    for row in trace:
        evaluated = plan(row['plan_file'], row['plan_sha256'], rounded=True)
        validate_logged_values(row, evaluated, 'rounded')
        # Raw checkpoint plans need not be persisted, so report their logged
        # values only as diagnostics and never mark them independently checked.
        require(all(finite(row['raw_'+key]) for key in ('loss', 'entropy', 'regularized_loss')),
                'Nonfinite raw checkpoint objective.')
        close(row['raw_regularized_loss'], row['raw_loss']+config['epsilon']*row['raw_entropy'],
              'Logged raw regularized objective')
        require(0 <= row['raw_residuals']['both_l1'] <= INNER_TOL, 'Raw checkpoint infeasible.')
        curve.append(dict(iteration=row['iteration'], **evaluated))
    initial = np.load(safe_file(directory, trace[0]['plan_file']), allow_pickle=False)
    require(np.allclose(initial, np.outer(a, b), rtol=0, atol=1e-15),
            'The initial coupling is not the frozen product initialization.')
    plans = {}
    require(set(metadata['saved_plans']) == set(PLAN_KEYS), 'Incomplete final/selected plan inventory.')
    for key in PLAN_KEYS:
        item = metadata['saved_plans'][key]
        plans[key] = plan(item['file'], item['file_sha256'], rounded=key.endswith('rounded'))
    validate_logged_values(trace[-1], plans['last_raw'], 'raw')
    validate_logged_values(trace[-1], plans['last_rounded'], 'rounded')
    validate_logged_values(metadata['selected_output'], plans['selected_raw'], 'raw')
    validate_logged_values(metadata['selected_output'], plans['selected_rounded'], 'rounded')
    close(curve[-1]['loss'], plans['last_rounded']['loss'], 'Final checkpoint loss')
    return dict(id=spec['id'], oracle=spec['oracle'], seed=spec['seed'],
                metadata_sha256=sha256(metadata_path), source_file_sha256=source,
                array_files_sha256=metadata['array_files_sha256'], counters=counters,
                schedule=metadata['schedule'], cmax=cmax,
                eta0=1/(4*cmax+2*config['epsilon']), certificates=certificates,
                selected_output=output, plans=plans, curve=curve,
                independent_saved_plan_evaluations=len(cache),
                full_loss_reduction_percent=100*(1-plans['last_rounded']['loss']/curve[0]['loss']),
                environment=metadata['environment'])


def aggregate(runs, checkpoints):
    rows = []
    for oracle in ('full', 'pair'):
        selected = [run for run in runs if run['oracle'] == oracle]
        if not selected:
            continue
        for k in checkpoints:
            observations = [next(row for row in run['curve'] if row['iteration'] == k) for run in selected]
            row = dict(oracle=oracle, iteration=k, n=len(selected))
            for field in ('loss', 'entropy', 'regularized_loss'):
                values = summarize_values([item[field] for item in observations])
                row.update({field+'_'+key: values[key] for key in ('median', 'minimum', 'maximum')})
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(report_dir, rows, epsilon, partial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    colors = dict(full='#b7492b', pair='#2469a5')
    for ax, field, ylabel in zip(axes, ('loss', 'regularized_loss'),
                                ('Full squared GW loss $f(P)$', r'Regularized objective $F_\varepsilon(P)$')):
        for oracle in ('full', 'pair'):
            group = [row for row in rows if row['oracle'] == oracle]
            if not group:
                continue
            x = [row['iteration'] for row in group]
            label = LABELS[oracle] + (' (one deterministic run)' if oracle == 'full' else f" (median of {group[0]['n']} seeds)")
            ax.plot(x, [row[field+'_median'] for row in group], color=colors[oracle], lw=2,
                    marker='o', ms=3.5, label=label)
            if oracle == 'pair':
                ax.fill_between(x, [row[field+'_minimum'] for row in group],
                                [row[field+'_maximum'] for row in group], color=colors[oracle], alpha=.17)
        ax.set_xlabel('Outer iterations')
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=.22)
    fig.suptitle(('PRELIMINARY — completed runs only\n' if partial else '')
                 + f'SCOT 1047 cells · FP64 · same entropy-MD update and theoretical schedule · ε={epsilon:g}', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=9, bbox_to_anchor=(.5, .055))
    fig.text(.5, .015, 'Current rounded plans; shaded region = observed seed range, not a confidence interval.\n'
             'Both methods include the exact entropy gradient; 1,024 pairs per stochastic update.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .2, 1, .91))
    for suffix in ('png', 'pdf'):
        fig.savefig(report_dir/('loss_vs_iterations.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def build_report(out, *, partial=False, make_plot=True):
    for key, value in THREAD_ENV.items():
        os.environ[key] = value
    import numpy as np
    import ot
    out = Path(out).resolve()
    config_path = out/'config.json'
    config = read(config_path)
    validate_config(config)
    arrays = load_arrays(config, np)
    runs, pending = [], []
    for spec in config['runs']:
        path = out/'runs'/spec['id']/'metadata.json'
        status = read(path).get('status') if path.is_file() else 'missing'
        if status != 'completed':
            require(partial, 'Missing completed run: '+spec['id'])
            pending.append(dict(id=spec['id'], status=status))
            continue
        runs.append(audit_run(out, spec, config, arrays, np, ot))
    require(bool(runs), 'No completed runs available; no report was produced.')
    complete = len(runs) == len(config['runs'])
    rows = aggregate(runs, config['checkpoint_iterations'])
    endpoints = [dict(id=run['id'], oracle=run['oracle'], seed=run['seed'],
                      initial_loss=run['curve'][0]['loss'], final_loss=run['plans']['last_rounded']['loss'],
                      final_entropy=run['plans']['last_rounded']['entropy'],
                      final_regularized_loss=run['plans']['last_rounded']['regularized_loss'],
                      full_loss_reduction_percent=run['full_loss_reduction_percent'],
                      maximum_inner_marginal_l1=run['certificates']['maxima']['marginal_l1'],
                      maximum_inner_normalized_gap=run['certificates']['maxima']['normalized_gap_upper_bound'])
                 for run in runs]
    endpoint_summary = {}
    for oracle in ('full', 'pair'):
        group = [row for row in endpoints if row['oracle'] == oracle]
        if group:
            endpoint_summary[oracle] = {field: summarize_values([row[field] for row in group])
                                       for field in ('final_loss', 'final_regularized_loss', 'full_loss_reduction_percent')}
    report_dir = out/('report' if complete else 'report_partial')
    report_dir.mkdir(exist_ok=True)
    write_csv(report_dir/'curve_data.csv', rows)
    write_csv(report_dir/'endpoints.csv', endpoints)
    lines = [('COMPLETE VALIDATED' if complete else 'PARTIAL: COMPLETED RUNS ONLY'),
             'Both methods optimize F_epsilon=f+epsilon*h, h=sum P(log(P)-1), epsilon=0.001.',
             'Identical product start, FP64 data and iterates, and eta_k=1/((4*Cmax+2*epsilon)*sqrt(k+1)).',
             f"Completed {len(runs)}/{len(config['runs'])} frozen runs; {config['iterations']} outer iterations per run.",
             'Every saved rounded checkpoint was independently evaluated with NumPy and POT.',
             'Both marginal and guarded normalized proximal gap tolerances are 1e-7.',
             'Numerical inner certificates do not establish an exact-proximal convergence theorem.',
             'Curves show current rounded outputs, not the independently sampled theorem output.']
    for oracle, values in endpoint_summary.items():
        loss = values['final_loss']
        lines.append(f"{LABELS[oracle]}: final median full GW loss {loss['median']:.12g}, "
                     f"range [{loss['minimum']:.12g}, {loss['maximum']:.12g}], n={loss['n']}; "
                     f"median reduction {values['full_loss_reduction_percent']['median']:.8g}%.")
    (report_dir/'summary.txt').write_text('\n'.join(lines)+'\n')
    if make_plot:
        plot(report_dir, rows, config['epsilon'], not complete)
    artifacts = {path.name: sha256(path) for path in report_dir.iterdir()
                 if path.is_file() and path.name != 'report.json'}
    report = dict(status='complete_validated' if complete else 'partial_validated',
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  config_sha256=sha256(config_path), reporter_sha256=sha256(__file__),
                  objective='F_epsilon=f+epsilon*sum(P*(log(P)-1)); full squared GW f has no factor 1/2',
                  epsilon=config['epsilon'], iterations=config['iterations'],
                  completed_run_count=len(runs), expected_run_count=len(config['runs']), pending=pending,
                  source_file_sha256=config['source_file_sha256'],
                  array_files_sha256=config['array_files_sha256'],
                  validation_environment=dict(numpy=np.__version__, POT=ot.__version__, thread_environment=THREAD_ENV),
                  marginal_accumulation=MARGINAL_ACCUMULATION,
                  endpoint_summary=endpoint_summary, runs=runs, artifacts_sha256=artifacts,
                  scope='Current rounded plan losses are diagnostics. No last-iterate or finite-prox-error theorem claim. Seed bands are observed ranges; timing is not compared.')
    write(report_dir/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--partial', action='store_true', help='Audit completed runs only and clearly label an incomplete report.')
    args = parser.parse_args()
    report = build_report(args.out, partial=args.partial)
    print(json.dumps(dict(status=report['status'], completed_runs=report['completed_run_count'],
                          expected_runs=report['expected_run_count'], endpoint_summary=report['endpoint_summary']), indent=2))


if __name__ == '__main__':
    main()
