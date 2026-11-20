"""Develop, freeze and run joint nonnegative L-BFGS-B at matched CT view budgets."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import fcntl
import math
import multiprocessing
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import traceback

try:
    from . import nonnegative_campaign as common
except ImportError:
    import nonnegative_campaign as common

read, write, frozen = common.read, common.write, common.frozen
require, digest, now = common.require, common.digest, common.now
CPUS, PYTHON, DEPS, THREAD_VARS = common.CPUS, common.PYTHON, common.DEPS, common.THREAD_VARS
EXPANDED, REFERENCE = common.EXPANDED, common.REFERENCE
ROOT = EXPANDED / 'joint_lbfgsb'
CAP, DEVELOPMENT_CAP = 69120, 34560
METHOD = 'joint_lbfgsb'
MEMORY_GRID = (5, 10, 20)
FIXED_PARAMETERS = dict(relative_kkt_tolerance=1e-3, ftol=1e-9, maxls=20)
NUMERICAL_SOURCES = ('joint_lbfgsb_run.py', 'joint_lbfgsb_solver.py', 'nonnegative_solvers.py', 'ct_problem.py')


def specification():
    return dict(method=METHOD, physical_evaluation_runs=28, development_runs=12,
        rank=64, grid=512, training_views=[360, 1080], calibration_seed=201,
        constraint_geometry='Joint entrywise nonnegative U,V; no spectral or Frobenius bounds.',
        initialization='Unmodified frozen calibrated factors shared with previous methods.',
        deterministic_repeats=1, solver_view_budget_per_direction=CAP,
        development_view_budget_per_direction=DEVELOPMENT_CAP,
        memory_grid=list(MEMORY_GRID), fixed_parameters=FIXED_PARAMETERS,
        parameter_selection='For each memory, median relative raw training loss over development slices at each view count, then median over the two view counts; minimize, breaking exact ties by smaller memory.',
        accounting='Every solver forward/adjoint action including rejected line-search trials counts; diagnostics are separate.',
        recording='Initialization, every accepted joint update, and terminal charged work; no interpolation.',
        stopping='Native L-BFGS-B projected-gradient or relative-function-decrease convergence is retained explicitly; line-search failure halts for review.',
        retained_physical_runs=504, rerun_previous_methods=False, additional_figures=False,
        optimizer_timing=False, failure_policy='Retain evidence and halt on failure; no failed-run filtering.')


def make_config(sid, n, entry, phase, memory):
    return dict(run_id=f'{phase}_{sid}_{n}_joint_lbfgsb' + (f'_memory{memory}' if phase == 'development' else ''),
        phase=phase, method=METHOD, slice_id=sid, n_train=n, grid=512, rank=64,
        calibration_seed=entry['calibration_seed'], record_every_iteration=True,
        view_budget=DEVELOPMENT_CAP if phase == 'development' else CAP,
        parameters=dict(FIXED_PARAMETERS, memory=memory),
        **{key: entry[key] for key in ('data_file', 'data_sha256', 'calibration_file', 'calibration_sha256')})


def make_development_plan(cohort, inputs):
    runs = [make_config(sid, n, inputs[sid], 'development', memory)
            for memory in MEMORY_GRID for sid in sorted(cohort['development_slices']) for n in (360, 1080)]
    require(len(runs) == 12 and all(c['calibration_seed'] == 110 for c in runs), 'Development split changed')
    return dict(phase='development', expected_runs=12, runs=runs,
        assignments={c['run_id']: CPUS[k % len(CPUS)] for k, c in enumerate(runs)},
        groups={sid: 'development' for sid in cohort['development_slices']},
        solver_view_budget_per_direction=DEVELOPMENT_CAP, deterministic_repeats=1)


def make_plan(cohort, inputs, parameters):
    require(set(parameters) == {METHOD} and parameters[METHOD] == dict(FIXED_PARAMETERS, memory=parameters[METHOD]['memory']), 'Unexpected selected parameters')
    memory = parameters[METHOD]['memory']
    require(memory in MEMORY_GRID, 'Selected memory was not developed')
    runs = [make_config(sid, n, inputs[sid], 'evaluation', memory)
            for sid in sorted(cohort['evaluation_slices']) for n in (360, 1080)]
    require(len(runs) == 28 and all(c['calibration_seed'] == 201 for c in runs), 'Evaluation cohort changed')
    # Each worker receives seven runs; the two view counts are balanced to within one run.
    assignments = {c['run_id']: CPUS[(k // 2 + 2 * (k % 2)) % 4] for k, c in enumerate(runs)}
    return dict(phase='evaluation', expected_runs=28, runs=runs, assignments=assignments,
        groups=cohort['evaluation_slice_groups'], worker_run_counts={cpu: 7 for cpu in CPUS},
        solver_view_budget_per_direction=CAP, deterministic_repeats=1)


def reference_anchors(expanded):
    prior = expanded / 'nonnegative_baselines/report_update.json'
    publication = read(prior)
    require(publication['status'] == 'completed_published', 'Predecessor publication is incomplete')
    summary = Path(publication['comparison_summary'])
    require(digest(summary) == publication['comparison_summary_sha256'], 'Previous summary changed')
    cases = read(summary)['cases']
    directories = {Path(c['source_directory']) for c in cases}
    require(len(cases) == len(directories) == 504, 'Expected all 504 previous physical runs')
    files = {str(summary): digest(summary)}
    for directory in sorted(directories):
        for name in ('metadata.json', 'result.json', 'trace.json'):
            files[str(directory / name)] = digest(directory / name)
    report = expanded / 'reports/paper_section/revised'
    assets = {str(report / name): digest(report / name) for name in publication['published_sha256'] if (report / name).is_file()}
    # This is the current editorial revision, not the older PDF hash in the frozen scientific audit.
    for path in (report / 'editorial_revision.json', report.parent / 'CT_REPRODUCIBILITY.md', prior,
                 *(Path(publication['stage']) / f'statistics_{n}.json' for n in (360, 1080))):
        require(path.is_file(), f'Missing predecessor document: {path}')
        assets[str(path)] = digest(path)
    return files, publication, assets


def prepare(root=ROOT, reference=REFERENCE):
    root, reference = Path(root), Path(reference)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        verify_prepared(root)
        return dict(status='already_prepared', protocol_sha256=digest(root / 'protocol.json'))
    gate = read(root / 'implementation_validation.json')
    require(gate.get('status') == 'passed', 'Implementation validation has not passed')
    required = {*NUMERICAL_SOURCES, 'joint_lbfgsb_campaign.py', 'nonnegative_campaign.py'}
    require(required <= set(gate['source_files']), 'Missing validated source dependency')
    sources = {}
    for name, record in sorted(gate['source_files'].items()):
        require(Path(name).name == name, 'Expected source basenames')
        common.check_record(record)
        target = root / 'frozen_sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = Path(record['path']).read_bytes()
        require(not target.exists() or target.read_bytes() == raw, 'Frozen source changed')
        if not target.exists():
            target.write_bytes(raw)
        sources[name] = record['sha256']
    require(sources['joint_lbfgsb_campaign.py'] == digest(__file__), 'Active controller was not validated')
    cohort, inputs = common.inspect_inputs(reference)
    require(all(v['calibration_problem_sha256'] == sources['ct_problem.py'] for v in inputs.values()), 'Calibration CT source changed')
    refs, predecessor, assets = reference_anchors(reference.parent)
    values = dict(specification=specification(), cohort=cohort,
        input_audit=dict(status='passed', ct_operator_calls=0, inputs=inputs),
        development_plan=make_development_plan(cohort, inputs), predecessor_publication=predecessor,
        publication_predecessor=dict(assets=assets),
        retained_report_inputs={p: sha for p, sha in refs.items() if Path(p).name in ('metadata.json', 'trace.json')},
        implementation_manifest=dict(source_files=gate['source_files'], frozen_sources=sources,
            implementation_validation_sha256=digest(root / 'implementation_validation.json')))
    for name, value in values.items():
        frozen(root / (name + '.json'), value)
    frozen(root / 'protocol.json', dict(host=os.uname().nodename, python=PYTHON, deps=DEPS,
        cpus=list(CPUS), blas_threads=1, inputs=inputs,
        numerical_sources={name: sources[name] for name in NUMERICAL_SOURCES}, frozen_sources=sources,
        frozen_files={name: digest(root / name) for name in ['implementation_validation.json', *(n + '.json' for n in values)]},
        reference_files=refs, reference_protocol_files={str(reference / name): digest(reference / name) for name in ('cohort.json', 'data_preparation_manifest.json', 'data_group_manifest.json')},
        solver_view_budget_per_direction=CAP, development_view_budget_per_direction=DEVELOPMENT_CAP,
        spectral_constraints=False, initializations_unchanged=True, no_evaluation_parameter_selection=True, no_optimizer_timing=True))
    verify_prepared(root)
    common.update(root, 'prepared', 'awaiting_launch', completed=0, total=28)
    return dict(status='prepared', development_runs=12, evaluation_runs=28, protocol_sha256=digest(root / 'protocol.json'))


def verify_prepared(root=ROOT):
    root = Path(root)
    protocol = read(root / 'protocol.json')
    require(digest(__file__) == protocol['frozen_sources']['joint_lbfgsb_campaign.py'], 'Active controller changed')
    require(digest(common.__file__) == protocol['frozen_sources']['nonnegative_campaign.py'], 'Shared controller dependency changed')
    for name, sha in protocol['frozen_files'].items():
        require(digest(root / name) == sha, f'Frozen protocol changed: {name}')
    for name, sha in protocol['frozen_sources'].items():
        require(digest(root / 'frozen_sources' / name) == sha, f'Frozen source changed: {name}')
    for entry in protocol['inputs'].values():
        for kind in ('data', 'calibration'):
            require(digest(entry[kind + '_file']) == entry[kind + '_sha256'], kind + ' changed')
    for path, sha in {**protocol['reference_files'], **protocol['reference_protocol_files']}.items():
        require(digest(path) == sha, f'Historical reference changed: {path}')
    require(read(root / 'specification.json') == specification(), 'Frozen specification changed')
    if (root / 'evaluation_protocol.json').exists():
        for name, sha in read(root / 'evaluation_protocol.json')['frozen_files'].items():
            require(digest(root / name) == sha, f'Evaluation input changed: {name}')
    return protocol


def validate_trace(trace, cfg):
    require(cfg['method'] == METHOD, 'Unexpected joint solver method')
    common.validate_trace(trace, cfg)
    for row in trace:
        require(row['completed_blocks'] == row['outer_sweeps'] == 0, 'Joint updates are not alternating blocks')
        require(row['rhs_backward_calls'] == row['rhs_backward_views'] == row['normal_operator_calls'] == 0, 'Unexpected standalone work')
        require(row['forward_calls'] == row['backward_calls'] == row['gradient_calls'] == row['oracle_calls'], 'Joint pair accounting mismatch')
    return True


def validate_result(cfg, cpu, plan, protocol, root=ROOT):
    root = Path(root)
    folder = root / 'runs' / cfg['run_id']
    meta, result, trace = (read(folder / name) for name in ('metadata.json', 'result.json', 'trace.json'))
    entry = protocol['inputs'][cfg['slice_id']]
    require(meta['config'] == cfg and meta['cpu'] == cpu, 'Run configuration or CPU changed')
    require(meta['source_sha256'] == protocol['numerical_sources'], 'Numerical source changed')
    require(meta['numpy'] == '2.5.4' and meta['scipy'] == '1.18.1', 'Frozen numerical environment changed')
    require(meta['geometry'] == 'nonnegative' and meta['spectral_caps_are_constraints'] is False and meta['deterministic'] is True, 'Unexpected optimization domain')
    require(meta['data_sha256'] == cfg['data_sha256'] and meta['calibration_sha256'] == cfg['calibration_sha256'], 'Data/initialization provenance changed')
    factor_hashes = dict(U=meta['initial_u_sha256'], V=meta['initial_v_sha256'])
    require(factor_hashes == entry['initial_factor_sha256'], 'Initial factors changed')
    require(meta['reference_spectral_caps'] == entry['historical_reference_caps'], 'Reference radii changed')
    require(meta['protocol_sha256'] == digest(root / 'protocol.json') and meta['optimizer'] == cfg['parameters'], 'Protocol/parameters changed')
    offsets = [0] if cfg['n_train'] == 360 else [0, 3, 6]
    require(meta['train_view_ids'] == sorted(base + offset for base in range(0, 3600, 10) for offset in offsets) and meta['validation_view_ids'] == list(range(9, 3600, 10)), 'View split changed')
    require(result.get('metadata') == meta and result['status'] == 'completed', 'Result metadata/completion mismatch')
    validate_trace(trace, cfg)
    final = trace[-1]
    require(result['initial'] == trace[0] and result['final'] == final and result['iteration'] == final['iteration'] and result['observations'] == len(trace), 'Result/native observations disagree')
    require(trace[0]['factor_u_sha256'] == factor_hashes['U'] and trace[0]['factor_v_sha256'] == factor_hashes['V'], 'Initial observed factors changed')
    require(result['termination'] in ('budget_stop', 'converged') and final['observation_type'] == result['termination'], 'Unexpected early termination')
    if result['termination'] == 'budget_stop':
        require(cfg['view_budget'] - max(final['forward_views'], final['backward_views']) < cfg['n_train'], 'Budget stop left an affordable gradient pair')
    else:
        event = final['last_step']
        reason = event['stop_reason']
        require(reason in ('kkt', 'relative_function_decrease'), 'Uncertified native convergence')
        require(event['native_task'] == [4, 401 if reason == 'kkt' else 402], 'Native stopping code changed')
        require(math.isfinite(event['solver_loss']) and event['solver_loss'] >= 0, 'Invalid converged solver loss')
        if reason == 'kkt':
            require(math.isfinite(event['kkt_inf_norm']) and math.isfinite(event['kkt_target']) and 0 <= event['kkt_inf_norm'] <= event['kkt_target'], 'Uncertified KKT convergence')
    return dict(status='completed', run_id=cfg['run_id'], method=METHOD, slice_id=cfg['slice_id'],
        n_train=cfg['n_train'], calibration_seed=cfg['calibration_seed'], group=plan['groups'][cfg['slice_id']], cpu=cpu,
        source_directory=str(folder.resolve()), metadata_sha256=digest(folder / 'metadata.json'),
        trace_sha256=digest(folder / 'trace.json'), result_sha256=digest(folder / 'result.json'),
        parameters=cfg['parameters'], initial=trace[0], endpoint=final, observations=len(trace),
        data_sha256=meta['data_sha256'], calibration_sha256=meta['calibration_sha256'],
        initial_factor_sha256=factor_hashes, initial_factor_frobenius_norm=entry['initial_factor_frobenius_norm'],
        termination=result['termination'], relative_loss=final['raw_loss']/trace[0]['raw_loss'],
        forward_view_slack=cfg['view_budget']-final['forward_views'], backward_view_slack=cfg['view_budget']-final['backward_views'],
        endpoint_exceeds_initial=final['raw_loss'] > trace[0]['raw_loss'])


def select_parameters(cases, plan):
    expected = {cfg['run_id']: cfg for cfg in plan['runs']}
    require(len(cases) == len(expected) == 12 and {c['run_id'] for c in cases} == set(expected), 'Selection requires all development runs')
    trials = []
    for memory in MEMORY_GRID:
        chosen = [c for c in cases if expected[c['run_id']]['parameters']['memory'] == memory]
        require(len(chosen) == 4 and all(c['status'] == 'completed' for c in chosen), 'Incomplete or failed development trial')
        scores = {n: median(c['relative_loss'] for c in chosen if c['n_train'] == n) for n in (360, 1080)}
        require(all(math.isfinite(v) and v >= 0 for v in scores.values()), 'Invalid tuning score')
        trials.append(dict(memory=memory, view_count_scores=scores, score=median(scores.values()), run_ids=sorted(c['run_id'] for c in chosen)))
    winner = min(trials, key=lambda c: (c['score'], c['memory']))
    return dict(status='selected_from_development_only', trials=trials, selected_memory=winner['memory'],
        criterion=specification()['parameter_selection'], parameters={METHOD: dict(FIXED_PARAMETERS, memory=winner['memory'])})


def prepare_evaluation(root, cases):
    root = Path(root)
    plan = read(root / 'development_plan.json')
    frozen(root / 'development_validation.json', dict(status='completed_validated', cases=cases, development_plan_sha256=digest(root / 'development_plan.json')))
    frozen(root / 'selection.json', select_parameters(cases, plan))
    frozen(root / 'evaluation_plan.json', make_plan(read(root / 'cohort.json'), read(root / 'protocol.json')['inputs'], read(root / 'selection.json')['parameters']))
    frozen(root / 'evaluation_protocol.json', dict(frozen_files={name: digest(root / name) for name in ('protocol.json', 'development_validation.json', 'selection.json', 'evaluation_plan.json')}))
    return read(root / 'evaluation_plan.json')


def worker(cfg, cpu, root):
    root = Path(root)
    require(not (root / 'execution_abort.json').exists(), 'Campaign was halted')
    for name in THREAD_VARS:
        os.environ[name] = '1'
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, DEPS)
    sys.path.insert(0, str(root / 'frozen_sources'))
    from joint_lbfgsb_run import run
    return run(cfg, cpu=cpu, root=root)


def run_plan(plan, protocol, root):
    completed, queues = {}, {cpu: [] for cpu in CPUS}
    for cfg in plan['runs']:
        cpu = plan['assignments'][cfg['run_id']]
        if (root / 'runs' / cfg['run_id'] / 'result.json').exists():
            completed[cfg['run_id']] = validate_result(cfg, cpu, plan, protocol, root)
        else:
            queues[cpu].append(cfg)
    common.update(root, 'running', plan['phase'], completed=len(completed), total=len(plan['runs']))
    for name in THREAD_VARS:
        os.environ[name] = '1'
    with ExitStack() as stack:
        pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))) for cpu in CPUS}
        pending = {}
        def submit(cpu):
            if queues[cpu]:
                cfg = queues[cpu].pop(0)
                pending[pools[cpu].submit(worker, cfg, cpu, str(root))] = cfg, cpu
        for cpu in CPUS:
            submit(cpu)
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                cfg, cpu = pending.pop(future)
                try:
                    future.result()
                    completed[cfg['run_id']] = validate_result(cfg, cpu, plan, protocol, root)
                except Exception as error:
                    write(root / 'execution_abort.json', dict(active=True, run_id=cfg['run_id'], error=repr(error), utc=now()))
                    for other in pending:
                        other.cancel()
                    raise
                common.update(root, 'running', plan['phase'], completed=len(completed), total=len(plan['runs']), last_completed=cfg['run_id'])
                submit(cpu)
    return [completed[cfg['run_id']] for cfg in plan['runs']]


def execute(root=ROOT, publish=False):
    root = Path(root)
    protocol = verify_prepared(root)
    require(os.uname().nodename == protocol['host'] and Path(sys.executable).absolute() == Path(PYTHON), 'Wrong host/Python environment')
    require(set(CPUS) <= os.sched_getaffinity(0), 'Worker CPUs unavailable')
    require(not (root / 'execution_abort.json').exists(), 'Review retained failure before resuming')
    launch = dict(protocol_sha256=digest(root / 'protocol.json'), publish=publish)
    if publish:
        common.check_publication(root)
        launch['publication_helper_sha256'] = digest(root / 'publication_helper.json')
    frozen(root / 'launch.json', launch)
    development = run_plan(read(root / 'development_plan.json'), protocol, root)
    plan = prepare_evaluation(root, development)
    verify_prepared(root)
    cases = run_plan(plan, protocol, root)
    verify_prepared(root)
    frozen(root / 'validation_summary.json', dict(status='completed_validated', expected_runs=28, completed_runs=len(cases), cases=cases,
        protocol_sha256=digest(root / 'protocol.json'), evaluation_protocol_sha256=digest(root / 'evaluation_protocol.json'),
        evaluation_plan_sha256=digest(root / 'evaluation_plan.json'), selection_sha256=digest(root / 'selection.json'),
        solver_view_budget_per_direction=CAP, spectral_constraints=False, deterministic_repeats=1,
        endpoint_exceeds_initial_count=sum(c['endpoint_exceeds_initial'] for c in cases)))
    common.update(root, 'evaluated', 'evaluation_validated', completed=28, total=28)
    if publish:
        require(digest(root / 'publication_helper.json') == launch['publication_helper_sha256'], 'Publication helper changed')
        subprocess.run(common.check_publication(root)['command'], check=True, cwd=root)
        common.update(root, 'completed', 'report_published', completed=28, total=28)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--verify', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            print(prepare(args.root))
        elif args.verify:
            verify_prepared(args.root)
            print(dict(status='verified', protocol_sha256=digest(args.root / 'protocol.json')))
        else:
            try:
                execute(args.root, publish=args.publish)
            except Exception as error:
                common.update(args.root, 'failed', 'execution', error=repr(error), traceback=traceback.format_exc())
                raise


if __name__ == '__main__':
    main()
