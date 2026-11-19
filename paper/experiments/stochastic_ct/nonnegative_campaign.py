"""Freeze and validate the nonnegative-only CT baseline campaign.

All solver forward/adjoint work is budgeted, including cached right-hand sides,
inner NNLS solves and unsuccessful trials. Diagnostic image evaluations are
reported separately. Historical SFRAMES results are read-only references.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
from statistics import median
import sys
import traceback


EXPANDED = Path('/home/tony/frames_ct_2detect_20261104/spectral_balls/expanded')
ROOT = EXPANDED / 'nonnegative_baselines'
REFERENCE = EXPANDED / 'convincing_4x_recovery'
PYTHON = '/home/tony/zofw_experiments/.venv/bin/python'
DEPS = '/home/tony/frames_ct_2detect_20261104/deps'
CPUS = (0, 2, 4, 6)
CAP = 69120
METHODS = ('anls', 'mu')
DEVELOPMENT_CAP = 34560
INNER_GRID = (5, 10, 20)
ANLS_PARAMETERS = dict(memory=10, relative_kkt_tolerance=1e-3, ftol=1e-9, maxls=20)
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
COUNTERS = ('forward_views', 'backward_views', 'forward_calls', 'backward_calls',
            'oracle_calls', 'gradient_calls', 'normal_operator_calls',
            'rhs_backward_calls', 'rhs_backward_views', 'outer_sweeps', 'completed_blocks')
NUMERICAL_SOURCES = ('nonnegative_run.py', 'nonnegative_solvers.py', 'ct_problem.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def frozen(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        require(read(path) == value, f'Frozen input changed: {path}')
    else:
        write(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


def check_record(record):
    path = Path(record['path'])
    require(path.is_file() and digest(path) == record['sha256'],
            f'Frozen file changed: {path}')


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def specification():
    return dict(
        physical_evaluation_runs=56, methods=list(METHODS), rank=64, grid=512,
        training_views=[360, 1080], calibration_seed=201,
        constraint_geometry='Nonnegative factors only; no spectral or Frobenius norm bounds.',
        repeats='One physical run per deterministic method, slice and view count; no duplicate seed runs.',
        initialization='Reuse the frozen calibrated U,V arrays without modifying or renormalizing them.',
        solver_view_budget_per_direction=CAP,
        accounting='Every solver forward and adjoint call, including cached right-hand sides, inner solves and unsuccessful trials, is charged. Diagnostic evaluations are separate.',
        directional_counts='Forward and adjoint totals are checked separately and need not agree.',
        recording='Initialization and every accepted native state, with terminal charged work retained; no interpolation.',
        anls_fixed_parameters=ANLS_PARAMETERS, anls_inner_maxiter_grid=list(INNER_GRID),
        development_runs=12, development_view_budget_per_direction=DEVELOPMENT_CAP,
        parameter_selection='For each inner_maxiter, take the median relative raw training loss across the two development slices at each view count, then the median across the two view counts. Minimize this score, breaking exact ties by smaller inner_maxiter. MU has no tuned parameters.',
        sframes_reruns=False, additional_figures=False, optimizer_timing=False,
        failure_policy='Keep all evidence and halt on operational or numerical failure; no failed-run filtering.')


def inspect_inputs(reference=REFERENCE):
    """Hash/audit prepared inputs without creating a CT projector."""
    for name in THREAD_VARS:
        os.environ[name] = '1'
    sys.path.insert(0, DEPS)
    import numpy as np

    reference = Path(reference)
    cohort = read(reference / 'cohort.json')
    slices = cohort['evaluation_slices']
    development = cohort['development_slices']
    require(len(slices) == len(set(slices)) == 14, 'Expected 14 evaluation slices')
    require(len(development) == len(set(development)) == 2
            and not (set(slices) & set(development)), 'Invalid development split')
    require(sorted(cohort['evaluation_training_views']) == [360, 1080], 'View split changed')
    groups = cohort['evaluation_slice_groups']
    require(set(groups) == set(slices) and len(set(groups.values())) == 6,
            'Expected six prespecified specimen groups')
    prepared = read(reference / 'data_preparation_manifest.json')
    require(prepared['status'] == 'complete', 'Data preparation is incomplete')
    inputs = {}
    for sid in sorted(slices + development):
        seed = 201 if sid in slices else cohort['development_calibration_seed']
        data = (reference / 'data' / f'{sid}.npz').resolve()
        calibration = (reference / 'calibration' / f'{sid}_seed{seed}.npz').resolve()
        data_hash = digest(data)
        if sid in prepared['records']:
            require(data_hash == prepared['records'][sid]['sha256'], 'Prepared data hash changed')
        with np.load(calibration, allow_pickle=False) as saved:
            require(str(saved['data_sha256']) == data_hash, 'Calibration belongs to different data')
            arrays = {name: np.array(saved[name], copy=True) for name in ('U', 'V')}
            for array in arrays.values():
                require(array.shape == (512, 64) and array.dtype == np.float64,
                        'Calibration dimensions or precision changed')
                require(np.isfinite(array).all() and np.all(array > 0),
                        'Shared MU initialization must be finite and strictly positive')
            inputs[sid] = dict(
                data_file=str(data), data_sha256=data_hash,
                calibration_file=str(calibration), calibration_sha256=digest(calibration),
                calibration_seed=seed,
                initial_factor_sha256={name: hashlib.sha256(
                    np.ascontiguousarray(array).tobytes()).hexdigest() for name, array in arrays.items()},
                initial_factor_frobenius_norm=float(np.sqrt(sum(np.sum(a*a) for a in arrays.values()))),
                calibration_problem_sha256=str(saved['problem_sha256']),
                historical_reference_caps=saved['caps'].tolist(),
                historical_reference_caps_are_constraints=False)
    return cohort, inputs


def make_plan(cohort, inputs, parameters):
    require(set(parameters) == set(METHODS), 'Incomplete fixed method parameters')
    configs = []
    for sid in sorted(cohort['evaluation_slices']):
        entry = inputs[sid]
        require(entry['calibration_seed'] == 201, 'Evaluation initialization seed changed')
        for n in (360, 1080):
            for method in METHODS:
                configs.append(dict(
                    run_id=f'evaluation_{sid}_{n}_{method}', phase='evaluation',
                    method=method, slice_id=sid, n_train=n, grid=512, rank=64,
                    calibration_seed=201, view_budget=CAP, record_every_iteration=True,
                    data_file=entry['data_file'], data_sha256=entry['data_sha256'],
                    calibration_file=entry['calibration_file'],
                    calibration_sha256=entry['calibration_sha256'], parameters=parameters[method]))
    require(len(configs) == 56 and len({c['run_id'] for c in configs}) == 56,
            'Expected 56 unique deterministic runs')
    # Equal total view caps dominate work; distribute each method/view stratum
    # evenly before balancing the deterministic queue lengths.
    loads = dict.fromkeys(CPUS, 0)
    strata = {(cpu, method, n): 0 for cpu in CPUS for method in METHODS for n in (360, 1080)}
    assignments = {}
    for cfg in configs:
        cpu = min(CPUS, key=lambda c: (strata[c, cfg['method'], cfg['n_train']], loads[c], c))
        assignments[cfg['run_id']] = cpu
        loads[cpu] += 1
        strata[cpu, cfg['method'], cfg['n_train']] += 1
    return dict(phase='evaluation', expected_runs=56, runs=configs,
                groups=cohort['evaluation_slice_groups'], assignments=assignments,
                worker_run_counts=loads, solver_view_budget_per_direction=CAP,
                deterministic_repeats=1)


def make_development_plan(cohort, inputs):
    runs = []
    for inner in INNER_GRID:
        for sid in sorted(cohort['development_slices']):
            entry = inputs[sid]
            for n in (360, 1080):
                runs.append(dict(
                    run_id=f'development_{sid}_{n}_anls_inner{inner}', phase='development',
                    method='anls', slice_id=sid, n_train=n, grid=512, rank=64,
                    calibration_seed=entry['calibration_seed'], view_budget=DEVELOPMENT_CAP,
                    record_every_iteration=True, parameters=dict(ANLS_PARAMETERS, inner_maxiter=inner),
                    data_file=entry['data_file'], data_sha256=entry['data_sha256'],
                    calibration_file=entry['calibration_file'], calibration_sha256=entry['calibration_sha256']))
    require(len(runs) == 12 and len({c['run_id'] for c in runs}) == 12,
            'Expected all 12 development configurations')
    return dict(phase='development', expected_runs=12, runs=runs,
                assignments={c['run_id']: CPUS[k % len(CPUS)] for k, c in enumerate(runs)},
                groups={sid: 'development' for sid in cohort['development_slices']},
                solver_view_budget_per_direction=DEVELOPMENT_CAP, deterministic_repeats=1)


def freeze_sources(root, gate):
    require(gate.get('status') == 'passed', 'Implementation validation has not passed')
    sources = gate['source_files']
    required = {*NUMERICAL_SOURCES, 'nonnegative_campaign.py'}
    require(required <= set(sources), 'Missing validated numerical/controller sources')
    hashes = {}
    for name, record in sorted(sources.items()):
        require(Path(name).name == name, 'Source freeze expects plain basenames')
        check_record(record)
        raw = Path(record['path']).read_bytes()
        target = root / 'frozen_sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(target.read_bytes() == raw, 'Attempt to overwrite a frozen source')
        else:
            target.write_bytes(raw)
        hashes[name] = record['sha256']
    return hashes


def reference_anchors(expanded):
    """Freeze current scientific references, without pinning mutable live audits."""
    publication = read(expanded / 'gradient_budget/report_update.json')
    require(publication['status'] == 'completed_published', 'Predecessor is not published')
    summary = Path(publication['gradient_comparison_summary'])
    require(digest(summary) == publication['gradient_comparison_summary_sha256'],
            'Published comparison summary changed')
    joint = Path(publication['anchors']['unchanged_joint_summary']['path'])
    require(digest(joint) == publication['anchors']['unchanged_joint_summary']['sha256'],
            'Published joint comparison changed')
    current = read(summary)
    joint_summary = read(joint)
    directories = {Path(a['source_directory']) for c in current['cases'] for a in c['arms'].values()}
    directories |= {Path(c['runs']['tuned_sframes']['source_directory']) for c in joint_summary['cases']}
    require(len(directories) == 448, 'Published six-arm comparison is incomplete')
    files = {str(summary): digest(summary), str(joint): digest(joint)}
    for directory in sorted(directories):
        for name in ('metadata.json', 'result.json', 'trace.json'):
            files[str(directory / name)] = digest(directory / name)
    report = expanded / 'reports/paper_section/revised'
    assets = {str(report / name): digest(report / name) for name in publication['published_sha256']
              if (report / name).is_file()}
    appendix = expanded / 'reports/paper_section/CT_REPRODUCIBILITY.md'
    assets[str(appendix)] = digest(appendix)
    for path in (summary, joint,
                 *(expanded / stage / 'report_update.json' for stage in (
                     'gradient_budget', 'trajectory_mean_sd', 'baseline_step_rules',
                     'spring_localcurvature', 'feasibility_retune')),
                 *(Path(publication['stage']) / f'statistics_{n}.json' for n in (360, 1080))):
        assets[str(path)] = digest(path)
    return files, publication, assets


def prepare(root=ROOT, reference=REFERENCE):
    root, reference = Path(root), Path(reference)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        verify_prepared(root)
        return dict(status='already_prepared', protocol_sha256=digest(root / 'protocol.json'))
    gate = read(root / 'implementation_validation.json')
    sources = freeze_sources(root, gate)
    require(sources['nonnegative_campaign.py'] == digest(__file__), 'Active controller was not validated')
    frozen(root / 'specification.json', specification())
    cohort, inputs = inspect_inputs(reference)
    require(all(entry['calibration_problem_sha256'] == sources['ct_problem.py'] for entry in inputs.values()),
            'Calibration was created with a different CT problem implementation')
    frozen(root / 'cohort.json', cohort)
    frozen(root / 'input_audit.json', dict(status='passed', ct_operator_calls=0, inputs=inputs))
    frozen(root / 'development_plan.json', make_development_plan(cohort, inputs))
    references, publication, assets = reference_anchors(reference.parent)
    frozen(root / 'retained_report_inputs.json', {path: sha for path, sha in references.items()
            if Path(path).name in ('metadata.json', 'trace.json')})
    frozen(root / 'predecessor_publication.json', publication)
    frozen(root / 'publication_predecessor.json', dict(assets=assets,
        original_audit=str(reference.parent / 'gradient_budget/report_update.json'),
        original_audit_sha256=digest(reference.parent / 'gradient_budget/report_update.json')))
    frozen(root / 'implementation_manifest.json', dict(
        numerical_sources=sources, source_files=gate['source_files'],
        implementation_validation_sha256=digest(root / 'implementation_validation.json')))
    frozen_names = ('implementation_validation.json', 'implementation_manifest.json',
                    'specification.json', 'cohort.json', 'input_audit.json',
                    'development_plan.json', 'predecessor_publication.json', 'publication_predecessor.json',
                    'retained_report_inputs.json')
    frozen(root / 'protocol.json', dict(
        host=os.uname().nodename, python=PYTHON, deps=DEPS, cpus=list(CPUS), blas_threads=1,
        numerical_sources={name: sources[name] for name in NUMERICAL_SOURCES}, frozen_sources=sources,
        frozen_files={name: digest(root / name) for name in frozen_names},
        inputs=inputs, reference_files=references,
        reference_protocol_files={str(reference / name): digest(reference / name)
                                  for name in ('cohort.json', 'data_preparation_manifest.json', 'data_group_manifest.json')},
        solver_view_budget_per_direction=CAP, development_view_budget_per_direction=DEVELOPMENT_CAP,
        initializations_unchanged=True, spectral_constraints=False,
        no_evaluation_parameter_selection=True, no_optimizer_timing=True))
    verify_prepared(root)
    update(root, 'prepared', 'awaiting_launch', completed=0, total=56)
    return dict(status='prepared', development_runs=12, evaluation_runs=56,
                protocol_sha256=digest(root / 'protocol.json'))


def verify_prepared(root=ROOT):
    root = Path(root)
    protocol = read(root / 'protocol.json')
    require(digest(__file__) == protocol['frozen_sources']['nonnegative_campaign.py'],
            'Active controller differs from the validated frozen source')
    for name, expected in protocol['frozen_files'].items():
        require(digest(root / name) == expected, f'Frozen protocol file changed: {name}')
    for name, expected in protocol['frozen_sources'].items():
        require(digest(root / 'frozen_sources' / name) == expected, f'Frozen source changed: {name}')
    for entry in protocol['inputs'].values():
        for key in ('data', 'calibration'):
            require(digest(entry[f'{key}_file']) == entry[f'{key}_sha256'], f'{key} changed')
    for path, expected in {**protocol['reference_files'], **protocol['reference_protocol_files']}.items():
        require(digest(path) == expected, f'Historical reference changed: {path}')
    require(read(root / 'specification.json') == specification(), 'Controller specification changed')
    evaluation = root / 'evaluation_protocol.json'
    if evaluation.exists():
        for name, expected in read(evaluation)['frozen_files'].items():
            require(digest(root / name) == expected, f'Frozen evaluation input changed: {name}')
    return protocol


def validate_trace(trace, cfg):
    require(trace and trace[0]['iteration'] == 0, 'Missing initialization observation')
    previous = dict.fromkeys(COUNTERS, 0)
    last_iteration = 0
    cap, n = cfg['view_budget'], cfg['n_train']
    for k, row in enumerate(trace):
        terminal = row.get('observation_type') in ('budget_stop', 'converged')
        require(type(row['iteration']) is int and row['iteration'] >= 0, 'Invalid native iteration')
        require((k == 0 and row['observation_type'] == 'initial')
                or (k > 0 and not terminal and row['iteration'] == last_iteration + 1)
                or (k > 0 and terminal and k == len(trace)-1 and row['iteration'] == last_iteration),
                'Missing native accepted state or repeated nonterminal iteration')
        for name in COUNTERS:
            require(type(row[name]) is int and row[name] >= previous[name],
                    f'Invalid/nonmonotone solver counter: {name}')
        if k == 0:
            require(all(row[name] == 0 for name in COUNTERS), 'Initialization must precede solver work')
        require(row['forward_views'] == n * row['forward_calls']
                and row['backward_views'] == n * row['backward_calls'], 'Directional view accounting mismatch')
        require(row['oracle_calls'] == row['gradient_calls'] + row['rhs_backward_calls'],
                'Solver operator calls omit charged work')
        require(row['forward_calls'] == row['gradient_calls']
                and row['backward_calls'] == row['gradient_calls'] + row['rhs_backward_calls']
                and row['rhs_backward_views'] == n * row['rhs_backward_calls'],
                'Incomplete or uncharged operator pair/RHS adjoint')
        require(row['outer_sweeps'] == row['completed_blocks'] // 2, 'Invalid outer-sweep counter')
        require(max(row['forward_views'], row['backward_views']) <= cap, 'Solver exceeded its view budget')
        for name in ('raw_loss', 'feasible_loss', 'validation_loss', 'd_distance'):
            require(isinstance(row[name], (int, float)) and math.isfinite(row[name]) and row[name] >= 0,
                    f'Invalid native metric: {name}')
        require(row['d_distance'] == 0 and math.isclose(row['raw_loss'], row['feasible_loss'], rel_tol=1e-12, abs_tol=1e-14),
                'Nonnegative baseline is infeasible or uses a different repaired loss')
        require(not any('seconds' in name for name in row), 'Optimizer timing is outside the protocol')
        if cfg['method'] == 'mu':
            require(row['normal_operator_calls'] == row['gradient_calls'] == row['iteration']
                    and row['completed_blocks'] == row['iteration'], 'MU update/work counts disagree')
            require(row['rhs_backward_calls'] == int(k > 0), 'Cached MU adjoint was not charged exactly once')
        else:
            require(row['normal_operator_calls'] == row['rhs_backward_calls'] == 0,
                    'Unexpected ANLS normal/RHS calls')
            require(row['iteration'] <= row['gradient_calls'], 'ANLS accepted steps lack gradient evaluations')
        if k:
            event = row['last_step']
            require(event['observation_type'] == row['observation_type']
                    and event['iteration'] == row['iteration']
                    and all(event[name] == row[name] for name in COUNTERS),
                    'Recorded event/cumulative counters disagree')
            require(terminal or row['observation_type'] == 'accepted_update', 'Unknown native observation')
            if terminal and row['iteration'] == last_iteration:
                require(all(row[name] == trace[k-1][name] for name in ('factor_u_sha256', 'factor_v_sha256')),
                        'Terminal unaccepted work changed factors')
        previous = {name: row[name] for name in COUNTERS}
        last_iteration = row['iteration']
    require(trace[0]['raw_loss'] > 0, 'Relative loss needs a positive initial loss')
    return True


def validate_result(cfg, cpu, plan, protocol, root=ROOT):
    root = Path(root)
    folder = root / 'runs' / cfg['run_id']
    meta, result, trace = (read(folder / name) for name in ('metadata.json', 'result.json', 'trace.json'))
    entry = protocol['inputs'][cfg['slice_id']]
    require(meta['config'] == cfg and meta['cpu'] == cpu, 'Run configuration or CPU changed')
    require(meta['source_sha256'] == protocol['numerical_sources'], 'Run numerical source changed')
    require(meta['numpy'] == '2.5.4' and meta['scipy'] == '1.18.1', 'Frozen numerical environment changed')
    require(meta['geometry'] == 'nonnegative' and meta['spectral_caps_are_constraints'] is False
            and meta['deterministic'] is True, 'Unexpected optimization constraint or repeats')
    require(meta['data_sha256'] == cfg['data_sha256'] and meta['calibration_sha256'] == cfg['calibration_sha256'],
            'Run data or initialization changed')
    factor_hashes = dict(U=meta['initial_u_sha256'], V=meta['initial_v_sha256'])
    require(factor_hashes == entry['initial_factor_sha256'], 'Initial factor arrays changed')
    require(meta['reference_spectral_caps'] == entry['historical_reference_caps'], 'Historical reference radii changed')
    require(meta['protocol_sha256'] == digest(root / 'protocol.json'), 'Run protocol changed')
    require(meta['optimizer'] == cfg['parameters'], 'Optimizer parameters changed')
    offsets = [0] if cfg['n_train'] == 360 else [0, 3, 6]
    require(meta['train_view_ids'] == sorted(base + offset for base in range(0, 3600, 10) for offset in offsets)
            and meta['validation_view_ids'] == list(range(9, 3600, 10)), 'View split changed')
    require(result.get('metadata') == meta, 'Result does not preserve metadata')
    require(result['status'] == 'completed' and result['final'] == trace[-1]
            and result['iteration'] == trace[-1]['iteration'], 'Result/trace completion mismatch')
    validate_trace(trace, cfg)
    final = trace[-1]
    require(result['initial'] == trace[0] and result['observations'] == len(trace), 'Initial/result observation mismatch')
    require(trace[0]['factor_u_sha256'] == factor_hashes['U'] and trace[0]['factor_v_sha256'] == factor_hashes['V'],
            'Initial observed factors differ from shared initialization')
    require(result['termination'] in ('budget_stop', 'converged')
            and final['observation_type'] == result['termination'], 'Unexpected early termination')
    if result['termination'] == 'budget_stop':
        require(cfg['view_budget'] - max(final['forward_views'], final['backward_views']) < cfg['n_train'],
                'Budget stop left room for a complete charged operator action')
    else:
        event = final['last_step']
        require(cfg['method'] == 'anls' and event['block_stop_reason'] == 'kkt'
                and math.isfinite(event['kkt_inf_norm']) and math.isfinite(event['kkt_target'])
                and 0 <= event['kkt_inf_norm'] <= event['kkt_target'], 'Uncertified early convergence')
    return dict(status='completed', run_id=cfg['run_id'], method=cfg['method'],
                slice_id=cfg['slice_id'], n_train=cfg['n_train'], calibration_seed=cfg['calibration_seed'],
                group=plan['groups'][cfg['slice_id']], cpu=cpu, source_directory=str(folder.resolve()),
                metadata_sha256=digest(folder / 'metadata.json'), trace_sha256=digest(folder / 'trace.json'),
                result_sha256=digest(folder / 'result.json'), parameters=cfg['parameters'],
                initial=trace[0], endpoint=final, observations=len(trace),
                data_sha256=meta['data_sha256'], calibration_sha256=meta['calibration_sha256'],
                initial_factor_sha256=factor_hashes, termination=result['termination'],
                initial_factor_frobenius_norm=entry['initial_factor_frobenius_norm'],
                relative_loss=final['raw_loss']/trace[0]['raw_loss'],
                forward_view_slack=cfg['view_budget']-final['forward_views'],
                backward_view_slack=cfg['view_budget']-final['backward_views'],
                endpoint_exceeds_initial=final['raw_loss'] > trace[0]['raw_loss'])


def select_parameters(cases, plan):
    require(len(cases) == len(plan['runs']) == 12, 'Selection requires all development runs')
    expected = {cfg['run_id']: cfg for cfg in plan['runs']}
    require({c['run_id'] for c in cases} == set(expected), 'Development selection cohort changed')
    trials = []
    for inner in INNER_GRID:
        selected = [c for c in cases if expected[c['run_id']]['parameters']['inner_maxiter'] == inner]
        require(len(selected) == 4 and all(c['status'] == 'completed' for c in selected),
                'Incomplete or failed development trial')
        scores = {n: median(c['relative_loss'] for c in selected if c['n_train'] == n) for n in (360, 1080)}
        require(all(math.isfinite(value) and value >= 0 for value in scores.values()), 'Invalid tuning score')
        trials.append(dict(inner_maxiter=inner, view_count_scores=scores, score=median(scores.values()),
                           run_ids=sorted(c['run_id'] for c in selected)))
    winner = min(trials, key=lambda row: (row['score'], row['inner_maxiter']))
    return dict(status='selected_from_development_only', trials=trials,
                criterion=specification()['parameter_selection'],
                parameters={'anls': dict(ANLS_PARAMETERS, inner_maxiter=winner['inner_maxiter']), 'mu': {}},
                selected_inner_maxiter=winner['inner_maxiter'])


def prepare_evaluation(root, development_cases, protocol):
    root = Path(root)
    dev = read(root / 'development_plan.json')
    frozen(root / 'development_validation.json', dict(status='completed_validated', cases=development_cases,
                                                    development_plan_sha256=digest(root / 'development_plan.json')))
    frozen(root / 'selection.json', select_parameters(development_cases, dev))
    selection = read(root / 'selection.json')
    frozen(root / 'evaluation_plan.json', make_plan(read(root / 'cohort.json'), protocol['inputs'], selection['parameters']))
    frozen(root / 'evaluation_protocol.json', dict(frozen_files={name: digest(root / name) for name in (
        'protocol.json', 'development_validation.json', 'selection.json', 'evaluation_plan.json')}))
    return read(root / 'evaluation_plan.json')


def update(root, status, phase, **values):
    value = dict(status=status, phase=phase, updated_utc=now(), pid=os.getpid(), **values)
    write(Path(root) / 'progress.json', value)
    print(json.dumps(value, separators=(',', ':')), flush=True)


def worker(cfg, cpu, root):
    root = Path(root)
    require(not (root / 'execution_abort.json').exists(), 'Campaign was halted')
    for name in THREAD_VARS:
        os.environ[name] = '1'
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, DEPS)
    sys.path.insert(0, str(root / 'frozen_sources'))
    from nonnegative_run import run
    return run(cfg, cpu=cpu, root=root)


def run_plan(plan, protocol, root=ROOT):
    root = Path(root)
    completed, queues = {}, {cpu: [] for cpu in CPUS}
    for cfg in plan['runs']:
        cpu = plan['assignments'][cfg['run_id']]
        if (root / 'runs' / cfg['run_id'] / 'result.json').exists():
            completed[cfg['run_id']] = validate_result(cfg, cpu, plan, protocol, root)
        else:
            queues[cpu].append(cfg)
    update(root, 'running', plan['phase'], completed=len(completed), total=len(plan['runs']))
    for name in THREAD_VARS:
        os.environ[name] = '1'
    with ExitStack() as stack:
        pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1,
                 mp_context=multiprocessing.get_context('spawn'))) for cpu in CPUS}
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
                except Exception as exc:
                    write(root / 'execution_abort.json', dict(active=True, run_id=cfg['run_id'], error=repr(exc), utc=now()))
                    for other in pending:
                        other.cancel()
                    raise
                update(root, 'running', plan['phase'], completed=len(completed), total=len(plan['runs']),
                       last_completed=cfg['run_id'])
                submit(cpu)
    return [completed[cfg['run_id']] for cfg in plan['runs']]


def check_publication(root):
    helper = read(root / 'publication_helper.json')
    require(helper.get('status') == 'passed', 'Report helper has not passed validation')
    for path, expected in helper['dependencies'].items():
        require(digest(path) == expected, f'Publication dependency changed: {path}')
    command = helper['command']
    require(len(command) >= 2 and command[0] == PYTHON and '--publish' in command
            and command[1] in helper['dependencies'], 'Invalid or unpinned publication command')
    return helper


def execute(root=ROOT, publish=False):
    root = Path(root)
    protocol = verify_prepared(root)
    require(os.uname().nodename == protocol['host'], 'Campaign is on the wrong host')
    require(Path(sys.executable).absolute() == Path(PYTHON), 'Wrong campaign Python environment')
    require(set(CPUS) <= os.sched_getaffinity(0), 'Frozen worker CPUs are unavailable')
    require(not (root / 'execution_abort.json').exists(), 'Review retained operational failure before resuming')
    launch = dict(protocol_sha256=digest(root / 'protocol.json'), publish=publish)
    if publish:
        check_publication(root)
        launch['publication_helper_sha256'] = digest(root / 'publication_helper.json')
    frozen(root / 'launch.json', launch)
    development = run_plan(read(root / 'development_plan.json'), protocol, root)
    plan = prepare_evaluation(root, development, protocol)
    verify_prepared(root)
    cases = run_plan(plan, protocol, root)
    verify_prepared(root)
    frozen(root / 'validation_summary.json', dict(
        status='completed_validated', expected_runs=56, completed_runs=len(cases), cases=cases,
        protocol_sha256=digest(root / 'protocol.json'), evaluation_protocol_sha256=digest(root / 'evaluation_protocol.json'),
        evaluation_plan_sha256=digest(root / 'evaluation_plan.json'), selection_sha256=digest(root / 'selection.json'),
        solver_view_budget_per_direction=CAP, spectral_constraints=False, deterministic_repeats=1,
        endpoint_exceeds_initial_count=sum(c['endpoint_exceeds_initial'] for c in cases)))
    update(root, 'evaluated', 'evaluation_validated', completed=56, total=56)
    if publish:
        require(digest(root / 'publication_helper.json') == launch['publication_helper_sha256'], 'Publication helper changed')
        helper = check_publication(root)
        subprocess.run(helper['command'], check=True, cwd=root)
        update(root, 'completed', 'report_published', completed=56, total=56)


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
            print(json.dumps(prepare(args.root), indent=2))
        elif args.verify:
            verify_prepared(args.root)
            print(json.dumps(dict(status='verified', protocol_sha256=digest(args.root / 'protocol.json'))))
        else:
            try:
                execute(args.root, publish=args.publish)
            except Exception as exc:
                update(args.root, 'failed', 'execution', error=repr(exc), traceback=traceback.format_exc())
                raise


if __name__ == '__main__':
    main()
