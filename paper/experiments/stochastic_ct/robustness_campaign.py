"""Predeclared, staged CT evaluation of stochastic FraMES robustness.

The controller never selects parameters from evaluation or confirmation data.
Scientific duplicates across evaluation studies share one physical trajectory.
Runtime sources, inputs, plans and selection records are immutable; checkpoints
and progress files are the only evolving numerical evidence.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
from statistics import median
import sys
import traceback


EXPANDED = Path('/home/tony/frames_ct_2detect_20261104/spectral_balls/expanded')
ROOT = EXPANDED / 'reviewer_robustness_20261122'
PYTHON = '/home/tony/zofw_experiments/.venv/bin/python'
DEPS = '/home/tony/frames_ct_2detect_20261104/deps'
CPUS = (0, 2, 4, 6)
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
CAP, SCREEN_CAP, DEVELOPMENT_CAP = 69120, 8640, 34560
VIEWS = (120, 360, 1080, 2160)
DEV_SEEDS, MAIN_SEEDS, STRESS_SEEDS = (110, 111, 112), tuple(range(202, 207)), tuple(range(202, 212))
BETA_GRID, GAMMA_GRID = (.25, 1., 4., 16., 64.), (.025, .05, .1, .2, .4)
CURVATURE_GRID, MEMORY_GRID = tuple(2.**k for k in range(-5, 3)), (5, 10, 20, 40, 80)
FEASIBILITY_THRESHOLDS = (.001, .005, .01, .02)
FAMILIES = {
    'sframes_b16': ('sframes', 16), 'sframes_b64': ('sframes', 64),
    'sframes_b256': ('sframes', 256), 'sframes_full': ('sframes', 'full'),
    'sframes_noavg_b64': ('sframes_noavg', 64), 'frames': ('frames', 'full'),
}
BASELINE_FIXED = dict(relative_kkt_tolerance=1e-3, ftol=1e-9, maxls=20)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def frozen(path, value):
    value = json.loads(canonical(value))
    if Path(path).exists():
        require(read(path) == value, 'Frozen record changed: ' + str(path))
    else:
        write(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


def update(root, status, phase, **details):
    write(Path(root) / 'progress.json', dict(status=status, phase=phase, updated_utc=now(), **details))


def specification():
    return dict(schema_version=1, training_views=list(VIEWS), grid=512, main_rank=64,
        main_oracle_seeds=list(MAIN_SEEDS), stress_oracle_seeds=list(STRESS_SEEDS),
        development_oracle_seeds=list(DEV_SEEDS), deterministic_repetitions=1,
        solver_view_budget_per_direction=CAP, screen_view_budget_per_direction=SCREEN_CAP,
        development_view_budget_per_direction=DEVELOPMENT_CAP,
        beta_scale_grid=list(BETA_GRID), gamma0_grid=list(GAMMA_GRID),
        families={name: dict(method=method, batch=batch) for name, (method, batch) in FAMILIES.items()},
        shortlist='Top three joint-target candidates plus the loss-only winner if distinct; at most four per family.',
        smoothing_selection='Use development data only. Maximize the fraction attaining repaired loss <=30% of initial and raw orthant distance <=1% of initial factor norm. Among positive-hit ties, minimize nested-median repaired loss, then distance. If no candidate attains either joint target, minimize nested-median max(loss_ratio/0.30,distance_ratio/0.01), then loss and distance. Exact ties prefer smaller beta_scale then gamma0.',
        loss_only_selection='Among confirmed candidates, minimize nested-median repaired loss, then raw distance, beta_scale and gamma0.',
        aggregation='Median over seeds within each slice/view case, then slices within each view count, then view counts. Target attainment uses means at the same levels.',
        coarse_screen='Two development slices, N1080, one oracle seed; full 5x5 grid for all six smoothing families.',
        confirmation='Shortlisted settings at N360 and N1080 on both development slices; three stochastic seeds or one deterministic run.',
        baseline_development=dict(prox_curvature_scale=list(CURVATURE_GRID), lbfgsb_memory=list(MEMORY_GRID),
            views=[360, 1080], budget=DEVELOPMENT_CAP, stochastic_seeds=list(DEV_SEEDS)),
        boundary_policy='The finite search grids are explicit. Boundary winners are recorded and never described as an interior optimum or evidence of broad tuning robustness.',
        evaluation_studies=['main', 'batch', 'averaging_control', 'radius', 'rank', 'initialization',
                            'stress', 'fresh_confirmation', 'condition_retuning', 'stationarity'],
        radius_scales=[1, 2, 4], ablation_ranks=[32, 64, 128],
        initializations=['standard', 'perturbed', 'random'],
        radius_definition='radius_scale times each standard reference factor operator norm; identical domain across initialization choices.',
        stationarity='The Mix1 and LavaStone panel slices at N1080; SFRAMES joint three seeds and deterministic FraMES once, at 4x the main cap. Record true full-gradient smoothed gaps at every iteration, with diagnostic work separate.',
        per_condition_retuning='Select from the same development confirmations separately at N360 and N1080. Evaluate on the six-slice panel only when parameters differ from the globally selected setting; three stochastic seeds or one deterministic run.',
        feasibility_thresholds=list(FEASIBILITY_THRESHOLDS), loss_threshold=.30,
        budget_checkpoints=[8640, 17280, 34560, 69120],
        work='Gradient/solver forward and adjoint views have separate caps. Curvature-estimation views remain separately recorded, with no optimizer timing. All L-BFGS-B line-search trials are charged.',
        recording='Initialization, every accepted native update, and terminal charged-work state; no interpolation.',
        failures='Nonfinite development candidates are retained and ineligible. Operational/projection/provenance failures and all evaluation failures halt the campaign.',
        paper_layout=dict(figures=2, tables=1), optimizer_timing=False,
        statistical_unit='Acquisition groups; repeated seeds and slices within a group are not independent specimens.',
        historical_reuse='Only cataloged dense trajectories with verified matching objective, initialization, parameters, stochastic streams and hard-budget semantics may satisfy a new planned case. Normalized records preserve the original executed source hashes and origin-artifact digests, explicitly mark historical_reuse, and never claim the new engine executed them. Unsupported or uncertain equivalences run anew.',
        old_results='Previously published runs and numerical sources are retained unchanged. New protocol results are analyzed separately before integration.')


def input_key(sid, rank=64, initialization='standard'):
    return f'{sid}:{rank}:{initialization}'


def deterministic(method, parameters):
    return method in ('frames', 'joint_lbfgsb') or (
        method in ('sframes', 'sframes_noavg') and parameters.get('batch') == 'full')


def make_config(inputs, *, phase, study, variant, sid, n, method, parameters,
                seed=None, cap=CAP, rank=64, initialization='standard', radius_scale=2,
                record_stationarity=False):
    entry = inputs[input_key(sid, rank, initialization)]
    cfg = dict(phase=phase, study=study, variant=variant, method=method, slice_id=sid,
        n_train=n, grid=512, rank=rank, initialization=initialization,
        radius_scale=radius_scale, seed=seed if seed is not None else (110 if phase == 'development' else 201),
        view_budget=cap, parameters=dict(parameters), record_stationarity=record_stationarity,
        record_every_iteration=True,
        **{key: entry[key] for key in ('data_file', 'data_sha256', 'calibration_file', 'calibration_sha256', 'calibration_seed')})
    # Study/variant labels describe analyses, not distinct scientific runs.
    signature = {key: value for key, value in cfg.items() if key not in ('study', 'variant')}
    cfg['run_id'] = f"{phase}_{sid}_n{n}_{method}_" + hashlib.sha256(canonical(signature).encode()).hexdigest()[:16]
    return cfg


def estimated_updates(cfg):
    method, p, n = cfg['method'], cfg['parameters'], cfg['n_train']
    if method in ('sframes', 'sframes_noavg', 'frames'):
        batch = n if p['batch'] == 'full' else p['batch']
    elif method == 'joint_lbfgsb':
        batch = n
    elif method == 'spring_sarah':
        batch = 4 * p.get('batch', 64)
    else:
        batch = 2 * max(1, int(math.sqrt(n)))
    return cfg['view_budget'] / batch * (n / 1080) * (cfg['rank'] / 64)


def make_plan(phase, configs, groups=None):
    """Deduplicate physical work while preserving all study/variant memberships."""
    runs, memberships = {}, {}
    for cfg in configs:
        rid = cfg['run_id']
        if rid in runs:
            ignored = {'study', 'variant'}
            require({k: v for k, v in cfg.items() if k not in ignored} ==
                    {k: v for k, v in runs[rid].items() if k not in ignored}, 'Run identifier collision')
        else:
            runs[rid] = dict(cfg)
        memberships.setdefault(rid, [])
        label = dict(study=cfg['study'], variant=cfg['variant'])
        if label not in memberships[rid]:
            memberships[rid].append(label)
    loads, assignments = dict.fromkeys(CPUS, 0.), {}
    for cfg in sorted(runs.values(), key=lambda c: (-estimated_updates(c), c['run_id'])):
        cpu = min(CPUS, key=lambda c: (loads[c], c))
        assignments[cfg['run_id']] = cpu
        loads[cpu] += estimated_updates(cfg)
    studies = {}
    for rid, labels in memberships.items():
        runs[rid]['memberships'] = labels
        runs[rid]['studies'] = sorted({label['study'] for label in labels})
        runs[rid]['variants'] = sorted({label['variant'] for label in labels})
        for label in labels:
            studies.setdefault(label['study'], set()).add(rid)
    return dict(phase=phase, stage=phase, expected_runs=len(runs), runs=list(runs.values()),
        assignments=assignments, memberships=memberships, groups=groups or {},
        studies={name: sorted(ids) for name, ids in sorted(studies.items())},
        study_run_counts={name: len(ids) for name, ids in sorted(studies.items())},
        worker_run_counts={cpu: sum(v == cpu for v in assignments.values()) for cpu in CPUS})


def smoothing_parameters(family, beta, gamma):
    return dict(batch=FAMILIES[family][1], beta_scale=beta, gamma0=gamma, rho0=1.)


def coarse_plan(cohort, inputs):
    configs = [make_config(inputs, phase='development', study='coarse', variant=family,
               sid=sid, n=1080, method=method, seed=110, cap=SCREEN_CAP,
               parameters=smoothing_parameters(family, beta, gamma))
        for family, (method, _) in FAMILIES.items() for beta in BETA_GRID for gamma in GAMMA_GRID
        for sid in cohort['development_slices']]
    return make_plan('coarse', configs)


def baseline_plan(cohort, inputs):
    configs = []
    for sid in cohort['development_slices']:
        for n in (360, 1080):
            for scale in CURVATURE_GRID:
                for seed in DEV_SEEDS:
                    configs.append(make_config(inputs, phase='development', study='baseline_development',
                        variant='prox_sarah', sid=sid, n=n, method='prox_sarah', seed=seed,
                        cap=DEVELOPMENT_CAP, parameters=dict(curvature_scale=scale)))
            for memory in MEMORY_GRID:
                configs.append(make_config(inputs, phase='development', study='baseline_development',
                    variant='joint_lbfgsb', sid=sid, n=n, method='joint_lbfgsb',
                    cap=DEVELOPMENT_CAP, parameters=dict(BASELINE_FIXED, memory=memory)))
    return make_plan('baseline_development', configs)


def nested_score(cases, field, *, average=False):
    reducer = (lambda values: sum(values) / len(values)) if average else median
    by_view = {}
    for n in sorted({c['n_train'] for c in cases}):
        by_slice = []
        for sid in sorted({c['slice_id'] for c in cases if c['n_train'] == n}):
            chosen = [c[field] for c in cases if c['n_train'] == n and c['slice_id'] == sid]
            by_slice.append(reducer(chosen))
        by_view[n] = reducer(by_slice)
    return reducer(list(by_view.values()))


def score_candidates(cases, plan, *, n_train=None):
    configs = {c['run_id']: c for c in plan['runs'] if n_train is None or c['n_train'] == n_train}
    records = {c['run_id']: c for c in cases if c['run_id'] in configs}
    require(set(records) == set(configs), 'Parameter selection requires every planned development outcome')
    require(len(records) == sum(c['run_id'] in configs for c in cases), 'Duplicate development outcome')
    require(all(c['phase'] == 'development' for c in configs.values()), 'Evaluation data entered parameter selection')
    grouped = {}
    for rid, cfg in configs.items():
        grouped.setdefault((cfg['variant'], canonical(cfg['parameters'])), []).append(records[rid])
    scores = []
    for (variant, encoded), chosen in sorted(grouped.items()):
        params = json.loads(encoded)
        valid = all(c['status'] == 'completed' for c in chosen)
        row = dict(variant=variant, parameters=params, eligible=valid, run_ids=sorted(c['run_id'] for c in chosen))
        if valid:
            enriched = [dict(c, joint_hit=float(c['relative_loss'] <= .30 and c['relative_distance'] <= .01),
                target_ratio=max(c['relative_loss']/.30, c['relative_distance']/.01)) for c in chosen]
            row.update(loss=nested_score(enriched, 'relative_loss'), distance=nested_score(enriched, 'relative_distance'),
                joint_attainment=nested_score(enriched, 'joint_hit', average=True),
                target_ratio=nested_score(enriched, 'target_ratio'),
                threshold_attainment={str(eps): nested_score([dict(c, hit=float(c['relative_loss'] <= .30 and c['relative_distance'] <= eps))
                    for c in chosen], 'hit', average=True) for eps in FEASIBILITY_THRESHOLDS})
        scores.append(row)
    return scores


def tie_key(row):
    p = row['parameters']
    return (p.get('beta_scale', 0), p.get('gamma0', 0), p.get('curvature_scale', 0), p.get('memory', 0))


def ordered(scores, mode='joint'):
    valid = [s for s in scores if s['eligible']]
    require(valid, 'No eligible development candidate remains')
    if mode == 'loss':
        return sorted(valid, key=lambda s: (s['loss'], s['distance'], tie_key(s)))
    has_hit = any(s['joint_attainment'] > 0 for s in valid)
    return sorted(valid, key=lambda s: ((-s['joint_attainment'], s['loss'], s['distance']) if has_hit
                                      else (s['target_ratio'], s['loss'], s['distance'])) + tie_key(s))


def choose_shortlist(cases, plan):
    scores = score_candidates(cases, plan)
    choices = {}
    for family in FAMILIES:
        candidates = [s for s in scores if s['variant'] == family]
        selected = ordered(candidates)[:3]
        loss = ordered(candidates, 'loss')[0]
        if loss not in selected:
            selected.append(loss)
        choices[family] = [s['parameters'] for s in selected]
    return dict(status='shortlisted_from_development_only', choices=choices, scores=scores,
                rule=specification()['shortlist'])


def confirmation_plan(cohort, inputs, shortlist):
    configs = []
    for family, settings in shortlist['choices'].items():
        method = FAMILIES[family][0]
        for params in settings:
            for sid in cohort['development_slices']:
                for n in (360, 1080):
                    seeds = (110,) if deterministic(method, params) else DEV_SEEDS
                    for seed in seeds:
                        configs.append(make_config(inputs, phase='development', study='development_confirmation',
                            variant=family, sid=sid, n=n, method=method, seed=seed,
                            cap=DEVELOPMENT_CAP, parameters=params))
    return make_plan('development_confirmation', configs)


def boundary_flags(params):
    grids = {'beta_scale': BETA_GRID, 'gamma0': GAMMA_GRID,
             'curvature_scale': CURVATURE_GRID, 'memory': MEMORY_GRID}
    return [key for key, grid in grids.items() if key in params and params[key] in (min(grid), max(grid))]


def select_parameters(smoothing_cases, smoothing_plan, baseline_cases, baselines):
    def winners(n=None):
        rows = score_candidates(smoothing_cases, smoothing_plan, n_train=n)
        rows += score_candidates(baseline_cases, baselines, n_train=n)
        answer = {}
        for family in (*FAMILIES, 'prox_sarah', 'joint_lbfgsb'):
            selected = [r for r in rows if r['variant'] == family]
            modes = ('joint', 'loss') if family in FAMILIES else ('loss',)
            answer[family] = {}
            for mode in modes:
                row = ordered(selected, mode)[0]
                answer[family][mode] = dict(parameters=row['parameters'], score=row,
                                            grid_boundary_fields=boundary_flags(row['parameters']))
        return answer, rows
    chosen, scores = winners()
    by_view = {str(n): winners(n)[0] for n in (360, 1080)}
    return dict(status='selected_from_development_only', global_selection=chosen,
        per_view_selection=by_view, scores=scores, evaluation_data_used=False,
        selection_criterion=specification()['smoothing_selection'],
        loss_only_criterion=specification()['loss_only_selection'],
        boundary_policy=specification()['boundary_policy'])


def main_variants(selection):
    s = selection
    return {
        'sframes_joint': ('sframes', s['sframes_b64']['joint']['parameters']),
        'sframes_loss': ('sframes', s['sframes_b64']['loss']['parameters']),
        'sframes_full': ('sframes', s['sframes_full']['joint']['parameters']),
        'frames': ('frames', s['frames']['joint']['parameters']),
        'prox_sarah': ('prox_sarah', s['prox_sarah']['loss']['parameters']),
        'joint_lbfgsb': ('joint_lbfgsb', s['joint_lbfgsb']['loss']['parameters']),
        'spring_sarah': ('spring_sarah', dict(batch=64, refresh_period=16)),
    }


def evaluation_plan(cohort, inputs, selection):
    configs = []
    main = main_variants(selection['global_selection'])
    panel = cohort['panel_slices']
    def add(study, variant, sid, n, method, params, *, seeds=MAIN_SEEDS, **kw):
        for seed in ((201,) if deterministic(method, params) else seeds):
            configs.append(make_config(inputs, phase='evaluation', study=study, variant=variant,
                sid=sid, n=n, method=method, parameters=params, seed=seed, **kw))
    for sid in cohort['evaluation_slices']:
        for n in VIEWS:
            for variant, (method, params) in main.items():
                add('main', variant, sid, n, method, params)
    for sid in panel:
        for family in FAMILIES:
            params = selection['global_selection'][family]['joint']['parameters']
            add('batch', family, sid, 1080, FAMILIES[family][0], params)
        add('averaging_control', 'sframes_noavg_paired', sid, 1080, 'sframes_noavg', main['sframes_joint'][1])
        for variant in ('sframes_joint', 'frames', 'prox_sarah', 'joint_lbfgsb'):
            method, params = main[variant]
            if variant != 'joint_lbfgsb':
                for radius in (1, 2, 4):
                    add('radius', variant, sid, 1080, method, params, seeds=MAIN_SEEDS[:3], radius_scale=radius)
            for rank in (32, 64, 128):
                add('rank', variant, sid, 1080, method, params, seeds=MAIN_SEEDS[:3], rank=rank)
            for initial in ('standard', 'perturbed', 'random'):
                add('initialization', variant, sid, 1080, method, params, seeds=MAIN_SEEDS[:3], initialization=initial)
        for variant in ('sframes_joint', 'prox_sarah'):
            method, params = main[variant]
            add('stress', variant, sid, 2160, method, params, seeds=STRESS_SEEDS)
        for n in (360, 1080):
            local = main_variants(selection['per_view_selection'][str(n)])
            for variant, (method, params) in local.items():
                if params != main[variant][1]:
                    add('condition_retuning', variant, sid, n, method, params, seeds=MAIN_SEEDS[:3])
    for sid in cohort['confirmation_slices']:
        for n in (360, 1080):
            for variant, (method, params) in main.items():
                add('fresh_confirmation', variant, sid, n, method, params)
    panel_groups = {cohort['evaluation_slice_groups'][sid]: sid for sid in panel}
    require(len(panel_groups) == 6, 'The diagnostic panel must contain one slice per group')
    diagnostic_slices = [panel_groups[name] for name in ('Mix1', 'LavaStone')]
    for sid in diagnostic_slices:
        for variant in ('sframes_joint', 'frames'):
            method, params = main[variant]
            add('stationarity', variant, sid, 1080, method, params, seeds=MAIN_SEEDS[:3],
                cap=4*CAP, record_stationarity=True)
    groups = cohort['evaluation_slice_groups'] | cohort['confirmation_slice_groups']
    plan = make_plan('evaluation', configs, groups)
    plan['stationarity_slices'] = diagnostic_slices
    return plan


def verify_inputs(cohort, inputs):
    dev, old, fresh = (set(cohort[name]) for name in ('development_slices', 'evaluation_slices', 'confirmation_slices'))
    require(len(dev) == 2 and len(old) == 14 and len(fresh) == 6, 'Unexpected prespecified cohort sizes')
    require(not (dev & old or dev & fresh or old & fresh), 'Development/evaluation/confirmation slices overlap')
    require(set(cohort['panel_slices']) <= old and len(cohort['panel_slices']) == 6, 'Invalid diagnostic panel')
    require(len({cohort['evaluation_slice_groups'][sid] for sid in cohort['panel_slices']}) == 6, 'Panel must span all groups')
    required = {input_key(sid) for sid in dev | old | fresh}
    required |= {input_key(sid, rank) for sid in cohort['panel_slices'] for rank in (32, 128)}
    required |= {input_key(sid, 64, name) for sid in cohort['panel_slices'] for name in ('perturbed', 'random')}
    require(required <= set(inputs), 'Input preparation is incomplete')
    checked = set()
    for key in required:
        record = inputs[key]
        for kind in ('data', 'calibration'):
            pair = (record[kind + '_file'], record[kind + '_sha256'])
            if pair not in checked:
                require(digest(pair[0]) == pair[1], 'Prepared input changed: ' + pair[0])
                checked.add(pair)


def prepare(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'protocol.json').exists():
        verify_prepared(root)
        return dict(status='already_prepared', protocol_sha256=digest(root / 'protocol.json'))
    gate = read(root / 'implementation_validation.json')
    require(gate.get('status') == 'passed', 'Implementation validation has not passed')
    cohort, inputs = read(root / 'cohort.json'), read(root / 'inputs.json')
    verify_inputs(cohort, inputs)
    sources = {}
    required = {'robustness_campaign.py', 'robustness_run.py', 'robustness_solvers.py', 'robustness_inputs.py'}
    if (root/'historical_reuse_catalog.json').exists():
        required.add('robustness_reuse.py')
    require(required <= set(gate['source_files']), 'Missing validated source dependencies')
    for name, record in gate['source_files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe frozen source path')
        require(digest(record['path']) == record['sha256'], 'Validated source changed: ' + name)
        destination = root / 'frozen_sources' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = Path(record['path']).read_bytes()
        require(not destination.exists() or destination.read_bytes() == raw, 'Frozen source changed: ' + name)
        if not destination.exists():
            destination.write_bytes(raw)
        sources[name] = record['sha256']
    require(sources['robustness_campaign.py'] == digest(__file__), 'Controller was not validated')
    numerical = gate['numerical_sources']
    require(all(sources.get(name) == sha for name, sha in numerical.items()), 'Numerical dependency gate mismatch')
    for name, value in [('specification.json', specification()), ('coarse_plan.json', coarse_plan(cohort, inputs)),
                        ('baseline_development_plan.json', baseline_plan(cohort, inputs))]:
        frozen(root / name, value)
        if name.endswith('_plan.json'):
            frozen(root/'plans'/name, value)
    names = ('implementation_validation.json', 'cohort.json', 'inputs.json', 'specification.json',
             'coarse_plan.json', 'baseline_development_plan.json',
             'plans/coarse_plan.json', 'plans/baseline_development_plan.json')
    if (root/'publication_config.json').exists():
        names += ('publication_config.json',)
    if (root/'historical_reuse_catalog.json').exists():
        names += ('historical_reuse_catalog.json',)
    frozen(root / 'protocol.json', dict(schema_version=1, host=os.uname().nodename, python=PYTHON,
        deps=DEPS, cpus=list(CPUS), blas_threads=1, frozen_sources=sources, numerical_sources=numerical,
        frozen_files={name: digest(root / name) for name in names}, inputs=inputs,
        specification_sha256=digest(root / 'specification.json'), optimizer_timing=False,
        prepared_utc=now()))
    update(root, 'prepared', 'awaiting_launch', completed=0,
           coarse_runs=read(root/'coarse_plan.json')['expected_runs'],
           baseline_development_runs=read(root/'baseline_development_plan.json')['expected_runs'])
    verify_prepared(root)
    return read(root / 'progress.json')


def verify_prepared(root=ROOT):
    root = Path(root)
    protocol = read(root / 'protocol.json')
    require(digest(__file__) == protocol['frozen_sources']['robustness_campaign.py'], 'Active controller changed')
    for name, sha in protocol['frozen_sources'].items():
        require(digest(root/'frozen_sources'/name) == sha, 'Frozen source changed: ' + name)
    for name, sha in protocol['frozen_files'].items():
        require(digest(root/name) == sha, 'Frozen protocol record changed: ' + name)
    require(read(root/'specification.json') == specification(), 'Specification differs from controller')
    verify_inputs(read(root/'cohort.json'), read(root/'inputs.json'))
    for path in root.glob('*_stage_receipt.json'):
        for name, sha in read(path)['files'].items():
            require(digest(root/name) == sha, 'Completed stage evidence changed: ' + name)
    return protocol


def budget_views(row, direction):
    return row.get('gradient_' + direction + '_views', row[direction + '_views'])


def reuse_module(root, protocol):
    """Load only the source/catalog explicitly included in the frozen protocol."""
    root=Path(root)
    name='historical_reuse_catalog.json'
    require(name in protocol['frozen_files'], 'Historical reuse was not declared in the protocol')
    require(digest(root/name)==protocol['frozen_files'][name], 'Historical reuse catalog changed')
    require('robustness_reuse.py' in protocol['frozen_sources'], 'Historical reuse helper was not frozen')
    sys.path.insert(0,str(root/'frozen_sources'))
    module=importlib.import_module('robustness_reuse')
    require(digest(module.__file__)==protocol['frozen_sources']['robustness_reuse.py'], 'Historical reuse helper changed')
    return module


def validate_result(cfg, cpu, protocol, root):
    folder = Path(root) / 'runs' / cfg['run_id']
    failure = folder / 'candidate_failure.json'
    if failure.exists():
        evidence = read(failure)
        require(cfg['phase'] == 'development' and evidence['config'] == cfg
            and evidence['protocol_sha256'] == digest(Path(root)/'protocol.json'), 'Invalid retained development failure')
        return dict(status='invalid_candidate', run_id=cfg['run_id'], method=cfg['method'],
            slice_id=cfg['slice_id'], n_train=cfg['n_train'], seed=cfg['seed'],
            error=evidence['error'], failure_sha256=digest(failure))
    meta, result, trace = (read(folder/name) for name in ('metadata.json', 'result.json', 'trace.json'))
    require(meta['config'] == cfg and meta.get('cpu') == cpu, 'Run configuration/CPU changed')
    historical = meta.get('producer') == 'historical_reuse'
    if historical:
        require(reuse_module(root,protocol).verify_materialized(cfg,cpu,root,protocol=protocol),
                'Historical reuse evidence is invalid')
    else:
        require(meta.get('producer') in (None,'robustness_run'), 'Unknown result producer')
        require(meta['source_sha256'] == protocol['numerical_sources'], 'Run numerical sources changed')
    require(meta.get('numpy') == '2.5.4' and
            (meta.get('scipy') in (None,'1.18.1') if historical else meta.get('scipy') == '1.18.1'),
            'Frozen numerical environment changed')
    require(meta.get('protocol_sha256') == digest(Path(root)/'protocol.json'), 'Run protocol changed')
    require(result.get('metadata') == meta and result.get('trace_sha256') == digest(folder/'trace.json'), 'Result metadata or trace digest changed')
    if result['status'] == 'numerical_failure':
        require(cfg['phase'] == 'development', 'Numerical failure in evaluation requires review')
        return dict(status='invalid_candidate', run_id=cfg['run_id'], method=cfg['method'],
            slice_id=cfg['slice_id'], n_train=cfg['n_train'], seed=cfg['seed'],
            error=result['error'], metadata_sha256=digest(folder/'metadata.json'),
            result_sha256=digest(folder/'result.json'), trace_sha256=digest(folder/'trace.json'))
    require(result['status'] == 'completed' and trace and trace[0]['iteration'] == 0, 'Incomplete run result')
    previous, work = 0, (0,0)
    for index,row in enumerate(trace):
        require(row['iteration'] in (previous, previous+1), 'Missing dense native observation')
        if index == 0:
            require(row.get('observation_type') == 'initial' and
                    all(budget_views(row,d) == 0 for d in ('forward','backward')), 'Initial solver work is nonzero')
        elif row['iteration'] == previous:
            require(index == len(trace)-1 and row.get('observation_type') in ('budget_stop','converged'),
                    'Repeated native iterate is not a terminal work observation')
        else:
            require(row.get('observation_type') == 'accepted_update', 'New native iterate lacks accepted-update event')
        previous = row['iteration']
        require(all(0 <= budget_views(row, d) <= cfg['view_budget'] for d in ('forward', 'backward')), 'Solver view cap exceeded')
        current = tuple(budget_views(row,d) for d in ('forward','backward'))
        require(all(new >= old for new,old in zip(current,work)), 'Solver work decreased')
        work = current
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ('raw_loss', 'feasible_loss', 'd_distance')), 'Invalid native diagnostic')
    initial, endpoint = trace[0], trace[-1]
    require(initial['feasible_loss'] > 0, 'Initial loss cannot normalize this run')
    norm = meta['initial_factor_frobenius_norm']
    require(norm > 0 and math.isfinite(norm), 'Invalid initial factor norm')
    require(result.get('final', endpoint) == endpoint, 'Result endpoint differs from dense trace')
    snapshots = {}
    for cap in specification()['budget_checkpoints']:
        eligible = [row for row in trace if max(budget_views(row,d) for d in ('forward','backward')) <= cap]
        if cap <= cfg['view_budget']:
            snapshots[str(cap)] = eligible[-1]
    return dict(status='completed', run_id=cfg['run_id'], method=cfg['method'],
        slice_id=cfg['slice_id'], n_train=cfg['n_train'], seed=cfg['seed'],
        producer=meta.get('producer','robustness_run'),
        initial=initial, endpoint=endpoint, observations=len(trace),
        relative_loss=endpoint['feasible_loss']/initial['feasible_loss'],
        relative_distance=endpoint['d_distance']/norm,
        initial_factor_frobenius_norm=norm, snapshots=snapshots,
        metadata_sha256=digest(folder/'metadata.json'), result_sha256=digest(folder/'result.json'),
        trace_sha256=digest(folder/'trace.json'), source_directory=str(folder))


def worker(cfg, cpu, root):
    root = Path(root)
    require(not (root/'execution_abort.json').exists(), 'Campaign was halted')
    for name in THREAD_VARS:
        os.environ[name] = '1'
    os.sched_setaffinity(0, {cpu})
    sys.path.insert(0, DEPS)
    sys.path.insert(0, str(root/'frozen_sources'))
    protocol=read(root/'protocol.json')
    if 'historical_reuse_catalog.json' in protocol['frozen_files']:
        if reuse_module(root,protocol).try_materialize(cfg,cpu,root,protocol=protocol):
            return dict(status='completed',producer='historical_reuse')
    from robustness_run import run
    return run(cfg, cpu=cpu, root=root)


def run_plan(plan, protocol, root):
    root = Path(root)
    completed, queues = {}, {cpu: [] for cpu in CPUS}
    for cfg in plan['runs']:
        cpu = plan['assignments'][cfg['run_id']]
        folder = root/'runs'/cfg['run_id']
        if (folder/'result.json').exists() or (folder/'candidate_failure.json').exists():
            completed[cfg['run_id']] = validate_result(cfg,cpu,protocol,root)
        else:
            queues[cpu].append(cfg)
    update(root, 'running', plan['phase'], completed=len(completed), total=plan['expected_runs'])
    for name in THREAD_VARS:
        os.environ[name] = '1'
    with ExitStack() as stack:
        pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'))) for cpu in CPUS}
        pending = {}
        def submit(cpu):
            if queues[cpu]:
                cfg = queues[cpu].pop(0)
                pending[pools[cpu].submit(worker,cfg,cpu,str(root))] = cfg,cpu
        for cpu in CPUS:
            submit(cpu)
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                cfg,cpu = pending.pop(future)
                try:
                    future.result()
                    completed[cfg['run_id']] = validate_result(cfg,cpu,protocol,root)
                except Exception as error:
                    write(root/'execution_abort.json', dict(active=True,run_id=cfg['run_id'],error=repr(error),utc=now()))
                    for other in pending:
                        other.cancel()
                    raise
                update(root,'running',plan['phase'],completed=len(completed),total=plan['expected_runs'],
                    invalid_candidates=sum(c['status']!='completed' for c in completed.values()),
                    reused_runs=sum(c.get('producer')=='historical_reuse' for c in completed.values()),last_completed=cfg['run_id'])
                submit(cpu)
    return [completed[cfg['run_id']] for cfg in plan['runs']]


def complete_stage(root, name, cases, files):
    root = Path(root)
    name_file = name + '_validation.json'
    frozen(root/name_file, dict(status='completed_validated', cases=cases,
        completed_runs=sum(c['status']=='completed' for c in cases),
        reused_runs=sum(c.get('producer')=='historical_reuse' for c in cases),
        invalid_candidates=sum(c['status']!='completed' for c in cases)))
    frozen(root/(name+'_stage_receipt.json'), dict(files={file:digest(root/file) for file in (*files,name_file)}))
    frozen(root/'stages'/(name+'.json'), dict(status='completed', stage=name,
        expected_runs=len(cases), completed_runs=sum(c['status']=='completed' for c in cases),
        reused_runs=sum(c.get('producer')=='historical_reuse' for c in cases),
        invalid_candidates=sum(c['status']!='completed' for c in cases),
        validation_file=name_file, validation_sha256=digest(root/name_file)))


def report_hook(root, stage, publish=False):
    protocol = read(Path(root)/'protocol.json')
    if 'robustness_report.py' not in protocol['frozen_sources']:
        require(not publish, 'Final publication requires the frozen report module')
        return
    require(digest(Path(root)/'frozen_sources/robustness_report.py') == protocol['frozen_sources']['robustness_report.py'],
            'Frozen report source changed')
    sys.path.insert(0,str(Path(root)/'frozen_sources'))
    module = importlib.import_module('robustness_report')
    module.summarize(root,stage)
    if publish:
        module.publish(root)


def execute(root=ROOT, publish=False):
    root=Path(root)
    protocol=verify_prepared(root)
    require(os.uname().nodename==protocol['host'] and Path(sys.executable).absolute()==Path(PYTHON), 'Wrong host/Python environment')
    require(set(CPUS)<=os.sched_getaffinity(0), 'Worker CPUs unavailable')
    require(not (root/'execution_abort.json').exists(), 'Review retained failure before resuming')
    if publish:
        require('publication_config.json' in protocol['frozen_files'], 'Publication destination was not frozen')
    frozen(root/'launch.json',dict(protocol_sha256=digest(root/'protocol.json'),publish=publish))
    cohort,inputs=read(root/'cohort.json'),read(root/'inputs.json')
    coarse=read(root/'coarse_plan.json')
    cases=run_plan(coarse,protocol,root)
    frozen(root/'shortlist.json',choose_shortlist(cases,coarse))
    confirm=confirmation_plan(cohort,inputs,read(root/'shortlist.json'))
    frozen(root/'confirmation_plan.json',confirm)
    frozen(root/'plans/confirmation_plan.json',confirm)
    complete_stage(root,'coarse',cases,('coarse_plan.json','shortlist.json','confirmation_plan.json',
                                     'plans/confirmation_plan.json'))
    report_hook(root,'coarse')
    confirmed=run_plan(confirm,protocol,root)
    complete_stage(root,'confirmation',confirmed,('confirmation_plan.json',))
    baselines=read(root/'baseline_development_plan.json')
    baseline_cases=run_plan(baselines,protocol,root)
    complete_stage(root,'baseline_development',baseline_cases,('baseline_development_plan.json',))
    frozen(root/'selection.json',select_parameters(confirmed,confirm,baseline_cases,baselines))
    plan=evaluation_plan(cohort,inputs,read(root/'selection.json'))
    frozen(root/'evaluation_plan.json',plan)
    frozen(root/'plans/evaluation_plan.json',plan)
    frozen(root/'development_stage_receipt.json',dict(files={name:digest(root/name)
        for name in ('selection.json','evaluation_plan.json','plans/evaluation_plan.json',
                     'confirmation_validation.json','baseline_development_validation.json')}))
    report_hook(root,'development')
    verify_prepared(root)
    evaluated=run_plan(plan,protocol,root)
    complete_stage(root,'evaluation',evaluated,('evaluation_plan.json','selection.json'))
    verify_prepared(root)
    frozen(root/'completion.json',dict(status='completed', evaluation_runs=len(evaluated),
        reused_evaluation_runs=sum(c.get('producer')=='historical_reuse' for c in evaluated),
        protocol_sha256=digest(root/'protocol.json'), selection_sha256=digest(root/'selection.json'),
        evaluation_plan_sha256=digest(root/'evaluation_plan.json'),
        stage_files={str(path.relative_to(root)):digest(path) for path in sorted((root/'stages').glob('*.json'))},
        study_run_counts=plan['study_run_counts']))
    try:
        report_hook(root,'evaluation',publish=publish)
    except Exception as error:
        update(root,'evaluated','publication_pending',completed=len(evaluated),total=len(evaluated),
               error=repr(error),traceback=traceback.format_exc())
        raise
    update(root,'completed','report_published' if publish else 'evaluation_validated',
           completed=len(evaluated),total=len(evaluated),study_run_counts=plan['study_run_counts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--verify',action='store_true')
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:
            print(json.dumps(prepare(args.root),indent=2))
        elif args.verify:
            verify_prepared(args.root)
            print(json.dumps(dict(status='verified',protocol_sha256=digest(args.root/'protocol.json'))))
        else:
            try:
                execute(args.root,publish=args.publish)
            except Exception as error:
                if not (args.root/'completion.json').exists():
                    update(args.root,'failed','execution',error=repr(error),traceback=traceback.format_exc())
                raise


if __name__=='__main__':
    main()
