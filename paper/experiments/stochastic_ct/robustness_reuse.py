"""Import verified dense historical trajectories without rerunning optimizers.

The catalog is frozen before new jobs. Imported artifacts retain the original
executed source hashes and complete original evidence. This adapter only adds
counter aliases and, when the old driver took one update beyond its cap,
retains the dense prefix which the new preflight guard would have accepted.
Unavailable historical diagnostic costs and factor hashes remain unavailable.
"""

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SOURCE = Path(__file__).resolve().parent
DEFAULT_HISTORY = Path('/home/tony/frames_ct_2detect_20261104/spectral_balls/expanded')
CATALOG = 'historical_reuse_catalog.json'
STAGES = ('joint_lbfgsb', 'convincing_4x_recovery', 'feasibility_retune',
          'baseline_step_rules', 'gradient_budget')
# These are the audited drivers, including their seed and stopping rules.
DRIVERS = {
    'joint_lbfgsb': ('joint_lbfgsb_run.py', 'a546470c8ed953b334f9cbf2ad37746a8bde98727dd6e027cc2ff537c93ce095'),
    'convincing_4x_recovery': ('run.py', '238836b8d64ff8040263b9723e53dd4fc7559430d25d4e9ccf2a6cef142a2744'),
    'feasibility_retune': ('run.py', 'df714761ccd26f660452ab2fecddf7e300bdad35d03cf9953529e9106c842ded'),
    'baseline_step_rules': ('run.py', '0578f7c60366c51b1ca828697270ec5c0df7c2842ccbb15f675d9def37e736a2'),
    'gradient_budget': ('run.py', '13cbb75c86ff2eeaf6c101c62899af1d1ea04359d39c0505cb263b011d15e115'),
}


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _semantic(path):
    """Ignore only top-level imports changed by vendoring the legacy modules."""
    tree = ast.parse(Path(path).read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.Try) and
                 all(isinstance(item, (ast.Import, ast.ImportFrom)) for item in node.body))]
    return ast.dump(tree, include_attributes=False)


def _parameters(config, metadata=None):
    method, n = config['method'], config['n_train']
    old = metadata is not None
    p = metadata['optimizer'] if old else config['parameters']
    if method == 'joint_lbfgsb':
        defaults = dict(relative_kkt_tolerance=.001, ftol=1e-9, maxls=20, memory=20)
        require(not p.keys()-defaults.keys(), 'Unknown L-BFGS-B parameters')
        return defaults | p
    if method == 'sframes':
        if old:
            return dict(batch=p['batch'], gamma0=p['gamma0'], rho0=p['rho0'],
                        beta_scale=config['beta_scale'])
        require(not p.keys()-{'batch','gamma0','rho0','beta_scale'}, 'Unknown SFRAMES parameters')
        return dict(batch=n if p.get('batch') == 'full' else p.get('batch',64),
                    gamma0=p.get('gamma0',.1), rho0=p.get('rho0',1.), beta_scale=p.get('beta_scale',4.))
    if method == 'prox_sarah':
        if old:
            require(p.get('paper_variant') == 'ProxSARAH-v3', 'Wrong historical ProxSARAH variant')
            return dict(curvature_scale=config['scale_multiplier'])
        require(not p.keys()-{'curvature_scale'}, 'Unknown ProxSARAH parameters')
        return dict(curvature_scale=p.get('curvature_scale',.25))
    if method == 'spring_sarah':
        if old:
            require(p.get('power_iterations') == 5 and p.get('step_coefficient') == .5,
                    'Wrong historical SPRING step rule')
        else:
            require(not p.keys()-{'batch','refresh_period'}, 'Unknown SPRING parameters')
        return dict(batch=p.get('batch',64), refresh_period=p.get('refresh_period',16.))
    raise ValueError('Unsupported historical method')


def _signature(config, metadata=None):
    p = _parameters(config, metadata)
    deterministic = config['method'] == 'joint_lbfgsb' or (
        config['method'] == 'sframes' and p['batch'] == config['n_train'])
    return dict(method=config['method'], phase=config['phase'], slice_id=config['slice_id'],
        n_train=config['n_train'], grid=config['grid'], rank=config['rank'],
        calibration_seed=config['calibration_seed'], seed=None if deterministic else config['seed'],
        view_budget=config.get('view_budget', config.get('gradient_view_budget')),
        data_sha256=(metadata or config)['data_sha256'],
        calibration_sha256=(metadata or config)['calibration_sha256'], parameters=p)


def _gradient(row, direction):
    return row.get('gradient_'+direction+'_views',
                   row[direction+'_views']-row.get('curvature_'+direction+'_views',0))


def _normalize(trace, metadata, cap):
    """Return only observed accepted states; never interpolate or replay updates."""
    require(trace and trace[0]['iteration'] == 0, 'Missing historical initialization')
    method = metadata['config']['method']
    previous, work, output, excluded = -1, (0,0), [], []
    initial_norm = metadata['initial_factor_frobenius_norm']
    for index, original in enumerate(trace):
        iteration = original['iteration']
        terminal = original.get('observation_type') in ('budget_stop','converged')
        require(iteration == previous+1 or (index == len(trace)-1 and terminal and iteration == previous),
                'Historical trajectory is not dense')
        previous = iteration
        counts = tuple(_gradient(original, d) for d in ('forward','backward'))
        require(all(math.isfinite(v) and v >= old for v, old in zip(counts,work)),
                'Historical work counters are invalid')
        work = counts
        require(all(math.isfinite(original[k]) and original[k] >= 0
                    for k in ('raw_loss','feasible_loss','d_distance')), 'Invalid historical diagnostic')
        if index == 0:
            require(counts == (0,0), 'Nonzero initial historical solver work')
        if max(counts) > cap:
            excluded.append(index)
            continue
        require(not excluded, 'Historical trace returned below the cap')
        row = deepcopy(original)
        row['observation_type'] = ('initial' if index == 0 else
                                  original.get('observation_type','accepted_update'))
        # An older L-BFGS-B driver calls its accepted event simply "accepted".
        if row['observation_type'] == 'accepted':
            row['observation_type'] = 'accepted_update'
        for direction, count in zip(('forward','backward'),counts):
            row['gradient_'+direction+'_views'] = count
            row.setdefault('curvature_'+direction+'_views',0)
        row.update(processed_views=max(counts), comparison_views=max(counts),
            total_processed_views=max(row['forward_views'],row['backward_views']),
            comparison_oracle_calls=row['gradient_calls'],
            relative_d_distance=row['d_distance']/initial_norm)
        if method == 'sframes':
            row['lmo_calls'] = 2*iteration
        output.append(row)
    require(output and len(output) > 1, 'No accepted historical updates within the cap')
    if excluded:
        require(method in ('prox_sarah','spring_sarah') and excluded == [len(trace)-1],
                'Only the audited one-update overshoot may be trimmed')
        require(trace[-1]['iteration'] == output[-1]['iteration']+1,
                'Overshoot is not the next native update')
    else:
        require(max(_gradient(trace[-1],d) for d in ('forward','backward')) == cap or
                trace[-1].get('observation_type') == 'converged', 'Historical trajectory stopped early')
    return output, excluded


def _resolve_sources(stage, metadata):
    paths = {}
    for name, sha in metadata['source_sha256'].items():
        choices = (stage/'frozen_sources'/name, stage/name, SOURCE/name,
                   *sorted((stage/'frozen_sources').glob('*/'+name)))
        matches = [p for p in choices if p.is_file() and digest(p) == sha]
        require(matches, 'Missing original executed source: '+str(stage/name))
        paths[name] = dict(path=str(matches[0].resolve()), sha256=sha)
    return paths


def _equivalence(stage_name, metadata, sources):
    driver, sha = DRIVERS[stage_name]
    require(metadata['source_sha256'].get(driver) == sha, 'Historical driver was not audited')
    require(metadata.get('numpy') == '2.5.4', 'Historical NumPy differs')
    require(metadata.get('factor_dtype') == 'float64' and
            metadata.get('projector') == 'ASTRA CPU line_fanflat FP32', 'Historical operator backend differs')
    method = metadata['config']['method']
    if method == 'joint_lbfgsb':
        require(metadata.get('scipy') == '1.18.1', 'Historical SciPy differs')
        names = ('ct_problem.py','joint_lbfgsb_solver.py','nonnegative_solvers.py')
    else:
        require(metadata.get('astra') == '2.5.0', 'Historical ASTRA differs')
        names = ('ct_problem.py','methods.py','spectral_reference_core.py','spectral_reference_baseline.py')
        if method == 'spring_sarah':
            names += ('local_curvature.py','curvature_reference.py')
    targets = {}
    for name in names:
        target = SOURCE/name if method == 'joint_lbfgsb' or name == 'ct_problem.py' else SOURCE/'robustness_reference'/name
        old = Path(sources[name]['path'])
        require(digest(old) == digest(target) or _semantic(old) == _semantic(target),
                'Native numerical source is not equivalent: '+name)
        targets[str(target)] = digest(target)
    return targets


def _eligible(stage_name, metadata):
    c = metadata['config']
    if c.get('rank') != 64 or c.get('grid') != 512 or c.get('n_train') not in (360,1080):
        return False
    if c.get('record_every_iteration') is not True:
        return False
    return ((stage_name == 'joint_lbfgsb' and c['method'] == 'joint_lbfgsb') or
        (c['phase'] == 'evaluation' and (
            (stage_name in ('convincing_4x_recovery','feasibility_retune') and c['method'] == 'sframes') or
            (stage_name == 'baseline_step_rules' and c['method'] == 'prox_sarah') or
            (stage_name == 'gradient_budget' and c['method'] == 'spring_sarah'))))


def build_catalog(root, history=DEFAULT_HISTORY):
    """Pin the audited completed runs before the new campaign begins."""
    root, history = Path(root), Path(history)
    path = root/CATALOG
    if path.exists():
        catalog = read(path)
        require(catalog['adapter_sha256'] == digest(__file__), 'Frozen reuse adapter changed')
        return catalog
    require(not any((root/'runs').glob('*/result.json')), 'Freeze historical reuse before new results exist')
    entries, targets, skipped = [], {}, []
    for stage_name in STAGES:
        stage = history/stage_name
        for folder in sorted((stage/'runs').glob('*')):
            if not (folder/'result.json').is_file():
                continue
            meta, result, trace = (read(folder/name) for name in ('metadata.json','result.json','trace.json'))
            if result.get('status') != 'completed' or not _eligible(stage_name,meta):
                continue
            driver, driver_sha = DRIVERS[stage_name]
            if meta['source_sha256'].get(driver) != driver_sha:
                skipped.append(dict(origin_directory=str(folder),reason='Historical driver version has not been audited'))
                continue
            c = meta['config']
            cap = c.get('view_budget',c.get('gradient_view_budget'))
            require(cap == (34560 if c['phase'] == 'development' else 69120), 'Unexpected old budget')
            require(result.get('metadata', {key:result.get(key) for key in meta}) == meta,
                    'Historical result metadata disagrees')
            require(result.get('final') == trace[-1], 'Historical endpoint disagrees')
            _, excluded = _normalize(trace,meta,cap)
            sources = _resolve_sources(stage,meta)
            targets.update(_equivalence(stage_name,meta,sources))
            n = c['n_train']
            expected = [i for i in range(3600) if i%10 == 0 or (n == 1080 and i%10 in (3,6))]
            require(meta['train_view_ids'] == expected and meta['validation_view_ids'] == list(range(9,3600,10)),
                    'Historical view split differs')
            entries.append(dict(signature=_signature(c,meta), stage=stage_name,
                origin_directory=str(folder.resolve()), original_sources=sources,
                artifacts={name:digest(folder/name) for name in ('metadata.json','result.json','trace.json')},
                excluded_overshoot_rows=excluded))
    catalog = dict(schema_version=1,adapter_sha256=digest(__file__),entries=entries,skipped=skipped,
        target_native_sources=targets,
        scope='Dense standard rank64 radius2 cases only. Sparse development traces are never reused.',
        limitations='Historical diagnostic work is unavailable; it is not replaced by zero. Legacy L-BFGS-B records NumPy/SciPy and operator backend but not ASTRA version; legacy non-L-BFGS-B records NumPy/ASTRA but not SciPy. Missing version fields remain absent.')
    write(path,catalog)
    return catalog


def _catalog(root, protocol):
    path = Path(root)/CATALOG
    if not path.exists():
        return None
    catalog = read(path)
    require(catalog['adapter_sha256'] == digest(__file__), 'Frozen reuse adapter changed')
    if protocol is not None:
        frozen = protocol.get('frozen_files',{})
        expected = frozen.get(CATALOG)
        if isinstance(expected,dict):
            expected = expected.get('sha256')
        if expected is not None:
            require(digest(path) == expected, 'Frozen historical catalog changed')
    for path, sha in catalog['target_native_sources'].items():
        require(digest(path) == sha, 'Target native numerical source changed')
    return catalog


def _entry(config, catalog):
    if (config.get('initialization','standard') != 'standard' or config.get('rank') != 64
            or config.get('radius_scale',2.) != 2. or config.get('record_stationarity',False)
            or config.get('record_every_iteration') is not True or config.get('n_train') not in (360,1080)
            or config.get('method') not in ('sframes','prox_sarah','spring_sarah','joint_lbfgsb')):
        return None
    signature = _signature(config)
    matches = [entry for entry in catalog['entries'] if entry['signature'] == signature]
    require(len(matches) <= 1, 'Historical signature is ambiguous')
    return matches[0] if matches else None


def _verify_origin(entry):
    folder = Path(entry['origin_directory'])
    for name, sha in entry['artifacts'].items():
        require(digest(folder/name) == sha, 'Original historical evidence changed: '+name)
    for record in entry['original_sources'].values():
        require(digest(record['path']) == record['sha256'], 'Original executed numerical source changed')


def _material(config,cpu,root,entry):
    root = Path(root)
    _verify_origin(entry)
    for field in ('data','calibration'):
        require(digest(config[field+'_file']) == config[field+'_sha256'], 'Reuse input bytes changed')
    old = read(Path(entry['origin_directory'])/'metadata.json')
    original_trace = read(Path(entry['origin_directory'])/'trace.json')
    trace, excluded = _normalize(original_trace,old,config['view_budget'])
    require(excluded == entry['excluded_overshoot_rows'], 'Historical prefix changed')
    provenance = dict(catalog_sha256=digest(root/CATALOG), adapter_sha256=digest(__file__),
        origin_directory=entry['origin_directory'], original_artifact_sha256=entry['artifacts'],
        original_cpu=old.get('cpu'),
        original_executed_source_sha256=old['source_sha256'], excluded_overshoot_rows=excluded,
        numerical_engine_executed=False, checkpoint_available=False,
        missing_diagnostic_costs=True,
        missing_factor_hashes=any('factor_u_sha256' not in row or 'factor_v_sha256' not in row for row in trace),
        unknown_dependency_versions=[key for key in ('numpy','scipy','astra') if old.get(key) is None],
        normalization='Only counter aliases, event names, relative distance, and the exact observed in-budget prefix. No interpolated observations or fabricated diagnostic counts.')
    meta = deepcopy(old)
    meta.update(config=deepcopy(config),cpu=cpu,cpu_role='materialization_worker',
        producer='historical_reuse',historical_reuse=provenance,
        protocol_sha256=digest(root/'protocol.json'),
        reference_spectral_caps=old.get('reference_spectral_caps',old.get('caps')),
        spectral_caps=old.get('reference_spectral_caps',old.get('caps')),
        spectral_caps_are_constraints=config['method'] != 'joint_lbfgsb')
    # The old executed source map and dependency versions are deliberately kept.
    result = dict(status='completed',termination='historical_dense_prefix' if excluded else 'historical_completed',
        metadata=meta,iteration=trace[-1]['iteration'],observations=len(trace),initial=trace[0],final=trace[-1])
    return meta,result,trace


def try_materialize(config,cpu,root,protocol=None):
    """Return False for a nonmatching case; otherwise write a verified import."""
    root = Path(root)
    catalog = _catalog(root,protocol)
    if catalog is None:
        return False
    entry = _entry(config,catalog)
    if entry is None:
        return False
    folder = root/'runs'/config['run_id']
    if (folder/'result.json').exists():
        return verify_materialized(config,cpu,root,protocol)
    require(not (folder/'checkpoint.pkl').exists(), 'Cannot replace an existing optimizer checkpoint')
    meta,result,trace = _material(config,cpu,root,entry)
    evidence = folder/'historical_origin'
    evidence.mkdir(parents=True,exist_ok=True)
    for name in entry['artifacts']:
        data = (Path(entry['origin_directory'])/name).read_bytes()
        if (evidence/name).exists():
            require((evidence/name).read_bytes() == data, 'Retained historical evidence changed')
        else:
            (evidence/name).write_bytes(data)
    write(folder/'metadata.json',meta)
    write(folder/'trace.json',trace)
    result['trace_sha256'] = digest(folder/'trace.json')
    write(folder/'result.json',result)
    return True


def verify_materialized(config,cpu,root,protocol=None):
    """Rebuild expected metadata/trace from pinned evidence and compare exactly."""
    root = Path(root)
    catalog = _catalog(root,protocol)
    require(catalog is not None, 'Historical catalog is missing')
    entry = _entry(config,catalog)
    require(entry is not None, 'Imported configuration is not in the frozen catalog')
    folder = root/'runs'/config['run_id']
    meta,result,trace = _material(config,cpu,root,entry)
    for name,sha in entry['artifacts'].items():
        require(digest(folder/'historical_origin'/name) == sha, 'Copied historical evidence changed')
    require(read(folder/'metadata.json') == meta, 'Imported metadata changed')
    require(read(folder/'trace.json') == trace, 'Imported trajectory changed')
    result['trace_sha256'] = digest(folder/'trace.json')
    require(read(folder/'result.json') == result, 'Imported result changed')
    return True


# Descriptive alias for callers using the originally proposed API name.
materialize = try_materialize
