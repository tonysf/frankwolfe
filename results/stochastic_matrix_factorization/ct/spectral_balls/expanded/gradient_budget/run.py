"""Continue frozen PALM and SPRING runs to a gradient-only view budget.

The source checkpoint's complete native trace, including its final update past
the former inclusive-work cap, is retained exactly. Only the stopping rule and
explicit continuation provenance change; optimizer parameters and RNG states
come from that checkpoint. Forward/adjoint totals continue to include curvature.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'

from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import pickle

import astra
import numpy as np

from ct_problem import CTProblem
from methods import op_norm, repair_factor
import palm_paper
import local_curvature


ROOT = Path(__file__).resolve().parent
COUNTS = ('forward_views', 'backward_views', 'gradient_calls', 'oracle_calls',
          'curvature_operator_calls', 'curvature_forward_views', 'curvature_backward_views',
          'projection_calls', 'projection_inner_iterations', 'projection_dykstra_calls',
          'projection_floor_count')
SOURCES = ('run.py', 'palm_paper.py', 'local_curvature.py', 'curvature_reference.py',
           'ct_problem.py', 'methods.py', 'spectral_reference_core.py', 'spectral_reference_baseline.py')
ORIGIN_FILES = ('metadata.json', 'trace.json', 'checkpoint.pkl', 'result.json')
COMMON_CONFIG = ('method', 'slice_id', 'n_train', 'seed', 'grid', 'rank', 'calibration_seed')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def write_json(path, value):
    atomic_write(path, (json.dumps(value, default=json_scalar, allow_nan=False, indent=2) + '\n').encode())


def gradient_views(row, direction='forward'):
    if direction not in ('forward', 'backward'):
        raise ValueError('Expected forward or backward views')
    value = row[f'{direction}_views'] - row[f'curvature_{direction}_views']
    if value < 0 or row.get(f'gradient_{direction}_views', value) != value:
        raise ValueError('Gradient and curvature view counters disagree')
    return value


def split_work(counters):
    return dict(gradient_forward_views=gradient_views(counters),
                gradient_backward_views=gradient_views(counters, 'backward'))


def observe(problem, U, V, caps, counters, iteration, last_step):
    # Exact reference diagnostic formulas, with two derived work columns.
    fixed_u, fixed_v = (repair_factor(matrix, cap) for matrix, cap in zip((U, V), caps))
    du = float(np.linalg.norm(U - fixed_u))
    dv = float(np.linalg.norm(V - fixed_v))
    image = fixed_u @ fixed_v.T
    repaired = problem.image_loss(image, problem.train_ids)
    raw = repaired if du + dv == 0 else problem.image_loss(U @ V.T, problem.train_ids)
    c_violation = max(op_norm(U) - caps[0], op_norm(V) - caps[1], 0.0)
    d_distance = float(np.hypot(np.linalg.norm(np.minimum(U, 0)), np.linalg.norm(np.minimum(V, 0))))
    violation = max(c_violation, float(-U.min()), float(-V.min()))
    reference = problem.reference
    error = None if reference is None else float(np.linalg.norm(image - reference) / np.linalg.norm(reference))
    result = dict(iteration=iteration, **counters, **split_work(counters), raw_loss=raw,
        feasible_loss=repaired, validation_loss=problem.image_loss(image, problem.validation_ids),
        reference_relative_error=error, max_constraint_violation=violation,
        repair_distance=float(np.hypot(du, dv)), c_violation=c_violation,
        d_distance=d_distance, last_step=last_step)
    if any(not np.isfinite(result[k]) for k in
           ('raw_loss', 'feasible_loss', 'validation_loss', 'd_distance', 'repair_distance')):
        raise FloatingPointError('nonfinite optimizer diagnostic')
    return result


def check_abort():
    path = ROOT / 'execution_abort.json'
    if path.exists() and read_json(path).get('active'):
        raise RuntimeError('Campaign halted; retained checkpoints are resumable after review')


def origin(config):
    """Read immutable source bytes and verify their complete checkpoint prefix."""
    directory = Path(config['origin_directory'])
    raw = {}
    for filename in ORIGIN_FILES:
        content = (directory / filename).read_bytes()
        field = f'origin_{Path(filename).stem}_sha256'
        if hashlib.sha256(content).hexdigest() != config[field]:
            raise ValueError(f'Continuation source changed: {filename}')
        raw[filename] = content
    meta, trace, result = (json.loads(raw[name]) for name in ('metadata.json', 'trace.json', 'result.json'))
    # Pickles are loaded only after matching the prepared local input digest.
    state = pickle.loads(raw['checkpoint.pkl'])
    if (meta['config'] != config['origin_config']
            or meta['config']['run_id'] != config['origin_run_id']
            or any(meta['config'][name] != config[name] for name in COMMON_CONFIG)
            or result['status'] != 'completed' or state['metadata'] != meta
            or any(result.get(k) != v for k, v in meta.items())
            or state['trace'] != trace or not trace or result['final'] != trace[-1]
            or result['iteration'] != state['solver']['k']
            or trace[-1]['iteration'] != state['solver']['k']
            or [row['iteration'] for row in trace] != list(range(len(trace)))
            or state['last_step'] != trace[-1]['last_step']
            or any(trace[-1][k] != v for k, v in state['counters'].items())):
        raise ValueError('Continuation checkpoint, trace, configuration or result disagree')
    if gradient_views(state['counters']) >= config['gradient_view_budget']:
        raise ValueError('The requested extension budget is already reached by the source checkpoint')
    return meta, state


def run(config, cpu=None, runs_dir=None):
    check_abort()
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    folder = (ROOT / 'runs' if runs_dir is None else Path(runs_dir)) / config['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, cpu, folder)


def _run(config, cpu, folder):
    if (config.get('phase') != 'evaluation' or config.get('method') not in ('palm', 'spring_sarah')
            or config.get('record_every_iteration') is not True
            or type(config.get('gradient_view_budget')) is not int or config['gradient_view_budget'] <= 0
            or 'view_budget' in config):
        raise ValueError('This engine requires dense, explicitly gradient-budgeted continuations')
    source = {name: sha256(ROOT / name) for name in SOURCES}
    manifest = read_json(ROOT / 'implementation_manifest.json')
    if source != manifest['current_source_sha256']:
        raise ValueError('Continuation implementation differs from frozen sources')
    if sha256(ROOT / 'specification.json') != manifest['specification_sha256']:
        raise ValueError('Continuation specification changed')
    old, initial_state = origin(config)
    for name in set(source).intersection(old['source_sha256']) - {'run.py'}:
        if source[name] != old['source_sha256'][name]:
            raise ValueError(f'A numerical source differs from the original run: {name}')
    data_file = ROOT / 'data' / f'{config["slice_id"]}.npz'
    calibration = ROOT / 'calibration' / f'{config["slice_id"]}_seed{config["calibration_seed"]}.npz'
    if sha256(data_file) != old['data_sha256'] or sha256(calibration) != old['calibration_sha256']:
        raise ValueError('Continuation data or common initialization changed')
    problem = CTProblem(data_file, config['grid'], config['n_train'])
    try:
        with np.load(calibration, allow_pickle=False) as saved:
            if str(saved['data_sha256']) != old['data_sha256'] or str(saved['problem_sha256']) != source['ct_problem.py']:
                raise ValueError('Calibration differs from the frozen problem')
            U, V = saved['U'], saved['V']
            caps, curvature = tuple(saved['caps'].tolist()), tuple(saved['curvature'].tolist())
        if (list(caps) != old['caps'] or list(curvature) != old['curvature']
                or problem.train_ids.tolist() != old['train_view_ids']
                or problem.validation_ids.tolist() != old['validation_view_ids']
                or np.__version__ != old['numpy'] or astra.__version__ != old['astra']):
            raise ValueError('Continuation problem or numerical environment changed')
        params = old['optimizer']
        factory = palm_paper if config['method'] == 'palm' else local_curvature
        factory_config = old.get('factory_config', params)
        solver = factory.make_solver(problem, U, V, caps, seed=config['seed'] + 100_000, config=factory_config)
        checkpoint_info = dict(source_directory=str(Path(config['origin_directory']).resolve()),
            run_id=config['origin_run_id'], iteration=initial_state['solver']['k'],
            gradient_forward_views=gradient_views(initial_state['counters']),
            gradient_backward_views=gradient_views(initial_state['counters'], 'backward'),
            source_metadata_sha256=config['origin_metadata_sha256'],
            source_trace_sha256=config['origin_trace_sha256'],
            source_checkpoint_sha256=config['origin_checkpoint_sha256'],
            source_result_sha256=config['origin_result_sha256'],
            source_code_sha256=old['source_sha256'], prefix_observations=len(initial_state['trace']),
            optimizer_and_rng_state_restored=True, source_trace_prefix_preserved=True,
            includes_source_final_update_past_inclusive_view_cap=True)
        metadata = deepcopy(old)
        metadata.update(config=config, source_sha256=source, cpu=cpu,
            implementation_manifest_sha256=sha256(ROOT / 'implementation_manifest.json'),
            specification_sha256=sha256(ROOT / 'specification.json'),
            authorized_implementation_changes=manifest['authorized_changes'],
            continuation=checkpoint_info,
            imported_checkpoint=str(Path(config['origin_directory']).resolve() / 'checkpoint.pkl'),
            recovery_origin='continued_from_completed_checkpoint',
            budget=dict(metric='gradient_views', per_direction=config['gradient_view_budget'],
                        definition='total_views - curvature_views', cumulative_from_initialization=True),
            diagnostics=dict(cadence='every_completed_iteration', includes_iteration_zero=True,
                optimizer_work_counters_only=True, curvature_work_recorded_separately=True,
                total_view_counters_include_curvature=True,
                oracle_calls_definition='gradient_calls + curvature_operator_calls',
                gradient_view_columns='Explicit in new observations; total minus curvature in the unchanged source prefix.'))
        metadata = json.loads(json.dumps(metadata, default=json_scalar, allow_nan=False))
        final_path = folder / 'result.json'
        if final_path.exists():
            result = read_json(final_path)
            if any(result.get(key) != value for key, value in metadata.items()):
                raise ValueError('Completed continuation provenance changed')
            return result['final']
        checkpoint = folder / 'checkpoint.pkl'
        state = pickle.loads(checkpoint.read_bytes()) if checkpoint.exists() else initial_state
        if checkpoint.exists() and state['metadata'] != metadata:
            raise ValueError('Continuation checkpoint provenance changed')
        if state['trace'][:len(initial_state['trace'])] != initial_state['trace']:
            raise ValueError('The original dense trace prefix changed')
        solver.load_state_dict(state['solver'])
        counters, trace, last_step = state['counters'], state['trace'], state['last_step']
        write_json(folder / 'metadata.json', metadata)

        def save(status):
            if trace[-1]['iteration'] != solver.k:
                trace.append(observe(problem, solver.U, solver.V, caps, counters, solver.k, last_step))
            atomic_write(checkpoint, pickle.dumps(dict(metadata=metadata, solver=solver.state_dict(),
                counters=counters, trace=trace, last_step=last_step), protocol=5))
            write_json(folder / 'trace.json', trace)
            write_json(folder / 'progress.json', dict(status=status, iteration=solver.k, **counters, **split_work(counters)))

        save('running')
        try:
            while gradient_views(counters) < config['gradient_view_budget']:
                check_abort()
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    step = solver.step()
                if (step['oracle_calls'] != step['gradient_calls'] + step['curvature_operator_calls']
                        or step['forward_views'] != step['backward_views']
                        or gradient_views(step) != gradient_views(step, 'backward')
                        or step['component_gradient_calls'] != gradient_views(step)):
                    raise ValueError('Gradient and curvature work accounting is inconsistent')
                for key in COUNTS:
                    counters[key] += step[key]
                counters['max_projection_distance_bound'] = max(counters['max_projection_distance_bound'],
                                                                step['projection_max_distance_bound'])
                last_step = json.loads(json.dumps(step, default=json_scalar, allow_nan=False))
                save('running')
            save('completed')
            write_json(final_path, dict(metadata, status='completed', iteration=solver.k, final=trace[-1]))
        except Exception as exc:
            write_json(folder / 'progress.json', dict(status='failed', iteration=solver.k,
                error=repr(exc), error_type=type(exc).__name__, **counters, **split_work(counters)))
            raise
        return trace[-1]
    finally:
        problem.close()
