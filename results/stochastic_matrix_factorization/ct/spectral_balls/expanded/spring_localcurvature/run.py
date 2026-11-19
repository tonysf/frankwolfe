"""Fresh, resumable SPRING evaluation with explicitly charged local-curvature work.

The problem, spectral projection and diagnostic formulas are unchanged from
convincing_4x_recovery. The SPRING factory and this recording engine are new,
declared sources; they are never presented as the historical optimizer code.
"""
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'

import fcntl
import hashlib
import json
from pathlib import Path
import pickle

import astra
import numpy as np

from ct_problem import CTProblem
from methods import op_norm, repair_factor
from local_curvature import make_solver


ROOT = Path(__file__).resolve().parent
COUNTS = ('forward_views', 'backward_views', 'gradient_calls', 'oracle_calls',
          'curvature_operator_calls', 'curvature_forward_views', 'curvature_backward_views',
          'projection_calls', 'projection_inner_iterations', 'projection_dykstra_calls',
          'projection_floor_count')
SOURCES = ('run.py', 'local_curvature.py', 'ct_problem.py', 'methods.py',
           'spectral_reference_core.py', 'spectral_reference_baseline.py')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_write(path, content):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(content)
    temp.replace(path)


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def write_json(path, value):
    atomic_write(path, (json.dumps(value, default=json_scalar, allow_nan=False, indent=2)+'\n').encode())


def observe(problem, U, V, caps, counters, iteration, last_step):
    # Keep exactly the reference diagnostic definitions, adding only work fields
    # and the local-curvature step information supplied by the optimizer.
    fixed_u, fixed_v = (repair_factor(matrix, cap) for matrix, cap in zip((U, V), caps))
    du = float(np.linalg.norm(U-fixed_u))
    dv = float(np.linalg.norm(V-fixed_v))
    image = fixed_u @ fixed_v.T
    repaired = problem.image_loss(image, problem.train_ids)
    raw = repaired if du+dv == 0 else problem.image_loss(U @ V.T, problem.train_ids)
    c_violation = max(op_norm(U)-caps[0], op_norm(V)-caps[1], 0.0)
    d_distance = float(np.hypot(np.linalg.norm(np.minimum(U, 0)), np.linalg.norm(np.minimum(V, 0))))
    violation = max(c_violation, float(-U.min()), float(-V.min()))
    reference = problem.reference
    error = None if reference is None else float(np.linalg.norm(image-reference)/np.linalg.norm(reference))
    result = dict(iteration=iteration, **counters, raw_loss=raw, feasible_loss=repaired,
                  validation_loss=problem.image_loss(image, problem.validation_ids),
                  reference_relative_error=error, max_constraint_violation=violation,
                  repair_distance=float(np.hypot(du, dv)), c_violation=c_violation,
                  d_distance=d_distance, last_step=last_step)
    if any(not np.isfinite(result[k]) for k in
           ('raw_loss', 'feasible_loss', 'validation_loss', 'd_distance', 'repair_distance')):
        raise FloatingPointError('nonfinite optimizer diagnostic')
    return result


def check_abort():
    path = ROOT/'execution_abort.json'
    if path.exists() and read_json(path).get('active'):
        raise RuntimeError('Campaign halted; retained checkpoints are resumable after review')


def run(config, cpu=None, runs_dir=None):
    check_abort()
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    folder = (ROOT/'runs' if runs_dir is None else Path(runs_dir))/config['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(config, cpu, folder)


def _run(config, cpu, folder):
    if (config.get('phase') != 'evaluation' or config.get('method') != 'spring_sarah'
            or config.get('record_every_iteration') is not True
            or any('continuation' in k or 'checkpoint_import' in k for k in config)):
        raise ValueError('This engine only runs fresh, dense SPRING configurations')
    source = {name:sha256(ROOT/name) for name in SOURCES}
    manifest = read_json(ROOT/'implementation_manifest.json')
    if source != manifest['current_source_sha256']:
        raise ValueError('Local-curvature implementation differs from frozen sources')
    if sha256(ROOT/'specification.json') != manifest['specification_sha256']:
        raise ValueError('SPRING specification changed')
    data_file = ROOT/'data'/f'{config["slice_id"]}.npz'
    calibration = ROOT/'calibration'/f'{config["slice_id"]}_seed{config["calibration_seed"]}.npz'
    problem = CTProblem(data_file, config['grid'], config['n_train'])
    try:
        data_hash = sha256(data_file)
        with np.load(calibration, allow_pickle=False) as saved:
            if str(saved['data_sha256']) != data_hash or str(saved['problem_sha256']) != source['ct_problem.py']:
                raise ValueError('Calibration differs from the frozen problem')
            U, V = saved['U'], saved['V']
            caps, curvature = tuple(saved['caps'].tolist()), tuple(saved['curvature'].tolist())
        params = config['optimizer']
        solver = make_solver(problem, U, V, caps, seed=config['seed']+100_000, config=params)
        metadata = dict(config=config, optimizer=params, caps=caps, curvature=curvature,
            initial_factor_frobenius_norm=float(np.hypot(np.linalg.norm(U),np.linalg.norm(V))),
            initial_factor_norm=float(np.hypot(np.linalg.norm(U),np.linalg.norm(V))),
            source_sha256=source, data_sha256=data_hash, calibration_sha256=sha256(calibration),
            implementation_manifest_sha256=sha256(ROOT/'implementation_manifest.json'),
            specification_sha256=sha256(ROOT/'specification.json'),
            authorized_implementation_changes=manifest['authorized_changes'],
            projector='ASTRA CPU line_fanflat FP32', factor_dtype='float64',
            numpy=np.__version__, astra=astra.__version__, cpu=cpu,
            train_view_ids=problem.train_ids.tolist(), validation_view_ids=problem.validation_ids.tolist(),
            initialization=dict(name='warm',seed=config['calibration_seed']),
            diagnostics=dict(cadence='every_completed_iteration',includes_iteration_zero=True,
                optimizer_work_counters_only=True,curvature_work_included=True,
                oracle_calls_definition='gradient_calls + curvature_operator_calls'),
            constraint_geometry='separate spectral norm balls and entrywise nonnegativity',
            continuation=None, imported_checkpoint=None, recovery_origin='fresh')
        metadata = json.loads(json.dumps(metadata,default=json_scalar))
        final_path = folder/'result.json'
        if final_path.exists():
            result = read_json(final_path)
            if any(result.get(key) != value for key,value in metadata.items()):
                raise ValueError('Completed run provenance changed')
            return result['final']
        counters = {key:0 for key in COUNTS}
        counters.update(max_projection_distance_bound=0.0,max_lmo_support_gap_bound=0.0)
        trace, last_step = [], None
        checkpoint = folder/'checkpoint.pkl'
        if checkpoint.exists():
            state = pickle.loads(checkpoint.read_bytes())
            if state['metadata'] != metadata:
                raise ValueError('Checkpoint provenance changed')
            solver.load_state_dict(state['solver'])
            counters,trace,last_step = state['counters'],state['trace'],state['last_step']
        write_json(folder/'metadata.json',metadata)

        def save(status):
            if not trace or trace[-1]['iteration'] != solver.k:
                trace.append(observe(problem,solver.U,solver.V,caps,counters,solver.k,last_step))
            atomic_write(checkpoint,pickle.dumps(dict(metadata=metadata,solver=solver.state_dict(),
                counters=counters,trace=trace,last_step=last_step),protocol=5))
            write_json(folder/'trace.json',trace)
            write_json(folder/'progress.json',dict(status=status,iteration=solver.k,**counters))

        save('running')
        try:
            while counters['forward_views'] < config['view_budget']:
                check_abort()
                with np.errstate(over='raise',invalid='raise',divide='raise'):
                    step = solver.step()
                if step['oracle_calls'] != step['gradient_calls']+step['curvature_operator_calls']:
                    raise ValueError('Step oracle accounting is inconsistent')
                if step['forward_views'] != step['backward_views']:
                    raise ValueError('Step forward/adjoint accounting is inconsistent')
                for key in COUNTS:
                    if key not in step:
                        raise ValueError(f'Missing charged optimizer counter: {key}')
                    counters[key] += step[key]
                counters['max_projection_distance_bound'] = max(counters['max_projection_distance_bound'],
                                                                step['projection_max_distance_bound'])
                last_step = json.loads(json.dumps(step,default=json_scalar,allow_nan=False))
                save('running')
            save('completed')
            write_json(final_path,dict(metadata,status='completed',iteration=solver.k,final=trace[-1]))
        except Exception as exc:
            write_json(folder/'progress.json',dict(status='failed',iteration=solver.k,
                error=repr(exc),error_type=type(exc).__name__,**counters))
            raise
        return trace[-1]
    finally:
        problem.close()
