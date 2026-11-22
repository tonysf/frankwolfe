"""Dense, resumable CT runs with frozen inputs and explicit diagnostic work.

Native iterations are accepted factor updates.  A terminal observation may
repeat the preceding iteration while accounting for rejected L-BFGS-B trials.
No optimizer timings are measured.  Full-gradient gap diagnostics concern only
the smoothing methods and use beta_k at the recorded state x_k.
"""

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
import traceback

import numpy as np
import scipy

try:
    from .robustness_solvers import make_solver, METHODS, SMOOTHING, NumericalFailure
except ImportError:
    from robustness_solvers import make_solver, METHODS, SMOOTHING, NumericalFailure

SOURCE_ROOT = Path(__file__).resolve().parent
SOURCES = ('robustness_run.py', 'robustness_solvers.py', 'robustness_inputs.py',
           'ct_problem.py', 'joint_lbfgsb_solver.py', 'nonnegative_solvers.py')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_hashes():
    names = (*SOURCES, *(str(p.relative_to(SOURCE_ROOT))
                         for p in sorted((SOURCE_ROOT / 'robustness_reference').glob('*.py'))))
    return {name: sha256(SOURCE_ROOT / name) for name in names}


def read(path):
    return json.loads(Path(path).read_text())


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def plain(value):
    return json.loads(json.dumps(value, default=json_scalar, allow_nan=False))


def atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(value)
    temporary.replace(path)


def write(path, value):
    atomic_bytes(path, (json.dumps(plain(value), indent=2, allow_nan=False) + '\n').encode())


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def make_problem(config):
    try:
        from .robustness_inputs import NewCTProblem
    except ImportError:
        from robustness_inputs import NewCTProblem
    return NewCTProblem(config['data_file'], config['grid'], config['n_train'])


def initial_diagnostics():
    return dict(forward_views=0, backward_views=0, forward_calls=0, backward_calls=0,
                full_gradient_calls=0, gap_sum=0., gap_observations=0)


def observe(problem, solver, initial_norm, diagnostics, event, record_stationarity=False):
    """Observe a new accepted state, charging diagnostic operators separately."""
    U, V = solver.U, solver.V
    if not np.isfinite(U).all() or not np.isfinite(V).all():
        raise NumericalFailure('Nonfinite accepted factors')
    positive = [np.maximum(factor, 0.) for factor in (U, V)]
    distance = float(np.hypot(np.linalg.norm(U - positive[0]), np.linalg.norm(V - positive[1])))
    fixed = [p.copy() for p in positive]
    if solver.spectral_constrained:
        for p, cap in zip(fixed, solver.caps):
            norm = float(np.linalg.norm(p, 2))
            if norm > cap:
                p *= cap / norm
    raw_image, image = U @ V.T, fixed[0] @ fixed[1].T

    def loss(value, ids):
        diagnostics['forward_views'] += len(ids)
        diagnostics['forward_calls'] += 1
        return problem.image_loss(value, ids)

    raw_loss = loss(raw_image, problem.train_ids)
    repair_distance = float(np.hypot(np.linalg.norm(U - fixed[0]), np.linalg.norm(V - fixed[1])))
    feasible_loss = raw_loss if repair_distance == 0 else loss(image, problem.train_ids)
    validation_loss = loss(image, problem.validation_ids)
    reference = problem.reference
    error = None if reference is None else float(np.linalg.norm(image - reference) / np.linalg.norm(reference))
    norms = [float(np.linalg.norm(factor, 2)) for factor in (U, V)]
    c_distance = max(0., *(norm-cap for norm, cap in zip(norms, solver.caps)))
    row = dict(iteration=solver.iteration, observation_type=event['observation_type'],
        **solver.counters, completed_blocks=solver.completed_blocks, outer_sweeps=solver.outer_sweeps,
        processed_views=max(solver.counters['gradient_forward_views'], solver.counters['gradient_backward_views']),
        total_processed_views=max(solver.counters['forward_views'], solver.counters['backward_views']),
        comparison_views=max(solver.counters['gradient_forward_views'], solver.counters['gradient_backward_views']),
        comparison_oracle_calls=solver.counters['gradient_calls'],
        raw_loss=raw_loss, feasible_loss=feasible_loss, validation_loss=validation_loss,
        reference_relative_error=error, d_distance=distance, relative_d_distance=distance/initial_norm,
        repair_distance=repair_distance, c_violation=c_distance,
        max_constraint_violation=max(distance, c_distance) if solver.spectral_constrained else distance,
        spectral_norm_u=norms[0], spectral_norm_v=norms[1],
        reference_spectral_ratio_u=norms[0]/solver.caps[0],
        reference_spectral_ratio_v=norms[1]/solver.caps[1],
        factor_u_sha256=array_hash(U), factor_v_sha256=array_hash(V), last_step=plain(event))
    if record_stationarity:
        if solver.method not in SMOOTHING:
            raise ValueError('The smoothed gap is only a diagnostic for smoothing algorithms')
        n = len(problem.train_ids)
        diagnostics['forward_views'] += n
        diagnostics['backward_views'] += n
        diagnostics['forward_calls'] += 1
        diagnostics['backward_calls'] += 1
        diagnostics['full_gradient_calls'] += 1
        gu, gv = problem.grad(U, V, problem.train_ids)
        beta = solver.beta()
        hu, hv = gu + np.minimum(U, 0.)/beta, gv + np.minimum(V, 0.)/beta
        support = solver.caps[0]*np.linalg.svd(hu, compute_uv=False).sum()
        support += solver.caps[1]*np.linalg.svd(hv, compute_uv=False).sum()
        inner = float(np.sum(hu*U) + np.sum(hv*V))
        gap = float(inner + support)
        if gap < -1e-10*max(1., abs(inner), float(support)):
            raise ValueError('Smoothed FW gap is negative beyond floating-point tolerance')
        gap = max(0., gap)
        diagnostics['gap_sum'] += gap
        diagnostics['gap_observations'] += 1
        row.update(stationarity_beta=beta, smoothed_fw_gap=gap,
            mean_smoothed_fw_gap=diagnostics['gap_sum']/diagnostics['gap_observations'],
            stationarity_observations=diagnostics['gap_observations'])
    row.update({f'diagnostic_{key}': int(diagnostics[key]) for key in
                ('forward_views', 'backward_views', 'forward_calls', 'backward_calls', 'full_gradient_calls')})
    for key, value in row.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise NumericalFailure('Nonfinite observation: ' + key)
    if solver.spectral_constrained and c_distance > 1e-9*max(solver.caps):
        raise ValueError('Accepted state exceeds the spectral domain')
    if solver.method not in SMOOTHING and distance != 0:
        raise ValueError('A nonnegative baseline produced an infeasible accepted state')
    return plain(row)


def terminal_observation(solver, previous, event):
    row = deepcopy(previous)
    row.update(solver.counters)
    row.update(iteration=solver.iteration, observation_type=event['observation_type'],
        processed_views=max(solver.counters['gradient_forward_views'], solver.counters['gradient_backward_views']),
        total_processed_views=max(solver.counters['forward_views'], solver.counters['backward_views']),
        comparison_views=max(solver.counters['gradient_forward_views'], solver.counters['gradient_backward_views']),
        comparison_oracle_calls=solver.counters['gradient_calls'], last_step=plain(event))
    return row


def check_abort(root):
    path = root / 'execution_abort.json'
    if path.exists() and read(path).get('active'):
        raise RuntimeError('Campaign halted; retained evidence requires review')


def run(config, cpu=None, root=None, output=None, problem_factory=None):
    if root is None:
        raise ValueError('An explicit campaign root is required')
    root = Path(root).resolve()
    folder = Path(output).resolve() if output is not None else root / 'runs' / config['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    if cpu is not None:
        os.sched_setaffinity(0, {int(cpu)})
    with (folder / 'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(plain(config), cpu, root, folder, problem_factory or make_problem)


def _run(config, cpu, root, folder, problem_factory):
    if (config['method'] not in METHODS or not config.get('phase')
            or config.get('record_every_iteration', True) is not True
            or isinstance(config['view_budget'], bool) or int(config['view_budget']) != config['view_budget']
            or config['view_budget'] <= 0):
        raise ValueError('Invalid robustness run configuration')
    sources = source_hashes()
    expected = read(root / 'protocol.json')['numerical_sources']
    if any(expected.get(name) != value for name, value in sources.items()):
        raise ValueError('Numerical code differs from frozen campaign sources')
    for field in ('data', 'calibration'):
        if sha256(config[field+'_file']) != config[field+'_sha256']:
            raise ValueError('Frozen input changed: ' + field)
    with np.load(config['calibration_file'], allow_pickle=False) as saved:
        if str(saved['data_sha256']) != config['data_sha256']:
            raise ValueError('Calibration data provenance differs')
        if str(saved['problem_sha256']) != sources['ct_problem.py']:
            raise ValueError('Calibration CT operator provenance differs')
        U, V = saved['U'].copy(), saved['V'].copy()
        reference_caps = tuple(float(x) for x in saved['caps'])
        curvature = tuple(float(x) for x in saved['curvature'])
    radius_scale = float(config.get('radius_scale', 2.))
    if not np.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError('radius_scale must be finite and positive')
    caps = tuple(cap*radius_scale/2. for cap in reference_caps)
    if (U.shape != (config['grid'], config['rank']) or V.shape != U.shape
            or not np.isfinite(U).all() or not np.isfinite(V).all()
            or np.any(U < 0) or np.any(V < 0)):
        raise ValueError('Invalid common nonnegative initialization')
    record_gap = bool(config.get('record_stationarity', False))
    if record_gap and config['method'] not in SMOOTHING:
        raise ValueError('Requested smoothed-gap diagnostic for an incompatible method')
    initial_norm = float(np.hypot(np.linalg.norm(U), np.linalg.norm(V)))
    if initial_norm == 0:
        raise ValueError('The shared initialization must be nonzero')
    problem = problem_factory(config)
    try:
        solver = make_solver(config['method'], problem, U, V, caps,
            seed=config.get('seed', 0), curvatures=curvature, config=config['parameters'])
        metadata = plain(dict(config=config, optimizer=solver.config, source_sha256=sources,
            protocol_sha256=sha256(root/'protocol.json'), cpu=cpu,
            geometry='spectral' if solver.spectral_constrained else 'nonnegative',
            spectral_caps_are_constraints=solver.spectral_constrained, reference_spectral_caps=reference_caps,
            spectral_caps=caps, initial_curvature=curvature, initial_factor_frobenius_norm=initial_norm,
            initial_u_sha256=array_hash(U), initial_v_sha256=array_hash(V),
            data_sha256=config['data_sha256'], calibration_sha256=config['calibration_sha256'],
            train_view_ids=problem.train_ids.tolist(), validation_view_ids=problem.validation_ids.tolist(),
            numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
            astra=getattr(sys.modules.get('astra'), '__version__', None), factor_dtype='float64',
            projector='ASTRA CPU line_fanflat FP32', oracle_seed_rule='100000 + seed label',
            deterministic=config['method'] in ('frames', 'joint_lbfgsb') or
                (config['method'] in SMOOTHING and solver.config['batch'] == len(problem.train_ids)),
            diagnostics=dict(cadence='Every accepted update; terminal records retain the last accepted state',
                record_stationarity=record_gap, stationarity_index='At x_k use beta_0/(k+1)^q; cumulative mean includes x_0 once',
                repair='Orthant clipping plus radial spectral repair' if solver.spectral_constrained else 'Orthant only',
                optimizer_timing=False, costs='Diagnostic FP/BP recorded separately and excluded from solver work'),
            accounting='Gradient views capped in each direction; curvature work recorded separately and in total views'))
        result_file, checkpoint = folder/'result.json', folder/'checkpoint.pkl'
        if result_file.exists():
            result = read(result_file)
            if (result.get('metadata') != metadata or result['status'] not in ('completed', 'numerical_failure')
                    or result.get('trace_sha256') != sha256(folder/'trace.json')):
                raise ValueError('Completed run provenance or trace changed')
            return result
        diagnostics, trace, last_event = initial_diagnostics(), [], None
        if checkpoint.exists():
            receipt = read(folder/'checkpoint.sha256.json')
            if sha256(checkpoint) not in (receipt['sha256'], receipt.get('previous_sha256')):
                raise ValueError('Checkpoint does not match its digest receipt')
            state = pickle.loads(checkpoint.read_bytes())
            if state['metadata'] != metadata:
                raise ValueError('Checkpoint provenance changed')
            solver.load_state_dict(state['solver'])
            trace, diagnostics = state['trace'], state['diagnostics']
            last_event = state['last_event']
            if solver.iteration != trace[-1]['iteration']:
                if solver.iteration != trace[-1]['iteration']+1 or last_event['observation_type'] != 'accepted_update':
                    raise ValueError('Checkpoint cannot reconstruct its missing accepted observation')
                trace.append(observe(problem, solver, initial_norm, diagnostics, last_event, record_gap))
            elif (last_event['observation_type'] in ('budget_stop', 'converged')
                    and trace[-1]['observation_type'] not in ('budget_stop', 'converged')):
                trace.append(terminal_observation(solver, trace[-1], last_event))
        else:
            last_event = dict(observation_type='initial', iteration=0)
            trace = [observe(problem, solver, initial_norm, diagnostics, last_event, record_gap)]
        write(folder/'metadata.json', metadata)

        def save(status, termination=None):
            pending = checkpoint.with_suffix('.next.pkl')
            atomic_bytes(pending, pickle.dumps(dict(metadata=metadata, solver=solver.state_dict(),
                trace=trace, diagnostics=diagnostics, last_event=last_event), protocol=5))
            write(folder/'checkpoint.sha256.json', dict(sha256=sha256(pending),
                previous_sha256=sha256(checkpoint) if checkpoint.exists() else None))
            pending.replace(checkpoint)
            write(folder/'trace.json', trace)
            write(folder/'progress.json', dict(status=status, termination=termination,
                iteration=solver.iteration, observations=len(trace), **plain(solver.counters)))

        def finish(kind, status='completed', error=None):
            result = dict(status=status, termination=kind, metadata=metadata, iteration=solver.iteration,
                observations=len(trace), initial=trace[0], final=trace[-1], trace_sha256=sha256(folder/'trace.json'))
            if error is not None:
                result['error'] = repr(error)
            write(result_file, result)
            return result

        if trace[-1]['observation_type'] in ('budget_stop', 'converged'):
            save('completed', trace[-1]['observation_type'])
            return finish(trace[-1]['observation_type'])
        save('running')
        while True:
            check_abort(root)
            previous_solver, previous_diagnostics = solver.state_dict(), deepcopy(diagnostics)
            previous_event = deepcopy(last_event)
            update_returned = row_committed = False
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    event = plain(solver.step(view_budget=config['view_budget']))
                last_event = event
                update_returned = True
                if max(solver.counters['gradient_forward_views'], solver.counters['gradient_backward_views']) > config['view_budget']:
                    raise ValueError('Optimizer exceeded its directional gradient-view budget')
                kind = event['observation_type']
                if kind == 'stalled':
                    raise RuntimeError('Native solver stalled: ' + json.dumps(event))
                if kind not in ('accepted_update', 'budget_stop', 'converged'):
                    raise ValueError('Unexpected solver event: ' + kind)
                if kind == 'accepted_update':
                    # The diagnostic checkpoint is reconstructed after an I/O failure.
                    row = observe(problem, solver, initial_norm, diagnostics, event, record_gap)
                else:
                    row = terminal_observation(solver, trace[-1], event)
                trace.append(row)
                row_committed = True
                terminal = kind in ('budget_stop', 'converged')
                save('completed' if terminal else 'running', kind if terminal else None)
                if terminal:
                    return finish(kind)
            except NumericalFailure as exc:
                failed_state = dict(error=repr(exc), traceback=traceback.format_exc(),
                    attempted_iteration=solver.iteration, last_valid_iteration=trace[-1]['iteration'],
                    attempted_counters=plain(solver.counters), last_valid_counters=previous_solver['counters'])
                # Numerical candidates are ineligible, not successful early stops.
                # Preserve a compatible last-valid checkpoint rather than NaNs.
                solver.load_state_dict(previous_solver)
                diagnostics, last_event = previous_diagnostics, previous_event
                save('numerical_failure', 'nonfinite_arithmetic')
                write(folder/'failure.json', failed_state)
                return finish('nonfinite_arithmetic', status='numerical_failure', error=exc)
            except Exception as exc:
                # A legacy update is a transaction: failures before its return
                # replay from the original factors, momentum and RNG. L-BFGS-B
                # instead retains its fully resumable reverse-communication state.
                discarded_diagnostics = {key: diagnostics[key]-previous_diagnostics[key]
                    for key in ('forward_views', 'backward_views', 'forward_calls',
                                'backward_calls', 'full_gradient_calls')}
                attempted_counters = plain(solver.counters)
                native_step_counters = {key: int(getattr(solver.native, key, 0))
                    for key in ('forward_views', 'backward_views', 'gradient_calls',
                                'curvature_forward_views', 'curvature_backward_views')}
                if not row_committed:
                    diagnostics = previous_diagnostics
                    if not update_returned and solver.method != 'joint_lbfgsb':
                        solver.load_state_dict(previous_solver)
                        last_event = previous_event
                # An accepted update whose observation failed is retained, while
                # its partial diagnostic sum is discarded and recomputed once.
                save('failed', type(exc).__name__)
                write(folder/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                    iteration=solver.iteration, counters=plain(solver.counters),
                    attempted_counters=attempted_counters, native_step_counters=native_step_counters,
                    discarded_diagnostic_attempt=discarded_diagnostics if not row_committed else {}))
                raise
    finally:
        problem.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    result = run(read(args.config), cpu=args.cpu, root=args.root, output=args.output)
    print(json.dumps(dict(status=result['status'], termination=result['termination'],
                         iteration=result['iteration'], final=result['final'])))


if __name__ == '__main__':
    main()
