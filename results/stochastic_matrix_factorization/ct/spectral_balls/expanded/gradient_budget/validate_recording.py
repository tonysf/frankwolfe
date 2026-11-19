"""Bounded continuation tests on tiny dense operators; no CT jobs."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np

import run as engine


class TinyProblem:
    def __init__(self, data_file, grid, n_train):
        self.train_ids = np.arange(n_train)
        self.validation_ids = np.arange(n_train, n_train + 2)
        self.reference = np.array([[.8, .2, .4], [.3, .7, .1], [.1, .4, .6]])
    def forward(self, image, ids):
        return (1 + np.asarray(ids)[:, None] / 100) * image.reshape(1, -1)
    def backward(self, values, ids):
        return np.sum((1 + np.asarray(ids)[:, None] / 100) * values, axis=0).reshape(3, 3)
    def grad(self, U, V, ids, block=None):
        residual = self.forward(U @ V.T - self.reference, ids)
        adjoint = self.backward(residual, ids) / residual.size
        if block == 0:
            return adjoint @ V
        if block == 1:
            return adjoint.T @ U
        return adjoint @ V, adjoint.T @ U
    def image_loss(self, image, ids):
        return float(.5 * np.mean(self.forward(image - self.reference, ids) ** 2))
    def close(self):
        pass


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def advance(problem, solver, caps, counters, trace):
    step = solver.step()
    for name in engine.COUNTS:
        counters[name] += step[name]
    counters['max_projection_distance_bound'] = max(counters['max_projection_distance_bound'],
                                                    step['projection_max_distance_bound'])
    step = json.loads(json.dumps(step, default=engine.json_scalar))
    trace.append(engine.observe(problem, solver.U, solver.V, caps, counters, solver.k, step))
    return step


def fixture_origin(root, method, sources):
    data = root / 'data/tiny.npz'
    U = np.array([[.6, .1], [.3, .5], [.2, .4]])
    V = np.array([[.5, .2], [.2, .6], [.4, .3]])
    caps, curvature = (2., 2.), (.2, .2)
    np.savez(root / 'calibration/tiny_seed201.npz', U=U, V=V, caps=caps, curvature=curvature,
             data_sha256=engine.sha256(data), problem_sha256=sources['ct_problem.py'])
    factory = engine.palm_paper if method == 'palm' else engine.local_curvature
    factory_config = (dict(power_iterations=5, step_coefficient=1.) if method == 'palm'
                      else dict(batch=3, refresh_period=16, power_iterations=5, step_coefficient=.5))
    params = engine.palm_paper.parameters(8, factory_config) if method == 'palm' else factory_config
    config = dict(run_id=f'original_{method}', phase='evaluation', method=method,
                  slice_id='tiny', n_train=8, grid=3, rank=2, calibration_seed=201, seed=202,
                  record_every_iteration=True, view_budget=1)
    problem = TinyProblem(data, 3, 8)
    solver = factory.make_solver(problem, U, V, caps, seed=100202, config=factory_config)
    counters = dict.fromkeys(engine.COUNTS, 0)
    counters.update(max_projection_distance_bound=0., max_lmo_support_gap_bound=0.)
    trace = [engine.observe(problem, U, V, caps, counters, 0, None)]
    for _ in range(2):
        step = advance(problem, solver, caps, counters, trace)
    # The source was completed at an inclusive work cap, then its overshooting
    # final update remains valid under the larger gradient-only cap.
    config['view_budget'] = trace[-2]['forward_views'] + 1
    for row in trace:
        del row['gradient_forward_views'], row['gradient_backward_views']
    metadata = dict(config=config, optimizer=params, caps=list(caps), curvature=list(curvature),
        initial_factor_frobenius_norm=float(np.hypot(np.linalg.norm(U), np.linalg.norm(V))),
        initial_factor_norm=float(np.hypot(np.linalg.norm(U), np.linalg.norm(V))),
        source_sha256=sources, data_sha256=engine.sha256(data),
        calibration_sha256=engine.sha256(root / 'calibration/tiny_seed201.npz'),
        numpy=np.__version__, astra=engine.astra.__version__, cpu=None,
        train_view_ids=problem.train_ids.tolist(), validation_view_ids=problem.validation_ids.tolist(),
        projector='tiny dense operator', factor_dtype='float64', initialization=dict(name='warm', seed=201),
        constraint_geometry='separate spectral norm balls and entrywise nonnegativity',
        continuation=None, imported_checkpoint=None, recovery_origin='fresh')
    if method == 'palm':
        metadata['factory_config'] = factory_config
    origin = root / 'origins' / config['run_id']
    origin.mkdir(parents=True)
    engine.write_json(origin / 'metadata.json', metadata)
    engine.write_json(origin / 'trace.json', trace)
    engine.write_json(origin / 'result.json', dict(metadata, status='completed', iteration=solver.k, final=trace[-1]))
    engine.atomic_write(origin / 'checkpoint.pkl', pickle.dumps(dict(metadata=metadata,
        solver=solver.state_dict(), counters=deepcopy(counters), trace=deepcopy(trace), last_step=step), protocol=5))
    continuation = {name: config[name] for name in engine.COMMON_CONFIG}
    continuation.update(run_id=f'continued_{method}', phase='evaluation', record_every_iteration=True,
        gradient_view_budget=77 if method == 'palm' else 81, origin_directory=str(origin),
        origin_run_id=config['run_id'], origin_config=config)
    for filename in engine.ORIGIN_FILES:
        continuation[f'origin_{Path(filename).stem}_sha256'] = engine.sha256(origin / filename)
    return continuation, problem, solver, caps, counters, trace


def validate():
    original_root, real_problem, real_abort = engine.ROOT, engine.CTProblem, engine.check_abort
    sources = {name: engine.sha256(original_root / name) for name in engine.SOURCES}
    checks = []
    with tempfile.TemporaryDirectory(prefix='gradient-continuation-') as directory:
        root = Path(directory)
        for name in engine.SOURCES:
            (root / name).symlink_to(original_root / name)
        (root / 'data').mkdir()
        (root / 'calibration').mkdir()
        np.savez(root / 'data/tiny.npz', fixture=np.ones(1))
        engine.write_json(root / 'specification.json', dict(test='bounded continuation fixture'))
        engine.write_json(root / 'implementation_manifest.json', dict(current_source_sha256=sources,
            specification_sha256=engine.sha256(root / 'specification.json'),
            authorized_changes=['continuation fixture']))
        engine.ROOT, engine.CTProblem = root, TinyProblem
        try:
            for method in ('palm', 'spring_sarah'):
                cfg, problem, direct, caps, counters, reference = fixture_origin(root, method, sources)
                prefix = deepcopy(reference)
                while engine.gradient_views(counters) < cfg['gradient_view_budget']:
                    advance(problem, direct, caps, counters, reference)
                calls = 0
                def interrupted():
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise RuntimeError('bounded injected interruption')
                engine.check_abort = interrupted
                try:
                    engine.run(cfg, runs_dir=root / 'resumed')
                except RuntimeError as exc:
                    assert str(exc) == 'bounded injected interruption'
                else:
                    raise AssertionError('Expected a saved-step interruption')
                engine.check_abort = real_abort
                folder = root / 'resumed' / cfg['run_id']
                partial = engine.read_json(folder / 'trace.json')
                assert len(partial) == len(prefix) + 1 and partial[:len(prefix)] == prefix
                engine.run(cfg, runs_dir=root / 'resumed')
                engine.run(cfg, runs_dir=root / 'uninterrupted')
                trace = engine.read_json(folder / 'trace.json')
                assert trace == reference == engine.read_json(root / 'uninterrupted' / cfg['run_id'] / 'trace.json')
                state = pickle.loads((folder / 'checkpoint.pkl').read_bytes())
                assert same(state['solver'], direct.state_dict())
                assert state['counters'] == counters
                assert [v['iteration'] for v in trace] == list(range(len(trace)))
                assert trace[:len(prefix)] == prefix
                assert engine.gradient_views(trace[-2]) < cfg['gradient_view_budget'] <= engine.gradient_views(trace[-1])
                for row in trace:
                    assert row['oracle_calls'] == row['gradient_calls'] + row['curvature_operator_calls']
                    assert engine.gradient_views(row) == engine.gradient_views(row, 'backward')
                hashes = {name: engine.sha256(folder / name) for name in ('metadata.json', 'trace.json', 'checkpoint.pkl', 'result.json')}
                engine.run(cfg, runs_dir=root / 'resumed')
                assert hashes == {name: engine.sha256(folder / name) for name in hashes}
                broken = dict(cfg, origin_checkpoint_sha256='0' * 64)
                try:
                    engine.run(broken, runs_dir=root / 'invalid')
                except ValueError as exc:
                    assert 'Continuation source changed' in str(exc)
                else:
                    raise AssertionError('Changed source checkpoint accepted')
                checks.append(dict(method=method, original_iterations=len(prefix) - 1, final_iteration=direct.k,
                    final_gradient_views=engine.gradient_views(trace[-1]), final_total_views=trace[-1]['forward_views'],
                    original_prefix_exact=True, continued_equals_uninterrupted=True,
                    interrupted_resume_exact=True, final_solver_and_rng_state_exact=True,
                    native_cap_crossing=True, completed_replay_idempotent=True, source_digest_rejection=True))
        finally:
            engine.ROOT, engine.CTProblem, engine.check_abort = original_root, real_problem, real_abort
    return dict(status='passed', source_sha256=sources, test_source_sha256=engine.sha256(Path(__file__)),
                no_real_ct_operator_or_production_job=True, cases=checks)


if __name__ == '__main__':
    result = validate()
    engine.write_json(Path(__file__).with_name('engine_validation.json'), result)
    print(json.dumps(result, indent=2))
