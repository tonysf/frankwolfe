"""One-update real CT continuation smoke for each method; never production runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np

import campaign
import run as engine
from validate_recording import same


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate():
    root = engine.ROOT
    protocol = campaign.verify_prepared()
    plan = engine.read_json(root / 'evaluation_plan.json')
    reports = []
    for method in ('palm', 'spring_sarah'):
        original = next(c for c in plan['runs'] if c['method'] == method and c['slice_id'] == 'slice03611'
                        and c['n_train'] == 360 and c['seed'] == 203)
        cfg = deepcopy(original)
        cfg['run_id'] = 'recording_smoke_' + cfg['run_id']
        directory = Path(cfg['origin_directory'])
        old_state = pickle.loads((directory / 'checkpoint.pkl').read_bytes())
        old_meta = engine.read_json(directory / 'metadata.json')
        prefix = old_state['trace']
        cfg['gradient_view_budget'] = engine.gradient_views(old_state['counters']) + (2 * cfg['n_train'] if method == 'palm' else 1)
        factory_path = directory.parent.parent / ('palm_paper.py' if method == 'palm' else 'local_curvature.py')
        factory = load(f'original_{method}_smoke_factory', factory_path)
        problem = engine.CTProblem(root / 'data' / f'{cfg["slice_id"]}.npz', cfg['grid'], cfg['n_train'])
        try:
            with np.load(root / 'calibration' / f'{cfg["slice_id"]}_seed{cfg["calibration_seed"]}.npz') as saved:
                U, V, caps = saved['U'], saved['V'], tuple(saved['caps'].tolist())
            direct = factory.make_solver(problem, U, V, caps, seed=cfg['seed'] + 100_000,
                                         config=old_meta.get('factory_config', old_meta['optimizer']))
            direct.load_state_dict(old_state['solver'])
            counters = deepcopy(old_state['counters'])
            step = direct.step()
            for name in engine.COUNTS:
                counters[name] += step[name]
            counters['max_projection_distance_bound'] = max(counters['max_projection_distance_bound'],
                                                            step['projection_max_distance_bound'])
            step = json.loads(json.dumps(step, default=engine.json_scalar, allow_nan=False))
            expected = engine.observe(problem, direct.U, direct.V, caps, counters, direct.k, step)
            expected_solver = direct.state_dict()
        finally:
            problem.close()
        engine.run(cfg, cpu=6, runs_dir=root / 'smoke_runs')
        folder = root / 'smoke_runs' / cfg['run_id']
        trace = engine.read_json(folder / 'trace.json')
        state = pickle.loads((folder / 'checkpoint.pkl').read_bytes())
        assert trace == [*prefix, expected]
        assert same(state['solver'], expected_solver) and state['counters'] == counters
        assert trace[-1]['iteration'] == prefix[-1]['iteration'] + 1
        hashes = {name: engine.sha256(folder / name) for name in engine.ORIGIN_FILES}
        engine.run(cfg, cpu=6, runs_dir=root / 'smoke_runs')
        assert hashes == {name: engine.sha256(folder / name) for name in hashes}
        # The production validator can inspect the smoke run through temporary
        # aliases, leaving every production run directory and plan untouched.
        with tempfile.TemporaryDirectory(prefix='smoke_validator_', dir=root) as temporary:
            validation_root = Path(temporary)
            (validation_root / 'runs').symlink_to(root / 'smoke_runs', target_is_directory=True)
            for name in ('implementation_manifest.json',):
                (validation_root / name).symlink_to(root / name)
            saved_root = campaign.ROOT
            try:
                campaign.ROOT = validation_root
                checked = campaign.validate_result(cfg, 6, plan, protocol)
                assert checked['prefix_preserved'] and checked['added_iterations'] == 1
            finally:
                campaign.ROOT = saved_root
        reports.append(dict(method=method, run_id=cfg['run_id'], source_directory=str(folder),
            source_file_sha256=hashes, original_factory=str(factory_path), original_factory_sha256=engine.sha256(factory_path),
            imported_iteration=prefix[-1]['iteration'], final_iteration=trace[-1]['iteration'],
            requested_gradient_view_budget=cfg['gradient_view_budget'],
            final_gradient_views=engine.gradient_views(trace[-1]), final_total_views=trace[-1]['forward_views'],
            step=step, raw_loss=trace[-1]['raw_loss'],
            original_prefix_exact=True, original_factory_next_iterate_exact=True,
            solver_and_rng_state_exact=True, completed_replay_idempotent=True, campaign_validator_passed=True))
        print(json.dumps(dict(method=method, status='passed', final_iteration=trace[-1]['iteration'])), flush=True)
    return dict(status='passed', source_sha256={name: engine.sha256(root / name) for name in engine.SOURCES},
        test_source_sha256=engine.sha256(Path(__file__)), protocol_sha256=engine.sha256(root / 'protocol.json'),
        cases=reports, production_runs_created=False, production_plan_or_selection_changed=False)


if __name__ == '__main__':
    result = validate()
    engine.write_json(Path(__file__).with_name('real_engine_validation.json'), result)
    print(json.dumps(result, indent=2))
