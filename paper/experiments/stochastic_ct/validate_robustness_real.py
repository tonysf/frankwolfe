"""Bounded real-ASTRA checks of the expanded-view CT runner.

Each case permits one or two native updates (L-BFGS-B may spend its three
pairs on line-search work). This validates the actual 512x64 data path and
accounting, rather than extrapolating from the dense unit-test projector.
"""

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import robustness_run as run
    from .robustness_solvers import make_solver, parameters
    from .robustness_inputs import NewCTProblem
    from .robustness_reference.methods import SFrames
except ImportError:
    import robustness_run as run
    from robustness_solvers import make_solver, parameters
    from robustness_inputs import NewCTProblem
    from robustness_reference.methods import SFrames


EXPANDED = Path('/home/tony/frames_ct_2detect_20261104/spectral_balls/expanded')


def validate(root, cpu=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = run.source_hashes()
    protocol = dict(numerical_sources=sources, purpose='Bounded real-data implementation validation')
    if (root/'protocol.json').exists() and run.read(root/'protocol.json') != protocol:
        raise ValueError('Choose a fresh validation root after changing source')
    run.write(root/'protocol.json', protocol)
    data = EXPANDED/'convincing_4x/data/slice03150.npz'
    calibration = EXPANDED/'convincing_4x_recovery/calibration/slice03150_seed110.npz'
    with np.load(calibration, allow_pickle=False) as saved:
        U, V = saved['U'].copy(), saved['V'].copy()
        caps, curvature = tuple(saved['caps']), tuple(saved['curvature'])
    cases = []
    for n in (120, 2160):
        for method in ('sframes', 'frames', 'prox_sarah', 'joint_lbfgsb', 'spring_sarah'):
            params = dict(batch='full' if method == 'frames' else 64, beta_scale=4., gamma0=.1) if method in ('sframes', 'frames') else {}
            if method == 'joint_lbfgsb':
                params = dict(memory=20)
            resolved = parameters(method, n, curvature, params)
            budget = {'sframes': 128, 'frames': 2*n, 'prox_sarah': n+2*resolved.get('batch', 0),
                      'joint_lbfgsb': 3*n, 'spring_sarah': 2*n}[method]
            cfg = dict(run_id=f'{method}_{n}', phase='validation', study='real_engine',
                variant=method, method=method, slice_id='slice03150', n_train=n,
                grid=512, rank=64, seed=110, initialization='standard', radius_scale=2.,
                view_budget=budget, parameters=params, record_every_iteration=True,
                record_stationarity=method in ('sframes', 'frames'),
                data_file=str(data), data_sha256=run.sha256(data),
                calibration_file=str(calibration), calibration_sha256=run.sha256(calibration))
            result = run.run(cfg, cpu=cpu, root=root)
            assert result['status'] == 'completed', result
            assert result['termination'] == 'budget_stop', result
            trace = run.read(root/'runs'/cfg['run_id']/'trace.json')
            unique = [row for row in trace if row['observation_type'] != 'budget_stop']
            assert [r['iteration'] for r in unique] == list(range(len(unique)))
            assert 1 <= result['iteration'] <= 2, (method, n, result['iteration'])
            for row in trace:
                assert max(row['gradient_forward_views'], row['gradient_backward_views']) <= budget
                assert row['c_violation'] <= 1e-8 if method != 'joint_lbfgsb' else True
                assert np.isfinite([row['raw_loss'], row['feasible_loss'], row['validation_loss']]).all()
                if method not in ('sframes', 'frames'):
                    assert row['d_distance'] == 0
            cases.append(dict(method=method, n_train=n, iterations=result['iteration'],
                gradient_forward_views=result['final']['gradient_forward_views'],
                gradient_backward_views=result['final']['gradient_backward_views'],
                curvature_forward_views=result['final']['curvature_forward_views'],
                diagnostic_forward_views=result['final']['diagnostic_forward_views'],
                final_loss=result['final']['raw_loss'],
                result_sha256=run.sha256(root/'runs'/cfg['run_id']/'result.json')))
            print(json.dumps(cases[-1]), flush=True)
    # Confirm that the package wrapper leaves the historical numerical update
    # bit-for-bit unchanged on the original view split and physical dimensions.
    problem = NewCTProblem(data, 512, 360)
    try:
        modern = make_solver('sframes', problem, U, V, caps, seed=110,
            curvatures=curvature, config=dict(batch=64, gamma0=.1, beta_scale=4.))
        old = SFrames(problem, U, V, caps, seed=100110,
            config=dict(batch=64, gamma0=.1, rho0=1., mu0=4./max(curvature)))
        for _ in range(2):
            modern.step(); old.step()
            np.testing.assert_array_equal(modern.U, old.U)
            np.testing.assert_array_equal(modern.V, old.V)
            np.testing.assert_array_equal(modern.native.mU, old.mU)
            np.testing.assert_array_equal(modern.native.mV, old.mV)
    finally:
        problem.close()
    summary = dict(status='passed', source_sha256=sources, cases=cases,
        historical_sframes_bit_identical=True, historical_sframes_steps=2,
        data_sha256=run.sha256(data), calibration_sha256=run.sha256(calibration),
        validate_source_sha256=run.sha256(__file__), optimizer_timing=False)
    run.write(root/'real_engine_validation.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()
    result = validate(args.root, args.cpu)
    print(json.dumps(dict(status=result['status'], cases=len(result['cases']),
                         historical_sframes_bit_identical=True)))


if __name__ == '__main__':
    main()
