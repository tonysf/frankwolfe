"""Algorithm, work-accounting and restart checks for the expanded CT study."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from paper.experiments.stochastic_ct import robustness_solvers as solvers
from paper.experiments.stochastic_ct import robustness_run as runner
from paper.experiments.stochastic_ct.robustness_reference import methods as legacy


class DenseProblem:
    def __init__(self, n=9):
        self.train_ids = np.arange(n)
        self.validation_ids = np.arange(n, n+3)
        rng = np.random.default_rng(8)
        self.A = rng.uniform(0.01, 0.4, (n+3, 2, 9))
        self.y = rng.uniform(0.05, 0.4, (n+3, 2))
        self.reference = np.full((3, 3), 0.2)
        self.calls = []

    def forward(self, image, views):
        self.calls.append(('forward', tuple(views)))
        return np.einsum('vdi,i->vd', self.A[views], image.ravel())

    def backward(self, sinogram, views):
        self.calls.append(('backward', tuple(views)))
        return np.einsum('vdi,vd->i', self.A[views], sinogram).reshape(3, 3)

    def grad(self, U, V, views, block=None):
        residual = self.forward(U @ V.T, views) - self.y[views]
        z = self.backward(residual, views)/residual.size
        return z @ V if block == 0 else z.T @ U if block == 1 else (z @ V, z.T @ U)

    def image_loss(self, image, views):
        residual = self.forward(image, views)-self.y[views]
        return float(0.5*np.mean(residual**2))

    def close(self):
        pass


U = np.array([[.2, .3], [.1, .2], [.3, .1]])
V = np.array([[.3, .1], [.1, .2], [.2, .2]])
CAPS = (2., 2.)
CURVATURE = (.5, .7)


def make(method, problem=None, config=None):
    problem = problem or DenseProblem()
    defaults = {'batch': 3, 'gamma0': .1, 'beta_scale': 4.} if method in solvers.SMOOTHING else {}
    if method == 'frames':
        defaults['batch'] = 'full'
    return solvers.make_solver(method, problem, U, V, CAPS, seed=42,
                              curvatures=CURVATURE, config=defaults | (config or {}))


def test_sframes_matches_historical_update_bit_for_bit():
    new = make('sframes')
    old = legacy.SFrames(DenseProblem(), U, V, CAPS, seed=100042,
        config=dict(batch=3, gamma0=.1, rho0=1., mu0=4./max(CURVATURE)))
    for _ in range(7):
        event, prior = new.step(), old.step()
        np.testing.assert_array_equal(new.U, old.U)
        np.testing.assert_array_equal(new.V, old.V)
        np.testing.assert_array_equal(new.native.mU, old.mU)
        assert event['native_step']['mu'] == prior['mu']
    assert new.counters['gradient_forward_views'] == 21
    assert new.counters['lmo_calls'] == 14


def test_deterministic_frames_uses_current_gradient_and_its_own_exponents():
    problem = DenseProblem()
    solver = make('frames', problem)
    left, right = U.copy(), V.copy()
    for k in range(5):
        gu, gv = problem.grad(left, right, problem.train_ids)
        beta = 4./max(CURVATURE)/(k+1)**(1./3.)
        gamma = .1/(k+1)**(2./3.)
        atom_u = legacy.spectral_lmo(gu+np.minimum(left, 0.)/beta, CAPS[0])
        atom_v = legacy.spectral_lmo(gv+np.minimum(right, 0.)/beta, CAPS[1])
        left = left+gamma*(atom_u-left)
        right = right+gamma*(atom_v-right)
        event = solver.step()
        np.testing.assert_array_equal(solver.U, left)
        np.testing.assert_array_equal(solver.V, right)
        assert event['native_step']['rho'] == 1.
        assert event['native_step']['gamma'] == gamma
    assert solver.counters['gradient_calls'] == 5
    assert solver.config['batch'] == 9
    with pytest.raises(ValueError, match='full gradient'):
        make('frames', config={'batch': 3})


def test_no_average_ablation_preserves_sampling_stream():
    averaged, fresh = make('sframes'), make('sframes_noavg')
    for k in range(6):
        ea, ef = averaged.step(), fresh.step()
        assert ea['native_step']['gamma'] == ef['native_step']['gamma']
        assert ea['native_step']['mu'] == ef['native_step']['mu']
        assert averaged.native.rng.bit_generator.state == fresh.native.rng.bit_generator.state
    assert averaged.problem.calls == fresh.problem.calls
    assert not np.array_equal(averaged.U, fresh.U)


@pytest.mark.parametrize('method', solvers.METHODS)
def test_exact_continuation_and_directional_budget(method):
    a, b = make(method), make(method)
    for _ in range(3):
        a.step()
    state = a.state_dict()
    b.load_state_dict(state)
    for _ in range(4):
        event_a, event_b = a.step(), b.step()
        np.testing.assert_array_equal(a.U, b.U)
        np.testing.assert_array_equal(a.V, b.V)
        assert event_a == event_b
        assert a.counters == b.counters
        if event_a['observation_type'] == 'converged':
            break
    cap = max(a.counters['gradient_forward_views'], a.counters['gradient_backward_views'])
    before = a.native.state_dict()
    calls = len(a.problem.calls)
    event = a.step(view_budget=cap)
    assert event['observation_type'] in ('budget_stop', 'converged')
    assert len(a.problem.calls) == calls
    assert a.counters['gradient_forward_views'] <= cap
    assert a.counters['gradient_backward_views'] <= cap
    # Non-LB updates never advance an RNG at a failed budget check.
    if method != 'joint_lbfgsb':
        assert pickle.dumps(a.native.state_dict()) == pickle.dumps(before)


def test_spring_preflight_preserves_rng_and_exact_gradient_cost():
    solver = make('spring_sarah', config={'batch': 3, 'refresh_period': 2})
    for _ in range(10):
        n = solver.next_gradient_views()
        before = solver.counters['gradient_forward_views']
        state = solver.native.state_dict()
        calls = len(solver.problem.calls)
        assert solver.step(before+n-1)['observation_type'] == 'budget_stop'
        assert pickle.dumps(solver.native.state_dict()) == pickle.dumps(state)
        assert len(solver.problem.calls) == calls
        assert solver.step(before+n)['observation_type'] == 'accepted_update'
        assert solver.counters['gradient_forward_views'] == before+n
    assert solver.counters['curvature_operator_calls'] == 120
    assert solver.counters['forward_views'] == solver.counters['gradient_forward_views']+solver.counters['curvature_forward_views']


def test_proxsarah_paper_parameters_and_snapshot_cost():
    params = solvers.parameters('prox_sarah', 1080, CURVATURE, {'curvature_scale': .25})
    assert params['batch'] == 48 and params['inner_steps'] == 32
    assert params['eta'] == (2./4.99)/(.25*sum(CURVATURE))
    solver = make('prox_sarah')
    assert solver.next_gradient_views() == 9
    solver.step(9)
    assert solver.next_gradient_views() == 2*solver.config['batch']


def test_true_smoothed_gap_uses_current_beta_and_full_gradient():
    solver = make('sframes')
    solver.step()
    diagnostics = runner.initial_diagnostics()
    row = runner.observe(solver.problem, solver, np.hypot(np.linalg.norm(U), np.linalg.norm(V)),
                         diagnostics, {'observation_type': 'accepted_update'}, True)
    gu, gv = solver.problem.grad(solver.U, solver.V, solver.problem.train_ids)
    beta = solver.config['mu0']/2**.25
    hu, hv = gu+np.minimum(solver.U, 0.)/beta, gv+np.minimum(solver.V, 0.)/beta
    support_gap = np.sum(hu*(solver.U-legacy.spectral_lmo(hu, CAPS[0])))
    support_gap += np.sum(hv*(solver.V-legacy.spectral_lmo(hv, CAPS[1])))
    assert row['stationarity_beta'] == beta
    assert row['smoothed_fw_gap'] == pytest.approx(support_gap, rel=1e-13)
    assert row['mean_smoothed_fw_gap'] == row['smoothed_fw_gap']
    assert diagnostics['backward_views'] == 9
    assert solver.counters['gradient_forward_views'] == 3


def runner_config(tmp_path, monkeypatch, method='sframes'):
    data = tmp_path/'data.bin'; data.write_bytes(b'fixed fake CT data')
    source = {'ct_problem.py': '0'*64, 'robustness_run.py': '1'*64}
    monkeypatch.setattr(runner, 'source_hashes', lambda: source)
    calibration = tmp_path/'calibration.npz'
    np.savez(calibration, U=U, V=V, caps=CAPS, curvature=CURVATURE,
             data_sha256=runner.sha256(data), problem_sha256=source['ct_problem.py'])
    runner.write(tmp_path/'protocol.json', {'numerical_sources': source})
    return dict(run_id='test', phase='development', study='test', variant='test',
        method=method, slice_id='fake', n_train=9, grid=3, rank=2, seed=42,
        radius_scale=2., view_budget=15, parameters=dict(batch=3, beta_scale=4., gamma0=.1),
        record_every_iteration=True, record_stationarity=True,
        data_file=str(data), data_sha256=runner.sha256(data),
        calibration_file=str(calibration), calibration_sha256=runner.sha256(calibration))


def test_dense_runner_idempotence_and_terminal_accounting(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    result = runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    trace = runner.read(tmp_path/'runs/test/trace.json')
    assert [row['iteration'] for row in trace] == [0, 1, 2, 3, 4, 5, 5]
    assert result['status'] == 'completed' and result['termination'] == 'budget_stop'
    assert result['final']['processed_views'] == 15
    assert result['final']['stationarity_observations'] == 6
    assert result['final']['diagnostic_backward_views'] == 54
    assert trace[-1]['factor_u_sha256'] == trace[-2]['factor_u_sha256']
    assert runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem()) == result


def test_runner_recovers_a_failed_accepted_observation(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    original = runner.observe
    failed = False
    def fail_once(problem, solver, *args, **kwargs):
        nonlocal failed
        if solver.iteration == 2 and not failed:
            failed = True
            raise OSError('simulated diagnostic write/IO failure')
        return original(problem, solver, *args, **kwargs)
    monkeypatch.setattr(runner, 'observe', fail_once)
    with pytest.raises(OSError):
        runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    resumed = runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    other = tmp_path/'uninterrupted'; other.mkdir()
    runner.write(other/'protocol.json', runner.read(tmp_path/'protocol.json'))
    clean = runner.run(config, root=other, problem_factory=lambda cfg: DenseProblem())
    assert resumed['final'] == clean['final']
    assert runner.read(tmp_path/'runs/test/trace.json') == runner.read(other/'runs/test/trace.json')


def test_numerical_candidate_failure_is_retained_and_not_success(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    original = solvers.Solver.step
    def fail(self, *args, **kwargs):
        if self.iteration == 2:
            raise solvers.NumericalFailure('Nonfinite synthetic candidate')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(solvers.Solver, 'step', fail)
    result = runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    assert result['status'] == 'numerical_failure'
    assert result['final']['iteration'] == 2
    assert (tmp_path/'runs/test/failure.json').exists()
    state = pickle.loads((tmp_path/'runs/test/checkpoint.pkl').read_bytes())
    assert state['solver']['native']['k'] == 2
    assert np.isfinite(state['solver']['native']['U']).all()


def test_frozen_inputs_and_sources_are_enforced(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    Path(config['data_file']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Frozen input'):
        runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())


def test_runner_discards_partial_diagnostic_aggregate_on_recovery(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    original = runner.observe
    failed = False
    def fail_after_observe(problem, solver, *args, **kwargs):
        nonlocal failed
        row = original(problem, solver, *args, **kwargs)
        if solver.iteration == 2 and not failed:
            failed = True
            raise OSError('failure after computing the diagnostic but before committing its row')
        return row
    monkeypatch.setattr(runner, 'observe', fail_after_observe)
    with pytest.raises(OSError):
        runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    resumed = runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    other = tmp_path/'uninterrupted'; other.mkdir()
    runner.write(other/'protocol.json', runner.read(tmp_path/'protocol.json'))
    clean = runner.run(config, root=other, problem_factory=lambda cfg: DenseProblem())
    assert resumed['final'] == clean['final']


def test_runner_replays_a_failed_native_update_from_its_original_rng(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    original = legacy.SFrames._update
    failed = False
    def fail_mid_update(self):
        nonlocal failed
        details = original(self)
        if self.k == 1 and not failed:
            failed = True
            raise OSError('failure after factor/momentum work before accepting native update')
        return details
    monkeypatch.setattr(legacy.SFrames, '_update', fail_mid_update)
    with pytest.raises(OSError):
        runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    resumed = runner.run(config, root=tmp_path, problem_factory=lambda cfg: DenseProblem())
    other = tmp_path/'uninterrupted'; other.mkdir()
    runner.write(other/'protocol.json', runner.read(tmp_path/'protocol.json'))
    clean = runner.run(config, root=other, problem_factory=lambda cfg: DenseProblem())
    assert resumed['final'] == clean['final']
