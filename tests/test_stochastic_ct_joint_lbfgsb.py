"""Independent numerical and recovery checks for joint CT L-BFGS-B."""

from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import minimize

from paper.experiments.stochastic_ct import joint_lbfgsb_solver as engine
from paper.experiments.stochastic_ct import joint_lbfgsb_run as runner


class DenseProblem:
    def __init__(self, grid=4, rank=2, n_train=4):
        rng = np.random.default_rng(17)
        self.grid = grid
        self.train_ids = np.arange(n_train)
        self.validation_ids = np.arange(n_train, n_train + 2)
        self.matrix = rng.uniform(.01, 1, (n_train+2, 6, grid*grid))
        self.reference = rng.uniform(.1, 1, (grid, rank)) @ rng.uniform(.1, 1, (grid, rank)).T
        self.y = self.matrix @ self.reference.ravel()
        self.y[0, 0] -= 3  # Retain signed/noisy data without modifying the model.
        self.forward_calls = self.backward_calls = 0

    def forward(self, image, ids):
        self.forward_calls += 1
        return self.matrix[ids] @ image.ravel()

    def backward(self, sinogram, ids):
        self.backward_calls += 1
        return np.einsum("vdi,vd->i", self.matrix[ids], sinogram).reshape(self.grid, self.grid)

    def image_loss(self, image, ids):
        return float(.5*np.mean((self.matrix[ids] @ image.ravel() - self.y[ids])**2))

    def close(self):
        pass


def fixture():
    problem = DenseProblem()
    rng = np.random.default_rng(43)
    U, V = rng.uniform(.05, .5, (4, 2)), rng.uniform(.05, .5, (4, 2))
    return problem, U, V


def objective(problem, U, V):
    shape, split = U.shape, U.size
    A = problem.matrix[problem.train_ids].reshape(-1, problem.grid**2)
    y = problem.y[problem.train_ids].ravel()

    def evaluate(x):
        a, b = x[:split].reshape(shape), x[split:].reshape(shape)
        residual = A @ (a @ b.T).ravel() - y
        image_gradient = (A.T @ residual).reshape(problem.grid, problem.grid)/y.size
        return .5*np.mean(residual**2), np.concatenate(((image_gradient @ b).ravel(),
                                                       (image_gradient.T @ a).ravel()))
    return evaluate


def same_state(a, b):
    if isinstance(a, np.ndarray):
        assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_state(a[k], b[k])
    else:
        assert a == b


@pytest.mark.parametrize("memory", [5, 10, 20])
def test_joint_accepted_steps_match_public_scipy_without_memory_restart(memory):
    problem, U, V = fixture()
    evaluate = objective(problem, U, V)
    x0 = np.concatenate((U.ravel(), V.ravel()))
    gtol = 1e-8*engine.projected_gradient_inf(x0, evaluate(x0)[1])
    expected = []
    reference = minimize(evaluate, x0, jac=True, method="L-BFGS-B",
        bounds=[(0, None)]*len(x0), options=dict(maxcor=memory, gtol=gtol, ftol=0,
                                              maxiter=12, maxls=20),
        callback=lambda x: expected.append(x.copy()))
    solver = engine.JointLBFGSB(problem, U, V,
        config=dict(memory=memory, relative_kkt_tolerance=1e-8, ftol=0))
    actual = []
    for _ in range(12):
        event = solver.step()
        assert event["accepted"] and event["updated_blocks"] == "both"
        actual.append(np.concatenate((solver.U.ravel(), solver.V.ravel())))
    assert len(actual) == len(expected) == 12
    for a, b in zip(actual, expected):
        assert_allclose(a, b, rtol=1e-10, atol=1e-11)
    assert_allclose(actual[-1], reference.x, rtol=1e-10, atol=1e-11)
    assert solver.counters["gradient_calls"] == reference.nfev
    assert solver.counters["forward_calls"] == problem.forward_calls == reference.nfev
    assert solver.counters["backward_calls"] == problem.backward_calls == reference.nfev
    assert solver.completed_blocks == solver.outer_sweeps == 0
    assert not np.array_equal(solver.U, U) and not np.array_equal(solver.V, V)


def test_both_trial_factor_gradients_match_independent_finite_differences():
    problem, U, V = fixture()
    evaluate = objective(problem, U, V)
    solver = engine.JointLBFGSB(problem, U, V)
    for _ in range(4):
        solver.advance()
    assert solver.phase == "setulb" and solver.iteration == 0
    x, g = solver.work["lb"]["x"], solver.work["lb"]["g"]
    assert not np.array_equal(x, np.concatenate((U.ravel(), V.ravel())))
    step = 1e-6
    fd = np.array([(evaluate(x+step*np.eye(len(x))[i])[0]
                    - evaluate(x-step*np.eye(len(x))[i])[0])/(2*step) for i in range(len(x))])
    assert_allclose(g, fd, rtol=2e-7, atol=1e-8)
    assert_array_equal(solver.U, U)
    assert_array_equal(solver.V, V)


def test_all_trial_work_is_budgeted_and_resume_preserves_pending_trial():
    problem, U, V = fixture()
    solver = engine.JointLBFGSB(problem, U, V)
    assert solver.step(view_budget=4)["observation_type"] == "budget_stop"
    assert solver.iteration == 0
    assert solver.counters["forward_views"] == solver.counters["backward_views"] == 4
    assert_array_equal(solver.U, U)
    assert_array_equal(solver.V, V)
    checkpoint = solver.state_dict()
    restarted = engine.JointLBFGSB(DenseProblem(), U, V)
    restarted.load_state_dict(checkpoint)
    direct = engine.JointLBFGSB(DenseProblem(), U, V)
    for state in (restarted, direct):
        for _ in range(100):
            if state.step(view_budget=80)["observation_type"] == "budget_stop":
                break
        else:
            pytest.fail("Expected budget stop")
    same_state(restarted.state_dict(), direct.state_dict())


def test_exact_resume_between_forward_and_adjoint():
    problem, U, V = fixture()
    solver = engine.JointLBFGSB(problem, U, V)
    for _ in range(3):
        solver.advance(view_budget=200)
    assert solver.phase == "backward" and solver.counters["forward_calls"] == 2
    checkpoint = solver.state_dict()
    restarted = engine.JointLBFGSB(DenseProblem(), U, V)
    restarted.load_state_dict(checkpoint)
    for _ in range(8):
        assert restarted.step(view_budget=200) == solver.step(view_budget=200)
        same_state(restarted.state_dict(), solver.state_dict())


@pytest.mark.parametrize("reason", ["kkt", "relative_function_decrease"])
def test_native_convergence_reasons_are_kept_distinct(reason):
    problem, U, V = fixture()
    config = dict(ftol=1.0, relative_kkt_tolerance=0.)
    if reason == "kkt":
        problem.y = problem.matrix @ (U @ V.T).ravel()
        config["ftol"] = 0.
    solver = engine.JointLBFGSB(problem, U, V, config=config)
    for _ in range(10):
        event = solver.step(view_budget=400)
        if event["observation_type"] == "converged":
            break
    else:
        pytest.fail("Expected native convergence")
    assert event["stop_reason"] == event["block_stop_reason"] == reason
    assert event["native_task"] == [4, 401 if reason == "kkt" else 402]
    assert solver.step(view_budget=400) == event


@pytest.mark.parametrize("corruption, match", [("scipy", "SciPy"), ("array", "restart array"),
                                               ("counters", "counters"), ("abi", "ABI")])
def test_invalid_checkpoint_is_rejected_before_native_call(corruption, match):
    problem, U, V = fixture()
    solver = engine.JointLBFGSB(problem, U, V)
    solver.advance()
    state = solver.state_dict()
    if corruption == "scipy": state["scipy_version"] = "0.0"
    if corruption == "array": state["work"]["lb"]["wa"] = np.zeros(1)
    if corruption == "counters": state["counters"]["forward_views"] += 1
    if corruption == "abi": state["lbfgsb_integer_dtype"] = "invalid"
    with pytest.raises(ValueError, match=match): solver.load_state_dict(state)


def setup_run(root):
    data = root / "data.npz"
    np.savez(data, fixture=np.array("synthetic only"))
    _, U, V = fixture()
    calibration = root / "calibration.npz"
    np.savez(calibration, U=U, V=V, caps=np.array([.01, .01]),
        data_sha256=np.array(runner.sha256(data)),
        problem_sha256=np.array(runner.source_hashes()["ct_problem.py"]))
    runner.write(root / "protocol.json", {"numerical_sources": runner.source_hashes()})
    return dict(run_id="joint", method="joint_lbfgsb", phase="development", grid=4, rank=2,
        n_train=4, slice_id="synthetic", calibration_seed=110, record_every_iteration=True,
        view_budget=4*8, data_file=str(data), calibration_file=str(calibration),
        data_sha256=runner.sha256(data), calibration_sha256=runner.sha256(calibration),
        parameters=engine.parameters())


def factory(config):
    return DenseProblem(config["grid"], config["rank"], config["n_train"])


def test_runner_dense_observations_and_terminal_receipt_recovery(tmp_path):
    cfg = setup_run(tmp_path)
    result = runner.run(cfg, root=tmp_path, problem_factory=factory)
    folder = tmp_path / "runs/joint"
    trace = runner.read(folder / "trace.json")
    accepted = [x for x in trace if x["observation_type"] == "accepted_update"]
    assert [x["iteration"] for x in accepted] == list(range(1, len(accepted)+1))
    assert trace[-1]["iteration"] == trace[-2]["iteration"]
    assert trace[-1]["forward_views"] == trace[-1]["backward_views"] == cfg["view_budget"]
    assert all(x["raw_loss"] == x["feasible_loss"] and x["d_distance"] == 0 for x in trace)
    assert trace[-1]["reference_spectral_ratio_u"] > 1
    assert trace[-1]["reference_spectral_ratio_v"] > 1
    assert runner.run(cfg, root=tmp_path, problem_factory=factory) == result
    (folder / "result.json").unlink()
    receipt = runner.read(folder / "checkpoint.sha256.json")
    receipt["previous_sha256"], receipt["sha256"] = receipt["sha256"], "0"*64
    runner.write(folder / "checkpoint.sha256.json", receipt)
    assert runner.run(cfg, root=tmp_path, problem_factory=factory) == result
    assert runner.read(folder / "trace.json") == trace


@pytest.mark.parametrize("kind", ["accepted_update", "budget_stop"])
def test_runner_recovers_interrupted_accepted_and_terminal_diagnostics(tmp_path, monkeypatch, kind):
    cfg = setup_run(tmp_path)
    direct = runner.run(cfg, root=tmp_path, output=tmp_path / "direct", problem_factory=factory)
    original = runner.observe

    def interrupted(problem, solver, caps, event):
        if event["observation_type"] == kind:
            raise RuntimeError("synthetic diagnostic interruption")
        return original(problem, solver, caps, event)

    monkeypatch.setattr(runner, "observe", interrupted)
    with pytest.raises(RuntimeError, match="synthetic diagnostic interruption"):
        runner.run(cfg, root=tmp_path, output=tmp_path / "interrupted", problem_factory=factory)
    monkeypatch.setattr(runner, "observe", original)
    resumed = runner.run(cfg, root=tmp_path, output=tmp_path / "interrupted", problem_factory=factory)
    assert resumed == direct
    assert runner.read(tmp_path / "interrupted/trace.json") == runner.read(tmp_path / "direct/trace.json")
