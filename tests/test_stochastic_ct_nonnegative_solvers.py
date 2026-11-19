"""Independent dense references for the matrix-free, budgeted NNLS/MU solvers."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import minimize, nnls


SPEC = importlib.util.spec_from_file_location(
    "ct_nonnegative_solvers",
    Path(__file__).resolve().parents[1] / "paper/experiments/stochastic_ct/nonnegative_solvers.py",
)
SOLVERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOLVERS)


class DenseTomography:
    def __init__(self, matrix, image_shape, observed, n_views):
        self.matrix = np.asarray(matrix, dtype=float)
        self.image_shape = image_shape
        self.y = np.asarray(observed, dtype=float).reshape(n_views, -1)
        self.train_ids = np.arange(n_views)
        self.forward_calls = self.backward_calls = 0

    def forward(self, image, ids):
        assert_array_equal(ids, self.train_ids)
        self.forward_calls += 1
        return (self.matrix @ image.ravel()).reshape(self.y.shape)

    def backward(self, sinogram, ids):
        assert_array_equal(ids, self.train_ids)
        self.backward_calls += 1
        return (self.matrix.T @ sinogram.ravel()).reshape(self.image_shape)

    def loss(self, U, V):
        # Independent diagnostic that does not use the instrumented operators.
        residual = self.matrix @ (U @ V.T).ravel() - self.y.ravel()
        return float(0.5 * np.mean(residual**2))


def fixture(seed=17, rank=2):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(.01, 1, size=(24, 12))
    truth = rng.uniform(0, 1, size=(3, rank)) @ rng.uniform(0, 1, size=(4, rank)).T
    observed = matrix @ truth.ravel() + rng.normal(0, .1, 24)
    problem = DenseTomography(matrix, (3, 4), observed, 4)
    U, V = rng.uniform(.05, 1, (3, rank)), rng.uniform(.05, 1, (4, rank))
    return problem, U, V


def same_state(a, b):
    if isinstance(a, np.ndarray):
        assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same_state(a[key], b[key])
    else:
        assert a == b


def test_anls_accepted_steps_match_off_the_shelf_lbfgsb():
    problem, U, V = fixture()
    config = dict(inner_maxiter=5, relative_kkt_tolerance=1e-7, ftol=0)
    expected_steps = []

    def objective(x):
        factor = x.reshape(U.shape)
        residual = problem.matrix @ (factor @ V.T).ravel() - problem.y.ravel()
        gradient = (problem.matrix.T @ residual).reshape(problem.image_shape) @ V / residual.size
        return .5*np.mean(residual**2), gradient.ravel()

    initial_gradient = objective(U.ravel())[1]
    reference = minimize(objective, U.ravel(), jac=True, method="L-BFGS-B",
        bounds=[(0, None)]*U.size,
        options=dict(maxcor=10, gtol=1e-7*np.max(np.abs(np.minimum(U.ravel(), initial_gradient))),
                     ftol=0, maxiter=5, maxls=20),
        callback=lambda x: expected_steps.append(x.copy()))
    solver = SOLVERS.make_solver("anls", problem, U, V, config)
    actual_steps = []
    while solver.completed_blocks < 1:
        event = solver.step()
        assert event["accepted"]
        actual_steps.append(solver.U.ravel().copy())
    assert len(actual_steps) == len(expected_steps) == 5
    for actual, expected in zip(actual_steps, expected_steps):
        assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    assert_allclose(solver.U.ravel(), reference.x, rtol=3e-13, atol=3e-13)
    assert solver.counters["gradient_calls"] == reference.nfev
    assert solver.counters["forward_calls"] == problem.forward_calls == reference.nfev
    assert solver.counters["backward_calls"] == problem.backward_calls == reference.nfev


def test_anls_block_matches_independent_nnls_and_kkt():
    rng = np.random.default_rng(202)
    matrix = rng.uniform(.01, 1, size=(40, 6))
    V = np.array([[.8], [.3]])
    U = np.ones((3, 1))*.4
    observed = matrix @ (np.array([[1.4], [0.], [.2]]) @ V.T).ravel()
    observed += rng.normal(0, .03, observed.size)
    problem = DenseTomography(matrix, (3, 2), observed, 4)
    design = np.column_stack([matrix @ (np.eye(3)[:, k, None] @ V.T).ravel()
                              for k in range(3)])
    expected, _ = nnls(design, observed)
    solver = SOLVERS.make_solver("anls", problem, U, V,
        dict(inner_maxiter=200, relative_kkt_tolerance=1e-10, ftol=0))
    # advance() observes the block boundary even when convergence adds no iterate.
    for _ in range(5000):
        event = solver.advance()
        if solver.completed_blocks:
            break
    else:
        pytest.fail("The small NNLS block did not terminate")
    assert_allclose(solver.U.ravel(), expected, rtol=1e-7, atol=1e-8)
    gradient = design.T @ (design @ solver.U.ravel() - observed) / observed.size
    assert SOLVERS.projected_gradient_inf(solver.U.ravel(), gradient) < 1e-8


@pytest.mark.parametrize("method", ["anls", "mu"])
def test_budget_preflight_retains_accepted_iterate_and_counts_every_call(method):
    problem, U, V = fixture()
    solver = SOLVERS.make_solver(method, problem, U, V)
    cap = len(problem.train_ids)
    event = solver.step(view_budget=cap)
    assert event["observation_type"] == "budget_stop"
    assert event["iteration"] == 0
    assert_array_equal(solver.U, U)
    assert_array_equal(solver.V, V)
    assert event["backward_views"] == cap
    assert event["forward_views"] == (cap if method == "anls" else 0)
    assert event["forward_calls"] == problem.forward_calls
    assert event["backward_calls"] == problem.backward_calls
    assert event["oracle_calls"] == event["gradient_calls"] + event["rhs_backward_calls"]
    assert solver.step(view_budget=cap) == event


@pytest.mark.parametrize("method", ["anls", "mu"])
def test_exact_resume_inside_operator_pair_and_trial(method):
    problem, U, V = fixture()
    solver = SOLVERS.make_solver(method, problem, U, V,
        dict(inner_maxiter=3) if method == "anls" else None)
    for _ in range(100):
        event = solver.advance(view_budget=400)
        if solver.phase == "backward" and solver.counters["forward_calls"] >= 2:
            break
    else:
        pytest.fail("No pending second forward/adjoint pair")
    if method == "anls":
        # The second evaluation is a line-search trial, still before acceptance.
        assert solver.iteration == 0
        assert not np.array_equal(solver.work["lb"]["x"], U.ravel())
        assert_array_equal(solver.U, U)
    checkpoint = solver.state_dict()
    restored_problem = DenseTomography(problem.matrix, problem.image_shape, problem.y.copy(), 4)
    restored = SOLVERS.make_solver(method, restored_problem, U, V, solver.config)
    restored.load_state_dict(checkpoint)
    same_state(restored.state_dict(), solver.state_dict())
    for _ in range(12):
        assert restored.step(view_budget=400) == solver.step(view_budget=400)
        same_state(restored.state_dict(), solver.state_dict())
    assert checkpoint["counters"] != solver.counters  # deep, independent snapshot


def test_signed_rhs_mu_descent_and_initial_adjoint_accounting():
    problem, U, V = fixture()
    problem.y[:, 0] -= 8  # Genuine signed observations; do not clip the data.
    assert np.any(problem.y < 0)
    solver = SOLVERS.make_solver("mu", problem, U, V)
    losses = [problem.loss(U, V)]
    for k in range(30):
        event = solver.step(view_budget=1000)
        assert event["accepted"] and event["iteration"] == k+1
        losses.append(problem.loss(solver.U, solver.V))
        assert np.min(solver.U) >= 0 and np.min(solver.V) >= 0
    assert np.max(np.diff(losses)) <= 1e-12
    assert losses[-1] < losses[0]
    assert solver.counters["rhs_backward_calls"] == 1
    assert solver.counters["backward_calls"] == 31
    assert solver.counters["forward_calls"] == 30
    assert solver.counters["gradient_calls"] == 30
    assert solver.outer_sweeps == 15


def test_mu_zero_cases_have_no_floor_or_unexplained_reactivation():
    actual = SOLVERS.multiplicative_update(
        np.array([2., 3., 0., 5.]), np.array([0., 2., 4., 3.]),
        np.array([0., -1., 2., 6.]))
    assert_array_equal(actual, [2., 0., 0., 10.])
    with pytest.raises(FloatingPointError, match="zero denominator"):
        SOLVERS.multiplicative_update(np.ones(1), np.zeros(1), np.ones(1))
    problem, U, V = fixture()
    U[0, 0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        SOLVERS.make_solver("mu", problem, U, V)


def test_no_spectral_cap_or_radial_repair_is_applied():
    problem = DenseTomography(np.eye(4), (2, 2), np.ones(4)*100, 2)
    solver = SOLVERS.make_solver("mu", problem, np.ones((2, 1)), np.ones((2, 1)))
    solver.step(view_budget=100)
    assert_array_equal(solver.U, np.full((2, 1), 100.))
    assert np.linalg.norm(solver.U, 2) > 100


def test_checkpoint_rejects_changed_scipy_and_accounting():
    problem, U, V = fixture()
    solver = SOLVERS.make_solver("anls", problem, U, V)
    checkpoint = solver.state_dict()
    wrong_version = deepcopy(checkpoint)
    wrong_version["scipy_version"] = "0.0"
    with pytest.raises(ValueError, match="SciPy"):
        solver.load_state_dict(wrong_version)
    wrong_counts = deepcopy(checkpoint)
    wrong_counts["counters"]["forward_views"] = 1
    with pytest.raises(ValueError, match="counters"):
        solver.load_state_dict(wrong_counts)


@pytest.mark.parametrize("method", ["anls", "mu"])
def test_budget_stop_can_resume_without_restarting_algorithm(method):
    problem, U, V = fixture()
    config = dict(inner_maxiter=3) if method == "anls" else {}
    stopped = SOLVERS.make_solver(method, problem, U, V, config)
    first_cap, final_cap = len(problem.train_ids), 12*len(problem.train_ids)
    assert stopped.step(view_budget=first_cap)["observation_type"] == "budget_stop"
    resumed = SOLVERS.make_solver(method, problem, U, V, config)
    resumed.load_state_dict(stopped.state_dict())
    fresh = SOLVERS.make_solver(method, problem, U, V, config)
    for solver in (resumed, fresh):
        for _ in range(100):
            if solver.step(view_budget=final_cap)["observation_type"] == "budget_stop":
                break
        else:
            pytest.fail("Did not exhaust the prescribed small budget")
    same_state(resumed.state_dict(), fresh.state_dict())


def test_bad_lbfgsb_restart_array_is_rejected_before_calling_c():
    problem, U, V = fixture()
    solver = SOLVERS.make_solver("anls", problem, U, V)
    solver.advance(view_budget=100)
    checkpoint = solver.state_dict()
    checkpoint["work"]["lb"]["wa"] = np.zeros(1)
    with pytest.raises(ValueError, match="restart array"):
        solver.load_state_dict(checkpoint)


def test_projected_gradient_norm_matches_native_lbfgsb_stopping_norm():
    problem = DenseTomography(np.eye(4), (2, 2), -np.ones(4), 2)
    U, V = np.ones((2, 1))*.1, np.ones((2, 1))
    solver = SOLVERS.make_solver("anls", problem, U, V)
    solver.advance()  # initial forward
    solver.advance()  # initial adjoint
    initial = solver.work["lb"]["initial_kkt"]
    solver.advance()  # native solver computes its projected norm and proposes a trial
    assert initial == pytest.approx(.1)
    assert solver.work["lb"]["dsave"][12] == pytest.approx(initial)
    assert solver.work["lb"]["pgtol"] == pytest.approx(1e-3*initial)


@pytest.mark.parametrize("first_reason, expected", [("line_search_failed", "stalled"),
                                                   ("kkt", "converged")])
def test_both_no_update_blocks_need_kkt_for_global_convergence(first_reason, expected):
    problem, U, V = fixture()
    solver = SOLVERS.make_solver("anls", problem, U, V)
    solver._start_block()
    solver._end_block(first_reason)
    solver._start_block()
    solver._end_block("kkt")
    assert solver.status == expected
    assert solver.block_termination_counts[first_reason] >= 1
    assert sum(solver.block_termination_counts.values()) == 2
