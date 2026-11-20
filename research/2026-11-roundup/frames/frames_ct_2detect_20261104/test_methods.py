import numpy as np
from numpy.testing import assert_array_equal
import pytest

from methods import make_solver


class TinyProblem:
    train_ids = np.arange(3, 12)

    def __init__(self):
        self.queries = []

    def grad(self, U, V, ids, *, block=None):
        self.queries.append((block, tuple(ids)))
        weights = np.asarray(ids)[:, None, None] / 12
        residual = (weights * (U @ V.T) - 0.2) * weights
        image_gradient = residual.mean(axis=0)
        if block == 0:
            return image_gradient @ V
        if block == 1:
            return image_gradient.T @ U
        return image_gradient @ V, image_gradient.T @ U


U = np.array([[0.2, 0.1], [0.3, 0.15]])
V = np.array([[0.1, 0.25], [0.15, 0.1], [0.3, 0.2]])
CAPS = (0.5, 0.6)
STEPS = {"eta_u": 0.6, "eta_v": 0.7}
CONFIGS = {
    "sframes": {"batch": 3, "gamma0": 0.5, "rho0": 0.4, "mu0": 0.2},
    "spring_sarah": {**STEPS, "batch": 3, "refresh_period": 3},
    "prox_sarah": {
        "batch": 3, "snapshot_batch": 7, "inner_steps": 3, "eta": 0.7,
        "relaxation": 0.7,
    },
    "palm": STEPS,
}


def run_one(solver, problem, method):
    problem.queries.clear()
    row = solver.step()
    queries = problem.queries.copy()
    assert row["gradient_calls"] == len(queries)
    assert row["forward_views"] == row["backward_views"] == sum(len(ids) for _, ids in queries)
    if method == "spring_sarah":
        n_u = 1 if row["refresh_u"] else 2
        n_v = 1 if row["refresh_v"] else 2
        assert [block for block, _ in queries] == [0] * n_u + [1] * n_v
        if n_u == 2:
            assert queries[0][1] == queries[1][1]
        if n_v == 2:
            assert queries[-2][1] == queries[-1][1]
    elif method == "palm":
        assert queries == [(0, tuple(problem.train_ids)), (1, tuple(problem.train_ids))]
    else:
        assert all(block is None for block, _ in queries)
        if method == "prox_sarah":
            assert row["forward_views"] == (7 if row["refresh"] else 6)
            if not row["refresh"]:
                assert queries[0][1] == queries[1][1]
    return row, queries


@pytest.mark.parametrize("method", CONFIGS)
def test_restart_preserves_recursive_history_and_oracle_counts(method):
    p = TinyProblem()
    a = make_solver(method, p, U, V, CAPS, seed=1, config=CONFIGS[method])
    b = make_solver(method, p, U, V, CAPS, seed=2, config=CONFIGS[method])
    for _ in range(5):
        run_one(a, p, method)
    b.load_state_dict(a.state_dict())
    for _ in range(8):
        assert run_one(a, p, method) == run_one(b, p, method)
        assert_array_equal(a.U, b.U)
        assert_array_equal(a.V, b.V)
    for factor, cap in zip((a.U, a.V), CAPS):
        assert np.linalg.norm(factor) <= cap * (1 + 1e-12)
        if method != "sframes":
            assert np.min(factor) >= 0


def test_always_refreshing_spring_matches_palm():
    p = TinyProblem()
    spring = make_solver(
        "spring_sarah", p, U, V, CAPS, seed=2,
        config={**STEPS, "batch": 3, "refresh_period": 1},
    )
    palm = make_solver("palm", p, U, V, CAPS, seed=2, config=STEPS)
    for _ in range(7):
        spring.step()
        palm.step()
        assert_array_equal(spring.U, palm.U)
        assert_array_equal(spring.V, palm.V)
