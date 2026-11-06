"""Independent projection solves and estimator/checkpoint regression checks."""
import math

import cvxpy as cp
import numpy as np
import pytest

from baseline import (ProjectionFailure, SpringSARAH, project_intersection,
                      project_l1, theorem_parameters)
from core import FactorizationProblem


@pytest.mark.parametrize("kind", ["nonnegative", "l1"])
def test_intersection_against_independent_conic_projection(kind):
    rng = np.random.default_rng(20261104)
    z = rng.normal(size=(5, 3)) * 2
    radius, budget = 1.1, 3.5
    x, cert = project_intersection(z, radius, kind=kind, l1_radius=budget,
                                    distance_tolerance=1e-7)
    variable = cp.Variable(z.shape)
    constraints = [cp.norm(variable, 2) <= radius]
    constraints += [variable >= 0] if kind == "nonnegative" else [cp.norm1(variable) <= budget]
    model = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(variable - z)), constraints)
    model.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_gap_rel=1e-11,
                tol_feas=1e-11, max_iter=500)
    assert model.status in ("optimal", "optimal_inaccurate")
    primal = 0.5 * np.sum((x - z) ** 2)
    assert primal == pytest.approx(model.value, abs=2e-7)
    assert np.linalg.norm(x - variable.value) < 1e-4
    assert cert.dual_objective <= model.value + 2e-8
    assert primal - model.value <= cert.duality_gap + 2e-8
    assert cert.primal_feasibility <= 1e-12
    assert np.linalg.norm(x, 2) <= radius + 1e-12
    if kind == "nonnegative":
        assert np.min(x) >= 0
    else:
        assert np.abs(x).sum() <= budget + 1e-12
        assert np.any(x < 0)  # V is signed, not silently nonnegative.


def test_interior_point_is_preserved_and_l1_is_entrywise():
    z = np.array([[0.1, 0.2], [0.3, 0.1]])
    x, cert = project_intersection(z, 2.0)
    np.testing.assert_array_equal(x, z)
    assert cert.duality_gap == 0
    signed = np.array([[2., -2.], [1., -1.]])
    np.testing.assert_allclose(project_l1(signed, 2), [[1, -1], [0, 0]])


def test_tiny_active_l1_step_uses_exact_single_projection_reduction():
    # Near-boundary optimizer steps can have correction energy many orders
    # below input energy. Their certificate floor must include subtraction error
    # in q=z-proj_l1(z), rather than spin until the Dykstra iteration cap.
    z = np.random.default_rng(11).uniform(0.1, 0.8, size=(5, 2))
    budget = float(np.sum(np.abs(z))) - 2e-5
    expected = project_l1(z, budget)
    x, cert = project_intersection(z, 10, kind="l1", l1_radius=budget,
                                   distance_tolerance=1e-10, max_iterations=1)
    np.testing.assert_array_equal(x, expected)
    assert cert.iterations == 1
    assert cert.method == "single_side_projection"
    assert cert.status == "numerical_floor"
    assert cert.attained_distance_bound > cert.requested_distance
    assert cert.roundoff_allowance >= 64 * np.finfo(float).eps * np.sum(z * z)


def test_spectral_only_projection_reduction():
    z = 3 * np.outer([1., 2., 3.], [1., 2.])
    x, cert = project_intersection(z, 1, kind="nonnegative", max_iterations=1)
    expected = z / np.linalg.norm(z, 2)
    np.testing.assert_allclose(x, expected, atol=1e-14)
    assert cert.method == "single_spectral_projection"
    assert cert.primal_feasibility == 0


def test_cap_failure_is_explicit_and_floor_is_not_exact_theorem():
    z = np.array([[2., -1., 0.7], [-1., 3., 2.], [1., -2., 4.]])
    with pytest.raises(ProjectionFailure):
        project_intersection(z, 1.0, max_iterations=1, distance_tolerance=1e-12)
    _, cert = project_intersection(z, 1.0, distance_tolerance=1e-14, iteration=100)
    assert cert.status == "numerical_floor"
    assert cert.attained_distance_bound > cert.requested_distance
    assert cert.exact_prox_theorem is False


def test_literal_corollary_constants():
    params = theorem_parameters(400, 2.0, 1.0, 1.5)
    assert params["step_u"] == 1 / (4 * math.sqrt(30 * 400))
    assert params["step_v"] == params["step_u"]
    assert params["refresh_probability"] == 1 / 400
    assert params["V1"] == 8.0
    assert params["rho"] == 1 / 400


def make_small(seed=9, batch=2, n=4, mode="theorem"):
    rng = np.random.default_rng(8)
    M = rng.normal(size=(5, n))
    problem = FactorizationProblem(M, 2, 3.0, 3.0, 20.0)
    U = rng.uniform(0.1, 0.3, size=(5, 2))
    V = rng.uniform(-0.1, 0.1, size=(n, 2))
    Mbound = (3 ** 2 + 3 ** 2 + 3 * 3 + np.linalg.norm(M, axis=0).max()) / 5
    solver = SpringSARAH(U, V, full_gradient=problem.full_grad,
                         batch_gradient=problem.sampled_grad_columns,
                         n_components=n, radius_u=3, radius_v=3, l1_radius=20,
                         component_lipschitz=Mbound, block_lipschitz_u=9 / 5,
                         block_lipschitz_v=9 / 5, batch_size=batch, seed=seed, mode=mode)
    return problem, solver


def test_full_batch_recursion_matches_alternating_exact_gradient_updates():
    problem, solver = make_small(batch=4)
    U, V = solver.U.copy(), solver.V.copy()
    # With the complete subset, every recursive difference telescopes exactly,
    # including the mixed point (U_{k+1},V_k) for the second block.
    for _ in range(4):
        gu, _ = problem.full_grad(U, V)
        U, _ = project_intersection(U - solver.parameters["step_u"] * gu, 3)
        _, gv = problem.full_grad(U, V)
        V, _ = project_intersection(V - solver.parameters["step_v"] * gv, 3,
                                   kind="l1", l1_radius=20)
        solver.step()
        np.testing.assert_allclose(solver.U, U, rtol=0, atol=1e-13)
        np.testing.assert_allclose(solver.V, V, rtol=0, atol=1e-13)


def test_checkpoint_preserves_randomness_and_momentum():
    _, solver = make_small()
    for _ in range(3):
        solver.step()
    state = solver.state_dict()
    _, resumed = make_small(seed=450)
    resumed.load_state_dict(state)
    for _ in range(5):
        actual, expected = solver.step(), resumed.step()
        np.testing.assert_array_equal(solver.U, resumed.U)
        np.testing.assert_array_equal(solver.V, resumed.V)
        assert actual == expected
    _, wrong = make_small(batch=1)
    with pytest.raises(ValueError):
        wrong.load_state_dict(state)


def test_fresh_block_subsets_and_oracle_accounting():
    problem, solver = make_small()
    first = solver.step()
    assert first["step_oracle_calls"] == 2 * problem.n
    assert first["refresh_u"] and first["refresh_v"]
    # An injected huge refresh period is allowed only for the explicitly
    # heuristic control and forces recursive updates in this deterministic test.
    solver.parameters["refresh_probability"] = 0
    seen = []
    def draw(rng, b):
        chosen = rng.choice(problem.n, b, replace=False)
        seen.append(chosen)
        return chosen
    solver.sample_indices = draw
    second = solver.step()
    assert second["step_oracle_calls"] == 4 * solver.batch_size
    assert len(seen) == 2
    assert all(len(np.unique(item)) == solver.batch_size for item in seen)
