"""Bounded independent-reference checks for the published ProxSARAH-v3 recipe.

No CT data, tomography backend, or production run is used. The rank-one
test has an exact orthant/ball projection, so its reference trajectory does
not reuse the numerical projection or SARAH implementation under test.
"""

from copy import deepcopy
import math
import unittest

import numpy as np

from prox_paper import make_solver, parameters


class DenseFactoredProblem:
    def __init__(self, n=17, objective_scale=1.0):
        rng = np.random.default_rng(4321)
        self.A = rng.normal(size=(n, 2, 3, 2))
        target = np.array([[3.0, -4.0], [-2.0, 5.0], [1.0, 1.5]])
        self.y = np.einsum("bdij,ij->bd", self.A, target)
        self.train_ids = rng.permutation(n)
        self.objective_scale = objective_scale
        self.queries = []

    def grad(self, U, V, ids, *, block=None):
        if block is not None:
            raise AssertionError("ProxSARAH must query the joint gradient")
        self.queries.append((U.copy(), V.copy(), np.array(ids, copy=True)))
        A = self.A[ids]
        residual = np.einsum("bdij,ij->bd", A, U @ V.T) - self.y[ids]
        G = np.einsum("bd,bdij->ij", residual, A) / (
            len(ids) * A.shape[1] * self.objective_scale
        )
        return G @ V, G.T @ U


def exact_gradient(problem, U, V, ids):
    """Independent loop implementation of the finite-sum factor gradient."""
    gu, gv = np.zeros_like(U), np.zeros_like(V)
    for i in ids:
        for a, y in zip(problem.A[i], problem.y[i]):
            residual = np.sum(a * (U @ V.T)) - y
            gu += residual * (a @ V)
            gv += residual * (a.T @ U)
    denominator = len(ids) * problem.A.shape[1] * problem.objective_scale
    return gu / denominator, gv / denominator


def exact_rank_one_projection(x, cap):
    z = np.maximum(x, 0.0)
    norm = np.linalg.norm(z)
    if norm > cap:
        z = z * (cap / norm)
    return z


class PublishedVariantChecks(unittest.TestCase):
    def setUp(self):
        self.U = np.array([[0.25], [0.20], [0.30]])
        self.V = np.array([[0.25], [0.35]])
        self.caps = (0.8, 0.7)
        self.seed = 91
        self.loss_scale = 0.7

    def solver(self, problem=None, *, loss_scale=None, config=None):
        return make_solver(
            DenseFactoredProblem() if problem is None else problem,
            self.U, self.V, self.caps, seed=self.seed,
            config=config if config is not None else {
                "loss_scale": self.loss_scale if loss_scale is None else loss_scale},
        )

    def test_published_parameters_and_epoch_work(self):
        for n, m, batch, views in ((360, 18, 27, 1332), (1080, 32, 48, 4152)):
            with self.subTest(n=n):
                p = parameters(n, 0.25)
                self.assertEqual(p["inner_steps"], m)
                self.assertEqual(p["batch"], batch)
                self.assertEqual(p["snapshot_batch"], n)
                self.assertEqual(p["updates_per_epoch"], m + 1)
                self.assertEqual(n + 2 * batch * m, views)
                self.assertEqual(p["gamma"], 0.99)
                self.assertEqual(p["relaxation"], 0.99)
                self.assertEqual(p["eta"], (2 / 4.99) / 0.25)
                # Literal experimental v3 deliberately differs from eq. (36).
                finite_n_batch = math.floor(m * n / (p["C"] * n + m - p["C"]))
                self.assertNotEqual(batch, finite_n_batch)

    def test_full_trajectory_against_exact_independent_reference(self):
        problem = DenseFactoredProblem()
        solver = self.solver(problem)
        rng = np.random.default_rng(self.seed)
        U, V = self.U.copy(), self.V.copy()
        eta, gamma = (2 / 4.99) / self.loss_scale, 0.99
        n = len(problem.train_ids)
        m, batch = math.isqrt(n), math.floor(math.sqrt(n) * 3 * 0.99**2 / 2)
        previous = estimate = None
        orthant_active = spectral_active = False
        for k in range(27):
            query = U.copy(), V.copy()
            snapshot = k % (m + 1) == 0
            offset = len(problem.queries)
            if snapshot:
                ids = problem.train_ids
                estimate = exact_gradient(problem, U, V, ids)
            else:
                ids = rng.choice(problem.train_ids, batch, replace=False)
                current = exact_gradient(problem, U, V, ids)
                prior = exact_gradient(problem, *previous, ids)
                estimate = tuple(old + new - prev for old, new, prev in zip(
                    estimate, current, prior))
            zU, zV = U - eta * estimate[0], V - eta * estimate[1]
            orthant_active |= bool(np.any(zU < 0) or np.any(zV < 0))
            spectral_active |= any(np.linalg.norm(np.maximum(z, 0)) > cap
                                   for z, cap in zip((zU, zV), self.caps))
            U = (1 - gamma) * U + gamma * exact_rank_one_projection(zU, self.caps[0])
            V = (1 - gamma) * V + gamma * exact_rank_one_projection(zV, self.caps[1])
            details = solver.step()
            np.testing.assert_allclose(solver.U, U, rtol=3e-12, atol=3e-12)
            np.testing.assert_allclose(solver.V, V, rtol=3e-12, atol=3e-12)
            self.assertEqual(details["refresh"], snapshot)
            self.assertEqual(details["epoch_position"], k % (m + 1))
            expected_queries = [query] if snapshot else [query, previous]
            new_queries = problem.queries[offset:]
            self.assertEqual(len(new_queries), len(expected_queries))
            for actual, expected in zip(new_queries, expected_queries):
                np.testing.assert_array_equal(actual[2], ids)
                np.testing.assert_allclose(actual[0], expected[0], rtol=3e-12, atol=3e-12)
                np.testing.assert_allclose(actual[1], expected[1], rtol=3e-12, atol=3e-12)
            expected_views = n if snapshot else 2 * batch
            for key in ("forward_views", "backward_views", "component_gradient_calls"):
                self.assertEqual(details[key], expected_views)
            for key in ("gradient_calls", "oracle_calls"):
                self.assertEqual(details[key], 1 if snapshot else 2)
            for key in ("curvature_operator_calls", "curvature_forward_views",
                        "curvature_backward_views"):
                self.assertEqual(details[key], 0)
            self.assertEqual(details["projection_calls"], 2)
            for factor, cap in ((solver.U, self.caps[0]), (solver.V, self.caps[1])):
                self.assertTrue(np.all(factor >= 0))
                self.assertLessEqual(np.linalg.norm(factor), cap * (1 + 1e-12))
            previous = query
        self.assertTrue(orthant_active, "fixture must exercise orthant projection")
        self.assertTrue(spectral_active, "fixture must exercise spectral-cap projection")

    def test_original_and_normalized_objectives_are_equivalent(self):
        original = self.solver(DenseFactoredProblem(), loss_scale=2.7)
        normalized = self.solver(DenseFactoredProblem(objective_scale=2.7), loss_scale=1.0)
        for _ in range(23):
            left, right = original.step(), normalized.step()
            np.testing.assert_allclose(original.U, normalized.U, rtol=3e-12, atol=3e-12)
            np.testing.assert_allclose(original.V, normalized.V, rtol=3e-12, atol=3e-12)
            for key in ("refresh", "gradient_calls", "oracle_calls", "forward_views",
                        "backward_views", "projection_calls"):
                self.assertEqual(left[key], right[key])

    def test_resume_restores_recursive_estimator_and_rng_across_snapshot(self):
        original = self.solver()
        for _ in range(3):
            original.step()
        checkpoint = original.state_dict()
        resumed = self.solver()
        resumed.load_state_dict(checkpoint)
        # Also check that loading does not retain aliases to the supplied state.
        checkpoint["U"][:] = 999.0
        checkpoint["est_u"][:] = 999.0
        for _ in range(14):
            self.assertEqual(original.step(), resumed.step())
            np.testing.assert_array_equal(original.U, resumed.U)
            np.testing.assert_array_equal(original.V, resumed.V)
            np.testing.assert_array_equal(original.est_u, resumed.est_u)
            np.testing.assert_array_equal(original.est_v, resumed.est_v)
        wrong = deepcopy(original.state_dict())
        wrong["config"]["loss_scale"] *= 2
        with self.assertRaises(ValueError):
            resumed.load_state_dict(wrong)

    def test_invalid_scales_and_arbitrary_parameter_overrides_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                self.solver(loss_scale=value)
        for override in ({}, {"loss_scale": 1, "batch": 64},
                         {"loss_scale": 1, "snapshot_batch": 4},
                         {"loss_scale": 1, "relaxation": 0.8},
                         {"loss_scale": 1, "inner_steps": 96},
                         {"loss_scale": 1, "eta": 1}):
            with self.subTest(config=override), self.assertRaises(ValueError):
                self.solver(config=override)
        for n in (0, -1, True, 1.5):
            with self.subTest(n=n), self.assertRaises(ValueError):
                parameters(n, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
