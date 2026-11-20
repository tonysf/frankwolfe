"""Small independent mathematical checks for the factorization experiment."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from core import FactorizationProblem, pair_norm_sq, project_l1_ball, project_spectral_ball, spectral_lmo


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(14)
        self.problem = FactorizationProblem(np.array([[1., 3., -2.], [4., 0.5, 2.],
                                                      [0.4, -1., 3.], [2., 0., 1.]]),
                                            rank=2, RU=1.7, RV=1.3, l1_radius=1.1)
        self.U = self.rng.normal(size=(4, 2)) * .3
        self.V = self.rng.normal(size=(3, 2)) * .3

    def test_full_gradient_directional_difference_and_scalar_loss(self):
        p = self.problem
        expected = sum((sum(self.U[i, k] * self.V[j, k] for k in range(2)) - p.M[i, j]) ** 2
                       for i in range(p.m) for j in range(p.n)) / (2 * p.N)
        self.assertAlmostEqual(p.objective(self.U, self.V), expected, places=13)
        dU, dV = self.rng.normal(size=self.U.shape), self.rng.normal(size=self.V.shape)
        gU, gV = p.full_grad(self.U, self.V)
        h = 1e-6
        numerical = (p.objective(self.U + h*dU, self.V + h*dV)
                     - p.objective(self.U - h*dU, self.V - h*dV)) / (2*h)
        assert_allclose(numerical, np.sum(gU*dU) + np.sum(gV*dV), rtol=1e-8, atol=1e-8)

    def test_image_oracle_exact_expectation_and_duplicate_weights(self):
        p = self.problem
        grads = [p.sampled_grad_columns(self.U, self.V, [j]) for j in range(p.n)]
        mean = tuple(sum(g[k] for g in grads) / p.n for k in range(2))
        for actual, target in zip(mean, p.full_grad(self.U, self.V)):
            assert_allclose(actual, target, rtol=1e-14, atol=1e-14)
        repeated = p.sampled_grad_columns(self.U, self.V, [2, 2, 0, 2])
        for k in range(2):
            assert_allclose(repeated[k], (3*grads[2][k] + grads[0][k]) / 4,
                            rtol=1e-14, atol=1e-14)
        # Check one component against its own scalar finite difference.
        j, h = 1, 1e-6
        dU, dV = self.rng.normal(size=self.U.shape), self.rng.normal(size=self.V.shape)
        def value(U, V):
            return np.sum((U @ V[j] - p.M[:, j])**2) / (2*p.m)
        numerical = (value(self.U+h*dU, self.V+h*dV)-value(self.U-h*dU,self.V-h*dV))/(2*h)
        assert_allclose(numerical, np.sum(grads[j][0]*dU)+np.sum(grads[j][1]*dV), atol=1e-8)

    def test_entry_oracle_exact_expectation_including_duplicates(self):
        p = self.problem
        rows, cols = np.indices(p.M.shape)
        actual = p.sampled_grad_indices(self.U, self.V, rows.ravel(), cols.ravel())
        for value, target in zip(actual, p.full_grad(self.U, self.V)):
            assert_allclose(value, target, atol=1e-14)
        single = p.sampled_grad_indices(self.U, self.V, [1], [2])
        repeated = p.sampled_grad_indices(self.U, self.V, [1, 1, 1], [2, 2, 2])
        for value, target in zip(single, repeated):
            assert_allclose(value, target, atol=1e-14)

    def test_spectral_lmo_dual_certificate_rectangular_and_zero(self):
        for shape in ((5, 2), (2, 5), (3, 3)):
            G = self.rng.normal(size=shape)
            radius = 1.4
            S = spectral_lmo(G, radius)
            singular = np.linalg.svd(G, compute_uv=False)
            assert_allclose(np.sum(G*S), -radius*singular.sum(), atol=1e-12)
            self.assertLessEqual(np.linalg.norm(S, 2), radius+1e-12)
        assert_allclose(spectral_lmo(np.zeros((3, 2)), 2), 0)

    def test_spectral_projection_variational_inequality(self):
        A = self.rng.normal(size=(5, 3)) * 3
        P = project_spectral_ball(A, 1.2)
        self.assertLessEqual(np.linalg.norm(P, 2), 1.2+1e-12)
        # Projection iff <A-P,Z-P> <= 0 for every feasible Z.
        for _ in range(25):
            Z = self.rng.normal(size=A.shape)
            Z *= self.rng.uniform(0, 1.2) / np.linalg.norm(Z, 2)
            self.assertLessEqual(np.sum((A-P)*(Z-P)), 1e-11)

    def test_spectral_lmo_leaves_exact_nullspace_zero_and_certifies_truncation(self):
        G = np.zeros((6, 3))
        G[1, 2] = 3.0
        S, cert = spectral_lmo(G, 2.0, return_certificate=True)
        expected = np.zeros_like(G)
        expected[1, 2] = -2.0
        assert_allclose(S, expected, atol=1e-14)
        self.assertEqual(cert["rank"], 1)
        self.assertAlmostEqual(np.linalg.norm(S), 2.0)
        G[0, 0] = 1e-17
        S, cert = spectral_lmo(G, 2.0, return_certificate=True)
        self.assertEqual(cert["rank"], 1)
        self.assertGreaterEqual(cert["support_gap_bound"], 2e-17*(1-1e-12))
        self.assertLessEqual(cert["support_gap_bound"], 2e-17*(1+1e-12))

    def test_l1_projection_kkt_and_D_projection_variational_inequality(self):
        A = np.array([[-3., 2.5, .4], [1.2, -.1, 0.]])
        P = project_l1_ball(A, 2.0)
        self.assertAlmostEqual(np.abs(P).sum(), 2.0, places=12)
        active = np.abs(P) > 0
        threshold = np.abs(A[active]) - np.abs(P[active])
        assert_allclose(threshold, threshold[0], atol=1e-14)
        self.assertTrue(np.all(np.abs(A[~active]) <= threshold[0]+1e-14))
        p = self.problem
        U, V = self.U*9, self.V*9
        pU, pV = p.project_D(U, V)
        for _ in range(25):
            zU = np.abs(self.rng.normal(size=U.shape))
            zV = self.rng.normal(size=V.shape)
            zV *= self.rng.uniform(0, p.l1_radius) / np.abs(zV).sum()
            self.assertLessEqual(np.sum((U-pU)*(zU-pU))+np.sum((V-pV)*(zV-pV)), 1e-11)

    def test_repair_satisfies_joint_constraints_without_mutating_inputs(self):
        for nonnegative in (False, True):
            p = FactorizationProblem(self.problem.M, 2, 0.7, 0.8, 1.1, nonnegative)
            U, V = self.U*100, self.V*100
            beforeU, beforeV = U.copy(), V.copy()
            fixedU, fixedV = p.repair(U, V)
            self.assertTrue(p.feasibility(fixedU, fixedV)["feasible"])
            assert_allclose(U, beforeU)
            assert_allclose(V, beforeV)
            if not nonnegative:
                self.assertTrue(np.any(fixedV < 0))
            else:
                self.assertTrue(np.all(fixedV >= 0))

    def test_true_smoothed_gradient_and_gap_dual_formula(self):
        p, mu = self.problem, .7
        U, V = p.project_C(self.U*3, self.V*3)
        dU, dV = self.rng.normal(size=U.shape), self.rng.normal(size=V.shape)
        gU, gV = p.smoothed_gradient(U, V, mu)
        h = 1e-6
        numerical = (p.smoothed_objective(U+h*dU,V+h*dV,mu)
                     -p.smoothed_objective(U-h*dU,V-h*dV,mu))/(2*h)
        assert_allclose(numerical, np.sum(gU*dU)+np.sum(gV*dV), atol=2e-8)
        expected_gap = (np.sum(gU*U)+np.sum(gV*V)
                        +p.RU*np.linalg.svd(gU,compute_uv=False).sum()
                        +p.RV*np.linalg.svd(gV,compute_uv=False).sum())
        assert_allclose(p.smoothed_fw_gap(U,V,mu), expected_gap, atol=1e-12)

    def test_certified_component_gradient_and_smoothness_bounds(self):
        p = self.problem
        bounds = p.certificates()
        U, V = p.project_C(self.U*5, self.V*5)
        W, Z = p.project_C(self.rng.normal(size=U.shape)*5,self.rng.normal(size=V.shape)*5)
        distance = np.sqrt(pair_norm_sq(U-W,V-Z))
        for j in range(p.n):
            gU,gV = p.sampled_grad_columns(U,V,[j])
            hU,hV = p.sampled_grad_columns(W,Z,[j])
            self.assertLessEqual(pair_norm_sq(gU,gV),bounds["image_gradient_squared_bound"]+1e-10)
            self.assertLessEqual(np.sqrt(pair_norm_sq(gU-hU,gV-hV)),bounds["image_L_bound"]*distance+1e-10)

    def test_feasible_nonconvexity_witness_and_zero_stationarity(self):
        p = FactorizationProblem(np.ones((2, 2)), 1, 1, 1, 1)
        U0,V0 = np.zeros((2,1)),np.zeros((2,1))
        U1,V1 = np.full((2,1),.2),np.full((2,1),.2)
        self.assertTrue(p.feasibility(U1,V1)["feasible"])
        self.assertGreater(p.objective(U1/2,V1/2),(p.objective(U0,V0)+p.objective(U1,V1))/2)
        assert_allclose(p.full_grad(U0,V0),0)


if __name__ == "__main__":
    unittest.main()
