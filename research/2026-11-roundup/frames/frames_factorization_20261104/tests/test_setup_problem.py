import unittest

import numpy as np
from numpy.testing import assert_allclose

from core import FactorizationProblem
from setup_problem import initialize, make_problem_config


class SetupTests(unittest.TestCase):
    def test_svd_floor_and_spectral_caps_known_diagonal(self):
        A = np.diag([9., 4., 1.])
        config = make_problem_config(A, 2)
        self.assertAlmostEqual(config["rank_r_loss_lower_bound"], 1/18)
        self.assertAlmostEqual(config["RU"], 3.75)
        self.assertAlmostEqual(config["RV"], 3.75)
        self.assertFalse(config["columns_unit_norm"])
        U = np.array([[3., 0.], [0., 2.], [0., 0.]])
        V = U.copy()
        p = FactorizationProblem(A, 2, config["RU"], config["RV"], config["l1_radius"])
        self.assertLess(np.linalg.norm(U, 2), p.RU)
        self.assertLess(np.linalg.norm(V, 2), p.RV)
        self.assertAlmostEqual(p.objective(U,V), config["rank_r_loss_lower_bound"])

    def test_initialization_reproducible_nonzero_full_column_rank_feasible(self):
        rng = np.random.default_rng(91)
        A = rng.uniform(0,1,size=(12,8))
        A /= np.linalg.norm(A,axis=0)
        config = make_problem_config(A, 3)
        self.assertTrue(config["columns_unit_norm"])
        p = FactorizationProblem(A, 3, config["RU"], config["RV"], config["l1_radius"])
        U,V = initialize(p,17)
        againU,againV = initialize(p,17)
        assert_allclose(U,againU,rtol=0,atol=0)
        assert_allclose(V,againV,rtol=0,atol=0)
        self.assertTrue(p.feasibility(U,V)["feasible"])
        self.assertEqual(np.linalg.matrix_rank(U),p.rank)
        self.assertEqual(np.linalg.matrix_rank(V),p.rank)
        self.assertTrue(np.all(U>0))
        self.assertTrue(np.all(V>0))
        self.assertLess(p.objective(U,V),config["zero_prediction_loss"])
        otherU,otherV=initialize(p,18)
        self.assertFalse(np.allclose(U,otherU))
        self.assertFalse(np.allclose(V,otherV))

    def test_full_smoothness_bound_for_complete_objective(self):
        rng=np.random.default_rng(12)
        A=rng.normal(size=(7,5))
        config=make_problem_config(A,3)
        p=FactorizationProblem(A,3,config["RU"],config["RV"],config["l1_radius"])
        U,V=p.project_C(rng.normal(size=(7,3))*3,rng.normal(size=(5,3))*3)
        X,Y=p.project_C(rng.normal(size=(7,3))*3,rng.normal(size=(5,3))*3)
        gU,gV=p.full_grad(U,V)
        hU,hV=p.full_grad(X,Y)
        norm=lambda B,C:np.sqrt(np.sum(B*B)+np.sum(C*C))
        self.assertLessEqual(norm(gU-hU,gV-hV),config["L_full_bound"]*norm(U-X,V-Y)+1e-12)


if __name__ == "__main__":
    unittest.main()
