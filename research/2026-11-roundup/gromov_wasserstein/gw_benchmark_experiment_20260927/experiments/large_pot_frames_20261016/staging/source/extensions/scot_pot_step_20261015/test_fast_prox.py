"""Parity, safety and total-budget tests for the optional CPU Sinkhorn path."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import ot
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fast_prox import FastCertifiedKLProx
spec = importlib.util.spec_from_file_location("pot_scale_fast_test_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    A = torch.tensor([[0., .035, .1], [.035, 0., .07], [.1, .07, 0.]], dtype=torch.float64)
    B = torch.tensor([[0., .06, .085], [.06, 0., .02], [.085, .02, 0.]], dtype=torch.float64)
    initial = torch.tensor([[.12, .03, .05], [.03, .17, .1], [.15, .2, .15]], dtype=torch.float64)
    problem = runner.RowShardedGW(A, B, initial.sum(1), initial.sum(0), exact_block_rows=2)
    return problem, initial, (problem.a[:, None] * problem.b[None, :]).log()


class TestFastCertifiedKLProx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matches_original_log_domain_plan_and_objective(self):
        problem, plan, logref = fixture()
        gradient = problem.gradient(plan)
        fast = FastCertifiedKLProx(problem, gap_tol=1e-11, certificate_block_rows=2)
        base = runner.CertifiedKLProx(problem, gap_tol=1e-11, certificate_block_rows=2)
        actual = fast(logref, gradient, eta=1000., tol=1e-13, check_every=1)
        expected = base(logref, gradient, eta=1000., tol=1e-13, check_every=1)
        self.assertTrue(actual[2]["converged"])
        self.assertEqual(actual[2]["sinkhorn_path"], "positive-kernel-matrix-vector")
        torch.testing.assert_close(actual[0], expected[0], rtol=0., atol=3e-13)
        self.assertAlmostEqual(actual[2]["primal_objective"], expected[2]["primal_objective"], places=13)

    def test_passes_production_independent_certificate_acceptance(self):
        problem, plan, logref = fixture()
        prox = FastCertifiedKLProx(problem, certificate_block_rows=2)
        Q, _, info = runner.pot_step(logref, problem.gradient(plan), iteration=0,
                                     epsilon=.001, prox=prox)
        self.assertLessEqual(info["normalized_gap_upper_bound"], 1e-7)
        self.assertLessEqual(prox.marginal_residuals(Q)["both_l1"], 1e-7)
        self.assertEqual(info["eta"], 1000.)

    def test_multiple_outer_updates_match_pot_pgd(self):
        problem, initial, logref = fixture()
        prox = FastCertifiedKLProx(problem, gap_tol=1e-11, certificate_block_rows=2)
        plan = initial.clone()
        for iteration in range(4):
            plan, _, _ = runner.pot_step(logref, problem.gradient(plan), iteration=iteration,
                                         epsilon=.001, prox=prox, inner_tol=1e-13,
                                         gap_tol=1e-11, check_every=1)
        expected = ot.gromov.entropic_gromov_wasserstein(
            problem.A.numpy(), problem.B.numpy(), problem.a.numpy(), problem.b.numpy(),
            loss_fun="square_loss", epsilon=.001, G0=initial.numpy(), solver="PGD",
            max_iter=4, tol=0., numItermax=10000, stopThr=1e-14)
        np.testing.assert_allclose(plan.numpy(), expected, rtol=0., atol=2e-12)

    def test_kernel_underflow_falls_back_without_clipping_or_lost_budget(self):
        A = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
        marginal = torch.full((2,), .5, dtype=torch.float64)
        problem = runner.RowShardedGW(A, A, marginal, marginal, exact_block_rows=1)
        logref = torch.full((2, 2), .25, dtype=torch.float64).log()
        prox = FastCertifiedKLProx(problem, certificate_block_rows=1)
        Q, _, info = prox(logref, A, eta=1000., max_iter=10, check_every=10)
        self.assertTrue(info["converged"])
        self.assertEqual(info["sinkhorn_path"], "original-log-domain-fallback")
        self.assertEqual(info["fast_scaling_iterations"], 0)
        self.assertEqual(info["inner_iterations"], 10)
        self.assertEqual(info["underflow_entries"], 2)
        self.assertEqual(int((Q == 0).sum()), 2)

    def test_late_fallback_honors_total_cap_and_records_both_paths(self):
        problem, plan, logref = fixture()
        class FailThird(FastCertifiedKLProx):
            calls = 0
            def _scaling_iteration(self, kernel, v):
                self.calls += 1
                if self.calls == 3:
                    raise ArithmeticError("deliberate test failure")
                return super()._scaling_iteration(kernel, v)
        prox = FailThird(problem, certificate_block_rows=2)
        _, _, info = prox(logref, problem.gradient(plan), eta=1000., max_iter=5, check_every=10)
        self.assertEqual(info["fast_scaling_iterations"], 3)
        self.assertEqual(info["log_domain_iterations"], 2)
        self.assertEqual(info["inner_iterations"], 5)
        self.assertEqual(info["total_inner_budget"], 5)
        self.assertFalse(info["converged"])

    def test_no_extra_fallback_iteration_when_budget_exhausted(self):
        problem, plan, logref = fixture()
        class FailFirst(FastCertifiedKLProx):
            def _scaling_iteration(self, kernel, v):
                raise ArithmeticError("deliberate test failure")
        prox = FailFirst(problem, certificate_block_rows=2)
        with self.assertRaises(runner.smd.ProximalSolveError) as caught:
            prox(logref, problem.gradient(plan), eta=1000., max_iter=1)
        self.assertEqual(caught.exception.certificate["inner_iterations"], 1)
        self.assertEqual(caught.exception.certificate["total_inner_budget"], 1)

    def test_nonconvergence_returns_certificate_and_exact_cap(self):
        problem, plan, logref = fixture()
        prox = FastCertifiedKLProx(problem, certificate_block_rows=2)
        _, _, info = prox(logref, problem.gradient(plan), eta=1000., max_iter=1)
        self.assertEqual(info["inner_iterations"], 1)
        self.assertEqual(info["certificate_checks"], 1)
        self.assertFalse(info["converged"])
        self.assertIn("primal_objective", info)
        self.assertIn("stable_primal_dual_gap", info)

    def test_inputs_are_unchanged(self):
        problem, plan, logref = fixture()
        gradient = problem.gradient(plan)
        saved_logref, saved_gradient = logref.clone(), gradient.clone()
        FastCertifiedKLProx(problem)(logref, gradient, eta=1000.)
        self.assertTrue(torch.equal(saved_logref, logref))
        self.assertTrue(torch.equal(saved_gradient, gradient))

    def test_zero_gradient_recovers_fixed_product(self):
        problem, plan, logref = fixture()
        Q, _, info = FastCertifiedKLProx(problem)(logref, torch.zeros_like(plan), eta=1000.)
        self.assertTrue(info["converged"])
        torch.testing.assert_close(Q, logref.exp(), rtol=1e-14, atol=1e-15)

    def test_invalid_arguments_fail_before_computation(self):
        problem, plan, logref = fixture()
        prox = FastCertifiedKLProx(problem)
        for options in ({"eta": 0.}, {"eta": float("nan")},
                        {"eta": 1000., "max_iter": 0}, {"eta": 1000., "check_every": 0}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    prox(logref, problem.gradient(plan), **options)


if __name__ == "__main__":
    unittest.main()
