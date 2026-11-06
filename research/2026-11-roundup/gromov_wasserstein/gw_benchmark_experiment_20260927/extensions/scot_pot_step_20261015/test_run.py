"""Mathematical and numerical checks for fixed POT-scale updates."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import ot
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pot_scale_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from oracle import PairSample, pair_gradient


def fixture():
    A = torch.tensor([[0., .35, 1.], [.35, 0., .7], [1., .7, 0.]], dtype=torch.float64)
    B = torch.tensor([[0., .6, .85], [.6, 0., .2], [.85, .2, 0.]], dtype=torch.float64)
    plan = torch.tensor([[.12, .03, .05], [.03, .17, .1], [.15, .2, .15]], dtype=torch.float64)
    # Keep ordinary POT's exponentials safely representable at epsilon=.001.
    problem = runner.RowShardedGW(.1*A, .1*B, plan.sum(1), plan.sum(0), exact_block_rows=2)
    return problem, plan


def reference(problem):
    return (problem.a[:, None] * problem.b[None, :]).log()


class TestPOTScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_one_and_multiple_updates_match_actual_pot_pgd_at_epsilon_001(self):
        problem, initial = fixture()
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                plan = initial.clone()
                prox = runner.CertifiedKLProx(problem, gap_tol=1e-11, certificate_block_rows=2)
                for iteration in range(iterations):
                    plan, _, info = runner.pot_step(
                        reference(problem), problem.gradient(plan), iteration=iteration,
                        epsilon=.001, prox=prox, inner_tol=1e-13, gap_tol=1e-11,
                        check_every=1)
                    self.assertEqual(info["eta"], 1000.)
                expected = ot.gromov.entropic_gromov_wasserstein(
                    problem.A.numpy(), problem.B.numpy(), problem.a.numpy(), problem.b.numpy(),
                    loss_fun="square_loss", epsilon=.001, G0=initial.numpy(),
                    solver="PGD", max_iter=iterations, tol=0.,
                    numItermax=10000, stopThr=1e-14)
                np.testing.assert_allclose(plan.numpy(), expected, rtol=0., atol=2e-12)

    def test_fixed_product_reference_matches_canonical_entropy_subproblem(self):
        problem, initial = fixture()
        gradient = problem.gradient(initial)
        prox = runner.CertifiedKLProx(problem, gap_tol=1e-11, certificate_block_rows=2)
        actual, _, _ = runner.pot_step(
            reference(problem), gradient, iteration=0, epsilon=.001, prox=prox,
            inner_tol=1e-13, gap_tol=1e-11, check_every=1)
        canonical, _, info = prox(torch.zeros_like(initial), gradient, eta=1000.,
                                  tol=1e-13, max_iter=10000, check_every=1)
        self.assertTrue(info["converged"])
        torch.testing.assert_close(actual, canonical, rtol=0., atol=2e-13)

    def test_update_is_iteration_independent_and_has_no_current_plan_reference(self):
        problem, initial = fixture()
        prox = runner.CertifiedKLProx(problem, certificate_block_rows=2)
        gradient = problem.gradient(initial)
        first = runner.pot_step(reference(problem), gradient, iteration=0, epsilon=.001, prox=prox)
        later = runner.pot_step(reference(problem), gradient, iteration=1999, epsilon=.001, prox=prox)
        self.assertTrue(torch.equal(first[0], later[0]))
        self.assertEqual(first[2]["eta"], later[2]["eta"])
        # Retaining the current nonproduct reference would silently change POT PGD to PPA.
        wrong, _, _ = prox(initial.log(), gradient, eta=1000., tol=1e-7,
                            max_iter=10000, check_every=10)
        self.assertGreater(torch.linalg.norm(first[0]-wrong).item(), .01)

    def test_enumerated_pair_oracle_is_unbiased_with_actual_mass_and_zeros(self):
        problem, plan = fixture()
        plan = 1.7 * plan
        plan[0, 1] = 0.
        expectation = torch.zeros_like(plan)
        mass = plan.sum()
        for k in range(problem.n):
            for l in range(problem.m):
                sample = PairSample(torch.tensor([k]), torch.tensor([l]), mass)
                expectation.add_(pair_gradient(problem, sample), alpha=(plan[k, l]/mass).item())
        torch.testing.assert_close(expectation, problem.gradient(plan), rtol=2e-15, atol=2e-15)

    def test_underflow_is_permitted_and_can_recover_support_next_update(self):
        A = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
        marginal = torch.full((2,), .5, dtype=torch.float64)
        problem = runner.RowShardedGW(A, A, marginal, marginal, exact_block_rows=1)
        prox = runner.CertifiedKLProx(problem, certificate_block_rows=1)
        diagonal, logs, info = runner.pot_step(
            reference(problem), A, iteration=0, epsilon=.001, prox=prox)
        self.assertEqual(info["underflow_entries"], 2)
        self.assertEqual(int((diagonal == 0).sum()), 2)
        self.assertTrue(torch.isfinite(logs).all())
        self.assertTrue(torch.isfinite(runner.entropy(diagonal)))
        dense, _, _ = runner.pot_step(
            reference(problem), torch.zeros_like(A), iteration=1, epsilon=.001, prox=prox)
        torch.testing.assert_close(dense, torch.full((2, 2), .25, dtype=torch.float64))

    def test_bad_numerical_certificates_are_rejected(self):
        problem, plan = fixture()
        true_prox = runner.CertifiedKLProx(problem, certificate_block_rows=2)
        original = true_prox(reference(problem), problem.gradient(plan), eta=1000.,
                             tol=1e-7, max_iter=10000, check_every=10)
        for change in ({"converged": False}, {"proximal_optimality_certified": False},
                       {"normalized_primal_dual_gap": .1}, {"gap_roundoff_tolerance": .1},
                       {"primal_witness_marginal_l1": 1e-5}):
            with self.subTest(change=change):
                class Broken:
                    marginal_residuals = true_prox.marginal_residuals
                    def __call__(self, *args, **kwargs):
                        return original[0], original[1], dict(original[2], **change)
                with self.assertRaises(runner.smd.ProximalSolveError):
                    runner.pot_step(reference(problem), problem.gradient(plan),
                                    iteration=0, epsilon=.001, prox=Broken())

    def test_actual_plan_feasibility_is_checked_independently(self):
        problem, plan = fixture()
        true_prox = runner.CertifiedKLProx(problem, certificate_block_rows=2)
        result = true_prox(reference(problem), problem.gradient(plan), eta=1000.,
                           tol=1e-7, max_iter=10000, check_every=10)
        class Broken:
            marginal_residuals = true_prox.marginal_residuals
            def __call__(self, *args, **kwargs):
                return result[0]*.99, result[1], result[2]
        with self.assertRaisesRegex(runner.smd.ProximalSolveError, "Actual summed marginal"):
            runner.pot_step(reference(problem), problem.gradient(plan),
                            iteration=0, epsilon=.001, prox=Broken())

    def test_existing_output_is_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "already_exists"
            out.mkdir()
            (out / "sentinel").write_text("preserve me")
            args = runner.parser().parse_args([
                "--instance", "not-loaded", "--out", str(out), "--oracle", "full", "--seed", "811"])
            with self.assertRaises(FileExistsError):
                runner.run(args)
            self.assertEqual([p.name for p in out.iterdir()], ["sentinel"])

    def test_bad_manifest_hash_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "A.npy", np.zeros((3, 3), dtype=np.float64))
            (path / "manifest.json").write_text(json.dumps({"array_files_sha256": {"A.npy": "bad"}}))
            with self.assertRaisesRegex(ValueError, "Input hash mismatch"):
                runner.load_problem(path)

    def test_zero_epsilon_is_rejected(self):
        args = runner.parser().parse_args([
            "--instance", "unused", "--out", "unused", "--oracle", "full", "--seed", "811",
            "--epsilon", "0"])
        with self.assertRaises(ValueError):
            runner.validate_args(args)


if __name__ == "__main__":
    unittest.main()
