"""Small mathematical regression tests; no production data or CPU campaign."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import ot
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("entropy_md_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from oracle import PairSample, pair_gradient


def fixture():
    A = torch.tensor([[0., .35, 1.], [.35, 0., .7], [1., .7, 0.]], dtype=torch.float64)
    B = torch.tensor([[0., .6, .85], [.6, 0., .2], [.85, .2, 0.]], dtype=torch.float64)
    plan = torch.tensor([[.12, .03, .05], [.03, .17, .1], [.15, .2, .15]], dtype=torch.float64)
    problem = runner.RowShardedGW(A, B, plan.sum(1), plan.sum(0), exact_block_rows=2)
    return problem, plan


class TestEntropyMD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_one_and_multiple_steps_match_pot_pgd_from_nonproduct(self):
        problem, initial = fixture()
        epsilon = .25
        for iterations in (1, 4):
            with self.subTest(iterations=iterations):
                plan, log_plan = initial.clone(), initial.log()
                prox = runner.CertifiedKLProx(problem, gap_tol=1e-10, certificate_block_rows=2)
                for _ in range(iterations):
                    plan, log_plan, _ = runner.entropic_md_step_at_eta(
                        log_plan, problem.gradient(plan), epsilon=epsilon,
                        eta=1/epsilon, prox=prox, tol=1e-13)
                expected = ot.gromov.entropic_gromov_wasserstein(
                    problem.A.numpy(), problem.B.numpy(), problem.a.numpy(), problem.b.numpy(),
                    loss_fun="square_loss", epsilon=epsilon, G0=initial.numpy(),
                    solver="PGD", max_iter=iterations, tol=0.,
                    numItermax=10000, stopThr=1e-14)
                np.testing.assert_allclose(plan.numpy(), expected, rtol=0., atol=2e-12)

    def test_nonproduct_state_detects_missing_entropy_gradient(self):
        problem, initial = fixture()
        epsilon = .25
        prox = runner.CertifiedKLProx(problem, gap_tol=1e-10, certificate_block_rows=2)
        gradient = problem.gradient(initial)
        corrected, _, _ = runner.entropic_md_step_at_eta(
            initial.log(), gradient, epsilon=epsilon, eta=1/epsilon, prox=prox)
        missing, _, _ = prox(initial.log(), gradient, eta=1/epsilon,
                              tol=1e-12, max_iter=10000, check_every=1)
        self.assertGreater(torch.linalg.norm(corrected-missing).item(), .01)

    def test_zero_entropy_is_existing_smd_bit_exact(self):
        problem, plan = fixture()
        prox = runner.CertifiedKLProx(problem, certificate_block_rows=2)
        gradient = problem.gradient(plan)
        actual = runner.scheduled_step(plan.log(), gradient, iteration=3,
                                       cmax=1., epsilon=0., prox=prox)
        expected = runner.smd.smd_step(
            plan.log(), gradient, iteration=3, cmax=1., prefactor_fraction=1.,
            prox=prox, marginal_residuals=prox.marginal_residuals,
            inner_tol=runner.INNER_TOL, inner_max=runner.INNER_MAX,
            check_every=runner.CHECK_EVERY, gap_tol=runner.INNER_GAP_TOL)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(actual[2], expected[2])

    def test_enumerated_pair_oracle_plus_entropy_is_unbiased(self):
        problem, plan = fixture()
        # Also exercise actual-mass scaling away from the feasible set.
        plan = 1.7 * plan
        epsilon = .001
        expectation = torch.zeros_like(plan)
        mass = plan.sum()
        for k in range(problem.n):
            for l in range(problem.m):
                sample = PairSample(torch.tensor([k]), torch.tensor([l]), mass)
                estimate = runner.regularized_gradient(
                    plan.log(), pair_gradient(problem, sample), epsilon)
                expectation.add_(estimate, alpha=(plan[k, l]/mass).item())
        expected = runner.regularized_gradient(plan.log(), problem.gradient(plan), epsilon)
        torch.testing.assert_close(expectation, expected, rtol=2e-15, atol=2e-15)

    def test_regularized_ambient_gradient_by_finite_differences(self):
        problem, plan = fixture()
        plan = .83 * plan
        epsilon = .001
        gradient = runner.regularized_gradient(plan.log(), problem.gradient(plan), epsilon)
        difference = torch.empty_like(plan)
        delta = 1e-6
        for k in range(problem.n):
            for l in range(problem.m):
                plus, minus = plan.clone(), plan.clone()
                plus[k, l] += delta
                minus[k, l] -= delta
                fp = problem.loss(plus) + epsilon*runner.entropy(plus)
                fm = problem.loss(minus) + epsilon*runner.entropy(minus)
                difference[k, l] = (fp-fm)/(2*delta)
        torch.testing.assert_close(gradient, difference, rtol=2e-8, atol=2e-10)

    def test_schedule_includes_entropy_relative_smoothness(self):
        problem, plan = fixture()
        prox = runner.CertifiedKLProx(problem, certificate_block_rows=2)
        _, _, info = runner.scheduled_step(plan.log(), problem.gradient(plan),
                                           iteration=3, cmax=1., epsilon=.001, prox=prox)
        self.assertAlmostEqual(info["eta"], 1/((4 + .002)*2), places=16)

    def test_entropy_zero_convention(self):
        plan = torch.tensor([0., .5, .5], dtype=torch.float64)
        self.assertAlmostEqual(runner.entropy(plan).item(), np.log(.5)-1, places=15)

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


if __name__ == "__main__":
    unittest.main()
