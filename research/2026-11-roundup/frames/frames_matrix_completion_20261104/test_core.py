"""Independent algebra and tiny-trajectory checks for the experiment.

Tests use small temporary instances; they do not launch the campaign or alter its data.
"""
from __future__ import annotations

import hashlib
import json
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import Problem, schedule, update
import run as runner


def tiny_problem():
    y = np.array([[.2, 1.3, -.7], [-1.4, .8, .5], [.9, -.4, .1]], dtype=np.float64)
    mask = np.array([[True, True, False], [True, False, True], [False, True, False]])
    return Problem(y, y / 2, mask, ~mask, 3.)


def finite_difference(function, x, step=1e-6):
    result = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[index] += step
        xm[index] -= step
        result[index] = (function(xp) - function(xm)) / (2 * step)
    return result


def independently_replay(problem, config):
    """Explicit equations, avoiding the production schedule/update/oracle/gap helpers."""
    rng = np.random.default_rng(config["seed"])
    x = np.zeros(problem.Y.shape, dtype=np.float64)
    momentum = None
    observations = list(zip(*np.nonzero(problem.mask)))
    history, gaps = [x.copy()], []
    stream = hashlib.sha256()
    for k in range(config["iterations"]):
        if config["method"] == "sframes":
            gamma = config["gamma0"] * (k + 1.) ** (-3. / 4)
            rho = config["rho0"] * (k + 1.) ** (-1. / 2)
            beta = config["beta0"] * (k + 1.) ** (-1. / 4)
        else:
            paper_k = k + 1.
            gamma = 9. / (paper_k + 8.)
            rho = 4. * (paper_k + 7.) ** (-2. / 3)
            beta = config["beta0"] * (paper_k + 8.) ** (-1. / 2)
        full = np.zeros_like(x)
        for i, j in observations:
            full[i, j] = (x[i, j] - problem.Y[i, j]) / len(observations)
        penalty = np.where(x > 1, x - 1, np.where(x < -1, x + 1, 0.)) / beta
        h = full + penalty
        gaps.append(float(np.vdot(h, x) + problem.tau * np.linalg.svd(h, compute_uv=False)[0]))
        draws = rng.integers(len(observations), size=config["batch"])
        stream.update(draws.astype("<i8").tobytes())
        gradient = np.zeros_like(x)
        for draw in draws:
            i, j = observations[draw]
            gradient[i, j] += (x[i, j] - problem.Y[i, j]) / config["batch"]
        if momentum is None:
            # First-sample initialization in SFRAMES; d0=0 in SHCGM.
            momentum = gradient.copy() if config["method"] == "sframes" else rho * gradient
        else:
            momentum = (1 - rho) * momentum + rho * gradient
        u, _, vt = np.linalg.svd(momentum + penalty, full_matrices=False)
        vertex = -problem.tau * (u[:, :1] @ vt[:1, :])
        x = (1 - gamma) * x + gamma * vertex
        history.append(x.copy())
    return history, np.asarray(gaps), momentum, stream.hexdigest()


class CoreChecks(unittest.TestCase):
    def setUp(self):
        self.problem = tiny_problem()

    def test_oracle_exact_expectation_and_variance(self):
        p = self.problem
        x = np.array([[.1, -.2, .3], [.6, -.1, .4], [.2, -.5, .8]])
        all_gradients = np.stack([p.sample_gradient(x, np.array([i])) for i in range(p.count)])
        mean = all_gradients.mean(axis=0)
        np.testing.assert_allclose(mean, p.gradient(x), atol=1e-15, rtol=1e-14)
        variance = np.mean(np.sum((all_gradients - mean) ** 2, axis=(1, 2)))
        expected = (1 - 1 / p.count) * np.mean((x[p.mask] - p.Y[p.mask]) ** 2)
        self.assertAlmostEqual(variance, expected, places=14)

    def test_repeated_sample_indices_are_accumulated(self):
        p = self.problem
        x = np.zeros_like(p.Y)
        draws = np.array([1, 1, 1, 3])
        expected = sum(p.sample_gradient(x, np.array([i])) for i in draws) / len(draws)
        np.testing.assert_allclose(p.sample_gradient(x, draws), expected, atol=1e-15)
        np.testing.assert_allclose(p.sample_gradient(x, np.array([2, 2, 2])),
                                   p.sample_gradient(x, np.array([2])), atol=1e-15)

    def test_full_loss_gradient_by_finite_difference(self):
        x = np.array([[1.4, -.6, -.2], [-1.3, .1, .9], [.5, -.2, .7]])
        np.testing.assert_allclose(self.problem.gradient(x), finite_difference(self.problem.loss, x),
                                   atol=1e-10, rtol=1e-8)

    def test_positive_penalty_gradient_by_finite_difference(self):
        x = np.array([[1.4, -.6, -1.2], [-1.3, .1, .9], [.5, 1., -1.]])
        beta = .7
        penalty = lambda z: float(np.sum((z - np.clip(z, -1, 1)) ** 2) / (2 * beta))
        np.testing.assert_allclose(self.problem.penalty_gradient(x, beta), finite_difference(penalty, x),
                                   atol=4e-7, rtol=1e-7)
        self.assertGreater(self.problem.penalty_gradient(x, beta)[0, 0], 0)
        self.assertLess(self.problem.penalty_gradient(x, beta)[0, 2], 0)

    def test_prescribed_schedule_constants_and_indexing(self):
        np.testing.assert_allclose(schedule("shcgm", 0, 12.), (1., 1., 4.), atol=1e-14)
        np.testing.assert_allclose(schedule("shcgm", 7, 12.), (9/16, 4/15**(2/3), 3.), atol=1e-14)
        # Arbitrary gamma0/rho0 inputs must not modify the published SHCGM schedule.
        np.testing.assert_allclose(schedule("shcgm", 7, 12., .2, .1),
                                   schedule("shcgm", 7, 12.), atol=1e-14)
        np.testing.assert_allclose(schedule("sframes", 0, 2., .6, .2), (.6, .2, 2.), atol=1e-14)
        np.testing.assert_allclose(schedule("sframes", 15, 2., .6, .2), (.6/8, .2/4, 1.), atol=1e-14)

    def test_lmo_feasibility_and_exact_linear_value(self):
        rng = np.random.default_rng(28)
        for h in (rng.normal(size=(3, 3)), np.ones((3, 3)), np.zeros((3, 3))):
            s = self.problem.lmo(h)
            nuclear = np.linalg.svd(s, compute_uv=False).sum()
            self.assertLessEqual(nuclear, self.problem.tau + 1e-12)
            expected = -self.problem.tau * np.linalg.svd(h, compute_uv=False)[0]
            self.assertAlmostEqual(float(np.sum(h * s)), float(expected), places=12)

    def test_radial_repair_preserves_both_constraints(self):
        p = self.problem
        matrices = [np.diag([2., -.5, .2]), np.ones((3, 3)) / 2, np.zeros((3, 3))]
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.normal(size=(3, 3))
            matrices.append(x * p.tau / np.linalg.svd(x, compute_uv=False).sum())
        for x in matrices:
            self.assertLessEqual(np.linalg.svd(x, compute_uv=False).sum(), p.tau + 1e-12)
            z = p.feasible_repair(x)
            self.assertLessEqual(np.max(np.abs(z)), 1 + 1e-14)
            self.assertLessEqual(np.linalg.svd(z, compute_uv=False).sum(), p.tau + 1e-12)
            np.testing.assert_allclose(p.feasible_repair(z), z, atol=1e-14)

    def test_first_sample_momentum_is_not_scaled_in_sframes(self):
        p = self.problem
        x = np.zeros_like(p.Y)
        g = p.sample_gradient(x, np.array([0, 1, 1, 3]))
        xn, momentum = update(p, x, None, g, method="sframes", iteration=0,
                              beta0=.8, gamma0=.6, rho0=.2)
        np.testing.assert_array_equal(momentum, g)
        np.testing.assert_allclose(xn, .6 * p.lmo(g), atol=1e-14)
        _, sh_momentum = update(p, x, None, g, method="shcgm", iteration=0, beta0=.8)
        np.testing.assert_allclose(sh_momentum, g, atol=1e-14)

    def test_short_trajectories_gap_indexing_and_paired_sample_streams(self):
        p = self.problem
        hashes = []
        with tempfile.TemporaryDirectory(prefix="frames_mc_test_") as temp:
            root = Path(temp)
            np.savez(root / "data.npz", Y=p.Y, signal=p.signal, mask=p.mask,
                     test_mask=p.test_mask, tau=p.tau, lower=p.lower, upper=p.upper)
            # run() fingerprints its own sources relative to ROOT.
            for name in ("core.py", "run.py"):
                (root / name).write_bytes((Path(__file__).parent / name).read_bytes())
            with patch.object(runner, "ROOT", root):
                for method in ("sframes", "shcgm"):
                    cfg = dict(method=method, seed=311, iterations=7, batch=4,
                               beta0=.8, gamma0=.6, rho0=.2)
                    out = root / method
                    runner.run(cfg, out)
                    result = json.loads((out / "result.json").read_text())
                    history, gaps, momentum, expected_hash = independently_replay(p, cfg)
                    with np.load(out / "final_state.npz") as state:
                        np.testing.assert_allclose(state["x"], history[-1], atol=1e-12, rtol=1e-12)
                        np.testing.assert_allclose(state["momentum"], momentum, atol=1e-12, rtol=1e-12)
                    self.assertEqual(result["sampled_stream_sha256"], expected_hash)
                    hashes.append(expected_hash)
                    self.assertEqual(result["sampled_entries"], cfg["iterations"] * cfg["batch"])
                    self.assertEqual(result["completed_updates"], cfg["iterations"])
                    self.assertEqual([row["iteration"] for row in result["trace"]], [0, 1, 2, 5, 7])
                    for row in result["trace"]:
                        k = row["iteration"]
                        if k == 0:
                            self.assertIsNone(row["mean_true_smoothed_gap"])
                        else:
                            self.assertAlmostEqual(row["mean_true_smoothed_gap"], float(gaps[:k].mean()), places=11)
                        x = history[k]
                        repaired = x / max(1., float(np.max(np.abs(x))))
                        expected_loss = float(np.mean((repaired[p.mask] - p.Y[p.mask]) ** 2) / 2)
                        self.assertAlmostEqual(row["feasible_loss"], expected_loss, places=12)
                        self.assertEqual(row["sampled_entries"], k * cfg["batch"])
                        self.assertGreaterEqual(row["total_seconds"], row["optimizer_seconds"] + row["diagnostic_seconds"])
                    self.assertGreater(result["optimizer_seconds"], 0.)
                    self.assertGreater(result["diagnostic_seconds"], 0.)
        self.assertEqual(hashes[0], hashes[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
