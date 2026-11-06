"""Small checks of source semantics and exact restart, without dataset runs."""
from pathlib import Path
import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("control_runner", ROOT / "control_runner.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class ControlChecks(unittest.TestCase):
    def test_original_threshold_order_and_strict_boundary(self):
        U = np.array([[-9., -2.], [4., 5.], [3., 4.], [2., -1.]])
        expected_outer = np.array([[0., 0.], [4., 5.], [3., 4.], [0., 0.]])
        expected_inner = np.array([[0., 0.], [0., 5.], [0., 0.], [0., 0.]])
        np.testing.assert_array_equal(control.source_outer_threshold(U, 2), expected_outer)
        np.testing.assert_array_equal(control.source_inner_threshold(U, 2), expected_inner)

    @staticmethod
    def fixture():
        rng = np.random.default_rng(90)
        U = rng.uniform(.01, .1, (80, 6))
        V = rng.uniform(.01, .1, (80, 6))
        A = rng.uniform(.1, 1., (80, 80))
        A /= np.linalg.norm(A, axis=0)
        return A, U, V

    def test_work_counts_and_feasibility_after_warmup(self):
        A, U, V = self.fixture()
        U, V, counts = control.outer_cycle(A, U, V, 1, np.random.default_rng(3), 20)
        self.assertEqual(counts, dict(residual_entries=2*A.size,
                                    partial_gradient_calls=40, power_method_calls=42))
        U, V, counts = control.outer_cycle(A, U, V, 2, np.random.default_rng(4), 20)
        self.assertEqual(counts, dict(residual_entries=6*A.size,
                                    partial_gradient_calls=82, power_method_calls=44))
        self.assertTrue(np.isfinite(U).all() and np.isfinite(V).all())
        self.assertTrue((U >= 0).all() and (V >= 0).all())
        self.assertLessEqual(int(np.count_nonzero(U, axis=0).max()), 20)

    def test_checkpoint_resume_is_exact(self):
        A, U, V = self.fixture()
        previous_root = control.ROOT
        try:
            with tempfile.TemporaryDirectory(prefix="spring_control_check_") as directory:
                control.ROOT = Path(directory)
                data = control.ROOT / "data" / "processed"
                data.mkdir(parents=True)
                np.savez(data / "orl64_unitnorm.npz", A=A, U0=U, V0=V)

                def run(name, epochs, resume=False):
                    args = argparse.Namespace(dataset="orl64", source_epochs=epochs,
                        seed=7, output=str(control.ROOT/name), threads=1,
                        checkpoint_every=1, max_seconds=None, resume=resume)
                    with contextlib.redirect_stdout(io.StringIO()):
                        control.run(args)

                run("full", 4)
                run("split", 2)
                run("split", 4, True)
                with np.load(control.ROOT/"full"/"checkpoint.npz") as full, \
                     np.load(control.ROOT/"split"/"checkpoint.npz") as split:
                    np.testing.assert_array_equal(full["U"], split["U"])
                    np.testing.assert_array_equal(full["V"], split["V"])
                    a, b = json.loads(str(full["state"])), json.loads(str(split["state"]))
                    self.assertEqual(a["counts"], b["counts"])
                    self.assertEqual(a["rng_state"], b["rng_state"])
        finally:
            control.ROOT = previous_root


if __name__ == "__main__":
    unittest.main()
