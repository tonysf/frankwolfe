"""Numerical parity and restart timing for the instrumented replay only."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load_module("original_wallclock_reference", HERE.parent/"large_pot_frames_20261016"/"runner.py")
runner = load_module("instrumented_wallclock_runner", HERE/"runner.py")


def write_instance(path):
    path.mkdir()
    arrays = dict(
        A=np.array([[0., .035, .1], [.035, 0., .07], [.1, .07, 0.]], dtype=np.float32),
        B=np.array([[0., .06, .085], [.06, 0., .02], [.085, .02, 0.]], dtype=np.float32),
        a=np.array([.2, .3, .5], dtype=np.float64),
        b=np.array([.3, .4, .3], dtype=np.float64),
    )
    for name, value in arrays.items():
        np.save(path/(name+".npy"), value)
    (path/"manifest.json").write_text(json.dumps(dict(n=3, m=3,
        array_files_sha256={name+".npy":runner.sha256(path/(name+".npy")) for name in arrays})))


def arguments(module, instance, output, method, chunk=6):
    return module.parser().parse_args([
        "--instance", str(instance), "--out", str(output), "--method", method,
        "--device", "cpu", "--iterations", "6", "--chunk-updates", str(chunk),
        "--pairs", "16", "--gamma0", ".3", "--rho0", ".1", "--mu-scale", "10",
        "--checkpoint-every", "2", "--diagnostic-every", "2", "--retain-checkpoint",
    ])


def without_timing(trace):
    return [{key:value for key,value in point.items() if key != "elapsed_seconds"} for point in trace]


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

    def assert_states_equal(self, expected, actual):
        for key in ("plan", "momentum", "generator_state", "torch_rng_state", "cuda_rng_state"):
            if expected[key] is None:
                self.assertIsNone(actual[key])
            else:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def assert_monotone_trace(self, record):
        times = [point["elapsed_seconds"] for point in record["trace"]]
        self.assertGreater(times[0], 0)
        self.assertTrue(all(left < right for left, right in zip(times, times[1:])))
        self.assertGreaterEqual(record["elapsed_seconds"], times[-1])

    def test_all_methods_bitwise_parity_with_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_instance(root/"input")
            for method in ("pot-full", "pot-pair", "frames"):
                with self.subTest(method=method):
                    before = original.run(arguments(original, root/"input", root/(method+"-before"), method))
                    after = runner.run(arguments(runner, root/"input", root/(method+"-after"), method))
                    self.assertEqual(before["trace"], without_timing(after["trace"]))
                    self.assertEqual(before["inner_trace"], after["inner_trace"])
                    self.assertEqual(before["counters"], after["counters"])
                    self.assertEqual(before["config"], after["config"])
                    self.assertEqual(before["final_loss"], after["final_loss"])
                    self.assert_monotone_trace(after)
                    expected, _ = runner.CheckpointStore(root/(method+"-before")/"checkpoints").load(identity=before["identity"])
                    actual, _ = runner.CheckpointStore(root/(method+"-after")/"checkpoints").load(identity=after["identity"])
                    self.assert_states_equal(expected, actual)
                    self.assertEqual(actual["trace"], after["trace"])

    def test_resume_preserves_trace_and_counts_post_checkpoint_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_instance(root/"input")
            for method in ("pot-full", "pot-pair", "frames"):
                with self.subTest(method=method):
                    full = runner.run(arguments(runner, root/"input", root/(method+"-full"), method))
                    partial = runner.run(arguments(runner, root/"input", root/(method+"-resume"), method, chunk=2))
                    self.assertEqual(partial["status"], "checkpointed")
                    state, _ = runner.CheckpointStore(root/(method+"-resume")/"checkpoints").load(identity=partial["identity"])
                    self.assertGreater(partial["elapsed_seconds"], state["elapsed_seconds"])
                    resumed = runner.run(arguments(runner, root/"input", root/(method+"-resume"), method))
                    self.assertEqual(resumed["status"], "completed")
                    self.assertEqual(resumed["timing"]["elapsed_prior_seconds"], partial["elapsed_seconds"])
                    self.assertEqual(resumed["timing"]["resume_elapsed_source"], "matching_post_checkpoint_progress")
                    self.assertEqual(resumed["trace"][:len(partial["trace"])], partial["trace"])
                    self.assertEqual(without_timing(full["trace"]), without_timing(resumed["trace"]))
                    self.assertEqual(full["counters"], resumed["counters"])
                    self.assert_monotone_trace(resumed)
                    self.assertGreater(resumed["trace"][len(partial["trace"])]["elapsed_seconds"], partial["elapsed_seconds"])
                    expected, _ = runner.CheckpointStore(root/(method+"-full")/"checkpoints").load(identity=full["identity"])
                    actual, _ = runner.CheckpointStore(root/(method+"-resume")/"checkpoints").load(identity=resumed["identity"])
                    self.assert_states_equal(expected, actual)

    def test_resume_rejects_unrelated_progress_generation_for_timing(self):
        state = {"elapsed_seconds": 10.}
        checkpoint = {"generation": "generation-current", "identity": {"test": 1}, "iteration": 2}
        unrelated = dict(checkpoint={"generation": "generation-other"}, identity={"test": 1},
                         completed_updates=2, elapsed_seconds=999.)
        self.assertEqual(runner.resume_elapsed_seconds(state, checkpoint, unrelated),
                         (10., "checkpoint_snapshot_before_write"))
        self.assertEqual(runner.resume_elapsed_seconds(state, checkpoint, None),
                         (10., "checkpoint_snapshot_before_write"))


if __name__ == "__main__":
    unittest.main()
