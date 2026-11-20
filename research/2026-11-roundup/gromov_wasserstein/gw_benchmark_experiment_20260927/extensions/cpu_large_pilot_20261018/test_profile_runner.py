"""The profiling boundary must not alter optimizer results or RNG state."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load("cpu_profile_test_subject", HERE / "profile_runner.py")
fixtures = load("cpu_profile_test_fixtures", HERE.parent / "wallclock_replay_20261018/test_runner.py")


class ProfileTests(unittest.TestCase):
    def test_all_methods_preserve_values_and_tensor_states(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures.write_instance(root / "input")
            for method in ("frames", "pot-full", "pot-pair"):
                with self.subTest(method=method):
                    plain_args = fixtures.arguments(profile.runner, root / "input", root / (method + "-plain"), method)
                    timed_args = fixtures.arguments(profile.runner, root / "input", root / (method + "-timed"), method)
                    before = profile.runner.run(plain_args)
                    after, summary = profile.run_profiled(timed_args)
                    self.assertEqual(fixtures.without_timing(before["trace"]), fixtures.without_timing(after["trace"]))
                    for key in ("identity", "counters", "inner_trace", "final_loss"):
                        self.assertEqual(before[key], after[key], key)
                    expected, _ = profile.runner.CheckpointStore(Path(plain_args.out) / "checkpoints").load(identity=before["identity"])
                    actual, _ = profile.runner.CheckpointStore(Path(timed_args.out) / "checkpoints").load(identity=after["identity"])
                    for key in ("plan", "momentum", "generator_state", "torch_rng_state", "cuda_rng_state"):
                        if expected[key] is None:
                            self.assertIsNone(actual[key])
                        else:
                            self.assertTrue(torch.equal(expected[key], actual[key]), key)
                    events = [json.loads(line) for line in Path(summary["events_path"]).read_text().splitlines()]
                    updates = [event for event in events if event["phase"] == "update"]
                    self.assertEqual(len(updates), 6)
                    self.assertEqual(summary["completed_updates_in_launch"], 6)
                    self.assertTrue(all(event["duration_seconds"] > 0 for event in updates))
                    self.assertTrue(all(event["peak_rss_bytes"] > 0 for event in events))
                    if method.startswith("pot-"):
                        self.assertEqual([event["inner_iterations"] for event in updates],
                                         [info["inner_iterations"] for info in after["inner_trace"]])


if __name__ == "__main__":
    unittest.main()
