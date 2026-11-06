"""Synthetic follow-up harness checks; no optimization or remote calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import freeze_plan
import run_followup as runner

COMMIT = "0f7857e94f6b16ac07733be248f68aa6904d287b"
BASE = Path("/home/tony/qpt_quiroga_resolution_20261006_1748/run_diagnostics.py")


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.plan = freeze_plan.build_plan(COMMIT, "synthetic")

    def test_explicit_counts_horizons_and_seeds(self):
        cases = runner.validate_plan(self.plan)
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(t["method"] == "printed-adafgd" for t in cases), 4)
        self.assertEqual(sum(t["method"] == "frames" for t in cases), 2)
        self.assertEqual(sum(t["method"] == "armijo-fgd" for t in cases), 6)
        for trial in cases:
            cfg = runner.configuration(self.plan, trial)
            self.assertEqual(cfg["initialization_seed"], 200000 + 10 * trial["channel"])
            self.assertEqual(cfg["sampling_seed"], 300000 + 100 * trial["channel"])
            if trial["method"] != "armijo-fgd":
                self.assertEqual(cfg["n_steps"], 100000)
                self.assertEqual(cfg["frames_steps"], 400000)
                self.assertEqual(cfg["metrics_frequency"], 1000)
                self.assertEqual(cfg["frames_metrics_frequency"], 4000)
                self.assertEqual(cfg["armijo_gradient_atol"], 0)
                self.assertEqual(cfg["armijo_gradient_rtol"], 0)
            else:
                self.assertEqual(cfg["n_steps"], 10000)
                self.assertEqual(cfg["metrics_frequency"], 100)
                self.assertEqual(cfg["armijo_gradient_atol"], 1e-12)
                self.assertEqual(cfg["armijo_gradient_rtol"], 1e-10)

    def test_immutable_plan_rejects_any_scientific_or_resource_change(self):
        changes = [lambda p: p["trials"][0].update(alpha=0.5),
                   lambda p: p["trials"][0].update(row_epochs=200000),
                   lambda p: p["shared"]["gate"].update(rmse_max=1e-4),
                   lambda p: p["resources"].update(minimum_seconds_before_new_trial=0),
                   lambda p: p.update(channels=[100, 101]),
                   lambda p: p["trials"].pop()]
        for change in changes:
            with self.subTest(change=change):
                plan = copy.deepcopy(self.plan)
                change(plan)
                with self.assertRaises(ValueError):
                    runner.validate_plan(plan)

    def test_frames_exact_fast_schedule_penalty(self):
        trial = next(t for t in self.plan["trials"] if t["method"] == "frames")
        cfg = runner.configuration(self.plan, trial)
        self.assertEqual(cfg["frames_beta0"], 10000)
        self.assertEqual(cfg["step_size_schedule"], {"scale": 4, "offset": 4, "exponent": 1, "cap": 1})
        self.assertEqual(cfg["frames_smoothing_schedule"],
                         {"scale": 10000, "offset": 1, "exponent": 0.25, "cap": None})
        actual = runner.actual_configuration(cfg)
        self.assertNotIn("show_progress", actual)
        self.assertEqual(actual["armijo_gradient_atol"], 0)
        self.assertEqual(actual["frames_initial_tp_weight_per_row_alpha"], 5e-5)

    def test_original_helper_is_hash_pinned_and_not_modified(self):
        base = runner.load_base(BASE, freeze_plan.BASE_SHA)
        self.assertEqual(base.file_hash(BASE), freeze_plan.BASE_SHA)
        with self.assertRaisesRegex(ValueError, "checksum"):
            runner.load_base(BASE, "0" * 64)

    def test_old_source_and_malformed_commit_rejected(self):
        for value in (freeze_plan.OLD_COMMIT, "bad", "a" * 39):
            with self.subTest(value=value), self.assertRaises(ValueError):
                freeze_plan.build_plan(value, "synthetic")

    def test_validate_only_without_scientific_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            base = runner.load_base(BASE, freeze_plan.BASE_SHA)
            path = Path(temp) / "plan.json"
            base.write_json(path, self.plan)
            with patch.object(runner, "source_hashes", side_effect=AssertionError("must not inspect source")):
                self.assertEqual(runner.main(["--plan", str(path), "--base-runner", str(BASE),
                                              "--validate-only"]), 0)

    def test_actual_execution_requires_slurm(self):
        with tempfile.TemporaryDirectory() as temp:
            base = runner.load_base(BASE, freeze_plan.BASE_SHA)
            path = Path(temp) / "plan.json"
            base.write_json(path, self.plan)
            with patch.dict(runner.os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "requires Slurm"):
                    runner.main(["--plan", str(path), "--base-runner", str(BASE)])


if __name__ == "__main__":
    unittest.main()
