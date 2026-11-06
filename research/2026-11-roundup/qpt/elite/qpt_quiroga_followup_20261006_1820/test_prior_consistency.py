"""Synthetic checkpoint and work-accounting checks; no optimizer execution."""
import copy
import unittest

import prior_consistency as prior


class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np

    def arrays(self, method, steps):
        prefix = method.replace("-", "_")
        steps = self.np.asarray(steps, dtype=self.np.int64)
        return {prefix + "_checkpoint_steps": steps,
                prefix + "_rmse": 1.0 / (steps + 1),
                prefix + "_process_fidelity": 1 - 1.0 / (steps + 2),
                prefix + "_cumulative_data_row_accesses": steps * (32 if method == "frames" else 128),
                prefix + "_optimizer_seconds": steps.astype(float) / 1000}

    def test_extended_quiroga_and_frames_common_checkpoints(self):
        for method, stride, horizon in (("printed-adafgd", 1000, 10000), ("frames", 4000, 40000)):
            with self.subTest(method=method):
                old = self.arrays(method, range(0, horizon + 1, stride // 10))
                new = self.arrays(method, range(0, 10 * horizon + 1, stride))
                check = prior.compare_arrays(new, old, method, self.np, ("rmse", "process_fidelity"))
                self.assertEqual(check["matching_checkpoint_count"], 11)
                self.assertEqual(check["matching_checkpoint_steps"][-1], horizon)
                self.assertEqual(check["max_absolute_deviation"]["rmse"], 0)

    def test_trajectory_change_or_missing_original_horizon_rejected(self):
        old = self.arrays("printed-adafgd", range(0, 10001, 100))
        new = self.arrays("printed-adafgd", range(0, 100001, 1000))
        changed = copy.deepcopy(new)
        changed["printed_adafgd_rmse"][1] += 1e-6
        with self.assertRaisesRegex(ValueError, "trajectory differs"):
            prior.compare_arrays(changed, old, "printed-adafgd", self.np, ("rmse",))
        with self.assertRaisesRegex(ValueError, "original completed horizon"):
            prior.compare_arrays(self.arrays("printed-adafgd", [0, 1000]), old,
                                 "printed-adafgd", self.np, ("rmse",))

    def test_armijo_between_checkpoint_stop_does_not_relabel_old_status(self):
        old = self.arrays("armijo-fgd", [0, 100, 200, 300])
        new = self.arrays("armijo-fgd", [0, 100, 150])
        check = prior.compare_arrays(new, old, "armijo-fgd", self.np, ("rmse",))
        self.assertEqual(check["matching_checkpoint_steps"], [0, 100])
        self.assertIn("terminal_status", check["excluded_fields"])

    def test_exact_candidate_matching_requires_schedule_and_channel(self):
        trial = {"channel": 0, "method": "frames", "alpha": 5e-5, "eta_scale": .01, "tau": 2,
                 "step_schedule": {"scale": 4, "offset": 4, "exponent": 1, "cap": 1}}
        other = {**trial, "channel": 1}
        report = {"cases": [{"trial_id": "other", "trial": other}, {"trial_id": "right", "trial": trial}]}
        self.assertEqual(prior.matching_case(report, trial, {})["trial_id"], "right")
        report["cases"].append({"trial_id": "duplicate", "trial": trial})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prior.matching_case(report, trial, {})

    def test_q_and_frames_row_accounting_and_failed_attempt(self):
        for method in ("printed-adafgd", "frames"):
            arrays = self.arrays(method, [0, 100, 200])
            prefix = method.replace("-", "_")
            prior.check_work(arrays, prefix, method, "complete", None, self.np)
            arrays[prefix + "_cumulative_data_row_accesses"][-1] += 128
            if method == "printed-adafgd":
                prior.check_work(arrays, prefix, method, "failed", "nonfinite update", self.np)
            with self.assertRaises(ValueError):
                prior.check_work(arrays, prefix, method, "complete", None, self.np)

    def test_armijo_work_counts_terminal_gradient_and_failed_search(self):
        arrays = self.arrays("armijo-fgd", [0, 2])
        arrays["armijo_fgd_diagnostic_objective_evaluations"] = self.np.array([1, 2])
        arrays["armijo_fgd_cumulative_data_row_accesses"][-1] = 128 * (2 + 3 + 1)
        check = prior.check_work(arrays, "armijo_fgd", "armijo-fgd", "converged", None, self.np)
        self.assertEqual(check["terminal_attempt_rows"], 128)
        arrays["armijo_fgd_cumulative_data_row_accesses"][-1] = 128 * (2 + 3 + 32)
        check = prior.check_work(arrays, "armijo_fgd", "armijo-fgd", "failed",
                                 "RuntimeError: Armijo line search failed after 31 trials.", self.np)
        self.assertEqual(check["terminal_attempt_rows"], 4096)
        arrays["armijo_fgd_cumulative_data_row_accesses"][-1] -= 128
        with self.assertRaisesRegex(ValueError, "row charge differs"):
            prior.check_work(arrays, "armijo_fgd", "armijo-fgd", "failed",
                             "RuntimeError: Armijo line search failed after 31 trials.", self.np)

    def test_nonfinite_time_rejected(self):
        arrays = self.arrays("frames", [0, 100])
        arrays["frames_optimizer_seconds"][-1] = float("nan")
        with self.assertRaisesRegex(ValueError, "time/work"):
            prior.check_work(arrays, "frames", "frames", "complete", None, self.np)


if __name__ == "__main__":
    unittest.main()
