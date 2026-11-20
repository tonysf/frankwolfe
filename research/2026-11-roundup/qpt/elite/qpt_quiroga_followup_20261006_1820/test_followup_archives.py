"""Synthetic archive checks; no scientific optimizers run."""
import json
from pathlib import Path
import tempfile
import unittest

import freeze_plan
import run_followup as runner


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        cls.base = runner.load_base(
            Path("/home/tony/qpt_quiroga_resolution_20261006_1748/run_diagnostics.py"), freeze_plan.BASE_SHA)
        cls.plan = freeze_plan.build_plan("0f7857e94f6b16ac07733be248f68aa6904d287b", "synthetic")

    def make_archive(self, directory, method, status="complete", bad_horizon=False, bad_stop=False):
        np = self.np
        trial = next(t for t in self.plan["trials"] if t["method"] == method)
        cfg = runner.configuration(self.plan, trial)
        count = cfg["frames_steps"] if method == "frames" else cfg["n_steps"]
        steps = np.linspace(0, count, 101, dtype=np.int64) if status == "complete" else np.array([0, 50])
        if bad_horizon:
            steps[-1] -= 1
        rows = steps * (32 if method == "frames" else 128)
        if status == "converged":
            rows = steps * 256
            rows[-1] += 128
        physical = {key: 0.0 for key in self.base.PHYSICAL_METRICS}
        physical["process_fidelity"] = 1.0
        endpoint = {"status": status, "failure_reason": "synthetic" if status == "failed" else None,
                    "completed_steps": int(steps[-1]), "cumulative_data_row_accesses": int(rows[-1]),
                    **physical, "penalized_objective": 0.0}
        if status == "converged":
            endpoint["termination"] = {"reason": "full_gradient_tolerance", "full_gradient_norm": 1e-13,
                                       "gradient_threshold": 1.01e-10 if not bad_stop else 1e-8,
                                       "initial_full_gradient_norm": 1.0}
        metadata = {"methods": {method: endpoint}, "configuration": runner.actual_configuration(cfg),
                    "diagnostic": {"trial": trial, "trial_id": trial["id"], "split": "calibration",
                                   "role": self.plan["role"], "requested_configuration": cfg,
                                   "full_design_endpoint": physical,
                                   "gate": self.base.gate_for(status, physical, self.plan["shared"]["gate"])}}
        prefix = method.replace("-", "_")
        payload = {"metadata_json": np.asarray(json.dumps(metadata)), "active_rows": np.arange(128),
                   prefix + "_final_factor": np.zeros((16, 1), dtype=np.complex128),
                   prefix + "_checkpoint_steps": steps, prefix + "_cumulative_data_row_accesses": rows}
        for key in (*self.base.PHYSICAL_METRICS, "penalized_objective"):
            payload[prefix + "_" + key] = np.full(steps.shape, endpoint[key], dtype=float)
        npz, js = Path(directory) / "result.npz", Path(directory) / "result.json"
        np.savez(npz, **payload)
        self.base.write_json(js, metadata)
        return npz, js, trial

    def test_complete_extended_horizons_validate(self):
        for method in ("printed-adafgd", "frames"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                npz, js, trial = self.make_archive(temp, method)
                runner.validate_saved(npz, js, self.plan, trial, {}, self.base, self.np)

    def test_early_complete_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            npz, js, trial = self.make_archive(temp, "frames", bad_horizon=True)
            with self.assertRaisesRegex(ValueError, "Incomplete complete"):
                runner.validate_saved(npz, js, self.plan, trial, {}, self.base, self.np)

    def test_converged_armijo_and_failed_status_preserved(self):
        for status in ("converged", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                npz, js, trial = self.make_archive(temp, "armijo-fgd", status=status)
                metadata = runner.validate_saved(npz, js, self.plan, trial, {}, self.base, self.np)
                self.assertEqual(metadata["methods"]["armijo-fgd"]["status"], status)
                self.assertEqual(metadata["diagnostic"]["gate"]["pass"], status == "converged")

    def test_wrong_stationarity_threshold_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            npz, js, trial = self.make_archive(temp, "armijo-fgd", status="converged", bad_stop=True)
            with self.assertRaisesRegex(ValueError, "Stopping threshold"):
                runner.validate_saved(npz, js, self.plan, trial, {}, self.base, self.np)


if __name__ == "__main__":
    unittest.main()
