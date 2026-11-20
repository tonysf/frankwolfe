"""Synthetic harness checks only: no scientific optimization or remote calls."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_diagnostics", HERE / "run_diagnostics.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def synthetic_plan():
    return {
        "schema_version": 1, "kind": "qpt_calibration_resolution_diagnostic",
        "campaign_id": "synthetic_test", "source": {"commit": runner.COMMIT,
            "repo": "/gpfs/workdir/silvetian/frankwolfe-quiroga-" + runner.COMMIT},
        "shared": copy.deepcopy(runner.SHARED), "channels": [0, 1],
        "resources": {"minimum_seconds_before_new_trial": 120},
        "role": "Synthetic calibration diagnostic",
        "trials": [{"id": f"test-c{c}-{method}", "channel": c,
                    "method": method, "alpha": 0.00005, "eta_scale": 0.01, "tau": 2.0}
                   for c in (0, 1) for method in runner.METHODS],
    }


class PlanTests(unittest.TestCase):
    def test_explicit_inventory_and_seed_budget_conversion(self):
        plan = synthetic_plan()
        self.assertEqual(len(runner.validate_plan(plan)), 6)
        config = runner.run_configuration(plan, plan["trials"][-1])
        self.assertEqual(config["methods"], ["frames"])
        self.assertEqual(config["initialization_seed"], 200010)
        self.assertEqual(config["sampling_seed"], 300100)
        self.assertEqual(config["n_steps"], 10000)
        self.assertEqual(config["frames_steps"], 40000)
        self.assertEqual(config["frames_metrics_frequency"], 400)
        self.assertEqual(config["tp_weight"], 0.0064)
        self.assertEqual(config["frames_beta0"], 10000)
        self.assertEqual(config["frames_smoothing_schedule"]["exponent"], 0.25)

    def test_frames_override_is_explicit_and_method_specific(self):
        plan = synthetic_plan()
        schedule = {"scale": 4, "offset": 4, "exponent": 1, "cap": 1}
        plan["trials"][-1]["step_schedule"] = schedule
        runner.validate_plan(plan)
        self.assertEqual(runner.run_configuration(plan, plan["trials"][-1])["step_size_schedule"], schedule)
        plan["trials"][0]["step_schedule"] = schedule
        with self.assertRaisesRegex(ValueError, "Only FRAMES"):
            runner.validate_plan(plan)

    def test_mutated_split_pin_shared_and_guard_rejected(self):
        changes = [
            lambda p: p.update(channels=[100, 101]),
            lambda p: p["source"].update(commit="a" * 40),
            lambda p: p["shared"].update(row_epochs=20000),
            lambda p: p["resources"].update(minimum_seconds_before_new_trial=119),
            lambda p: p["trials"][0].update(channel=100),
            lambda p: p["trials"][-1].update(tau=1.99),
        ]
        for change in changes:
            with self.subTest(change=change):
                plan = synthetic_plan()
                change(plan)
                with self.assertRaises(ValueError):
                    runner.validate_plan(plan)

    def test_unsafe_duplicate_extra_and_nonfinite_trial_rejected(self):
        changes = [
            {"id": "../escape"}, {"alpha": float("nan")}, {"alpha": float("inf")},
            {"alpha": True}, {"eta_scale": 0}, {"method": "new-method"},
            {"unimplemented_hyperparameter": 1},
        ]
        for change in changes:
            with self.subTest(change=change):
                plan = synthetic_plan()
                plan["trials"][0].update(change)
                with self.assertRaises(ValueError):
                    runner.validate_plan(plan)
        plan = synthetic_plan()
        plan["trials"][1]["id"] = plan["trials"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            runner.validate_plan(plan)

    def test_missing_values_never_pass_gate(self):
        exact = {"process_fidelity": 1.0, "rmse": 0.0, "normalized_tp_violation": 0.0}
        gate = runner.SHARED["gate"]
        self.assertTrue(runner.gate_for("complete", exact, gate)["pass"])
        self.assertTrue(runner.gate_for("converged", exact, gate)["pass"])
        self.assertFalse(runner.gate_for("failed", exact, gate)["pass"])
        for value in (None, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.assertFalse(runner.gate_for("complete", {**exact, "rmse": value}, gate)["pass"])

    def test_no_overwrite_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            runner.write_json(path, {"state": "prepared"})
            with self.assertRaises(FileExistsError):
                runner.write_json(path, {"state": "overwritten"})
            runner.write_json(path, {"state": "complete"}, replace=True)
            self.assertEqual(json.loads(path.read_text()), {"state": "complete"})

    def test_validation_only_does_not_import_science(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            runner.write_json(path, synthetic_plan())
            with patch.object(runner, "verify_source", side_effect=AssertionError("source inspection forbidden")):
                self.assertEqual(runner.main(["--plan", str(path), "--validate-only"]), 0)

    def test_actual_execution_requires_slurm_before_scientific_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            runner.write_json(path, synthetic_plan())
            with patch.dict(runner.os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "requires a Slurm"):
                    runner.main(["--plan", str(path), "--array-index", "0", "--out",
                                 str(Path(temp) / "out"), "--deadline-epoch", "9999999999"])
            self.assertFalse((Path(temp) / "out").exists())


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np

    def test_failed_endpoint_overflow_is_retained_without_fallback(self):
        trace = types.SimpleNamespace(status="failed", final_factor=self.np.ones((16, 1)))
        data = types.SimpleNamespace(m=128)
        def overflow(*args):
            raise FloatingPointError("synthetic overflow")
        metrics, errors = runner.independent_endpoint(data, trace, self.np.zeros(128), self.np, overflow)
        self.assertTrue(all(metrics[key] is None for key in runner.PHYSICAL_METRICS))
        self.assertIn("synthetic overflow", errors["full_design_metrics"])
        self.assertAlmostEqual(metrics["factor_operator_norm"], 4.0)
        self.assertFalse(runner.gate_for(trace.status, metrics, runner.SHARED["gate"])["pass"])
        trace.status = "complete"
        with self.assertRaises(FloatingPointError):
            runner.independent_endpoint(data, trace, self.np.zeros(128), self.np, overflow)

    def test_unexpected_endpoint_exception_not_classified_as_numerical_failure(self):
        trace = types.SimpleNamespace(status="failed", final_factor=self.np.ones((16, 1)))
        def broken(*args):
            raise RuntimeError("infrastructure or implementation error")
        with self.assertRaises(RuntimeError):
            runner.independent_endpoint(types.SimpleNamespace(m=128), trace,
                                        self.np.zeros(128), self.np, broken)

    def _archive(self, directory, *, failed=False, corrupt_steps=False):
        trial = synthetic_plan()["trials"][0]
        steps = self.np.array([0, 200]) if failed else self.np.arange(0, 10001, 100)
        if corrupt_steps:
            steps[-1] -= 1
        work = steps * 128
        endpoint = {"status": "failed" if failed else "complete",
                    "failure_reason": "synthetic failure" if failed else None,
                    "completed_steps": int(steps[-1]),
                    "cumulative_data_row_accesses": int(work[-1])}
        metadata = {"methods": {"printed-adafgd": endpoint},
                    "diagnostic": {"trial": trial, "gate": {"pass": False}}}
        payload = {"metadata_json": self.np.asarray(json.dumps(metadata)),
                   "printed_adafgd_checkpoint_steps": steps,
                   "printed_adafgd_cumulative_data_row_accesses": work,
                   "printed_adafgd_final_factor": self.np.zeros((16, 1)),
                   "active_rows": self.np.arange(128),
                   "active_observations": self.np.zeros(128)}
        npz, js = Path(directory) / "result.npz", Path(directory) / "result.json"
        self.np.savez(npz, **payload)
        runner.write_json(js, metadata)
        return npz, js, trial

    def test_complete_and_failed_archives_validate(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temp:
                npz, js, trial = self._archive(temp, failed=failed)
                self.assertEqual(runner.validate_saved(npz, js, trial, self.np)["methods"][trial["method"]]["status"],
                                 "failed" if failed else "complete")

    def test_early_complete_archive_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            npz, js, trial = self._archive(temp, corrupt_steps=True)
            with self.assertRaisesRegex(ValueError, "stopped early"):
                runner.validate_saved(npz, js, trial, self.np)


if __name__ == "__main__":
    unittest.main()
