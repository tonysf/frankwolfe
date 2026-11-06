"""Tiny local checks; no experiment, scheduler, or external archive dependency."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import freeze_plan
import paired_reference
import run_diagnostic as runner
import support
from work_accounting import check_work


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.plan = freeze_plan.build_plan("a" * 40, "synthetic")

    def test_counts_budgets_and_seeds(self):
        cases = runner.validate_plan(self.plan)
        self.assertEqual(len(cases), 8)
        self.assertEqual([sum(t["method"] == m for t in cases) for m in support.METHODS], [4, 2, 2])
        for trial in cases:
            cfg = runner.configuration(self.plan, trial)
            self.assertEqual(cfg["initialization_basis"], "gell-mann")
            self.assertEqual(cfg["initialization_seed"], 200000 + 10 * trial["channel"])
            self.assertEqual(cfg["sampling_seed"], 300000 + 100 * trial["channel"])
            if trial["method"] == "armijo-fgd":
                self.assertEqual((cfg["n_steps"], cfg["metrics_frequency"]), (10000, 100))
                self.assertEqual((cfg["armijo_gradient_atol"], cfg["armijo_gradient_rtol"]), (1e-12, 1e-10))
            else:
                self.assertEqual((cfg["n_steps"], cfg["frames_steps"]), (100000, 400000))
                self.assertEqual((cfg["metrics_frequency"], cfg["frames_metrics_frequency"]), (1000, 4000))

    def test_frozen_plan_rejects_changes(self):
        for field, value in (("channel", 100), ("alpha", .5), ("row_epochs", 1)):
            plan = copy.deepcopy(self.plan)
            plan["trials"][0][field] = value
            with self.assertRaises(ValueError):
                runner.validate_plan(plan)
        plan = copy.deepcopy(self.plan)
        plan["resources"]["minimum_seconds_before_new_trial"] = 120
        with self.assertRaises(ValueError):
            runner.validate_plan(plan)

    def test_old_commit_rejected(self):
        for commit in (freeze_plan.PRIOR_COMMIT, "bad", "a" * 39):
            with self.assertRaises(ValueError):
                freeze_plan.build_plan(commit, "synthetic")

    def test_validate_only_and_slurm_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            support.write_json(plan, self.plan)
            with patch.object(runner, "source_hashes", side_effect=AssertionError):
                self.assertEqual(runner.main(["--plan", str(plan), "--validate-only"]), 0)
            with patch.dict(runner.os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "requires Slurm"):
                    runner.main(["--plan", str(plan)])

    def test_gate_does_not_relabel_failed_physical_success(self):
        metrics = {"process_fidelity": 1., "rmse": 0., "normalized_tp_violation": 0.}
        for status in ("complete", "converged", "failed"):
            gate = support.gate_for(status, metrics, self.plan["shared"]["gate"])
            self.assertEqual(gate["pass"], status != "failed")

    def test_json_no_overwrite_and_atomic_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            support.write_json(path, {"state": "pending"})
            with self.assertRaises(FileExistsError):
                support.write_json(path, {})
            support.write_json(path, {"state": "complete"}, replace=True)
            self.assertEqual(json.loads(path.read_text()), {"state": "complete"})

    def test_independent_dense_basis_and_actual_configuration(self):
        from paper.experiments.qpt_quiroga_initialization import gell_mann_coefficients_to_choi
        from paper.experiments.qpt_quiroga_campaign import run_campaign
        from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
        from paper.experiments.quantum_process_tomography import PowerSchedule
        coefficients = np.arange(16).reshape(16, 1) + 1j
        w = paired_reference.dense_gell_mann(np)
        np.testing.assert_allclose(w.conj().T @ w, np.eye(16), atol=1e-14)
        np.testing.assert_allclose(gell_mann_coefficients_to_choi(coefficients), w @ coefficients)
        cfg = runner.configuration(self.plan, self.plan["trials"][0])
        cfg.update(n_steps=1, frames_steps=1)
        kwargs = dict(cfg)
        for key in ("rho_schedule", "step_size_schedule", "frames_smoothing_schedule"):
            kwargs[key] = PowerSchedule(**kwargs[key])
        metadata, _, _ = run_campaign(generate_quiroga_sensing_data(2), **kwargs)
        self.assertEqual(metadata["configuration"], runner.actual_configuration(cfg))

    def test_paired_reference_and_config_tamper(self):
        trial = self.plan["trials"][0]
        rng = np.random.default_rng(200000)
        coefficients = rng.uniform(size=(16, 1)) + 1j * rng.uniform(size=(16, 1))
        coefficients *= 2 / np.linalg.norm(coefficients, ord="fro")
        arrays = {"active_rows": np.arange(128), "active_observations": np.zeros(128),
                  "truth_factor": coefficients.copy(), "initial_factor": coefficients.copy()}
        oldcfg = runner.configuration(self.plan, trial)
        del oldcfg["initialization_basis"]
        old = {"configuration": oldcfg, "diagnostic": {"requested_configuration": oldcfg}}
        newcfg = {**oldcfg, "initialization_basis": "gell-mann"}
        new = {"configuration": newcfg, "diagnostic": {"requested_configuration": newcfg}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            support.write_json(path, old)
            np.savez(path.with_suffix(".npz"), **arrays, metadata_json=json.dumps(old))
            case = {"trial_id": trial["id"], "trial": trial, "status": "complete",
                    "endpoint_physical": {}, "gate": {}, "completed_steps": 0,
                    "actual_optimizer_row_accesses": 0, "json_path": str(path),
                    "json_sha256": support.file_hash(path), "npz_sha256": support.file_hash(path.with_suffix(".npz"))}
            arrays["initial_factor"] = paired_reference.dense_gell_mann(np) @ coefficients
            result = paired_reference.check_prior({trial["id"]: case}, trial, arrays, new, np)
            self.assertEqual(result["dense_initializer_max_absolute_error"], 0.)
            changed = copy.deepcopy(new)
            changed["configuration"]["eta_scale"] = 3.
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                paired_reference.check_prior({trial["id"]: case}, trial, arrays, changed, np)
            arrays["initial_factor"] = coefficients
            with self.assertRaisesRegex(ValueError, "Initializer differs"):
                paired_reference.check_prior({trial["id"]: case}, trial, arrays, new, np)

    def test_work_terminal_gradient(self):
        prefix = "armijo_fgd"
        arrays = {prefix + "_checkpoint_steps": np.array([0, 1]),
                  prefix + "_cumulative_data_row_accesses": np.array([0, 512]),
                  prefix + "_optimizer_seconds": np.array([0., 1.]),
                  prefix + "_diagnostic_objective_evaluations": np.array([2])}
        self.assertEqual(check_work(arrays, prefix, "armijo-fgd", "converged", None, np)["terminal_attempt_rows"], 128)
        with self.assertRaises(ValueError):
            check_work(arrays, prefix, "armijo-fgd", "complete", None, np)


if __name__ == "__main__":
    unittest.main()
