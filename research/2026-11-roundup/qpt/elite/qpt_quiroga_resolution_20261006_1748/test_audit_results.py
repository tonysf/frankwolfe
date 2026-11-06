"""Synthetic archive-audit tests; no optimizers or remote operations."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit_results", HERE / "audit_results.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np

    def test_first_physical_gate_is_not_terminal_reclassification(self):
        prefix = "armijo_fgd"
        arrays = {prefix + "_" + key: self.np.array([0.1, 0.0, 0.0])
                  for key in audit.runner.PHYSICAL_METRICS}
        arrays[prefix + "_process_fidelity"] = self.np.array([0.1, 1.0, 1.0])
        arrays[prefix + "_checkpoint_steps"] = self.np.array([0, 1100, 3000])
        arrays[prefix + "_cumulative_data_row_accesses"] = self.np.array([0, 300000, 802944])
        first = audit.first_physical_gate(arrays, prefix, audit.runner.SHARED["gate"], self.np)
        self.assertEqual(first["step"], 1100)
        self.assertEqual(first["charged_rows"], 300000)
        self.assertFalse(audit.runner.gate_for("failed", first["checkpoint_metrics"],
                                             audit.runner.SHARED["gate"])["pass"])

    def test_nonfinite_checkpoint_cannot_be_first_gate(self):
        prefix = "printed_adafgd"
        arrays = {prefix + "_" + key: self.np.array([float("nan")])
                  for key in audit.runner.PHYSICAL_METRICS}
        arrays[prefix + "_checkpoint_steps"] = self.np.array([0])
        arrays[prefix + "_cumulative_data_row_accesses"] = self.np.array([0])
        self.assertIsNone(audit.first_physical_gate(arrays, prefix, audit.runner.SHARED["gate"], self.np))

    def test_shape_header_hash_distinguishes_shapes(self):
        a = self.np.array([1.0, 2.0])
        b = a.reshape(2, 1)
        self.assertNotEqual(audit.array_hash(a, self.np), audit.array_hash(b, self.np))
        self.assertEqual(audit.array_hash(a, self.np, shape_header=False),
                         audit.array_hash(b, self.np, shape_header=False))

    def test_complete_inventory_required_but_partial_omission_explicit(self):
        expected = {"trial": {"channel": 0}}
        self.assertEqual(audit.audit_inventories([], expected, {}, {}, False), ([], set()))
        with self.assertRaisesRegex(ValueError, "Both complete"):
            audit.audit_inventories([], expected, {}, {}, True)

    def test_inventory_hash_mismatch_rejected(self):
        trial = {"id": "trial", "channel": 0}
        case = {"status": "failed", "gate": {"pass": False},
                "actual_optimizer_row_accesses": 128, "json_sha256": "j", "npz_sha256": "n"}
        record = {"trial": trial, "status": "archived_method_failure", "method_status": "failed",
                  "gate_pass": False, "actual_optimizer_row_accesses": 128,
                  "json_sha256": "j", "npz_sha256": "wrong"}
        inventory = {"schema_version": 1, "kind": "qpt_diagnostic_inventory",
                     "channel": 0, "state": "complete", "trials": [record]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inventory.json"
            audit.runner.write_json(path, inventory)
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                audit.audit_inventories([path], {"trial": trial}, {"trial": case}, {}, False)

    def test_explicit_schedule_override_in_campaign_configuration(self):
        plan = {"shared": audit.runner.SHARED}
        trial = {"id": "f", "channel": 0, "method": "frames", "alpha": 5e-6,
                 "eta_scale": 0.01, "tau": 2,
                 "step_schedule": {"scale": 4, "offset": 4, "exponent": 1, "cap": 1}}
        config = audit.expected_campaign_configuration(plan, trial)
        self.assertEqual(config["step_size_schedule"], {"type": "PowerSchedule", **trial["step_schedule"]})
        self.assertAlmostEqual(config["frames_beta0"], 100000)
        self.assertEqual(config["tp_weight_lambda"], 128 * 5e-6)
        self.assertNotIn("show_progress", config)


if __name__ == "__main__":
    unittest.main()
