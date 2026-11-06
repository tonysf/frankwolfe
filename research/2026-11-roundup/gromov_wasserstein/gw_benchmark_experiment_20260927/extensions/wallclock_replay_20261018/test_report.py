"""Synthetic-only checks for actual-time aggregation and report validation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("wallclock_report_test_target", HERE/"report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_synthetic_fixture(root):
    config = report.read(HERE/"campaign.json")
    for dataset in report.DATASETS:
        initial = .16 if dataset == "bmmc6224" else .094
        for index, task in enumerate(report.tasks(config, dataset)):
            duration = 120 if index == 0 else 600+index*60
            times = [2+index, 10+index, 60+index, duration]
            trace = [dict(iteration=k, elapsed_seconds=t, rounded_loss=initial-drop)
                     for k,t,drop in zip([0, 1, 2, task["iterations"]], times, [0, .003, .008, .014+index*.0001])]
            meta = dict(status="completed", completed_updates=task["iterations"], target_updates=task["iterations"],
                        method=task["method"], seed=task["seed"], world_size=1,
                        config=dict(task, dtype="float64", epsilon=.001, pairs=1024),
                        dimensions=dict(n=config["instances"][dataset]["n"], m=config["instances"][dataset]["n"]),
                        timing=dict(clock="time.monotonic", fixture="SYNTHETIC TEST DATA ONLY"),
                        elapsed_seconds=duration+2, trace=trace, final_loss=trace[-1]["rounded_loss"])
            path = root/"results"/dataset/task["task_id"]/"metadata.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(meta))
    (root/"SYNTHETIC_FIXTURE_ONLY.txt").write_text("Fabricated timings and losses for report tests; these are not experiment results.\n")
    return config


class ReportTests(unittest.TestCase):
    def test_common_time_grid_never_extrapolates(self):
        traces = [[dict(elapsed_seconds=t, rounded_loss=loss) for t,loss in values] for values in (
            [(1, 10), (3, 8), (5, 6)], [(2, 20), (4, 16), (6, 12)], [(3, 30), (5, 24), (7, 18)])]
        actual = report.common_time_summary(traces)
        np.testing.assert_array_equal(actual["elapsed_seconds"], [3, 4, 5])
        np.testing.assert_allclose(actual["median"], [18, 16, 14])
        np.testing.assert_allclose(actual["minimum"], [8, 7, 6])
        np.testing.assert_allclose(actual["maximum"], [30, 27, 24])

    def test_bad_or_missing_measured_times_rejected(self):
        with tempfile.TemporaryDirectory(prefix="gw_wallclock_SYNTHETIC_FIXTURE_") as directory:
            root = Path(directory)
            config = write_synthetic_fixture(root)
            dataset = "bmmc6224"
            task = report.tasks(config, dataset)[0]
            meta = report.read(root/"results"/dataset/task["task_id"]/"metadata.json")
            for invalid in (None, float("nan"), -1, 2, 1):
                with self.subTest(invalid=invalid):
                    changed = copy.deepcopy(meta)
                    changed["trace"][1]["elapsed_seconds"] = invalid
                    with self.assertRaisesRegex(ValueError, "strictly increasing"):
                        report.validate_trace(changed, task, config, dataset)
            changed = copy.deepcopy(meta)
            del changed["trace"][1]["elapsed_seconds"]
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                report.validate_trace(changed, task, config, dataset)
            changed = copy.deepcopy(meta)
            changed["elapsed_seconds"] = 100
            with self.assertRaisesRegex(ValueError, "exceeds total"):
                report.validate_trace(changed, task, config, dataset)

    def test_synthetic_complete_report_renders_and_records_all_points(self):
        with tempfile.TemporaryDirectory(prefix="gw_wallclock_SYNTHETIC_FIXTURE_") as directory:
            root = Path(directory)
            config = write_synthetic_fixture(root)
            original_titles = report.TITLES.copy()
            try:
                report.TITLES.update({key:"SYNTHETIC FIXTURE ONLY · "+value for key,value in original_titles.items()})
                summary = report.report(root, root/"reports", config)
            finally:
                report.TITLES.update(original_titles)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len((root/"reports"/"traces.csv").read_text().splitlines()), 33)
            for dataset in report.DATASETS:
                self.assertEqual(len(summary["datasets"][dataset]["runs"]), 4)
                for filename in summary["figures"][dataset]:
                    self.assertGreater((root/"reports"/filename).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
