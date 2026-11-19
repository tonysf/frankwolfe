"""Small arithmetic checks for weighting and native work-axis alignment."""
import hashlib
import json
from pathlib import Path

import numpy as np
import mean_sd_plots as plotting


def run(values, work):
    return {"trace": [{"iteration": k, "forward_views": x, "oracle_calls": x,
                       "raw_loss": v, "feasible_loss": v,
                       "validation_loss": v, "d_distance": 0.0}
                      for k, (x, v) in enumerate(zip(work, values))]}


def main():
    groups = {"a": "group1", "b": "group1", "c": "group2"}
    slices = list(groups)
    runs = {("a", 1, 360, "test"): run([1, 2, 3], [0, 10, 30]),
            ("a", 2, 360, "test"): run([1, 4, 7], [0, 20, 40]),
            ("b", 1, 360, "test"): run([1, 6, 9], [0, 10, 30]),
            ("c", 1, 360, "test"): run([1, 8, 15], [0, 5, 50])}
    _, weights = plotting.select_and_weight(runs, slices, groups, 360, "test")
    np.testing.assert_array_equal(weights, [0.125, 0.125, 0.25, 0.5])
    grid, mean, sd = plotting.summarize(runs, slices, groups, 360, "test", "validation_loss", "forward_views")
    np.testing.assert_array_equal(grid, [0, 5, 10, 20, 30])
    # Independent manually aligned arrays verify no future-value lookahead,
    # no linear interpolation, and no continuation beyond shared support.
    manual = np.array([[1, 1, 2, 2, 3], [1, 1, 1, 4, 4],
                       [1, 1, 6, 6, 9], [1, 8, 8, 8, 8]], dtype=float)
    expected_mean = manual[0] / 8 + manual[1] / 8 + manual[2] / 4 + manual[3] / 2
    expected_sd = np.sqrt(sum(w * (v - expected_mean) ** 2 for w, v in zip(weights, manual)))
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=0)
    np.testing.assert_allclose(sd, expected_sd, rtol=0, atol=0)
    iteration, iter_mean, iter_sd = plotting.summarize(runs, slices, groups, 360, "test", "validation_loss", "iteration")
    np.testing.assert_array_equal(iteration, [0, 1, 2])
    np.testing.assert_allclose(iter_mean, [1, 6.25, 11.0], rtol=0, atol=0)
    _, zero, zero_sd = plotting.summarize(runs, slices, groups, 360, "test", "d_distance", "oracle_calls")
    np.testing.assert_array_equal(zero, np.zeros(5))
    np.testing.assert_array_equal(zero_sd, np.zeros(5))
    root = Path(__file__).resolve().parent
    report = dict(status="passed", optimizer_runs_performed=False,
        checks=["equal groups despite unequal slice and repetition counts", "weighted mean and descriptive SD",
                "native step alignment without interpolation or lookahead", "fixed shared observed support",
                "exact integer iteration alignment", "zero feasibility and zero variance preserved"],
        renderer_sha256=hashlib.sha256((root / "mean_sd_plots.py").read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root / "plot_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
