"""Synthetic-only tests; never read or rank the actual calibration outcomes."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyzer_under_test", Path(__file__).with_name("analyze_calibration.py"))
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def plan():
    return {"channels": [0, 1], "initialization_indices": [0], "sampling_indices": [0, 1, 2],
            "printed_candidates": [{"id": "q-a", "alpha": 0.05, "eta_scale": 0.001},
                                   {"id": "q-b", "alpha": 0.005, "eta_scale": 0.003}],
            "frames_candidates": [{"id": "f-a", "alpha0": 0.005},
                                  {"id": "f-b", "alpha0": 0.0005}]}


def outcome(method, candidate, channel, sample, *, passed=False, fidelity=0.5,
            rmse=0.1, tp=0.1, status="complete", label=None):
    return {"method": method, "config_id": candidate, "channel": channel,
            "initialization_index": 0, "sampling_index": sample,
            "status": status, "failure_reason": "numerical failure" if status == "failed" else None,
            "gate_passed": passed,
            "gate_components": {"process_fidelity": passed, "rmse": passed,
                                "normalized_tp_violation": passed},
            "metrics": {"process_fidelity": fidelity, "rmse": rmse, "normalized_tp_violation": tp},
            "rmse_source": "full_design", "data_instance_sha256": f"data-{channel}",
            "paired_instance_sha256": f"pair-{channel}", "initial_factor_sha256": f"u0-{channel}",
            "active_rows_sha256": "rows", "active_observations_sha256": f"y-{channel}",
            "full_observations_sha256": f"y-{channel}", "truth_factor_sha256": f"truth-{channel}",
            "endpoint_without_timings": {"status": status},
            "endpoint_diagnostics": {"final_factor_sha256": f"factor-{method}-{candidate}-{channel}-{sample}"},
            "source_trial_ids": [label or f"{method}-{candidate}-{channel}-{sample}"]}


def all_outcomes(p):
    rows = []
    for method, candidates, samples in (
        ("printed-adafgd", p["printed_candidates"], [None]),
        ("frames", p["frames_candidates"], p["sampling_indices"])):
        rows.extend(outcome(method, c["id"], channel, sample)
                    for c in candidates for channel in p["channels"] for sample in samples)
    return rows


def test_native_repetitions_are_verified_then_deduplicated():
    p = plan()
    rows = all_outcomes(p)
    duplicate = copy.deepcopy(rows[0])
    duplicate["source_trial_ids"] = ["opposite-method-anchor-copy"]
    unique = analyzer.deduplicate(rows + [duplicate], p)
    assert len(unique) == 16
    repeated = next(r for r in unique if r["method"] == "printed-adafgd" and r["config_id"] == "q-a"
                    and r["channel"] == 0)
    assert len(repeated["source_trial_ids"]) == 2
    duplicate["endpoint_diagnostics"]["final_factor_sha256"] = "changed"
    with pytest.raises(ValueError, match="Repeated native endpoint"):
        analyzer.deduplicate(rows + [duplicate], p)


def test_missing_native_replication_and_unpaired_data_are_rejected():
    p = plan()
    rows = all_outcomes(p)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        analyzer.deduplicate(rows[:-1], p)
    rows[-1]["initial_factor_sha256"] = "not-shared-u0"
    with pytest.raises(ValueError, match="do not share data/U0"):
        analyzer.deduplicate(rows, p)


def test_sampling_means_precede_equal_channel_median():
    p = plan()
    candidate = p["frames_candidates"][0]
    rows = [outcome("frames", candidate["id"], channel, sample,
                    fidelity=values[sample], passed=(channel == 1 or sample == 0))
            for channel, values in ((0, [0.2, 0.8, 0.5]), (1, [0.7, 0.9, 1.0]))
            for sample in range(3)]
    _, result = analyzer.candidate_summary(rows, p, "frames", candidate)
    assert result["gate_pass_rate"] == pytest.approx(2 / 3)
    assert result["mean_gate_components_passed"] == pytest.approx(2)
    median = result["physical_ranking_channel_medians"]["process_fidelity"]["value"]
    assert median == pytest.approx((0.5 + 2.6 / 3) / 2)
    assert median != pytest.approx(0.75)  # pooled-stream median would differ


def test_failure_denominator_and_missing_metrics_are_conservative_not_omitted():
    p = plan()
    candidate = p["frames_candidates"][0]
    rows = [outcome("frames", candidate["id"], channel, sample, passed=True,
                    fidelity=1, rmse=0, tp=0) for channel in p["channels"] for sample in range(3)]
    failed = rows[0]
    failed.update(status="failed", failure_reason="overflow", gate_passed=False,
                  gate_components={key: False for key in analyzer.METRICS},
                  metrics={key: None for key in analyzer.METRICS})
    key, result = analyzer.candidate_summary(rows, p, "frames", candidate)
    assert result["unique_native_outcomes"] == 6
    assert result["status_counts"] == {"failed": 1, "complete": 5}
    assert result["gate_pass_rate"] == pytest.approx(5 / 6)
    assert result["physical_ranking_channel_medians"]["process_fidelity"]["value"] == pytest.approx(5 / 6)
    assert key[3] == math.inf and key[4] == math.inf
    assert result["physical_ranking_channel_medians"]["rmse"] == {
        "value": None, "ranking_sentinel": "+infinity"}
    assert result["undefined_metric_counts"] == dict.fromkeys(analyzer.METRICS, 1)
    json.dumps(result, allow_nan=False)


def test_finite_failed_diagnostics_are_preserved_but_gate_is_false():
    p = plan()
    candidate = p["printed_candidates"][0]
    rows = [outcome("printed-adafgd", candidate["id"], channel, None,
                    status="failed", passed=False, fidelity=0.95, rmse=0.002, tp=0.003)
            for channel in p["channels"]]
    _, result = analyzer.candidate_summary(rows, p, "printed-adafgd", candidate)
    assert result["gate_pass_rate"] == 0
    assert result["physical_ranking_channel_medians"]["process_fidelity"]["value"] == 0.95
    assert result["undefined_metric_counts"] == dict.fromkeys(analyzer.METRICS, 0)


def test_methods_select_independently_and_config_id_breaks_exact_ties():
    p = plan()
    rows = all_outcomes(p)
    for row in rows:
        if row["method"] == "printed-adafgd" and row["config_id"] == "q-b":
            row["gate_passed"] = True
            row["gate_components"] = dict.fromkeys(analyzer.METRICS, True)
    summary, chosen = analyzer.select_candidates(rows, p)
    assert chosen["printed-adafgd"]["id"] == "q-b"
    assert chosen["frames"]["id"] == "f-a"
    assert summary["candidates"]["printed-adafgd"][0]["rank"] == 1


def test_gate_priority_precedes_secondary_metrics_and_never_uses_objective():
    p = plan()
    rows = all_outcomes(p)
    for row in rows:
        row["native_penalized_objective"] = -1e300 if row["config_id"].endswith("a") else 1e300
        if row["config_id"] == "q-b":
            row["gate_passed"] = True
            row["metrics"]["process_fidelity"] = 0.1  # synthetic inconsistent gate for sort-priority test only
    _, chosen = analyzer.select_candidates(rows, p)
    assert chosen["printed-adafgd"]["id"] == "q-b"


def test_freeze_refuses_incomplete_calibration_without_writing(tmp_path):
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Cannot freeze incomplete"):
        analyzer.freeze_selection(destination, {}, {"status": "incomplete", "paired_trial_count": 351}, [])
    assert not destination.exists()


def test_safe_infinite_encoding_and_stable_large_metric_arithmetic():
    assert analyzer.rank_value(None, "process_fidelity") == 0
    assert analyzer.rank_value(float("nan"), "rmse") == math.inf
    assert analyzer.mean([1e308, 1e308]) == 1e308
    assert analyzer.median([1e308, 1e308]) == 1e308
    assert analyzer.encode_rank(math.inf) == {"value": None, "ranking_sentinel": "+infinity"}
