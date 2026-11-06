"""Numerical failure must not abort the paired archive's diagnostic pass."""

import json

import numpy as np
import pytest

from paper.experiments import qpt_quiroga_campaign as campaign
from paper.experiments import qpt_quiroga_direct_comparison as direct
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data


def _small_pair(*, sampling_seed=0, trial_id="failure-test", rows=None):
    data = generate_quiroga_sensing_data(1, channel_seed=0)
    return direct.run_direct_comparison(
        data, rows=rows, row_epochs=2, checkpoint_epochs=1,
        printed_tp_weight_per_row=0.05, frames_initial_tp_weight_per_row=0.005,
        eta_scale=0.0001, batch_size=4, spectral_norm_method="dense",
        initialization_seed=200000, sampling_seed=sampling_seed,
        trial_id=trial_id, cohort_id="failure-archive-test", split="calibration")


def test_actual_aggressive_printed_run_archives_failure_and_intact_frames(tmp_path):
    archive = tmp_path / "aggressive.npz"
    # The production candidate fails well before this reduced 100-epoch budget.
    # All sensing, initial-factor, penalty and adaptive-step controls are exact.
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(SystemExit) as error:
        direct.main([
            "--n-qubits", "2", "--row-epochs", "100", "--checkpoint-epochs", "100",
            "--printed-tp-weight-per-row", "0.5",
            "--frames-initial-tp-weight-per-row", "0.005",
            "--frames-penalty-exponent", "0.25", "--eta-scale", "0.03",
            "--rank", "1", "--tau", "2", "--batch-size", "32",
            "--channel-seed", "0", "--initialization-seed", "200000",
            "--sampling-seed", "300000", "--subset-seed", "400000",
            "--spectral-norm-method", "dense", "--split", "calibration",
            "--cohort-id", "real-overflow-regression", "--trial-id", "real-p05",
            "--save", str(archive), "--quiet", "--fail-on-method-failure"])
    assert error.value.code == 2
    metadata = json.loads(archive.with_suffix(".json").read_text())
    printed, frames = (metadata["methods"][name] for name in direct.DIRECT_METHODS)
    assert printed["status"] == "failed" and printed["failure_reason"]
    assert printed["completed_steps"] < 100
    assert frames["status"] == "complete" and frames["completed_steps"] == 400
    diagnostics = metadata["comparison"]["endpoint_diagnostics"]["printed-adafgd"]
    assert all(value is None for value in diagnostics["full_design_metrics"].values())
    assert diagnostics["unavailable_diagnostic_reasons"]["full_design_metrics"]
    gate = metadata["comparison"]["method_gate"]["printed-adafgd"]
    assert not gate["status_eligible"] and not gate["passed"]
    assert gate["metric_values"]["rmse"] is None
    with np.load(archive, allow_pickle=False) as payload:
        assert json.loads(payload["metadata_json"].item()) == metadata
        factor = payload["printed_adafgd_final_factor"]
        assert np.isfinite(factor).all()
        assert direct._array_sha256(factor) == diagnostics["final_factor_sha256"]
        assert not np.array_equal(factor, payload["initial_factor"])
        assert np.isfinite(payload["frames_final_factor"]).all()
    summary = direct.aggregate_sidecars([metadata])
    assert summary["sampling_trial_count"] == 1
    assert summary["methods"]["printed-adafgd"][
        "sampling_trial_status_counts_descriptive"] == {"failed": 1}
    assert summary["methods"]["printed-adafgd"][
        "sampling_trial_gate_descriptive"] == {"pass_count": 0, "pass_rate": 0.0}
    json.dumps(metadata, allow_nan=False)


def test_unavailable_failed_norms_and_full_metrics_preserve_factor_and_denominator(monkeypatch):
    original = campaign.run_printed_adafgd

    def failed_retained_factor(*args, **kwargs):
        trace = original(*args, **kwargs)
        trace.status = "failed"
        trace.failure_reason = "synthetic finite factor with overflowing diagnostics"
        trace.final_factor = np.full_like(trace.final_factor, 1e200 + 0j)
        for metric in campaign.COMMON_METRICS:
            getattr(trace, metric)[-1] = np.nan
        return trace

    monkeypatch.setattr(campaign, "run_printed_adafgd", failed_retained_factor)
    original_norm = direct.np.linalg.norm

    def unavailable_operator_norm(array, *args, **kwargs):
        if kwargs.get("ord") == 2 and np.max(np.abs(array)) > 1e190:
            raise np.linalg.LinAlgError("synthetic endpoint SVD failure")
        return original_norm(array, *args, **kwargs)

    monkeypatch.setattr(direct.np.linalg, "norm", unavailable_operator_norm)
    with np.errstate(over="ignore", invalid="ignore"):
        first, payload, traces = _small_pair(sampling_seed=0, trial_id="failed-q-s0")
        second, _, _ = _small_pair(sampling_seed=1, trial_id="failed-q-s1")
    diagnostics = first["comparison"]["endpoint_diagnostics"]["printed-adafgd"]
    assert diagnostics["factor_frobenius_norm"] is None
    assert diagnostics["factor_operator_norm"] is None
    assert diagnostics["factor_operator_norm_over_tau"] is None
    assert set(diagnostics["unavailable_diagnostic_reasons"]) == {
        "full_design_metrics", "factor_frobenius_norm", "factor_operator_norm"}
    np.testing.assert_array_equal(payload["printed_adafgd_final_factor"],
                                  traces["printed-adafgd"].final_factor)
    assert first["methods"]["frames"]["status"] == "complete"
    summary = direct.aggregate_sidecars([first, second])
    assert summary["sampling_trial_count"] == 2
    assert summary["optimization_instance_count"] == 1
    assert summary["methods"]["printed-adafgd"][
        "sampling_trial_status_counts_descriptive"] == {"failed": 2}
    assert summary["methods"]["printed-adafgd"][
        "data_instance_gate"]["cohort_nested_pass_rate"] == 0
    json.dumps(first, allow_nan=False)


def test_missing_full_design_rmse_does_not_use_failed_subset_training_rmse(monkeypatch):
    original_runner = campaign.run_printed_adafgd
    original_metrics = direct.evaluate_common_metrics

    def mark_failed(*args, **kwargs):
        trace = original_runner(*args, **kwargs)
        trace.status = "failed"
        trace.failure_reason = "synthetic failed optimizer"
        return trace

    def unavailable_printed_full_metrics(data, factor, rows, targets, tp_weight):
        if tp_weight == 0.4:
            raise FloatingPointError("synthetic full-design-only overflow")
        return original_metrics(data, factor, rows, targets, tp_weight)

    monkeypatch.setattr(campaign, "run_printed_adafgd", mark_failed)
    monkeypatch.setattr(direct, "evaluate_common_metrics", unavailable_printed_full_metrics)
    metadata, _, _ = _small_pair(rows=np.arange(8, dtype=np.int64))
    assert metadata["methods"]["printed-adafgd"]["rmse"] is not None
    gate = metadata["comparison"]["method_gate"]["printed-adafgd"]
    assert gate["rmse_source"] == "full_design" and gate["metric_values"]["rmse"] is None
    assert not gate["component_passes"]["rmse"] and not gate["passed"]
    direct.aggregate_sidecars([metadata])


def test_complete_method_diagnostic_errors_are_not_hidden(monkeypatch):
    def fail_full_metrics(*args, **kwargs):
        raise FloatingPointError("unexpected complete endpoint overflow")

    monkeypatch.setattr(direct, "evaluate_common_metrics", fail_full_metrics)
    with pytest.raises(FloatingPointError, match="unexpected complete endpoint"):
        _small_pair()


def test_complete_method_norm_errors_are_not_hidden():
    class CompleteTrace:
        status = "complete"
        final_factor = np.array([[1e200 + 0j]])

    with pytest.raises(FloatingPointError):
        direct._endpoint_factor_norm(CompleteTrace(), "fro", "norm", {})
