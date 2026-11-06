"""Opt-in stationarity termination must not change other optimizer rules."""

import json

import numpy as np
import pytest

import paper.experiments.qpt_quiroga_campaign as campaign
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.quantum_process_tomography import make_factor_initial_point, unpack_factor


def problem():
    data = generate_quiroga_sensing_data(1, channel_seed=3)
    rows = np.arange(data.m)
    factor = unpack_factor(make_factor_initial_point(data, seed=5), data.process_dimension, 1)
    return data, factor, rows, data.all_observations()


def run(*, gradient_atol=0.0, gradient_rtol=0.0, n_steps=2):
    return campaign.run_armijo_fgd(
        *problem(), n_steps=n_steps, metrics_frequency=2, tp_weight=0.5,
        gradient_atol=gradient_atol, gradient_rtol=gradient_rtol)


def test_disabled_default_and_explicit_zeros_preserve_trace():
    original_default = campaign.run_armijo_fgd(
        *problem(), n_steps=4, metrics_frequency=2, tp_weight=0.5)
    explicit = run(n_steps=4)
    assert original_default.status == explicit.status == "complete"
    for name in ("final_factor", "checkpoint_steps", "cumulative_data_row_accesses",
                 *campaign.COMMON_METRICS):
        np.testing.assert_array_equal(getattr(original_default, name), getattr(explicit, name))
    assert set(original_default.diagnostics) == {
        "step_size", "backtracks", "objective_evaluations"}
    for key in original_default.diagnostics:
        np.testing.assert_array_equal(original_default.diagnostics[key], explicit.diagnostics[key])
    assert "termination" not in original_default.endpoint()


def test_machine_precision_stall_is_only_converged_when_opted_in(monkeypatch):
    data, factor, rows, targets = problem()
    direction = np.zeros_like(factor)
    direction[0, 0] = 1e-16
    monkeypatch.setattr(campaign, "_summed_objective_and_half_gradient",
                        lambda *args: (1e-30, direction))
    trial_calls = []

    def rounded_objective(*args):
        trial_calls.append(True)
        return 1e-30

    monkeypatch.setattr(campaign, "_summed_objective", rounded_objective)
    legacy = run()
    assert legacy.status == "failed"
    assert "31 trials" in legacy.failure_reason
    assert len(trial_calls) == 31
    assert legacy.cumulative_data_row_accesses[-1] == 32 * rows.size

    trial_calls.clear()
    stopped = run(gradient_atol=1e-12)
    assert stopped.status == "converged" and not stopped.failure_reason
    assert not trial_calls
    np.testing.assert_array_equal(stopped.final_factor, factor)
    np.testing.assert_array_equal(stopped.checkpoint_steps, [0])
    assert stopped.cumulative_data_row_accesses[-1] == rows.size
    termination = stopped.endpoint()["termination"]
    assert termination["reason"] == "full_gradient_tolerance"
    assert termination["full_gradient_norm"] == pytest.approx(2e-16, abs=0)
    assert termination["gradient_threshold"] == 1e-12
    assert termination["initial_full_gradient_norm"] == pytest.approx(2e-16, abs=0)


def test_threshold_uses_full_gradient_not_half_direction(monkeypatch):
    direction = np.zeros_like(problem()[1])
    direction[0, 0] = 0.75
    monkeypatch.setattr(campaign, "_summed_objective_and_half_gradient",
                        lambda *args: (1.0, direction))
    monkeypatch.setattr(campaign, "_summed_objective", lambda *args: 0.0)
    assert run(gradient_atol=1.0, n_steps=1).status == "complete"
    stopped = run(gradient_atol=1.5, n_steps=1)
    assert stopped.status == "converged"
    assert stopped.endpoint()["termination"]["full_gradient_norm"] == 1.5


def test_relative_threshold_uses_initial_norm_and_counts_final_gradient(monkeypatch):
    directions = []
    for value in (2.0, 0.25):
        direction = np.zeros_like(problem()[1])
        direction[0, 0] = value
        directions.append(direction)
    calls = iter(directions)
    monkeypatch.setattr(campaign, "_summed_objective_and_half_gradient",
                        lambda *args: (1.0, next(calls)))
    monkeypatch.setattr(campaign, "_summed_objective", lambda *args: 0.0)
    trace = run(gradient_atol=0.125, gradient_rtol=0.25)
    assert trace.status == "converged"
    np.testing.assert_array_equal(trace.checkpoint_steps, [0, 1])
    assert trace.cumulative_data_row_accesses[-1] == 3 * problem()[2].size
    termination = trace.endpoint()["termination"]
    assert termination["initial_full_gradient_norm"] == 4.0
    assert termination["full_gradient_norm"] == 0.5
    assert termination["gradient_threshold"] == 1.125


@pytest.mark.parametrize("key", ["gradient_atol", "gradient_rtol"])
@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan"), True, [1.0]])
def test_invalid_tolerances_are_rejected(key, bad):
    with pytest.raises(ValueError, match="nonnegative finite"):
        run(**{key: bad})


def test_nonfinite_gradient_cannot_be_mistaken_for_convergence(monkeypatch):
    monkeypatch.setattr(campaign, "_summed_objective_and_half_gradient",
                        lambda *args: (1.0, np.full_like(problem()[1], np.nan)))
    trace = run(gradient_atol=1e10)
    assert trace.status == "failed"
    assert "nonfinite" in trace.failure_reason
    assert "termination" not in trace.endpoint()


def test_overflowed_tolerance_threshold_is_failure_not_convergence(monkeypatch):
    monkeypatch.setattr(campaign, "_summed_objective_and_half_gradient",
                        lambda *args: (1.0, np.ones_like(problem()[1])))
    trace = run(gradient_rtol=1e308)
    assert trace.status == "failed"
    assert "stopping threshold is nonfinite" in trace.failure_reason


def test_cli_propagates_and_archives_opt_in_rule(tmp_path):
    archive = tmp_path / "stopped.npz"
    metadata, _, traces = campaign.main([
        "--n-qubits", "1", "--methods", "armijo-fgd", "--steps", "2",
        "--metrics-every", "1", "--eta-scale", "0.001", "--tp-weight", "1",
        "--armijo-gradient-atol", "1e-12", "--armijo-gradient-rtol", "1",
        "--save", str(archive), "--quiet",
    ])
    assert traces["armijo-fgd"].status == "converged"
    assert metadata["configuration"]["armijo_gradient_atol"] == 1e-12
    assert metadata["configuration"]["armijo_gradient_rtol"] == 1.0
    assert metadata["methods"]["armijo-fgd"]["termination"]["reason"] == "full_gradient_tolerance"
    with np.load(archive, allow_pickle=False) as arrays:
        assert all(not arrays[key].dtype.hasobject for key in arrays.files)
        assert json.loads(str(arrays["metadata_json"].item())) == metadata
        assert arrays["armijo_fgd_diagnostic_termination_reason"].item() == "full_gradient_tolerance"


def test_opt_in_armijo_controls_do_not_change_printed_or_frames():
    data = problem()[0]
    kwargs = dict(methods=("printed-adafgd", "frames"), n_steps=2, frames_steps=2,
                  metrics_frequency=1, eta_scale=1e-4, tp_weight=0.5, batch_size=2)
    _, _, original = campaign.run_campaign(data, **kwargs)
    _, _, opted_in = campaign.run_campaign(
        data, **kwargs, armijo_gradient_atol=1.0, armijo_gradient_rtol=1.0)
    for method in original:
        assert original[method].status == opted_in[method].status
        for name in ("final_factor", "checkpoint_steps", "cumulative_data_row_accesses",
                     *campaign.COMMON_METRICS):
            np.testing.assert_array_equal(getattr(original[method], name), getattr(opted_in[method], name))


def test_near_truth_stationarity_is_detected_without_line_search():
    data, _, rows, targets = problem()
    trace = campaign.run_armijo_fgd(
        data, data.truth_factor * (1 + 1e-14), rows, targets,
        n_steps=2, metrics_frequency=1, tp_weight=0.5, gradient_atol=1e-10)
    assert trace.status == "converged"
    assert trace.checkpoint_steps[-1] == 0
    assert trace.rmse[-1] < 1e-12
    assert trace.endpoint()["termination"]["full_gradient_norm"] <= 1e-10
