"""Small algebra, stopping, accounting and archive tests for a named variant."""

import json

import numpy as np
import pytest

from paper.experiments import qpt_quiroga_campaign as campaign
from paper.experiments import qpt_quiroga_linesearch as search
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.quantum_process_tomography import make_factor_initial_point, unpack_factor


def problem(n=1, rank=1, subset=False):
    data = generate_quiroga_sensing_data(n, channel_seed=3)
    rows = (data.fixed_row_subset(data.m - 3, seed=7) if subset
            else np.arange(data.m, dtype=np.int64))
    targets = data.observations_for_rows(rows)
    factor = unpack_factor(make_factor_initial_point(data, rank=rank, seed=5),
                           data.process_dimension, rank)
    return data, rows, targets, factor


def dense_objective(data, matrices, factor, targets, weight):
    choi = factor @ factor.conj().T
    residual = np.einsum("aij,ji->a", matrices, choi).real - targets
    partial = np.einsum("iojo->ij", choi.reshape(data.d, data.d, data.d, data.d))
    tp = partial - np.eye(data.d)
    return 0.5 * residual @ residual + weight * np.vdot(tp, tp).real


@pytest.mark.parametrize("n,rank,subset", [(1, 1, False), (1, 2, True), (2, 1, True)])
def test_exact_polynomial_slope_and_positive_ray_global_minimum(n, rank, subset):
    data, rows, targets, factor = problem(n, rank, subset)
    weight = 0.03
    objective, direction = campaign._summed_objective_and_half_gradient(
        data, factor, rows, targets, weight)
    norm = np.linalg.norm(direction)
    ray = direction / norm
    coefficients = search.factor_ray_coefficients(data, factor, ray, rows, targets, weight)
    assert coefficients[0] == pytest.approx(objective, abs=1e-14)
    assert coefficients[1] == pytest.approx(-2 * norm, rel=1e-12)
    matrices = data.design.dense_sensing_matrices(rows)
    roots = search.positive_stationary_points(coefficients)
    assert roots.size > 0
    for distance in (-0.5, 0.0, 0.1, 0.8, 2.0, *roots):
        expected = dense_objective(data, matrices, factor - distance * ray, targets, weight)
        assert np.polynomial.polynomial.polyval(distance, coefficients) == pytest.approx(
            expected, rel=1e-11, abs=2e-13)
    values = [dense_objective(data, matrices, factor - distance * ray, targets, weight)
              for distance in roots]
    selected = roots[np.argmin(values)]
    grid = np.linspace(0, max(4.0, 1.5 * roots.max()), 101)
    assert min(values) < objective
    assert min(values) <= min(dense_objective(data, matrices, factor - t * ray,
                                            targets, weight) for t in grid) + 1e-13
    trace = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1, tp_weight=weight)
    assert trace.status == "complete"
    np.testing.assert_allclose(trace.final_factor, factor - selected * ray, atol=1e-13)
    assert trace.diagnostics["step_size"][0] == pytest.approx(selected / norm)


def test_stationary_roots_include_both_positive_minima():
    # (t-1)^2(t-3)^2: two minima and the intervening maximum.
    coefficients = np.polynomial.polynomial.polymul([1, -2, 1], [9, -6, 1])
    np.testing.assert_allclose(search.positive_stationary_points(coefficients), [1, 2, 3])
    assert search.positive_stationary_points([1, 0, 0, 0, 0]).size == 0
    assert search.positive_stationary_points([1, 1, 0, 0, 0]).size == 0


@pytest.mark.parametrize("coefficients", [[1, 2], [1, 2, np.nan, 3, 4]])
def test_bad_coefficients_rejected(coefficients):
    with pytest.raises(ValueError, match="five finite"):
        search.positive_stationary_points(coefficients)


def assert_work_reconstruction(trace, m):
    diag = trace.diagnostics
    cumulative = np.concatenate(([0], np.cumsum(
        diag["direction_passes"] + diag["polynomial_forward_passes"]
        + diag["objective_evaluations"]))) * m
    expected = cumulative[trace.checkpoint_steps]
    expected[-1] += int(diag["terminal_attempt_rows"][0])
    np.testing.assert_array_equal(trace.cumulative_data_row_accesses, expected)
    assert diag["terminal_attempt_rows"][0] == m * sum(
        diag[f"terminal_{key}"][0] for key in
        ("direction_passes", "polynomial_forward_passes", "objective_evaluations"))


def test_budget_completion_and_checkpoint_work_are_exact():
    data, rows, targets, factor = problem()
    trace = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=7, metrics_frequency=3, tp_weight=0.01,
        gradient_atol=0, gradient_rtol=0)
    assert trace.status == "complete"
    np.testing.assert_array_equal(trace.checkpoint_steps, [0, 3, 6, 7])
    assert np.all(np.diff(trace.penalized_objective) < 0)
    np.testing.assert_array_equal(trace.diagnostics["direction_passes"], np.ones(7))
    np.testing.assert_array_equal(trace.diagnostics["polynomial_forward_passes"], np.full(7, 2))
    assert trace.diagnostics["terminal_attempt_rows"][0] == 0
    assert_work_reconstruction(trace, rows.size)


def test_true_stationarity_stops_before_line_search_and_charges_terminal_pass():
    data, rows, targets, _ = problem()
    trace = search.run_exact_linesearch_fgd(
        data, data.truth_factor, rows, targets, n_steps=3, metrics_frequency=1, tp_weight=0.01)
    assert trace.status == "converged"
    np.testing.assert_array_equal(trace.checkpoint_steps, [0])
    assert trace.diagnostics["termination_reason"][0] == "full_gradient_tolerance"
    assert trace.diagnostics["terminal_direction_passes"][0] == 1
    assert trace.diagnostics["terminal_objective_evaluations"][0] == 0
    assert trace.diagnostics["terminal_polynomial_forward_passes"][0] == 0
    assert_work_reconstruction(trace, rows.size)


def test_near_truth_tiny_gradient_still_takes_a_real_decreasing_step():
    data, rows, targets, _ = problem()
    rng = np.random.default_rng(33)
    perturbation = rng.normal(size=data.truth_factor.shape) + 1j * rng.normal(size=data.truth_factor.shape)
    factor = data.truth_factor + 1e-10 * perturbation
    trace = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1, tp_weight=0.01,
        gradient_atol=1e-14, gradient_rtol=0)
    assert trace.status == "complete", trace.failure_reason
    assert trace.penalized_objective[-1] < trace.penalized_objective[0]
    assert trace.diagnostics["line_search_distance"][0] < 1e-8
    assert trace.diagnostics["step_size"][0] > 0
    assert_work_reconstruction(trace, rows.size)


@pytest.mark.parametrize("frequency", [1, 2])
def test_terminal_gradient_pass_is_retained_after_an_accepted_update(frequency):
    data, rows, targets, factor = problem()
    first = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1, tp_weight=0.01)
    initial_norm = first.diagnostics["full_gradient_norm"][0]
    _, final_direction = campaign._summed_objective_and_half_gradient(
        data, first.final_factor, rows, targets, 0.01)
    final_norm = 2 * np.linalg.norm(final_direction)
    assert final_norm < initial_norm
    trace = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=3, metrics_frequency=frequency, tp_weight=0.01,
        gradient_atol=(initial_norm + final_norm) / 2, gradient_rtol=0)
    assert trace.status == "converged"
    assert trace.checkpoint_steps[-1] == 1
    assert trace.diagnostics["terminal_attempt_rows"][0] == rows.size
    np.testing.assert_array_equal(trace.final_factor, first.final_factor)
    assert_work_reconstruction(trace, rows.size)


def test_failed_polynomial_forward_is_charged_without_recording_an_update(monkeypatch):
    data, rows, targets, factor = problem()
    original = type(data.design).full_values
    calls = 0

    def fail_on_second_polynomial_forward(self, value, **kwargs):
        nonlocal calls
        calls += 1
        # Initial checkpoint, current prediction, V, U+V.
        if calls == 4:
            raise RuntimeError("injected forward failure")
        return original(self, value, **kwargs)

    monkeypatch.setattr(type(data.design), "full_values", fail_on_second_polynomial_forward)
    trace = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1, tp_weight=0.01)
    assert trace.status == "failed"
    assert "injected forward failure" in trace.failure_reason
    assert trace.diagnostics["terminal_polynomial_forward_passes"][0] == 2
    assert trace.diagnostics["objective_evaluations"].size == 0
    assert_work_reconstruction(trace, rows.size)


def test_no_root_failure_retains_last_factor_and_all_attempted_work(monkeypatch):
    data, rows, targets, factor = problem()
    monkeypatch.setattr(search, "positive_stationary_points", lambda coefficients: np.array([]))
    trace = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=3, metrics_frequency=2, tp_weight=0.01)
    assert trace.status == "failed"
    assert "no fallback" in trace.failure_reason
    np.testing.assert_array_equal(trace.final_factor, factor)
    assert trace.diagnostics["step_size"].size == 0
    assert trace.diagnostics["terminal_attempt_rows"][0] == 3 * rows.size
    assert_work_reconstruction(trace, rows.size)


def test_non_decreasing_roots_are_failure_not_zero_step_success(monkeypatch):
    data, rows, targets, factor = problem()
    monkeypatch.setattr(search, "positive_stationary_points", lambda coefficients: np.array([1e6]))
    trace = search.run_exact_linesearch_fgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1, tp_weight=0.01)
    assert trace.status == "failed"
    np.testing.assert_array_equal(trace.final_factor, factor)
    assert trace.diagnostics["terminal_objective_evaluations"][0] == 1
    assert_work_reconstruction(trace, rows.size)


def test_default_methods_unchanged_and_variant_repeats_without_contaminating_other_methods():
    data, _, _, _ = problem()
    kwargs = dict(n_steps=3, metrics_frequency=1, eta_scale=0.001, tp_weight=0.01,
                  initialization_seed=7, sampling_seed=8)
    assert campaign.METHODS == ("printed-adafgd", "armijo-fgd", "frames")
    default_meta, default_payload, default_traces = campaign.run_campaign(data, **kwargs)
    metadata, payload, traces = campaign.run_campaign(
        data, methods=campaign.ALL_METHODS, **kwargs)
    repeated_meta, repeated_payload, repeated = campaign.run_campaign(
        data, methods=(search.METHOD,), **kwargs)
    assert "qpt_quiroga_linesearch.py" not in default_meta["source_sha256"]
    assert "linesearch_rule" not in default_meta["configuration"]
    assert "qpt_quiroga_linesearch.py" in metadata["source_sha256"]
    assert "not published" in metadata["method_domains"][search.METHOD]["role"]
    np.testing.assert_array_equal(payload["initial_factor"], default_payload["initial_factor"])
    np.testing.assert_array_equal(payload["initial_factor"], repeated_payload["initial_factor"])
    for name in campaign.METHODS:
        np.testing.assert_array_equal(traces[name].final_factor, default_traces[name].final_factor)
        np.testing.assert_array_equal(traces[name].penalized_objective, default_traces[name].penalized_objective)
    np.testing.assert_array_equal(traces[search.METHOD].final_factor, repeated[search.METHOD].final_factor)


def test_cli_pickle_free_archive_identifies_variant_and_stopping(tmp_path):
    destination = tmp_path / "variant.npz"
    metadata, payload, traces = campaign.main([
        "--n-qubits", "1", "--methods", search.METHOD, "--steps", "2",
        "--metrics-every", "1", "--eta-scale", "1", "--tp-weight", "0.01",
        "--linesearch-gradient-atol", "1e-12", "--linesearch-gradient-rtol", "1e-10",
        "--save", str(destination), "--quiet",
    ])
    assert metadata["configuration"]["linesearch_gradient_atol"] == 1e-12
    assert metadata["configuration"]["linesearch_gradient_rtol"] == 1e-10
    assert metadata["configuration"]["linesearch_rule"] == "exact_positive_factor_ray_quartic_v1"
    stored = json.loads(destination.with_suffix(".json").read_text())
    with np.load(destination, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata_json"])) == stored
        assert all(not archive[key].dtype.hasobject for key in archive)
        np.testing.assert_array_equal(archive["exact_linesearch_fgd_diagnostic_terminal_attempt_rows"], [0])
    assert_work_reconstruction(traces[search.METHOD], metadata["active_row_count"])
