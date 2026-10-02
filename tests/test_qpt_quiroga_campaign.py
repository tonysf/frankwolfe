"""Fast regression tests for the reproducible Quiroga campaign runner."""

import json

import numpy as np
import pytest

import paper.experiments.qpt_quiroga_campaign as campaign

from paper.experiments.qpt_quiroga_campaign import (
    _active_rows_and_targets,
    _summed_objective,
    _summed_objective_and_half_gradient,
    evaluate_common_metrics,
    main,
    run_armijo_fgd,
    run_campaign,
    run_printed_adafgd,
    save_campaign,
)
from paper.experiments.qpt_quiroga_frames import (
    _adafgd_spectral_norm,
    quiroga_adafgd_step,
)
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.quantum_process_tomography import (
    PowerSchedule,
    make_factor_initial_point,
    unpack_factor,
)


def _problem(n=1, subset=None):
    data = generate_quiroga_sensing_data(n, channel_seed=3)
    rows = (np.arange(data.m, dtype=np.int64) if subset is None
            else data.fixed_row_subset(subset, seed=7))
    targets = data.observations_for_rows(rows)
    factor = unpack_factor(
        make_factor_initial_point(data, rank=1, seed=5), data.process_dimension, 1)
    return data, rows, targets, factor


def test_printed_loop_is_the_validated_literal_step():
    data, rows, targets, factor = _problem()
    expected, info = quiroga_adafgd_step(
        data, factor, eta_scale=1e-3, tp_weight=0.5, rows=rows,
        observations=targets, spectral_norm_method="dense")
    trace = run_printed_adafgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1,
        eta_scale=1e-3, tp_weight=0.5, spectral_norm_method="dense")

    assert trace.status == "complete"
    np.testing.assert_array_equal(trace.checkpoint_steps, [0, 1])
    np.testing.assert_allclose(trace.final_factor, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(trace.diagnostics["step_size"], [info["eta"]])
    np.testing.assert_array_equal(trace.cumulative_data_row_accesses, [0, rows.size])
    expected_metrics = evaluate_common_metrics(data, expected, rows, targets, 0.5)
    for name, value in expected_metrics.items():
        assert getattr(trace, name)[-1] == pytest.approx(value)


def test_armijo_uses_the_correct_half_gradient_slope_and_decreases_objective():
    data, rows, targets, factor = _problem()
    objective, direction = _summed_objective_and_half_gradient(
        data, factor, rows, targets, tp_weight=0.5)
    epsilon = 1e-7
    finite_difference = (
        _summed_objective(data, factor - epsilon * direction, rows, targets, 0.5)
        - _summed_objective(data, factor + epsilon * direction, rows, targets, 0.5)
    ) / (2 * epsilon)
    expected_slope = -2.0 * np.vdot(direction, direction).real
    assert objective == pytest.approx(
        evaluate_common_metrics(data, factor, rows, targets, 0.5)["penalized_objective"])
    assert finite_difference == pytest.approx(expected_slope, rel=2e-6, abs=1e-9)

    trace = run_armijo_fgd(
        data, factor, rows, targets, n_steps=4, metrics_frequency=1,
        tp_weight=0.5, initial_step=1.0, shrink=0.5,
        armijo_constant=1e-4, max_backtracks=30)
    assert trace.status in {"complete", "converged"}
    assert np.all(np.diff(trace.penalized_objective) <= 1e-12)
    assert np.all(trace.diagnostics["step_size"] > 0)
    trials = trace.diagnostics["objective_evaluations"].sum()
    completed = trace.diagnostics["step_size"].size
    assert trace.cumulative_data_row_accesses[-1] == rows.size * (completed + trials)


def test_campaign_shares_inputs_and_matches_frames_penalty_scaling():
    data = generate_quiroga_sensing_data(1, channel_seed=11, observation_mode="gaussian",
                                         noise_std=1e-3, noise_seed=13)
    rows = data.fixed_row_subset(12, seed=17)
    metadata, payload, traces = run_campaign(
        data, methods=("printed-adafgd", "armijo-fgd", "frames"), rows=rows,
        initialization_seed=19, sampling_seed=23, n_steps=2, frames_steps=2,
        metrics_frequency=1, eta_scale=1e-4, tp_weight=0.75,
        batch_size=3, spectral_norm_method="dense", show_progress=False)

    assert metadata["configuration"]["frames_beta"] == pytest.approx(rows.size / 1.5)
    assert metadata["active_rows_sha256"]
    assert metadata["active_observations_sha256"]
    np.testing.assert_array_equal(payload["active_rows"], rows)
    for trace in traces.values():
        assert trace.status == "complete"
        np.testing.assert_allclose(trace.measurement_loss_sum[0],
                                   traces["armijo-fgd"].measurement_loss_sum[0])
        np.testing.assert_allclose(trace.tp_violation[0],
                                   traces["armijo-fgd"].tp_violation[0])
    frames = traces["frames"]
    np.testing.assert_allclose(
        frames.penalized_objective,
        rows.size * frames.measurement_loss_mean + 0.75 * frames.tp_violation**2,
    )
    np.testing.assert_array_equal(frames.cumulative_data_row_accesses, [0, 3, 6])


def test_atomic_archive_is_pickle_free_and_refuses_to_clobber(tmp_path):
    data = generate_quiroga_sensing_data(1, channel_seed=2)
    metadata, payload, _ = run_campaign(
        data, methods=("armijo-fgd",), n_steps=1, metrics_frequency=1,
        eta_scale=1e-3, tp_weight=1.0)
    archive, sidecar = tmp_path / "run.npz", tmp_path / "run.json"
    save_campaign(archive, sidecar, metadata, payload)
    before_archive, before_sidecar = archive.read_bytes(), sidecar.read_bytes()

    with np.load(archive, allow_pickle=False) as loaded:
        assert not any(loaded[name].dtype.hasobject for name in loaded.files)
        stored = json.loads(str(loaded["metadata_json"].item()))
        assert stored["format"] == "qpt_quiroga_campaign"
        np.testing.assert_array_equal(loaded["active_rows"], np.arange(data.m))
    assert json.loads(sidecar.read_text())["schema_version"] == 1

    with pytest.raises(FileExistsError):
        save_campaign(archive, sidecar, metadata, payload)
    assert archive.read_bytes() == before_archive
    assert sidecar.read_bytes() == before_sidecar


def test_cli_writes_deterministic_subset_and_metadata(tmp_path):
    archive = tmp_path / "smoke.npz"
    metadata, _, traces = main([
        "--n-qubits", "1", "--methods", "armijo-fgd", "--steps", "1",
        "--metrics-every", "1", "--subset-size", "8", "--subset-seed", "31",
        "--channel-seed", "29", "--initialization-seed", "37",
        "--eta-scale", "0.001", "--tp-weight", "1", "--save", str(archive),
        "--quiet",
    ])
    assert traces["armijo-fgd"].status == "complete"
    assert archive.exists() and archive.with_suffix(".json").exists()
    expected = generate_quiroga_sensing_data(1, channel_seed=29).fixed_row_subset(8, 31)
    with np.load(archive, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["active_rows"], expected)
    assert metadata["configuration"]["subset_seed"] == 31

    with pytest.raises(SystemExit):
        main([
            "--n-qubits", "1", "--methods", "armijo-fgd", "--steps", "1",
            "--eta-scale", "0.001", "--tp-weight", "1", "--save", str(archive),
            "--quiet",
        ])


def test_failed_printed_attempt_retains_elapsed_and_row_work(monkeypatch):
    data, rows, targets, factor = _problem()

    def fail_after_call_begins(*args, **kwargs):
        raise RuntimeError("synthetic printed-step failure")

    monkeypatch.setattr(campaign, "quiroga_adafgd_step", fail_after_call_begins)
    trace = run_printed_adafgd(
        data, factor, rows, targets, n_steps=2, metrics_frequency=1,
        eta_scale=1e-3, tp_weight=0.5, spectral_norm_method="dense")

    assert trace.status == "failed"
    assert "synthetic printed-step failure" in trace.failure_reason
    np.testing.assert_array_equal(trace.checkpoint_steps, [0])
    assert trace.optimizer_seconds[-1] >= 0.0
    assert trace.cumulative_data_row_accesses[-1] == rows.size


def test_checkpoint_failure_is_archivable_with_a_retained_terminal(tmp_path, monkeypatch):
    data, rows, targets, factor = _problem()
    original = campaign.evaluate_common_metrics
    calls = 0

    def fail_after_initial(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise FloatingPointError("synthetic metric failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign, "evaluate_common_metrics", fail_after_initial)
    trace = run_armijo_fgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1,
        tp_weight=0.5, initial_step=1e-3)

    assert trace.status == "failed"
    assert trace.checkpoint_steps[-1] == 1
    assert trace.cumulative_data_row_accesses[-1] >= 2 * rows.size
    assert np.isnan(trace.measurement_loss_sum[-1])
    assert trace.endpoint()["measurement_loss_sum"] is None
    archive, sidecar = tmp_path / "failed.npz", tmp_path / "failed.json"
    save_campaign(
        archive, sidecar, {"methods": {"armijo-fgd": trace.endpoint()}},
        {"measurement_loss_sum": trace.measurement_loss_sum})
    assert archive.exists() and sidecar.exists()
    assert json.loads(sidecar.read_text())["methods"]["armijo-fgd"][
        "measurement_loss_sum"] is None


def test_spectral_norm_diagnostics_preserve_legacy_return_and_reach_endpoint():
    data, rows, targets, factor = _problem()
    residual = data.design.full_values(factor) - data.all_observations()
    legacy = _adafgd_spectral_norm(
        data.design, residual, method="dense", dense_max_process_dimension=32)
    dense_value, dense_method, dense_diagnostics = _adafgd_spectral_norm(
        data.design, residual, method="dense", dense_max_process_dimension=32,
        return_diagnostics=True)
    matrix_free_value, matrix_free_method, matrix_free_diagnostics = (
        _adafgd_spectral_norm(
            data.design, residual, method="matrix-free",
            dense_max_process_dimension=32, return_diagnostics=True))

    assert len(legacy) == 2
    assert legacy[0] == pytest.approx(dense_value)
    assert dense_method == "dense"
    assert dense_diagnostics["seconds"] >= 0.0
    assert dense_diagnostics["operator_calls"] == 1
    assert dense_diagnostics["adjoint_column_equivalent"] == data.process_dimension
    assert matrix_free_method == "matrix-free"
    assert matrix_free_value == pytest.approx(dense_value, rel=1e-10, abs=1e-12)
    assert matrix_free_diagnostics["operator_calls"] > 0
    assert (matrix_free_diagnostics["adjoint_column_equivalent"]
            == matrix_free_diagnostics["operator_calls"])

    trace = run_printed_adafgd(
        data, factor, rows, targets, n_steps=1, metrics_frequency=1,
        eta_scale=1e-3, tp_weight=0.5, spectral_norm_method="dense")
    endpoint = trace.endpoint()
    assert endpoint["spectral_norm_seconds_total"] >= 0.0
    assert endpoint["spectral_norm_operator_calls_total"] == 1
    assert (endpoint["spectral_norm_adjoint_column_equivalent_total"]
            == data.process_dimension)


def test_programmatic_metadata_records_schedules_domains_and_provenance():
    data = generate_quiroga_sensing_data(1, channel_seed=41)
    rho = PowerSchedule(3.0, 5.0, 0.7, cap=0.9)
    step = PowerSchedule(1.5, 3.0, 1.0, cap=0.8)
    metadata, _, _ = run_campaign(
        data, methods=("armijo-fgd",), n_steps=1, metrics_frequency=1,
        eta_scale=1e-3, tp_weight=0.5,
        rho_schedule=rho, step_size_schedule=step)

    configuration = metadata["configuration"]
    assert configuration["rho_schedule"] == {
        "type": "PowerSchedule", "scale": 3.0, "offset": 5.0,
        "exponent": 0.7, "cap": 0.9}
    assert configuration["step_size_schedule"] == {
        "type": "PowerSchedule", "scale": 1.5, "offset": 3.0,
        "exponent": 1.0, "cap": 0.8}
    assert metadata["method_domains"]["frames"]["constraint"] == "operator_norm_ball"
    assert metadata["method_domains"]["armijo-fgd"]["constraint"] == "none"
    assert "not a total compute/work-equivalent" in metadata["work_counter"]
    assert metadata["scipy_version"]
    assert all(len(digest) == 64 for digest in metadata["source_sha256"].values())


def test_campaign_explicitly_requires_rank_one_truth():
    data = generate_quiroga_sensing_data(1, channel_seed=43)
    data.truth_factor = np.column_stack((data.truth_factor, data.truth_factor))
    with pytest.raises(ValueError, match="rank-one truth factor"):
        run_campaign(
            data, methods=("armijo-fgd",), n_steps=1, metrics_frequency=1,
            eta_scale=1e-3, tp_weight=0.5)


def test_canonical_all_rows_use_only_full_design_apis(monkeypatch):
    data, rows, targets, factor = _problem()

    def forbid_sampled_api(*args, **kwargs):
        raise AssertionError("sampled-row API should not be used for canonical full rows")

    monkeypatch.setattr(type(data.design), "values", forbid_sampled_api)
    monkeypatch.setattr(type(data.design), "loss_and_gradient", forbid_sampled_api)
    active_rows, active_targets = _active_rows_and_targets(data, None)
    np.testing.assert_array_equal(active_rows, rows)
    np.testing.assert_allclose(active_targets, targets)
    metrics = evaluate_common_metrics(data, factor, rows, targets, tp_weight=0.5)
    objective, gradient = _summed_objective_and_half_gradient(
        data, factor, rows, targets, tp_weight=0.5)
    objective_only = _summed_objective(
        data, factor, rows, targets, tp_weight=0.5)
    assert all(np.isfinite(list(metrics.values())))
    assert objective == pytest.approx(objective_only)
    assert np.all(np.isfinite(gradient))


def test_strict_cli_saves_before_nonzero_exit(tmp_path, monkeypatch):
    archive = tmp_path / "strict-failure.npz"
    monkeypatch.setattr(campaign, "_summed_objective", lambda *args, **kwargs: np.inf)

    with pytest.raises(SystemExit) as exit_info:
        main([
            "--n-qubits", "1", "--methods", "armijo-fgd", "--steps", "1",
            "--metrics-every", "1", "--eta-scale", "0.001",
            "--tp-weight", "1", "--save", str(archive), "--quiet",
            "--fail-on-method-failure",
        ])

    assert exit_info.value.code == 2
    assert archive.exists() and archive.with_suffix(".json").exists()
    stored = json.loads(archive.with_suffix(".json").read_text())
    assert stored["methods"]["armijo-fgd"]["status"] == "failed"
