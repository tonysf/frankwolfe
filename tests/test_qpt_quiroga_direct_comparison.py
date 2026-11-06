"""Focused tests for paired Quiroga/FRAMES comparison protocol."""

import copy
import json

import numpy as np
import pytest

from paper.experiments.qpt_quiroga_direct_comparison import (
    DIRECT_METHODS,
    FORMAT,
    RecoveryGate,
    aggregate_sidecars,
    derive_matched_row_budget,
    main,
    run_direct_comparison,
)
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data


_SELECTION_HASH = "a" * 64


def _pair(*, data=None, rows=None, initialization_seed=5, sampling_seed=7,
          trial_id=None, **overrides):
    data = generate_quiroga_sensing_data(1, channel_seed=3) if data is None else data
    arguments = {
        "rows": rows, "row_epochs": 2, "checkpoint_epochs": 1,
        "printed_tp_weight_per_row": 0.05,
        "frames_initial_tp_weight_per_row": 0.005,
        "frames_smoothing_exponent": 0.25, "eta_scale": 1e-4,
        "initialization_seed": initialization_seed, "sampling_seed": sampling_seed,
        "batch_size": 4, "spectral_norm_method": "dense", "trial_id": trial_id,
        "cohort_id": "n1-test", "split": "heldout",
        "selection_manifest_sha256": _SELECTION_HASH,
        "printed_config_id": "printed-best-v1", "frames_config_id": "frames-best-v1",
        "show_progress": False,
    }
    arguments.update(overrides)
    return run_direct_comparison(data, **arguments)


def test_matched_budget_derivation_and_validation():
    budget = derive_matched_row_budget(8192, 32, 2000, 100)
    assert budget.printed_adafgd_steps == 2000
    assert budget.frames_steps == 512000
    assert budget.printed_adafgd_metrics_frequency == 100
    assert budget.frames_metrics_frequency == 25600
    assert budget.nominal_row_accesses == 16384000
    assert budget.as_dict()["compute_equivalent"] is False
    for arguments in ((0, 4, 1, 1), (10, 6, 1, 1), (16, 4, 1, 2)):
        with pytest.raises(ValueError):
            derive_matched_row_budget(*arguments)


def test_gate_is_inclusive_and_failed_status_cannot_pass():
    gate = RecoveryGate()
    endpoint = {"status": "complete", "process_fidelity": 0.99,
                "rmse": 1e-6, "normalized_tp_violation": 1e-3}
    result = gate.evaluate(endpoint, rmse=1e-6, rmse_source="full_design")
    assert result["passed"] and result["rmse_source"] == "full_design"
    endpoint["status"] = "failed"
    assert not gate.evaluate(endpoint)["passed"]
    endpoint["status"], endpoint["rmse"] = "complete", None
    assert not gate.evaluate(endpoint)["passed"]


def test_direct_pair_uses_method_native_penalties_and_shared_physical_budget():
    data = generate_quiroga_sensing_data(1, channel_seed=3)
    rows = data.fixed_row_subset(8, seed=11)
    metadata, payload, traces = _pair(data=data, rows=rows)
    configuration = metadata["configuration"]
    assert tuple(configuration["methods"]) == DIRECT_METHODS
    assert configuration["printed_tp_weight_per_row_alpha"] == pytest.approx(0.05)
    assert configuration["tp_weight_lambda"] == pytest.approx(0.4)
    assert configuration["frames_initial_tp_weight_per_row_alpha"] == pytest.approx(0.005)
    assert configuration["frames_initial_tp_weight_lambda"] == pytest.approx(0.04)
    assert configuration["frames_beta0"] == pytest.approx(100.0)
    assert configuration["frames_smoothing_schedule"] == {
        "type": "PowerSchedule", "scale": 100.0, "offset": 1.0,
        "exponent": 0.25, "cap": None,
    }

    comparison = metadata["comparison"]
    penalties = comparison["method_penalties"]
    assert penalties["printed-adafgd"]["summed_tp_weight_lambda"] == pytest.approx(0.4)
    assert penalties["frames"]["initial_smoothing_beta"] == pytest.approx(100.0)
    assert penalties["frames"]["endpoint_smoothing_beta"] == pytest.approx(
        100.0 / traces["frames"].checkpoint_steps[-1] ** 0.25)
    assert comparison["selection_provenance"]["selection_manifest_sha256"] == _SELECTION_HASH
    assert comparison["budget"]["nominal_row_accesses"] == 16
    assert "penalized_objective" not in comparison["endpoint_differences"]["values"]
    assert "native_penalized_objective" not in comparison[
        "endpoint_differences"]["values"]
    assert "optimizer_seconds" not in comparison["endpoint_differences"]["values"]
    assert "factor_operator_norm" not in comparison[
        "endpoint_differences"]["values"]
    assert set(traces) == set(DIRECT_METHODS)
    assert set(payload) >= {
        "printed_adafgd_final_factor", "frames_final_factor",
        "frames_diagnostic_checkpoint_tp_weight_per_row_alpha",
    }
    for trace in traces.values():
        assert trace.cumulative_data_row_accesses[-1] == 16
    np.testing.assert_array_equal(
        traces["printed-adafgd"].cumulative_data_row_accesses,
        traces["frames"].cumulative_data_row_accesses,
    )
    betas = traces["frames"].diagnostics["smoothing_parameter"]
    np.testing.assert_allclose(
        betas, 100.0 / np.arange(1, betas.size + 1) ** 0.25)
    for metric in ("measurement_loss_sum", "rmse", "tp_violation", "process_fidelity"):
        np.testing.assert_allclose(getattr(traces["printed-adafgd"], metric)[0],
                                   getattr(traces["frames"], metric)[0])
    assert comparison["initial_native_penalized_objective"]["printed-adafgd"] != pytest.approx(
        comparison["initial_native_penalized_objective"]["frames"])
    for method, trace in traces.items():
        diagnostics = comparison["endpoint_diagnostics"][method]
        assert diagnostics["full_design_metrics"]["rmse"] >= 0
        assert diagnostics["factor_frobenius_norm"] == pytest.approx(
            np.linalg.norm(trace.final_factor, ord="fro"))
        assert diagnostics["factor_operator_norm"] == pytest.approx(
            np.linalg.norm(trace.final_factor, ord=2))
        decision = comparison["method_gate"][method]
        assert decision["rmse_source"] == "full_design"
        assert decision["metric_values"]["rmse"] == pytest.approx(
            diagnostics["full_design_metrics"]["rmse"])
    printed_full = comparison["endpoint_diagnostics"]["printed-adafgd"][
        "full_design_metrics"]
    assert printed_full["native_penalized_objective"] == pytest.approx(
        printed_full["measurement_loss_sum"]
        + configuration["printed_summed_tp_weight_lambda"]
        * printed_full["tp_violation"] ** 2)
    assert comparison["paired_instance"]["initial_factor_sha256"]
    assert comparison["data_instance"]["full_observations_sha256"]
    assert comparison["domain_audit"]["truth_inside_frames_domain"]
    for method in DIRECT_METHODS:
        assert len(comparison["endpoint_diagnostics"][method]["final_factor_sha256"]) == 64
    assert metadata["provenance"]["git"]["commit"]
    assert "worktree_dirty" in metadata["provenance"]["git"]
    assert "tracked_dirty" not in metadata["provenance"]["git"]
    assert metadata["provenance"]["source_sha256"]
    assert "frank_wolfe/algorithms/base.py" in metadata["provenance"]["source_sha256"]
    assert set(comparison["cohort_spec"]["software"]) == {
        "python_version", "numpy_version", "scipy_version"}
    assert not comparison["first_gate_passage"]["frames"]["available"]
    assert "checkpoint factors" in comparison["first_gate_passage"]["frames"]["reason"]
    json.dumps(metadata, allow_nan=False)


def test_cli_writes_one_pickle_free_pair_and_refuses_to_clobber(tmp_path):
    archive = tmp_path / "pair.npz"
    arguments = [
        "--n-qubits", "1", "--row-epochs", "1", "--checkpoint-epochs", "1",
        "--printed-tp-weight-per-row", "0.05",
        "--frames-initial-tp-weight-per-row", "0.005",
        "--frames-penalty-exponent", "0.25", "--eta-scale", "0.0001",
        "--batch-size", "4", "--spectral-norm-method", "dense",
        "--trial-id", "cli-trial", "--cohort-id", "cli-test",
        "--split", "heldout",
        "--selection-manifest-sha256", _SELECTION_HASH,
        "--printed-config-id", "printed-best-v1",
        "--frames-config-id", "frames-best-v1",
        "--save", str(archive), "--quiet"]
    metadata, _, traces = main(arguments)
    assert archive.exists() and archive.with_suffix(".json").exists()
    assert metadata["format"] == FORMAT and set(traces) == set(DIRECT_METHODS)
    stored = json.loads(archive.with_suffix(".json").read_text())
    assert stored["comparison"]["trial_id"] == "cli-trial"
    assert stored["comparison"]["first_gate_passage"]["frames"]["available"]
    with np.load(archive, allow_pickle=False) as loaded:
        assert not any(loaded[name].dtype.hasobject for name in loaded.files)
        assert "printed_adafgd_final_factor" in loaded
        assert "frames_final_factor" in loaded
    with pytest.raises(SystemExit):
        main(arguments)
    same_path = tmp_path / "same-output.npz"
    same_arguments = list(arguments)
    same_arguments[same_arguments.index(str(archive))] = str(same_path)
    same_arguments.extend(["--metadata", str(same_path)])
    with pytest.raises(SystemExit):
        main(same_arguments)
    assert not same_path.exists()


def test_aggregation_nests_sampling_replicates_and_checks_duplicates_and_mismatch():
    first_data = generate_quiroga_sensing_data(1, channel_seed=9)
    second_data = generate_quiroga_sensing_data(1, channel_seed=10)
    first, _, _ = _pair(data=first_data, initialization_seed=5,
                        sampling_seed=1, trial_id="data-a-init-5-sample-1")
    second, _, _ = _pair(data=first_data, initialization_seed=5,
                         sampling_seed=2, trial_id="data-a-init-5-sample-2")
    third, _, _ = _pair(data=first_data, initialization_seed=6,
                        sampling_seed=3, trial_id="data-a-init-6-sample-3")
    fourth, _, _ = _pair(data=second_data, initialization_seed=5,
                         sampling_seed=4, trial_id="data-b-init-5-sample-4")
    records = [first, second, third, fourth]
    for record, passes in zip(records, (True, False, True, False)):
        endpoint = record["methods"]["frames"]
        endpoint.update({
            "status": "complete", "process_fidelity": 1.0 if passes else 0.0,
            "rmse": 0.0, "normalized_tp_violation": 0.0,
        })
        diagnostics = record["comparison"]["endpoint_diagnostics"]["frames"]
        diagnostics["full_design_metrics"]["rmse"] = 0.0
        gate = RecoveryGate.from_dict(record["comparison"]["gate"])
        record["comparison"]["method_gate"]["frames"] = gate.evaluate(
            endpoint, rmse=0.0, rmse_source="full_design")
    before = json.dumps(records, sort_keys=True)
    summary = aggregate_sidecars(records)
    assert summary["primary_analysis_unit"] == "data_instance"
    assert summary["data_instance_count"] == 2
    assert summary["optimization_instance_count"] == 3
    assert summary["sampling_trial_count"] == 4
    assert sorted(item["sampling_replicate_count"]
                  for item in summary["data_instances"]) == [1, 3]
    assert set(summary["methods"]) == set(DIRECT_METHODS)
    frames = summary["methods"]["frames"]
    assert frames["sampling_trial_metrics_descriptive"][
        "process_fidelity"]["count"] == 4
    assert frames["optimization_instance_mean_metrics"][
        "process_fidelity"]["count"] == 3
    assert frames["data_instance_mean_metrics"][
        "process_fidelity"]["count"] == 2
    fidelity = lambda record: record["methods"]["frames"]["process_fidelity"]
    expected_data_a = ((fidelity(first) + fidelity(second)) / 2 + fidelity(third)) / 2
    expected_primary_mean = (expected_data_a + fidelity(fourth)) / 2
    assert frames["data_instance_mean_metrics"]["process_fidelity"]["mean"] == pytest.approx(
        expected_primary_mean)
    data_gate = frames["data_instance_gate"]
    assert len(data_gate["instances"]) == 2
    assert data_gate["cohort_nested_pass_rate"] == pytest.approx(0.375)
    assert sorted(item["nested_pass_rate"] for item in data_gate["instances"]) == [
        0.0, 0.75]
    assert frames["sampling_trial_gate_descriptive"]["pass_rate"] == pytest.approx(0.5)
    assert data_gate["secondary_all_pass_robustness"]["rate"] == 0.0
    assert summary["paired_differences"]["data_instance_mean_metrics"][
        "process_fidelity"]["count"] == 2
    assert "metadata sidecars only" in summary["validation_scope"]
    assert "not a causal speed comparison" in summary["timing_interpretation"]
    assert json.dumps(records, sort_keys=True) == before
    with pytest.raises(ValueError, match="Duplicate trial_id"):
        aggregate_sidecars([first, first])
    mismatched = copy.deepcopy(second)
    mismatched["comparison"]["cohort_id"] = "different-cohort"
    with pytest.raises(ValueError, match="cohort"):
        aggregate_sidecars([first, mismatched])
    disagreeing = copy.deepcopy(second)
    disagreeing["comparison"]["endpoint_diagnostics"]["printed-adafgd"][
        "final_factor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Deterministic printed-adaFGD results disagree"):
        aggregate_sidecars([first, disagreeing])


def test_same_sampling_seed_and_instance_is_rejected_even_with_new_label():
    first, _, _ = _pair(sampling_seed=4, trial_id="first-label")
    duplicate = copy.deepcopy(first)
    duplicate["comparison"]["trial_id"] = "second-label"
    with pytest.raises(ValueError, match="Duplicate scientific trials"):
        aggregate_sidecars([first, duplicate])


def test_default_noiseless_gate_is_rejected_for_noisy_data():
    data = generate_quiroga_sensing_data(
        1, channel_seed=3, observation_mode="gaussian", noise_std=0.01,
        noise_seed=8)
    with pytest.raises(ValueError, match="noiseless_recovery_v1"):
        _pair(data=data)


def test_domain_schedule_and_dirty_worktree_preflights(monkeypatch):
    with pytest.raises(ValueError, match="excludes the truth factor"):
        _pair(tau=0.5)
    with pytest.raises(TypeError, match="rho_schedule"):
        _pair(rho_schedule=lambda step: 1.0)

    from paper.experiments import qpt_quiroga_direct_comparison as module
    provenance = module.collect_source_provenance()
    provenance["git"]["worktree_dirty"] = True
    monkeypatch.setattr(module, "collect_source_provenance", lambda: provenance)
    with pytest.raises(ValueError, match="including untracked files"):
        _pair(require_clean=True)


def test_native_objectives_are_method_scaled_and_never_paired():
    metadata, _, _ = _pair(trial_id="native-objective")
    methods = metadata["methods"]
    diagnostics = metadata["comparison"]["endpoint_diagnostics"]

    printed = methods["printed-adafgd"]
    printed_lambda = metadata["configuration"]["printed_summed_tp_weight_lambda"]
    assert printed["native_penalized_objective"] == pytest.approx(
        printed["measurement_loss_sum"] + printed_lambda * printed["tp_violation"] ** 2)

    frames = methods["frames"]
    frames_alpha = diagnostics["frames"]["native_tp_weight_per_row_alpha"]
    assert frames["native_penalized_objective"] == pytest.approx(
        frames["measurement_loss_mean"] + frames_alpha * frames["tp_violation"] ** 2)
    assert frames["summed_equivalent_penalized_objective"] == pytest.approx(
        metadata["active_row_count"] * frames["native_penalized_objective"])

    summary = aggregate_sidecars([metadata])
    assert "native_penalized_objective" in summary["methods"]["frames"][
        "data_instance_mean_metrics"]
    assert "native_penalized_objective" not in summary["paired_differences"][
        "data_instance_mean_metrics"]


def test_sidecar_rejects_native_objective_penalty_and_schedule_tampering():
    metadata, _, _ = _pair(trial_id="untampered")

    native = copy.deepcopy(metadata)
    native["methods"]["frames"]["native_penalized_objective"] += 1.0
    with pytest.raises(ValueError, match="native objective"):
        aggregate_sidecars([native])

    diagnostic = copy.deepcopy(metadata)
    diagnostic["comparison"]["endpoint_diagnostics"]["frames"][
        "native_tp_weight_per_row_alpha"] *= 2.0
    with pytest.raises(ValueError, match="endpoint penalties"):
        aggregate_sidecars([diagnostic])

    schedule = copy.deepcopy(metadata)
    schedule["configuration"]["frames_smoothing_schedule"]["exponent"] = 0.5
    with pytest.raises(ValueError, match="beta schedule"):
        aggregate_sidecars([schedule])


def test_selection_provenance_and_finite_penalty_schedule_preflights():
    missing = {
        "selection_manifest_sha256": None,
        "printed_config_id": None,
        "frames_config_id": None,
    }
    with pytest.raises(ValueError, match="validation and heldout"):
        _pair(split="validation", **missing)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _pair(selection_manifest_sha256="not-a-sha")
    with pytest.raises(ValueError, match="underflows or overflows"):
        _pair(frames_smoothing_exponent=1e308)

    first, _, _ = _pair(trial_id="selected-a")
    second, _, _ = _pair(
        trial_id="selected-b", printed_config_id="printed-best-v2")
    with pytest.raises(ValueError, match="different cohorts"):
        aggregate_sidecars([first, second])


def test_legacy_direct_comparison_schema_and_protocol_are_rejected():
    metadata, _, _ = _pair(trial_id="current-protocol")

    schema_v1 = copy.deepcopy(metadata)
    schema_v1["schema_version"] = 1
    with pytest.raises(ValueError, match="supported direct comparison"):
        aggregate_sidecars([schema_v1])

    protocol_v1 = copy.deepcopy(metadata)
    protocol_v1["comparison"]["protocol"] = "matched_nominal_row_accesses_v1"
    with pytest.raises(ValueError, match="incompatible protocol"):
        aggregate_sidecars([protocol_v1])


def test_failed_frames_replicate_remains_in_completed_replicate_cohort(monkeypatch):
    from paper.experiments import qpt_quiroga_campaign as campaign

    complete, _, _ = _pair(sampling_seed=7, trial_id="frames-complete")

    def fail_frames(*args, **kwargs):
        raise FloatingPointError("synthetic FRAMES failure")

    monkeypatch.setattr(
        campaign, "run_quiroga_sensing_stochastic_frames", fail_frames)
    failed, _, traces = _pair(sampling_seed=8, trial_id="frames-failed")

    assert traces["frames"].status == "failed"
    assert not failed["comparison"]["method_gate"]["frames"]["passed"]
    assert complete["comparison"]["method_penalties"]["frames"][
        "endpoint_completed_steps"] > 0
    assert failed["comparison"]["method_penalties"]["frames"][
        "endpoint_completed_steps"] == 0
    assert (complete["comparison"]["cohort_fingerprint"]
            == failed["comparison"]["cohort_fingerprint"])
    summary = aggregate_sidecars([complete, failed])
    assert summary["sampling_trial_count"] == 2
    expected_passes = int(
        complete["comparison"]["method_gate"]["frames"]["passed"])
    gate_summary = summary["methods"]["frames"][
        "sampling_trial_gate_descriptive"]
    assert gate_summary["pass_count"] == expected_passes
    assert gate_summary["pass_rate"] == pytest.approx(expected_passes / 2)
    assert summary["methods"]["frames"][
        "sampling_trial_status_counts_descriptive"] == {
            "complete": 1,
            "failed": 1,
        }
