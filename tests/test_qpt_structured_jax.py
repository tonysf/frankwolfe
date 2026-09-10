"""End-to-end parity and bounded-state checks for structured QPT."""

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from paper.experiments.qpt_structured_data import (  # noqa: E402
    StructuredQPTData, standard_local_basis, standard_local_measurements,
)
from paper.experiments.quantum_process_tomography import (  # noqa: E402
    QPTData, QPTMeasurementObjective, create_operator_norm_factor_lmo,
    make_factor_initial_point, pack_factor,
)
from paper.experiments.quantum_process_tomography_jax import (  # noqa: E402
    run_qpt_stochastic_frames_jax,
)
from paper.experiments.quantum_process_tomography_structured_jax import (  # noqa: E402
    _build_scan, _executable_memory_estimate, load_structured_result, main, run_qpt_structured_jax,
    save_structured_result,
)


def kron(factors):
    result = np.ones((1, 1), dtype=complex)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def test_optional_compiler_memory_accounting_is_not_required():
    from types import SimpleNamespace

    assert _executable_memory_estimate(object()) is None
    assert _executable_memory_estimate(SimpleNamespace(memory_analysis=lambda: None)) is None
    stats = SimpleNamespace(argument_size_in_bytes=100, output_size_in_bytes=50,
                            temp_size_in_bytes=20, alias_size_in_bytes=10)
    assert _executable_memory_estimate(SimpleNamespace(memory_analysis=lambda: stats)) == 160


def paired_data(n=1):
    """Include complex truth and nonzero fixed noise in the original row order."""
    rng = np.random.default_rng(601 + n)
    truth = rng.normal(size=(4**n, 1)) + 1j * rng.normal(size=(4**n, 1))
    truth *= np.sqrt(2**n) / np.linalg.norm(truth)
    structured = StructuredQPTData(
        n, standard_local_measurements(), standard_local_basis(),
        truth_factor=truth, observation_mode="noiseless",
    )
    symbols = structured.indices_to_symbols(np.arange(structured.m))
    observed = structured.observations_for_symbols(symbols) + rng.normal(scale=0.04, size=structured.m)
    structured = StructuredQPTData(
        n, structured.local_measurements, structured.local_basis,
        observations=observed, truth_factor=truth,
    )
    dense = QPTData(
        observed,
        np.stack([kron(structured.local_measurements[row]) for row in symbols]),
        np.stack([kron(structured.local_basis[list(row)]) for row in itertools.product(range(4), repeat=n)]),
        chi_star=truth @ truth.conj().T,
    )
    return structured, dense


def run_options(rank=1, n_steps=4):
    return dict(
        n_steps=n_steps, rank=rank, tau=3.0, batch_size=5,
        initialization_seed=3, sampling_seed=17,
        rho_schedule=lambda k: 0.7 / (k + 1)**0.3,
        smoothing_schedule=lambda k: 2.0 / (k + 1)**0.2,
        step_size_schedule=lambda k: 0.2 / (k + 1)**0.5,
        metrics_frequency=1, device="cpu", precision="64",
        warmup=False, show_progress=False,
    )


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rank", [1, 2])
def test_dense_trajectory_and_full_metrics_match(n, rank):
    structured, dense = paired_data(n)
    options = run_options(rank)
    options["x0"] = make_factor_initial_point(dense, rank, seed=91)
    reference = run_qpt_stochastic_frames_jax(dense, execution_mode="scan", **options)
    result = run_qpt_structured_jax(
        structured, chunk_steps=2, metric_mode="full", metric_batch_size=37,
        store_checkpoints=True, **options,
    )
    np.testing.assert_allclose(result.final_x, reference.final_x, atol=3e-11)
    np.testing.assert_allclose(result.final_gradient_estimate, reference.final_gradient_estimate, atol=3e-11)
    np.testing.assert_allclose(result.estimated_gaps, reference.estimated_gaps, atol=3e-11)
    for name in ("measurement_loss", "tp_violation", "smoothed_objective", "process_fidelity_proxy"):
        np.testing.assert_allclose(getattr(result, name), getattr(reference, name), atol=3e-11)
    np.testing.assert_allclose(result.smoothed_gap, reference.exact_smoothed_gap, atol=3e-11)
    np.testing.assert_array_equal(result.checkpoint_smoothing_parameters, reference.checkpoint_smoothing_parameters)
    symbols = structured.indices_to_symbols(reference.batch_indices.ravel()).astype(np.int32)
    assert result.metadata["batch_symbols_sha256"] == hashlib.sha256(symbols.tobytes()).hexdigest()
    assert result.metadata["gap_is_exact"] is True
    assert result.metadata["penalty_coefficient"] == "1/(2*beta)"
    assert result.metadata["x0_provided"] is True
    assert result.metadata["initialization_seed"] is None
    initial = result.checkpoint_factors[0]
    assert result.metadata["initial_factor_sha256"] == hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()


def test_chunk_boundaries_and_metric_sampling_do_not_change_optimizer():
    data, _ = paired_data(2)
    options = run_options(n_steps=7)
    options["metrics_frequency"] = 3
    first = run_qpt_structured_jax(
        data, chunk_steps=1, metric_mode="sampled", metric_samples=19,
        metric_seed=8, metric_batch_size=6, **options,
    )
    second = run_qpt_structured_jax(
        data, chunk_steps=5, metric_mode="sampled", metric_samples=19,
        metric_seed=8, metric_batch_size=11, **options,
    )
    full = run_qpt_structured_jax(
        data, chunk_steps=5, metric_mode="full", metric_batch_size=43, **options,
    )
    for result in (second, full):
        np.testing.assert_allclose(result.final_factor, first.final_factor, atol=2e-12)
        np.testing.assert_allclose(result.gradient_estimate, first.gradient_estimate, atol=2e-12)
        np.testing.assert_allclose(result.estimated_gaps, first.estimated_gaps, atol=2e-12)
        assert result.metadata["batch_symbols_sha256"] == first.metadata["batch_symbols_sha256"]
    np.testing.assert_array_equal(first.metric_symbols, second.metric_symbols)
    np.testing.assert_allclose(first.measurement_loss, second.measurement_loss, atol=2e-12)
    np.testing.assert_allclose(first.smoothed_gap, second.smoothed_gap, atol=2e-12)
    np.testing.assert_array_equal(first.checkpoint_steps, [0, 3, 6, 7])
    assert first.checkpoint_factors.shape == (0, 16, 1)
    assert not hasattr(first, "iterate_history")
    assert not hasattr(first, "final_chi")
    assert first.metadata["gap_is_exact"] is False


def test_rank_one_measurement_fast_path_matches_general_tensor_trajectory():
    data, _ = paired_data(2)
    options = run_options(rank=2, n_steps=3)
    tensor = run_qpt_structured_jax(data, metric_samples=17, measurement_backend="tensor", **options)
    fast = run_qpt_structured_jax(data, metric_samples=17, measurement_backend="rank-one", **options)
    np.testing.assert_allclose(fast.final_factor, tensor.final_factor, atol=2e-12)
    np.testing.assert_allclose(fast.gradient_estimate, tensor.gradient_estimate, atol=2e-12)
    np.testing.assert_allclose(fast.estimated_gaps, tensor.estimated_gaps, atol=2e-12)
    np.testing.assert_allclose(fast.smoothed_gap, tensor.smoothed_gap, atol=2e-12)
    assert fast.metadata["measurement_backend"] == "rank-one"
    assert tensor.metadata["measurement_backend"] == "tensor"


def test_fixed_sample_metrics_use_stored_noisy_observations_at_every_checkpoint():
    data, dense = paired_data(1)
    options = run_options(rank=2, n_steps=3)
    result = run_qpt_structured_jax(
        data, chunk_steps=2, metric_mode="sampled", metric_samples=13,
        metric_batch_size=5, metric_seed=20, store_checkpoints=True, **options,
    )
    indices = data.symbols_to_indices(result.metric_symbols)
    expected_indices = np.random.default_rng(20).integers(0, data.m, size=13)
    np.testing.assert_array_equal(indices, expected_indices)
    sampled_data = QPTData(dense.f_vector[indices], dense.D_tensors[indices], dense.A_basis)
    objective = QPTMeasurementObjective(sampled_data, rank=2)
    lmo = create_operator_norm_factor_lmo(data.process_dimension, 2, options["tau"])
    expected_losses, expected_gaps = [], []
    for factor, beta in zip(result.checkpoint_factors, result.checkpoint_smoothing_parameters):
        vector = pack_factor(factor)
        loss, gradient = objective.loss_and_gradient(vector)
        residual = objective.trace_preserving_residual(vector)
        gradient += objective.linear_operator_adjoint_at(vector, residual) / beta
        expected_losses.append(loss)
        expected_gaps.append(np.dot(gradient, vector - lmo(gradient)))
    np.testing.assert_allclose(result.measurement_loss, expected_losses, atol=2e-12)
    np.testing.assert_allclose(result.smoothed_gap, expected_gaps, atol=2e-12)
    # The truth is present for fidelity only: discarding fixed noise changes the run.
    noiseless = StructuredQPTData(
        1, data.local_measurements, data.local_basis,
        truth_factor=data.truth_factor, observation_mode="noiseless",
    )
    other = run_qpt_structured_jax(noiseless, metric_samples=13, **options)
    assert np.linalg.norm(result.final_factor - other.final_factor) > 1e-5
    assert result.metadata["observation_mode"] == "stored"


def test_compiled_scan_emits_only_final_state_and_scalar_history():
    import jax.numpy as jnp

    data, _ = paired_data()
    with jax.experimental.enable_x64():
        factor = jnp.zeros((4, 2), dtype=jnp.complex128)
        scan = _build_scan(jax, jnp, 3.0, False)
        for steps in (3, 31):
            result = jax.eval_shape(
                scan, (factor, factor), jnp.asarray(0, dtype=jnp.int32),
                jnp.zeros((steps, 5, 1), dtype=jnp.int32), jnp.zeros((steps, 5)),
                jnp.ones(steps), jnp.ones(steps), jnp.ones(steps),
                jnp.asarray(data.local_measurements), jnp.asarray(data.local_basis),
                jnp.asarray(data.truth_factor),
            )
            state, scalar_history = result
            assert [leaf.shape for leaf in state] == [(4, 2), (4, 2)]
            assert scalar_history.shape == (steps,)


def test_result_round_trip_uses_no_pickle_and_preserves_metadata(tmp_path):
    data, _ = paired_data()
    result = run_qpt_structured_jax(data, metric_samples=7, **run_options(n_steps=2))
    path = tmp_path / "result.npz"
    save_structured_result(path, result)
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[key].dtype.hasobject for key in archive.files)
        assert not any(key in archive for key in ("D_tensors", "B_tensors", "iterate_history", "final_chi"))
        assert json.loads(archive["metadata_json"].item())["observation_mode"] == "stored"
    restored = load_structured_result(path)
    assert restored.metadata == result.metadata
    np.testing.assert_array_equal(restored.final_factor, result.final_factor)
    np.testing.assert_array_equal(restored.metric_symbols, result.metric_symbols)
    np.testing.assert_array_equal(restored.measurement_loss, result.measurement_loss)
    assert restored.metadata["compiled_memory_estimate_bytes"] >= 0
    assert restored.metadata["optimizer_seconds"] >= 0
    assert restored.metadata["compile_seconds"] >= 0
    assert restored.metadata["x0_provided"] is False
    assert restored.metadata["initialization_seed"] == 3


@pytest.mark.parametrize("mode", ["sampled", "full"])
def test_cli_saves_a_loadable_result(tmp_path, mode):
    data, _ = paired_data()
    data_path, result_path = tmp_path / "data.npz", tmp_path / "result.npz"
    data.save_npz(data_path)
    assert main([
        "--data", str(data_path), "--save", str(result_path), "--steps", "2",
        "--batch-size", "3", "--chunk-steps", "1", "--metrics-every", "0",
        "--metric-mode", mode, "--metric-samples", "7", "--metric-batch-size", "5",
        "--device", "cpu", "--step-scale", "0.1", "--quiet",
    ]) == 0
    result = load_structured_result(result_path)
    assert result.metadata["metric_mode"] == mode
    assert result.final_factor.shape == (4, 1)


def test_cli_module_help_is_executable():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "paper.experiments.quantum_process_tomography_structured_jax", "--help"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "--chunk-steps" in completed.stdout
    assert "--metric-mode" in completed.stdout


def test_benchmark_cli_compares_fresh_processes_and_labels_estimates(tmp_path):
    h5py = pytest.importorskip("h5py")
    _, dense = paired_data()
    h5_path = tmp_path / "original.h5"
    with h5py.File(h5_path, "w") as handle:
        handle["f_jax_vector"] = dense.f_vector
        handle["D_jax_tensors"] = dense.D_tensors
        handle["A_jax_basis"] = dense.A_basis
    root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(root / "scripts/benchmark_qpt_structured.py"),
         "--h5", str(h5_path), "--save", str(report_path), "--device", "cpu",
         "--steps", "2", "--batch-size", "3", "--chunk-steps", "1",
         "--metrics-every", "0", "--metric-mode", "full", "--metric-batch-size", "7",
         "--step-scale", "0.1", "--repeats", "1"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    report = json.loads(report_path.read_text())
    assert report["observations_preserved"] is True
    assert report["final_factor_max_abs_difference"] < 1e-10
    assert report["final_full_loss_abs_difference"] < 1e-10
    assert report["final_full_gap_abs_difference"] < 1e-10
    assert len(report["runs"]) == 2
    assert all(row["device_platform"] == "cpu" for row in report["runs"])
    assert all(row["optimizer_compile_seconds"] >= 0 for row in report["runs"])
    assert "estimates" in report["memory_estimates"]["kind"]
