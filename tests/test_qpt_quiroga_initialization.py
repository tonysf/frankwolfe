"""Independent dense checks of opt-in coefficient-coordinate initialization."""

import hashlib
import json
from pathlib import Path
import tracemalloc

import numpy as np
import pytest

from paper.experiments import qpt_quiroga_campaign as campaign
from paper.experiments.qpt_quiroga_frames import quiroga_adafgd_step
from paper.experiments.qpt_quiroga_initialization import (
    GELL_MANN_CONVENTION,
    gell_mann_coefficients_to_choi,
    initialization_descriptor,
)
from paper.experiments.qpt_quiroga_sensing import (
    generate_quiroga_sensing_data,
    trace_preserving_loss_and_gradient,
    trace_preserving_residual,
)
from paper.experiments.quantum_process_tomography import (
    make_factor_initial_point,
    unpack_factor,
)


def _dense_basis(d):
    """Build each operator independently; never call the production mapper."""
    basis = [np.eye(d, dtype=complex) / np.sqrt(d)]
    for imaginary in (False, True):
        for j in range(d):
            for k in range(j + 1, d):
                operator = np.zeros((d, d), dtype=complex)
                operator[j, k] = -1j if imaginary else 1
                operator[k, j] = 1j if imaginary else 1
                basis.append(operator / np.sqrt(2))
    for level in range(1, d):
        entries = [1] * level + [-level] + [0] * (d - level - 1)
        basis.append(np.diag(entries) / np.sqrt(level * (level + 1)))
    return np.asarray(basis)


def _dense_change(d):
    return np.column_stack([operator.T.ravel() for operator in _dense_basis(d)])


@pytest.mark.parametrize("d", [2, 4])
def test_every_basis_column_matches_independent_dense_map(d):
    basis, change = _dense_basis(d), _dense_change(d)
    np.testing.assert_allclose(basis, basis.conj().transpose(0, 2, 1), atol=0)
    np.testing.assert_allclose(change.conj().T @ change, np.eye(d * d), atol=3e-16)
    np.testing.assert_allclose(
        gell_mann_coefficients_to_choi(np.eye(d * d)), change, atol=3e-16)
    # This explicitly catches a conjugation/transposition sign error at n=1.
    first_antisymmetric = 1 + d * (d - 1) // 2
    assert change[1, first_antisymmetric] == pytest.approx(1j / np.sqrt(2))
    assert change[d, first_antisymmetric] == pytest.approx(-1j / np.sqrt(2))


@pytest.mark.parametrize("d,rank", [(2, 1), (2, 3), (4, 1), (4, 3)])
def test_complex_mapping_preserves_gram_and_singular_values(d, rank):
    rng = np.random.default_rng(981)
    coefficients = rng.normal(size=(d * d, rank)) + 1j * rng.normal(size=(d * d, rank))
    before = coefficients.copy()
    mapped = gell_mann_coefficients_to_choi(coefficients)
    np.testing.assert_array_equal(coefficients, before)
    np.testing.assert_allclose(mapped, _dense_change(d) @ coefficients, rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(mapped.conj().T @ mapped, coefficients.conj().T @ coefficients,
                               rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(np.linalg.svd(mapped, compute_uv=False),
                               np.linalg.svd(coefficients, compute_uv=False), rtol=2e-15)


@pytest.mark.parametrize("n", [1, 2])
def test_measurement_and_tp_gradients_are_covariant_in_dense_basis(n):
    data = generate_quiroga_sensing_data(n, channel_seed=3)
    coefficients = unpack_factor(make_factor_initial_point(data, rank=2, seed=5),
                                  data.process_dimension, 2)
    change, basis = _dense_change(data.d), _dense_basis(data.d)
    factor = gell_mann_coefficients_to_choi(coefficients)
    rows = np.arange(data.m)
    targets = data.all_observations()
    dense_rows = data.design.dense_sensing_matrices(rows)
    coefficient_rows = np.asarray([change.conj().T @ row @ change for row in dense_rows])
    predictions = np.einsum("ia,sij,ja->s", coefficients.conj(), coefficient_rows,
                            coefficients).real
    loss, gradient = data.design.loss_and_gradient(factor, rows, targets, reduction="sum")
    residual = predictions - targets
    expected_gradient = 2 * np.einsum("s,sij,ja->ia", residual, coefficient_rows, coefficients)
    np.testing.assert_allclose(data.design.values(factor, rows), predictions, atol=3e-16)
    assert loss == pytest.approx(0.5 * residual @ residual, abs=2e-15)
    np.testing.assert_allclose(change.conj().T @ gradient, expected_gradient, atol=2e-15)

    kraus = np.einsum("ba,bij->aij", coefficients, basis)
    tp_kraus = sum(operator.conj().T @ operator for operator in kraus) - np.eye(data.d)
    np.testing.assert_allclose(trace_preserving_residual(factor), tp_kraus.T, atol=8e-16)
    tp_loss, tp_gradient = trace_preserving_loss_and_gradient(factor)
    expected_tp_gradient = np.einsum("bij,aij->ba", basis.conj(), 2 * kraus @ tp_kraus)
    assert tp_loss == pytest.approx(0.5 * np.vdot(tp_kraus, tp_kraus).real, abs=2e-15)
    np.testing.assert_allclose(change.conj().T @ tp_gradient, expected_tp_gradient, atol=3e-15)

    # Independently assemble the literal printed step in coefficient space,
    # including its matrix-space spectral numerator (not a factor norm).
    eta_scale, tp_weight = 0.123, 0.7
    matrix_gradient = np.einsum("s,sij->ij", residual, coefficient_rows)
    numerator = np.max(np.abs(np.linalg.eigvalsh(matrix_gradient)))
    denominator = np.linalg.norm(predictions)
    eta = eta_scale * numerator / denominator
    expected_next = coefficients - eta * (
        matrix_gradient @ coefficients + tp_weight * expected_tp_gradient)
    updated, diagnostics = quiroga_adafgd_step(
        data, factor, eta_scale=eta_scale, tp_weight=tp_weight, rows=rows,
        observations=targets, spectral_norm_method="dense")
    assert diagnostics["spectral_norm_numerator"] == pytest.approx(numerator, rel=2e-14)
    assert diagnostics["prediction_norm_denominator"] == pytest.approx(denominator, rel=2e-14)
    assert diagnostics["eta"] == pytest.approx(eta, rel=2e-14)
    np.testing.assert_allclose(updated, change @ expected_next, rtol=2e-14, atol=3e-15)


@pytest.mark.parametrize("bad", [np.ones(4), np.ones((4, 0)), np.ones((3, 1)),
                                np.ones((9, 1)), np.ones((1, 1)), np.ones((4, 1, 1)),
                                np.full((4, 1), np.nan), np.full((4, 1), np.inf)])
def test_invalid_coefficients_rejected(bad):
    with pytest.raises(ValueError):
        gell_mann_coefficients_to_choi(bad)


@pytest.mark.parametrize("bad", [None, True, "Gell-Mann", "pauli", ["choi"]])
def test_invalid_basis_rejected_before_optimization(bad):
    data = generate_quiroga_sensing_data(1, channel_seed=2)
    with pytest.raises(ValueError, match="initialization_basis"):
        campaign.run_campaign(data, initialization_basis=bad, n_steps=1,
                              eta_scale=1e-4, tp_weight=0.5)


def test_n8_mapping_has_factor_sized_memory_not_dense_basis_storage():
    d, rank = 256, 2
    coefficients = np.ones((d * d, rank), dtype=np.complex128)
    tracemalloc.start()
    try:
        mapped = gell_mann_coefficients_to_choi(coefficients)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert mapped.shape == coefficients.shape
    assert peak < 32 * 1024**2  # A dense d**2 by d**2 complex map would need 64 GiB.
    assert np.linalg.norm(mapped) == pytest.approx(np.linalg.norm(coefficients), rel=3e-15)


def _campaign_arguments():
    return dict(n_steps=2, frames_steps=2, metrics_frequency=1, rank=2,
                initialization_seed=29, sampling_seed=31, eta_scale=1e-4,
                tp_weight=0.5, batch_size=4, spectral_norm_method="dense")


def test_default_and_explicit_choi_preserve_legacy_factor_and_all_trajectories():
    data = generate_quiroga_sensing_data(1, channel_seed=3)
    metadata, payload, default = campaign.run_campaign(data, **_campaign_arguments())
    _, explicit_payload, explicit = campaign.run_campaign(
        data, initialization_basis="choi", **_campaign_arguments())
    old = unpack_factor(make_factor_initial_point(data, rank=2, seed=29), data.process_dimension, 2)
    np.testing.assert_array_equal(payload["initial_factor"], old)
    np.testing.assert_array_equal(explicit_payload["initial_factor"], old)
    assert metadata["configuration"]["initialization_basis"] == "choi"
    assert metadata["initialization"] == initialization_descriptor("choi")
    for method in campaign.METHODS:
        assert default[method].status == explicit[method].status == "complete"
        for name in ("final_factor", "checkpoint_steps", "cumulative_data_row_accesses",
                     *campaign.COMMON_METRICS):
            np.testing.assert_array_equal(getattr(default[method], name), getattr(explicit[method], name))
        for key in default[method].diagnostics:
            if "seconds" not in key:
                np.testing.assert_array_equal(default[method].diagnostics[key], explicit[method].diagnostics[key])


def test_gell_mann_reuses_same_coefficient_draw_across_calls_and_methods():
    data = generate_quiroga_sensing_data(1, channel_seed=3)
    arguments = dict(_campaign_arguments(), initialization_basis="gell-mann")
    metadata, payload, traces = campaign.run_campaign(data, **arguments)
    _, alone_payload, alone_traces = campaign.run_campaign(data, methods=("printed-adafgd",), **arguments)
    old = unpack_factor(make_factor_initial_point(data, rank=2, seed=29), data.process_dimension, 2)
    np.testing.assert_allclose(payload["initial_factor"], _dense_change(data.d) @ old,
                               rtol=2e-15, atol=3e-16)
    np.testing.assert_array_equal(alone_payload["initial_factor"], payload["initial_factor"])
    np.testing.assert_array_equal(alone_traces["printed-adafgd"].final_factor,
                                  traces["printed-adafgd"].final_factor)
    assert np.linalg.norm(payload["initial_factor"]) ** 2 == pytest.approx(data.d, rel=1e-15)
    assert not np.allclose(payload["initial_factor"], old)
    assert metadata["initialization"]["operator_basis"]["convention"] == GELL_MANN_CONVENTION
    for trace in traces.values():
        assert trace.measurement_loss_sum[0] == pytest.approx(traces["printed-adafgd"].measurement_loss_sum[0])
        assert trace.tp_violation[0] == pytest.approx(traces["printed-adafgd"].tp_violation[0])


def test_cli_and_pickle_free_archive_record_initializer_and_dependencies(tmp_path):
    archive = tmp_path / "gm.npz"
    metadata, payload, _ = campaign.main([
        "--n-qubits", "1", "--methods", "armijo-fgd", "--steps", "1",
        "--metrics-every", "1", "--initialization-basis", "gell-mann",
        "--initialization-seed", "47", "--eta-scale", "0.0001", "--tp-weight", "0.5",
        "--save", str(archive), "--quiet",
    ])
    assert metadata["configuration"]["initialization_basis"] == "gell-mann"
    assert metadata["initialization"] == initialization_descriptor("gell-mann")
    for name in ("qpt_quiroga_initialization.py", "quantum_process_tomography.py"):
        path = Path(campaign.__file__).with_name(name)
        assert metadata["source_sha256"][name] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(archive, allow_pickle=False) as arrays:
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)
        assert json.loads(str(arrays["metadata_json"].item())) == metadata
        np.testing.assert_array_equal(arrays["initial_factor"], payload["initial_factor"])
    assert json.loads(archive.with_suffix(".json").read_text()) == metadata
