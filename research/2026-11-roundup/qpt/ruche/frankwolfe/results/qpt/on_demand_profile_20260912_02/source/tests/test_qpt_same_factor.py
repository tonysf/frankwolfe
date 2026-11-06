"""Independent full-data CPU references for identical saved QPT factors."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from paper.experiments import qpt_same_factor as evaluator
from paper.experiments.qpt_structured_data import (
    StructuredQPTData, standard_local_basis, standard_local_measurements,
)
from paper.experiments.quantum_process_tomography import (
    QPTData, QPTMeasurementObjective, create_operator_norm_factor_lmo,
    pack_factor, unpack_factor,
)


def fixture(n=1, rank=1, custom_basis=False):
    rng = np.random.default_rng(902 + n + rank)
    basis = standard_local_basis()
    if custom_basis:
        # Preserve orthonormality while making no Hermitian-basis assumption.
        basis = basis * np.exp(1j * np.array([0.1, -0.3, 0.25, 0.4]))[:, None, None]
    data = StructuredQPTData(
        n, standard_local_measurements(), basis,
        observations=rng.normal(size=24**n),
        truth_factor=np.ones((4**n, 1), dtype=np.complex128),
    )
    factor = (rng.normal(size=(4**n, rank)) + 1j * rng.normal(size=(4**n, rank))) / 3
    return data, factor


def kron(matrices):
    result = np.ones((1, 1), dtype=complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def monolithic(data, factor, beta=2.5, tau=10):
    symbols = data.indices_to_symbols(np.arange(data.m))
    dense = np.stack([kron(data.local_measurements[row]) for row in symbols])
    basis = np.stack([kron(data.local_basis[list(row)])
                      for row in itertools.product(range(4), repeat=data.n_qubits)])
    objective = QPTMeasurementObjective(QPTData(data.observations, dense, basis), rank=factor.shape[1])
    packed = pack_factor(factor)
    loss, gradient = objective.loss_and_gradient(packed)
    residual = objective.trace_preserving_residual(packed)
    tp_gradient = objective.linear_operator_adjoint_at(packed, residual)
    combined = gradient + tp_gradient / beta
    atom = create_operator_norm_factor_lmo(*factor.shape, tau)(combined)
    return {
        "measurement_loss": loss,
        "measurement_gradient": unpack_factor(gradient, *factor.shape),
        "tp_residual": residual,
        "tp_violation": np.linalg.norm(residual),
        "tp_loss": 0.5 * np.vdot(residual, residual).real,
        "tp_gradient": unpack_factor(tp_gradient, *factor.shape),
        "moreau_gradient": unpack_factor(tp_gradient / beta, *factor.shape),
        "full_gradient": unpack_factor(combined, *factor.shape),
        "smoothed_objective": loss + 0.5 * np.vdot(residual, residual).real / beta,
        "lmo_atom": unpack_factor(atom, *factor.shape),
        "smoothed_gap": float(np.dot(combined, packed - atom)),
        "factor_operator_norm": np.linalg.norm(factor, ord=2),
        "feasibility_violation": max(0.0, np.linalg.norm(factor, ord=2) - tau),
    }


@pytest.mark.parametrize("n,rank", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("backend", ["tensor", "rank-one"])
def test_full_data_streaming_matches_monolithic_legacy(n, rank, backend):
    data, factor = fixture(n, rank)
    before = [value.copy() for value in (factor, data.observations, data.local_measurements, data.local_basis)]
    expected = monolithic(data, factor)
    result = evaluator.evaluate_same_factor(data, factor, 2.5, 10, batch_size=7, measurement_backend=backend)
    assert result["m"] == 24**n
    assert result["batches"] == (data.m + 6) // 7
    assert result["evaluated_measurement_backend"] == backend
    assert result["dense_measurement_entries"] == data.m * data.process_dimension**2
    for path in ("dense", "structured"):
        assert set(result[path]) == set(expected)
        for name, value in expected.items():
            np.testing.assert_allclose(result[path][name], value, rtol=2e-10, atol=2e-12, err_msg=name)
    for actual, original in zip((factor, data.observations, data.local_measurements, data.local_basis), before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("n,rank", [(1, 2), (2, 1)])
def test_custom_complex_orthonormal_basis_and_real_antisymmetric_y(n, rank):
    data, factor = fixture(n, rank, custom_basis=True)
    assert not np.allclose(data.local_basis, data.local_basis.conj().swapaxes(1, 2))
    expected = monolithic(data, factor)
    result = evaluator.evaluate_same_factor(data, factor, 2.5, 10, batch_size=19)
    for path in ("dense", "structured"):
        for name in ("tp_residual", "tp_gradient", "smoothed_objective", "smoothed_gap"):
            np.testing.assert_allclose(result[path][name], expected[name], atol=2e-11)


def test_partial_final_batch_uses_every_original_observation_without_regeneration(monkeypatch):
    data, factor = fixture()
    expected = monolithic(data, factor)
    original_gradient = evaluator.measurement_loss_and_gradient
    seen = []

    def observed(u, symbols, observations, bank, **kwargs):
        seen.append(observations.copy())
        return original_gradient(u, symbols, observations, bank, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Stored observations must not be regenerated or resampled")

    monkeypatch.setattr(evaluator, "measurement_loss_and_gradient", observed)
    monkeypatch.setattr(data, "observations_for_symbols", forbidden)
    monkeypatch.setattr(data, "sample_symbols", forbidden)
    result = evaluator.evaluate_same_factor(data, factor, 2.5, 10, batch_size=7)
    assert list(map(len, seen)) == [7, 7, 7, 3]
    np.testing.assert_array_equal(np.concatenate(seen), data.observations)
    for path in ("dense", "structured"):
        np.testing.assert_allclose(result[path]["measurement_loss"], expected["measurement_loss"], atol=1e-13)
        np.testing.assert_allclose(result[path]["measurement_gradient"], expected["measurement_gradient"], atol=1e-13)


def test_dense_decoder_is_independent_of_structured_adapter(monkeypatch):
    data, factor = fixture(2)
    expected = monolithic(data, factor)
    original_decoder = data.indices_to_symbols

    def wrong_decoder(indices):
        result = original_decoder(indices)
        result[:, 0] = (result[:, 0] + 1) % 24
        return result

    monkeypatch.setattr(data, "indices_to_symbols", wrong_decoder)
    result = evaluator.evaluate_same_factor(data, factor, 2.5, 10, batch_size=31)
    np.testing.assert_allclose(result["dense"]["measurement_gradient"], expected["measurement_gradient"], atol=1e-12)
    assert not np.allclose(result["structured"]["measurement_gradient"], expected["measurement_gradient"], atol=1e-8)


@pytest.mark.parametrize("path", ["dense", "structured"])
def test_full_gradient_matches_real_directional_finite_difference(path):
    data, factor = fixture(1, 2, custom_basis=True)
    rng = np.random.default_rng(31)
    direction = rng.normal(size=factor.shape) + 1j * rng.normal(size=factor.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    center = evaluator.evaluate_same_factor(data, factor, 2.5, 10, batch_size=7)[path]
    plus = evaluator.evaluate_same_factor(data, factor + epsilon * direction, 2.5, 10, batch_size=7)[path]
    minus = evaluator.evaluate_same_factor(data, factor - epsilon * direction, 2.5, 10, batch_size=7)[path]
    derivative = (plus["smoothed_objective"] - minus["smoothed_objective"]) / (2 * epsilon)
    np.testing.assert_allclose(np.vdot(center["full_gradient"], direction).real, derivative, rtol=1e-6, atol=1e-9)


def test_reference_cap_rejects_before_any_global_dense_construction(monkeypatch):
    data, factor = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference construction must follow the cap check")

    monkeypatch.setattr(evaluator, "_global_reference_basis", forbidden)
    with pytest.raises(ValueError, match="exceeds max_reference_bytes"):
        evaluator.evaluate_same_factor(data, factor, 1, 10, max_reference_bytes=1)
    assert evaluator.reference_allocation_estimate(64, 1, 32) < 128 * 2**20
    assert evaluator.reference_allocation_estimate(256, 1, 32) > 128 * 2**20


def test_no_full_measurement_bank_is_allocated(monkeypatch):
    data, factor = fixture(2)
    original_empty = evaluator.np.empty
    allocated_rows = []

    def bounded_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 3 and shape[1:] == (16, 16):
            allocated_rows.append(shape[0])
            assert shape[0] <= 7
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(evaluator.np, "empty", bounded_empty)
    evaluator.evaluate_same_factor(data, factor, 2.5, 10, batch_size=7)
    assert max(allocated_rows) == 7
    assert sum(allocated_rows) == data.m


@pytest.mark.parametrize("field,value", [
    ("beta", 0), ("beta", np.nan), ("beta", 1j), ("tau", -1), ("tau", True),
    ("batch_size", 0), ("batch_size", True), ("batch_size", 1.5),
    ("max_reference_bytes", 0), ("max_reference_bytes", -1),
    ("measurement_backend", "auto"),
])
def test_invalid_arguments_rejected(field, value):
    data, factor = fixture()
    args = dict(beta=2.5, tau=10)
    args[field] = value
    with pytest.raises((ValueError, TypeError)):
        evaluator.evaluate_same_factor(data, factor, **args)


@pytest.mark.parametrize("factor_kind", ["real", "complex64", "nan", "wrong_shape"])
def test_factor_requires_finite_complex128_with_matching_shape(factor_kind):
    data, factor = fixture()
    if factor_kind == "real":
        factor = factor.real
    elif factor_kind == "complex64":
        factor = factor.astype(np.complex64)
    elif factor_kind == "nan":
        factor[0, 0] = np.nan
    else:
        factor = factor[:, 0]
    with pytest.raises((ValueError, TypeError)):
        evaluator.evaluate_same_factor(data, factor, 1, 10)


def test_noiseless_mode_is_rejected_and_infeasibility_is_reported():
    data, factor = fixture()
    noiseless = StructuredQPTData(1, data.local_measurements, data.local_basis,
                                 truth_factor=data.truth_factor, observation_mode="noiseless")
    with pytest.raises(ValueError, match="stored"):
        evaluator.evaluate_same_factor(noiseless, factor, 1, 10)
    result = evaluator.evaluate_same_factor(data, factor, 1, 0.01)
    for path in ("dense", "structured"):
        assert result[path]["feasibility_violation"] > 0
        np.testing.assert_allclose(np.linalg.norm(result[path]["lmo_atom"], ord=2), 0.01)


def test_import_and_evaluation_do_not_import_jax():
    code = """
import importlib.abc
import sys
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith('jax.') or fullname == 'jaxlib':
            raise AssertionError('CPU-only evaluator attempted JAX import')
sys.meta_path.insert(0, NoJax())
import numpy as np
from paper.experiments.qpt_structured_data import StructuredQPTData, standard_local_basis, standard_local_measurements
from paper.experiments.qpt_same_factor import evaluate_same_factor
data = StructuredQPTData(1, standard_local_measurements(), standard_local_basis(), observations=np.arange(24, dtype=float))
result = evaluate_same_factor(data, np.ones((4,1), dtype=np.complex128)/2, 2.5, 10)
assert result['m'] == 24
assert not any(name == 'jax' or name.startswith('jax.') or name == 'jaxlib' for name in sys.modules)
"""
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
