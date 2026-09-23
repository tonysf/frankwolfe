"""Independent equivalence checks for the memory-bounded product-state backend."""

import os
import numpy as np
import pytest

from paper.experiments.qpt_structured_data import standard_local_basis, standard_local_measurements
from paper.experiments.qpt_structured_operators import (
    product_state_measurement_bank, product_state_measurement_values,
    product_state_measurement_loss_and_gradient, measurement_loss_and_gradient,
    rank_one_measurement_vectors, rank_one_measurement_values,
    pauli_matrices_to_coefficients,
)


@pytest.mark.parametrize("n,rank", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("complex_basis", [False, True])
def test_same_factor_predictions_gradients_and_directional_derivatives(n, rank, complex_basis):
    rng = np.random.default_rng(63)
    basis, measurements = standard_local_basis(), standard_local_measurements()
    if complex_basis:
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))
        basis = np.einsum("kj,jab->kab", rotation, basis)
        # Independent physical matrices define a complex-basis sensing bank.
        left = rng.normal(size=(24, 2)) + 1j*rng.normal(size=(24, 2))
        right = rng.normal(size=(24, 2)) + 1j*rng.normal(size=(24, 2))
        matrices = left[:, :, None] * right[:, None, :].conj()
        vectors = pauli_matrices_to_coefficients(matrices, basis).T
        measurements = vectors[:, :, None] * vectors[:, None, :].conj()
    bank = product_state_measurement_bank(measurements, basis)
    assert bank is not None
    u = (rng.normal(size=(4**n, rank)) + 1j*rng.normal(size=(4**n, rank)))/np.sqrt(4**n)
    symbols = rng.integers(0, 24, (31, n), dtype=np.int32)
    observations = rng.normal(size=31)
    values = product_state_measurement_values(u, symbols, bank)
    reference = rank_one_measurement_values(u, symbols, rank_one_measurement_vectors(measurements))
    np.testing.assert_allclose(values, reference, rtol=3e-12, atol=3e-12)
    loss, gradient = product_state_measurement_loss_and_gradient(u, symbols, observations, bank)
    old_loss, old_gradient = measurement_loss_and_gradient(u, symbols, observations, measurements)
    np.testing.assert_allclose(loss, old_loss, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(gradient, old_gradient, rtol=3e-11, atol=3e-11)
    direction = rng.normal(size=u.shape) + 1j*rng.normal(size=u.shape)
    delta = 1e-6
    plus = product_state_measurement_loss_and_gradient(u+delta*direction, symbols, observations, bank)[0]
    minus = product_state_measurement_loss_and_gradient(u-delta*direction, symbols, observations, bank)[0]
    np.testing.assert_allclose((plus-minus)/(2*delta), np.vdot(gradient, direction).real, rtol=2e-7, atol=2e-7)


def test_product_state_bank_rejects_invalid_factorizations():
    basis, measurements = standard_local_basis(), standard_local_measurements()
    assert product_state_measurement_bank(measurements, 2*basis) is None
    bad = np.zeros_like(measurements)
    bad[:, 0, 0] = 1  # Coefficient rank one, but physical Q=I/sqrt(2) has rank two.
    assert product_state_measurement_bank(bad, basis) is None


@pytest.mark.parametrize("rank", [1, 2])
def test_product_state_scan_matches_existing_noisy_trajectory(rank):
    pytest.importorskip("jax")
    from paper.experiments.qpt_generate_data import generate_on_demand_data
    from paper.experiments.quantum_process_tomography_structured_jax import run_qpt_structured_jax
    data = generate_on_demand_data(2, channel_seed=5, noise_seed=3)
    options = dict(n_steps=19, rank=rank, tau=3., batch_size=11, chunk_steps=4,
                   metrics_frequency=4, metric_mode="sampled", metric_samples=17,
                   metric_batch_size=5, rho_schedule=lambda k: .7/(k+1)**.3,
                   smoothing_schedule=lambda k: 20/(k+1)**.25,
                   step_size_schedule=lambda k: .2/(k+1)**.6, initialization_seed=7,
                   sampling_seed=9, device=os.environ.get("QPT_TEST_DEVICE", "cpu"),
                   precision="64", store_checkpoints=True, show_progress=False)
    old = run_qpt_structured_jax(data, measurement_backend="rank-one", **options)
    new = run_qpt_structured_jax(data, measurement_backend="product-state", **options)
    for field in ("checkpoint_factors", "gradient_estimate", "measurement_loss", "tp_violation",
                  "smoothed_gap", "process_fidelity_proxy"):
        np.testing.assert_allclose(getattr(new, field), getattr(old, field), rtol=2e-9, atol=2e-10)
    assert new.metadata["batch_symbols_sha256"] == old.metadata["batch_symbols_sha256"]
    assert new.metadata["measurement_backend"] == "product-state"
