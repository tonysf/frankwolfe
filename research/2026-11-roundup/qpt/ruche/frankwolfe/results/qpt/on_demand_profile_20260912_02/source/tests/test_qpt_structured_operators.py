"""Small dense references protect the structured operators' complex conventions."""

import itertools

import numpy as np
import pytest

from paper.experiments.qpt_structured_operators import (
    apply_measurements,
    measurement_loss_and_gradient,
    measurement_values,
    pack_factor,
    pauli_coefficients_to_matrices,
    pauli_matrices_to_coefficients,
    product_measurement_vectors,
    rank_one_measurement_loss_and_gradient,
    rank_one_measurement_values,
    rank_one_measurement_vectors,
    trace_preserving_jacobian_adjoint,
    trace_preserving_loss_and_gradient,
    trace_preserving_map,
    trace_preserving_residual,
)


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _local_basis(kind="real_y"):
    identity = np.eye(2)
    pauli_x = np.array([[0, 1], [1, 0]])
    real_y = np.array([[0, -1], [1, 0]])
    pauli_z = np.diag([1, -1])
    if kind == "arbitrary":
        return _complex(np.random.default_rng(5), (4, 2, 2)) / 3
    y = 1j * real_y if kind == "hermitian_y" else real_y
    return np.asarray([identity, pauli_x, y, pauli_z], dtype=complex) / np.sqrt(2)


def _kron_product(factors):
    result = np.ones((1, 1), dtype=complex)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def _global_basis(local, n):
    return np.stack(
        [_kron_product(local[list(row)]) for row in itertools.product(range(4), repeat=n)]
    )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("rank", [1, 2])
def test_local_measurement_application_matches_arbitrary_complex_kronecker(n, rank):
    rng = np.random.default_rng(10 + n)
    local = _complex(rng, (24, 4, 4))
    symbols = rng.integers(0, 24, size=(5, n))
    factor = _complex(rng, (4**n, rank))
    expected = np.stack([_kron_product(local[row]) @ factor for row in symbols])
    np.testing.assert_allclose(apply_measurements(factor, symbols, local), expected, atol=1e-11)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rank", [1, 2])
def test_measurement_values_and_packed_gradient_match_dense_and_finite_difference(n, rank):
    rng = np.random.default_rng(32 + n)
    raw = _complex(rng, (24, 4, 4)) / 4
    local = 0.5 * (raw + raw.conj().swapaxes(1, 2))
    symbols = rng.integers(0, 24, size=(7, n))
    factor = _complex(rng, (4**n, rank)) / 3
    observations = rng.normal(size=7)
    dense = np.stack([_kron_product(local[row]) for row in symbols])
    chi = factor @ factor.conj().T
    expected_values = np.einsum("bij,ij->b", dense.conj(), chi).real
    residual = expected_values - observations
    expected_gradient = 2 * np.einsum("b,bij->ij", residual, dense) @ factor / len(symbols)
    loss, gradient = measurement_loss_and_gradient(factor, symbols, observations, local)
    np.testing.assert_allclose(measurement_values(factor, symbols, local), expected_values, atol=1e-12)
    np.testing.assert_allclose(loss, 0.5 * np.mean(residual**2), atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)

    direction = _complex(rng, factor.shape)
    epsilon = 1e-6
    plus, _ = measurement_loss_and_gradient(factor + epsilon * direction, symbols, observations, local)
    minus, _ = measurement_loss_and_gradient(factor - epsilon * direction, symbols, observations, local)
    finite_difference = (plus - minus) / (2 * epsilon)
    packed_derivative = np.dot(pack_factor(gradient), pack_factor(direction))
    np.testing.assert_allclose(packed_derivative, finite_difference, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("rank", [1, 2])
def test_rank_one_measurements_match_generic_application_and_ignore_eigenvector_phase(n, rank):
    rng = np.random.default_rng(153 + n)
    original_vectors = _complex(rng, (24, 4)) / 2
    bank = np.einsum("bi,bj->bij", original_vectors, original_vectors.conj())
    recovered_vectors = rank_one_measurement_vectors(bank)
    assert recovered_vectors is not None
    np.testing.assert_allclose(
        np.einsum("bi,bj->bij", recovered_vectors, recovered_vectors.conj()), bank, atol=1e-12
    )
    symbols = rng.integers(0, 24, size=(7, n))
    factor = _complex(rng, (4**n, rank)) / 3
    observations = rng.normal(size=7)
    expected_values = measurement_values(factor, symbols, bank)
    expected_loss, expected_gradient = measurement_loss_and_gradient(factor, symbols, observations, bank)
    arbitrary_phases = np.exp(1j * rng.normal(size=24))[:, None]
    for vectors in (original_vectors, recovered_vectors, recovered_vectors * arbitrary_phases):
        product = product_measurement_vectors(symbols, vectors)
        expected_product = np.stack([_kron_product(vectors[row]).reshape(-1) for row in symbols])
        np.testing.assert_allclose(product, expected_product, atol=1e-12)
        np.testing.assert_allclose(rank_one_measurement_values(factor, symbols, vectors), expected_values, atol=1e-11)
        loss, gradient = rank_one_measurement_loss_and_gradient(factor, symbols, observations, vectors)
        np.testing.assert_allclose(loss, expected_loss, atol=1e-11)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-11)


@pytest.mark.parametrize("kind", ["rank_two", "negative", "nonhermitian"])
def test_rank_one_detection_rejects_incompatible_banks(kind):
    bank = np.zeros((24, 4, 4), dtype=complex)
    bank[:, 0, 0] = 1
    if kind == "rank_two":
        bank[17, 1, 1] = 0.2
    elif kind == "negative":
        bank[17, 0, 0] = -1
    else:
        bank[17, 1, 0] = 0.2j
    assert rank_one_measurement_vectors(bank) is None


def test_rank_one_detection_and_gradient_support_zero_rows():
    vectors = rank_one_measurement_vectors(np.zeros((24, 4, 4), dtype=complex))
    np.testing.assert_array_equal(vectors, np.zeros((24, 4)))
    loss, gradient = rank_one_measurement_loss_and_gradient(
        np.ones((16, 2), dtype=complex), np.array([[0, 1], [2, 3]]), np.array([1, 3]), vectors
    )
    np.testing.assert_allclose(loss, 2.5)
    np.testing.assert_array_equal(gradient, np.zeros((16, 2)))


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("kind", ["real_y", "hermitian_y", "arbitrary"])
def test_pauli_transform_and_hermitian_adjoint_match_dense(n, kind):
    rng = np.random.default_rng(18 + n)
    local = _local_basis(kind)
    basis = _global_basis(local, n)
    factor = _complex(rng, (4**n, 2))
    matrices = _complex(rng, (2, 2**n, 2**n))
    transformed = pauli_coefficients_to_matrices(factor, local)
    adjoint = pauli_matrices_to_coefficients(matrices, local)
    np.testing.assert_allclose(transformed, np.einsum("ka,kij->aij", factor, basis), atol=1e-12)
    np.testing.assert_allclose(adjoint, np.einsum("kij,aij->ka", basis.conj(), matrices), atol=1e-12)
    np.testing.assert_allclose(np.vdot(transformed, matrices), np.vdot(factor, adjoint), atol=1e-12)
    if kind != "arbitrary":
        np.testing.assert_allclose(pauli_matrices_to_coefficients(transformed, local), factor, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("kind", ["real_y", "arbitrary"])
def test_trace_map_and_jacobian_adjoint_match_dense_for_complex_factors(n, rank, kind):
    rng = np.random.default_rng(53 + n)
    local = _local_basis(kind)
    basis = _global_basis(local, n)
    dense_b = np.einsum("mki,nkj->nmij", basis.conj(), basis)
    factor = _complex(rng, (4**n, rank)) / 3
    chi = factor @ factor.conj().T
    expected_map = np.einsum("nm,nmij->ij", chi, dense_b)
    np.testing.assert_allclose(trace_preserving_map(factor, local), expected_map, atol=1e-12)
    np.testing.assert_allclose(trace_preserving_residual(factor, local), expected_map - np.eye(2**n), atol=1e-12)

    matrix = _complex(rng, (2**n, 2**n))  # Deliberately non-Hermitian.
    chi_gradient = np.einsum("nmij,ij->nm", dense_b.conj(), matrix)
    expected_gradient = (chi_gradient + chi_gradient.conj().T) @ factor
    gradient = trace_preserving_jacobian_adjoint(factor, matrix, local)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12)

    direction = _complex(rng, factor.shape)
    epsilon = 1e-6
    derivative = (
        trace_preserving_map(factor + epsilon * direction, local)
        - trace_preserving_map(factor - epsilon * direction, local)
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        np.vdot(derivative, matrix).real,
        np.dot(pack_factor(gradient), pack_factor(direction)),
        rtol=1e-6,
        atol=1e-8,
    )


def test_moreau_penalty_gradient_matches_legacy_objective_and_finite_difference():
    from paper.experiments.quantum_process_tomography import QPTData, QPTMeasurementObjective

    rng = np.random.default_rng(73)
    local = _local_basis("real_y")
    basis = _global_basis(local, 2)
    raw = _complex(rng, (24, 4, 4)) / 4
    bank = raw + raw.conj().swapaxes(1, 2)
    symbols = rng.integers(0, 24, size=(5, 2))
    data = QPTData(
        f_vector=rng.normal(size=5),
        D_tensors=np.stack([_kron_product(bank[row]) for row in symbols]),
        A_basis=basis,
    )
    objective = QPTMeasurementObjective(data, rank=2)
    factor = _complex(rng, (16, 2)) / 5
    vector = pack_factor(factor)
    loss, measurement_gradient = measurement_loss_and_gradient(factor, symbols, data.f_vector, bank)
    expected_loss, expected_gradient = objective.loss_and_gradient(vector)
    np.testing.assert_allclose(loss, expected_loss, atol=1e-12)
    np.testing.assert_allclose(pack_factor(measurement_gradient), expected_gradient, atol=1e-12)

    beta = 0.73
    residual = trace_preserving_residual(factor, local)
    penalty_gradient = trace_preserving_jacobian_adjoint(factor, residual, local) / beta
    combined_loss, combined_gradient = trace_preserving_loss_and_gradient(factor, local)
    np.testing.assert_allclose(combined_loss, 0.5 * np.linalg.norm(residual)**2, atol=1e-12)
    np.testing.assert_allclose(combined_gradient / beta, penalty_gradient, atol=1e-12)
    np.testing.assert_allclose(
        pack_factor(penalty_gradient),
        objective.linear_operator_adjoint_at(vector, objective.trace_preserving_residual(vector)) / beta,
        atol=1e-12,
    )
    direction = _complex(rng, factor.shape)
    epsilon = 1e-6
    plus = trace_preserving_residual(factor + epsilon * direction, local)
    minus = trace_preserving_residual(factor - epsilon * direction, local)
    finite_difference = (np.linalg.norm(plus)**2 - np.linalg.norm(minus)**2) / (4 * epsilon * beta)
    np.testing.assert_allclose(np.vdot(penalty_gradient, direction).real, finite_difference, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("rank_one", [False, True])
def test_jax_jit_operators_and_autodiff_match_numpy(rank_one):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(91)
    local = _local_basis("real_y")
    raw = _complex(rng, (24, 4, 4)) / 3
    bank = 0.5 * (raw + raw.conj().swapaxes(1, 2))
    if rank_one:
        local_vectors = _complex(rng, (24, 4)) / 3
        bank = np.einsum("bi,bj->bij", local_vectors, local_vectors.conj())
        measurement_function = rank_one_measurement_loss_and_gradient
        compiled_measurements = local_vectors
    else:
        measurement_function = measurement_loss_and_gradient
        compiled_measurements = bank
    symbols = rng.integers(0, 24, size=(7, 2))
    observations = rng.normal(size=7)
    factor = _complex(rng, (16, 2)) / 5
    beta = 0.3
    expected_loss, expected_gradient = measurement_loss_and_gradient(factor, symbols, observations, bank)
    residual = trace_preserving_residual(factor, local)
    expected_gradient += trace_preserving_jacobian_adjoint(factor, residual, local) / beta
    expected_loss += np.linalg.norm(residual)**2 / (2 * beta)

    def compiled_objective(vector, local_matrices, paulis, selected, observed):
        factor_jax = vector[:32].reshape((16, 2)) + 1j * vector[32:].reshape((16, 2))
        loss, gradient = measurement_function(factor_jax, selected, observed, local_matrices, xp=jnp)
        tp_loss, tp_gradient = trace_preserving_loss_and_gradient(factor_jax, paulis, xp=jnp)
        gradient += tp_gradient / beta
        loss += tp_loss / beta
        return loss, pack_factor(gradient, xp=jnp)

    # Limit precision configuration to the test, rather than changing the
    # process-wide JAX mode used by other experiment tests.
    with jax.experimental.enable_x64():
        arguments = tuple(jnp.asarray(value) for value in (pack_factor(factor), compiled_measurements, local, symbols, observations))
        loss, gradient = jax.jit(compiled_objective)(*arguments)
        autodiff = jax.jit(jax.grad(lambda *args: compiled_objective(*args)[0]))(*arguments)
        np.testing.assert_allclose(loss, expected_loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gradient, pack_factor(expected_gradient), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gradient, autodiff, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("factor", [np.ones(4), np.ones((5, 1)), np.ones((4, 0)), np.ones((1, 1))])
def test_invalid_factor_shapes_are_rejected(factor):
    with pytest.raises(ValueError):
        pauli_coefficients_to_matrices(factor, _local_basis())


@pytest.mark.parametrize("symbols,error", [([[0.0]], TypeError), ([[24]], IndexError), ([[-1]], IndexError), ([], ValueError)])
def test_invalid_measurement_symbols_are_rejected(symbols, error):
    with pytest.raises(error):
        apply_measurements(np.ones((4, 1)), symbols, np.zeros((24, 4, 4)))
