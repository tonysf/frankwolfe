"""Independent dense references for the Quiroga-style global-probe/POVM backend."""

import math

import numpy as np
import pytest

from paper.experiments.qpt_generate_data import _haar_unitary
from paper.experiments.qpt_observation_noise import fixed_row_noise
from paper.experiments.qpt_quiroga_sensing import (
    QuirogaSensingData,
    QuirogaSensingDesign,
    choi_output_partial_trace,
    default_povm_constants,
    generate_quiroga_sensing_data,
    haar_unitary_choi_truth,
    kraus_operators,
    main,
    povm_elements,
    probe_states,
    pure_target_process_fidelity,
    scaling_summary,
    throwaway_min_eigenvalue,
    trace_preserving_jacobian_adjoint,
    trace_preserving_loss_and_gradient,
    trace_preserving_residual,
    unitary_choi_factor,
    validate_povm_constants,
)
from paper.experiments.qpt_structured_operators import pack_factor


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _reference_probes(d):
    """Quiroga's generic states (Baldwin et al. Eq. (9)), listed directly."""
    basis = np.eye(d, dtype=complex)
    pairs = [(k, l) for k in range(d - 1) for l in range(k + 1, d)]
    return np.asarray(
        [basis[k] for k in range(d)]
        + [(basis[k] + basis[l]) / np.sqrt(2) for k, l in pairs]
        + [(basis[k] + 1j * basis[l]) / np.sqrt(2) for k, l in pairs]
    )


def _reference_povm(d, a, b):
    """Baldwin et al. (2014) Eq. (18) with E_2d = I - sum(others)."""
    basis = np.eye(d, dtype=complex)

    def outer(x, y):
        return np.outer(x, y.conj())

    elements = [a * outer(basis[0], basis[0])]
    elements += [b * (np.eye(d) + outer(basis[0], basis[m]) + outer(basis[m], basis[0])) for m in range(1, d)]
    elements += [b * (np.eye(d) + 1j * outer(basis[0], basis[m]) - 1j * outer(basis[m], basis[0]))
                 for m in range(1, d)]
    elements.append(np.eye(d) - sum(elements))
    return np.asarray(elements)


def _reference_sensing(d, a, b):
    """Rows s = 2d p + j with D_s = rho_p^T (x) E_j in input-major Choi order."""
    povm = _reference_povm(d, a, b)
    return np.asarray([np.kron(np.outer(psi, psi.conj()).T, element)
                       for psi in _reference_probes(d) for element in povm])


def _born(kraus, d, a, b):
    """Tr(E_j E(rho_p)) with E(rho) = sum_a K_a rho K_a^H."""
    values = []
    for psi in _reference_probes(d):
        output = sum(K @ np.outer(psi, psi.conj()) @ K.conj().T for K in kraus)
        values.extend(np.trace(element @ output).real for element in _reference_povm(d, a, b))
    return np.asarray(values)


def _partial_trace_output(chi, d):
    return np.trace(chi.reshape(d, d, d, d), axis1=1, axis2=3)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_default_povm_is_psd_complete_and_matches_baldwin_eq18(n):
    d = 2**n
    a, b = default_povm_constants(d)
    assert a == b == pytest.approx(2 / (4 * d - 1 + math.sqrt(8 * d - 7)), rel=1e-15)
    elements = povm_elements(d)
    assert elements.shape == (2 * d, d, d)
    np.testing.assert_allclose(elements, _reference_povm(d, a, b), atol=1e-15)
    np.testing.assert_array_equal(elements, elements.conj().swapaxes(1, 2))
    eigenvalues = np.linalg.eigvalsh(elements)
    assert eigenvalues.min() >= -1e-14
    np.testing.assert_allclose(elements.sum(axis=0), np.eye(d), atol=1e-14)
    # The throw-away element keeps the documented spectral floor b > 0.
    assert eigenvalues[-1].min() == pytest.approx(b, rel=1e-10)
    assert throwaway_min_eigenvalue(d, a, b) == pytest.approx(b, rel=1e-12)
    w = np.r_[0, np.full(d - 1, 1 - 1j)]
    e0 = np.eye(d)[0]
    closed_form = ((1 - 2 * b * (d - 1)) * np.eye(d) - a * np.outer(e0, e0)
                   - b * (np.outer(e0, w.conj()) + np.outer(w, e0)))
    np.testing.assert_allclose(elements[-1], closed_form, atol=1e-15)
    stacked = np.concatenate((elements.real.reshape(2 * d, -1), elements.imag.reshape(2 * d, -1)), axis=1)
    assert np.linalg.matrix_rank(stacked) == 2 * d


@pytest.mark.parametrize("n", [1, 2, 3])
def test_povm_determines_generic_pure_states_by_flammia_inversion(n):
    d = 2**n
    a, b = default_povm_constants(d)
    rng = np.random.default_rng(40 + n)
    state = _complex(rng, d)
    state *= np.exp(-1j * np.angle(state[0])) / np.linalg.norm(state)
    probabilities = np.einsum("i,jik,k->j", state.conj(), povm_elements(d), state).real
    r0 = np.sqrt(probabilities[0] / a)
    real = (probabilities[1:d] - b) / (2 * b * r0)
    imag = (b - probabilities[d:2 * d - 1]) / (2 * b * r0)
    np.testing.assert_allclose(np.r_[r0, real + 1j * imag], state, atol=1e-12)


@pytest.mark.parametrize("a, b", [(0.0, 0.1), (0.1, 0.0), (-0.1, 0.1), (0.1, 0.2), (1.0, 0.01), (np.nan, 0.1)])
def test_invalid_povm_constants_are_rejected(a, b):
    with pytest.raises(ValueError):
        validate_povm_constants(4, a, b)
    with pytest.raises(ValueError):
        QuirogaSensingDesign(2, a, b)


def test_povm_constants_must_be_given_together():
    with pytest.raises(ValueError, match="both"):
        QuirogaSensingDesign(2, 0.1, None)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_probe_order_span_and_row_indexing(n):
    d = 2**n
    states = probe_states(d)
    np.testing.assert_allclose(states, _reference_probes(d), atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0)
    projectors = np.einsum("pi,pj->pij", states, states.conj()).reshape(d * d, -1)
    assert np.linalg.matrix_rank(projectors) == d * d
    design = QuirogaSensingDesign(n)
    assert (design.probe_count, design.outcome_count, design.row_count) == (d * d, 2 * d, 2 * d**3)
    rows = np.arange(design.row_count)
    probes, outcomes = design.decode_rows(rows)
    np.testing.assert_array_equal(probes, rows // (2 * d))
    np.testing.assert_array_equal(outcomes, rows % (2 * d))
    np.testing.assert_array_equal(design.encode_rows(probes, outcomes), rows)
    for bad, error in (([design.row_count], IndexError), ([-1], IndexError), (np.array([0.5]), ValueError),
                       (np.empty(0, dtype=int), ValueError)):
        with pytest.raises(error):
            design.validate_rows(bad)


def test_two_qubit_generic_probes_are_not_all_local_product_states():
    schmidt_ranks = [np.linalg.matrix_rank(state.reshape(2, 2)) for state in probe_states(4)]
    # (|00> + |11>)/sqrt2, (|01> + |10>)/sqrt2 and their i-phase partners.
    assert schmidt_ranks.count(2) == 4 and schmidt_ranks.count(1) == 12


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("constants", [None, (0.3, 0.05)])
def test_structured_rows_match_dense_reference(n, rank, constants):
    design = QuirogaSensingDesign(n) if constants is None else QuirogaSensingDesign(n, *constants)
    d, rows = design.d, np.arange(design.row_count)
    dense = _reference_sensing(d, design.povm_a, design.povm_b)
    np.testing.assert_allclose(design.dense_sensing_matrices(), dense, atol=1e-15)
    rng = np.random.default_rng(100 * n + 10 * rank + (constants is None))
    factor = _complex(rng, (d * d, rank)) / 3
    chi = factor @ factor.conj().T
    expected = np.einsum("sij,ij->s", dense.conj(), chi).real
    np.testing.assert_allclose(expected, _born(kraus_operators(factor), d, design.povm_a, design.povm_b), atol=1e-13)
    np.testing.assert_allclose(design.values(factor, rows), expected, atol=1e-13)
    np.testing.assert_allclose(design.full_values(factor), expected, atol=1e-13)
    np.testing.assert_allclose(design.full_values(factor, probe_chunk=3), expected, atol=1e-13)
    np.testing.assert_allclose(design.probe_values(factor, np.arange(d * d)).reshape(-1), expected, atol=1e-13)

    targets = expected + rng.normal(size=expected.shape)
    residual = expected - targets
    for reduction, scale in (("mean", 1 / rows.size), ("sum", 1.0)):
        dense_gradient = 2 * scale * np.einsum("s,sij->ij", residual, dense) @ factor
        for loss, gradient in (
            design.loss_and_gradient(factor, rows, targets, reduction=reduction),
            design.full_loss_and_gradient(factor, targets, reduction=reduction),
            design.full_loss_and_gradient(factor, targets, reduction=reduction, probe_chunk=5),
        ):
            np.testing.assert_allclose(loss, 0.5 * scale * np.sum(residual**2), rtol=1e-12)
            np.testing.assert_allclose(gradient, dense_gradient, atol=1e-12)

    sample = rng.integers(0, rows.size, size=11)
    loss, gradient = design.loss_and_gradient(factor, sample, targets[sample])
    np.testing.assert_allclose(loss, 0.5 * np.mean(residual[sample] ** 2), rtol=1e-12)
    np.testing.assert_allclose(
        gradient, 2 * np.einsum("s,sij->ij", residual[sample], dense[sample]) @ factor / sample.size, atol=1e-12)

    weights = rng.normal(size=rows.size)
    matrix = _complex(rng, (d * d, 3))
    dense_adjoint = np.einsum("s,sij->ij", weights, dense)
    np.testing.assert_allclose(design.full_adjoint(matrix, weights), dense_adjoint @ matrix, atol=1e-12)
    np.testing.assert_allclose(design.adjoint(matrix, rows, weights), dense_adjoint @ matrix, atol=1e-12)
    np.testing.assert_allclose(design.dense_adjoint(weights), dense_adjoint, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rank", [1, 2])
def test_gradients_match_central_finite_differences(n, rank):
    design = QuirogaSensingDesign(n)
    rng = np.random.default_rng(300 + 10 * n + rank)
    factor = _complex(rng, (design.process_dimension, rank)) / 3
    targets = rng.uniform(size=design.row_count)
    rows = rng.integers(0, design.row_count, size=9)
    direction = _complex(rng, factor.shape)
    epsilon = 1e-6
    for function in (
        lambda u: design.full_loss_and_gradient(u, targets, reduction="sum"),
        lambda u: design.full_loss_and_gradient(u, targets),
        lambda u: design.loss_and_gradient(u, rows, targets[rows]),
        trace_preserving_loss_and_gradient,
    ):
        _, gradient = function(factor)
        plus = function(factor + epsilon * direction)[0]
        minus = function(factor - epsilon * direction)[0]
        np.testing.assert_allclose(np.dot(pack_factor(gradient), pack_factor(direction)),
                                   (plus - minus) / (2 * epsilon), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("n, independent_rows, real_dimension", [(1, 16, 16), (2, 128, 256)])
def test_full_row_set_is_linearly_independent(n, independent_rows, real_dimension):
    dense = QuirogaSensingDesign(n).dense_sensing_matrices()
    stacked = np.concatenate((dense.real.reshape(dense.shape[0], -1), dense.imag.reshape(dense.shape[0], -1)), axis=1)
    assert dense.shape[0] == independent_rows == np.linalg.matrix_rank(stacked)
    # n=1 is informationally complete for all Choi matrices; n=2 is not.
    assert dense.shape[1] ** 2 == real_dimension


@pytest.mark.parametrize("n", [1, 2])
def test_full_design_identifies_a_unitary_channel_up_to_phase(n):
    data = generate_quiroga_sensing_data(n, channel_seed=0)
    design, truth, rows = data.design, data.truth_factor, np.arange(data.m)
    size = truth.size

    def unpack(x):
        return (x[:size] + 1j * x[size:]).reshape(truth.shape)

    # Rows are quadratic in U, so polarization gives exact directional derivatives.
    jacobian = np.stack([0.5 * (design.values(truth + unpack(e), rows) - design.values(truth - unpack(e), rows))
                         for e in np.eye(2 * size)], axis=1)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    assert np.sum(singular > 1e-8 * singular[0]) == 2 * size - 1
    np.testing.assert_allclose(jacobian @ pack_factor(1j * truth), 0.0, atol=1e-13)

    # A generic quasi-Newton solve of the summed loss recovers the truth.
    from scipy.optimize import minimize

    targets = data.all_observations()

    def objective(x):
        loss, gradient = design.full_loss_and_gradient(unpack(x), targets, reduction="sum")
        return loss, pack_factor(gradient)

    start = _complex(np.random.default_rng(100), truth.shape) * np.sqrt(data.d / (2 * size))
    solution = minimize(objective, pack_factor(start), jac=True, method="L-BFGS-B",
                        options=dict(maxiter=5000, ftol=1e-30, gtol=1e-14))
    assert solution.fun < 1e-20 and data.fidelity(unpack(solution.x)) > 1 - 1e-9


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rank", [1, 2])
def test_trace_preservation_is_exact_in_choi_coordinates(n, rank):
    d = 2**n
    rng = np.random.default_rng(500 + 10 * n + rank)
    factor = _complex(rng, (d * d, rank)) / 2
    chi = factor @ factor.conj().T
    mapped = _partial_trace_output(chi, d)
    np.testing.assert_allclose(choi_output_partial_trace(factor), mapped, atol=1e-13)
    kraus = kraus_operators(factor)
    np.testing.assert_allclose(mapped, sum(K.conj().T @ K for K in kraus).T, atol=1e-13)
    residual = trace_preserving_residual(factor)
    np.testing.assert_allclose(residual, mapped - np.eye(d), atol=1e-13)
    loss, gradient = trace_preserving_loss_and_gradient(factor)
    np.testing.assert_allclose(loss, 0.5 * np.linalg.norm(residual) ** 2, rtol=1e-12)
    np.testing.assert_allclose(gradient, 2 * np.kron(residual, np.eye(d)) @ factor, atol=1e-12)

    matrix, direction = _complex(rng, (d, d)), _complex(rng, factor.shape)
    derivative = _partial_trace_output(direction @ factor.conj().T + factor @ direction.conj().T, d)
    np.testing.assert_allclose(np.vdot(pack_factor(trace_preserving_jacobian_adjoint(factor, matrix)),
                                       pack_factor(direction)),
                               np.real(np.vdot(matrix, derivative)), rtol=1e-12)

    isometry, _ = np.linalg.qr(_complex(rng, (rank * d, d)))
    channel = np.concatenate([unitary_choi_factor(isometry[a * d:(a + 1) * d]) for a in range(rank)], axis=1)
    tp_loss, tp_gradient = trace_preserving_loss_and_gradient(channel)
    assert tp_loss < 1e-28 and np.linalg.norm(tp_gradient) < 1e-13
    np.testing.assert_allclose(trace_preserving_residual(np.sqrt(0.8) * channel), -0.2 * np.eye(d), atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_haar_truth_is_a_unitary_channel_with_normalized_probabilities(n):
    d = 2**n
    data = generate_quiroga_sensing_data(n, channel_seed=11)
    unitary = _haar_unitary(d, 11)
    np.testing.assert_allclose(data.truth_factor, unitary_choi_factor(unitary))
    np.testing.assert_allclose(data.truth_factor, haar_unitary_choi_truth(n, 11))
    np.testing.assert_allclose(kraus_operators(data.truth_factor)[0], unitary)
    probabilities = data.all_observations()
    a, b = data.design.povm_a, data.design.povm_b
    np.testing.assert_allclose(probabilities, _born([unitary], d, a, b), atol=1e-14)
    probabilities = probabilities.reshape(d * d, 2 * d)
    assert probabilities.min() >= 0
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-13)
    assert np.vdot(data.truth_factor, data.truth_factor).real == pytest.approx(d)
    assert np.linalg.norm(trace_preserving_residual(data.truth_factor)) < 1e-13
    assert data.metadata["design"]["rows"] == 2 * d**3

    truth = data.truth_factor
    assert data.fidelity(truth) == pytest.approx(1.0)
    assert data.fidelity(2 * np.exp(0.7j) * truth) == pytest.approx(1.0)
    other = unitary_choi_factor(_haar_unitary(d, 12))
    chi = other @ other.conj().T
    expected = (truth.conj().T @ chi @ truth).real.item() / (d * np.trace(chi).real)
    assert data.fidelity(other) == pytest.approx(expected) and data.fidelity(other) < 1
    assert pure_target_process_fidelity(other, truth[:, 0]) == pytest.approx(expected)


def test_observation_models_are_deterministic_and_random_access():
    n, d = 2, 4
    base = generate_quiroga_sensing_data(n, channel_seed=1)
    noiseless = base.all_observations()
    rows = np.array([5, 0, 127, 5, 64])
    np.testing.assert_allclose(base.observations_for_rows(rows), noiseless[rows], atol=1e-15)

    gaussian = generate_quiroga_sensing_data(n, channel_seed=1, observation_mode="gaussian",
                                             noise_std=0.05, noise_seed=7)
    full = gaussian.all_observations()
    np.testing.assert_array_equal(full, gaussian.all_observations(probe_chunk=3))
    np.testing.assert_allclose(full - noiseless, fixed_row_noise(np.arange(gaussian.m), 7, 0.05), atol=1e-15)
    np.testing.assert_allclose(gaussian.observations_for_rows(rows), full[rows], atol=1e-15)
    reseeded = generate_quiroga_sensing_data(n, channel_seed=1, observation_mode="gaussian",
                                             noise_std=0.05, noise_seed=8)
    assert not np.allclose(reseeded.all_observations(), full)
    stored = gaussian.materialize()
    assert stored.observation_mode == "stored" and stored.metadata["materialized_from"] == "gaussian"
    assert stored.noise_std == 0.0 and stored.metadata["noise_std"] == 0.05
    np.testing.assert_array_equal(stored.observations_for_rows(rows), full[rows])
    np.testing.assert_array_equal(stored.all_observations(), full)

    shots = generate_quiroga_sensing_data(n, channel_seed=1, observation_mode="shots", shots=1000, shot_seed=3)
    frequencies = shots.all_observations()
    np.testing.assert_array_equal(frequencies, shots.all_observations(probe_chunk=5))
    np.testing.assert_array_equal(shots.observations_for_rows(rows), frequencies[rows])
    per_probe = frequencies.reshape(d * d, 2 * d)
    np.testing.assert_allclose(per_probe.sum(axis=1), 1.0)
    np.testing.assert_allclose(1000 * per_probe, np.round(1000 * per_probe), atol=1e-9)
    assert per_probe.min() >= 0 and np.max(np.abs(frequencies - noiseless)) < 0.1
    np.testing.assert_array_equal(shots.shot_frequencies([7, 2]), per_probe[[7, 2]])
    reseeded = generate_quiroga_sensing_data(n, channel_seed=1, observation_mode="shots", shots=1000, shot_seed=4)
    assert not np.array_equal(reseeded.all_observations(), frequencies)


def test_gaussian_noise_has_the_requested_scale():
    data = generate_quiroga_sensing_data(4, channel_seed=2, observation_mode="gaussian", noise_std=0.1, noise_seed=9)
    noise = data.all_observations() - data.noiseless_values()
    assert noise.size == 8192
    assert abs(noise.mean()) < 4 * 0.1 / np.sqrt(noise.size)
    assert noise.std() == pytest.approx(0.1, rel=0.05)


def test_row_sampling_and_fixed_subsets_are_seed_deterministic():
    data = generate_quiroga_sensing_data(2)
    first = data.sample_rows(np.random.default_rng(4), 50)
    np.testing.assert_array_equal(first, data.sample_rows(np.random.default_rng(4), 50))
    np.testing.assert_array_equal(first, np.random.default_rng(4).integers(0, data.m, size=50))
    subset = data.fixed_row_subset(96, seed=0)
    assert subset.size == 96 and np.all(np.diff(subset) > 0) and subset.max() < data.m
    np.testing.assert_array_equal(subset, data.fixed_row_subset(96, seed=0))
    assert not np.array_equal(subset, data.fixed_row_subset(96, seed=1))
    sampled = data.sample_rows(np.random.default_rng(2), 20, population=subset)
    assert np.isin(sampled, subset).all()
    with pytest.raises(ValueError):
        data.fixed_row_subset(data.m + 1, seed=0)


def test_invalid_data_and_inputs_are_rejected():
    design = QuirogaSensingDesign(1)
    truth = haar_unitary_choi_truth(1, 0)
    for kwargs in (dict(), dict(truth_factor=np.ones((3, 1))),
                   dict(truth_factor=truth, observation_mode="shots"),
                   dict(truth_factor=truth, observation_mode="gaussian", noise_std=-1.0),
                   dict(truth_factor=truth, observation_mode="noiseless", shots=10),
                   dict(truth_factor=truth, observations=np.ones(16)),
                   dict(observation_mode="stored", observations=np.ones(3)),
                   dict(truth_factor=truth, observation_mode="bogus"),
                   dict(truth_factor=truth, noise_seed=-1),
                   # A noise level outside gaussian mode would be silently ignored.
                   dict(truth_factor=truth, noise_std=0.1),
                   dict(truth_factor=truth, observation_mode="shots", shots=10, noise_std=0.1),
                   dict(observation_mode="stored", observations=np.ones(16), noise_std=0.1)):
        with pytest.raises(ValueError):
            QuirogaSensingData(design, **kwargs)
    with pytest.raises(ValueError, match="gaussian"):
        generate_quiroga_sensing_data(1, noise_std=0.05)
    stored = QuirogaSensingData(design, observation_mode="stored", observations=np.arange(16))
    assert stored.observations.dtype == np.float64
    assert QuirogaSensingDesign(20).row_count == 2**61
    with pytest.raises(ValueError, match="int64"):
        QuirogaSensingDesign(21)
    non_tp = QuirogaSensingData(design, 2 * truth, observation_mode="shots", shots=10)
    with pytest.raises(ValueError, match="CPTP"):
        non_tp.all_observations()
    with pytest.raises(ValueError, match="reduction"):
        design.full_loss_and_gradient(truth, np.zeros(design.row_count), reduction="median")
    with pytest.raises(ValueError):
        design.values(np.ones((16, 1)), [0])
    with pytest.raises(ValueError):
        design.full_adjoint(truth, np.zeros(design.row_count) + 1j)
    with pytest.raises(ValueError, match="max_process_dimension"):
        QuirogaSensingDesign(6).dense_adjoint(np.zeros(2 * 64**3))
    with pytest.raises(ValueError, match="max_entries"):
        QuirogaSensingDesign(4).dense_sensing_matrices()


def test_scaling_summary_counts_n2_through_n8(capsys):
    for n in range(2, 9):
        row = scaling_summary(n)
        d = 2**n
        assert row["rows"] == 2 * 8**n == 2 * d**3 and row["probes"] == 4**n and row["outcomes"] == 2 * d
        assert row["local_pauli_rows"] == 24**n
        assert row["adafgd_dense_matrix_bytes"] == 16 * 16**n
    assert scaling_summary(8)["adafgd_dense_matrix_bytes"] == 64 * 2**30
    assert scaling_summary(8)["observation_bytes"] == 256 * 2**20
    rows = main(["--min-qubits", "2", "--max-qubits", "3"])
    assert [row["n_qubits"] for row in rows] == [2, 3]
    assert '"rows": 1024' in capsys.readouterr().out
    for bad in (["--rank", "0"], ["--repeats", "0"], ["--min-qubits", "3", "--max-qubits", "2"]):
        with pytest.raises(SystemExit):
            main(bad)


@pytest.mark.parametrize("n", [1, 2])
def test_jax_operators_match_numpy(n):
    jax = pytest.importorskip("jax")
    from paper.experiments.quantum_process_tomography_jax import _configure_jax_precision
    _configure_jax_precision(jax, "64")
    import jax.numpy as jnp

    design = QuirogaSensingDesign(n)
    rng = np.random.default_rng(700 + n)
    factor = _complex(rng, (design.process_dimension, 2)) / 3
    targets = rng.uniform(size=design.row_count)
    rows = rng.integers(0, design.row_count, size=13)
    weights = rng.normal(size=rows.size)

    @jax.jit
    def evaluate(u, r, t, w):
        loss, gradient = design.loss_and_gradient(u, r, t[r], xp=jnp)
        full_loss, full_gradient = design.full_loss_and_gradient(u, t, reduction="sum", xp=jnp, probe_chunk=3)
        tp_loss, tp_gradient = trace_preserving_loss_and_gradient(u, xp=jnp)
        return (design.values(u, r, xp=jnp), loss, gradient, full_loss, full_gradient,
                design.adjoint(u, r, w, xp=jnp), design.full_values(u, xp=jnp), tp_loss, tp_gradient,
                pure_target_process_fidelity(u[:, :1], u[:, 1:], xp=jnp))

    got = evaluate(jnp.asarray(factor), jnp.asarray(rows), jnp.asarray(targets), jnp.asarray(weights))
    expected = (design.values(factor, rows), *design.loss_and_gradient(factor, rows, targets[rows]),
                *design.full_loss_and_gradient(factor, targets, reduction="sum"),
                design.adjoint(factor, rows, weights), design.full_values(factor),
                *trace_preserving_loss_and_gradient(factor),
                pure_target_process_fidelity(factor[:, :1], factor[:, 1:]))
    for actual, reference in zip(got, expected):
        np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-12, atol=1e-12)
