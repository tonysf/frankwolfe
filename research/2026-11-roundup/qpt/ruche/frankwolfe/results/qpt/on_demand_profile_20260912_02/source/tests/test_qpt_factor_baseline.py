import numpy as np

from paper.experiments.qpt_bfw_factor_baseline import (
    QPTFactorObjective,
    create_operator_norm_factor_lmo,
    make_factor_initial_point,
    pack_factor,
    run_qpt_factor_stochastic_frames,
    save_qpt_factor_result,
    unpack_factor,
)
from paper.experiments.quantum_process_tomography import QPTData


def make_tiny_qpt_data(seed=0):
    rng = np.random.default_rng(seed)
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    A_basis = np.stack(
        [identity, pauli_x, pauli_y, pauli_z], axis=0
    ) / np.sqrt(2.0)
    vectors = rng.standard_normal((7, 4)) + 1j * rng.standard_normal((7, 4))
    D_tensors = np.einsum(
        "mi,mj->mij", vectors, vectors.conj(), optimize=True
    )
    D_tensors /= np.linalg.norm(D_tensors, axis=(1, 2))[:, None, None]
    chi_star = np.eye(4, dtype=np.complex128) / 2.0
    f_vector = np.einsum(
        "mab,ab->m", D_tensors.conj(), chi_star, optimize=True
    ).real
    return QPTData(
        f_vector=f_vector,
        D_tensors=D_tensors,
        A_basis=A_basis,
        chi_star=chi_star,
    )


def test_factor_pack_roundtrip_preserves_real_inner_products():
    rng = np.random.default_rng(2)
    first = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    second = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))

    np.testing.assert_allclose(unpack_factor(pack_factor(first), 4, 2), first)
    np.testing.assert_allclose(
        np.dot(pack_factor(first), pack_factor(second)),
        np.vdot(first, second).real,
    )


def test_factor_gradient_matches_packed_real_directional_derivative():
    data = make_tiny_qpt_data()
    objective = QPTFactorObjective(data, rank=2, lam=0.05)
    x = make_factor_initial_point(data, rank=2, seed=3)
    rng = np.random.default_rng(4)
    direction = rng.standard_normal(x.shape)
    epsilon = 1e-6

    finite_difference = (
        objective.evaluate(x + epsilon * direction)
        - objective.evaluate(x - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = np.dot(objective.gradient(x), direction)

    np.testing.assert_allclose(finite_difference, analytic, rtol=1e-6, atol=1e-8)


def test_factor_singleton_gradients_average_to_full_gradient():
    data = make_tiny_qpt_data()
    objective = QPTFactorObjective(data, rank=1, lam=0.05)
    x = make_factor_initial_point(data, rank=1, seed=5)

    singleton_average = np.mean(
        [
            objective.gradient_for_indices(x, [index])
            for index in range(data.m)
        ],
        axis=0,
    )

    np.testing.assert_allclose(singleton_average, objective.gradient(x), atol=1e-12)


def test_operator_norm_lmo_attains_negative_dual_nuclear_norm():
    rng = np.random.default_rng(6)
    gradient = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    packed_gradient = pack_factor(gradient)
    tau = 3.0
    lmo = create_operator_norm_factor_lmo(4, 2, tau)

    atom = unpack_factor(lmo(packed_gradient), 4, 2)

    np.testing.assert_allclose(np.linalg.norm(atom, ord=2), tau)
    expected = -tau * np.linalg.svd(gradient, compute_uv=False).sum()
    np.testing.assert_allclose(np.vdot(gradient, atom).real, expected)


def test_factor_runner_preserves_original_parameters_and_inert_smoothing():
    data = make_tiny_qpt_data()
    common = dict(
        n_steps=3,
        rank=1,
        tau=10.0,
        lam=0.05,
        batch_size=2,
        initialization_seed=7,
        sampling_seed=8,
        rho_schedule=0.5,
        step_size_schedule=0.25,
        metrics_frequency=2,
        show_progress=False,
    )

    first = run_qpt_factor_stochastic_frames(
        data, smoothing_schedule=0.1, **common
    )
    second = run_qpt_factor_stochastic_frames(
        data, smoothing_schedule=100.0, **common
    )

    np.testing.assert_allclose(first.final_x, second.final_x)
    np.testing.assert_array_equal(first.checkpoint_steps, [0, 2, 3])
    np.testing.assert_allclose(first.momentum_weights, [0.5] * 3)
    np.testing.assert_allclose(first.smoothing_parameters, [0.1] * 3)
    np.testing.assert_allclose(first.step_sizes, [0.25] * 3)
    np.testing.assert_allclose(
        first.qpt_bfw_exact_gap, 0.5 * first.exact_fw_gap
    )
    np.testing.assert_allclose(
        first.qpt_bfw_estimated_gaps, 0.5 * first.estimated_gaps
    )
    np.testing.assert_array_equal(
        first.cumulative_sampled_measurements, [2, 4, 6]
    )
    assert np.linalg.norm(first.final_factor, ord=2) <= 10.0 + 1e-10


def test_factor_archive_includes_configuration_and_both_gap_conventions(tmp_path):
    result = run_qpt_factor_stochastic_frames(
        make_tiny_qpt_data(),
        n_steps=1,
        rank=1,
        tau=9.0,
        lam=0.04,
        batch_size=2,
        metrics_frequency=0,
        show_progress=False,
    )
    path = tmp_path / "factor_result.npz"

    save_qpt_factor_result(path, result)

    with np.load(path, allow_pickle=False) as archive:
        assert archive["meta_rank"].item() == 1
        assert archive["meta_tau"].item() == 9.0
        assert archive["meta_lam"].item() == 0.04
        assert archive["meta_batch_size"].item() == 2
        np.testing.assert_allclose(
            archive["qpt_bfw_exact_gap"],
            0.5 * archive["exact_fw_gap"],
        )
