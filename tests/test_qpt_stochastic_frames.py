import numpy as np
import pytest

from paper.experiments.quantum_process_tomography import (
    PowerSchedule,
    QPTData,
    QPTMeasurementObjective,
    checkpoint_iterations,
    create_operator_norm_factor_lmo,
    make_factor_initial_point,
    pack_factor,
    run_qpt_stochastic_frames,
    save_qpt_result,
    unpack_factor,
)


def make_tiny_qpt_data(seed=0):
    rng = np.random.default_rng(seed)
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    A_basis = np.stack(
        [identity, pauli_x, pauli_y, pauli_z],
        axis=0,
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


def test_trace_preserving_jacobian_and_adjoint_match_real_inner_product():
    data = make_tiny_qpt_data()
    objective = QPTMeasurementObjective(data, rank=2)
    rng = np.random.default_rng(8)
    vector = make_factor_initial_point(data, rank=2, seed=4)
    direction = rng.standard_normal(vector.shape)
    raw_matrix = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    matrix = 0.5 * (raw_matrix + raw_matrix.conj().T)
    epsilon = 1e-6

    derivative = (
        objective.linear_operator(vector + epsilon * direction)
        - objective.linear_operator(vector - epsilon * direction)
    ) / (2.0 * epsilon)
    lhs = np.vdot(derivative, matrix).real
    rhs = np.dot(
        direction,
        objective.linear_operator_adjoint_at(vector, matrix),
    )

    np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-9)


def test_measurement_minibatch_gradients_average_to_the_full_gradient():
    data = make_tiny_qpt_data()
    objective = QPTMeasurementObjective(data)
    vector = make_factor_initial_point(data, seed=2)

    full_gradient = objective.gradient(vector)
    singleton_average = np.mean(
        [
            objective.gradient_for_indices(vector, [index])
            for index in range(data.m)
        ],
        axis=0,
    )
    full_batch_gradient = objective.gradient_for_indices(
        vector, np.arange(data.m)
    )

    np.testing.assert_allclose(singleton_average, full_gradient, atol=1e-12)
    np.testing.assert_allclose(full_batch_gradient, full_gradient, atol=1e-12)


def test_measurement_gradient_matches_real_directional_derivative():
    data = make_tiny_qpt_data()
    objective = QPTMeasurementObjective(data)
    rng = np.random.default_rng(21)
    vector = make_factor_initial_point(data, seed=3)
    direction = rng.standard_normal(vector.shape)
    epsilon = 1e-6

    finite_difference = (
        objective.evaluate(vector + epsilon * direction)
        - objective.evaluate(vector - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = np.dot(objective.gradient(vector), direction)

    np.testing.assert_allclose(finite_difference, analytic, rtol=1e-6, atol=1e-9)


def test_full_moreau_gradient_matches_real_directional_derivative():
    data = make_tiny_qpt_data()
    objective = QPTMeasurementObjective(data)
    rng = np.random.default_rng(22)
    vector = make_factor_initial_point(data, seed=4)
    direction = rng.standard_normal(vector.shape)
    identity = np.eye(data.d)
    beta = 0.7
    epsilon = 1e-6

    def smoothed_objective(point):
        residual = objective.linear_operator(point) - identity
        return objective.evaluate(point) + 0.5 * np.linalg.norm(residual) ** 2 / beta

    residual = objective.linear_operator(vector) - identity
    gradient = objective.gradient(vector) + (
        objective.linear_operator_adjoint_at(vector, residual) / beta
    )
    finite_difference = (
        smoothed_objective(vector + epsilon * direction)
        - smoothed_objective(vector - epsilon * direction)
    ) / (2.0 * epsilon)

    np.testing.assert_allclose(
        finite_difference,
        np.dot(gradient, direction),
        rtol=1e-6,
        atol=1e-9,
    )


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


def test_checkpoint_iterations_always_include_nonmultiple_final_step():
    np.testing.assert_array_equal(checkpoint_iterations(5, 2), [0, 2, 4, 5])
    np.testing.assert_array_equal(checkpoint_iterations(5, 0), [0, 5])


def test_qpt_runner_forwards_schedules_and_uses_sparse_posthoc_metrics():
    data = make_tiny_qpt_data()
    result = run_qpt_stochastic_frames(
        data,
        n_steps=3,
        rank=1,
        tau=3.0,
        batch_size=2,
        sampling_seed=11,
        rho_schedule=lambda iteration: [1.0, 0.5, 0.25][iteration],
        smoothing_schedule=lambda iteration: [2.0, 1.0, 0.5][iteration],
        step_size_schedule=lambda iteration: [0.2, 0.1, 0.05][iteration],
        metrics_frequency=2,
        show_progress=False,
    )

    np.testing.assert_array_equal(result.checkpoint_steps, [0, 2, 3])
    np.testing.assert_allclose(result.momentum_weights, [1.0, 0.5, 0.25])
    np.testing.assert_allclose(result.smoothing_parameters, [2.0, 1.0, 0.5])
    np.testing.assert_allclose(
        result.checkpoint_smoothing_parameters, [2.0, 0.5, 0.5]
    )
    np.testing.assert_allclose(result.step_sizes, [0.2, 0.1, 0.05])
    np.testing.assert_array_equal(
        result.cumulative_sampled_measurements, [2, 4, 6]
    )
    assert np.isnan(result.algorithm.func_vals).all()
    assert result.measurement_loss.shape == (3,)
    assert result.exact_smoothed_gap.shape == (3,)
    assert result.rank == 1
    assert result.tau == 3.0
    assert np.linalg.norm(result.final_factor, ord=2) <= 3.0 + 1e-10
    np.testing.assert_allclose(result.final_chi, result.final_chi.conj().T)
    assert np.linalg.eigvalsh(result.final_chi)[0] >= -1e-10

    initial_x = make_factor_initial_point(data, rank=1, seed=0)
    objective = QPTMeasurementObjective(data, rank=1)
    residual = objective.trace_preserving_residual(initial_x)
    combined_gradient = objective.gradient(initial_x) + (
        objective.linear_operator_adjoint_at(initial_x, residual) / 2.0
    )
    lmo = create_operator_norm_factor_lmo(data.process_dimension, 1, 3.0)
    expected_gap = np.dot(
        combined_gradient,
        initial_x - lmo(combined_gradient),
    )
    np.testing.assert_allclose(result.exact_smoothed_gap[0], expected_gap)
    np.testing.assert_allclose(
        result.smoothed_objective[0],
        objective.evaluate(initial_x) + 0.25 * np.linalg.norm(residual) ** 2,
    )


def test_smoothing_schedule_changes_the_nonconvex_factor_trajectory():
    data = make_tiny_qpt_data()
    common = dict(
        n_steps=2,
        tau=3.0,
        batch_size=2,
        initialization_seed=7,
        sampling_seed=8,
        rho_schedule=0.25,
        step_size_schedule=0.25,
        metrics_frequency=0,
        show_progress=False,
    )

    strong = run_qpt_stochastic_frames(
        data, smoothing_schedule=0.1, **common
    )
    weak = run_qpt_stochastic_frames(
        data, smoothing_schedule=100.0, **common
    )

    assert not np.allclose(strong.final_x, weak.final_x)
    np.testing.assert_allclose(strong.smoothing_parameters, [0.1, 0.1])
    np.testing.assert_allclose(weak.smoothing_parameters, [100.0, 100.0])


def test_seeded_stochastic_oracles_draw_identical_measurement_batches():
    data = make_tiny_qpt_data()
    vector = make_factor_initial_point(data, seed=9)
    first = QPTMeasurementObjective(data, batch_size=3, seed=19)
    second = QPTMeasurementObjective(data, batch_size=3, seed=19)

    for _ in range(4):
        np.testing.assert_allclose(
            first.stochastic_gradient(vector),
            second.stochastic_gradient(vector),
        )
        np.testing.assert_array_equal(
            first.last_batch_indices, second.last_batch_indices
        )


def test_power_schedule_supports_caps_and_constant_exponent():
    capped = PowerSchedule(scale=4.0, offset=1.0, exponent=1.0, cap=1.0)
    constant = PowerSchedule(scale=0.25, offset=3.0, exponent=0.0)

    assert capped(0) == 1.0
    assert capped(7) == 0.5
    assert constant(100) == 0.25


def test_qpt_hdf5_loader_maps_source_dataset_names(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = make_tiny_qpt_data()
    path = tmp_path / "qpt.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("f_jax_vector", data=source.f_vector)
        handle.create_dataset("D_jax_tensors", data=source.D_tensors)
        handle.create_dataset("A_jax_basis", data=source.A_basis)
        handle.create_dataset("B_jax_tensors", data=source.B_tensors)
        handle.create_dataset("Chi_star_tensor", data=source.chi_star)

    loaded = QPTData.from_hdf5(path)

    np.testing.assert_allclose(loaded.f_vector, source.f_vector)
    np.testing.assert_allclose(loaded.D_tensors, source.D_tensors)
    np.testing.assert_allclose(loaded.A_basis, source.A_basis)
    np.testing.assert_allclose(loaded.B_tensors, source.B_tensors)
    np.testing.assert_allclose(loaded.chi_star, source.chi_star)


def test_saved_qpt_result_contains_checkpoint_beta_and_scalar_metadata(tmp_path):
    data = make_tiny_qpt_data()
    result = run_qpt_stochastic_frames(
        data,
        n_steps=1,
        metrics_frequency=0,
        show_progress=False,
    )
    path = tmp_path / "result.npz"

    save_qpt_result(
        path,
        result,
        metadata={
            "formulation": "nonconvex_factor_nonlinear_tp_smoothing"
        },
    )

    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_allclose(
            archive["checkpoint_smoothing_parameters"],
            result.checkpoint_smoothing_parameters,
        )
        assert archive["meta_formulation"].item() == (
            "nonconvex_factor_nonlinear_tp_smoothing"
        )
        assert archive["meta_nonlinear_composite"].item() == (
            "jacobian_adjoint_as_linear"
        )
        assert archive["meta_rank"].item() == 1
        assert archive["meta_tau"].item() == 10.0
        np.testing.assert_allclose(archive["final_factor"], result.final_factor)


def test_malformed_qpt_shapes_are_rejected_before_optimization():
    with pytest.raises(ValueError, match="same number of measurements"):
        QPTData(
            f_vector=np.zeros(2),
            D_tensors=np.zeros((3, 4, 4)),
            A_basis=np.zeros((4, 2, 2)),
        )


def test_empty_dataset_and_nonintegral_indices_are_rejected():
    with pytest.raises(ValueError, match="at least one measurement"):
        QPTData(
            f_vector=np.zeros(0),
            D_tensors=np.zeros((0, 4, 4)),
            A_basis=make_tiny_qpt_data().A_basis,
        )

    data = make_tiny_qpt_data()
    objective = QPTMeasurementObjective(data)
    vector = make_factor_initial_point(data)
    with pytest.raises(TypeError, match="indices must be integers"):
        objective.gradient_for_indices(vector, [0.5])


def test_nonfinite_initial_factor_is_rejected_explicitly():
    data = make_tiny_qpt_data()
    x0 = make_factor_initial_point(data)
    x0[0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        run_qpt_stochastic_frames(
            data,
            n_steps=1,
            x0=x0,
            show_progress=False,
        )
