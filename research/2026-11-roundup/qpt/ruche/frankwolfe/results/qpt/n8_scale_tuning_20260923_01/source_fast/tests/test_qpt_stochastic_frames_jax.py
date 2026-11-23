import numpy as np
import pytest


jax = pytest.importorskip("jax")

from paper.experiments.quantum_process_tomography import (  # noqa: E402
    QPTData,
    QPTMeasurementObjective,
    make_factor_initial_point,
    run_qpt_stochastic_frames,
)
from paper.experiments.quantum_process_tomography_jax import (  # noqa: E402
    run_qpt_stochastic_frames_jax,
    save_qpt_jax_result,
    select_jax_device,
)


def make_tiny_qpt_data(seed=0):
    rng = np.random.default_rng(seed)
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    A_basis = np.stack(
        [identity, pauli_x, pauli_y, pauli_z], axis=0
    ) / np.sqrt(2.0)

    vectors = rng.standard_normal((6, 4)) + 1j * rng.standard_normal((6, 4))
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


def run_tiny(data, **overrides):
    options = dict(
        n_steps=2,
        rank=1,
        tau=3.0,
        batch_size=2,
        initialization_seed=3,
        sampling_seed=9,
        rho_schedule=[0.4, 0.6].__getitem__,
        smoothing_schedule=[2.0, 1.5].__getitem__,
        step_size_schedule=[0.2, 0.1].__getitem__,
        metrics_frequency=1,
        device="cpu",
        precision="64",
        warmup=True,
        show_progress=False,
    )
    options.update(overrides)
    return run_qpt_stochastic_frames_jax(data, **options)


def test_cpu_scan_and_step_smoke_are_seed_reproducible():
    data = make_tiny_qpt_data()

    scan = run_tiny(data, execution_mode="scan")
    step = run_tiny(data, execution_mode="step")

    np.testing.assert_array_equal(scan.batch_indices, step.batch_indices)
    np.testing.assert_allclose(scan.final_x, step.final_x, atol=1e-11)
    np.testing.assert_allclose(
        scan.final_gradient_estimate,
        step.final_gradient_estimate,
        atol=1e-11,
    )
    np.testing.assert_allclose(scan.estimated_gaps, step.estimated_gaps)
    for result in (scan, step):
        assert result.backend == "jax"
        assert result.device_platform == "cpu"
        assert result.precision == "64"
        assert result.final_x.shape == (8,)
        assert np.isfinite(result.final_x).all()
        assert np.isfinite(result.measurement_loss).all()
        np.testing.assert_array_equal(result.checkpoint_steps, [0, 1, 2])


@pytest.mark.parametrize("rank", [1, 2])
def test_jax_trajectory_matches_numpy_with_identical_sampling(rank):
    data = make_tiny_qpt_data(seed=11)
    options = dict(
        n_steps=3,
        rank=rank,
        tau=3.0,
        batch_size=2,
        initialization_seed=5,
        sampling_seed=17,
        rho_schedule=[0.3, 0.5, 0.7].__getitem__,
        smoothing_schedule=[2.0, 1.5, 1.0].__getitem__,
        step_size_schedule=[0.2, 0.15, 0.1].__getitem__,
        metrics_frequency=1,
        show_progress=False,
    )
    numpy_result = run_qpt_stochastic_frames(data, **options)
    jax_result = run_qpt_stochastic_frames_jax(
        data,
        **options,
        device="cpu",
        precision="64",
        execution_mode="scan",
        warmup=True,
    )

    expected_batches = np.random.default_rng(
        options["sampling_seed"]
    ).integers(
        0,
        data.m,
        size=(options["n_steps"], options["batch_size"]),
    )
    np.testing.assert_array_equal(jax_result.batch_indices, expected_batches)
    np.testing.assert_array_equal(
        numpy_result.algorithm.objective.last_batch_indices,
        expected_batches[-1],
    )
    np.testing.assert_allclose(
        jax_result.final_x, numpy_result.final_x, atol=1e-11
    )
    np.testing.assert_allclose(
        jax_result.final_gradient_estimate,
        numpy_result.algorithm.gradient_estimate,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        jax_result.estimated_gaps, numpy_result.estimated_gaps, atol=1e-11
    )


def test_explicit_gpu_selection_never_silently_falls_back():
    try:
        gpu_devices = list(jax.devices("gpu"))
    except RuntimeError:
        gpu_devices = []

    if gpu_devices:
        assert select_jax_device("gpu", 0).platform == "gpu"
    else:
        with pytest.raises(RuntimeError, match="backend is unavailable"):
            select_jax_device("gpu", 0)


def test_first_sample_initializes_estimator_before_momentum_recurrence():
    data = make_tiny_qpt_data(seed=2)
    x0 = make_factor_initial_point(data, rank=1, seed=7)
    rho = [0.2, 0.6]
    result = run_tiny(
        data,
        x0=x0,
        sampling_seed=13,
        rho_schedule=rho.__getitem__,
        smoothing_schedule=2.0,
        step_size_schedule=0.0,
        execution_mode="scan",
    )

    objective = QPTMeasurementObjective(data, rank=1)
    first = objective.gradient_for_indices(x0, result.batch_indices[0])
    second = objective.gradient_for_indices(x0, result.batch_indices[1])
    expected = (1.0 - rho[1]) * first + rho[1] * second

    np.testing.assert_allclose(result.final_x, x0, atol=1e-12)
    np.testing.assert_allclose(
        result.final_gradient_estimate, expected, atol=1e-11
    )


def test_callable_schedules_are_materialized_once_and_returned():
    data = make_tiny_qpt_data(seed=3)
    requested = {
        "rho": [0.8, 0.6, 0.4],
        "beta": [3.0, 2.0, 1.0],
        "step": [0.3, 0.2, 0.1],
    }
    calls = {name: [] for name in requested}

    def schedule(name):
        def value(iteration):
            calls[name].append(iteration)
            return requested[name][iteration]

        return value

    result = run_tiny(
        data,
        n_steps=3,
        rho_schedule=schedule("rho"),
        smoothing_schedule=schedule("beta"),
        step_size_schedule=schedule("step"),
        metrics_frequency=0,
        execution_mode="scan",
    )

    assert calls == {name: [0, 1, 2] for name in requested}
    np.testing.assert_allclose(result.momentum_weights, requested["rho"])
    np.testing.assert_allclose(
        result.smoothing_parameters, requested["beta"]
    )
    np.testing.assert_allclose(result.step_sizes, requested["step"])


def test_rank_two_is_finite_and_save_records_device_versions(tmp_path):
    data = make_tiny_qpt_data(seed=4)
    result = run_tiny(
        data,
        n_steps=1,
        rank=2,
        rho_schedule=0.5,
        smoothing_schedule=2.0,
        step_size_schedule=0.1,
        metrics_frequency=0,
        execution_mode="scan",
    )

    assert result.final_x.shape == (16,)
    assert np.isfinite(result.final_x).all()
    assert np.isfinite(result.final_gradient_estimate).all()
    assert np.isfinite(result.estimated_gaps).all()
    assert result.jax_version
    assert result.jaxlib_version

    path = tmp_path / "jax-result.npz"
    save_qpt_jax_result(path, result)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["meta_backend"].item() == "jax"
        assert archive["meta_device_platform"].item() == "cpu"
        assert archive["meta_device_kind"].item() == result.device_kind
        assert archive["meta_jax_version"].item() == result.jax_version
        assert archive["meta_jaxlib_version"].item() == result.jaxlib_version
        np.testing.assert_array_equal(
            archive["batch_indices"], result.batch_indices
        )


def test_rank_two_zero_gradient_keeps_the_zero_iterate():
    data = QPTData(
        f_vector=np.zeros(1),
        D_tensors=np.zeros((1, 1, 1), dtype=np.complex128),
        A_basis=np.zeros((1, 1, 1), dtype=np.complex128),
    )
    result = run_qpt_stochastic_frames_jax(
        data,
        n_steps=1,
        rank=2,
        tau=2.0,
        batch_size=1,
        x0=np.zeros(4),
        rho_schedule=0.2,
        smoothing_schedule=1.0,
        step_size_schedule=1.0,
        metrics_frequency=0,
        device="cpu",
        precision="64",
        execution_mode="scan",
        warmup=True,
        show_progress=False,
    )

    np.testing.assert_array_equal(result.final_x, np.zeros(4))
    np.testing.assert_array_equal(result.final_gradient_estimate, np.zeros(4))
    np.testing.assert_array_equal(result.estimated_gaps, np.zeros(1))
