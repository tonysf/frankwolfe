"""The Quiroga-style FRAMES/adaFGD hooks against the existing dense runners."""

import inspect

import numpy as np
import pytest

from paper.experiments.qpt_quiroga_frames import (
    QuirogaSensingObjective,
    quiroga_adafgd_step,
    run_quiroga_sensing_stochastic_frames,
)
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.quantum_process_tomography import (
    PowerSchedule,
    QPTMeasurementObjective,
    make_factor_initial_point,
    run_qpt_stochastic_frames,
    unpack_factor,
)


def _noisy(n, channel_seed=2):
    return generate_quiroga_sensing_data(n, channel_seed=channel_seed, observation_mode="gaussian",
                                         noise_std=0.03, noise_seed=5)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("subset", [False, True])
def test_objective_matches_the_dense_legacy_objective(n, rank, subset):
    data = _noisy(n)
    rows = data.fixed_row_subset(data.m // 2 + 1, seed=3) if subset else None
    new = QuirogaSensingObjective(data, rank=rank, batch_size=4, seed=9, rows=rows)
    old = QPTMeasurementObjective(data.to_dense_qpt_data(rows), rank=rank, batch_size=4, seed=9)
    x = make_factor_initial_point(data, rank=rank, seed=1)
    for got, expected in zip(new.loss_and_gradient(x), old.loss_and_gradient(x)):
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(new.sensing_values(x), old.sensing_values(x), atol=1e-13)
    np.testing.assert_allclose(new.sensing_values(x, [1, 0, 1]), old.sensing_values(x, [1, 0, 1]), atol=1e-13)
    for _ in range(3):
        np.testing.assert_allclose(new.stochastic_gradient(x), old.stochastic_gradient(x), atol=1e-12)
        np.testing.assert_array_equal(new.last_batch_indices, old.last_batch_indices)
    # The Choi TP map is the transpose of the legacy map in this basis; the
    # residual norm and the Jacobian-adjoint Moreau gradient coincide.
    identity = np.eye(data.d)
    np.testing.assert_allclose(new.linear_operator(x), old.linear_operator(x).T, atol=1e-13)
    np.testing.assert_allclose(new.linear_operator_adjoint_at(x, new.linear_operator(x) - identity),
                               old.linear_operator_adjoint_at(x, old.linear_operator(x) - identity), atol=1e-12)


def test_runner_defaults_are_those_of_the_dense_runner():
    new = inspect.signature(run_quiroga_sensing_stochastic_frames).parameters
    old = inspect.signature(run_qpt_stochastic_frames).parameters
    assert set(old) <= set(new) and set(new) - set(old) == {"rows", "observations"}
    for name, parameter in old.items():
        assert new[name].default == parameter.default, name
    assert new["rows"].default is None and new["observations"].default is None


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("schedules", ["explicit", "default"])
def test_structured_frames_run_reproduces_the_existing_dense_runner(n, subset, schedules):
    data = _noisy(n, channel_seed=4)
    rows = data.fixed_row_subset(3 * data.m // 4, seed=0) if subset else None
    common = dict(
        n_steps=40, rank=1, tau=3.0, batch_size=8, initialization_seed=0, sampling_seed=6,
        metrics_frequency=10, show_progress=False,
    )
    if schedules == "explicit":
        common.update(
            rho_schedule=PowerSchedule(2.0, 4.0, 0.6, cap=1.0),
            smoothing_schedule=PowerSchedule(10.0, 1.0, 0.25),
            step_size_schedule=PowerSchedule(2.0, 2.0, 1.0, cap=1.0),
        )
    else:
        # The default 1/sqrt(k+1) steps amplify the kernels' round-off
        # differences faster (to about 1e-10 by step 40), so stop earlier.
        common.update(n_steps=16, metrics_frequency=4)
    old = run_qpt_stochastic_frames(data.to_dense_qpt_data(rows), **common)
    new = run_quiroga_sensing_stochastic_frames(data, rows=rows, **common)
    np.testing.assert_array_equal(new.checkpoint_steps, old.checkpoint_steps)
    np.testing.assert_allclose(new.final_x, old.final_x, rtol=1e-10, atol=1e-12)
    for name, legacy in (("measurement_loss", "measurement_loss"), ("tp_violation", "tp_violation"),
                         ("smoothed_objective", "smoothed_objective"), ("exact_smoothed_gap", "exact_smoothed_gap"),
                         ("process_fidelity", "process_fidelity_proxy"), ("estimated_gaps", "estimated_gaps"),
                         ("step_sizes", "step_sizes"), ("smoothing_parameters", "smoothing_parameters")):
        np.testing.assert_allclose(getattr(new, name), getattr(old, legacy), rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(new.cumulative_sampled_measurements, old.cumulative_sampled_measurements)
    assert new.metadata["active_rows"] == (data.m if rows is None else rows.size)


@pytest.mark.parametrize("subset", [False, True])
def test_dense_bridge_feeds_the_existing_jax_runner(subset):
    pytest.importorskip("jax")
    from paper.experiments.quantum_process_tomography_jax import run_qpt_stochastic_frames_jax

    data = _noisy(2, channel_seed=7)
    rows = data.fixed_row_subset(96, seed=1) if subset else None
    common = dict(
        n_steps=12, rank=1, tau=3.0, batch_size=4, initialization_seed=2, sampling_seed=3,
        metrics_frequency=4, show_progress=False,
        rho_schedule=PowerSchedule(2.0, 4.0, 0.6, cap=1.0),
        smoothing_schedule=PowerSchedule(10.0, 1.0, 0.25),
        step_size_schedule=PowerSchedule(2.0, 2.0, 1.0, cap=1.0),
    )
    old = run_qpt_stochastic_frames_jax(data.to_dense_qpt_data(rows), device="cpu", precision="64",
                                        execution_mode="scan", warmup=False, **common)
    new = run_quiroga_sensing_stochastic_frames(data, rows=rows, **common)
    active = data.m if rows is None else rows.size
    expected_batches = np.random.default_rng(3).integers(0, active, size=(12, 4))
    np.testing.assert_array_equal(old.batch_indices, expected_batches)
    np.testing.assert_array_equal(new.algorithm.objective.last_batch_indices, expected_batches[-1])
    np.testing.assert_allclose(new.final_x, old.final_x, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(new.algorithm.gradient_estimate, old.final_gradient_estimate, atol=1e-11)
    for name, legacy in (("measurement_loss", "measurement_loss"), ("tp_violation", "tp_violation"),
                         ("exact_smoothed_gap", "exact_smoothed_gap"), ("estimated_gaps", "estimated_gaps"),
                         ("process_fidelity", "process_fidelity_proxy")):
        np.testing.assert_allclose(getattr(new, name), getattr(old, legacy), rtol=1e-9, atol=1e-11)


def test_structured_frames_runs_where_dense_matrices_do_not_fit():
    data = generate_quiroga_sensing_data(5, channel_seed=1, observation_mode="gaussian",
                                         noise_std=0.01, noise_seed=2)
    with pytest.raises(ValueError, match="max_entries"):
        data.to_dense_qpt_data()
    result = run_quiroga_sensing_stochastic_frames(
        data, n_steps=6, tau=np.sqrt(data.d), batch_size=32, metrics_frequency=3, show_progress=False)
    assert result.final_factor.shape == (data.process_dimension, 1)
    for trace in (result.measurement_loss, result.tp_violation, result.exact_smoothed_gap):
        assert trace.shape == (3,) and np.all(np.isfinite(trace))
    assert np.all((result.process_fidelity >= 0) & (result.process_fidelity <= 1))
    assert result.metadata["design"]["rows"] == 2 * 32**3


def test_exact_adafgd_step_matches_the_dense_published_update():
    data = _noisy(2, channel_seed=3)
    rows = data.fixed_row_subset(96, seed=0)  # Quiroga's underdetermined n=2 size.
    targets = data.observations_for_rows(rows)
    factor = unpack_factor(make_factor_initial_point(data, rank=1, seed=0), data.process_dimension, 1)
    updated, info = quiroga_adafgd_step(data, factor, eta_scale=0.3, tp_weight=0.7, rows=rows)

    dense = data.design.dense_sensing_matrices(rows)
    predicted = np.einsum("sij,ij->s", dense.conj(), factor @ factor.conj().T).real
    residual = predicted - targets
    matrix_gradient = np.einsum("s,sij->ij", residual, dense)
    eta = 0.3 * np.linalg.norm(matrix_gradient, 2) / np.linalg.norm(predicted)
    partial = np.trace((factor @ factor.conj().T).reshape(4, 4, 4, 4), axis1=1, axis2=3) - np.eye(4)
    tp_term = 2 * np.kron(partial, np.eye(4)) @ factor
    np.testing.assert_allclose(updated, factor - eta * (matrix_gradient @ factor + 0.7 * tp_term), atol=1e-12)
    assert info["eta"] == pytest.approx(eta, rel=1e-12)
    assert info["measurement_loss_sum"] == pytest.approx(0.5 * np.sum(residual**2), rel=1e-12)
    assert info["tp_penalty_unhalved"] == pytest.approx(np.linalg.norm(partial) ** 2, rel=1e-12)
    # The published factor term is half the real-Frobenius gradient of the summed loss.
    _, summed_gradient = data.design.loss_and_gradient(factor, rows, targets, reduction="sum")
    np.testing.assert_allclose(matrix_gradient @ factor, 0.5 * summed_gradient, atol=1e-12)
    again, _ = quiroga_adafgd_step(data, factor, eta_scale=0.3, tp_weight=0.7, rows=rows, observations=targets)
    np.testing.assert_array_equal(again, updated)


def test_exact_adafgd_refuses_the_dense_spectral_norm_at_scale():
    data = generate_quiroga_sensing_data(6)
    with pytest.raises(ValueError, match="dense 4096x4096"):
        quiroga_adafgd_step(data, data.truth_factor, eta_scale=1.0, tp_weight=1.0)
