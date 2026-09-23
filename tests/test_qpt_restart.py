"""Restart must retain momentum, sample RNG, and absolute schedule position."""
import os
import numpy as np
import pytest

pytest.importorskip("jax")
from paper.experiments.qpt_generate_data import generate_on_demand_data
from paper.experiments.quantum_process_tomography_structured_jax import (
    run_qpt_structured_jax, load_structured_restart,
)
from paper.experiments.qpt_structured_operators import pack_factor


def options():
    return dict(rank=1, tau=3., batch_size=7, chunk_steps=3, metrics_frequency=4,
                metric_samples=11, metric_batch_size=5, precision="64",
                device=os.environ.get("QPT_TEST_DEVICE", "cpu"),
                measurement_backend="product-state", show_progress=False,
                rho_schedule=lambda k: .7/(k+1)**.3,
                smoothing_schedule=lambda k: 20/(k+1)**.25,
                step_size_schedule=lambda k: .2/(k+1)**.6)


def test_restart_matches_uninterrupted_trajectory_with_changed_chunk_and_metrics(tmp_path):
    data = generate_on_demand_data(2, channel_seed=3, noise_seed=5)
    cfg = options()
    whole = run_qpt_structured_jax(data, n_steps=19, **cfg)
    path = tmp_path / "restart.npz"
    first = run_qpt_structured_jax(data, n_steps=7, restart_path=path, prefetch=True, **cfg)
    saved = load_structured_restart(path)
    assert saved["metadata"]["completed_steps"] == 7
    np.testing.assert_array_equal(saved["gradient_estimate"], first.gradient_estimate)
    cfg.update(chunk_steps=5, metrics_frequency=6, metric_samples=13)
    second = run_qpt_structured_jax(data, n_steps=19, resume=path, prefetch=True,
                                    restart_path=tmp_path/"continued.npz", **cfg)
    np.testing.assert_allclose(second.final_factor, whole.final_factor, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(second.gradient_estimate, whole.gradient_estimate, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(np.r_[first.estimated_gaps, second.estimated_gaps], whole.estimated_gaps,
                               rtol=2e-11, atol=2e-12)
    assert second.checkpoint_steps[0] == 7 and second.checkpoint_steps[-1] == 19
    assert second.metadata["start_step"] == 7 and second.metadata["segment_steps"] == 12
    assert second.metadata["sampled_measurements"] == 12*7


def test_prefetch_retains_sampling_plan_and_all_checkpoint_values():
    data = generate_on_demand_data(2, noise_seed=6)
    baseline = run_qpt_structured_jax(data, n_steps=17, **options())
    prefetched = run_qpt_structured_jax(data, n_steps=17, prefetch=True, **options())
    assert prefetched.metadata["batch_symbols_sha256"] == baseline.metadata["batch_symbols_sha256"]
    for field in ("final_factor", "gradient_estimate", "estimated_gaps", "process_fidelity_proxy", "tp_violation"):
        np.testing.assert_array_equal(getattr(prefetched, field), getattr(baseline, field))


def test_restart_rejects_changed_noise_batch_or_schedule(tmp_path):
    data = generate_on_demand_data(1, noise_seed=7)
    path = tmp_path/"restart.npz"
    run_qpt_structured_jax(data, n_steps=3, restart_path=path, **options())
    for overrides in (dict(batch_size=8), dict(smoothing_schedule=lambda k: 3.)):
        with pytest.raises(ValueError, match="does not match"):
            run_qpt_structured_jax(data, n_steps=5, resume=path, **(options() | overrides))
    with pytest.raises(ValueError, match="does not match"):
        run_qpt_structured_jax(generate_on_demand_data(1, noise_seed=8), n_steps=5, resume=path, **options())


def test_fidelity_stopping_saves_zero_step_state_without_optimizing(tmp_path):
    data = generate_on_demand_data(1)
    path = tmp_path/"restart.npz"
    result = run_qpt_structured_jax(data, n_steps=100, x0=pack_factor(data.truth_factor),
                                     fidelity_target=.99, restart_path=path, **options())
    assert result.metadata["fidelity_target_reached"] and result.metadata["n_steps"] == 0
    assert result.estimated_gaps.size == 0
    np.testing.assert_array_equal(result.final_factor, data.truth_factor)
    assert load_structured_restart(path)["metadata"]["completed_steps"] == 0
