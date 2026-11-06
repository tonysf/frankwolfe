"""Fixed virtual noise, materialized equivalence, and bounded generation."""

import json
import os
import numpy as np
import pytest

from paper.experiments.qpt_generate_data import generate_on_demand_data
from paper.experiments.qpt_observation_noise import fixed_row_noise
from paper.experiments.qpt_structured_data import StructuredQPTData


def test_random_access_noise_matches_independent_scalar_reference():
    mask = (1 << 64) - 1
    def mix(x):
        x = (x + 0x9E3779B97F4A7C15) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        return x ^ (x >> 31)
    rows = np.array([0, 5, 5, 2**32 + 5, 24**10 - 1], dtype=np.int64)
    for seed in (0, 123, 2**64 - 1):
        key = mix(seed ^ 0xD2B74407B1CE6E93)
        expected = []
        for row in rows:
            u = ((mix(int(row) ^ key) >> 11) + 1) * 2.0**-53
            v = (mix(int(row) ^ mix(key)) >> 11) * 2.0**-53
            expected.append(0.05 * np.sqrt(-2*np.log(u)) * np.cos(2*np.pi*v))
        np.testing.assert_array_equal(fixed_row_noise(rows, seed, 0.05), expected)
        np.testing.assert_array_equal(fixed_row_noise(rows[::-1], seed, 0.05), np.asarray(expected)[::-1])
        assert expected[1] == expected[2] and expected[1] != expected[3]


def test_noise_moments_and_batch_partition_are_stable():
    rows = np.arange(100_000, dtype=np.int64)
    z = fixed_row_noise(rows, 17, 1.0)
    np.testing.assert_array_equal(z, np.concatenate([fixed_row_noise(r, 17, 1.0) for r in np.array_split(rows, 11)]))
    assert abs(z.mean()) < 0.02
    assert abs(z.var() - 1) < 0.03
    assert abs(np.corrcoef(z[:-1], z[1:])[0, 1]) < 0.02


def test_virtual_archive_roundtrip_and_fixed_targets(tmp_path):
    data = generate_on_demand_data(2, channel_seed=3, noise_seed=7)
    assert data.observations is None
    symbols = data.indices_to_symbols(np.array([0, 14, 14, data.m-1]))
    target = data.observations_for_symbols(symbols)
    assert target[1] == target[2]
    path = tmp_path / "virtual.npz"
    data.save_npz(path)
    with np.load(path) as archive:
        assert "observations" not in archive.files
    loaded = StructuredQPTData.load_npz(path)
    np.testing.assert_array_equal(loaded.observations_for_symbols(symbols), target)
    assert loaded.metadata == data.metadata
    assert data.metadata["truth_tp_violation"] < 1e-12


def test_truth_memory_guard_rejects_before_allocating():
    with pytest.raises(ValueError, match="Truth generation estimate"):
        generate_on_demand_data(10, max_memory_bytes=1024)


@pytest.mark.parametrize("backend", ["tensor", "rank-one"])
def test_virtual_and_materialized_optimizer_trajectories_match(backend):
    pytest.importorskip("jax")
    from paper.experiments.quantum_process_tomography_structured_jax import run_qpt_structured_jax
    data = generate_on_demand_data(2, channel_seed=3, noise_seed=11)
    symbols = data.indices_to_symbols(np.arange(data.m))
    stored = StructuredQPTData(2, data.local_measurements, data.local_basis,
                               observations=data.observations_for_symbols(symbols), truth_factor=data.truth_factor)
    options = dict(n_steps=17, tau=3., rank=1, batch_size=5, chunk_steps=4,
                   metrics_frequency=4, metric_mode="full", metric_batch_size=32,
                   rho_schedule=lambda k: .7/(k+1)**.3, smoothing_schedule=lambda k: 20/(k+1)**.25,
                   step_size_schedule=lambda k: .2/(k+1)**.6, initialization_seed=3, sampling_seed=17,
                   device=os.environ.get("QPT_TEST_DEVICE", "cpu"), precision="64",
                   store_checkpoints=True, measurement_backend=backend)
    actual = run_qpt_structured_jax(data, **options)
    expected = run_qpt_structured_jax(stored, **options)
    for field in ("checkpoint_factors", "gradient_estimate", "measurement_loss", "tp_violation",
                  "smoothed_gap", "process_fidelity_proxy"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=2e-10, atol=2e-11)
    options.update(chunk_steps=7, metric_mode="sampled", metric_samples=13, metrics_frequency=5)
    other = run_qpt_structured_jax(data, **options)
    np.testing.assert_allclose(other.final_factor, actual.final_factor, rtol=2e-10, atol=2e-11)
    assert other.metadata["batch_symbols_sha256"] == actual.metadata["batch_symbols_sha256"]
    assert actual.metadata["observation_mode"] == "synthetic-noisy"
    json.dumps(actual.metadata, allow_nan=False)


@pytest.mark.parametrize("backend,prefetch", [("auto", False), ("product-state", True)])
def test_benchmark_explicitly_admits_virtual_data(tmp_path, backend, prefetch):
    pytest.importorskip("jax")
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data_path, report_path = tmp_path / "data.npz", tmp_path / "report.json"
    generate_on_demand_data(1).save_npz(data_path)
    command = [sys.executable, str(root / "scripts/benchmark_qpt_structured.py"),
        "--data", str(data_path), "--allow-synthetic-noisy", "--device", "cpu",
        "--steps", "3", "--repeats", "1", "--tau", "3", "--metric-samples", "7",
        "--measurement-backend", backend, "--save", str(report_path)]
    if prefetch:
        command.append("--prefetch")
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "complete"
    assert report["observation_mode"] == "synthetic-noisy"
    assert report["memory_estimates"]["structured_host_observation_bytes"] == 0
    assert report["observations_fingerprint"] is None
    assert report["observations_preserved"] is False
    assert Path(report["runs"][0]["structured_result_path"]).is_file()
