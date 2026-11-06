"""Independent small-n checks of directly generated compact noisy QPT data."""

import hashlib
import itertools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from paper.experiments.qpt_generate_data import (
    generate_compact_data, generation_allocation_estimate,
)
from paper.experiments.qpt_structured_data import StructuredQPTData


ROOT = Path(__file__).resolve().parents[1]


def kron_all(matrices):
    result = np.ones((1, 1), dtype=np.complex128)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def independent_haar(dimension, seed):
    rng = domain_rng(seed, 0)
    gaussian = (rng.standard_normal((dimension, dimension))
                + 1j * rng.standard_normal((dimension, dimension))) / np.sqrt(2)
    unitary, triangular = np.linalg.qr(gaussian)
    diagonal = np.diag(triangular)
    return unitary * (diagonal / np.abs(diagonal))[None, :]


def domain_rng(seed, domain):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(domain,))))


def independent_physical_arrays(n_qubits):
    """Legacy trace formula and original label loops, without compact helpers."""
    local_basis = np.array([
        [[1, 0], [0, 1]], [[0, 1], [1, 0]],
        [[0, -1], [1, 0]], [[1, 0], [0, -1]],
    ], dtype=np.complex128) / np.sqrt(2)
    basis = np.array([
        kron_all(matrices)
        for matrices in itertools.product(local_basis, repeat=n_qubits)
    ])
    inputs = np.array([[1, 0], [0, 1], [1, 1], [1, 1j]], dtype=complex)
    inputs[2:] /= np.sqrt(2)
    outputs = np.array([
        [[1, 1], [1, -1]], [[1, 1j], [1, -1j]],
        [[np.sqrt(2), 0], [0, np.sqrt(2)]],
    ], dtype=complex) / np.sqrt(2)
    rows = []
    for input_label in itertools.product(range(4), repeat=n_qubits):
        state = kron_all([inputs[index][None, :] for index in reversed(input_label)]).ravel()
        rho = np.outer(state, state.conj())
        for axis_label in itertools.product(range(3), repeat=n_qubits):
            for outcomes in itertools.product(range(2), repeat=n_qubits):
                state = kron_all([
                    outputs[axis, outcome][None, :]
                    for axis, outcome in zip(reversed(axis_label), outcomes)
                ]).ravel()
                projector = np.outer(state, state.conj())
                rows.append(np.einsum(
                    "ab,ncb,cd,mda->mn", rho, basis.conj(), projector, basis,
                    optimize=True,
                ))
    return basis, np.asarray(rows)


@pytest.mark.parametrize("n_qubits", [1, 2])
def test_all_generated_rows_match_independent_legacy_trace_and_gaussian_noise(n_qubits):
    basis, rows = independent_physical_arrays(n_qubits)
    unitary = independent_haar(2**n_qubits, 13)
    coefficients = np.array([np.trace(matrix.conj().T @ unitary) for matrix in basis])
    chi = np.outer(coefficients, coefficients.conj())
    noise = domain_rng(72, 1).standard_normal(24**n_qubits)
    expected = np.array([np.vdot(row, chi).real for row in rows]) + .05 * noise
    generated = generate_compact_data(
        n_qubits, channel_seed=13, noise_seed=72, noise_std=.05, batch_size=7,
    )
    np.testing.assert_allclose(generated.truth_factor[:, 0], coefficients, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(generated.observations, expected, rtol=1e-13, atol=1e-14)
    assert generated.observation_mode == "stored"
    assert generated.observations.dtype == np.float64
    assert generated.truth_factor.dtype == np.complex128


@pytest.mark.parametrize("n_qubits", [1, 2])
def test_unitary_truth_has_correct_basis_normalization_and_trace_preservation(n_qubits):
    generated = generate_compact_data(n_qubits, channel_seed=4, batch_size=13)
    basis, _ = independent_physical_arrays(n_qubits)
    kraus = np.einsum("k,kij->ij", generated.truth_factor[:, 0], basis)
    dimension = 2**n_qubits
    np.testing.assert_allclose(kraus.conj().T @ kraus, np.eye(dimension), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(generated.truth_factor), np.sqrt(dimension), rtol=1e-14)
    assert generated.truth_factor.shape == (4**n_qubits, 1)


def test_reproducible_values_are_independent_of_generation_batch_size():
    outputs = [generate_compact_data(2, channel_seed=9, noise_seed=18, batch_size=batch)
               for batch in (1, 7, 1024)]
    for generated in outputs[1:]:
        np.testing.assert_array_equal(generated.truth_factor, outputs[0].truth_factor)
        np.testing.assert_array_equal(generated.observations, outputs[0].observations)
        assert generated.metadata["noise_standard_normals_sha256"] == outputs[0].metadata["noise_standard_normals_sha256"]


def test_seed_streams_are_independent_and_noise_is_unclipped_additive_gaussian():
    clean = generate_compact_data(2, noise_std=0, channel_seed=6, noise_seed=71)
    noisy = generate_compact_data(2, noise_std=10, channel_seed=6, noise_seed=71)
    other_noise = generate_compact_data(2, noise_std=10, channel_seed=6, noise_seed=72)
    other_channel = generate_compact_data(2, noise_std=10, channel_seed=7, noise_seed=71)
    np.testing.assert_array_equal(noisy.truth_factor, clean.truth_factor)
    np.testing.assert_array_equal(noisy.truth_factor, other_noise.truth_factor)
    assert not np.array_equal(noisy.observations, other_noise.observations)
    assert not np.array_equal(noisy.truth_factor, other_channel.truth_factor)
    np.testing.assert_allclose(noisy.observations - clean.observations,
                               10 * domain_rng(71, 1).standard_normal(576),
                               rtol=1e-14, atol=1e-14)
    assert np.any(noisy.observations < 0)
    assert np.any(noisy.observations > 1)


def test_equal_user_seeds_still_domain_separate_channel_and_noise_draws():
    generated = generate_compact_data(2, channel_seed=0, noise_seed=0)
    channel_prefix = domain_rng(0, 0).standard_normal(16)
    noise = domain_rng(0, 1).standard_normal(576)
    assert not np.array_equal(channel_prefix, noise[:16])
    assert generated.metadata["noise_standard_normals_sha256"] == hashlib.sha256(noise.tobytes()).hexdigest()


def test_generated_metadata_describes_new_data_and_fingerprints_actual_arrays():
    generated = generate_compact_data(2, channel_seed=5, noise_seed=18, batch_size=7)
    metadata = generated.metadata
    assert metadata["source_kind"] == "synthetic_compact"
    assert metadata["verification"] == "constructed"
    assert metadata["legacy_observations_reused"] is False
    assert metadata["noise_clipped"] is False
    assert metadata["shot_noise"] is False
    assert metadata["channel_model"] == "haar_unitary"
    assert metadata["noise_model"] == "iid_additive_gaussian"
    assert metadata["channel_seed"] == 5
    assert metadata["noise_seed"] == 18
    assert metadata["generation_batch_size"] == 7
    assert metadata["generated_observations"] == 576
    assert metadata["resource_estimate_bytes"] == generation_allocation_estimate(2, 7)
    for name in ("observations", "truth_factor"):
        assert metadata[name + "_sha256"] == hashlib.sha256(
            np.ascontiguousarray(getattr(generated, name)).tobytes(),
        ).hexdigest()
    for name, digest in metadata["numerical_source_sha256"].items():
        assert digest == hashlib.sha256((ROOT / "paper" / "experiments" / name).read_bytes()).hexdigest()


def test_generation_sensing_batches_are_bounded_and_never_build_dense_kronecker_or_chi(monkeypatch):
    from paper.experiments import qpt_generate_data as generator
    calls = []
    original = generator.product_measurement_vectors

    def tracked(symbols, local_vectors, **kwargs):
        assert symbols.shape[0] <= 7
        calls.append(symbols.shape[0])
        result = original(symbols, local_vectors, **kwargs)
        assert result.shape == (symbols.shape[0], 16)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("dense tensor/process-matrix construction is forbidden")

    monkeypatch.setattr(generator, "product_measurement_vectors", tracked)
    monkeypatch.setattr(np, "kron", forbidden)
    monkeypatch.setattr(np, "outer", forbidden)
    generated = generate_compact_data(2, batch_size=7)
    assert sum(calls) == generated.m
    assert calls[-1] == 2


def test_compact_data_round_trip_has_no_dense_arrays_or_pickle(tmp_path):
    generated = generate_compact_data(2, batch_size=7)
    path = tmp_path / "direct.npz"
    generated.save_npz(path)
    loaded = StructuredQPTData.load_npz(path)
    np.testing.assert_array_equal(loaded.observations, generated.observations)
    np.testing.assert_array_equal(loaded.truth_factor, generated.truth_factor)
    assert loaded.metadata == generated.metadata
    with np.load(path, allow_pickle=False) as archive:
        assert not {"D_jax_tensors", "A_jax_basis", "B_jax_tensors", "Chi_star_tensor"} & set(archive.files)
        assert all(archive[key].dtype.kind != "O" for key in archive.files)


@pytest.mark.parametrize("kwargs", [
    {"n_qubits": 0}, {"n_qubits": -1}, {"n_qubits": 1.5}, {"n_qubits": True},
    {"batch_size": 0}, {"batch_size": -1}, {"batch_size": True},
    {"noise_std": -1}, {"noise_std": np.nan}, {"noise_std": np.inf},
    {"channel_seed": -1}, {"noise_seed": -1},
    {"channel_seed": True}, {"noise_seed": True},
    {"channel_seed": 2**64}, {"noise_seed": 2**64},
    {"noise_std": True}, {"noise_std": 1j},
    {"n_qubits": 10**100},
    {"max_observations": 0}, {"max_memory_bytes": 0},
])
def test_invalid_generation_parameters_are_rejected(kwargs):
    arguments = dict(n_qubits=1)
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        generate_compact_data(**arguments)


@pytest.mark.parametrize("kwargs", [
    {"n_qubits": 6},
    {"n_qubits": 2, "max_observations": 575},
    {"n_qubits": 2, "max_memory_bytes": 1},
])
def test_sizing_guards_run_before_random_or_array_allocation(monkeypatch, kwargs):
    from paper.experiments import qpt_generate_data as generator
    def forbidden(*args, **options):
        raise AssertionError("allocation or RNG started before rejecting oversized request")
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(generator, "_haar_unitary", forbidden)
    with pytest.raises(ValueError):
        generate_compact_data(**kwargs)


def test_default_generation_budget_admits_four_and_five_qubit_array_estimates():
    assert generation_allocation_estimate(4, 1024) < 512 * 2**20
    assert generation_allocation_estimate(5, 1024) < 512 * 2**20
    assert generation_allocation_estimate(6, 1024) > 512 * 2**20


def cli_environment():
    result = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        result[name] = "1"
    return result


def run_cli(arguments):
    return subprocess.run(
        [sys.executable, "-m", "paper.experiments.qpt_generate_data"] + arguments,
        cwd=ROOT, capture_output=True, text=True, env=cli_environment(), timeout=30,
    )


def test_generator_cli_saves_new_compact_data_and_refuses_overwrite(tmp_path):
    path = tmp_path / "generated.npz"
    arguments = ["--n-qubits", "1", "--channel-seed", "7", "--noise-seed", "19",
                 "--noise-std", "0.05", "--batch-size", "7", "--save", str(path)]
    completed = run_cli(arguments)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    original = path.read_bytes()
    generated = StructuredQPTData.load_npz(path)
    expected = generate_compact_data(1, channel_seed=7, noise_seed=19, noise_std=.05, batch_size=7)
    np.testing.assert_array_equal(generated.observations, expected.observations)
    repeated = run_cli(arguments)
    assert repeated.returncode != 0
    assert path.read_bytes() == original


def test_cli_guard_does_not_create_output(tmp_path):
    path = tmp_path / "oversized.npz"
    completed = run_cli(["--n-qubits", "6", "--save", str(path)])
    assert completed.returncode != 0
    assert not path.exists()


@pytest.mark.parametrize("broken", [False, True])
def test_cli_refuses_existing_symlink_without_touching_target(tmp_path, broken):
    target = tmp_path / "original.npz"
    if not broken:
        target.write_bytes(b"Original user data, not a valid archive")
    path = tmp_path / "link.npz"
    path.symlink_to(target)
    completed = run_cli(["--n-qubits", "1", "--save", str(path)])
    assert completed.returncode != 0
    assert path.is_symlink()
    if not broken:
        assert target.read_bytes() == b"Original user data, not a valid archive"
    else:
        assert not target.exists()


def test_generator_does_not_import_jax_or_h5py():
    completed = subprocess.run([
        sys.executable, "-c",
        "import sys; from paper.experiments.qpt_generate_data import generate_compact_data; "
        "generate_compact_data(1); "
        "assert not any(k.split('.')[0] in ('jax', 'jaxlib', 'h5py') for k in sys.modules)",
    ], cwd=ROOT, capture_output=True, text=True, env=cli_environment(), timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
