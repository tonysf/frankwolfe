"""Independent physical-model and persistence checks for compact QPT data."""

import itertools

import numpy as np
import pytest

from paper.experiments.qpt_structured_data import (
    ROW_ORDER,
    StructuredQPTData,
    main,
    standard_local_basis,
    standard_local_measurements,
)


def kron_all(matrices):
    result = np.ones((1, 1), dtype=complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


@pytest.fixture(scope="module")
def physical_two_qubit_arrays():
    """Reproduce the upstream trace formula and label loops independently."""
    basis_local = np.array(
        [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1], [1, 0]], [[1, 0], [0, -1]]],
        dtype=complex,
    ) / np.sqrt(2)
    basis = np.stack([np.kron(left, right) for left in basis_local for right in basis_local])
    inputs = np.array([[1, 0], [0, 1], [1, 1], [1, 1j]], dtype=complex)
    inputs[2:] /= np.sqrt(2)
    outputs = np.array(
        [[[1, 1], [1, -1]], [[1, 1j], [1, -1j]], [[np.sqrt(2), 0], [0, np.sqrt(2)]]],
        dtype=complex,
    ) / np.sqrt(2)
    measurements = []
    # Global rho and axis labels are reversed by Qiskit's qubit convention;
    # projection outcomes remain ordered left-to-right in tensor products.
    for input_label in itertools.product(range(4), repeat=2):
        vector = kron_all([inputs[index][None, :] for index in reversed(input_label)]).ravel()
        rho = np.outer(vector, vector.conj())
        for axis_label in itertools.product(range(3), repeat=2):
            for outcomes in itertools.product(range(2), repeat=2):
                vector = kron_all([
                    outputs[axis, outcome][None, :]
                    for axis, outcome in zip(reversed(axis_label), outcomes)
                ]).ravel()
                projector = np.outer(vector, vector.conj())
                # D[m,n] = Tr(rho P_n^H E P_m), without the compact bank.
                measurements.append(np.einsum(
                    "ab,ncb,cd,mda->mn", rho, basis.conj(), projector, basis,
                    optimize=True,
                ))
    measurements = np.asarray(measurements)
    # Noise deliberately differs from any possible generated truth values.
    observations = np.random.default_rng(52).normal(0.5, 0.2, size=576).astype(np.float32)
    return observations, measurements, basis


@pytest.fixture
def physical_hdf5(tmp_path, physical_two_qubit_arrays):
    h5py = pytest.importorskip("h5py")
    observations, measurements, basis = physical_two_qubit_arrays
    path = tmp_path / "physical_qpt.h5"
    with h5py.File(path, "w") as handle:
        handle["f_jax_vector"] = observations
        handle["D_jax_tensors"] = measurements
        handle["A_jax_basis"] = basis
        handle["B_jax_tensors"] = np.einsum("mki,nkj->nmij", basis.conj(), basis)
        # Deliberately unusable sentinel: conversion must never inspect it.
        handle["Chi_star_tensor"] = np.array([np.nan])
    return path


def make_stored(observations):
    return StructuredQPTData(2, standard_local_measurements(), standard_local_basis(), observations)


def test_converter_cli_cannot_overwrite_source_hdf5(physical_hdf5):
    from paper.experiments.qpt_structured_data import main

    before = physical_hdf5.read_bytes()
    with pytest.raises(SystemExit):
        main(["--h5", str(physical_hdf5), "--save", str(physical_hdf5)])
    assert physical_hdf5.read_bytes() == before


def test_all_576_legacy_rows_match_independent_physical_formula(physical_two_qubit_arrays):
    observations, measurements, _ = physical_two_qubit_arrays
    data = make_stored(observations)
    rows = np.arange(data.m)
    symbols = data.indices_to_symbols(rows)
    np.testing.assert_array_equal(data.symbols_to_indices(symbols), rows)
    actual = data.apply_measurements(np.eye(16), symbols)
    np.testing.assert_allclose(actual, measurements, atol=2e-16)
    # Explicit mixed ordering regression: R=1, I=2, J=1 -> (input1,axis2,out0),(input0,axis0,out1).
    np.testing.assert_array_equal(data.indices_to_symbols(np.array([45])), [[10, 1]])


def test_basis_is_orthonormal_with_upstream_real_antisymmetric_y():
    basis = standard_local_basis()
    np.testing.assert_array_equal(basis[2].real, np.array([[0, -1], [1, 0]]) / np.sqrt(2))
    np.testing.assert_array_equal(basis[2].imag, np.zeros((2, 2)))
    np.testing.assert_allclose(np.einsum("kij,lij->kl", basis.conj(), basis), np.eye(4), atol=3e-16)
    bank = standard_local_measurements()
    np.testing.assert_allclose(bank, bank.conj().swapaxes(-1, -2), atol=0)
    np.testing.assert_allclose(np.linalg.eigvalsh(bank)[:, -1], 1, atol=1e-15)


def test_sample_symbols_exactly_preserve_legacy_rng_sequence(physical_two_qubit_arrays):
    data = make_stored(physical_two_qubit_arrays[0])
    expected_rng = np.random.default_rng(100)
    actual_rng = np.random.default_rng(100)
    for batch_size in [1, 17, 32]:
        expected = expected_rng.integers(0, data.m, size=batch_size)
        actual = data.symbols_to_indices(data.sample_symbols(actual_rng, batch_size))
        np.testing.assert_array_equal(actual, expected)


def test_noiseless_mode_evaluates_low_rank_truth_without_observation_table(physical_two_qubit_arrays):
    rng = np.random.default_rng(5)
    truth = rng.normal(size=(16, 2)) + 1j * rng.normal(size=(16, 2))
    data = StructuredQPTData(
        2, standard_local_measurements(), standard_local_basis(),
        truth_factor=truth, observation_mode="noiseless",
    )
    rows = np.array([0, 2, 34, 109, 575, 2])
    expected = np.einsum(
        "sij,ij->s", physical_two_qubit_arrays[1][rows].conj(), truth @ truth.conj().T,
    ).real
    np.testing.assert_allclose(data.observations_for_symbols(data.indices_to_symbols(rows)), expected, atol=1e-14)
    assert data.observations is None


def test_hdf5_converter_streams_bounded_chunks_and_does_not_read_chi(
    physical_hdf5, physical_two_qubit_arrays, monkeypatch,
):
    import h5py

    original_getitem = h5py.Dataset.__getitem__
    read_rows = {"/D_jax_tensors": 0, "/A_jax_basis": 0, "/B_jax_tensors": 0}

    def guarded_getitem(dataset, key, *args, **kwargs):
        assert dataset.name != "/Chi_star_tensor"
        if dataset.name in read_rows:
            if dataset.name == "/B_jax_tensors":
                assert isinstance(key, tuple) and len(key) == 2
                assert isinstance(key[0], int)
                selected_slice = key[1]
            else:
                selected_slice = key
            assert isinstance(selected_slice, slice)
            start, stop, step = selected_slice.indices(dataset.shape[0])
            assert step == 1
            assert stop - start <= 7
            read_rows[dataset.name] += stop - start
        return original_getitem(dataset, key, *args, **kwargs)

    def forbidden_array(dataset, *args, **kwargs):
        raise AssertionError(f"Whole-dataset materialization: {dataset.name}")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", forbidden_array)
    data = StructuredQPTData.from_hdf5(physical_hdf5, chunk_size=7)
    assert read_rows == {"/D_jax_tensors": 576, "/A_jax_basis": 16, "/B_jax_tensors": 256}
    assert data.observations.dtype == np.float32
    np.testing.assert_array_equal(data.observations, physical_two_qubit_arrays[0])
    assert data.truth_factor is None
    assert data.metadata["verification"] == "full"
    assert data.metadata["observations_preserved"] is True
    assert data.metadata["verified_penalty_blocks"] == 256
    assert data.metadata["source_fingerprint"]["size_bytes"] == physical_hdf5.stat().st_size


@pytest.mark.parametrize("dataset,row", [("D_jax_tensors", 313), ("A_jax_basis", 9)])
def test_converter_rejects_incompatible_rows(physical_hdf5, dataset, row):
    import h5py

    with h5py.File(physical_hdf5, "r+") as handle:
        matrix = handle[dataset][row]
        matrix[0, 0] += 0.1
        handle[dataset][row] = matrix
    with pytest.raises(ValueError, match=f"{dataset} row {row}"):
        StructuredQPTData.from_hdf5(physical_hdf5)


def test_sampled_conversion_records_incomplete_verification(physical_hdf5):
    data = StructuredQPTData.from_hdf5(
        physical_hdf5, verification="sampled", verification_samples=9, chunk_size=4,
    )
    assert data.metadata["verification"] == "sampled"
    assert data.metadata["verified_measurement_rows"] == 9
    assert data.metadata["verified_basis_rows"] == 16
    assert data.metadata["verified_penalty_blocks"] == 256


def test_custom_penalty_is_rejected_instead_of_silently_replaced(physical_hdf5):
    import h5py

    with h5py.File(physical_hdf5, "r+") as handle:
        handle["B_jax_tensors"][7, 3, 0, 0] += 0.01
    with pytest.raises(ValueError, match=r"B_jax_tensors block \(7, 3\)"):
        StructuredQPTData.from_hdf5(physical_hdf5, verification="sampled", verification_samples=2)


def test_complex64_archive_requires_explicit_cli_tolerances_and_preserves_targets(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "single_precision.h5"
    destination = tmp_path / "compact.npz"
    observations = np.random.default_rng(81).normal(size=24).astype(np.float32)
    basis = standard_local_basis()
    with h5py.File(source, "w") as handle:
        handle["f_jax_vector"] = observations
        handle["D_jax_tensors"] = standard_local_measurements().astype(np.complex64)
        handle["A_jax_basis"] = basis.astype(np.complex64)
        handle["B_jax_tensors"] = np.einsum("mki,nkj->nmij", basis.conj(), basis).astype(np.complex64)

    # Odd-qubit bases contain irrational normalization coefficients; rounding
    # them to complex64 exceeds the deliberately strict default tolerances.
    with pytest.raises(ValueError, match="A_jax_basis row 0"):
        StructuredQPTData.from_hdf5(source)
    main([
        "--h5", str(source), "--save", str(destination),
        "--rtol", "1e-6", "--atol", "1e-7",
    ])
    data = StructuredQPTData.load_npz(destination)
    assert data.metadata["verification_rtol"] == 1e-6
    assert data.metadata["verification_atol"] == 1e-7
    assert data.observations.dtype == observations.dtype
    assert data.observations.tobytes() == observations.tobytes()


@pytest.mark.parametrize("mode", ["stored", "noiseless"])
def test_npz_roundtrip_has_no_pickled_objects(tmp_path, physical_two_qubit_arrays, mode):
    if mode == "stored":
        data = make_stored(physical_two_qubit_arrays[0])
    else:
        data = StructuredQPTData(
            2, standard_local_measurements(), standard_local_basis(),
            truth_factor=np.arange(16), observation_mode="noiseless",
        )
    data.metadata = {"test": "roundtrip", "verification": "full"}
    path = tmp_path / "nested" / "compact.npz"
    data.save_npz(path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        assert archive["row_order"].item() == ROW_ORDER
    loaded = StructuredQPTData.load_npz(path)
    assert loaded.observation_mode == mode
    assert loaded.metadata == data.metadata
    symbols = data.indices_to_symbols(np.array([0, 1, 575]))
    np.testing.assert_array_equal(loaded.observations_for_symbols(symbols), data.observations_for_symbols(symbols))


def test_invalid_input_and_noiseless_modes_are_rejected(physical_two_qubit_arrays):
    bank, basis = standard_local_measurements(), standard_local_basis()
    with pytest.raises(ValueError, match="requires observations"):
        StructuredQPTData(2, bank, basis)
    with pytest.raises(ValueError, match="requires truth_factor"):
        StructuredQPTData(2, bank, basis, observation_mode="noiseless")
    with pytest.raises(ValueError, match="must not include"):
        StructuredQPTData(2, bank, basis, np.zeros(576), np.zeros((16, 1)), "noiseless")
    with pytest.raises(ValueError, match="orthonormal"):
        StructuredQPTData(2, bank, basis * 2, np.zeros(576))
    bank[0, 0, 1] += 1j
    with pytest.raises(ValueError, match="Hermitian"):
        StructuredQPTData(2, bank, basis, np.zeros(576))
    data = make_stored(physical_two_qubit_arrays[0])
    with pytest.raises(IndexError):
        data.indices_to_symbols(np.array([576]))
    with pytest.raises(ValueError):
        data.indices_to_symbols(np.array([1.5]))
    with pytest.raises(IndexError):
        data.symbols_to_indices(np.array([[24, 0]]))
    with pytest.raises(TypeError):
        data.symbols_to_indices(np.array([[1.2, 0]]))
    with pytest.raises(ValueError):
        data.sample_symbols(np.random.default_rng(0), 0)
