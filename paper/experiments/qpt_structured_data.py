"""Compact QPT_BFW operators and a bounded-memory HDF5 converter.

The legacy row order is *not* ordinary base-24 tensor-product order.  Its
input and measurement-axis labels run in the opposite direction to the
outcome label.  This module retains that order, and retains every stored
observation (including its original noise realization).

The upstream basis uses ``[I, X, -1j*Y, Z] / sqrt(2)``.  In particular, its
third element is real and anti-Hermitian; replacing it with the usual Pauli
Y changes the coordinates of the optimization problem.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional

import numpy as np


FORMAT_VERSION = 1
ROW_ORDER = "qpt_bfw_input_axis_reversed_outcome_forward"


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _numeric_array(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in "iufc":
        raise ValueError(f"{name} must contain numeric values.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result


def standard_local_basis():
    """Return the exact single-qubit orthonormal basis used by QPT_BFW."""
    return np.asarray(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1], [1, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)


def standard_local_measurements():
    """Return 24 Hermitian rows indexed by ``6*input + 2*axis + outcome``.

    Inputs are ``|0>, |1>, |+>, |+i>``; axes are X, Y, Z; outcomes are
    the positive and negative eigenstates of each physical Pauli axis.
    """
    root_two = np.sqrt(2.0)
    inputs = np.asarray([[1, 0], [0, 1], [1, 1], [1, 1j]], dtype=complex)
    inputs[2:] /= root_two
    eigenvectors = np.asarray(
        [
            [[1, 1], [1, -1]],
            [[1, 1j], [1, -1j]],
            [[root_two, 0], [0, root_two]],
        ],
        dtype=complex,
    ) / root_two
    amplitudes = np.einsum(
        "aji,kil,pl->pajk", eigenvectors.conj(), standard_local_basis(), inputs
    ).reshape(24, 4)
    return np.einsum("sk,sl->skl", amplitudes, amplitudes.conj())


# Descriptive aliases for callers constructing a new synthetic problem.
local_pauli_basis = standard_local_basis
local_qpt_measurements = standard_local_measurements


def _tensor_product(matrices):
    result = np.ones((1, 1), dtype=np.complex128)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


@dataclass
class StructuredQPTData:
    """Tensor-product QPT operators with stored or explicitly noiseless data.

    ``observations`` always follows the legacy HDF5 row order.  Noiseless
    mode evaluates the supplied low-rank truth factor on demand and never
    allocates an array of length ``24**n_qubits``.  It is a different data
    model from a noisy HDF5 experiment and is recorded explicitly on disk.
    """

    n_qubits: int
    local_measurements: np.ndarray
    local_basis: np.ndarray
    observations: Optional[np.ndarray] = None
    truth_factor: Optional[np.ndarray] = None
    observation_mode: str = "stored"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_qubits = _positive_integer(self.n_qubits, "n_qubits")
        bank = _numeric_array(self.local_measurements, "local_measurements")
        if bank.shape != (24, 4, 4):
            raise ValueError("local_measurements must have shape (24, 4, 4).")
        if not np.allclose(bank, bank.conj().swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
            raise ValueError("local_measurements must be Hermitian.")
        basis = _numeric_array(self.local_basis, "local_basis")
        if basis.shape != (4, 2, 2):
            raise ValueError("local_basis must have shape (4, 2, 2).")
        gram = np.einsum("kij,lij->kl", basis.conj(), basis)
        if not np.allclose(gram, np.eye(4), rtol=1e-10, atol=1e-12):
            raise ValueError("local_basis must be Hilbert-Schmidt orthonormal.")
        self.local_measurements = np.asarray(bank, dtype=np.complex128)
        self.local_basis = np.asarray(basis, dtype=np.complex128)
        if self.observation_mode not in ("stored", "noiseless"):
            raise ValueError("observation_mode must be 'stored' or 'noiseless'.")
        if self.observations is not None:
            observations = _numeric_array(self.observations, "observations")
            if observations.shape != (self.m,):
                raise ValueError(f"observations must have shape ({self.m},).")
            if np.iscomplexobj(observations):
                if np.any(observations.imag != 0):
                    raise ValueError("observations must be real-valued.")
                observations = observations.real
            # Preserve the original real dtype and every stored value.
            self.observations = observations
        if self.truth_factor is not None:
            truth = _numeric_array(self.truth_factor, "truth_factor")
            if truth.ndim == 1:
                truth = truth[:, None]
            if truth.ndim != 2 or truth.shape[0] != self.process_dimension or truth.shape[1] < 1:
                raise ValueError("truth_factor must have shape (4**n_qubits, rank).")
            self.truth_factor = np.asarray(truth, dtype=np.complex128)
        if self.observation_mode == "stored" and self.observations is None:
            raise ValueError("stored mode requires observations.")
        if self.observation_mode == "noiseless":
            if self.truth_factor is None:
                raise ValueError("noiseless mode requires truth_factor.")
            if self.observations is not None:
                raise ValueError("noiseless mode must not include stored observations.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON-compatible dictionary.")
        # Round-tripping also rejects arrays, arbitrary objects, and NaN metadata.
        self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))

    @property
    def m(self):
        return 24**self.n_qubits

    @property
    def d(self):
        return 2**self.n_qubits

    @property
    def process_dimension(self):
        return 4**self.n_qubits

    def _validate_symbols(self, symbols):
        symbols = np.asarray(symbols)
        if symbols.ndim != 2 or symbols.shape[1] != self.n_qubits:
            raise ValueError("symbols must have shape (batch_size, n_qubits).")
        if symbols.dtype.kind not in "iu":
            raise TypeError("symbols must be integers.")
        if np.any(symbols < 0) or np.any(symbols >= 24):
            raise IndexError("local measurement symbol is out of range.")
        return symbols.astype(np.int64, copy=False)

    def indices_to_symbols(self, indices):
        """Decode legacy flat rows into left-to-right tensor factors."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("indices must be a one-dimensional integer array.")
        if np.any(indices < 0) or np.any(indices >= self.m):
            raise IndexError("measurement index is out of range.")
        if self.m > np.iinfo(np.int64).max:
            raise ValueError("flat indices exceed int64 capacity; use local symbols.")
        indices = indices.astype(np.int64, copy=False)
        outcomes = indices % (2**self.n_qubits)
        axes = (indices // (2**self.n_qubits)) % (3**self.n_qubits)
        inputs = indices // (6**self.n_qubits)
        symbols = np.empty((indices.size, self.n_qubits), dtype=np.int64)
        for qubit in range(self.n_qubits):
            symbols[:, qubit] = (
                6 * ((inputs // (4**qubit)) % 4)
                + 2 * ((axes // (3**qubit)) % 3)
                + ((outcomes // (2 ** (self.n_qubits - qubit - 1))) % 2)
            )
        return symbols

    def symbols_to_indices(self, symbols):
        """Encode tensor-factor symbols in the original HDF5 row order."""
        symbols = self._validate_symbols(symbols)
        if self.m > np.iinfo(np.int64).max:
            raise ValueError("flat indices exceed int64 capacity; use local symbols.")
        inputs = np.zeros(symbols.shape[0], dtype=np.int64)
        axes = np.zeros_like(inputs)
        outcomes = np.zeros_like(inputs)
        for qubit in range(self.n_qubits):
            inputs += (symbols[:, qubit] // 6) * (4**qubit)
            axes += ((symbols[:, qubit] // 2) % 3) * (3**qubit)
            outcomes += (symbols[:, qubit] % 2) * (2 ** (self.n_qubits - qubit - 1))
        return (inputs * (3**self.n_qubits) + axes) * (2**self.n_qubits) + outcomes

    def sample_symbols(self, rng, batch_size):
        """Uniform sampling with replacement, matching the legacy RNG stream."""
        batch_size = _positive_integer(batch_size, "batch_size")
        if self.m <= np.iinfo(np.int64).max:
            return self.indices_to_symbols(rng.integers(0, self.m, size=batch_size))
        return rng.integers(0, 24, size=(batch_size, self.n_qubits), dtype=np.int64)

    def apply_measurements(self, factor, symbols):
        """NumPy reference for ``D_s @ factor`` without constructing any D_s."""
        symbols = self._validate_symbols(symbols)
        factor = _numeric_array(factor, "factor")
        if factor.ndim == 1:
            factor = factor[:, None]
        if factor.ndim != 2 or factor.shape[0] != self.process_dimension:
            raise ValueError("factor must have shape (4**n_qubits, rank).")
        result = np.empty((symbols.shape[0],) + factor.shape, dtype=np.complex128)
        shape = (4,) * self.n_qubits + (factor.shape[1],)
        for row, local_symbols in enumerate(symbols):
            tensor = factor.reshape(shape)
            for axis, symbol in enumerate(local_symbols):
                tensor = np.tensordot(self.local_measurements[symbol], tensor, axes=(1, axis))
                tensor = np.moveaxis(tensor, 0, axis)
            result[row] = tensor.reshape(factor.shape)
        return result

    def observations_for_symbols(self, symbols):
        """Fetch actual observations or evaluate an explicitly noiseless model."""
        symbols = self._validate_symbols(symbols)
        if self.observation_mode == "stored":
            return self.observations[self.symbols_to_indices(symbols)]
        applied = self.apply_measurements(self.truth_factor, symbols)
        return np.einsum("kr,bkr->b", self.truth_factor.conj(), applied).real

    def save_npz(self, path):
        """Save portable arrays and JSON metadata; no pickled Python objects."""
        payload = {
            "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
            "n_qubits": np.asarray(self.n_qubits, dtype=np.int64),
            "row_order": np.asarray(ROW_ORDER),
            "observation_mode": np.asarray(self.observation_mode),
            "local_measurements": self.local_measurements,
            "local_basis": self.local_basis,
            "metadata_json": np.asarray(json.dumps(self.metadata, allow_nan=False)),
        }
        if self.observations is not None:
            payload["observations"] = self.observations
        if self.truth_factor is not None:
            payload["truth_factor"] = self.truth_factor
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, **payload)

    @classmethod
    def load_npz(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            def scalar(name):
                array = archive[name]
                if array.shape != ():
                    raise ValueError(f"{name} must be a scalar.")
                return array.item()

            if scalar("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported structured QPT format version.")
            if scalar("row_order") != ROW_ORDER:
                raise ValueError("Unsupported structured QPT row order.")
            return cls(
                n_qubits=scalar("n_qubits"),
                local_measurements=archive["local_measurements"],
                local_basis=archive["local_basis"],
                observations=archive["observations"] if "observations" in archive else None,
                truth_factor=archive["truth_factor"] if "truth_factor" in archive else None,
                observation_mode=scalar("observation_mode"),
                metadata=json.loads(scalar("metadata_json")),
            )

    @classmethod
    def from_hdf5(
        cls, path, *, verification="full", chunk_size=32,
        verification_samples=256, rtol=1e-10, atol=1e-12,
    ):
        """Convert QPT_BFW input with bounded dense working storage.

        Full verification (the default) checks every D row and every basis
        element.  Explicit sampled verification checks deterministic rows
        and records that it is incomplete.  If ``B_jax_tensors`` is present,
        every block is checked using bounded slices.  ``Chi_star_tensor``
        is never read.  Observations are preserved,
        including their noise, rather than reconstructed from ground truth.
        """
        try:
            import h5py
        except ImportError as error:
            raise ImportError("HDF5 conversion requires h5py; install the 'qpt' extra.") from error
        chunk_size = _positive_integer(chunk_size, "chunk_size")
        verification_samples = _positive_integer(verification_samples, "verification_samples")
        if verification not in ("full", "sampled"):
            raise ValueError("verification must be 'full' or 'sampled'.")
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("verification tolerances must be finite and nonnegative.")
        local_basis = standard_local_basis()
        local_measurements = standard_local_measurements()
        with h5py.File(path, "r") as handle:
            required = ("f_jax_vector", "D_jax_tensors", "A_jax_basis")
            missing = [key for key in required if key not in handle]
            if missing:
                raise KeyError("Missing required QPT dataset(s): " + ", ".join(missing))
            basis = handle["A_jax_basis"]
            if len(basis.shape) != 3 or basis.shape[1] != basis.shape[2]:
                raise ValueError("A_jax_basis must have shape (4**n, 2**n, 2**n).")
            d = int(basis.shape[1])
            if d < 2 or d & (d - 1):
                raise ValueError("A_jax_basis dimension must be a positive power of two, at least two.")
            n_qubits = d.bit_length() - 1
            dimension = d * d
            measurement_count = 24**n_qubits
            if basis.shape != (dimension, d, d):
                raise ValueError("A_jax_basis has inconsistent process dimension.")
            measurements = handle["D_jax_tensors"]
            if measurements.shape != (measurement_count, dimension, dimension):
                raise ValueError("D_jax_tensors shape is incompatible with the QPT_BFW product model.")
            f_vector = handle["f_jax_vector"]
            if f_vector.shape != (measurement_count,):
                raise ValueError("f_jax_vector has inconsistent measurement count.")
            data = cls(n_qubits, local_measurements, local_basis, observations=f_vector[:])
            basis_digits = [
                [(index // (4**q)) % 4 for q in reversed(range(n_qubits))]
                for index in range(dimension)
            ]
            # Validate the complete basis even when D verification is sampled.
            for start in range(0, dimension, chunk_size):
                actual = basis[start : start + chunk_size]
                for offset, matrix in enumerate(actual):
                    index = start + offset
                    expected = _tensor_product(local_basis[basis_digits[index]])
                    if not np.allclose(matrix, expected, rtol=rtol, atol=atol):
                        raise ValueError(f"A_jax_basis row {index} does not match the QPT_BFW basis.")
            verified_penalty_blocks = 0
            if "B_jax_tensors" in handle:
                penalty = handle["B_jax_tensors"]
                if penalty.shape != (dimension, dimension, d, d):
                    raise ValueError("B_jax_tensors has inconsistent process dimension.")
                local_products = np.einsum("mki,nkj->nmij", local_basis.conj(), local_basis)
                for left in range(dimension):
                    for start in range(0, dimension, chunk_size):
                        actual = penalty[left, start : start + chunk_size]
                        for offset, matrix in enumerate(actual):
                            right = start + offset
                            expected = _tensor_product([
                                local_products[n_digit, m_digit]
                                for n_digit, m_digit in zip(basis_digits[left], basis_digits[right])
                            ])
                            if not np.allclose(matrix, expected, rtol=rtol, atol=atol):
                                raise ValueError(
                                    f"B_jax_tensors block ({left}, {right}) does not match "
                                    "the QPT_BFW penalty. Conversion would change the experiment."
                                )
                            verified_penalty_blocks += 1
            if verification == "full":
                row_count = measurement_count
                selected = None
            else:
                # Fixed seed and forced endpoints make conversion reproducible.
                count = min(verification_samples, measurement_count)
                if count == measurement_count:
                    selected = np.arange(measurement_count, dtype=np.int64)
                elif count == 1:
                    selected = np.asarray([0], dtype=np.int64)
                else:
                    interior = np.random.default_rng(0).choice(
                        measurement_count - 2, size=count - 2, replace=False,
                    ) + 1
                    selected = np.sort(np.concatenate(([0], interior, [measurement_count - 1])))
                row_count = selected.size
            for start in range(0, row_count, chunk_size):
                if selected is None:
                    rows = np.arange(start, min(start + chunk_size, row_count), dtype=np.int64)
                    actual = measurements[start : start + chunk_size]
                else:
                    rows = selected[start : start + chunk_size]
                    actual = measurements[rows]
                symbols = data.indices_to_symbols(rows)
                for offset, row in enumerate(rows):
                    expected = _tensor_product(local_measurements[symbols[offset]])
                    if not np.allclose(actual[offset], expected, rtol=rtol, atol=atol):
                        raise ValueError(
                            f"D_jax_tensors row {row} does not match the QPT_BFW product model. "
                            "Conversion would change the experiment."
                        )
            source_stat = Path(path).stat()
            data.metadata = {
                "source_path": str(Path(path)),
                "source_fingerprint": {
                    "method": "filesystem_stat",
                    "size_bytes": source_stat.st_size,
                    "mtime_ns": source_stat.st_mtime_ns,
                },
                "verification": verification,
                "verified_measurement_rows": int(row_count),
                "verified_basis_rows": dimension,
                "verified_penalty_blocks": verified_penalty_blocks,
                "verification_rtol": float(rtol),
                "verification_atol": float(atol),
                "observations_preserved": True,
            }
            return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Existing QPT_BFW HDF5 dataset.")
    parser.add_argument("--save", required=True, help="Destination compact .npz file.")
    parser.add_argument("--verification", choices=("full", "sampled"), default="full")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--verification-samples", type=int, default=256)
    parser.add_argument("--rtol", type=float, default=1e-10, help="Relative operator-verification tolerance.")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute operator-verification tolerance.")
    args = parser.parse_args(argv)
    source, destination = Path(args.h5), Path(args.save)
    if source.resolve() == destination.resolve() or (
        source.exists() and destination.exists() and source.samefile(destination)
    ):
        parser.error("--save must differ from the source HDF5 file.")
    data = StructuredQPTData.from_hdf5(
        args.h5, verification=args.verification, chunk_size=args.chunk_size,
        verification_samples=args.verification_samples, rtol=args.rtol, atol=args.atol,
    )
    data.save_npz(args.save)
    print(
        f"Saved {args.save}: {data.n_qubits} qubits, {data.m} preserved observations; "
        f"{args.verification} verification of {data.metadata['verified_measurement_rows']} measurement rows."
    )
    if args.verification == "sampled":
        print("Sampled verification is incomplete; unexamined rows have not been validated.")


if __name__ == "__main__":
    main()
