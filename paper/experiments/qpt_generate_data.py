"""Generate compact noisy QPT data without any dense measurement/process tensor.

This constructs a NEW synthetic experiment with the upstream QPT_BFW model:
Haar-unitary coefficients in its normalized real Pauli basis, followed by
legacy quadratic sensing and independent additive Gaussian noise. It does not
reproduce a historical HDF5 realization or run a shot simulator. Run larger
generation jobs in a CPU allocation, not on a cluster login node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from time import perf_counter

import numpy as np

from .qpt_benchmark_resources import process_peak_rss_bytes
from .qpt_structured_data import (
    ROW_ORDER, StructuredQPTData, standard_local_basis, standard_local_measurements,
)
from .qpt_structured_operators import (
    pauli_matrices_to_coefficients, product_measurement_vectors,
    rank_one_measurement_vectors, trace_preserving_residual,
)


UPSTREAM_GENERATOR = (
    "https://github.com/LeNavil/QPT_BFW/blob/"
    "48e9aa80250e8da065734de593afe549f05912ce/qutomo_gt_gen.ipynb"
)
DEFAULT_MAX_OBSERVATIONS = 10_000_000
DEFAULT_MAX_MEMORY_BYTES = 512 * 2**20


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _dimensions(n_qubits):
    n_qubits = _integer(n_qubits, "n_qubits")
    # Reject impossible flat enumeration before evaluating enormous powers.
    if n_qubits > 13:
        raise ValueError("Stored full observations exceed int64 row-index capacity above 13 qubits.")
    return n_qubits, 2**n_qubits, 4**n_qubits, 24**n_qubits


def generation_allocation_estimate(n_qubits, batch_size):
    """Conservative major-array bytes, NOT a measured or enforced RSS limit.

    Covers stored observations, validation/compression headroom, batched sensing
    vectors and conjugates, unitary QR/basis-transform work, and row symbols.
    It contains no term for dense D, B, the global basis, or chi.
    """
    n_qubits, _, dimension, count = _dimensions(n_qubits)
    batch_size = min(_integer(batch_size, "batch_size"), count)
    return int(3 * 8 * count + 4 * 16 * batch_size * dimension
               + 32 * 16 * dimension + 8 * batch_size * (4 * n_qubits + 8)
               + 2**20)


def _validate(n_qubits, noise_std, channel_seed, noise_seed, batch_size,
              max_observations, max_memory_bytes):
    n_qubits, d, dimension, count = _dimensions(n_qubits)
    batch_size = _integer(batch_size, "batch_size")
    max_observations = _integer(max_observations, "max_observations")
    max_memory_bytes = _integer(max_memory_bytes, "max_memory_bytes")
    channel_seed = _integer(channel_seed, "channel_seed", 0)
    noise_seed = _integer(noise_seed, "noise_seed", 0)
    if max(channel_seed, noise_seed) > np.iinfo(np.uint64).max:
        raise ValueError("Seeds must fit uint64.")
    if (isinstance(noise_std, (bool, np.bool_)) or np.ndim(noise_std) != 0
            or not np.isrealobj(noise_std)):
        raise ValueError("noise_std must be a finite nonnegative real scalar.")
    noise_std = float(noise_std)
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be a finite nonnegative real scalar.")
    if count > max_observations:
        raise ValueError(f"{count} observations exceed max_observations={max_observations}.")
    estimate = generation_allocation_estimate(n_qubits, batch_size)
    if estimate > max_memory_bytes:
        raise ValueError(f"Generation estimate {estimate} bytes exceeds max_memory_bytes={max_memory_bytes}.")
    return n_qubits, d, dimension, count, noise_std, channel_seed, noise_seed, batch_size, estimate


def _array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _haar_unitary(d, seed):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(0,))))
    matrix = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    unitary, triangular = np.linalg.qr(matrix)
    diagonal = np.diag(triangular)
    if np.any(np.abs(diagonal) == 0):
        raise FloatingPointError("Degenerate QR diagonal while constructing the Haar unitary.")
    unitary *= (diagonal / np.abs(diagonal))[None, :]
    if not np.allclose(unitary.conj().T @ unitary, np.eye(d), rtol=1e-11, atol=1e-12):
        raise FloatingPointError("Generated matrix failed its unitary check.")
    return unitary


def generate_compact_data(n_qubits, *, noise_std=0.05, channel_seed=0, noise_seed=0,
                          batch_size=1024, max_observations=DEFAULT_MAX_OBSERVATIONS,
                          max_memory_bytes=DEFAULT_MAX_MEMORY_BYTES):
    """Return a stored-observation ``StructuredQPTData`` with rank-one truth.

    ``c_k = Tr(P_k^H H)`` has norm sqrt(2**n), with no extra 1/d. Targets
    follow the upstream literal ``real(vdot(D_s, c c^H))`` convention, equal
    to ``c^H D_s c``. No conjugation correction, clipping, renormalization,
    or shot noise is added. With this real basis and upstream D convention,
    these targets also equal Born probabilities of conjugate(H), not H.

    Domain-separated PCG64 streams control channel and Gaussian noise, even
    when both user seeds are equal (SeedSequence spawn keys 0 and 1). Within the
    same numerical environment, batch size changes neither the noise stream
    nor rowwise summation order. Cross-platform QR/NumPy bitwise identity is
    not promised; source, versions, seeds and numeric hashes are recorded.
    """
    (n_qubits, d, dimension, count, noise_std, channel_seed, noise_seed,
     batch_size, estimate) = _validate(n_qubits, noise_std, channel_seed, noise_seed,
                                     batch_size, max_observations, max_memory_bytes)
    began = perf_counter()
    basis, bank = standard_local_basis(), standard_local_measurements()
    unitary = _haar_unitary(d, channel_seed)
    truth = np.ascontiguousarray(pauli_matrices_to_coefficients(unitary[None, :, :], basis, xp=np))
    tp_violation = float(np.linalg.norm(trace_preserving_residual(truth, basis, xp=np)))
    if not np.isfinite(tp_violation) or tp_violation > 1e-10 * np.sqrt(d):
        raise FloatingPointError("Generated coefficients failed the trace-preservation check.")
    if not np.isclose(np.vdot(truth, truth).real, d, rtol=1e-11, atol=1e-12):
        raise FloatingPointError("Generated coefficients have the wrong normalized-basis trace.")
    # This small adapter supplies legacy row decoding without a placeholder b.
    adapter = StructuredQPTData(n_qubits, bank, basis, truth_factor=truth,
                                observation_mode="noiseless")
    vectors = rank_one_measurement_vectors(bank)
    if vectors is None:
        raise ValueError("Standard local bank failed its rank-one reconstruction check.")
    observations = np.empty(count, dtype=np.float64)
    noise_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(noise_seed, spawn_key=(1,))))
    noise_digest = hashlib.sha256()
    predicted_min, predicted_max = math.inf, -math.inf
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            symbols = adapter.indices_to_symbols(np.arange(start, stop, dtype=np.int64))
            selected = product_measurement_vectors(symbols, vectors, xp=np)
            # Fixed rowwise reduction avoids BLAS batch-shape kernel changes.
            amplitudes = np.einsum("bi,i->b", selected.conj(), truth[:, 0], optimize=False)
            predicted = (amplitudes.conj() * amplitudes).real
            if not np.all(np.isfinite(predicted)) or np.any(predicted < -1e-12) or np.any(predicted > 1 + 1e-10):
                raise FloatingPointError("Noiseless legacy sensing probabilities failed validation.")
            predicted_min = min(predicted_min, float(predicted.min()))
            predicted_max = max(predicted_max, float(predicted.max()))
            noise = noise_rng.standard_normal(stop - start)
            noise_digest.update(memoryview(noise).cast("B"))
            observations[start:stop] = predicted + noise_std * noise
    if not np.all(np.isfinite(observations)):
        raise FloatingPointError("Generated observations are nonfinite.")
    metadata = {
        "generator": "qpt_generate_data_v1", "source_kind": "synthetic_compact",
        "verification": "constructed", "upstream_generator": UPSTREAM_GENERATOR,
        "channel_model": "haar_unitary", "channel_seed": channel_seed,
        "channel_rng": "Generator(PCG64(SeedSequence(channel_seed, spawn_key=(0,))))", "truth_rank": 1,
        "coefficient_convention": "c_k=Tr(P_k^H H), no 1/d; P local=[I,X,-iY,Z]/sqrt(2)",
        "observation_formula": "real(vdot(D_s, c c^H)) + noise_std * z_s; z_s iid N(0,1)",
        "physical_convention_note": "Preserves upstream D/chi convention; sensing is Born(conjugate(H)) in the real basis.",
        "noise_model": "iid_additive_gaussian", "noise_std": noise_std, "noise_seed": noise_seed,
        "noise_rng": "Generator(PCG64(SeedSequence(noise_seed, spawn_key=(1,))))", "noise_clipped": False,
        "shot_noise": False, "legacy_observations_reused": False,
        "row_order": ROW_ORDER, "generation_batch_size": batch_size,
        "generated_observations": count, "resource_estimate_bytes": estimate,
        "resource_estimate_note": "Major-array estimate with headroom, not measured/enforced total process RAM.",
        "noiseless_min": predicted_min, "noiseless_max": predicted_max,
        "truth_tp_violation": tp_violation, "truth_trace": float(np.vdot(truth, truth).real),
        "unitary_sha256": _array_sha256(unitary), "truth_factor_sha256": _array_sha256(truth),
        "observations_sha256": _array_sha256(observations),
        "noise_standard_normals_sha256": noise_digest.hexdigest(),
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "numerical_source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("qpt_generate_data.py", "qpt_structured_data.py", "qpt_structured_operators.py")
        },
        "generation_seconds": perf_counter() - began,
        "generator_process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    return StructuredQPTData(n_qubits, bank, basis, observations=observations,
                             truth_factor=truth, metadata=metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--channel-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-observations", type=int, default=DEFAULT_MAX_OBSERVATIONS)
    parser.add_argument("--max-memory-mib", type=float, default=512)
    parser.add_argument("--save", type=Path, required=True, help="New compact NPZ; never overwrite an existing path.")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_memory_mib) or args.max_memory_mib <= 0:
        parser.error("--max-memory-mib must be finite and positive.")
    options = dict(noise_std=args.noise_std, channel_seed=args.channel_seed, noise_seed=args.noise_seed,
                   batch_size=args.batch_size, max_observations=args.max_observations,
                   max_memory_bytes=int(args.max_memory_mib * 2**20))
    try:
        validated = _validate(args.n_qubits, **options)
    except ValueError as error:
        parser.error(str(error))
    # Do not resolve through a symlink at the destination, including a broken one.
    destination = Path(os.path.abspath(args.save))
    if os.path.lexists(destination):
        raise FileExistsError(f"Refusing to overwrite existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    estimated_archive_bytes = 8 * validated[3] + 16 * validated[2] + 2**20
    if shutil.disk_usage(destination.parent).free < estimated_archive_bytes:
        raise OSError("Insufficient free disk space for the uncompressed-sized compact archive estimate.")
    print(f"Generating {validated[3]} observations for n={args.n_qubits}; "
          f"estimated major-array storage={validated[-1] / 2**20:.2f} MiB", flush=True)
    data = generate_compact_data(args.n_qubits, **options)
    # Publish only a complete archive, atomically and without clobbering a path
    # created by another process while generation was running.
    with tempfile.TemporaryDirectory(prefix=".qpt-generate-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "data.npz"
        data.save_npz(staged)
        os.link(staged, destination)
    print(json.dumps({
        "status": "complete", "data": str(destination), "n_qubits": data.n_qubits,
        "observations": data.m, "observation_bytes": int(data.observations.nbytes),
        "truth_factor_sha256": data.metadata["truth_factor_sha256"],
        "observations_sha256": data.metadata["observations_sha256"],
        "generation_seconds": data.metadata["generation_seconds"],
        "generator_process_peak_rss_bytes": process_peak_rss_bytes(),
    }, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
