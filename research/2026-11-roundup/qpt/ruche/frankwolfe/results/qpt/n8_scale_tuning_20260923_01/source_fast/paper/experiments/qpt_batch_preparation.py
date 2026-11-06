"""Exact host-only batch preparation for the int64 on-demand QPT model.

Keep the legacy PCG64 row stream and the existing NumPy noise implementation.
This is an experimental execution optimization, not a new observation model.
"""
import numpy as np
from paper.experiments.qpt_observation_noise import fixed_row_noise


def decode_rows(indices, n_qubits):
    """Decode the legacy mixed-radix rows, using bounded uint32 digits."""
    if not 1 <= n_qubits <= 13:
        raise ValueError('Optimized decoder supports n=1..13 (int64 row indices).')
    rows = np.asarray(indices, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= 24**n_qubits):
        raise ValueError('Invalid row indices.')
    inputs = (rows // (6**n_qubits)).astype(np.uint32)
    axes = ((rows // (2**n_qubits)) % (3**n_qubits)).astype(np.uint32)
    outcomes = (rows % (2**n_qubits)).astype(np.uint32)
    symbols = np.empty((n_qubits, rows.size), dtype=np.int32)
    for q in range(n_qubits):
        symbols[q] = (6 * (inputs & 3) + 2 * (axes % 3)
                      + ((outcomes >> (n_qubits-q-1)) & 1))
        inputs >>= 2
        axes //= 3
    return np.ascontiguousarray(symbols.T)


def prepare_on_demand_batch(data, rng, count):
    """Return exactly the legacy symbols and fixed noise, without re-encoding."""
    rows = rng.integers(0, data.m, size=count)
    symbols = decode_rows(rows, data.n_qubits)
    noise = fixed_row_noise(rows, data.metadata['noise_seed'], data.metadata['noise_std'])
    return symbols, noise
