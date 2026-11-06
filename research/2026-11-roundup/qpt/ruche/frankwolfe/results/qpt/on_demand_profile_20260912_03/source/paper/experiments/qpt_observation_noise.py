"""Random-access Gaussian observation noise, with no dataset-sized state.

Each legacy row index owns one fixed pseudorandom draw. SplitMix64 provides
two domain-separated uniform streams; Box-Muller maps them to a normal draw.
This defines a NEW noise realization, not the sequential PCG64 archive stream.
Only small host batches use this module; noiseless targets are computed on GPU.
"""

import numpy as np


NOISE_GENERATOR = "splitmix64_box_muller_row_v1"


def _mix64(value):
    with np.errstate(over="ignore"):
        value = np.asarray(value, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return value ^ (value >> np.uint64(31))


def fixed_row_noise(indices, seed, std):
    """Return fixed float64 noise, invariant to batching, order and repeats."""
    indices = np.asarray(indices)
    if indices.dtype.kind not in "iu" or np.any(indices < 0):
        raise ValueError("Noise indices must be nonnegative integers.")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed <= np.iinfo(np.uint64).max):
        raise ValueError("Noise seed must be a uint64 integer.")
    if not np.isfinite(std) or std < 0:
        raise ValueError("Noise std must be finite and nonnegative.")
    rows = indices.astype(np.uint64, copy=False)
    key = _mix64(np.uint64(seed) ^ np.uint64(0xD2B74407B1CE6E93))
    first = _mix64(rows ^ key)
    second = _mix64(rows ^ _mix64(key))
    # u1 in (0, 1], u2 in [0, 1): log never receives zero.
    u1 = ((first >> np.uint64(11)).astype(np.float64) + 1.0) * 2.0**-53
    u2 = (second >> np.uint64(11)).astype(np.float64) * 2.0**-53
    return float(std) * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
