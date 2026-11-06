"""Deterministic fixed measurement subsets shared by both optimizers."""
import hashlib

import numpy as np


def fixed_subset(data, size, seed):
    if not 0 < size <= data.m:
        raise ValueError("invalid subset size")
    rng = np.random.default_rng(seed)
    indices = rng.choice(data.m, size=size, replace=False)
    symbols = data.indices_to_symbols(indices).astype(np.int32)
    observations = np.asarray(data.observations_for_symbols(symbols), dtype=np.float64)
    digest = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    return indices, symbols, observations, dict(
        subset_indices_sha256=digest(indices),
        subset_symbols_sha256=digest(symbols),
        subset_observations_sha256=digest(observations),
    )
