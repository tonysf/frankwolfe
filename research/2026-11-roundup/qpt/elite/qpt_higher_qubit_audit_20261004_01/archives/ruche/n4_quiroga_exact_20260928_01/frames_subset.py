"""Run FRAMES with minibatches drawn only from a fixed measurement subset."""
import argparse
from pathlib import Path

import numpy as np

from paper.experiments import quantum_process_tomography_structured_jax as qpt
from paper.experiments.qpt_structured_data import StructuredQPTData
from subset import fixed_subset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--save", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--subset-size", type=int, default=8192)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--initialization-seed", type=int, default=0)
    p.add_argument("--beta0", type=float, required=True)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    data = StructuredQPTData.load_npz(a.data)
    _, symbols, _, hashes = fixed_subset(data, a.subset_size, a.subset_seed)
    original_sample = data.sample_symbols

    def sample_subset(rng, batch_size):
        return symbols[rng.integers(0, symbols.shape[0], size=batch_size)].copy()

    data.sample_symbols = sample_subset
    try:
        result = qpt.run_qpt_structured_jax(
            data, n_steps=a.steps, rank=1, tau=20.0, batch_size=32,
            initialization_seed=a.initialization_seed, sampling_seed=a.subset_seed,
            beta0=a.beta0, chunk_steps=100, metrics_frequency=100,
            metric_mode="sampled", metric_samples=512, metric_batch_size=32,
            metric_seed=12345, device="gpu", precision="64",
            show_progress=not a.quiet, measurement_backend="tensor", prefetch=False,
        )
    finally:
        data.sample_symbols = original_sample
    result.metadata.update(
        optimizer="frames_fixed_subset", subset_size=int(a.subset_size),
        subset_seed=int(a.subset_seed), subset_without_replacement=True,
        optimization_population="fixed subset; minibatches sampled with replacement",
        beta0=float(a.beta0), **hashes,
    )
    qpt.save_structured_result(a.save, result)
    print(f"Saved {a.save}; fidelity={result.process_fidelity_proxy[-1]:.12g}; TP={result.tp_violation[-1]:.12g}")


if __name__ == "__main__":
    main()
