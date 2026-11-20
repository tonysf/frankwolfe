#!/usr/bin/env python3
"""Isolated synchronized QPT kernel timings; run only in an allocation.

These timings include per-call dispatch/synchronization and do not add up to
the fused optimizer scan time. Compilation and warmup are excluded. The data
archive remains compact; only one bounded batch and current factors are used.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result", type=Path, help="Use the saved final factor instead of initialization.")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--measurement-backend", choices=("tensor", "rank-one"), default="tensor")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.repeats <= 0:
        parser.error("batch size and repeats must be positive")
    if args.save.exists():
        raise FileExistsError(args.save)

    import numpy as np
    from paper.experiments.qpt_structured_data import StructuredQPTData
    from paper.experiments.quantum_process_tomography import make_factor_initial_point, unpack_factor
    from paper.experiments.quantum_process_tomography_jax import _require_jax, _configure_jax_precision, select_jax_device
    from paper.experiments.quantum_process_tomography_structured_jax import _executable_memory_estimate
    from paper.experiments.qpt_structured_operators import (
        measurement_values, measurement_loss_and_gradient,
        rank_one_measurement_values, rank_one_measurement_loss_and_gradient,
        rank_one_measurement_vectors, trace_preserving_loss_and_gradient,
    )
    jax = _require_jax()
    _configure_jax_precision(jax, "64")
    import jax.numpy as jnp
    device = select_jax_device(args.device)
    data = StructuredQPTData.load_npz(args.data)
    if data.observation_mode != "synthetic-noisy":
        raise ValueError("This profiler expects explicit on-demand synthetic noisy data.")
    if args.result:
        with np.load(args.result, allow_pickle=False) as z:
            factor = z["final_factor"]
    else:
        factor = unpack_factor(make_factor_initial_point(data, 1, 0), data.process_dimension, 1)
    if factor.shape != (data.process_dimension, 1):
        raise ValueError("Expected a matching rank-one factor.")
    symbols = data.sample_symbols(np.random.default_rng(12345), args.batch_size)
    noise = data.noise_for_symbols(symbols)
    rank_one = args.measurement_backend == "rank-one"
    bank = rank_one_measurement_vectors(data.local_measurements) if rank_one else data.local_measurements
    if bank is None:
        raise ValueError("Bank failed rank-one validation.")
    values = rank_one_measurement_values if rank_one else measurement_values
    gradient = rank_one_measurement_loss_and_gradient if rank_one else measurement_loss_and_gradient
    u, truth, s, eps, bank, basis = [jax.device_put(x, device) for x in
                                   (factor, data.truth_factor, symbols.astype(np.int32), noise, bank, data.local_basis)]
    jax.block_until_ready((u, truth, s, eps, bank, basis))

    def targets(truth, symbols, noise, bank):
        return values(truth, symbols, bank, xp=jnp) + noise

    def sampled_gradient(factor, symbols, targets, bank):
        return gradient(factor, symbols, targets, bank, xp=jnp)

    def generated_gradient(factor, truth, symbols, noise, bank):
        return gradient(factor, symbols, targets(truth, symbols, noise, bank), bank, xp=jnp)

    def tp(factor, basis):
        return trace_preserving_loss_and_gradient(factor, basis, xp=jnp)

    y = jax.jit(targets)(truth, s, eps, bank)
    jax.block_until_ready(y)
    kernels = {}
    for name, fn, inputs in (
        ("truth_targets", targets, (truth, s, eps, bank)),
        ("measurement_gradient_with_ready_targets", sampled_gradient, (u, s, y, bank)),
        ("generated_targets_and_measurement_gradient", generated_gradient, (u, truth, s, eps, bank)),
        ("exact_tp_gradient", tp, (u, basis)),
    ):
        began = perf_counter()
        executable = jax.jit(fn).lower(*inputs).compile()
        compilation = perf_counter() - began
        jax.block_until_ready(executable(*inputs))
        samples = []
        for _ in range(args.repeats):
            began = perf_counter()
            jax.block_until_ready(executable(*inputs))
            samples.append(perf_counter() - began)
        kernels[name] = dict(median_seconds=statistics.median(samples), min_seconds=min(samples),
                             samples_seconds=samples, compile_seconds=compilation,
                             compiled_memory_estimate_bytes=_executable_memory_estimate(executable))
    chunk_symbols = data.sample_symbols(np.random.default_rng(0), 100 * args.batch_size)
    samples = []
    for _ in range(args.repeats):
        began = perf_counter()
        data.noise_for_symbols(chunk_symbols)
        samples.append(perf_counter() - began)
    report = dict(status="complete", n_qubits=data.n_qubits, precision="64", rank=1,
                  batch_size=args.batch_size, measurement_backend=args.measurement_backend,
                  device=str(device), device_kind=device.device_kind, hostname=socket.gethostname(),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), kernels=kernels,
                  fixed_noise_100_step_chunk_median_seconds=statistics.median(samples),
                  timing_scope="Isolated calls include dispatch/synchronization; not additive fused-scan timings.",
                  data=str(args.data.resolve()), result=str(args.result.resolve()) if args.result else None,
                  factor_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(factor)).cast("B")).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  jax_version=jax.__version__)
    args.save.parent.mkdir(parents=True, exist_ok=True)
    with args.save.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"n_qubits": data.n_qubits, "backend": args.measurement_backend,
                      "kernel_milliseconds": {k: 1000*v["median_seconds"] for k,v in kernels.items()}}, indent=2))


if __name__ == "__main__":
    main()
