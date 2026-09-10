#!/usr/bin/env python3
"""Compare dense and structured QPT in fresh processes on identical stored data.

Run inside a GPU allocation, for example:
  python scripts/benchmark_qpt_structured.py --h5 /scratch/input.h5 \
      --device gpu --steps 1000 --metric-mode full --save results/comparison.json

Optimizer timings synchronize the device and exclude explicit compilation and
warmup. Memory figures are estimates, never measured GPU peak allocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--h5", type=Path, required=True)
    result.add_argument("--save", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu")
    result.add_argument("--device-index", type=int, default=0)
    result.add_argument("--precision", choices=("64", "32"), default="64")
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--rank", type=int, default=1)
    result.add_argument("--tau", type=float, default=10.0)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--chunk-steps", type=int, default=100)
    result.add_argument("--metrics-every", type=int, default=100)
    result.add_argument("--metric-mode", choices=("sampled", "full"), default="full")
    result.add_argument("--metric-samples", type=int, default=512)
    result.add_argument("--metric-batch-size", type=int, default=32)
    result.add_argument("--metric-seed", type=int, default=12345)
    result.add_argument("--initialization-seed", type=int, default=0)
    result.add_argument("--sampling-seed", type=int, default=0)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--backends", nargs="+", choices=("dense", "structured"), default=["dense", "structured"])
    result.add_argument("--measurement-backend", choices=("auto", "tensor", "rank-one"), default="auto")
    result.add_argument("--rtol", type=float, default=1e-10, help="HDF5 operator-verification relative tolerance.")
    result.add_argument("--atol", type=float, default=1e-12, help="HDF5 operator-verification absolute tolerance.")
    for name, scale, offset, exponent in (
        ("rho", 4.0, 8.0, 2.0 / 3.0),
        ("smoothing", 10.0, 1.0, 0.25),
        ("step", 1.0, 1.0, 0.5),
    ):
        result.add_argument(f"--{name}-scale", type=float, default=scale)
        result.add_argument(f"--{name}-offset", type=float, default=offset)
        result.add_argument(f"--{name}-exponent", type=float, default=exponent)
    result.add_argument("--worker", choices=("dense", "structured"), help=argparse.SUPPRESS)
    result.add_argument("--compact-data", type=Path, help=argparse.SUPPRESS)
    return result


def common_options(args):
    from paper.experiments.quantum_process_tomography import PowerSchedule

    schedules = {
        name: PowerSchedule(
            getattr(args, name + "_scale"), getattr(args, name + "_offset"),
            getattr(args, name + "_exponent"), cap=None if name == "smoothing" else 1.0,
        )
        for name in ("rho", "smoothing", "step")
    }
    return dict(
        n_steps=args.steps, rank=args.rank, tau=args.tau, batch_size=args.batch_size,
        initialization_seed=args.initialization_seed, sampling_seed=args.sampling_seed,
        rho_schedule=schedules["rho"], smoothing_schedule=schedules["smoothing"],
        step_size_schedule=schedules["step"], metrics_frequency=args.metrics_every,
        device=args.device, device_index=args.device_index, precision=args.precision,
        warmup=True, show_progress=False,
    )


def run_worker(args):
    import numpy as np

    from paper.experiments.qpt_structured_data import StructuredQPTData
    from paper.experiments.quantum_process_tomography import QPTData
    from paper.experiments import quantum_process_tomography_jax as dense_module
    from paper.experiments.quantum_process_tomography_structured_jax import run_qpt_structured_jax

    load_started = perf_counter()
    data = QPTData.from_hdf5(args.h5) if args.worker == "dense" else StructuredQPTData.load_npz(args.compact_data)
    load_seconds = perf_counter() - load_started
    started = perf_counter()
    if args.worker == "dense":
        metric_seconds = 0.0
        original_metrics = dense_module._posthoc_metrics

        def timed_metrics(*values, **kwargs):
            nonlocal metric_seconds
            start = perf_counter()
            try:
                return original_metrics(*values, **kwargs)
            finally:
                metric_seconds += perf_counter() - start

        # Instrument the legacy runner without editing or replacing its math.
        with patch.object(dense_module, "_posthoc_metrics", timed_metrics):
            result = dense_module.run_qpt_stochastic_frames_jax(
                data, execution_mode="scan", **common_options(args),
            )
        row = dict(
            backend="dense", optimizer_seconds=result.total_seconds,
            optimizer_compile_seconds=result.compile_seconds, metric_seconds=metric_seconds,
            metric_location="CPU NumPy; full measurements and process-matrix eigendecomposition",
            metric_mode="full", metric_compile_included=False,
            device_platform=result.device_platform, device_kind=result.device_kind,
            jax_version=result.jax_version, jaxlib_version=result.jaxlib_version,
            batch_indices_sha256=hashlib.sha256(result.batch_indices.astype(np.int64).tobytes()).hexdigest(),
            final_measurement_loss=float(result.measurement_loss[-1]),
            final_smoothed_gap=float(result.exact_smoothed_gap[-1]),
            warmup="one complete scan, outside optimizer_seconds",
        )
    else:
        result = run_qpt_structured_jax(
            data, chunk_steps=args.chunk_steps, metric_mode=args.metric_mode,
            metric_samples=args.metric_samples, metric_batch_size=args.metric_batch_size,
            metric_seed=args.metric_seed, measurement_backend=args.measurement_backend,
            **common_options(args),
        )
        meta = result.metadata
        row = dict(
            backend="structured", optimizer_seconds=meta["optimizer_seconds"],
            optimizer_compile_seconds=meta["compile_seconds"], metric_seconds=meta["metric_seconds"],
            metric_location="selected JAX device; streamed measurement batches and exact TP",
            metric_mode=meta["metric_mode"], metric_compile_included=True,
            metric_sample_count=meta["metric_samples"],
            device_platform=meta["device_platform"], device_kind=meta["device_kind"],
            jax_version=meta["jax_version"], jaxlib_version=meta["jaxlib_version"],
            batch_symbols_sha256=meta["batch_symbols_sha256"],
            compiled_memory_estimate_bytes=meta["compiled_memory_estimate_bytes"],
            sampling_transfer_seconds=meta["sampling_transfer_seconds"],
            measurement_backend=meta["measurement_backend"],
            final_measurement_loss=float(result.measurement_loss[-1]),
            final_smoothed_gap=float(result.smoothed_gap[-1]),
            warmup="one chunk, outside optimizer_seconds",
        )
    row["runner_wall_seconds"] = perf_counter() - started
    row["load_seconds"] = load_seconds
    row["optimizer_steps_per_second"] = args.steps / row["optimizer_seconds"]
    # Exchange a numeric array on disk, not a potentially enormous JSON list.
    factor_path = args.compact_data.parent / (args.worker + "-factor.npy")
    np.save(factor_path, result.final_factor, allow_pickle=False)
    print(json.dumps(row, allow_nan=False))
    return 0


def memory_estimates(data, args):
    """Static array accounting, not a prediction of device peak memory."""
    complex_bytes = 16 if args.precision == "64" else 8
    real_bytes = complex_bytes // 2
    n, d, m = data.process_dimension, data.d, data.m
    dense_d = m * n * n * complex_bytes
    dense_b = n * n * d * d * complex_bytes
    return {
        "kind": "array-size estimates; excludes allocator reservation, copies, compiler and kernel workspace",
        "dense_D_device_bytes": dense_d,
        "dense_B_device_bytes": dense_b,
        "dense_static_device_bytes": dense_d + dense_b + m * real_bytes + d * d * complex_bytes,
        "dense_scan_iterate_history_device_bytes": args.steps * n * args.rank * complex_bytes,
        "dense_host_iterate_history_bytes": args.steps * n * args.rank * complex_bytes,
        "dense_history_note": "one scan history only; warmup/history copies can coexist, so these are not peak estimates",
        "structured_host_observation_bytes": data.observations.nbytes,
        "structured_full_local_bank_device_bytes": (24 * 4 * 4 + 4 * 2 * 2) * complex_bytes,
        "factor_device_bytes": n * args.rank * complex_bytes,
        "structured_checkpoint_factor_bytes": 0,
        "structured_host_scalar_traces_scale": "O(steps), not O(steps * process_dimension * rank)",
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        return run_worker(args)
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    if len(args.backends) != len(set(args.backends)):
        raise ValueError("--backends must not contain duplicates.")
    from paper.experiments.qpt_structured_data import StructuredQPTData
    import numpy as np

    # This one-time CPU conversion validates the exact stored operators and f.
    converted_started = perf_counter()
    data = StructuredQPTData.from_hdf5(
        args.h5, verification="full", chunk_size=args.metric_batch_size,
        rtol=args.rtol, atol=args.atol,
    )
    conversion_seconds = perf_counter() - converted_started
    report = dict(
        schema_version=1, h5=str(args.h5.resolve()), n_qubits=data.n_qubits,
        observations_preserved=True, verification="full", conversion_seconds=conversion_seconds,
        configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                       if key not in {"worker", "compact_data"}},
        memory_estimates=memory_estimates(data, args), runs=[],
        interpretation=[
            "Device optimizer calls are synchronized; optimizer compilation and warmup are excluded.",
            "Structured metric time includes first-call metric JIT compilation; optimizer_compile_seconds does not.",
            "Dense metrics run on CPU and include an eigendecomposition; metric timings cover different implementations.",
            "Full structured metrics use all original observations; sampled metrics reuse a fixed sample and are not exact full gaps.",
            "Each run uses a fresh process. GPU memory numbers are array/compiler estimates, not measured peak usage.",
            "A CPU timing ratio is not evidence of A100 speedup. Compare optimizer_seconds on the target GPU.",
        ],
    )
    with tempfile.TemporaryDirectory(prefix="qpt-structured-benchmark-") as temporary:
        compact = Path(temporary) / "data.npz"
        data.save_npz(compact)
        base = [sys.executable, str(Path(__file__).resolve())]
        source_args = list(argv) if argv is not None else sys.argv[1:]
        for repeat in range(args.repeats):
            for backend in args.backends:
                command = base + source_args + ["--worker", backend, "--compact-data", str(compact)]
                started = perf_counter()
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=os.environ.copy())
                if process.returncode:
                    raise RuntimeError(f"{backend} benchmark failed:\n{process.stderr}\n{process.stdout}")
                row = json.loads(process.stdout.strip().splitlines()[-1])
                row.update(repeat=repeat, process_wall_seconds=perf_counter() - started)
                report["runs"].append(row)
                print(f"{backend} repeat {repeat+1}: optimizer={row['optimizer_seconds']:.4f}s, "
                      f"compile={row['optimizer_compile_seconds']:.4f}s, metrics={row['metric_seconds']:.4f}s", flush=True)
        if set(args.backends) == {"dense", "structured"}:
            first = np.load(Path(temporary) / "dense-factor.npy", allow_pickle=False)
            second = np.load(Path(temporary) / "structured-factor.npy", allow_pickle=False)
            report["final_factor_max_abs_difference"] = float(np.max(np.abs(first - second)))
    report["medians"] = {
        backend: {
            key: statistics.median(row[key] for row in report["runs"] if row["backend"] == backend)
            for key in ("optimizer_seconds", "optimizer_compile_seconds", "metric_seconds", "runner_wall_seconds")
        }
        for backend in args.backends
    }
    if set(args.backends) == {"dense", "structured"}:
        report["dense_over_structured_optimizer_ratio"] = (
            report["medians"]["dense"]["optimizer_seconds"] / report["medians"]["structured"]["optimizer_seconds"]
        )
        report["dense_over_structured_runner_wall_ratio"] = (
            report["medians"]["dense"]["runner_wall_seconds"] / report["medians"]["structured"]["runner_wall_seconds"]
        )
        dense = next(row for row in report["runs"] if row["backend"] == "dense")
        structured = next(row for row in report["runs"] if row["backend"] == "structured")
        if args.metric_mode == "full":
            report["final_full_loss_abs_difference"] = abs(dense["final_measurement_loss"] - structured["final_measurement_loss"])
            report["final_full_gap_abs_difference"] = abs(dense["final_smoothed_gap"] - structured["final_smoothed_gap"])
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
