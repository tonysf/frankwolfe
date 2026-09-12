#!/usr/bin/env python3
"""Matched QPT timings and separate CPU/GPU memory profiling.

Run from an existing GPU environment; this script never connects to a cluster.
Timing repetitions use fresh, unmonitored processes. --profile-memory adds one
separate monitored process per backend. Driver VRAM samples, allocator highwater
counters, and compiler/array estimates are deliberately reported separately.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
from time import perf_counter
import traceback
from unittest.mock import patch
import zipfile


SCRIPT_STARTED = perf_counter()
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUMERICAL_SOURCE_FILES = (
    "scripts/benchmark_qpt_structured.py",
    "paper/experiments/qpt_structured_data.py",
    "paper/experiments/qpt_structured_operators.py",
    "paper/experiments/quantum_process_tomography.py",
    "paper/experiments/quantum_process_tomography_jax.py",
    "paper/experiments/quantum_process_tomography_structured_jax.py",
    "paper/experiments/qpt_benchmark_resources.py",
)
SCHEDULER_ENVIRONMENT_KEYS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")


class BenchmarkParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.backends is None:
            parsed.backends = ["structured"] if parsed.data is not None else ["dense", "structured"]
        if parsed.metric_mode is None:
            parsed.metric_mode = "sampled" if parsed.data is not None else "full"
        if parsed.data is not None and ("dense" in parsed.backends or parsed.worker == "dense"):
            self.error("--data supports only the structured backend; dense operators are never reconstructed.")
        return parsed


def parser():
    result = BenchmarkParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--h5", type=Path, help="Legacy HDF5 to convert and compare (default backends: dense structured).")
    source.add_argument("--data", type=Path, help="Existing compact NPZ; structured-only, with no HDF5 access or conversion.")
    result.add_argument("--max-data-mib", type=float, default=256,
                        help="Compact input compressed plus uncompressed size guard (default: 256 MiB).")
    result.add_argument("--save", type=Path, required=True)
    result.add_argument("--artifacts-dir", type=Path, help="New directory for compact data, worker logs, JSON and factors.")
    result.add_argument("--device", choices=("cpu", "gpu", "auto"), default="gpu")
    result.add_argument("--device-index", type=int, default=0)
    result.add_argument("--precision", choices=("64", "32"), default="64")
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--rank", type=int, default=1)
    result.add_argument("--tau", type=float, default=10.0)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--chunk-steps", type=int, default=100)
    result.add_argument("--metrics-every", type=int, default=100)
    result.add_argument("--metric-mode", choices=("sampled", "full"),
                        help="Default: full with --h5, sampled with --data.")
    result.add_argument("--metric-samples", type=int, default=512)
    result.add_argument("--metric-batch-size", type=int, default=32)
    result.add_argument("--metric-seed", type=int, default=12345)
    result.add_argument("--initialization-seed", type=int, default=0)
    result.add_argument("--sampling-seed", type=int, default=0)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--backends", nargs="+", choices=("dense", "structured"))
    result.add_argument("--measurement-backend", choices=("auto", "tensor", "rank-one"), default="auto")
    result.add_argument("--rtol", type=float, default=1e-10, help="HDF5 operator-verification relative tolerance.")
    result.add_argument("--atol", type=float, default=1e-12, help="HDF5 operator-verification absolute tolerance.")
    result.add_argument("--allocator", choices=("grow", "default"), default="grow",
                        help="Identical worker allocator policy: grow disables preallocation; default enables it.")
    result.add_argument("--profile-memory", action="store_true", help="Add separate profiling runs; exclude them from timing medians.")
    result.add_argument("--memory-poll-ms", type=float, default=100.0)
    result.add_argument("--require-gpu-memory", action="store_true",
                        help="Fail the report if a profiling run has neither a GPU allocator peak nor process-VRAM samples.")
    for name, scale, offset, exponent in (
        ("rho", 4.0, 8.0, 2.0 / 3.0), ("smoothing", 10.0, 1.0, 0.25), ("step", 1.0, 1.0, 0.5),
    ):
        result.add_argument(f"--{name}-scale", type=float, default=scale)
        result.add_argument(f"--{name}-offset", type=float, default=offset)
        result.add_argument(f"--{name}-exponent", type=float, default=exponent)
    result.add_argument("--worker", choices=("dense", "structured"), help=argparse.SUPPRESS)
    result.add_argument("--compact-data", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--compact-sha256", help=argparse.SUPPRESS)
    result.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--worker-factor", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--purpose", choices=("timing", "memory"), default="timing", help=argparse.SUPPRESS)
    return result


ALLOCATOR_ENVIRONMENT_KEYS = (
    "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION", "TF_GPU_ALLOCATOR",
    "JAX_ENABLE_COMPILATION_CACHE", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS",
    "XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
)


def configured_worker_environment(args):
    """Set the same policy before either worker imports JAX; do not alter GPU binding."""
    environment = os.environ.copy()
    for name in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION",
                 "TF_GPU_ALLOCATOR", "JAX_COMPILATION_CACHE_DIR"):
        environment.pop(name, None)
    environment["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" if args.allocator == "grow" else "true"
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def write_json(path, value):
    """Atomically publish even partial reports; never emit non-standard NaN JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_fingerprint(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(memoryview(array)).hexdigest())


def finite_scalar(value, name):
    scalar = float(value)
    if not math.isfinite(scalar):
        raise FloatingPointError(f"Nonfinite {name}.")
    return scalar


def guarded_compact_input(path, max_data_mib):
    """Inspect ZIP sizes before NumPy materializes any compact arrays."""
    if not math.isfinite(max_data_mib) or max_data_mib <= 0:
        raise ValueError("--max-data-mib must be finite and positive.")
    path = Path(path).resolve()
    size = path.stat().st_size
    limit = int(max_data_mib * 2**20)
    if size > limit:
        raise ValueError("Compact file exceeds --max-data-mib; no data loaded.")
    with zipfile.ZipFile(path) as archive:
        expanded = sum(item.file_size for item in archive.infolist())
    if size + expanded > limit:
        raise ValueError("Compact compressed plus uncompressed data exceed --max-data-mib; no data loaded.")
    return dict(path=str(path), sha256=file_sha256(path), file_bytes=size,
                uncompressed_bytes=expanded, input_size_guard_bytes=limit,
                size_guard_note="Compressed plus uncompressed file sizes; not a total process RAM limit.")


def common_options(args):
    from paper.experiments.quantum_process_tomography import PowerSchedule
    schedules = {
        name: PowerSchedule(getattr(args, name + "_scale"), getattr(args, name + "_offset"),
                            getattr(args, name + "_exponent"), cap=None if name == "smoothing" else 1.0)
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
    """Measure one unmodified algorithm in a clean process, retaining phase boundaries."""
    from paper.experiments.qpt_benchmark_resources import process_peak_rss_bytes, device_memory_snapshot

    row = dict(backend=args.worker, purpose=args.purpose, status="running", pid=os.getpid(), hostname=socket.gethostname(),
               allocator_environment={key: os.environ.get(key) for key in ALLOCATOR_ENVIRONMENT_KEYS},
               scheduler_environment={key: os.environ.get(key) for key in SCHEDULER_ENVIRONMENT_KEYS})
    phases = row["phases"] = {}
    row["memory"] = {"cpu_process_peak_rss_bytes": None, "jax_allocator": None, "external_sampler": None}
    selected = None
    try:
        started = perf_counter()
        import numpy as np
        from paper.experiments.qpt_structured_data import StructuredQPTData
        from paper.experiments.quantum_process_tomography import QPTData
        from paper.experiments import quantum_process_tomography_jax as dense_module
        from paper.experiments.quantum_process_tomography_structured_jax import run_qpt_structured_jax
        phases["python_import_seconds"] = perf_counter() - started

        if args.data is not None:
            started = perf_counter()
            compact_check = guarded_compact_input(args.compact_data or args.data, args.max_data_mib)
            if args.compact_sha256 is not None and compact_check["sha256"] != args.compact_sha256:
                raise RuntimeError("Compact input differs from the parent's recorded SHA-256.")
            row["compact_input_sha256"] = compact_check["sha256"]
            phases["input_validation_seconds"] = perf_counter() - started

        started = perf_counter()
        jax = dense_module._require_jax()
        dense_module._configure_jax_precision(jax, args.precision)
        selected = dense_module.select_jax_device(args.device, args.device_index)
        probe = jax.device_put(np.zeros(1, dtype=np.float64), selected)
        jax.block_until_ready(probe)
        del probe
        phases["device_initialization_seconds"] = perf_counter() - started
        row["device_platform"], row["device_kind"], row["device"] = selected.platform, selected.device_kind, str(selected)
        row["memory"]["jax_allocator_before_data"] = device_memory_snapshot(selected)

        started = perf_counter()
        data = (QPTData.from_hdf5(args.h5) if args.worker == "dense"
                else StructuredQPTData.load_npz(args.compact_data or args.data))
        if args.data is not None and data.observation_mode != "stored":
            raise ValueError("Compact benchmark requires stored observations, including their fixed noise realization.")
        row["load_seconds"] = phases["data_load_seconds"] = perf_counter() - started
        runner_started = perf_counter()

        if args.worker == "dense":
            original_metrics, original_scan = dense_module._posthoc_metrics, dense_module._run_scan_mode
            scan_finished = metrics_finished = None

            def timed_scan(*values, **kwargs):
                nonlocal scan_finished
                phases["runner_setup_seconds"] = perf_counter() - runner_started
                began = perf_counter()
                output = original_scan(*values, **kwargs)
                scan_finished = perf_counter()
                phases["scan_setup_and_warmup_seconds"] = max(0.0, scan_finished - began - output[3] - output[4])
                return output

            def timed_metrics(*values, **kwargs):
                nonlocal metrics_finished
                began = perf_counter()
                phases["post_optimizer_transfer_and_preparation_seconds"] = began - scan_finished
                output = original_metrics(*values, **kwargs)
                metrics_finished = perf_counter()
                phases["metric_seconds"] = metrics_finished - began
                return output

            with patch.object(dense_module, "_posthoc_metrics", timed_metrics), patch.object(dense_module, "_run_scan_mode", timed_scan):
                result = dense_module.run_qpt_stochastic_frames_jax(data, execution_mode="scan", **common_options(args))
            runner_finished = perf_counter()
            phases["runner_finalize_seconds"] = runner_finished - metrics_finished
            row.update(
                optimizer_seconds=result.total_seconds, optimizer_compile_seconds=result.compile_seconds,
                metric_seconds=phases["metric_seconds"],
                metric_location="CPU NumPy; full measurements and process-matrix eigendecomposition",
                metric_mode="full", metric_compile_included=False,
                jax_version=result.jax_version, jaxlib_version=result.jaxlib_version,
                batch_indices_sha256=hashlib.sha256(result.batch_indices.astype(np.int64).tobytes()).hexdigest(),
                final_measurement_loss=finite_scalar(result.measurement_loss[-1], "measurement loss"),
                final_smoothed_gap=finite_scalar(result.exact_smoothed_gap[-1], "smoothed gap"),
                warmup="native legacy behavior: one complete scan",
                sampling_transfer_seconds=None,
            )
            phases["optimizer_seconds"], phases["optimizer_compile_seconds"] = result.total_seconds, result.compile_seconds
        else:
            result = run_qpt_structured_jax(
                data, chunk_steps=args.chunk_steps, metric_mode=args.metric_mode,
                metric_samples=args.metric_samples, metric_batch_size=args.metric_batch_size,
                metric_seed=args.metric_seed, measurement_backend=args.measurement_backend, **common_options(args),
            )
            runner_finished = perf_counter()
            meta = result.metadata
            row.update(
                optimizer_seconds=meta["optimizer_seconds"], optimizer_compile_seconds=meta["compile_seconds"],
                metric_seconds=meta["metric_seconds"],
                metric_location="selected JAX device; streamed measurement batches and exact TP",
                metric_mode=meta["metric_mode"], metric_compile_included=True, metric_sample_count=meta["metric_samples"],
                jax_version=meta["jax_version"], jaxlib_version=meta["jaxlib_version"],
                batch_symbols_sha256=meta["batch_symbols_sha256"],
                compiled_memory_estimate_bytes=meta["compiled_memory_estimate_bytes"],
                sampling_transfer_seconds=meta["sampling_transfer_seconds"],
                measurement_backend=meta["measurement_backend"],
                final_measurement_loss=finite_scalar(result.measurement_loss[-1], "measurement loss"),
                final_smoothed_gap=finite_scalar(result.smoothed_gap[-1], "smoothed gap"),
                warmup="native structured behavior: one chunk",
            )
            for key in ("setup_seconds", "initial_transfer_seconds", "warmup_seconds",
                        "sampling_transfer_seconds", "metric_seconds", "optimizer_seconds"):
                phases[key] = meta[key]
            phases["optimizer_compile_seconds"] = meta["compile_seconds"]
            phases["runner_other_seconds"] = max(0.0, runner_finished - runner_started - sum(
                phases[key] for key in ("setup_seconds", "initial_transfer_seconds", "warmup_seconds",
                                       "sampling_transfer_seconds", "metric_seconds", "optimizer_seconds",
                                       "optimizer_compile_seconds")
            ))
            row["structured_runner_metadata"] = meta

        row["runner_wall_seconds"] = runner_finished - runner_started
        row["final_tp_violation"] = finite_scalar(result.tp_violation[-1], "TP violation")
        row["final_smoothed_objective"] = finite_scalar(result.smoothed_objective[-1], "smoothed objective")
        fidelity = float(result.process_fidelity_proxy[-1])
        row["final_process_fidelity_proxy"] = fidelity if math.isfinite(fidelity) else None
        if not all(math.isfinite(row[key]) for key in (
            "final_measurement_loss", "final_smoothed_gap", "final_tp_violation", "final_smoothed_objective",
        )) or not np.all(np.isfinite(result.final_factor)):
            raise FloatingPointError("Nonfinite final factor or optimization metrics.")
        row["final_factor_fingerprint"] = array_fingerprint(result.final_factor)
        row["optimizer_steps_per_second"] = args.steps / row["optimizer_seconds"]
        row["memory"]["jax_allocator"] = device_memory_snapshot(selected)
        began = perf_counter()
        # Save the same minimal numeric output on both paths, not dense histories.
        np.save(args.worker_factor, result.final_factor, allow_pickle=False)
        phases["factor_save_seconds"] = perf_counter() - began
        row["final_factor_path"] = str(args.worker_factor)
        if args.data is not None:
            # A structured-only experiment can retain its scalar traces without
            # introducing unequal output work into the legacy matched comparison.
            from paper.experiments.quantum_process_tomography_structured_jax import save_structured_result
            began = perf_counter()
            result_path = args.worker_factor.with_name(args.worker_factor.stem + "_result.npz")
            save_structured_result(result_path, result)
            row["structured_result_path"] = str(result_path)
            row["structured_result_sha256"] = file_sha256(result_path)
            phases["structured_result_save_seconds"] = perf_counter() - began
            began = perf_counter()
            if file_sha256(args.compact_data or args.data) != row["compact_input_sha256"]:
                raise RuntimeError("Compact input changed during the worker run.")
            phases["input_revalidation_seconds"] = perf_counter() - began
            row["compact_input_unchanged"] = True
        row["status"] = "success"
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        if selected is not None:
            row["memory"]["jax_allocator"] = device_memory_snapshot(selected)
    finally:
        row["memory"]["cpu_process_peak_rss_bytes"] = process_peak_rss_bytes()
        row["worker_wall_seconds"] = perf_counter() - SCRIPT_STARTED
        # In-worker overhead not covered by named phases; parent process time also
        # covers interpreter startup, final JSON serialization and shutdown.
        row["worker_unattributed_seconds"] = max(0.0, row["worker_wall_seconds"] - sum(phases.values()))
        write_json(args.worker_report, row)
    print(json.dumps(row, allow_nan=False), flush=True)
    return 0 if row["status"] == "success" else 1


def memory_estimates(data, args):
    """Array accounting only; kept separate from all runtime measurements."""
    complex_bytes = 16 if args.precision == "64" else 8
    real_bytes = complex_bytes // 2
    n, d, m = data.process_dimension, data.d, data.m
    dense_d, dense_b = m * n * n * complex_bytes, n * n * d * d * complex_bytes
    return {
        "kind": "array-size estimates; not measured peak memory",
        "dense_D_device_bytes": dense_d, "dense_B_device_bytes": dense_b,
        "dense_static_device_bytes": dense_d + dense_b + m * real_bytes + d * d * complex_bytes,
        "dense_scan_iterate_history_device_bytes": args.steps * n * args.rank * complex_bytes,
        "dense_host_iterate_history_bytes": args.steps * n * args.rank * complex_bytes,
        "dense_history_note": "one history only; native warmup/history copies can coexist",
        "structured_host_observation_bytes": int(data.observations.nbytes),
        "structured_full_local_bank_device_bytes": (24 * 4 * 4 + 4 * 2 * 2) * complex_bytes,
        "factor_device_bytes": n * args.rank * complex_bytes, "structured_checkpoint_factor_bytes": 0,
        "structured_host_scalar_traces_scale": "O(steps), not O(steps * process_dimension * rank)",
    }


def allocator_peak(row):
    snapshot = row.get("memory", {}).get("jax_allocator") or {}
    value = snapshot.get("stats", {}).get("peak_bytes_in_use")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def sampled_gpu_peak(row):
    sampler = row.get("memory", {}).get("external_sampler") or {}
    value = sampler.get("gpu", {}).get("sampled_peak_process_bytes")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _positive_ratio(first, second):
    if first is None or second is None or second <= 0:
        return None
    return first / second


def source_fingerprint(path):
    stat = Path(path).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def collect_run(args, source_args, backend, purpose, repeat, artifacts, compact, environment):
    """Keep stdout/stderr on disk so neither monitoring nor verbose errors fill pipes."""
    from paper.experiments.qpt_benchmark_resources import ProcessMemorySampler
    stem = f"{purpose}_{backend}_{repeat}"
    worker_report, factor_path = artifacts / (stem + ".json"), artifacts / (stem + ".npy")
    stdout_path, stderr_path = artifacts / (stem + ".stdout.log"), artifacts / (stem + ".stderr.log")
    source_option = ["--data", str(args.data)] if args.data is not None else ["--h5", str(args.h5)]
    command = [sys.executable, str(Path(__file__).resolve())] + source_args + [
        *source_option, "--worker", backend, "--compact-data", str(compact), "--purpose", purpose,
        "--worker-report", str(worker_report), "--worker-factor", str(factor_path),
    ]
    if args.compact_sha256 is not None:
        command.extend(["--compact-sha256", args.compact_sha256])
    started = perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, env=environment)
        sampler = ProcessMemorySampler(
            process.pid, interval_seconds=args.memory_poll_ms / 1000.0, enable_gpu=args.device != "cpu",
        ) if purpose == "memory" else None
        try:
            with sampler if sampler is not None else nullcontext():
                returncode = process.wait()
                process_finished = perf_counter()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    row = json.loads(worker_report.read_text()) if worker_report.exists() else {
        "backend": backend, "purpose": purpose, "status": "failed", "memory": {},
        "error": f"Worker exited {returncode} without a report (see stderr log).",
    }
    row.update(repeat=repeat, process_wall_seconds=process_finished - started, returncode=returncode,
               stdout_path=str(stdout_path), stderr_path=str(stderr_path), worker_report_path=str(worker_report))
    if returncode:
        row["status"] = "failed"
    if sampler is not None:
        row["monitor_teardown_and_report_read_seconds"] = perf_counter() - process_finished
        row["memory"]["external_sampler"] = sampler.summary()
        measured_gpu = row.get("device_platform") == "gpu" and (
            allocator_peak(row) is not None or sampled_gpu_peak(row) is not None
        )
        if args.require_gpu_memory and not measured_gpu:
            message = "Required GPU-memory measurement unavailable: no GPU allocator peak or PID VRAM samples."
            row["memory_requirement_error"] = message
            row.setdefault("error", message)
            row["status"] = "failed"
    write_json(worker_report, row)
    return row


def summarize(report, args):
    import numpy as np
    successes = [row for row in report["runs"] if row["status"] == "success"]
    timings = [row for row in successes if row["purpose"] == "timing"]
    report["timing_complete"] = len(timings) == args.repeats * len(args.backends)
    identities = {(row.get("hostname"), row.get("device_platform"), row.get("device_kind"), row.get("device"))
                  for row in successes}
    report["matched_device_identity"] = len(identities) == 1 if identities else None
    report["medians"] = {
        backend: {
            key: statistics.median(row[key] for row in timings if row["backend"] == backend)
            for key in ("optimizer_seconds", "optimizer_compile_seconds", "metric_seconds",
                        "runner_wall_seconds", "load_seconds", "process_wall_seconds")
        }
        for backend in args.backends if any(row["backend"] == backend for row in timings)
    }
    report["memory_profiles"] = [row for row in report["runs"] if row["purpose"] == "memory"]
    profiles = {row["backend"]: row for row in report["memory_profiles"] if row["status"] == "success"}
    report["memory_ratios"] = {}
    if set(profiles) == {"dense", "structured"}:
        dense, structured = profiles["dense"], profiles["structured"]
        report["memory_ratios"] = {
            "dense_over_structured_cpu_process_peak_rss": _positive_ratio(
                dense["memory"].get("cpu_process_peak_rss_bytes"), structured["memory"].get("cpu_process_peak_rss_bytes")),
            "dense_over_structured_jax_allocator_peak_bytes_in_use": _positive_ratio(allocator_peak(dense), allocator_peak(structured)),
            "dense_over_structured_sampled_process_vram_peak": _positive_ratio(sampled_gpu_peak(dense), sampled_gpu_peak(structured)),
            "note": "One profile per backend; sampled VRAM peaks can miss brief peaks. These are different memory definitions.",
        }
    if report["timing_complete"] and report["matched_device_identity"] and set(report["medians"]) == {"dense", "structured"}:
        dense, structured = report["medians"]["dense"], report["medians"]["structured"]
        report["dense_over_structured_optimizer_ratio"] = _positive_ratio(dense["optimizer_seconds"], structured["optimizer_seconds"])
        report["dense_over_structured_runner_wall_ratio"] = _positive_ratio(dense["runner_wall_seconds"], structured["runner_wall_seconds"])
        report["dense_over_structured_process_wall_ratio"] = _positive_ratio(dense["process_wall_seconds"], structured["process_wall_seconds"])

    # Check every timing repetition, not just whichever factor file was saved last.
    comparisons = []
    for repeat in range(args.repeats):
        pair = {row["backend"]: row for row in timings if row["repeat"] == repeat}
        if set(pair) != {"dense", "structured"}:
            continue
        dense, structured = pair["dense"], pair["structured"]
        first = np.load(dense["final_factor_path"], allow_pickle=False)
        second = np.load(structured["final_factor_path"], allow_pickle=False)
        tolerance = 1e-4 if args.precision == "32" else 1e-8
        item = {"repeat": repeat, "factor_max_abs_difference": float(np.max(np.abs(first - second))),
                "factor_matches": bool(np.allclose(first, second, rtol=tolerance, atol=tolerance))}
        if args.metric_mode == "full":
            for key, name in (("loss", "final_measurement_loss"), ("gap", "final_smoothed_gap")):
                item[f"full_{key}_abs_difference"] = abs(dense[name] - structured[name])
                item[f"full_{key}_matches"] = bool(np.isclose(dense[name], structured[name], rtol=tolerance, atol=tolerance))
        comparisons.append(item)
    report["parity"] = comparisons
    report["dense_comparison_performed"] = bool(comparisons)
    if comparisons:
        report["final_factor_max_abs_difference"] = max(item["factor_max_abs_difference"] for item in comparisons)
        if args.metric_mode == "full":
            for key in ("loss", "gap"):
                report[f"final_full_{key}_abs_difference"] = max(item[f"full_{key}_abs_difference"] for item in comparisons)
        report["parity_passed"] = all(value for item in comparisons for key, value in item.items() if key.endswith("_matches"))
    report["repeatability"] = {}
    for backend in args.backends:
        selected = sorted((row for row in timings if row["backend"] == backend), key=lambda row: row["repeat"])
        fingerprints = [row.get("final_factor_fingerprint") for row in selected]
        valid = all(value is not None for value in fingerprints)
        report["repeatability"][backend] = dict(
            successful_timing_repeats=len(selected), final_factor_fingerprints=fingerprints,
            final_factors_bitwise_equal=(all(value == fingerprints[0] for value in fingerprints[1:])
                                         if valid and len(fingerprints) > 1 else None),
            note="Observed numeric factor hashes for the fixed seeds; not a cross-backend parity or global determinism claim.",
        )
    report["status"] = "complete" if (
        all(row["status"] == "success" for row in report["runs"]) and report.get("parity_passed", True)
        and report["matched_device_identity"] and report["timing_complete"]
    ) else "failed"


def main(argv=None):
    source_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser().parse_args(source_args)
    if args.worker:
        return run_worker(args)
    if args.repeats < 1 or args.steps < 1:
        raise ValueError("--repeats and --steps must be positive.")
    if len(args.backends) != len(set(args.backends)):
        raise ValueError("--backends must not contain duplicates.")
    if not 0 < args.memory_poll_ms < float("inf"):
        raise ValueError("--memory-poll-ms must be finite and positive.")
    if args.require_gpu_memory and not args.profile_memory:
        raise ValueError("--require-gpu-memory requires --profile-memory.")
    if not math.isfinite(args.max_data_mib) or args.max_data_mib <= 0:
        raise ValueError("--max-data-mib must be finite and positive.")
    if os.path.lexists(args.save):
        raise FileExistsError(f"Refusing to overwrite existing report: {args.save}")
    artifacts = (args.artifacts_dir or args.save.parent / (args.save.stem + "_artifacts")).resolve()
    artifacts.mkdir(parents=True, exist_ok=False)
    if args.h5 is not None:
        args.h5 = args.h5.resolve()
    if args.data is not None:
        args.data = args.data.resolve()
    environment = configured_worker_environment(args)
    report = dict(
        schema_version=2, status="running", h5=str(args.h5) if args.h5 is not None else None,
        data=str(args.data) if args.data is not None else None,
        input_mode="compact" if args.data is not None else "hdf5", artifacts_dir=str(artifacts),
        configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                       if key not in {"worker", "compact_data", "compact_sha256", "worker_report", "worker_factor", "purpose"}},
        allocator_environment={key: environment.get(key) for key in ALLOCATOR_ENVIRONMENT_KEYS},
        scheduler_environment={key: os.environ.get(key) for key in SCHEDULER_ENVIRONMENT_KEYS},
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_source_sha256={name: file_sha256(ROOT / name) for name in NUMERICAL_SOURCE_FILES},
        runs=[], observations_preserved=False,
        interpretation=[
            "Timing medians use fresh unmonitored workers; separate memory profiles are excluded.",
            "Device optimizer calls are synchronized; optimizer compilation and native warmup are excluded.",
            "Structured metric time includes first-call metric JIT; dense metrics use NumPy and include eigendecomposition.",
            "Native warmup differs: dense discards one full scan; structured discards one chunk. Phase timings disclose this.",
            "The same explicit allocator policy is used for every worker before JAX import; persistent JAX compilation caching is disabled.",
            "CPU process peak RSS is an OS lifetime highwater; allocator peaks are JAX allocator counters, not total driver VRAM.",
            "Worker CPU RSS excludes the coordinating parent. Parent loading/conversion peaks are separate lifetime highwaters, not simultaneous total RAM.",
            "nvidia-smi process VRAM is externally sampled and may miss brief peaks; unavailable values are null, never zero placeholders.",
            "Array and compiler memory estimates remain separate from runtime counters and sampled measurements.",
            "HDF5 mode pays conversion once; compact mode reads the supplied NPZ directly and never reconstructs dense operators.",
            "Per-worker process time includes imports, input checks, device initialization, load, run, save and shutdown.",
            "No result plots are generated. Compact-only runs also save bounded structured result archives with scalar traces and fidelity.",
            "Full metrics use all observations; sampled metric values/gaps are not comparable to dense full-data metrics.",
            "Structured-only completion is not a dense parity claim; supplied compact verification metadata are provenance, not a new HDF5 audit.",
        ],
    )
    write_json(args.save, report)
    try:
        from paper.experiments.qpt_structured_data import StructuredQPTData
        from paper.experiments.qpt_benchmark_resources import process_peak_rss_bytes
        if args.data is not None:
            compact = args.data
            report["compact_input"] = guarded_compact_input(compact, args.max_data_mib)
            args.compact_sha256 = report["compact_input"]["sha256"]
            started = perf_counter()
            data = StructuredQPTData.load_npz(compact)
            report["compact_parent_load_seconds"] = perf_counter() - started
            if data.observation_mode != "stored":
                raise ValueError("Compact benchmark requires stored observations, including their fixed noise realization.")
            report.update(conversion_seconds=None, compact_save_seconds=None, hdf5_accessed=False,
                          verification=data.metadata.get("verification", "not_recorded"),
                          verification_note="Recorded compact provenance only; no HDF5 conversion or verification performed.")

            def check_input_unchanged():
                if file_sha256(compact) != args.compact_sha256:
                    raise RuntimeError("Compact source changed during the benchmark.")
        else:
            fingerprint = source_fingerprint(args.h5)
            started = perf_counter()
            data = StructuredQPTData.from_hdf5(
                args.h5, verification="full", chunk_size=args.metric_batch_size, rtol=args.rtol, atol=args.atol,
            )
            report["conversion_seconds"] = perf_counter() - started
            compact = artifacts / "data.npz"
            started = perf_counter()
            data.save_npz(compact)
            report.update(compact_save_seconds=perf_counter() - started, verification="full", hdf5_accessed=True,
                          source_fingerprint=fingerprint, converter_parent_peak_rss_bytes=process_peak_rss_bytes())

            def check_input_unchanged():
                if source_fingerprint(args.h5) != fingerprint:
                    raise RuntimeError("HDF5 source changed during the benchmark; comparison would not be matched.")
        report.update(n_qubits=data.n_qubits, measurement_count=data.m, process_dimension=data.process_dimension,
                      observations_preserved=True, observation_mode=data.observation_mode, source_metadata=data.metadata,
                      observations_fingerprint=array_fingerprint(data.observations),
                      truth_factor_fingerprint=array_fingerprint(data.truth_factor) if data.truth_factor is not None else None,
                      memory_estimates=memory_estimates(data, args))
        if args.data is not None:
            for key, fingerprint in (("observations", report["observations_fingerprint"]),
                                     ("truth_factor", report["truth_factor_fingerprint"])):
                recorded_hash = data.metadata.get(key + "_sha256")
                if recorded_hash is not None and (fingerprint is None or recorded_hash != fingerprint["sha256"]):
                    raise ValueError(f"Compact {key} does not match its recorded numeric SHA-256.")
            report["compact_parent_peak_rss_bytes"] = process_peak_rss_bytes()
        check_input_unchanged()
        # The parent needs only provenance and scalar shape accounting now.
        # Each worker loads its own inputs; do not retain a redundant b vector.
        del data
        write_json(args.save, report)
        schedule = [
            (backend, "timing", repeat)
            for repeat in range(args.repeats)
            for backend in (args.backends if repeat % 2 == 0 else list(reversed(args.backends)))
        ]
        if args.profile_memory:
            schedule.extend((backend, "memory", 0) for backend in args.backends)
        for backend, purpose, repeat in schedule:
            check_input_unchanged()
            row = collect_run(args, source_args, backend, purpose, repeat, artifacts, compact, environment)
            report["runs"].append(row)
            write_json(args.save, report)
            if row["status"] == "success":
                print(f"{purpose} {backend} repeat {repeat+1}: optimizer={row['optimizer_seconds']:.4f}s, "
                      f"runner={row['runner_wall_seconds']:.4f}s, process={row['process_wall_seconds']:.4f}s", flush=True)
            else:
                print(f"{purpose} {backend} failed: {row.get('error', 'see worker stderr')}", flush=True)
        check_input_unchanged()
        report["inputs_unchanged"] = True
        report["numerical_sources_unchanged"] = all(
            file_sha256(ROOT / name) == digest for name, digest in report["numerical_source_sha256"].items()
        )
        if not report["numerical_sources_unchanged"]:
            raise RuntimeError("Benchmark numerical sources changed during the run.")
        summarize(report, args)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        report["benchmark_wall_seconds"] = perf_counter() - SCRIPT_STARTED
        write_json(args.save, report)
    print(f"Saved {args.save} ({report['status']}); worker logs: {artifacts}", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
