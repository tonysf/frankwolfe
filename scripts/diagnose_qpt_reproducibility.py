#!/usr/bin/env python3
"""Replay one unchanged QPT scan executable, then compare fixed-input operators.

This is a diagnostic, not a performance benchmark. Run inside an existing
allocation. It never connects to a cluster, submits a job, or edits an optimizer.
Numerical differences are findings; a nonzero exit means the diagnostic failed
to execute or its input/device controls did not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_ENABLE_X64",
    "XLA_FLAGS", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
    "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION",
    "JAX_ENABLE_COMPILATION_CACHE", "JAX_COMPILATION_CACHE_DIR",
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
)
NUMERICAL_SOURCE_FILES = (
    "scripts/diagnose_qpt_reproducibility.py",
    "paper/experiments/quantum_process_tomography_structured_jax.py",
    "paper/experiments/qpt_structured_operators.py",
    "paper/experiments/quantum_process_tomography.py",
    "paper/experiments/quantum_process_tomography_jax.py",
    "paper/experiments/qpt_reproducibility_probes.py",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--benchmark-report", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing directories are never reused.")
    result.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    result.add_argument("--processes", type=int, default=3)
    result.add_argument("--replays", type=int, default=3)
    result.add_argument("--rtol", type=float, default=1e-10,
                        help="Diagnostic comparison tolerance, not an optimizer change.")
    result.add_argument("--atol", type=float, default=1e-12)
    result.add_argument("--max-plan-mib", type=float, default=64)
    result.add_argument("--max-trace-mib", type=float, default=128)
    result.add_argument("--max-reference-mib", type=float, default=128)
    result.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    result.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    return result


def write_json(path, value):
    path = Path(path)
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
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def input_fingerprint(arrays):
    descriptions = {name: array_fingerprint(value) for name, value in sorted(arrays.items())}
    digest = hashlib.sha256(json.dumps(descriptions, sort_keys=True).encode()).hexdigest()
    return descriptions, digest


def compare_arrays(reference, actual, rtol, atol):
    import numpy as np
    reference, actual = np.asarray(reference), np.asarray(actual)
    same_shape = reference.shape == actual.shape
    finite = bool(np.all(np.isfinite(reference)) and np.all(np.isfinite(actual)))
    result = dict(same_shape=same_shape, finite=finite, exact=False, allclose=False,
                  max_abs_difference=None, relative_l2_difference=None)
    if not same_shape or not finite:
        return result
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        difference = actual - reference
        norm = float(np.linalg.norm(reference.ravel()))
        error = float(np.linalg.norm(difference.ravel()))
        relative = error / norm if norm else (0.0 if error == 0 else None)
        maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
        matches = bool(np.allclose(reference, actual, rtol=rtol, atol=atol))
    result.update(exact=bool(np.array_equal(reference, actual)), allclose=matches,
                  max_abs_difference=maximum if math.isfinite(maximum) else None,
                  relative_l2_difference=relative if relative is None or math.isfinite(relative) else None)
    return result


def compare_trajectories(reference, actual, boundaries, rtol, atol):
    import numpy as np
    comparisons = {name: compare_arrays(reference[name], actual[name], rtol, atol)
                   for name in ("factors", "momentum", "gaps")}
    state_boundaries = []
    for index, step in enumerate(boundaries):
        row = {name: compare_arrays(reference[name][index], actual[name][index], rtol, atol)
               for name in ("factors", "momentum")}
        state_boundaries.append(dict(step=int(step), **row))
    first = next((row["step"] for row in state_boundaries
                  if not all(row[name]["exact"] for name in ("factors", "momentum"))), None)
    unequal_gaps = np.flatnonzero(reference["gaps"] != actual["gaps"])
    return dict(comparisons=comparisons, boundaries=state_boundaries,
                exact=all(value["exact"] for value in comparisons.values()),
                allclose=all(value["allclose"] for value in comparisons.values()),
                first_different_boundary=first,
                first_gap_difference_iteration=int(unequal_gaps[0]) if unequal_gaps.size else None)


def _positive_config_integer(configuration, name):
    value = configuration.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Benchmark configuration {name!r} must be a positive integer.")
    return value


def freeze_inputs(args):
    """Reproduce the host sampling plan once, then save every numerical input."""
    import numpy as np
    from paper.experiments.qpt_structured_data import StructuredQPTData
    from paper.experiments.qpt_structured_operators import rank_one_measurement_vectors
    from paper.experiments.quantum_process_tomography import (
        checkpoint_iterations, make_factor_initial_point, unpack_factor,
    )
    from paper.experiments.quantum_process_tomography_jax import _materialize_schedules
    from scripts.benchmark_qpt_structured import common_options

    source_path = args.benchmark_report.resolve()
    source_hash = file_sha256(source_path)
    source = json.loads(source_path.read_text())
    if source.get("observations_preserved") is not True or source.get("verification") != "full":
        raise ValueError("Source report must record observations_preserved=true and verification=full.")
    configuration = dict(source["configuration"])
    steps, rank, batch, chunk = (_positive_config_integer(configuration, key)
                                for key in ("steps", "rank", "batch_size", "chunk_steps"))
    if steps > np.iinfo(np.int32).max:
        raise ValueError("Step count exceeds the existing scan's int32 iteration range.")
    if configuration.get("precision") not in ("32", "64"):
        raise ValueError("Source precision must be '32' or '64'.")
    requested_backend = configuration.get("measurement_backend", "auto")
    if requested_backend not in ("auto", "tensor", "rank-one"):
        raise ValueError("Invalid source measurement_backend.")
    compact = Path(source["artifacts_dir"]) / "data.npz"
    if not compact.is_absolute():
        compact = source_path.parent / compact
    compact = compact.resolve()
    compact_hash = file_sha256(compact)
    data = StructuredQPTData.load_npz(compact)
    if data.observation_mode != "stored":
        raise ValueError("This diagnostic requires the original stored noisy observations.")

    complex_dtype = np.complex128 if configuration["precision"] == "64" else np.complex64
    real_dtype = np.float64 if configuration["precision"] == "64" else np.float32
    plan_bytes = steps * batch * (data.n_qubits * 4 + np.dtype(real_dtype).itemsize)
    plan_bytes += 3 * steps * np.dtype(real_dtype).itemsize
    if plan_bytes > args.max_plan_mib * 2**20:
        raise ValueError("Frozen sampling plan exceeds --max-plan-mib; no GPU work started.")
    checkpoints = checkpoint_iterations(steps, configuration["metrics_every"])
    lengths, completed = [], 0
    for checkpoint in checkpoints[1:]:
        while completed < checkpoint:
            length = min(chunk, int(checkpoint) - completed)
            lengths.append(length)
            completed += length
    if len(set(lengths)) != 1:
        raise ValueError("Source uses multiple scan lengths; this diagnostic requires one unchanged chunk shape.")
    scan_length = lengths[0]
    trace_bytes = (args.replays * (len(lengths) + 1) * data.process_dimension * rank
                   * np.dtype(complex_dtype).itemsize * 2 + args.replays * steps * np.dtype(real_dtype).itemsize)
    if trace_bytes > args.max_trace_mib * 2**20:
        raise ValueError("Saved trajectories exceed --max-trace-mib; no GPU work started.")

    raw_factor = unpack_factor(make_factor_initial_point(data, rank, configuration["initialization_seed"]),
                               data.process_dimension, rank)
    if not np.isfinite(configuration["tau"]) or configuration["tau"] <= 0:
        raise ValueError("Source tau must be finite and positive.")
    if np.linalg.norm(raw_factor, ord=2) > configuration["tau"] + 1e-10:
        raise ValueError("Source initial factor is not feasible.")
    initial_hash = array_fingerprint(raw_factor)["sha256"]
    rng = np.random.default_rng(configuration["sampling_seed"])
    symbols = np.concatenate([data.sample_symbols(rng, length * batch) for length in lengths])
    symbols = symbols.astype(np.int32).reshape(steps, batch, data.n_qubits)
    sampling_hash = array_fingerprint(symbols)["sha256"]
    observations = data.observations_for_symbols(symbols.reshape(-1, data.n_qubits)).reshape(steps, batch)
    vectors = rank_one_measurement_vectors(data.local_measurements)
    if requested_backend == "rank-one" and vectors is None:
        raise ValueError("Source rank-one backend is incompatible with the compact operator bank.")
    use_rank_one = requested_backend != "tensor" and vectors is not None
    effective_backend = "rank-one" if use_rank_one else "tensor"
    for row in source.get("runs", []):
        metadata = row.get("structured_runner_metadata") or {}
        for key, expected in (("initial_factor_sha256", initial_hash),
                              ("batch_symbols_sha256", sampling_hash),
                              ("measurement_backend", effective_backend)):
            if metadata.get(key) is not None and metadata[key] != expected:
                raise ValueError(f"Frozen {key} differs from the source run; use the original configuration/environment.")
    options = common_options(argparse.Namespace(**configuration))
    rho, beta, gamma = _materialize_schedules(
        steps, 10.0, options["rho_schedule"], options["smoothing_schedule"], options["step_size_schedule"],
    )
    factor = np.asarray(raw_factor, dtype=complex_dtype)
    arrays = dict(
        factor0=factor, momentum0=np.zeros_like(factor), symbols=symbols,
        observations=np.asarray(observations, dtype=real_dtype),
        rho=np.asarray(rho, dtype=real_dtype), beta=np.asarray(beta, dtype=real_dtype),
        gamma=np.asarray(gamma, dtype=real_dtype),
        bank=np.asarray(vectors if use_rank_one else data.local_measurements, dtype=complex_dtype),
        basis=np.asarray(data.local_basis, dtype=complex_dtype),
        local_measurements=np.asarray(data.local_measurements, dtype=complex_dtype),
        truth=np.asarray(data.truth_factor if data.truth_factor is not None
                         else np.empty((data.process_dimension, 0)), dtype=complex_dtype),
        boundaries=np.asarray([0] + list(np.cumsum(lengths)), dtype=np.int64),
    )
    descriptions, digest = input_fingerprint(arrays)
    bundle = args.output_dir / "input_bundle.npz"
    np.savez(bundle, **arrays)
    if file_sha256(source_path) != source_hash or file_sha256(compact) != compact_hash:
        raise RuntimeError("Source report or compact data changed during freezing.")
    return dict(
        configuration=configuration, measurement_backend=effective_backend,
        scan_chunk_steps=scan_length, source=dict(report=str(source_path), report_sha256=source_hash,
        compact_data=str(compact), compact_sha256=compact_hash, benchmark_status=source.get("status")),
        frozen_inputs=dict(arrays=descriptions, input_sha256=digest, bundle_sha256=file_sha256(bundle),
                           initial_factor_sha256=initial_hash, batch_symbols_sha256=sampling_hash),
        controls=dict(replays=args.replays, device=args.device, rtol=args.rtol, atol=args.atol,
                      max_reference_bytes=int(args.max_reference_mib * 2**20)),
        numerical_source_sha256={name: file_sha256(ROOT / name) for name in NUMERICAL_SOURCE_FILES},
        storage=dict(plan_array_bytes=plan_bytes, per_worker_trace_array_bytes=trace_bytes,
                     note="Diagnostic freezes the complete plan and saves chunk states; not a memory benchmark."),
    )


def _worker_environment(configuration):
    from scripts.benchmark_qpt_structured import configured_worker_environment
    environment = configured_worker_environment(argparse.Namespace(allocator=configuration.get("allocator", "grow")))
    # Preserve inherited XLA/CUBLAS controls and GPU binding; report them explicitly.
    return environment


def run_worker(args):
    started = perf_counter()
    directory = args.output_dir / f"worker_{args.worker:03d}"
    report_path = directory / "report.json"
    row = dict(status="running", worker=args.worker, pid=os.getpid(), hostname=socket.gethostname(),
               environment={name: os.environ.get(name) for name in ENVIRONMENT_KEYS},
               replays_completed=0, scan_compilation_count=0, phases={})
    write_json(report_path, row)
    try:
        import numpy as np
        from paper.experiments.quantum_process_tomography_jax import (
            _require_jax, _configure_jax_precision, select_jax_device,
        )
        from paper.experiments.quantum_process_tomography_structured_jax import _build_scan

        manifest_bytes = (args.output_dir / "input_manifest.json").read_bytes()
        row["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        if row["manifest_sha256"] != args.manifest_sha256:
            raise RuntimeError("Frozen manifest/scalar controls changed before the worker started.")
        manifest = json.loads(manifest_bytes)
        row["numerical_source_sha256"] = {name: file_sha256(ROOT / name) for name in NUMERICAL_SOURCE_FILES}
        if row["numerical_source_sha256"] != manifest["numerical_source_sha256"]:
            raise RuntimeError("Numerical source files changed between workers.")
        configuration = manifest["configuration"]
        controls = manifest["controls"]
        bundle = args.output_dir / "input_bundle.npz"
        if file_sha256(bundle) != manifest["frozen_inputs"]["bundle_sha256"]:
            raise RuntimeError("Frozen input bundle file changed.")
        with np.load(bundle, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        _, digest = input_fingerprint(arrays)
        row["input_sha256"] = digest
        if digest != manifest["frozen_inputs"]["input_sha256"]:
            raise RuntimeError("Decoded input fingerprints do not match the frozen manifest.")
        for value in arrays.values():
            value.setflags(write=False)
        row["phases"]["imports_and_input_load_seconds"] = perf_counter() - started

        began = perf_counter()
        jax = _require_jax()
        _configure_jax_precision(jax, configuration["precision"])
        import jax.numpy as jnp
        import jaxlib
        selected = select_jax_device(args.device, configuration.get("device_index", 0))
        row.update(device=str(selected), device_kind=selected.device_kind, device_platform=selected.platform,
                   jax_version=jax.__version__, jaxlib_version=jaxlib.__version__, numpy_version=np.__version__,
                   measurement_backend=manifest["measurement_backend"])
        put = lambda value: jax.device_put(value, selected)
        initial_state = (put(arrays["factor0"]), put(arrays["momentum0"]))
        bank, basis, truth = (put(arrays[name]) for name in ("bank", "basis", "truth"))
        boundaries = arrays["boundaries"]
        chunks = []
        for first, last in zip(boundaries[:-1], boundaries[1:]):
            chunks.append((put(np.asarray(first, dtype=np.int32)),) + tuple(
                put(np.ascontiguousarray(arrays[name][first:last]))
                for name in ("symbols", "observations", "rho", "beta", "gamma")
            ) + (bank, basis, truth))
        jax.block_until_ready((initial_state, chunks))
        row["phases"]["device_initialization_and_frozen_transfer_seconds"] = perf_counter() - began

        began = perf_counter()
        scan = _build_scan(jax, jnp, float(configuration["tau"]), False,
                           manifest["measurement_backend"] == "rank-one")
        executable = scan.lower(initial_state, *chunks[0]).compile()
        row["scan_compilation_count"] = 1
        row["phases"]["scan_compile_seconds"] = perf_counter() - began
        try:
            executable_text = executable.as_text()
            if not isinstance(executable_text, str):
                raise ValueError("Executable text unavailable.")
            text_path = directory / "compiled_scan.txt"
            text_path.write_text(executable_text, encoding="utf-8")
            row["executable"] = dict(status="available", text_sha256=file_sha256(text_path), path=str(text_path),
                note="Unnormalized executable IR text, not a GPU-binary hash; metadata differences can change it.")
        except Exception as error:
            row["executable"] = dict(status="unavailable", error=str(error))

        began = perf_counter()
        warm_state, warm_gaps = executable(initial_state, *chunks[0])
        jax.block_until_ready((warm_state, warm_gaps))
        warm_host = [np.array(jax.device_get(value), copy=True) for value in (*warm_state, warm_gaps)]
        del warm_state, warm_gaps
        row["phases"]["discarded_first_chunk_warmup_seconds"] = perf_counter() - began
        factor_shape = arrays["factor0"].shape
        factors = np.empty((args.replays, len(boundaries)) + factor_shape, dtype=arrays["factor0"].dtype)
        momentum = np.empty_like(factors)
        gaps = np.empty((args.replays, configuration["steps"]), dtype=arrays["rho"].dtype)
        trajectory_path = directory / "trajectories.npz"
        row["trajectory_path"] = str(trajectory_path)
        row["replay_seconds"] = []
        for replay in range(args.replays):
            began = perf_counter()
            state = initial_state
            factors[replay, 0], momentum[replay, 0] = arrays["factor0"], arrays["momentum0"]
            for index, chunk_inputs in enumerate(chunks):
                state, scalar_gaps = executable(state, *chunk_inputs)
                jax.block_until_ready((state, scalar_gaps))
                factors[replay, index + 1], momentum[replay, index + 1] = (
                    np.array(jax.device_get(value), copy=True) for value in state
                )
                first, last = boundaries[index:index + 2]
                gaps[replay, first:last] = np.asarray(jax.device_get(scalar_gaps))
                if not all(np.all(np.isfinite(value)) for value in (
                    factors[replay, index + 1], momentum[replay, index + 1], gaps[replay, first:last],
                )):
                    raise FloatingPointError(f"Nonfinite replay {replay} at chunk ending {last}.")
            row["replay_seconds"].append(perf_counter() - began)
            row["replays_completed"] = replay + 1
            # Publish only completed replays; never serialize uninitialized slots.
            np.savez(trajectory_path, factors=factors[:replay + 1], momentum=momentum[:replay + 1],
                     gaps=gaps[:replay + 1], boundaries=boundaries, warmup_factor=warm_host[0],
                     warmup_momentum=warm_host[1], warmup_gaps=warm_host[2])
            write_json(report_path, row)
            print(f"worker {args.worker}: replay {replay + 1}/{args.replays} complete", flush=True)

        row["initial_device_state_unchanged"] = all(np.array_equal(jax.device_get(value), arrays[name])
            for value, name in zip(initial_state, ("factor0", "momentum0")))
        if not row["initial_device_state_unchanged"]:
            raise RuntimeError("Initial device state changed during replay.")
        first = dict(factors=factors[0], momentum=momentum[0], gaps=gaps[0])
        row["same_executable_comparisons"] = [dict(replay=replay, **compare_trajectories(
            first, dict(factors=factors[replay], momentum=momentum[replay], gaps=gaps[replay]),
            boundaries, args.rtol, args.atol,
        )) for replay in range(1, args.replays)]
        row["warmup_vs_first_chunk"] = {name: compare_arrays(a, b, args.rtol, args.atol)
            for name, a, b in zip(("factor", "momentum", "gaps"), warm_host,
                                 (factors[0, 1], momentum[0, 1], gaps[0, :manifest["scan_chunk_steps"]]))}

        # These extra kernels are deliberately compiled only AFTER all replays.
        from paper.experiments.qpt_reproducibility_probes import run_operator_probes
        began = perf_counter()
        probe_indices = sorted({0, (len(boundaries) - 1) // 2})
        row["operator_probes"] = []
        for index in probe_indices:
            iteration = int(boundaries[index])
            probe = run_operator_probes(
                jax, jnp, selected, factor=factors[0, index], previous=momentum[0, index],
                symbols=arrays["symbols"][iteration], observations=arrays["observations"][iteration],
                local_measurements=arrays["local_measurements"], local_basis=arrays["basis"],
                selected_bank=arrays["bank"], rho=float(arrays["rho"][iteration]),
                beta=float(arrays["beta"][iteration]), gamma=float(arrays["gamma"][iteration]),
                tau=float(configuration["tau"]), measurement_backend=manifest["measurement_backend"],
                iteration=iteration, repeats=args.replays, rtol=args.rtol, atol=args.atol,
                max_reference_bytes=controls["max_reference_bytes"],
            )
            row["operator_probes"].append(probe)
            write_json(report_path, row)
        row["phases"]["post_replay_operator_probes_seconds"] = perf_counter() - began
        row["status"] = "complete"
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        row["worker_seconds"] = perf_counter() - started
        write_json(report_path, row)
    return 0 if row["status"] == "complete" else 1


def summarize(report, args):
    import numpy as np
    workers = report["workers"]
    complete = [row for row in workers if row["status"] == "complete"]
    expected_hash = report["frozen_inputs"]["input_sha256"]
    inputs_match = bool(complete) and all(
        row.get("input_sha256") == expected_hash and row.get("manifest_sha256") == report["manifest_sha256"]
        for row in complete
    )
    identities = {(row.get("hostname"), row.get("device"), row.get("device_kind"), row.get("device_platform"),
                   row.get("jax_version"), row.get("jaxlib_version"), row.get("numpy_version")) for row in complete}
    devices_match = len(identities) == 1
    report["across_process_comparisons"] = []
    if len(complete) > 1:
        with np.load(complete[0]["trajectory_path"], allow_pickle=False) as archive:
            reference = {name: archive[name][0] for name in ("factors", "momentum", "gaps")}
            boundaries = archive["boundaries"]
        for row in complete[1:]:
            with np.load(row["trajectory_path"], allow_pickle=False) as archive:
                actual = {name: archive[name][0] for name in reference}
                if not np.array_equal(boundaries, archive["boundaries"]):
                    raise RuntimeError("Worker chunk boundaries differ.")
            report["across_process_comparisons"].append(dict(
                reference_worker=complete[0]["worker"], worker=row["worker"],
                **compare_trajectories(reference, actual, boundaries, args.rtol, args.atol),
            ))
    within = [item for row in complete for item in row["same_executable_comparisons"]]
    across = report["across_process_comparisons"]
    reference_states = [probe["cpu_reference"]["status"] for row in complete for probe in row["operator_probes"]]
    reference_status = ("mismatch" if "mismatch" in reference_states else
                        "validated" if reference_states and all(s == "validated" for s in reference_states) else
                        "incomplete")
    report["findings"] = dict(
        inputs_match=inputs_match, devices_match=devices_match,
        same_executable_exact=all(item["exact"] for item in within) if within else None,
        same_executable_allclose=all(item["allclose"] for item in within) if within else None,
        cross_process_exact=all(item["exact"] for item in across) if across else None,
        cross_process_allclose=all(item["allclose"] for item in across) if across else None,
        cpu_reference_status=reference_status,
        operator_probe_replays_exact=all(probe["replay_exact"] for row in complete
                                         for probe in row["operator_probes"]) if reference_states else None,
    )
    report["status"] = "complete" if (len(complete) == args.processes and inputs_match and devices_match) else "failed"


def main(argv=None):
    args = parser().parse_args(argv)
    if args.processes < 1 or args.replays < 2:
        parser().error("--processes must be >= 1 and --replays must be >= 2.")
    for key in ("rtol", "atol", "max_plan_mib", "max_trace_mib", "max_reference_mib"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key.startswith("max_") and value == 0):
            parser().error(f"--{key.replace('_', '-')} must be finite and {'positive' if key.startswith('max_') else 'nonnegative'}.")
    args.output_dir = args.output_dir.resolve()
    if args.worker is not None:
        return run_worker(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    report = dict(schema_version=1, status="running", workers=[],
                  diagnostic_script_sha256=file_sha256(__file__),
                  interpretation=[
                      "Numerical differences are diagnostic findings, not execution failure or proof of an algebraic bug.",
                      "Each worker compiles the existing scan once; all replays reset both state components and reuse inputs.",
                      "Only one full-plan replay per worker is used for cross-process comparison; within-worker checks are separate.",
                      "Stable within-process but varying across processes suggests compile/process-dependent behavior, not proof of autotuning.",
                      "The frozen bank also controls host eigendecomposition differences; original per-process bank variability is not reproduced.",
                      "Freezing all inputs and collecting chunk states changes memory layout and timing; a negative test does not exclude the original symptom.",
                      "CPU operator probes run after replays in separate diagnostic kernels, not the production scan's fused executable.",
                      "Gradient agreement applies only at the tested factors and batches; it is not a global mathematical proof.",
                      "IR text hashes are not GPU-binary hashes. Timing here includes diagnostics and must not replace benchmark medians.",
                  ])
    report_path = args.output_dir / "report.json"
    write_json(report_path, report)
    try:
        manifest = freeze_inputs(args)
        write_json(args.output_dir / "input_manifest.json", manifest)
        report.update(manifest)
        report["manifest_sha256"] = file_sha256(args.output_dir / "input_manifest.json")
        environment = _worker_environment(manifest["configuration"])
        report["environment"] = {name: environment.get(name) for name in ENVIRONMENT_KEYS}
        write_json(report_path, report)
        for worker in range(args.processes):
            directory = args.output_dir / f"worker_{worker:03d}"
            directory.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--benchmark-report", str(args.benchmark_report.resolve()),
                       "--output-dir", str(args.output_dir), "--worker", str(worker),
                       "--manifest-sha256", report["manifest_sha256"],
                       "--device", args.device, "--processes", str(args.processes), "--replays", str(args.replays),
                       "--rtol", str(args.rtol), "--atol", str(args.atol)]
            with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                try:
                    returncode = process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
            worker_report = directory / "report.json"
            row = json.loads(worker_report.read_text()) if worker_report.exists() else dict(
                worker=worker, status="failed", error="Worker exited without a report.")
            row.update(returncode=returncode, stdout_path=str(directory / "stdout.log"),
                       stderr_path=str(directory / "stderr.log"), worker_report_path=str(worker_report))
            if returncode:
                row["status"] = "failed"
            report["workers"].append(row)
            write_json(report_path, report)
            print(f"worker {worker + 1}/{args.processes}: {row['status']}" +
                  (f" ({row['error']})" if row.get("error") else ""), flush=True)
        summarize(report, args)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        report["diagnostic_seconds"] = perf_counter() - started
        write_json(report_path, report)
    print(json.dumps({"status": report["status"], "findings": report.get("findings"),
                      "report": str(report_path)}, indent=2), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
