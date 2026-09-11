#!/usr/bin/env python3
"""Cross-evaluate saved QPT factors on CPU, without rerunning optimization.

Inputs are read-only; outputs require a new directory. Numerical disagreements
are findings, not execution failures. On a cluster, use a CPU allocation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
from time import perf_counter
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_qpt_reproducibility import (
    array_fingerprint, compare_arrays, file_sha256, write_json,
)


SOURCE_FILES = (
    "scripts/diagnose_qpt_same_factor.py",
    "scripts/diagnose_qpt_reproducibility.py",
    "paper/experiments/qpt_same_factor.py",
    "paper/experiments/qpt_structured_data.py",
    "paper/experiments/qpt_structured_operators.py",
    "paper/experiments/quantum_process_tomography.py",
)
SCALAR_METRICS = ("measurement_loss", "tp_violation", "smoothed_objective", "smoothed_gap")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--benchmark-report", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten.")
    result.add_argument("--artifacts-dir", type=Path,
                        help="Explicit relocation of data.npz and timing_BACKEND_REPEAT.npy files.")
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--rtol", type=float, default=1e-10)
    result.add_argument("--atol", type=float, default=1e-12)
    result.add_argument("--max-reference-mib", type=float, default=128)
    result.add_argument("--max-input-mib", type=float, default=128)
    result.add_argument("--max-measurements", type=int, default=1_000_000)
    return result


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def resolve_recorded_path(path, source_report):
    result = Path(path)
    return (result if result.is_absolute() else source_report.parent / result).resolve()


def output_summary(value):
    import numpy as np
    value = np.asarray(value)
    result = array_fingerprint(value)
    result["finite"] = bool(np.all(np.isfinite(value)))
    if value.ndim == 0 and not np.iscomplexobj(value):
        scalar = float(value)
        result["value"] = scalar if math.isfinite(scalar) else None
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            norm = float(np.linalg.norm(value.ravel()))
        result["l2_norm"] = norm if math.isfinite(norm) else None
    return result


def compare_factors(reference, actual, *, rtol, atol, max_reference_bytes):
    """Keep raw trajectory differences distinct from factor gauge freedom."""
    import numpy as np
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError("Factor comparison requires matching matrix shapes.")
    dimension, rank = reference.shape
    estimate = 8 * 16 * (dimension**2 + dimension * rank + rank**2)
    if estimate > max_reference_bytes:
        raise ValueError("Process-matrix comparison exceeds --max-reference-mib.")
    left, singular_values, right_h = np.linalg.svd(actual.conj().T @ reference, full_matrices=False)
    aligned = actual @ (left @ right_h)
    reference_process = reference @ reference.conj().T
    actual_process = actual @ actual.conj().T
    return dict(
        raw_factor=compare_arrays(reference, actual, rtol, atol),
        aligned_factor=compare_arrays(reference, aligned, rtol, atol),
        process_matrix=compare_arrays(reference_process, actual_process, rtol, atol),
        alignment="right-unitary Procrustes; global phase for rank one",
        overlap_singular_values=[float(value) for value in singular_values],
        note="Gauge/process agreement does not establish identical optimization trajectories.",
    )


def prepare_inputs(args):
    import numpy as np
    from paper.experiments.qpt_structured_data import StructuredQPTData
    from paper.experiments.quantum_process_tomography import PowerSchedule
    from paper.experiments.qpt_same_factor import reference_allocation_estimate

    source_path = args.benchmark_report.resolve()
    input_cap = int(args.max_input_mib * 2**20)
    used_bytes = 0
    fingerprints = {}

    def register(path, *, npz=False):
        nonlocal used_bytes
        path = path.resolve()
        key = str(path)
        if key in fingerprints:
            return
        size = path.stat().st_size
        used_bytes += size
        if used_bytes > input_cap:
            raise ValueError("Input files exceed --max-input-mib.")
        if npz:
            with zipfile.ZipFile(path) as archive:
                used_bytes += sum(item.file_size for item in archive.infolist())
            if used_bytes > input_cap:
                raise ValueError("Uncompressed compact data exceed --max-input-mib.")
        fingerprints[key] = dict(sha256=file_sha256(path), file_bytes=size)

    register(source_path)
    source = json.loads(source_path.read_text())
    if source.get("observations_preserved") is not True or source.get("verification") != "full":
        raise ValueError("Source benchmark must record full verification and preserved observations.")
    configuration = source["configuration"]
    if configuration.get("precision") != "64" or configuration.get("metric_mode") != "full":
        raise ValueError("This check requires a precision=64 benchmark with full metrics.")
    steps, rank, repeats = (positive_integer(configuration.get(key), key)
                            for key in ("steps", "rank", "repeats"))
    tau = float(configuration["tau"])
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Source tau must be finite and positive.")
    beta = float(PowerSchedule(*(configuration["smoothing_" + key]
                                for key in ("scale", "offset", "exponent")))(steps - 1))
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("Terminal smoothing beta must be finite and positive.")
    artifacts = (args.artifacts_dir.resolve() if args.artifacts_dir is not None
                 else resolve_recorded_path(source["artifacts_dir"], source_path))
    compact = artifacts / "data.npz"
    register(compact, npz=True)
    data = StructuredQPTData.load_npz(compact)
    if data.observation_mode != "stored":
        raise ValueError("Original stored noisy observations are required.")
    if data.m > args.max_measurements:
        raise ValueError("Full enumeration exceeds --max-measurements; no evaluation started.")
    metadata = data.metadata
    if (metadata != source.get("source_metadata") or metadata.get("verification") != "full"
            or metadata.get("observations_preserved") is not True
            or metadata.get("verified_measurement_rows") != data.m
            or metadata.get("verified_basis_rows") != data.process_dimension):
        raise ValueError("Compact-data full-verification metadata do not match the benchmark.")
    if source.get("n_qubits") != data.n_qubits:
        raise ValueError("Compact-data qubit count differs from the benchmark.")
    estimate = reference_allocation_estimate(data.process_dimension, rank, min(args.batch_size, data.m))
    reference_cap = int(args.max_reference_mib * 2**20)
    if estimate > reference_cap:
        raise ValueError("Dense reference exceeds --max-reference-mib; no evaluation started.")

    rows = [row for row in source["runs"] if row.get("purpose") == "timing"]
    if len(rows) != 2 * repeats:
        raise ValueError("Exactly one successful timing row per backend/repeat is required.")
    expected = {(backend, repeat) for backend in ("dense", "structured") for repeat in range(repeats)}
    keys = [(row.get("backend"), row.get("repeat")) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected or any(row.get("status") != "success" for row in rows):
        raise ValueError("Exactly one successful timing row per backend/repeat is required.")
    backends = {row.get("measurement_backend") or
                (row.get("structured_runner_metadata") or {}).get("measurement_backend")
                for row in rows if row["backend"] == "structured"}
    if len(backends) != 1 or next(iter(backends)) not in ("tensor", "rank-one"):
        raise ValueError("Source timing rows must identify one resolved structured measurement backend.")
    measurement_backend = next(iter(backends))
    requested_backend = configuration.get("measurement_backend", "auto")
    if requested_backend not in ("auto", measurement_backend):
        raise ValueError("Source configured and resolved structured backends differ.")

    factors, records = {}, []
    for row in sorted(rows, key=lambda item: (item["backend"], item["repeat"])):
        name = f"{row['backend']}:{row['repeat']}"
        path = (artifacts / f"timing_{row['backend']}_{row['repeat']}.npy"
                if args.artifacts_dir is not None else resolve_recorded_path(row["final_factor_path"], source_path))
        register(path)
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.shape != (data.process_dimension, rank) or array.dtype != np.dtype("complex128"):
            raise ValueError(f"Saved factor {name} has unexpected shape or precision.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Saved factor {name} contains nonfinite values.")
        factor = np.array(array, copy=True)
        factor.setflags(write=False)
        factors[name] = factor
        record = dict(id=name, backend=row["backend"], repeat=row["repeat"], path=str(path.resolve()),
                      fingerprint=array_fingerprint(factor),
                      original_metrics={key: row.get(key) for key in ("final_measurement_loss", "final_smoothed_gap")})
        for key, value in record["original_metrics"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing/nonfinite source metric {key} for {name}.")
        records.append(record)
    provenance = dict(
        benchmark_report=str(source_path), benchmark_status=source.get("status"),
        benchmark_parity_passed=source.get("parity_passed"), configuration=configuration,
        compact_data=str(compact), artifacts_relocated=args.artifacts_dir is not None,
        input_files=fingerprints, input_storage_estimate_bytes=used_bytes,
        inherited_verification=metadata,
        original_workers=[{key: row.get(key) for key in (
            "backend", "repeat", "hostname", "device_kind", "jax_version", "jaxlib_version")}
            for row in rows],
        note="HDF5 is not opened or revalidated. Current hashes identify supplied artifacts; original full verification is inherited evidence.",
    )
    settings = dict(beta=beta, smoothing_iteration=steps - 1, tau=tau, rank=rank,
                    precision="64", measurement_backend=measurement_backend, batch_size=args.batch_size,
                    rtol=args.rtol, atol=args.atol, reference_allocation_estimate_bytes=estimate,
                    max_reference_bytes=reference_cap, max_input_bytes=input_cap,
                    max_measurements=args.max_measurements)
    return data, factors, records, provenance, settings


def run_check(args, report):
    import numpy as np
    from paper.experiments.qpt_same_factor import evaluate_same_factor

    data, factors, records, provenance, settings = prepare_inputs(args)
    report.update(source=provenance, settings=settings, factors=records, evaluations=[],
                  environment=dict(hostname=socket.gethostname(), python=platform.python_version(),
                                   numpy=np.__version__, platform=platform.platform(),
                                   threads={key: os.environ.get(key) for key in (
                                       "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}))
    report["numerical_source_sha256"] = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    evaluated = {}
    by_id = {}
    for record in records:
        digest = record["fingerprint"]["sha256"]
        if digest not in evaluated:
            name = f"evaluation_{len(evaluated):03d}"
            began = perf_counter()
            outputs = evaluate_same_factor(
                data, factors[record["id"]], settings["beta"], settings["tau"],
                batch_size=args.batch_size, max_reference_bytes=settings["max_reference_bytes"],
                measurement_backend=settings["measurement_backend"],
            )
            differences = {key: compare_arrays(outputs["dense"][key], value, args.rtol, args.atol)
                           for key, value in outputs["structured"].items()}
            arrays = {backend + "__" + key: value for backend in ("dense", "structured")
                      for key, value in outputs[backend].items()}
            if not all(np.all(np.isfinite(value)) for value in arrays.values()):
                raise FloatingPointError("Nonfinite CPU evaluation; no validated finding can be produced.")
            artifact = args.output_dir / (name + ".npz")
            np.savez(artifact, **arrays)
            row = dict(id=name, factor_ids=[], artifact=str(artifact), artifact_sha256=file_sha256(artifact),
                       seconds=perf_counter() - began, measurement_count=data.m,
                       batches=outputs["batches"], dense_measurement_entries=outputs["dense_measurement_entries"],
                       same_factor_comparisons=differences,
                       same_factor_allclose=all(value["allclose"] for value in differences.values()),
                       **{backend: {key: output_summary(value) for key, value in outputs[backend].items()}
                          for backend in ("dense", "structured")})
            evaluated[digest] = (row, outputs)
            report["evaluations"].append(row)
            print(f"{name}: full CPU cross-evaluation finished at {record['id']}", flush=True)
        row, outputs = evaluated[digest]
        row["factor_ids"].append(record["id"])
        record["evaluation_id"] = row["id"]
        by_id[record["id"]] = outputs
        record["cpu_vs_saved_metrics"] = {
            backend: {key: compare_arrays(np.asarray(record["original_metrics"][saved]),
                                          outputs[backend][key], args.rtol, args.atol)
                      for key, saved in (("measurement_loss", "final_measurement_loss"),
                                         ("smoothed_gap", "final_smoothed_gap"))}
            for backend in ("dense", "structured")}
        write_json(args.output_dir / "report.json", report)

    geometry_args = dict(rtol=args.rtol, atol=args.atol, max_reference_bytes=settings["max_reference_bytes"])
    report["repeatability"] = {}
    for backend in ("dense", "structured"):
        selected = [record for record in records if record["backend"] == backend]
        reference = selected[0]["id"]
        comparisons = [dict(reference=reference, actual=record["id"], **compare_factors(
            factors[reference], factors[record["id"]], **geometry_args)) for record in selected[1:]]
        report["repeatability"][backend] = dict(
            comparisons=comparisons,
            exact=all(row["raw_factor"]["exact"] for row in comparisons) if comparisons else None,
            allclose=all(row["raw_factor"]["allclose"] for row in comparisons) if comparisons else None,
        )
    report["cross_backend"] = []
    for repeat in range(provenance["configuration"]["repeats"]):
        reference, actual = f"dense:{repeat}", f"structured:{repeat}"
        metrics = {backend: {
            key: dict(reference=float(by_id[reference][backend][key]),
                      actual=float(by_id[actual][backend][key]),
                      **compare_arrays(by_id[reference][backend][key], by_id[actual][backend][key], args.rtol, args.atol))
            for key in SCALAR_METRICS} for backend in ("dense", "structured")}
        report["cross_backend"].append(dict(
            repeat=repeat, reference=reference, actual=actual,
            **compare_factors(factors[reference], factors[actual], **geometry_args),
            common_evaluator_metrics=metrics,
        ))
    unchanged = all(file_sha256(path) == value["sha256"] for path, value in provenance["input_files"].items())
    code_unchanged = all(file_sha256(ROOT / path) == value for path, value in report["numerical_source_sha256"].items())
    no_jax = not any(name in ("jax", "jaxlib") or name.startswith(("jax.", "jaxlib."))
                     for name in sys.modules)
    report["findings"] = dict(
        inputs_unchanged=unchanged, numerical_sources_unchanged=code_unchanged, no_jax_imported=no_jax,
        same_factor_formulations_allclose=all(row["same_factor_allclose"] for row in report["evaluations"]),
        dense_repeats_exact=report["repeatability"]["dense"]["exact"],
        structured_repeats_exact=report["repeatability"]["structured"]["exact"],
        cross_backend_raw_allclose=all(row["raw_factor"]["allclose"] for row in report["cross_backend"]),
        cross_backend_aligned_allclose=all(row["aligned_factor"]["allclose"] for row in report["cross_backend"]),
        cross_backend_process_allclose=all(row["process_matrix"]["allclose"] for row in report["cross_backend"]),
        saved_metrics_match_both_cpu_evaluators=all(
            value["allclose"] for record in records for backend in record["cpu_vs_saved_metrics"].values()
            for value in backend.values()),
    )
    if not unchanged or not code_unchanged or not no_jax:
        raise RuntimeError("Input/source immutability or CPU-only controls failed.")
    report["status"] = "complete"


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("batch_size", "max_measurements"):
        if getattr(args, name) < 1:
            parser().error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("rtol", "atol", "max_reference_mib", "max_input_mib"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name.startswith("max_") and value == 0):
            parser().error(f"Invalid --{name.replace('_', '-')}.")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    report = dict(schema_version=1, status="running", interpretation=[
        "CPU full-data cross-evaluation of saved final factors, not an optimizer rerun or performance benchmark.",
        "Each distinct numeric factor is evaluated once by both formulations; all timing repeats are retained.",
        "The full-gradient gap uses beta[T-1], not a stochastic momentum estimator.",
        "Numerical discrepancies are findings; completion does not mean dense/structured benchmark parity passed.",
        "CPU batchwise summation differs from the original monolithic dense metrics and GPU fusion.",
        "Agreement at saved final factors does not validate every intermediate GPU update or establish harmless roundoff.",
        "Original HDF5 verification is inherited; the source HDF5 file is neither loaded nor regenerated.",
        "Input/reference byte guards are conservative array estimates, not total process RAM limits.",
    ])
    path = args.output_dir / "report.json"
    write_json(path, report)
    try:
        run_check(args, report)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        report["seconds"] = perf_counter() - began
        write_json(path, report)
    print(json.dumps(dict(status=report["status"], findings=report.get("findings"), report=str(path)), indent=2), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
