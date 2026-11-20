"""CPU-only saved-factor CLI checks; no scheduler, connection, or GPU use."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_qpt_same_factor.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_fingerprints(directory):
    return {str(path.relative_to(directory)): digest(path)
            for path in directory.rglob("*") if path.is_file()}


def cpu_environment():
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def run_cli(source, output, *extra, code=None):
    executable = [sys.executable, str(SCRIPT)] if code is None else [sys.executable, "-c", code]
    return subprocess.run(
        executable + ["--benchmark-report", str(source), "--output-dir", str(output),
                      "--batch-size", "7", *extra],
        cwd=ROOT, env=cpu_environment(), capture_output=True, text=True, timeout=60,
    )


def assert_failed_report(result, output):
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]
    assert not list(output.glob("evaluation_*.npz"))
    json.dumps(report, allow_nan=False)
    return report


@pytest.fixture(scope="module")
def cli_module():
    spec = importlib.util.spec_from_file_location("qpt_same_factor_cli_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark_bundle(tmp_path):
    h5py = pytest.importorskip("h5py")
    from paper.experiments.qpt_same_factor import evaluate_same_factor
    from paper.experiments.qpt_structured_data import (
        StructuredQPTData, standard_local_basis, standard_local_measurements,
    )
    from paper.experiments.quantum_process_tomography import PowerSchedule

    directory = tmp_path / "source"
    artifacts = directory / "artifacts"
    artifacts.mkdir(parents=True)
    h5 = directory / "original_noisy.h5"
    basis = standard_local_basis()
    observations = np.random.default_rng(954).uniform(-0.05, 1.05, 24)
    with h5py.File(h5, "w") as handle:
        handle["f_jax_vector"] = observations
        handle["D_jax_tensors"] = standard_local_measurements()
        handle["A_jax_basis"] = basis
        handle["B_jax_tensors"] = np.einsum("mki,nkj->nmij", basis.conj(), basis)
    data = StructuredQPTData.from_hdf5(h5, verification="full", chunk_size=5)
    compact = artifacts / "data.npz"
    data.save_npz(compact)
    configuration = dict(
        precision="64", metric_mode="full", steps=6, rank=1, repeats=2, tau=10.0,
        smoothing_scale=10.0, smoothing_offset=1.0, smoothing_exponent=0.25,
        measurement_backend="tensor",
    )
    beta = PowerSchedule(10.0, 1.0, 0.25)(configuration["steps"] - 1)
    base = np.asarray([1.0, 0.2 - 0.1j, 0.05 + 0.02j, 0.3j], dtype=np.complex128)[:, None]
    shifted = base.copy()
    shifted[0, 0] += 0.001 + 0.002j
    runs = []
    for backend, factor in (("dense", base), ("structured", shifted)):
        # This integration fixture delegates numerical scalar construction to
        # the evaluator. Its algebra is independently tested in the helper's
        # test module; these CLI tests concern provenance and report semantics.
        metrics = evaluate_same_factor(data, factor, beta, 10.0, batch_size=7)["dense"]
        for repeat in range(2):
            path = artifacts / f"timing_{backend}_{repeat}.npy"
            np.save(path, factor)
            runs.append(dict(
                backend=backend, purpose="timing", repeat=repeat, status="success",
                final_factor_path=str(path), final_measurement_loss=metrics["measurement_loss"],
                final_smoothed_gap=metrics["smoothed_gap"],
                structured_runner_metadata=dict(measurement_backend="tensor") if backend == "structured" else {},
            ))
    report = dict(
        schema_version=2, status="failed", parity_passed=False, observations_preserved=True,
        verification="full", n_qubits=1, configuration=configuration, artifacts_dir=str(artifacts),
        h5=str(h5), source_metadata=data.metadata, runs=runs,
    )
    source = directory / "result.json"
    source.write_text(json.dumps(report, indent=2))
    return dict(directory=directory, source=source, report=report, data=data, compact=compact,
                artifacts=artifacts, h5=h5, beta=beta)


def rewrite_report(bundle):
    bundle["source"].write_text(json.dumps(bundle["report"], indent=2))


def test_full_cpu_cli_separates_trajectory_differences_and_same_factor_agreement(tmp_path, benchmark_bundle):
    bundle = benchmark_bundle
    # A completed, verified conversion suffices: the CLI must not reopen HDF5.
    bundle["h5"].rename(bundle["h5"].with_suffix(".backup"))
    before = tree_fingerprints(bundle["directory"])
    output = tmp_path / "check"
    result = run_cli(bundle["source"], output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert tree_fingerprints(bundle["directory"]) == before
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["source"]["benchmark_status"] == "failed"
    assert report["source"]["benchmark_parity_passed"] is False
    assert report["source"]["inherited_verification"] == bundle["data"].metadata
    assert report["settings"]["smoothing_iteration"] == 5
    assert report["settings"]["beta"] == bundle["beta"]
    assert report["settings"]["measurement_backend"] == "tensor"
    assert len(report["factors"]) == 4
    assert len(report["evaluations"]) == 2
    assert len(report["cross_backend"]) == 2
    for key in ("inputs_unchanged", "numerical_sources_unchanged", "no_jax_imported",
                "same_factor_formulations_allclose", "dense_repeats_exact", "structured_repeats_exact",
                "saved_metrics_match_both_cpu_evaluators"):
        assert report["findings"][key] is True
    for key in ("cross_backend_raw_allclose", "cross_backend_aligned_allclose", "cross_backend_process_allclose"):
        assert report["findings"][key] is False
    for row in report["evaluations"]:
        assert row["measurement_count"] == 24
        assert row["batches"] == 4
        assert row["dense_measurement_entries"] == 24 * 4 * 4
        assert len(row["factor_ids"]) == 2
        assert row["same_factor_allclose"] is True
        assert row["artifact_sha256"] == digest(row["artifact"])
        with np.load(row["artifact"], allow_pickle=False) as values:
            assert "dense__full_gradient" in values.files
            assert "structured__full_gradient" in values.files
            assert all(np.all(np.isfinite(values[name])) for name in values.files)
    for row in report["cross_backend"]:
        for evaluator in ("dense", "structured"):
            gap = row["common_evaluator_metrics"][evaluator]["smoothed_gap"]
            assert gap["allclose"] is False
            assert gap["max_abs_difference"] > 0
            assert gap["reference"] != gap["actual"]
    for path, fingerprint in report["source"]["input_files"].items():
        assert fingerprint["sha256"] == digest(path)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("field,value", [("precision", "32"), ("metric_mode", "sampled")])
def test_rejects_unsupported_source_precision_or_metric_mode(tmp_path, benchmark_bundle, field, value):
    benchmark_bundle["report"]["configuration"][field] = value
    rewrite_report(benchmark_bundle)
    output = tmp_path / "rejected"
    report = assert_failed_report(run_cli(benchmark_bundle["source"], output), output)
    assert "precision=64" in report["error"] or "full metrics" in report["error"]


@pytest.mark.parametrize("field,value", [("observations_preserved", False), ("verification", "sampled")])
def test_source_must_preserve_observations_and_have_full_verification(tmp_path, benchmark_bundle, field, value):
    benchmark_bundle["report"][field] = value
    rewrite_report(benchmark_bundle)
    output = tmp_path / "unverified"
    report = assert_failed_report(run_cli(benchmark_bundle["source"], output), output)
    assert "full verification" in report["error"]


def test_compact_full_verification_counts_must_match_all_rows(tmp_path, benchmark_bundle):
    bundle = benchmark_bundle
    bundle["data"].metadata["verified_measurement_rows"] = 23
    bundle["data"].save_npz(bundle["compact"])
    bundle["report"]["source_metadata"] = bundle["data"].metadata
    rewrite_report(bundle)
    output = tmp_path / "incomplete_verification"
    report = assert_failed_report(run_cli(bundle["source"], output), output)
    assert "full-verification metadata" in report["error"]


@pytest.mark.parametrize("target", ["report", "compact", "factor"])
def test_missing_recorded_paths_fail_without_guessing_adjacent_files(tmp_path, benchmark_bundle, target):
    bundle = benchmark_bundle
    source = bundle["source"]
    if target == "report":
        source = tmp_path / "missing.json"
    elif target == "compact":
        bundle["report"]["artifacts_dir"] = str(tmp_path / "missing_artifacts")
        rewrite_report(bundle)
    else:
        bundle["report"]["runs"][0]["final_factor_path"] = str(tmp_path / "missing_factor.npy")
        rewrite_report(bundle)
    before = tree_fingerprints(bundle["directory"])
    output = tmp_path / "rejected"
    assert_failed_report(run_cli(source, output), output)
    assert tree_fingerprints(bundle["directory"]) == before


def test_explicit_artifact_relocation_uses_copied_inputs_and_preserves_originals(tmp_path, benchmark_bundle):
    bundle = benchmark_bundle
    relocated = tmp_path / "relocated"
    shutil.copytree(bundle["artifacts"], relocated)
    bundle["report"]["artifacts_dir"] = str(tmp_path / "unavailable_original")
    for row in bundle["report"]["runs"]:
        row["final_factor_path"] = str(tmp_path / "unavailable_original" / Path(row["final_factor_path"]).name)
    rewrite_report(bundle)
    before = tree_fingerprints(bundle["directory"])
    copied_before = tree_fingerprints(relocated)
    output = tmp_path / "relocation_check"
    result = run_cli(bundle["source"], output, "--artifacts-dir", str(relocated))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["source"]["artifacts_relocated"] is True
    assert Path(report["source"]["compact_data"]).parent == relocated
    assert all(Path(row["path"]).parent == relocated for row in report["factors"])
    assert tree_fingerprints(bundle["directory"]) == before
    assert tree_fingerprints(relocated) == copied_before


@pytest.mark.parametrize("defect", ["duplicate", "missing", "failed"])
def test_all_expected_timing_rows_are_required_once(tmp_path, benchmark_bundle, defect):
    rows = benchmark_bundle["report"]["runs"]
    if defect == "duplicate":
        rows.append(dict(rows[0]))
    elif defect == "missing":
        rows.pop()
    else:
        rows[0]["status"] = "failed"
    rewrite_report(benchmark_bundle)
    output = tmp_path / "rejected"
    report = assert_failed_report(run_cli(benchmark_bundle["source"], output), output)
    assert "successful timing row" in report["error"]


def test_huge_claimed_repeat_count_rejected_before_expected_set_allocation(tmp_path, benchmark_bundle):
    benchmark_bundle["report"]["configuration"]["repeats"] = 10**12
    rewrite_report(benchmark_bundle)
    output = tmp_path / "huge_repeat_count"
    report = assert_failed_report(run_cli(benchmark_bundle["source"], output), output)
    assert "timing row" in report["error"]


@pytest.mark.parametrize("option,value", [
    ("--batch-size", "0"), ("--batch-size", "-1"), ("--max-measurements", "0"),
    ("--rtol", "nan"), ("--atol", "-1"), ("--max-reference-mib", "0"),
    ("--max-input-mib", "inf"),
])
def test_invalid_cli_controls_rejected_before_output_creation(tmp_path, benchmark_bundle, option, value):
    output = tmp_path / "invalid"
    result = run_cli(benchmark_bundle["source"], output, option, value)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not output.exists()


@pytest.mark.parametrize("defect", ["dtype", "shape", "nan", "inf", "object"])
def test_invalid_saved_factor_arrays_fail_without_evaluation(tmp_path, benchmark_bundle, defect):
    path = Path(benchmark_bundle["report"]["runs"][0]["final_factor_path"])
    factor = np.load(path, allow_pickle=False)
    if defect == "dtype":
        factor = factor.astype(np.complex64)
    elif defect == "shape":
        factor = factor[:, 0]
    elif defect == "object":
        factor = factor.astype(object)
    else:
        factor[0, 0] = np.nan if defect == "nan" else np.inf
    np.save(path, factor)
    before = tree_fingerprints(benchmark_bundle["directory"])
    output = tmp_path / "invalid_array"
    assert_failed_report(run_cli(benchmark_bundle["source"], output), output)
    assert tree_fingerprints(benchmark_bundle["directory"]) == before


@pytest.mark.parametrize("option,value", [
    ("--max-measurements", "23"), ("--max-input-mib", "0.001"),
    ("--max-reference-mib", "0.001"),
])
def test_work_and_storage_guards_stop_before_full_evaluation(tmp_path, benchmark_bundle, option, value):
    output = tmp_path / "capped"
    report = assert_failed_report(run_cli(benchmark_bundle["source"], output, option, value), output)
    assert option in report["error"]


def test_existing_output_and_source_content_are_not_overwritten(tmp_path, benchmark_bundle):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "report.json"
    sentinel.write_text("user content, not a diagnostic report\n")
    before = tree_fingerprints(benchmark_bundle["directory"])
    sentinel_before = digest(sentinel)
    result = run_cli(benchmark_bundle["source"], output)
    assert result.returncode != 0
    assert digest(sentinel) == sentinel_before
    assert list(output.iterdir()) == [sentinel]
    assert tree_fingerprints(benchmark_bundle["directory"]) == before


def test_saved_repeat_variability_is_reported_without_execution_failure(tmp_path, benchmark_bundle):
    from paper.experiments.qpt_same_factor import evaluate_same_factor

    bundle = benchmark_bundle
    row = next(row for row in bundle["report"]["runs"] if row["backend"] == "structured" and row["repeat"] == 1)
    path = Path(row["final_factor_path"])
    factor = np.load(path, allow_pickle=False)
    factor[1, 0] += 0.03j
    np.save(path, factor)
    metrics = evaluate_same_factor(bundle["data"], factor, bundle["beta"], 10.0, batch_size=7)["dense"]
    row.update(final_measurement_loss=metrics["measurement_loss"], final_smoothed_gap=metrics["smoothed_gap"])
    rewrite_report(bundle)
    output = tmp_path / "repeat_variability"
    result = run_cli(bundle["source"], output)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["findings"]["dense_repeats_exact"] is True
    assert report["findings"]["structured_repeats_exact"] is False
    assert report["findings"]["same_factor_formulations_allclose"] is True
    assert report["findings"]["saved_metrics_match_both_cpu_evaluators"] is True
    assert len(report["evaluations"]) == 3


def test_injected_numerical_disagreement_is_a_finding_not_execution_failure(tmp_path, benchmark_bundle):
    # Use a fresh interpreter so unrelated suite tests importing JAX cannot
    # contaminate the CLI's deliberately strict CPU-only module check.
    code = """
import sys
from paper.experiments import qpt_same_factor
from scripts.diagnose_qpt_same_factor import main
original = qpt_same_factor.evaluate_same_factor
def changed(*args, **kwargs):
    outputs = original(*args, **kwargs)
    outputs['structured']['measurement_loss'] += 0.01
    return outputs
qpt_same_factor.evaluate_same_factor = changed
raise SystemExit(main(sys.argv[1:]))
"""
    output = tmp_path / "disagreement"
    result = run_cli(benchmark_bundle["source"], output, code=code)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["findings"]["same_factor_formulations_allclose"] is False
    assert report["findings"]["saved_metrics_match_both_cpu_evaluators"] is False
    assert report["findings"]["no_jax_imported"] is True
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("rank", [1, 2])
def test_gauge_alignment_preserves_process_matrix_without_changing_inputs(cli_module, rank):
    generator = np.random.default_rng(109)
    reference = generator.normal(size=(4, rank)) + 1j * generator.normal(size=(4, rank))
    gauge = np.asarray([[np.exp(0.7j)]]) if rank == 1 else np.asarray([[0, 1j], [1, 0]], dtype=complex)
    actual = reference @ gauge
    before = reference.copy(), actual.copy()
    compared = cli_module.compare_factors(reference, actual, rtol=1e-10, atol=1e-12, max_reference_bytes=2**20)
    assert compared["raw_factor"]["allclose"] is False
    assert compared["aligned_factor"]["allclose"] is True
    assert compared["process_matrix"]["allclose"] is True
    np.testing.assert_array_equal(reference, before[0])
    np.testing.assert_array_equal(actual, before[1])
    json.dumps(compared, allow_nan=False)


def test_geometry_guard_precedes_svd_allocation(cli_module, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("SVD must not run above the geometry cap")
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    factor = np.ones((4, 1), dtype=complex)
    with pytest.raises(ValueError, match="max-reference-mib"):
        cli_module.compare_factors(factor, factor, rtol=1e-10, atol=1e-12, max_reference_bytes=1)
