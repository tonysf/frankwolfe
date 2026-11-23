"""Local-only QPT diagnostic tests; no cluster access or scheduler submission."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_qpt_reproducibility.py"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_diagnostic(source, output, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--benchmark-report", str(source),
         "--output-dir", str(output), "--device", "cpu", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )


@pytest.fixture(scope="module")
def diagnostic_module():
    return load_script(SCRIPT, "qpt_reproducibility_test")


@pytest.fixture
def source_benchmark(tmp_path):
    h5py = pytest.importorskip("h5py")
    from paper.experiments.qpt_structured_data import (
        StructuredQPTData, standard_local_basis, standard_local_measurements,
    )

    source = tmp_path / "source"
    artifacts = source / "artifacts"
    artifacts.mkdir(parents=True)
    h5_path = source / "tiny.h5"
    basis = standard_local_basis()
    # Stored targets deliberately include arbitrary noise; no truth-generated
    # target path can accidentally stand in for the actual observations.
    observations = np.random.default_rng(763).uniform(-0.05, 1.05, 24)
    with h5py.File(h5_path, "w") as handle:
        handle["f_jax_vector"] = observations
        handle["D_jax_tensors"] = standard_local_measurements()
        handle["A_jax_basis"] = basis
        handle["B_jax_tensors"] = np.einsum("mki,nkj->nmij", basis.conj(), basis)
    data = StructuredQPTData.from_hdf5(h5_path, verification="full", chunk_size=3)
    compact = artifacts / "data.npz"
    data.save_npz(compact)

    benchmark = load_script(ROOT / "scripts" / "benchmark_qpt_structured.py", "qpt_repro_source_benchmark")
    path = source / "result.json"
    arguments = benchmark.parser().parse_args([
        "--h5", str(h5_path), "--save", str(path), "--artifacts-dir", str(artifacts),
        "--device", "cpu", "--precision", "64", "--steps", "6", "--chunk-steps", "2",
        "--batch-size", "3", "--rank", "1", "--tau", "10",
        "--metrics-every", "0", "--metric-mode", "full", "--metric-batch-size", "3",
        "--measurement-backend", "tensor", "--initialization-seed", "0", "--sampling-seed", "0",
        "--rho-scale", "2", "--rho-offset", "4", "--rho-exponent", "0.6",
        "--smoothing-scale", "10", "--smoothing-offset", "1", "--smoothing-exponent", "0.25",
        "--step-scale", "2", "--step-offset", "2", "--step-exponent", "1",
    ])
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(arguments).items()}
    report = {
        "schema_version": 2,
        "status": "failed",
        "parity_passed": False,
        "configuration": configuration,
        "artifacts_dir": str(artifacts),
        "h5": str(h5_path),
        "n_qubits": 1,
        "observations_preserved": True,
        "verification": "full",
        "source_metadata": data.metadata,
        "runs": [],
    }
    path.write_text(json.dumps(report, indent=2))
    return path, compact, h5_path


@pytest.mark.parametrize("option,value", [
    ("--processes", "0"), ("--processes", "-1"),
    ("--replays", "0"), ("--replays", "1"), ("--replays", "-1"),
    ("--device", "auto"),
])
def test_cli_rejects_invalid_replay_configuration(tmp_path, source_benchmark, option, value):
    source, compact, h5_path = source_benchmark
    before = {path: file_digest(path) for path in (source, compact, h5_path)}
    result = run_diagnostic(source, tmp_path / "invalid", option, value)
    assert result.returncode != 0, result.stdout + result.stderr
    assert option.lstrip("-") in result.stdout + result.stderr
    assert {path: file_digest(path) for path in before} == before


def test_cli_refuses_existing_output_without_overwriting(tmp_path, source_benchmark):
    source, compact, h5_path = source_benchmark
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "report.json"
    sentinel.write_text('{"user_content": "must survive"}\n')
    before = {path: file_digest(path) for path in (source, compact, h5_path, sentinel)}
    result = run_diagnostic(source, output, "--processes", "1", "--replays", "2")
    assert result.returncode != 0, result.stdout + result.stderr
    message = (result.stdout + result.stderr).lower()
    assert "overwrite" in message or "exist" in message
    assert {path: file_digest(path) for path in before} == before
    assert set(output.iterdir()) == {sentinel}


@pytest.mark.parametrize("field,value", [
    ("observations_preserved", False),
    ("verification", "sampled"),
])
def test_cli_rejects_unverified_or_nonpreserved_source(tmp_path, source_benchmark, field, value):
    source, compact, h5_path = source_benchmark
    report = json.loads(source.read_text())
    report[field] = value
    source.write_text(json.dumps(report))
    before = {path: file_digest(path) for path in (source, compact, h5_path)}
    result = run_diagnostic(source, tmp_path / "rejected", "--processes", "1", "--replays", "2")
    assert result.returncode != 0, result.stdout + result.stderr
    expected_word = "preserv" if field == "observations_preserved" else "full"
    assert expected_word in (result.stdout + result.stderr).lower()
    assert {path: file_digest(path) for path in before} == before


def test_cli_requires_existing_compact_data(tmp_path, source_benchmark):
    source, compact, h5_path = source_benchmark
    report = json.loads(source.read_text())
    report["artifacts_dir"] = str(tmp_path / "absent_artifacts")
    source.write_text(json.dumps(report))
    before = {path: file_digest(path) for path in (source, compact, h5_path)}
    # A different adjacent artifacts/data.npz must not silently substitute for
    # the exact compact-data path named by the source report.
    assert compact.is_file()
    result = run_diagnostic(source, tmp_path / "rejected", "--processes", "1", "--replays", "2")
    assert result.returncode != 0, result.stdout + result.stderr
    message = (result.stdout + result.stderr).lower()
    assert "data.npz" in message or "compact" in message
    assert {path: file_digest(path) for path in before} == before


def test_cpu_replays_freeze_inputs_preserve_sources_and_report_comparisons(tmp_path, source_benchmark):
    pytest.importorskip("jax")
    from paper.experiments.qpt_structured_data import StructuredQPTData

    source, compact, h5_path = source_benchmark
    before = {path: file_digest(path) for path in (source, compact, h5_path)}
    output = tmp_path / "diagnostic"
    result = run_diagnostic(source, output, "--processes", "2", "--replays", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path: file_digest(path) for path in before} == before

    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["configuration"]["steps"] == 6
    assert report["configuration"]["chunk_steps"] == 2
    assert report["configuration"]["batch_size"] == 3
    assert report["configuration"]["measurement_backend"] == "tensor"
    assert report["manifest_sha256"] == file_digest(output / "input_manifest.json")
    assert report["frozen_inputs"]["bundle_sha256"] == file_digest(output / "input_bundle.npz")
    for field in ("input_sha256", "initial_factor_sha256", "batch_symbols_sha256"):
        assert len(report["frozen_inputs"][field]) == 64
    assert report["findings"]["inputs_match"] is True
    assert report["findings"]["devices_match"] is True
    assert report["findings"]["same_executable_exact"] is True
    assert report["findings"]["same_executable_allclose"] is True
    assert report["findings"]["cross_process_exact"] in (True, False)
    assert report["findings"]["cross_process_allclose"] in (True, False)
    assert report["findings"]["cpu_reference_status"] == "validated"
    assert len(report["across_process_comparisons"]) == 1
    assert len(report["workers"]) == 2
    assert len({row["pid"] for row in report["workers"]}) == 2

    data = StructuredQPTData.load_npz(compact)
    expected_symbols = data.sample_symbols(np.random.default_rng(0), 6 * 3).astype(np.int32).reshape(6, 3, 1)
    expected_targets = data.observations_for_symbols(expected_symbols.reshape(-1, 1)).reshape(6, 3)
    with np.load(output / "input_bundle.npz", allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["symbols"], expected_symbols)
        np.testing.assert_array_equal(bundle["observations"], expected_targets)
        np.testing.assert_array_equal(bundle["momentum0"], 0)
        initial_factor = bundle["factor0"].copy()
        assert hashlib.sha256(initial_factor.tobytes()).hexdigest() == report["frozen_inputs"]["initial_factor_sha256"]
        assert hashlib.sha256(bundle["symbols"].tobytes()).hexdigest() == report["frozen_inputs"]["batch_symbols_sha256"]
        for name in bundle.files:
            description = report["frozen_inputs"]["arrays"][name]
            assert description["shape"] == list(bundle[name].shape)
            assert description["dtype"] == str(bundle[name].dtype)
            assert description["sha256"] == hashlib.sha256(bundle[name].tobytes()).hexdigest()

    for index, row in enumerate(report["workers"]):
        worker = output / f"worker_{index:03d}"
        assert row["status"] == "complete"
        assert row["device_platform"] == "cpu"
        assert row["input_sha256"] == report["frozen_inputs"]["input_sha256"]
        assert row["manifest_sha256"] == report["manifest_sha256"]
        assert row["scan_compilation_count"] == 1
        assert row["replays_completed"] == 2
        assert len(row["same_executable_comparisons"]) == 1
        assert len(row["operator_probes"]) == 2
        assert row["operator_probes"][0]["iteration"] == 0
        assert row["operator_probes"][1]["iteration"] in (2, 4)
        for probe in row["operator_probes"]:
            assert probe["cpu_reference"]["status"] == "validated"
            assert probe["replay_exact"] is True
            assert probe["replay_allclose"] is True
            assert probe["all_outputs_finite"] is True
            assert len(probe["runs"]) == 2
        assert Path(row["trajectory_path"]).is_file()
        for name in ("report.json", "trajectories.npz", "stdout.log", "stderr.log"):
            assert (worker / name).is_file()
        worker_report = json.loads((worker / "report.json").read_text())
        assert worker_report["pid"] == row["pid"]
        with np.load(worker / "trajectories.npz", allow_pickle=False) as trajectory:
            np.testing.assert_array_equal(trajectory["boundaries"], [0, 2, 4, 6])
            assert trajectory["factors"].shape == (2, 4, 4, 1)
            assert trajectory["momentum"].shape == (2, 4, 4, 1)
            assert trajectory["gaps"].shape == (2, 6)
            for name in ("factors", "momentum", "gaps"):
                assert np.isfinite(trajectory[name]).all()
                np.testing.assert_array_equal(trajectory[name][0], trajectory[name][1])
            np.testing.assert_array_equal(trajectory["momentum"][:, 0], 0)
            for replay in range(2):
                np.testing.assert_array_equal(trajectory["factors"][replay, 0], initial_factor)
        executable = row["executable"]
        if executable.get("path"):
            assert Path(executable["path"]).is_file()
            assert executable["text_sha256"] == file_digest(executable["path"])


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_comparisons_are_not_equal_and_remain_strict_json(diagnostic_module, nonfinite):
    result = diagnostic_module.compare_arrays(np.asarray([nonfinite]), np.asarray([nonfinite]), 1e-10, 1e-12)
    assert result["finite"] is False
    assert result["exact"] is False
    assert result["allclose"] is False
    assert result["max_abs_difference"] is None
    assert result["relative_l2_difference"] is None
    json.dumps(result, allow_nan=False)


def test_finite_overflow_comparison_remains_strict_json(diagnostic_module):
    with np.errstate(over="ignore", invalid="ignore"):
        result = diagnostic_module.compare_arrays(np.asarray([1e308]), np.asarray([-1e308]), 1e-10, 1e-12)
    assert result["exact"] is False
    assert result["allclose"] is False
    assert result["max_abs_difference"] is None
    assert result["relative_l2_difference"] is None
    json.dumps(result, allow_nan=False)


def test_zero_reference_relative_error_is_unavailable_not_infinite(diagnostic_module):
    result = diagnostic_module.compare_arrays(np.zeros(2), np.ones(2), 1e-10, 1e-12)
    assert result["finite"] is True
    assert result["max_abs_difference"] == 1
    assert result["relative_l2_difference"] is None
    json.dumps(result, allow_nan=False)


def test_trajectory_comparison_locates_first_recorded_state_and_gap_difference(diagnostic_module):
    reference = dict(factors=np.zeros((4, 4, 1), dtype=complex),
                     momentum=np.zeros((4, 4, 1), dtype=complex), gaps=np.zeros(6))
    actual = {key: value.copy() for key, value in reference.items()}
    actual["factors"][1, 0, 0] = 0.01j
    actual["gaps"][3] = 0.1
    result = diagnostic_module.compare_trajectories(reference, actual, np.asarray([0, 2, 4, 6]), 1e-10, 1e-12)
    assert result["exact"] is False
    assert result["allclose"] is False
    assert result["first_different_boundary"] == 2
    assert result["first_gap_difference_iteration"] == 3
    json.dumps(result, allow_nan=False)


@pytest.fixture
def make_completed_worker(tmp_path):
    def build(index, *, offset=0, within_exact=True, reference_status="validated"):
        path = tmp_path / f"worker_{index}.npz"
        factors = np.zeros((1, 4, 4, 1), dtype=complex)
        factors[0, 1:] = offset
        np.savez(path, factors=factors, momentum=np.zeros_like(factors), gaps=np.zeros((1, 6)),
                 boundaries=np.asarray([0, 2, 4, 6]))
        return dict(
            worker=index, status="complete", pid=1000 + index,
            input_sha256="frozen-input", manifest_sha256="frozen-manifest", hostname="local-test", device="TFRT_CPU_0",
            device_platform="cpu", device_kind="cpu", jax_version="test", jaxlib_version="test",
            numpy_version="test", trajectory_path=str(path),
            same_executable_comparisons=[dict(exact=within_exact, allclose=within_exact)],
            operator_probes=[dict(cpu_reference=dict(status=reference_status), replay_exact=True)],
        )
    return build


def test_numerical_variability_is_a_completed_finding_not_execution_failure(diagnostic_module, make_completed_worker):
    report = dict(frozen_inputs=dict(input_sha256="frozen-input"), manifest_sha256="frozen-manifest", workers=[
        make_completed_worker(0, within_exact=False), make_completed_worker(1, offset=0.1),
    ])
    diagnostic_module.summarize(report, SimpleNamespace(processes=2, rtol=1e-10, atol=1e-12))
    assert report["status"] == "complete"
    assert report["findings"]["same_executable_exact"] is False
    assert report["findings"]["same_executable_allclose"] is False
    assert report["findings"]["cross_process_exact"] is False
    assert report["findings"]["cross_process_allclose"] is False
    assert report["findings"]["cpu_reference_status"] == "validated"
    json.dumps(report, allow_nan=False)


def test_skipped_reference_and_missing_cross_process_comparison_do_not_pass(diagnostic_module, make_completed_worker):
    report = dict(frozen_inputs=dict(input_sha256="frozen-input"), manifest_sha256="frozen-manifest",
                  workers=[make_completed_worker(0, reference_status="skipped")])
    diagnostic_module.summarize(report, SimpleNamespace(processes=1, rtol=1e-10, atol=1e-12))
    assert report["status"] == "complete"
    assert report["findings"]["cross_process_exact"] is None
    assert report["findings"]["cross_process_allclose"] is None
    assert report["findings"]["cpu_reference_status"] == "incomplete"
    assert report["across_process_comparisons"] == []
    json.dumps(report, allow_nan=False)


def test_worker_execution_failure_cannot_be_reported_complete(diagnostic_module, make_completed_worker):
    report = dict(frozen_inputs=dict(input_sha256="frozen-input"), manifest_sha256="frozen-manifest", workers=[
        make_completed_worker(0), dict(worker=1, status="failed", error="test worker crashed"),
    ])
    diagnostic_module.summarize(report, SimpleNamespace(processes=2, rtol=1e-10, atol=1e-12))
    assert report["status"] == "failed"
    assert report["workers"][1]["error"] == "test worker crashed"
    assert report["findings"]["cross_process_exact"] is None
    assert report["findings"]["cross_process_allclose"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("fingerprint", ["input_sha256", "manifest_sha256"])
def test_mismatched_worker_inputs_fail_control_checks(diagnostic_module, make_completed_worker, fingerprint):
    worker = make_completed_worker(0)
    worker[fingerprint] = "changed-input"
    report = dict(frozen_inputs=dict(input_sha256="frozen-input"), manifest_sha256="frozen-manifest", workers=[worker])
    diagnostic_module.summarize(report, SimpleNamespace(processes=1, rtol=1e-10, atol=1e-12))
    assert report["status"] == "failed"
    assert report["findings"]["inputs_match"] is False
