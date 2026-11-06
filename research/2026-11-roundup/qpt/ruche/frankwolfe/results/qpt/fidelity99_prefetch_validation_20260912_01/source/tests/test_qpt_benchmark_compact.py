"""Compact-input benchmark guards and tiny CPU-only integration checks."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_qpt_structured.py"


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("qpt_compact_benchmark_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compact(tmp_path):
    from paper.experiments.qpt_structured_data import (
        StructuredQPTData, standard_local_basis, standard_local_measurements,
    )
    bank = standard_local_measurements()
    truth = np.asarray([[np.sqrt(2)], [0], [0], [0]], dtype=np.complex128)
    observations = np.einsum("i,bij,j->b", truth[:, 0].conj(), bank, truth[:, 0]).real
    observations += np.random.default_rng(99).normal(0, 0.05, 24)
    data = StructuredQPTData(
        1, bank, standard_local_basis(), observations=observations, truth_factor=truth,
        metadata=dict(source_kind="synthetic_compact", verification="constructed",
                      observations_sha256=hashlib.sha256(observations.tobytes()).hexdigest(),
                      truth_factor_sha256=hashlib.sha256(truth.tobytes()).hexdigest()),
    )
    path = tmp_path / "original.npz"
    data.save_npz(path)
    return path


def arguments(source, report):
    return ["--data", str(source), "--save", str(report), "--device", "cpu",
            "--steps", "2", "--batch-size", "3", "--chunk-steps", "1",
            "--metrics-every", "1", "--metric-samples", "7", "--metric-batch-size", "3",
            "--step-scale", "0.1", "--repeats", "2", "--measurement-backend", "tensor"]


def test_mode_defaults_and_dense_rejection(benchmark, tmp_path):
    base = ["--save", str(tmp_path / "report.json")]
    compact = benchmark.parser().parse_args(base + ["--data", "data.npz"])
    assert compact.backends == ["structured"]
    assert compact.metric_mode == "sampled"
    legacy = benchmark.parser().parse_args(base + ["--h5", "data.h5"])
    assert legacy.backends == ["dense", "structured"]
    assert legacy.metric_mode == "full"
    for conflicting in (["--data", "data.npz", "--h5", "data.h5"],
                        ["--data", "data.npz", "--backends", "dense"],
                        ["--data", "data.npz", "--worker", "dense"]):
        with pytest.raises(SystemExit):
            benchmark.parser().parse_args(base + conflicting)
    explicit = benchmark.parser().parse_args(base + ["--data", "data.npz", "--metric-mode", "full"])
    assert explicit.metric_mode == "full"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metric_rejected_before_report_assignment(benchmark, value):
    with pytest.raises(FloatingPointError, match="Nonfinite test metric"):
        benchmark.finite_scalar(value, "test metric")


def test_compact_cpu_workers_keep_source_and_full_bounded_results(compact, tmp_path, benchmark):
    pytest.importorskip("jax")
    before = benchmark.file_sha256(compact)
    target = tmp_path / "report.json"
    environment = os.environ.copy()
    environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    environment["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    environment["SLURM_JOB_ID"] = "test-job-id"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)] + arguments(compact, target)
        + ["--profile-memory", "--memory-poll-ms", "20"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(target.read_text())
    assert report["status"] == "complete"
    assert report["input_mode"] == "compact"
    assert report["h5"] is None and report["hdf5_accessed"] is False
    assert report["verification"] == "constructed"
    assert report["conversion_seconds"] is None
    assert report["compact_save_seconds"] is None
    assert report["inputs_unchanged"] is True
    assert report["numerical_sources_unchanged"] is True
    assert report["numerical_source_sha256"]["scripts/benchmark_qpt_structured.py"] == benchmark.file_sha256(SCRIPT)
    assert report["scheduler_environment"]["SLURM_JOB_ID"] == "test-job-id"
    if sys.platform in ("linux", "darwin"):
        assert report["compact_parent_peak_rss_bytes"] > 0
    assert report["compact_input"]["sha256"] == before == benchmark.file_sha256(compact)
    assert report["configuration"]["backends"] == ["structured"]
    assert report["configuration"]["metric_mode"] == "sampled"
    assert report["measurement_count"] == 24 and report["process_dimension"] == 4
    assert report["truth_factor_fingerprint"]["shape"] == [4, 1]
    assert report["dense_comparison_performed"] is False
    assert report["parity"] == [] and "parity_passed" not in report
    assert "dense_over_structured_optimizer_ratio" not in report
    assert not (Path(report["artifacts_dir"]) / "data.npz").exists()
    assert len(report["runs"]) == 3
    assert len({row["pid"] for row in report["runs"]}) == 3
    assert report["repeatability"]["structured"]["final_factors_bitwise_equal"] is True
    timing_values = [row["optimizer_seconds"] for row in report["runs"] if row["purpose"] == "timing"]
    assert report["medians"]["structured"]["optimizer_seconds"] == np.median(timing_values)
    from paper.experiments.quantum_process_tomography_structured_jax import load_structured_result
    for row in report["runs"]:
        assert row["status"] == "success" and row["compact_input_unchanged"] is True
        assert row["scheduler_environment"]["SLURM_JOB_ID"] == "test-job-id"
        assert row["compact_input_sha256"] == before
        assert row["allocator_environment"]["XLA_FLAGS"] == environment["XLA_FLAGS"]
        assert row["metric_mode"] == "sampled"
        for field in ("final_tp_violation", "final_smoothed_objective", "final_process_fidelity_proxy"):
            assert np.isfinite(row[field])
        saved = load_structured_result(row["structured_result_path"])
        assert saved.checkpoint_factors.shape == (0, 4, 1)
        assert saved.checkpoint_steps.tolist() == [0, 1, 2]
        assert "final_chi" not in vars(saved)
        np.testing.assert_array_equal(saved.final_factor, np.load(row["final_factor_path"], allow_pickle=False))
        assert saved.metadata["observation_mode"] == "stored"
        assert row["final_factor_fingerprint"] == benchmark.array_fingerprint(saved.final_factor)
        assert row["structured_result_sha256"] == benchmark.file_sha256(row["structured_result_path"])
        if row["purpose"] == "timing":
            assert row["memory"]["external_sampler"] is None
        else:
            assert row["memory"]["external_sampler"]["gpu"]["status"] == "disabled"


def test_size_guard_runs_before_data_load_or_worker(compact, tmp_path, benchmark, monkeypatch):
    from paper.experiments.qpt_structured_data import StructuredQPTData
    def forbidden(*args, **kwargs):
        raise AssertionError("No loading or workers permitted after failed size guard")
    monkeypatch.setattr(StructuredQPTData, "load_npz", forbidden)
    monkeypatch.setattr(benchmark, "collect_run", forbidden)
    target = tmp_path / "small-cap.json"
    assert benchmark.main(arguments(compact, target) + ["--max-data-mib", "0.0001"]) == 1
    report = json.loads(target.read_text())
    assert report["status"] == "failed" and report["runs"] == []
    assert "max-data-mib" in report["error"]


def test_broken_output_symlink_is_preserved(compact, tmp_path, benchmark):
    target = tmp_path / "report.json"
    destination = tmp_path / "missing.json"
    target.symlink_to(destination)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        benchmark.main(arguments(compact, target))
    assert target.is_symlink() and not destination.exists()


def test_no_hdf_access_and_input_mutation_is_rejected(compact, tmp_path, benchmark, monkeypatch):
    from paper.experiments.qpt_structured_data import StructuredQPTData
    from paper.experiments.quantum_process_tomography import QPTData
    def forbidden(*args, **kwargs):
        raise AssertionError("Compact mode must never open HDF5")
    monkeypatch.setattr(StructuredQPTData, "from_hdf5", forbidden)
    monkeypatch.setattr(QPTData, "from_hdf5", forbidden)
    def mutate(*args, **kwargs):
        with compact.open("ab") as handle:
            handle.write(b"changed input")
        return dict(backend="structured", purpose="timing", repeat=0,
                    status="failed", error="synthetic failed worker", memory={})
    monkeypatch.setattr(benchmark, "collect_run", mutate)
    target = tmp_path / "changed.json"
    assert benchmark.main(arguments(compact, target) + ["--repeats", "1"]) == 1
    report = json.loads(target.read_text())
    assert "Compact source changed" in report["error"]
    assert len(report["runs"]) == 1


def test_noiseless_or_corrupted_numeric_provenance_rejected(compact, tmp_path, benchmark, monkeypatch):
    from paper.experiments.qpt_structured_data import StructuredQPTData
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid compact input must not start workers")
    monkeypatch.setattr(benchmark, "collect_run", forbidden)
    for name in ("noiseless", "corrupt"):
        data = StructuredQPTData.load_npz(compact)
        if name == "noiseless":
            data.observation_mode, data.observations = "noiseless", None
        else:
            data.observations[0] += 0.5
        path = tmp_path / (name + ".npz")
        data.save_npz(path)
        target = tmp_path / (name + ".json")
        assert benchmark.main(arguments(path, target)) == 1
        report = json.loads(target.read_text())
        assert report["status"] == "failed" and report["runs"] == []
        assert ("stored observations" if name == "noiseless" else "numeric SHA-256") in report["error"]
