"""Measured-benchmark integration: tiny CPU workers, never cluster access."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_qpt_structured.py"


@pytest.fixture(scope="module")
def benchmark_module():
    spec = importlib.util.spec_from_file_location("qpt_measured_benchmark_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def small_qpt_h5(tmp_path):
    h5py = pytest.importorskip("h5py")
    from paper.experiments.qpt_structured_data import standard_local_basis, standard_local_measurements
    path = tmp_path / "tiny.h5"
    basis = standard_local_basis()
    with h5py.File(path, "w") as handle:
        handle["f_jax_vector"] = np.random.default_rng(81).uniform(0.0, 1.0, 24)
        handle["D_jax_tensors"] = standard_local_measurements()
        handle["A_jax_basis"] = basis
        handle["B_jax_tensors"] = np.einsum("mki,nkj->nmij", basis.conj(), basis)
    return path


def tiny_arguments(source, report):
    return [
        "--h5", str(source), "--save", str(report), "--device", "cpu",
        "--steps", "2", "--batch-size", "3", "--chunk-steps", "1",
        "--metrics-every", "0", "--metric-mode", "full", "--metric-batch-size", "7",
        "--step-scale", "0.1", "--repeats", "1", "--profile-memory",
        "--memory-poll-ms", "20",
    ]


def test_profile_cli_cpu_preserves_artifacts_and_separates_timing_medians(
    tmp_path, small_qpt_h5, benchmark_module,
):
    pytest.importorskip("jax")
    path = tmp_path / "measured.json"
    arguments = tiny_arguments(small_qpt_h5, path)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)] + arguments,
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(path.read_text())
    assert report["status"] == "complete"
    assert report["observations_preserved"] is True
    assert report["parity_passed"] is True
    assert report["final_factor_max_abs_difference"] < 1e-10
    assert report["final_full_loss_abs_difference"] < 1e-10
    assert report["final_full_gap_abs_difference"] < 1e-10
    assert len(report["runs"]) == 4
    assert {(row["backend"], row["purpose"]) for row in report["runs"]} == {
        (backend, purpose) for backend in ("dense", "structured") for purpose in ("timing", "memory")
    }
    assert len({row["pid"] for row in report["runs"]}) == 4
    assert (Path(report["artifacts_dir"]) / "data.npz").is_file()
    for row in report["runs"]:
        assert row["status"] == "success"
        assert row["returncode"] == 0
        assert row["device_platform"] == "cpu"
        assert row["allocator_environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert row["allocator_environment"]["JAX_ENABLE_COMPILATION_CACHE"] == "false"
        if sys.platform in ("linux", "darwin"):
            assert row["memory"]["cpu_process_peak_rss_bytes"] > 0
        for field in ("final_factor_path", "stdout_path", "stderr_path", "worker_report_path"):
            assert Path(row[field]).is_file()
        np.testing.assert_equal(np.load(row["final_factor_path"], allow_pickle=False).shape, (4, 1))
        if row["purpose"] == "timing":
            assert row["memory"]["external_sampler"] is None
            assert report["medians"][row["backend"]]["optimizer_seconds"] == row["optimizer_seconds"]
        else:
            sampler = row["memory"]["external_sampler"]
            assert sampler["gpu"]["status"] == "disabled"
            assert sampler["gpu"]["sampled_peak_process_bytes"] is None
            assert sampler["gpu"]["query_count"] == 0
            assert sampler["sampling"]["sample_count"] > 0
            assert sampler["sampling"]["thread_still_running"] is False
            saved_worker = json.loads(Path(row["worker_report_path"]).read_text())
            assert saved_worker["memory"]["external_sampler"] == sampler
    assert report["memory_ratios"]["dense_over_structured_sampled_process_vram_peak"] is None

    # A deliberately enormous profiling time must not contaminate timing medians.
    modified = copy.deepcopy(report)
    for row in modified["runs"]:
        if row["purpose"] == "memory":
            row["optimizer_seconds"] = 1e12
    benchmark_module.summarize(modified, benchmark_module.parser().parse_args(arguments))
    assert modified["medians"] == report["medians"]


def test_require_gpu_memory_cpu_failure_still_saves_partial_report(tmp_path, small_qpt_h5):
    pytest.importorskip("jax")
    path = tmp_path / "required.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)] + tiny_arguments(small_qpt_h5, path)
        + ["--require-gpu-memory", "--backends", "structured"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert len(report["runs"]) == 2
    timing, profile = report["runs"]
    assert timing["status"] == "success"
    assert profile["status"] == "failed"
    assert "GPU-memory" in profile["error"]
    assert profile["memory"]["external_sampler"]["gpu"]["sampled_peak_process_bytes"] is None
    assert Path(profile["final_factor_path"]).exists()
    assert json.loads(Path(profile["worker_report_path"]).read_text())["status"] == "failed"
    assert report["medians"]["structured"]["optimizer_seconds"] == timing["optimizer_seconds"]


@pytest.mark.parametrize("allocator,preallocation", [("grow", "false"), ("default", "true")])
def test_worker_environment_explicit_allocator_without_changing_gpu_binding(
    benchmark_module, monkeypatch, allocator, preallocation,
):
    inherited = {
        "CUDA_VISIBLE_DEVICES": "GPU-slurm-bound",
        "JAX_PLATFORMS": "cuda,cpu",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "inherited",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.98",
        "XLA_CLIENT_MEM_FRACTION": "0.99",
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        "JAX_COMPILATION_CACHE_DIR": "/some/cache",
        "JAX_ENABLE_COMPILATION_CACHE": "true",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    configured = benchmark_module.configured_worker_environment(SimpleNamespace(allocator=allocator))
    assert configured["XLA_PYTHON_CLIENT_ALLOCATOR"] == "default"
    assert configured["XLA_PYTHON_CLIENT_PREALLOCATE"] == preallocation
    assert configured["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    for key in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION", "TF_GPU_ALLOCATOR", "JAX_COMPILATION_CACHE_DIR"):
        assert key not in configured
    assert configured["CUDA_VISIBLE_DEVICES"] == "GPU-slurm-bound"
    assert configured["JAX_PLATFORMS"] == "cuda,cpu"
    # Building worker settings does not alter the parent's environment.
    assert benchmark_module.os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"


def test_collect_run_crashed_worker_without_json_keeps_failure_artifacts(tmp_path, benchmark_module, monkeypatch):
    commands = []
    class FailedProcess:
        pid = 12345
        def wait(self):
            return 9
    def launch(command, **kwargs):
        commands.append(command)
        return FailedProcess()
    monkeypatch.setattr(benchmark_module.subprocess, "Popen", launch)
    args = benchmark_module.parser().parse_args([
        "--h5", str(tmp_path / "source.h5"), "--save", str(tmp_path / "report.json"), "--device", "cpu",
    ])
    row = benchmark_module.collect_run(
        args, ["--h5", "relative-source.h5"], "dense", "timing", 0, tmp_path, tmp_path / "data.npz", {},
    )
    assert row["status"] == "failed"
    assert row["returncode"] == 9
    assert "without a report" in row["error"]
    for key in ("worker_report_path", "stdout_path", "stderr_path"):
        assert Path(row[key]).is_file()
    assert json.loads(Path(row["worker_report_path"]).read_text())["status"] == "failed"
    # The final duplicate option wins argparse parsing in the worker, preserving
    # the parent's resolved path even though child cwd is the repository root.
    command = commands[0]
    final_h5_option = max(index for index, value in enumerate(command) if value == "--h5")
    assert command[final_h5_option + 1] == str(args.h5)


@pytest.mark.parametrize("value", [0, 1024, None, -1, float("nan"), float("inf"), True])
def test_gpu_peak_extractors_keep_real_zero_distinct_from_missing(benchmark_module, value):
    row = {"memory": {
        "jax_allocator": {"stats": {"peak_bytes_in_use": value}},
        "external_sampler": {"gpu": {"sampled_peak_process_bytes": value}},
    }}
    expected = value if type(value) is int and value >= 0 else None
    assert benchmark_module.allocator_peak(row) == expected
    assert benchmark_module.sampled_gpu_peak(row) == expected


def test_parent_failure_report_survives_worker_failure(
    tmp_path, small_qpt_h5, benchmark_module, monkeypatch,
):
    def failed_worker(*args, **kwargs):
        return {"backend": "structured", "purpose": "timing", "repeat": 0,
                "status": "failed", "returncode": 9, "error": "simulated worker OOM", "memory": {}}
    monkeypatch.setattr(benchmark_module, "collect_run", failed_worker)
    report_path = tmp_path / "failure.json"
    result = benchmark_module.main([
        "--h5", str(small_qpt_h5), "--save", str(report_path), "--device", "cpu",
        "--steps", "2", "--repeats", "1", "--backends", "structured",
    ])
    assert result == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["runs"][0]["error"] == "simulated worker OOM"
    assert report["medians"] == {}
    assert report["conversion_seconds"] >= 0
    assert (Path(report["artifacts_dir"]) / "data.npz").is_file()
