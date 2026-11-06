"""CPU-only tests for optional GPU/CPU benchmark resource telemetry."""

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from paper.experiments import qpt_benchmark_resources as resources


@pytest.mark.parametrize("platform,multiplier", [("linux", 1024), ("darwin", 1)])
def test_process_peak_rss_converts_platform_units(monkeypatch, platform, multiplier):
    monkeypatch.setattr(resources.sys, "platform", platform)
    monkeypatch.setattr(resources, "resource", SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda target: SimpleNamespace(ru_maxrss=321),
    ))
    assert resources.process_peak_rss_bytes() == 321 * multiplier


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_self_peak_is_unavailable(monkeypatch, value):
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda target: SimpleNamespace(ru_maxrss=value),
    ))
    assert resources.process_peak_rss_bytes() is None


def test_unsupported_or_missing_resource_is_unavailable(monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "unsupported")
    assert resources.process_peak_rss_bytes() is None
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources, "resource", None)
    assert resources.process_peak_rss_bytes() is None


def test_peak_resource_os_error_is_unavailable(monkeypatch):
    def fail(_):
        raise OSError("not supported")
    monkeypatch.setattr(resources.sys, "platform", "linux")
    monkeypatch.setattr(resources, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=fail))
    assert resources.process_peak_rss_bytes() is None


def test_device_snapshot_is_json_numeric_without_initializing_jax():
    device = SimpleNamespace(memory_stats=lambda: {
        "bytes_in_use": 1024, "peak_bytes_in_use": 4096,
        "ratio": 0.5, "nan": float("nan"), "object": object(), "flag": True,
    })
    snapshot = resources.device_memory_snapshot(device)
    assert snapshot["status"] == "available"
    assert snapshot["stats"] == {"bytes_in_use": 1024, "peak_bytes_in_use": 4096, "ratio": 0.5}
    assert snapshot["omitted_stat_count"] == 3
    json.dumps(snapshot, allow_nan=False)


@pytest.mark.parametrize("value", [None, [], {}, {"bad": float("inf")}])
def test_unsupported_allocator_stats(value):
    snapshot = resources.device_memory_snapshot(SimpleNamespace(memory_stats=lambda: value))
    assert snapshot["status"] == "unavailable"
    assert snapshot["stats"] == {}


def test_allocator_errors_are_bounded_and_nonfatal():
    def fail():
        raise RuntimeError("x" * 5000)
    snapshot = resources.device_memory_snapshot(SimpleNamespace(memory_stats=fail))
    assert snapshot["status"] == "unavailable"
    assert len(snapshot["error"]) == 1024
    assert resources.device_memory_snapshot(object())["status"] == "unavailable"


def test_proc_parser_reports_resident_not_virtual_bytes():
    parsed = resources._parse_proc_status("VmSize:\t999 kB\nVmRSS:\t41 kB\nVmHWM:\t53 kB\n")
    assert parsed == {"rss_bytes": 41 * 1024, "high_water_rss_bytes": 53 * 1024}


@pytest.mark.parametrize("text", ["", "VmRSS: N/A kB", "VmRSS: -1 kB", "VmRSS: 8 MB", "VmHWM: 1"])
def test_proc_parser_unknown_values_are_null(text):
    assert resources._parse_proc_status(text) == {"rss_bytes": None, "high_water_rss_bytes": None}


def test_nvidia_parser_only_target_pid_and_mib_units():
    parsed = resources._parse_nvidia_smi(
        "999, GPU-other-user, 9000\n42, GPU-aaa, 12\n42, GPU-bbb, 1.5 MiB\n", 42,
    )
    assert parsed["per_gpu_uuid_bytes"] == {"GPU-aaa": 12 * 1024**2, "GPU-bbb": 1572864}
    assert parsed["process_bytes"] == 14155776
    assert parsed["matched_rows"] == 2
    assert parsed["complete"] is True
    assert "other-user" not in str(parsed)


@pytest.mark.parametrize("raw", ["N/A", "[N/A]", "[Not Supported]", "nan", "inf", "-1", "5 KB", "1e308"])
def test_missing_or_invalid_gpu_measurement_is_not_zero(raw):
    parsed = resources._parse_nvidia_smi(f"42, GPU-aaa, {raw}\n", 42)
    assert parsed["matched_rows"] == 1
    assert parsed["unavailable_rows"] == 1
    assert parsed["process_bytes"] is None
    assert parsed["per_gpu_uuid_bytes"] == {}


def test_actual_zero_is_valid_but_no_pid_match_is_null():
    assert resources._parse_nvidia_smi("42, GPU-a, 0", 42)["process_bytes"] == 0
    for text in ("", "41, GPU-a, 20", "No running processes found"):
        assert resources._parse_nvidia_smi(text, 42)["process_bytes"] is None


def test_partial_devices_and_duplicates_are_explicit_and_not_double_counted():
    parsed = resources._parse_nvidia_smi("42, GPU-a, 2\n42, GPU-a, 3\n42, GPU-b, N/A", 42)
    assert parsed["process_bytes"] == 3 * 1024**2
    assert parsed["duplicate_rows"] == 1
    assert parsed["unavailable_rows"] == 1
    assert parsed["complete"] is False


def test_telemetry_uuid_storage_is_bounded():
    parsed = resources._parse_nvidia_smi("\n".join(f"42, GPU-{i}, 1" for i in range(100)), 42)
    assert len(parsed["per_gpu_uuid_bytes"]) == 64
    assert parsed["omitted_devices"] == 36
    assert parsed["complete"] is False


def make_sampler_without_proc(monkeypatch, **kwargs):
    monitor = resources.ProcessMemorySampler(42, **kwargs)
    monitor._cpu_supported = False
    return monitor


def test_external_queries_are_pid_filtered_and_shell_free(monkeypatch):
    calls = []
    outputs = iter([
        "42, GPU-a, 5\n42, GPU-b, 1\n100, GPU-z, 8000",
        "42, GPU-a, 1\n42, GPU-b, 8",
        "100, GPU-z, 8000",
    ])
    def query(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=next(outputs), stderr="")
    monkeypatch.setattr(resources.subprocess, "run", query)
    monitor = make_sampler_without_proc(monkeypatch)
    for _ in range(3):
        monitor._sample()
    summary = monitor.summary()
    assert summary["gpu"]["sampled_peak_process_bytes"] == 9 * 1024**2
    # Individual maxima occurred at different instants: their sum is not peak.
    assert summary["gpu"]["per_gpu_uuid_peak_bytes"] == {"GPU-a": 5 * 1024**2, "GPU-b": 8 * 1024**2}
    assert summary["gpu"]["valid_memory_samples"] == 2
    assert summary["gpu"]["no_matching_pid_samples"] == 1
    assert summary["sampling"]["actual_start_gap"]["count"] == 2
    assert summary["sampling"]["gpu_query_latency"]["count"] == 3
    assert all(command == ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"] for command, _ in calls)
    assert all(not kwargs.get("shell", False) and kwargs["timeout"] == 2.0 for _, kwargs in calls)
    json.dumps(summary, allow_nan=False)


def test_missing_nvidia_smi_disables_repeated_queries_but_not_monitor(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(resources.subprocess, "run", missing)
    monitor = make_sampler_without_proc(monkeypatch)
    monitor._sample()
    monitor._sample()
    result = monitor.summary()
    assert result["gpu"]["query_count"] == 1
    assert result["gpu"]["sampled_peak_process_bytes"] is None
    assert result["gpu"]["status"] == "unavailable"
    assert "FileNotFoundError" in result["gpu"]["reason"]
    assert result["sampling"]["sample_count"] == 2


@pytest.mark.parametrize("failure", ["timeout", "returncode"])
def test_gpu_query_failures_do_not_stop_sampling(monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("nvidia-smi", 2)
        return SimpleNamespace(returncode=9, stdout="", stderr="driver unavailable")
    monkeypatch.setattr(resources.subprocess, "run", fail)
    monitor = make_sampler_without_proc(monkeypatch)
    monitor._sample()
    monitor._sample()
    gpu = monitor.summary()["gpu"]
    assert gpu["failed_queries"] == 2
    assert gpu["query_count"] == 2
    assert gpu["sampled_peak_process_bytes"] is None
    assert gpu["last_error"]


@pytest.mark.parametrize("stdout,reason", [("41, GPU-a, 9", "not observed"), ("42, GPU-a, N/A", "no numeric")])
def test_unavailable_gpu_reasons_distinguish_no_pid_and_na(monkeypatch, stdout, reason):
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout, stderr=""))
    monitor = make_sampler_without_proc(monkeypatch)
    monitor._sample()
    gpu = monitor.summary()["gpu"]
    assert reason in gpu["reason"]
    assert gpu["sampled_peak_process_bytes"] is None


def test_context_stop_interrupts_long_interval_and_propagates_exception(monkeypatch):
    sampled = threading.Event()
    def fake_sample(self):
        sampled.set()
    monkeypatch.setattr(resources.ProcessMemorySampler, "_sample", fake_sample)
    monitor = resources.ProcessMemorySampler(42, interval_seconds=3600, enable_gpu=False)
    started = time.monotonic()
    with pytest.raises(ValueError, match="caller error"):
        with monitor:
            assert sampled.wait(timeout=2)
            raise ValueError("caller error")
    assert time.monotonic() - started < 2
    assert not monitor.summary()["sampling"]["thread_still_running"]
    monitor.stop()  # Idempotent.
    with pytest.raises(RuntimeError, match="only once"):
        monitor.start()


@pytest.mark.parametrize("kwargs", [{"pid": -1}, {"pid": True}, {"pid": 1.2}, {"pid": 42, "interval_seconds": 0}, {"pid": 42, "interval_seconds": float("nan")}, {"pid": 42, "gpu_query_timeout_seconds": -1}])
def test_invalid_sampler_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        resources.ProcessMemorySampler(**kwargs)


def test_gpu_disabled_does_not_launch_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("GPU-disabled sampling must not launch a subprocess")
    monkeypatch.setattr(resources.subprocess, "run", forbidden)
    monitor = make_sampler_without_proc(monkeypatch, enable_gpu=False)
    monitor._sample()
    assert monitor.summary()["gpu"]["status"] == "disabled"
    assert monitor.summary()["gpu"]["query_count"] == 0


def test_proc_sampler_tracks_peak_and_missing_process_without_zero(monkeypatch):
    values = iter([
        "VmRSS: 400 kB\nVmHWM: 450 kB",
        "VmRSS: 300 kB\nVmHWM: 900 kB",
    ])
    def status(path):
        assert str(path) == "/proc/42/status"
        try:
            return next(values)
        except StopIteration:
            raise FileNotFoundError("process exited")
    monkeypatch.setattr(resources.Path, "read_text", status)
    monitor = resources.ProcessMemorySampler(42, enable_gpu=False)
    monitor._cpu_supported = True
    for _ in range(3):
        monitor._sample()
    cpu = monitor.summary()["cpu"]
    assert cpu["peak_rss_bytes"] == 400 * 1024
    assert cpu["peak_high_water_rss_bytes"] == 900 * 1024
    assert cpu["valid_samples"] == 2
    assert cpu["failed_samples"] == 1
    assert cpu["last_error"]


def test_stopping_during_gpu_query_waits_for_its_bounded_timeout(monkeypatch):
    querying = threading.Event()
    def timeout(*args, **kwargs):
        querying.set()
        time.sleep(kwargs["timeout"])
        raise subprocess.TimeoutExpired("nvidia-smi", kwargs["timeout"])
    monkeypatch.setattr(resources.subprocess, "run", timeout)
    monitor = make_sampler_without_proc(monkeypatch, gpu_query_timeout_seconds=0.02)
    with monitor:
        assert querying.wait(timeout=2)
    assert not monitor.summary()["sampling"]["thread_still_running"]
    assert monitor.summary()["gpu"]["failed_queries"] == 1


@pytest.mark.skipif(sys.platform != "linux", reason="external RSS reads Linux /proc")
def test_linux_child_resident_memory_smoke():
    # Keep allocation alive until stdin closes.  No GPU and no remote process.
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; a=bytearray(8*1024*1024); print('ready',flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with resources.ProcessMemorySampler(child.pid, interval_seconds=0.01, enable_gpu=False) as monitor:
            deadline = time.monotonic() + 2
            while monitor.summary()["cpu"]["valid_samples"] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
        summary = monitor.summary()
        assert summary["cpu"]["peak_rss_bytes"] >= 8 * 1024**2
        assert summary["cpu"]["peak_high_water_rss_bytes"] >= summary["cpu"]["peak_rss_bytes"]
        assert child.poll() is None  # Stopping the monitor never stops the worker.
    finally:
        child.stdin.close()
        child.wait(timeout=5)
        child.stdout.close()
