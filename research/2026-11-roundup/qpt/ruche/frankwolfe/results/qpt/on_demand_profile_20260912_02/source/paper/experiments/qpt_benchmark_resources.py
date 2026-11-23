"""Optional, bounded resource telemetry for isolated QPT benchmark workers.

This module imports neither JAX nor a GPU library.  The worker reports its own
``ru_maxrss`` and existing JAX device allocator statistics; its parent can sample
Linux ``/proc`` and ``nvidia-smi`` for that worker PID only.  Missing observations
are represented by ``None``, never by a fabricated zero-byte GPU peak.

The sampled NVIDIA peak is a lower bound: short allocations between polls may
be missed.  It includes the process's driver-visible GPU memory, while JAX's
``peak_bytes_in_use`` describes its allocator, not total driver memory.  See
https://docs.nvidia.com/deploy/nvidia-smi/index.html and
https://docs.jax.dev/en/latest/gpu_memory_allocation.html.
"""

from __future__ import annotations

import csv
import io
import math
import numbers
from pathlib import Path
import subprocess
import sys
import threading
import time
from collections.abc import Mapping

try:
    import resource
except ImportError:  # Windows, for example.
    resource = None


def process_peak_rss_bytes():
    """Current process lifetime maximum resident bytes, or None if unavailable.

    Linux reports ``ru_maxrss`` in KiB; macOS reports bytes.  Other platforms
    remain unsupported rather than guessing their units.  This does not include
    children, virtual address space, or GPU memory.
    """
    if resource is None or sys.platform not in ("linux", "darwin"):
        return None
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if not math.isfinite(value) or value < 0:
            return None
        return int(value) * (1024 if sys.platform == "linux" else 1)
    except (AttributeError, OSError, ValueError, OverflowError):
        return None


def _error_text(error):
    """Bound diagnostic storage, including unexpectedly verbose driver errors."""
    return (type(error).__name__ + ": " + str(error))[:1024]


def device_memory_snapshot(device):
    """Read numeric allocator stats from an already selected JAX device.

    Never imports JAX, selects a device, initializes a backend, or synchronizes
    work.  Call after the benchmark's own synchronization.  CPU/older backends
    may not provide statistics.  Non-numeric and non-finite fields are omitted
    so this result remains safe for strict JSON serialization.
    """
    try:
        raw = device.memory_stats()
    except Exception as error:  # Optional driver/backend instrumentation.
        return {"status": "unavailable", "stats": {}, "error": _error_text(error)}
    if raw is None:
        return {"status": "unavailable", "stats": {}, "reason": "backend returned no memory statistics"}
    if not isinstance(raw, Mapping):
        return {"status": "unavailable", "stats": {}, "reason": "backend returned an unsupported statistics format"}
    stats = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, bool):
            continue
        if isinstance(value, numbers.Integral):
            stats[key] = int(value)
        elif isinstance(value, numbers.Real) and math.isfinite(value):
            stats[key] = float(value)
    return {
        "status": "available" if stats else "unavailable",
        "stats": stats,
        "omitted_stat_count": len(raw) - len(stats),
        "scope": "selected JAX device allocator; not total process driver memory",
    }


def _parse_proc_status(text):
    """Extract only Linux resident and resident-high-water values, in bytes."""
    values = {"rss_bytes": None, "high_water_rss_bytes": None}
    names = {"VmRSS:": "rss_bytes", "VmHWM:": "high_water_rss_bytes"}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[0] not in names or fields[2] != "kB":
            continue
        try:
            value = int(fields[1])
        except ValueError:
            continue
        if value >= 0:
            values[names[fields[0]]] = value * 1024
    return values


def _parse_nvidia_smi(text, pid):
    """Parse PID, UUID, memory-MiB CSV; never retain another process's data.

    Duplicate rows for one UUID (possible with unusual/MIG configurations) are
    not added together: use their maximum and report the ambiguity.  A partial
    set of available device measurements is explicitly marked incomplete.
    """
    by_uuid = {}
    matched_rows = unavailable_rows = malformed_rows = duplicate_rows = 0
    omitted_devices = 0
    for fields in csv.reader(io.StringIO(text)):
        if not fields or not any(field.strip() for field in fields):
            continue
        try:
            row_pid = int(fields[0].strip())
        except ValueError:
            malformed_rows += 1
            continue
        if row_pid != pid:
            continue
        matched_rows += 1
        if len(fields) != 3:
            malformed_rows += 1
            unavailable_rows += 1
            continue
        uuid = fields[1].strip()
        raw_memory = fields[2].strip()
        if raw_memory.endswith("MiB"):
            raw_memory = raw_memory[:-3].strip()
        try:
            memory_mib = float(raw_memory)
            valid = bool(uuid) and uuid not in ("N/A", "[N/A]") and math.isfinite(memory_mib) and memory_mib >= 0
        except ValueError:
            valid = False
        if not valid:
            unavailable_rows += 1
            continue
        try:
            memory_bytes = int(memory_mib * 1024 * 1024)
        except OverflowError:
            unavailable_rows += 1
            continue
        if uuid in by_uuid:
            duplicate_rows += 1
            by_uuid[uuid] = max(by_uuid[uuid], memory_bytes)
        elif len(by_uuid) < 64:
            by_uuid[uuid] = memory_bytes
        else:
            omitted_devices += 1
    return {
        "per_gpu_uuid_bytes": by_uuid,
        "process_bytes": sum(by_uuid.values()) if by_uuid else None,
        "matched_rows": matched_rows,
        "unavailable_rows": unavailable_rows,
        "malformed_rows": malformed_rows,
        "duplicate_rows": duplicate_rows,
        "omitted_devices": omitted_devices,
        "complete": bool(by_uuid) and not (unavailable_rows or duplicate_rows or omitted_devices),
    }


class _Aggregate:
    """Constant-space timings, including actual rather than requested cadence."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def summary(self):
        return {
            "count": self.count,
            "min_seconds": self.minimum,
            "max_seconds": self.maximum,
            "mean_seconds": self.total / self.count if self.count else None,
            "total_seconds": self.total,
        }


class ProcessMemorySampler:
    """Sample one child PID from its parent; never launch or stop that child.

    ``with ProcessMemorySampler(child.pid) as monitor: child.wait()`` followed
    by ``monitor.summary()`` is the usual usage.  Polls run in a daemon thread,
    use no shell, and have a bounded NVIDIA query timeout.  ``stop()`` interrupts
    the between-poll wait and joins with a bounded timeout, even for a very long
    requested interval.  Aggregation keeps no sample history.  Per-GPU storage
    is capped at 64 UUIDs and diagnostics at their most recent 1024 characters.
    """

    def __init__(self, pid, interval_seconds=0.1, enable_gpu=True, gpu_query_timeout_seconds=2.0):
        if isinstance(pid, bool) or not isinstance(pid, numbers.Integral) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        for name, value in (("interval_seconds", interval_seconds), ("gpu_query_timeout_seconds", gpu_query_timeout_seconds)):
            if not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        self.pid = int(pid)
        self.interval_seconds = float(interval_seconds)
        self.enable_gpu = bool(enable_gpu)
        self.gpu_query_timeout_seconds = float(gpu_query_timeout_seconds)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._started = self._stopped = self._last_sample = None
        self._sample_count = 0
        self._gaps = _Aggregate()
        self._latencies = _Aggregate()
        self._cpu_samples = self._cpu_failures = 0
        self._cpu_rss = self._cpu_hwm = None
        self._cpu_last_error = None
        self._cpu_supported = sys.platform == "linux"
        self._gpu_queries = self._gpu_successes = self._gpu_failures = 0
        self._gpu_no_match = self._gpu_matched = self._gpu_valid = self._gpu_partial = 0
        self._gpu_unavailable_rows = self._gpu_malformed_rows = self._gpu_duplicates = 0
        self._gpu_omitted_devices = 0
        self._gpu_peak = None
        self._gpu_per_uuid = {}
        self._gpu_last_error = None
        self._gpu_disabled_reason = None if enable_gpu else "GPU sampling disabled"

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    def start(self):
        if self._thread is not None:
            raise RuntimeError("a ProcessMemorySampler can be started only once")
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"qpt-memory-{self.pid}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.gpu_query_timeout_seconds + 1.0)
        if self._stopped is None:
            self._stopped = time.monotonic()

    def _run(self):
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._sample()
            except Exception as error:  # Never bring down the actual benchmark.
                with self._lock:
                    self._gpu_last_error = "monitor exception: " + _error_text(error)
            self._stop_event.wait(max(0.0, self.interval_seconds - (time.monotonic() - started)))

    def _sample(self):
        now = time.monotonic()
        with self._lock:
            self._sample_count += 1
            if self._last_sample is not None:
                self._gaps.add(now - self._last_sample)
            self._last_sample = now
        if self._cpu_supported:
            try:
                values = _parse_proc_status(Path(f"/proc/{self.pid}/status").read_text())
                with self._lock:
                    if values["rss_bytes"] is not None:
                        self._cpu_samples += 1
                        self._cpu_rss = max(self._cpu_rss or 0, values["rss_bytes"])
                    if values["high_water_rss_bytes"] is not None:
                        self._cpu_hwm = max(self._cpu_hwm or 0, values["high_water_rss_bytes"])
                    if all(value is None for value in values.values()):
                        self._cpu_failures += 1
                        self._cpu_last_error = "resident values unavailable (process may have exited)"
            except (OSError, UnicodeError) as error:
                with self._lock:
                    self._cpu_failures += 1
                    self._cpu_last_error = _error_text(error)
        if self._gpu_disabled_reason is None:
            self._sample_gpu()

    def _sample_gpu(self):
        started = time.monotonic()
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=self.gpu_query_timeout_seconds,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(f"nvidia-smi exit {completed.returncode}: {completed.stderr.strip()[:768]}")
            values = _parse_nvidia_smi(completed.stdout, self.pid)
        except (OSError, subprocess.SubprocessError, ValueError, csv.Error, RuntimeError) as error:
            with self._lock:
                self._gpu_failures += 1
                self._gpu_last_error = _error_text(error)
                if isinstance(error, (FileNotFoundError, PermissionError)):
                    self._gpu_disabled_reason = self._gpu_last_error
        else:
            with self._lock:
                self._gpu_successes += 1
                if values["matched_rows"]:
                    self._gpu_matched += 1
                else:
                    self._gpu_no_match += 1
                if values["process_bytes"] is not None:
                    self._gpu_valid += 1
                    self._gpu_peak = max(self._gpu_peak or 0, values["process_bytes"])
                    self._gpu_partial += int(not values["complete"])
                    for uuid, memory in values["per_gpu_uuid_bytes"].items():
                        if uuid in self._gpu_per_uuid or len(self._gpu_per_uuid) < 64:
                            self._gpu_per_uuid[uuid] = max(self._gpu_per_uuid.get(uuid, 0), memory)
                        else:
                            self._gpu_omitted_devices += 1
                self._gpu_unavailable_rows += values["unavailable_rows"]
                self._gpu_malformed_rows += values["malformed_rows"]
                self._gpu_duplicates += values["duplicate_rows"]
                self._gpu_omitted_devices += values["omitted_devices"]
        finally:
            with self._lock:
                self._gpu_queries += 1
                self._latencies.add(time.monotonic() - started)

    def summary(self):
        """Return a detached, strict-JSON-safe telemetry snapshot."""
        with self._lock:
            if self._gpu_valid:
                gpu_status = "sampled"
                gpu_reason = "sampled lower bound; brief peaks and unavailable device rows may be missed"
            elif self._gpu_disabled_reason:
                gpu_status = "disabled" if not self.enable_gpu else "unavailable"
                gpu_reason = self._gpu_disabled_reason
            elif self._gpu_matched:
                gpu_status = "unavailable"
                gpu_reason = "PID matched, but no numeric GPU memory measurement was available"
            elif self._gpu_successes:
                gpu_status = "unavailable"
                gpu_reason = "target PID was not observed in compute-process rows; absence is not zero memory"
            else:
                gpu_status = "unavailable"
                gpu_reason = "no successful GPU query" if self._gpu_queries else "no GPU query performed"
            ended = self._stopped if self._stopped is not None else time.monotonic()
            return {
                "pid": self.pid,
                "cpu": {
                    "status": "sampled" if self._cpu_samples else "unavailable",
                    "source": "/proc/PID/status VmRSS and VmHWM (Linux only)",
                    "peak_rss_bytes": self._cpu_rss,
                    "peak_high_water_rss_bytes": self._cpu_hwm,
                    "valid_samples": self._cpu_samples,
                    "failed_samples": self._cpu_failures,
                    "reason": None if self._cpu_samples else (self._cpu_last_error or ("no samples available" if self._cpu_supported else "Linux /proc is unsupported on this platform")),
                    "last_error": self._cpu_last_error,
                },
                "gpu": {
                    "status": gpu_status,
                    "reason": gpu_reason,
                    "source": "nvidia-smi compute-process memory for the target PID only; MiB converted to bytes",
                    "sampled_peak_process_bytes": self._gpu_peak,
                    "per_gpu_uuid_peak_bytes": dict(self._gpu_per_uuid),
                    "aggregate_note": "peak of each sample's sum; not sum of per-device peaks",
                    "query_count": self._gpu_queries,
                    "successful_queries": self._gpu_successes,
                    "failed_queries": self._gpu_failures,
                    "matched_samples": self._gpu_matched,
                    "valid_memory_samples": self._gpu_valid,
                    "no_matching_pid_samples": self._gpu_no_match,
                    "partial_memory_samples": self._gpu_partial,
                    "unavailable_memory_rows": self._gpu_unavailable_rows,
                    "malformed_rows": self._gpu_malformed_rows,
                    "duplicate_device_rows": self._gpu_duplicates,
                    "omitted_device_observations": self._gpu_omitted_devices,
                    "last_error": self._gpu_last_error,
                },
                "sampling": {
                    "requested_interval_seconds": self.interval_seconds,
                    "elapsed_seconds": max(0.0, ended - self._started) if self._started is not None else 0.0,
                    "sample_count": self._sample_count,
                    "actual_start_gap": self._gaps.summary(),
                    "gpu_query_latency": self._latencies.summary(),
                    "gpu_query_timeout_seconds": self.gpu_query_timeout_seconds,
                    "thread_still_running": self._thread is not None and self._thread.is_alive(),
                    "limitations": "polling overhead is included in monitored wall time; no sample history is retained; short peaks may be missed",
                },
            }
