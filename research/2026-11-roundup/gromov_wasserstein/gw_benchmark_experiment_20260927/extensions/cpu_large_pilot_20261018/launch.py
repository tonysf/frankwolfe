"""Serial, resource-bounded CPU calibration of the frozen large GW runners.

No parameter search or GPU submission. Each pilot is an initial prefix of the
original horizon and keeps a resumable checkpoint. Limits are operational:
accepted optimizer steps retain the production arithmetic and certificates.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/cpu_large_pilot_20261018"
PYTHON = ROOT / ".venv/bin/python"
SOFT_SECONDS = 300
HARD_SECONDS = 900
MAX_RSS = int(9.5 * 1024**3)
MIN_AVAILABLE = int(1.5 * 1024**3)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def memory(pid):
    def kb_values(path):
        result = {}
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[2] == "kB":
                result[fields[0].rstrip(":")] = int(fields[1]) * 1024
        return result
    try:
        proc = kb_values(Path(f"/proc/{pid}/status"))
    except FileNotFoundError:
        proc = {}
    return dict(rss_bytes=proc.get("VmRSS", 0),
                high_water_bytes=proc.get("VmHWM", 0),
                available_bytes=kb_values(Path("/proc/meminfo"))["MemAvailable"])


def run_pilot(dataset, method):
    run_dir = OUT / "runs" / f"{dataset}_{method}"
    run_dir.mkdir(parents=True, exist_ok=False)
    args = [str(PYTHON), str(ROOT / "extensions/cpu_large_pilot_20261018/profile_runner.py"),
            "--profile-out", str(run_dir),
            "--instance", str(OUT / "data" / dataset), "--out", str(run_dir),
            "--method", method, "--device", "cpu", "--dtype", "float64",
            "--seed", "811" if method == "frames" else "810",
            "--iterations", "20000" if method == "frames" else "2000",
            "--chunk-updates", "100" if method == "frames" else "30",
            "--max-seconds", str(SOFT_SECONDS), "--checkpoint-seconds", "1800",
            "--checkpoint-every", "5000", "--retain-checkpoint",
            "--pairs", "1024", "--epsilon", ".001", "--gamma0", ".3",
            "--rho0", "1" if dataset == "bmmc6224" else ".1",
            "--mu-scale", "10", "--inner-max", "100000",
            "--diagnostic-every", "500", "--exact-block-rows", "128"]
    args = ["taskset", "-c", "0,2,4,6"] + args
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
               OMP_PROC_BIND="true", OMP_PLACES="cores", MALLOC_ARENA_MAX="2")
    record = dict(dataset=dataset, method=method, argv=args, started_at=now(),
                  soft_seconds=SOFT_SECONDS, hard_seconds=HARD_SECONDS,
                  cpu_ids=[0, 2, 4, 6], num_threads=4, status="running",
                  environment={k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_PROC_BIND",
                      "OMP_PLACES", "MALLOC_ARENA_MAX")})
    started = time.monotonic()
    peak = 0
    min_available = memory(os.getpid())["available_bytes"]
    reason = None
    with (run_dir / "process.log").open("wb") as log:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = child.pid
        save(run_dir / "receipt.json", record)
        while child.poll() is None:
            try:
                child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                observed = memory(child.pid)
                peak = max(peak, observed["rss_bytes"], observed["high_water_bytes"])
                min_available = min(min_available, observed["available_bytes"])
                if observed["rss_bytes"] > MAX_RSS or observed["available_bytes"] < MIN_AVAILABLE:
                    reason = "memory_guard"
                elif time.monotonic() - started > HARD_SECONDS:
                    reason = "hard_wall_limit"
                if reason:
                    # TERM cannot interrupt this runner's inner Sinkhorn loop.
                    # Enforce the resource bound, keeping prior committed state.
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    break
    record.update(finished_at=now(), process_seconds=time.monotonic()-started,
                  returncode=child.returncode, stop_reason=reason,
                  peak_rss_bytes_sampled=peak, minimum_available_bytes_sampled=min_available,
                  status="finished" if child.returncode == 0 else "interrupted" if reason else "failed")
    metadata = run_dir / "metadata.json"
    if metadata.exists():
        final = json.loads(metadata.read_text())
        record.update(runner_status=final["status"], completed_updates=final["completed_updates"],
                      runner_elapsed_seconds=final["elapsed_seconds"])
    save(run_dir / "receipt.json", record)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "campaign.json").exists():
        raise RuntimeError("Pilot already launched; refusing duplicate work")
    data = {}
    for dataset in ("bmmc6224", "snare_full"):
        folder = OUT / "data" / dataset
        manifest = json.loads((folder / "manifest.json").read_text())
        hashes = {name: digest(folder / name) for name in ("A.npy", "B.npy", "a.npy", "b.npy")}
        if any(hashes[k] != manifest["array_files_sha256"][k] for k in hashes):
            raise RuntimeError(f"Data checksum mismatch: {dataset}")
        data[dataset] = dict(hashes=hashes, manifest_sha256=digest(folder / "manifest.json"),
                             n=manifest["n"], m=manifest["m"])
    campaign = dict(started_at=now(), host=platform.node(), platform=platform.platform(),
                    git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    sources={str(p.relative_to(ROOT)): digest(p) for p in Path(__file__).parent.glob("*.py")},
                    lscpu=subprocess.check_output(["lscpu"], text=True), data=data,
                    status="running", runs=[], pid=os.getpid(),
                    note="Initial-prefix timing pilots, not new full-horizon loss comparisons; one four-core process at a time.")
    save(OUT / "campaign.json", campaign)
    for dataset, method in (("bmmc6224", "frames"), ("bmmc6224", "pot-full"),
                            ("snare_full", "frames"), ("snare_full", "pot-full")):
        result = run_pilot(dataset, method)
        campaign["runs"].append(result)
        save(OUT / "campaign.json", campaign)
        if result.get("stop_reason") == "memory_guard":
            campaign["status"] = "stopped_memory_guard"
            break
    else:
        campaign["status"] = "finished"
    campaign["finished_at"] = now()
    save(OUT / "campaign.json", campaign)


if __name__ == "__main__":
    main()
