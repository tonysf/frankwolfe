"""Run the requested BMMC large-batch CPU experiment through 20,000 updates.

The immutable numerical snapshot is the existing FP64 runner. Clean 12-hour
chunks resume plan, momentum, RNG and schedule counter until the target is met.
An unsuccessful chunk stops the controller and preserves its evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/bmmc_cpu_batch4096_20261018"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison(metadata, config):
    points = metadata["trace"]
    latest = points[-1]
    best = min(points, key=lambda p: p["rounded_loss"])
    target = config["pot_reference"]["feasible_unregularized_loss"]
    return dict(status=metadata["status"], updated_at=now(),
                completed_updates=metadata["completed_updates"], target_updates=20000,
                batch_size=4096, seed=811, initial_loss=points[0]["rounded_loss"],
                latest_diagnostic=latest, best_diagnostic=best, pot_target=target,
                latest_measured_loss_below_pot=latest["rounded_loss"] < target,
                best_measured_loss_below_pot=best["rounded_loss"] < target,
                latest_loss_minus_pot=latest["rounded_loss"] - target,
                elapsed_seconds=metadata["elapsed_seconds"],
                interpretation="Single-run feasible unregularized GW loss comparison; best observed and final/last observed losses are separate. CPU and CUDA RNG streams can differ. This does not establish superiority across seeds.")


def main():
    config = json.loads((OUT / "config.json").read_text())
    with (OUT / "controller.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, expected in config["source_sha256"].items():
            if sha256(OUT / "source" / name) != expected:
                raise RuntimeError(f"Frozen source checksum mismatch: {name}")
        run_dir = OUT / "runs" / "frames_q4096_seed811"
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = run_dir / "metadata.json"
        state_path = OUT / "execution.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            started_at=now(), chunks=[], config_sha256=sha256(OUT / "config.json"))
        if state["config_sha256"] != sha256(OUT / "config.json"):
            raise RuntimeError("Run configuration changed")
        state.update(status="running", controller_pid=os.getpid(), updated_at=now())
        save(state_path, state)
        while True:
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
            previous = metadata["completed_updates"] if metadata else 0
            if metadata and metadata["status"] == "completed":
                if previous != 20000:
                    raise RuntimeError("Completed run has the wrong horizon")
                save(OUT / "comparison.json", comparison(metadata, config))
                state.update(status="completed", completed_updates=previous, finished_at=now())
                save(state_path, state)
                return
            tick = time.monotonic()
            chunk = dict(index=len(state["chunks"]), started_at=now(), from_update=previous)
            env = dict(os.environ, **config["thread_environment"])
            with (OUT / "process.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(config["argv"], cwd=OUT / "source", env=env,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                chunk.update(pid=child.pid, status="running")
                state["chunks"].append(chunk)
                save(state_path, state)
                result = child.wait()
            chunk.update(returncode=result, process_seconds=time.monotonic()-tick, finished_at=now())
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
            current = metadata["completed_updates"] if metadata else previous
            chunk.update(completed_updates=current, runner_status=metadata.get("status") if metadata else None)
            valid = result == 0 and metadata and metadata["status"] in ("completed", "checkpointed") and current > previous
            chunk["status"] = "finished" if valid else "failed"
            state.update(status="running" if valid else "failed", completed_updates=current, updated_at=now())
            save(state_path, state)
            if not valid:
                raise RuntimeError("CPU chunk failed or made no progress; inspect process.log and run receipts")
            save(OUT / "comparison.json", comparison(metadata, config))


if __name__ == "__main__":
    main()
