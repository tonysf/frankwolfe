"""Rerun each original SFRAMES arm with a paired random LMO start."""

from run import ROOT, run, sha256, write_json

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def check_pair(config):
    name = config["run_id"]
    original = ROOT.parent / "runs" / name
    rerun = ROOT / "runs" / name
    traces = [read_json(folder / "trace.json") for folder in (original, rerun)]
    keys = ("iteration", "forward_views", "backward_views", "gradient_calls")
    original_work, rerun_work = (
        [tuple(probe[key] for key in keys) for probe in trace] for trace in traces)
    if original_work != rerun_work:
        raise ValueError(f"Probe work differs for {name}")
    states = [pickle.loads((folder / "checkpoint.pkl").read_bytes())["solver"]
              for folder in (original, rerun)]
    if states[0]["rng"] != states[1]["rng"]:
        raise ValueError(f"Terminal minibatch RNG differs for {name}")
    trace = traces[1]
    max_cap_excess = max(row["c_violation"] for row in trace)
    if max_cap_excess > 1e-11 or trace[0]["d_distance"] <= 0:
        raise ValueError(f"Expected C-feasible trajectory and D-infeasible start for {name}")
    return dict(matching_probes=len(trace), initial_d_distance=trace[0]["d_distance"],
                final_d_distance=trace[-1]["d_distance"], max_c_violation=max_cap_excess)


def main():
    lock = (ROOT / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = ROOT.parent
    original_manifest = read_json(parent / "manifest.json")
    if any(sha256(parent / name) != digest for name, digest in original_manifest["sources"].items()):
        raise ValueError("Original source no longer matches the frozen CT campaign")
    paired = [read_json(path) for path in sorted((parent / "runs").glob("*sframes*/result.json"))]
    if len(paired) != 6 or any(result["config"]["method"] != "sframes" for result in paired):
        raise ValueError("Expected the two development and four evaluation SFRAMES runs")
    if not {result["cpu"] for result in paired}.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Original worker affinities unavailable")
    manifest = dict(
        source_sha256={name: sha256(ROOT / name)
                       for name in ("campaign.py", "run.py", "ct_problem.py", "methods.py")},
        original_manifest_sha256=sha256(parent / "manifest.json"),
        design="Original SFRAMES settings and calibration; change only the initial factors.",
        direction_rng="Separate NumPy Generator with seed=original config seed+200000.",
        rate_selection="Reuse both development rates and the frozen evaluation rate.",
        runs=[dict(config=r["config"], cpu=r["cpu"], original_data_sha256=r["data_sha256"])
              for r in paired])
    path = ROOT / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Rerun provenance changed")
    write_json(path, manifest)
    state = dict(status="running", started_utc=now(),
                 runs={r["config"]["run_id"]: {"status": "queued"} for r in paired})
    status_path = ROOT / "progress.json"
    write_json(status_path, state)
    try:
        for phase in ("development", "evaluation"):
            cohort = [r for r in paired if r["config"]["phase"] == phase]
            state["phase"] = phase
            write_json(status_path, state)
            with ProcessPoolExecutor(max_workers=len(cohort), mp_context=mp.get_context("spawn")) as pool:
                submitted = {pool.submit(run, r["config"], r["cpu"]): r["config"] for r in cohort}
                for future in as_completed(submitted):
                    config = submitted[future]
                    try:
                        row = future.result()
                        paired_check = check_pair(config)
                        item = dict(status="completed", iteration=row["iteration"],
                                    forward_views=row["forward_views"], pairing=paired_check)
                    except Exception as exc:
                        item = dict(status="failed", error=repr(exc))
                    state["runs"][config["run_id"]] = item
                    write_json(status_path, state)
                    print(config["run_id"], item, flush=True)
        subprocess.run([sys.executable, str(ROOT / "report.py")], check=True)
        if any(item["status"] != "completed" for item in state["runs"].values()):
            raise RuntimeError("At least one initialization rerun failed")
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(status_path, dict(state, finished_utc=now()))


if __name__ == "__main__":
    main()
