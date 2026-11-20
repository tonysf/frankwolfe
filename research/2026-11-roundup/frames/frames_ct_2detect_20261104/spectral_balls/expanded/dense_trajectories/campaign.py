"""Replay the frozen evaluation cohort with diagnostics at every iteration."""

from run import ROOT, run, sha256, write_json

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys


CPUS = (0, 2, 4, 6)
ORIGINAL = ROOT.parent
SOURCES = ("campaign.py", "run.py", "methods.py", "ct_problem.py",
           "spectral_reference_core.py", "spectral_reference_baseline.py")
SCIENTIFIC_FIELDS = (
    "iteration", "forward_views", "backward_views", "gradient_calls",
    "projection_calls", "projection_inner_iterations", "projection_dykstra_calls",
    "projection_floor_count", "max_projection_distance_bound", "max_lmo_support_gap_bound",
    "raw_loss", "feasible_loss", "validation_loss", "reference_relative_error",
    "max_constraint_violation", "repair_distance", "c_violation", "d_distance",
)


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen configuration changed at {path}")
    write_json(path, value)


def validate(config):
    folder = ROOT / "runs" / config["run_id"]
    dense = read(folder / "trace.json")
    result = read(folder / "result.json")
    old_folder = ORIGINAL / "runs" / config["run_id"]
    old = read(old_folder / "trace.json")
    if [row["iteration"] for row in dense] != list(range(result["iteration"] + 1)):
        raise ValueError(f"Dense trace is not contiguous for {config['run_id']}")
    if result["final"] != dense[-1] or result["status"] != "completed":
        raise ValueError(f"Final result and trace disagree for {config['run_id']}")
    for row in dense:
        if row["forward_views"] != row["backward_views"]:
            raise ValueError("Forward and adjoint view counts differ")
        if any("seconds" in key for key in row):
            raise ValueError("Timing fields found in a dense trace")
    mismatches = [
        dict(iteration=row["iteration"], field=field,
             original=row[field], dense=dense[row["iteration"]][field])
        for row in old for field in SCIENTIFIC_FIELDS
        if row[field] != dense[row["iteration"]][field]
    ]
    audit = dict(run_id=config["run_id"], logged_iterations=len(dense),
                 original_probes=len(old), compared_fields=list(SCIENTIFIC_FIELDS),
                 exact_original_probe_match=not mismatches, mismatches=mismatches)
    write_json(folder / "validation.json", audit)
    if mismatches:
        raise ValueError(f"Scientific results differ from the original run: {config['run_id']}")
    return audit


def main():
    with (ROOT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute()


def execute():
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("The frozen physical CPU assignment is unavailable")
    original_manifest = read(ORIGINAL / "manifest.json")
    jobs = []
    for path in sorted((ORIGINAL / "runs").glob("evaluation_*/metadata.json")):
        metadata = read(path)
        config = dict(metadata["config"])
        config["probe_every"] = 1
        config.pop("time_budget_seconds", None)
        old_trace = read(path.with_name("trace.json"))
        diagnostic_views = (2 * config["n_train"] + 360 if config["method"] == "sframes"
                            else config["n_train"] + 360)
        weight = old_trace[-1]["iteration"] * diagnostic_views + 2 * config["view_budget"]
        jobs.append(dict(config=config, estimated_view_work=weight))
    if len(jobs) != 132:
        raise ValueError(f"Expected 132 frozen evaluation runs, found {len(jobs)}")
    loads = {cpu: 0 for cpu in CPUS}
    assignments = []
    for job in sorted(jobs, key=lambda row: (-row["estimated_view_work"], row["config"]["run_id"])):
        cpu = min(loads, key=loads.get)
        loads[cpu] += job["estimated_view_work"]
        assignments.append(dict(job, cpu=cpu))
    design = dict(original_manifest["design"])
    for key in ("evaluation_time_budget_seconds", "timing"):
        design.pop(key, None)
    design["diagnostic_cadence"] = "iteration_zero_and_every_completed_iteration"
    manifest = dict(
        original_manifest_sha256=sha256(ORIGINAL / "manifest.json"),
        original_selection_sha256=sha256(ORIGINAL / "selection.json"),
        original_sources=original_manifest["sources"],
        sources={name: sha256(ROOT / name) for name in SOURCES},
        data=original_manifest["data"], data_sidecars=original_manifest["data_sidecars"],
        host=platform.node(), python=sys.executable, cpus=list(CPUS), blas_threads=1,
        design=design, diagnostics="Full original metrics at iteration zero and every update.",
        timing_measurements=False,
    )
    frozen(ROOT / "manifest.json", manifest)
    frozen(ROOT / "jobs.json", assignments)
    state = dict(status="running", phase="evaluation", pid=os.getpid(), started_utc=now(),
                 expected_runs=len(jobs), runs={})
    state["runs"] = {row["config"]["run_id"]: dict(status="queued", cpu=row["cpu"])
                     for row in assignments}
    write_json(ROOT / "progress.json", state)
    try:
        ctx = mp.get_context("spawn")
        with ExitStack() as stack:
            pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx))
                     for cpu in CPUS}
            pending = {pools[row["cpu"]].submit(run, row["config"], row["cpu"]): row
                       for row in assignments}
            for future in as_completed(pending):
                job = pending[future]
                name = job["config"]["run_id"]
                try:
                    final = future.result()
                    audit = validate(job["config"])
                    item = dict(status="completed", cpu=job["cpu"], iteration=final["iteration"],
                                forward_views=final["forward_views"],
                                loss=final["feasible_loss"],
                                logged_iterations=audit["logged_iterations"],
                                exact_original_probe_match=True)
                except Exception as exc:
                    item = dict(status="failed", cpu=job["cpu"], error=repr(exc))
                state["runs"][name] = item
                write_json(ROOT / "progress.json", state)
                print(now(), name, item, flush=True)
        failed = [name for name, item in state["runs"].items() if item["status"] != "completed"]
        state.update(status="completed" if not failed else "completed_with_failures",
                     phase="validated", failed_runs=failed)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = now()
        write_json(ROOT / "progress.json", state)
    if state["failed_runs"]:
        raise RuntimeError(f"Dense reruns failed: {state['failed_runs']}")


if __name__ == "__main__":
    main()
