"""Confirm joint-selected SFRAMES settings with dense native trajectories."""

from campaign import CPUS, SOURCES, job, now, read, frozen
from run import ROOT, method_config, run, sha256, write_json

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time


ORIGINAL = ROOT.parent / "dense_trajectories"
PAIR_FIELDS = ("data_sha256", "calibration_sha256", "caps", "curvature",
               "train_view_ids", "validation_view_ids", "initialization", "constraint_geometry")
NUMERICAL = ("ct_problem.py", "methods.py", "spectral_reference_core.py", "spectral_reference_baseline.py")


def validation(item):
    folder = ROOT / "runs" / item["run_id"]
    result = read(folder / "result.json")
    metadata = read(folder / "metadata.json")
    trace = read(folder / "trace.json")
    original = read(Path(item["original_directory"]) / "result.json")
    if metadata["config"] != result["config"] or metadata["source_sha256"] != result["source_sha256"]:
        raise ValueError(f"Result and metadata disagree for {folder}")
    if result["status"] != "completed" or result["final"] != trace[-1]:
        raise ValueError(f"Result and trace disagree for {folder}")
    if [row["iteration"] for row in trace] != list(range(result["iteration"] + 1)):
        raise ValueError(f"Missing per-update observations in {folder}")
    if trace[-1]["forward_views"] < item["expected_config"]["view_budget"]:
        raise ValueError(f"Confirmation stopped below budget in {folder}")
    for field in PAIR_FIELDS:
        if metadata[field] != original[field]:
            raise ValueError(f"Original problem changed on {field} in {folder}")
    for name in NUMERICAL:
        if metadata["source_sha256"][name] != original["source_sha256"][name]:
            raise ValueError(f"Original numerical code changed on {name}")
    expected = method_config(item["expected_config"], metadata["curvature"])
    if metadata["optimizer"] != expected or result["optimizer"] != expected:
        raise ValueError(f"Actual optimizer differs from selected settings in {folder}")
    first = read(Path(item["original_directory"]) / "trace.json")[0]
    for field in ("raw_loss", "feasible_loss", "validation_loss", "d_distance", "reference_relative_error"):
        if first[field] != trace[0][field]:
            raise ValueError(f"Initialization changed in {folder}")
    for row in trace:
        if row["forward_views"] != row["backward_views"] or any("seconds" in key for key in row):
            raise ValueError(f"Invalid work counters in {folder}")
    return dict(observations=len(trace), iteration=result["iteration"],
                forward_views=trace[-1]["forward_views"],
                feasible_loss=trace[-1]["feasible_loss"])


def plan(selection):
    items = []
    for metadata_path in sorted((ORIGINAL / "runs").glob("evaluation_*/metadata.json")):
        metadata = read(metadata_path)
        original_config = metadata["config"]
        variant = original_config["variant"]
        if variant in ("sframes", "sframes_full"):
            chosen = selection["selected"][variant]
            config = job("evaluation", original_config["slice_id"], original_config["n_train"],
                         variant, chosen["beta_scale"], chosen["gamma0"], original_config["seed"])
            reused = method_config(config, metadata["curvature"]) == metadata["optimizer"]
        else:
            config = dict(original_config)
            reused = True
        actual_id = original_config["run_id"] if reused else config["run_id"]
        if reused:
            link = ROOT / "runs" / actual_id
            target = metadata_path.parent.resolve()
            if link.exists() or link.is_symlink():
                if not link.is_symlink() or link.resolve() != target:
                    raise ValueError(f"Reused run path already has different contents: {link}")
            else:
                link.symlink_to(target, target_is_directory=True)
        final = read(metadata_path.with_name("result.json"))["final"]
        diagnostic_views = 2 * config["n_train"] + 360
        weight = final["iteration"] * diagnostic_views + 2 * config["view_budget"]
        items.append(dict(run_id=actual_id, expected_config=config, reused=reused,
                          original_directory=str(metadata_path.parent.resolve()),
                          estimated_view_work=weight))
    if len(items) != 132 or len({item["run_id"] for item in items}) != 132:
        raise ValueError("Confirmation plan does not contain the frozen 132-arm cohort")
    loads = {cpu: 0 for cpu in CPUS}
    for item in sorted((item for item in items if not item["reused"]),
                       key=lambda row: (-row["estimated_view_work"], row["run_id"])):
        cpu = min(loads, key=loads.get)
        item["cpu"] = cpu
        loads[cpu] += item["estimated_view_work"]
    return items


def execute():
    selection = read(ROOT / "selection.json")
    if selection["status"] != "selected":
        raise ValueError("Development search has not frozen its choices")
    old_state = read(ROOT / "progress.json")
    development_progress = ROOT / "development_progress.json"
    if not development_progress.exists():
        if old_state["status"] != "completed" or old_state["phase"] != "development_selected":
            raise ValueError("Development search has not completed")
        write_json(development_progress, old_state)
    (ROOT / "runs").mkdir(exist_ok=True)
    items = plan(selection)
    frozen(ROOT / "evaluation_plan.json", dict(
        selected_sha256=sha256(ROOT / "selection.json"),
        original_dense_manifest_sha256=sha256(ORIGINAL / "manifest.json"),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        reused_runs=sum(item["reused"] for item in items),
        new_runs=sum(not item["reused"] for item in items), runs=items,
    ))
    state = dict(status="running", phase="evaluation", started_utc=now(),
                 selected=selection["selected"], expected_runs=132, runs={})
    for item in items:
        if item["reused"]:
            details = validation(item)
            state["runs"][item["run_id"]] = dict(status="completed", reused=True, **details)
        else:
            state["runs"][item["run_id"]] = dict(status="queued", reused=False, cpu=item["cpu"])
    write_json(ROOT / "progress.json", state)
    try:
        pending_items = [item for item in items if not item["reused"]]
        if pending_items:
            ctx = mp.get_context("spawn")
            with ExitStack() as stack:
                pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx))
                         for cpu in CPUS}
                pending = {pools[item["cpu"]].submit(run, item["expected_config"], item["cpu"]): item
                           for item in pending_items}
                for future in as_completed(pending):
                    item = pending[future]
                    try:
                        future.result()
                        details = validation(item)
                        row = dict(status="completed", reused=False, cpu=item["cpu"], **details)
                    except Exception as exc:
                        row = dict(status="failed", reused=False, cpu=item["cpu"], error=repr(exc))
                    state["runs"][item["run_id"]] = row
                    write_json(ROOT / "progress.json", state)
                    print(now(), item["run_id"], row, flush=True)
        failed = [key for key, row in state["runs"].items() if row["status"] != "completed"]
        state.update(status="completed" if not failed else "completed_with_failures",
                     phase="evaluation_validated", failed_runs=failed)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = now()
        write_json(ROOT / "progress.json", state)
    if state["failed_runs"]:
        raise RuntimeError(f"Confirmation cases failed: {state['failed_runs']}")
    subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--no-plots"], check=True)
    subprocess.run([sys.executable, str(ROOT / "plot_selection.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "report_update.py")], check=True)
    write_json(ROOT / "completion.json", dict(
        completed_utc=now(), selected=selection["selected"], reused_runs=sum(item["reused"] for item in items),
        new_runs=sum(not item["reused"] for item in items),
        observations=sum(row["observations"] for row in state["runs"].values()),
        paper_pdf=str(ROOT.parent / "reports" / "paper_section" / "sframes_ct_section.pdf"),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    with (ROOT / "confirmation.lock").open("a") as confirmation_lock:
        fcntl.flock(confirmation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = read(ROOT / "progress.json")
            if (ROOT / "development_progress.json").exists() or state["status"] != "running":
                break
            if not args.wait:
                raise RuntimeError("Development search is still running")
            time.sleep(30)
        if state["status"] == "failed":
            raise RuntimeError(f"Development or confirmation failed: {state.get('error')}")
        with (ROOT / "campaign.lock").open("a") as campaign_lock:
            fcntl.flock(campaign_lock, fcntl.LOCK_EX)
            execute()


if __name__ == "__main__":
    main()
