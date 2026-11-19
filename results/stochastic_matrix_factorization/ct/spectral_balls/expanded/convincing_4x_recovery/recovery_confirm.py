"""Resume the frozen 308-arm evaluation with explicit historical-source provenance."""

from run import ROOT, NUMERICAL_SOURCES, method_config, read_json as read, run, sha256, write_json

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

import numpy as np
from spectral_reference_baseline import ProjectionFailure


OLD = ROOT.parent / "convincing_4x"
CPUS = tuple(int(value) for value in os.environ.get("FRAMES_CAMPAIGN_CPUS", "0,2,4,6").split(","))
if not CPUS or len(set(CPUS)) != len(CPUS) or not set(CPUS).issubset({0, 2, 4, 6}):
    raise ValueError("FRAMES_CAMPAIGN_CPUS must name distinct supported physical CPUs")


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen recovery input changed: {path}")
    write_json(path, value)


def source_hashes(folder):
    return {name: sha256(folder / name) for name in
            ("metadata.json", "trace.json", "checkpoint.pkl", "result.json") if (folder / name).exists()}


def validate_native(folder, config, *, complete):
    metadata = read(folder / "metadata.json")
    trace = read(folder / "trace.json")
    if metadata["config"] != config or not trace:
        raise ValueError(f"Native arm differs from its frozen configuration: {folder}")
    if metadata["optimizer"] != method_config(config, metadata["curvature"]):
        raise ValueError(f"Actual optimizer differs from the selected configuration: {folder}")
    if [row["iteration"] for row in trace] != list(range(trace[-1]["iteration"] + 1)):
        raise ValueError(f"Native history has missing iterations: {folder}")
    for row in trace:
        if row["forward_views"] != row["backward_views"] or any("seconds" in name for name in row):
            raise ValueError(f"Invalid optimization-work accounting: {folder}")
        if not all(np.isfinite(row[name]) for name in ("raw_loss", "feasible_loss", "d_distance")):
            raise ValueError(f"Nonfinite saved scientific diagnostic: {folder}")
    if complete:
        result = read(folder / "result.json")
        if result["status"] != "completed" or result["final"] != trace[-1] or result["iteration"] != trace[-1]["iteration"]:
            raise ValueError(f"Completed result and trace disagree: {folder}")
        if any(result[name] != value for name, value in metadata.items()):
            raise ValueError(f"Completed result and metadata disagree: {folder}")
        if trace[-1]["forward_views"] < config["view_budget"]:
            raise ValueError(f"Completed evaluation stopped below budget: {folder}")
    for field in ("continuation", "imported_checkpoint"):
        audit = metadata.get(field)
        if audit is None:
            continue
        source = Path(audit["source_directory"])
        if sha256(source / "trace.json") != audit["source_trace_sha256"]:
            raise ValueError(f"Immutable imported prefix changed: {source}")
        prefix = read(source / "trace.json")
        if trace[:len(prefix)] != prefix:
            raise ValueError(f"The recovered native history changes an imported observation: {folder}")
    return metadata, trace


def prepare_plan():
    old_plan = read(OLD / "evaluation_plan.json")
    if old_plan["expected_runs"] != 308 or len(old_plan["runs"]) != 308:
        raise ValueError("The original frozen evaluation cohort is not the expected 308 arms")
    configurations, origins = [], {}
    for original in old_plan["runs"]:
        config = dict(original)
        folder = OLD / "runs" / config["run_id"]
        destination = ROOT / "runs" / config["run_id"]
        result = read(folder / "result.json") if (folder / "result.json").exists() else None
        if result is not None and result.get("status") == "completed":
            validate_native(folder, original, complete=True)
            if destination.exists() or destination.is_symlink():
                if not destination.is_symlink() or destination.resolve() != folder.resolve():
                    raise ValueError(f"Completed historical arm has a different recovery destination: {destination}")
            else:
                destination.symlink_to(folder.resolve(), target_is_directory=True)
            origin = dict(kind="reuse_completed", source_directory=str(folder.resolve()),
                          source_file_sha256=source_hashes(folder))
        elif (folder / "checkpoint.pkl").exists():
            validate_native(folder, original, complete=False)
            checkpoint = pickle.loads((folder / "checkpoint.pkl").read_bytes())
            if json.loads(json.dumps(checkpoint["metadata"])) != read(folder / "metadata.json"):
                raise ValueError(f"Historical checkpoint metadata changed: {folder}")
            if checkpoint["trace"] != read(folder / "trace.json"):
                raise ValueError(f"Historical checkpoint and trace disagree: {folder}")
            config["checkpoint_import_directory"] = str(folder.resolve())
            origin = dict(kind="import_checkpoint", source_directory=str(folder.resolve()),
                          source_file_sha256=source_hashes(folder),
                          source_final_iteration=checkpoint["solver"]["k"],
                          source_forward_views=checkpoint["counters"]["forward_views"])
        elif config.get("continuation_directory"):
            source = Path(config["continuation_directory"])
            origin = dict(kind="continue_original_prefix", source_directory=str(source.resolve()),
                          source_file_sha256=source_hashes(source))
        else:
            origin = dict(kind="fresh", source_directory=None)
            if folder.exists():
                origin["unstarted_original_directory"] = str(folder.resolve())
                origin["unstarted_original_file_sha256"] = source_hashes(folder)
        configurations.append(config)
        origins[config["run_id"]] = origin
    if len({config["run_id"] for config in configurations}) != 308:
        raise ValueError("The recovery plan has duplicate or missing arms")
    plan = dict(version=1, original_campaign_root=str(OLD),
                original_evaluation_plan_sha256=sha256(OLD / "evaluation_plan.json"),
                selected_sha256=sha256(ROOT / "selection.json"), cohort_sha256=sha256(ROOT / "cohort.json"),
                compatibility_manifest_sha256=sha256(ROOT / "compatibility_manifest.json"),
                source_sha256=read(ROOT / "compatibility_manifest.json")["current_source_sha256"],
                orchestrator_sha256=sha256(Path(__file__)), expected_runs=308,
                runs=configurations, origins=origins)
    frozen(ROOT / "evaluation_plan.json", plan)
    return plan


def prepare_manifest(plan):
    manifest = deepcopy(read(OLD / "manifest.json"))
    manifest["historical_original_manifest_sha256"] = manifest.get("original_manifest_sha256")
    manifest.update(
        original_campaign_root=str(OLD), original_manifest_sha256=sha256(OLD / "manifest.json"),
        original_sources=read(OLD / "manifest.json")["sources"],
        sources=read(ROOT / "compatibility_manifest.json")["current_source_sha256"],
        compatibility_manifest_sha256=sha256(ROOT / "compatibility_manifest.json"),
        original_evaluation_plan_sha256=sha256(OLD / "evaluation_plan.json"),
        evaluation_plan_sha256=sha256(ROOT / "evaluation_plan.json"),
        historical_development_selection_sha256=sha256(OLD / "selection.json"),
        recovery_origin_counts={kind: sum(row["kind"] == kind for row in plan["origins"].values())
                                for kind in ("reuse_completed", "import_checkpoint", "continue_original_prefix", "fresh")},
    )
    frozen(ROOT / "manifest.json", manifest)
    return manifest


def isolated_numerical_failure(exc):
    return isinstance(exc, (FloatingPointError, np.linalg.LinAlgError, ProjectionFailure))


def execute(plan):
    from audit_provenance import audit_evaluation, validate_evaluation_metadata
    path = ROOT / "evaluation_progress.json"
    state = read(path) if path.exists() else dict(runs={})
    state.update(status="running", phase="recovery_evaluation", started_utc=now(), pid=os.getpid(),
                 expected_runs=308, original_campaign_root=str(OLD))
    write_json(ROOT / "execution_abort.json", dict(active=False, phase="recovery_evaluation"))
    loads, jobs = {cpu: 0 for cpu in CPUS}, []
    for config in plan["runs"]:
        name, folder = config["run_id"], ROOT / "runs" / config["run_id"]
        if (folder / "result.json").exists():
            metadata, trace = validate_native(folder, config, complete=True)
            validate_evaluation_metadata(ROOT, metadata, expected_config=config)
            state["runs"][name] = dict(status="completed", origin=plan["origins"][name]["kind"],
                                       observations=len(trace), iteration=trace[-1]["iteration"],
                                       forward_views=trace[-1]["forward_views"], feasible_loss=trace[-1]["feasible_loss"])
            continue
        cpu = min(loads, key=loads.get)
        if (folder / "checkpoint.pkl").exists():
            previous_cpu = read(folder / "metadata.json")["cpu"]
            if previous_cpu not in CPUS:
                raise ValueError("A partial recovery checkpoint requires its original frozen CPU assignment")
            cpu = previous_cpu
        steps = config["view_budget"] / (config["batch"] if config["method"] == "sframes" else 256)
        loads[cpu] += 2 * config["view_budget"] + steps * (2 * config["n_train"] + 360)
        jobs.append((config, cpu))
        state["runs"][name] = dict(status="queued", cpu=cpu, origin=plan["origins"][name]["kind"])
    write_json(path, state)
    shared_error = None
    try:
        ctx = mp.get_context("spawn")
        with ExitStack() as stack:
            pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx)) for cpu in CPUS}
            pending = {pools[cpu].submit(run, config, cpu): (config, cpu) for config, cpu in jobs}
            for future in as_completed(pending):
                config, cpu = pending[future]
                name, folder = config["run_id"], ROOT / "runs" / config["run_id"]
                try:
                    future.result()
                    metadata, trace = validate_native(folder, config, complete=True)
                    validate_evaluation_metadata(ROOT, metadata, expected_config=config)
                    item = dict(status="completed", cpu=cpu, origin=plan["origins"][name]["kind"],
                                observations=len(trace), iteration=trace[-1]["iteration"],
                                forward_views=trace[-1]["forward_views"], feasible_loss=trace[-1]["feasible_loss"])
                except Exception as exc:
                    isolated = isolated_numerical_failure(exc)
                    item = dict(status="failed", cpu=cpu, origin=plan["origins"][name]["kind"],
                                error=repr(exc), error_type=type(exc).__name__, traceback=traceback.format_exc(),
                                isolated_numerical_failure=isolated, checkpoint_retained=(folder / "checkpoint.pkl").exists())
                    if not isolated:
                        shared_error = f"Shared input, source or infrastructure failure in {name}: {exc!r}"
                state["runs"][name] = item
                write_json(path, state)
                print(now(), name, item, flush=True)
                if shared_error:
                    write_json(ROOT / "execution_abort.json", dict(active=True, phase="recovery_evaluation", error=shared_error))
                    for outstanding in pending:
                        outstanding.cancel()
                    raise RuntimeError(shared_error)
        failed = [name for name, item in state["runs"].items() if item["status"] != "completed"]
        state.update(status="completed" if not failed else "completed_with_failures",
                     phase="evaluation_validated" if not failed else "evaluation_incomplete", failed_runs=failed)
        if failed:
            raise RuntimeError(f"Independent evaluation failures remain; publication withheld: {failed}")
        audit = audit_evaluation(ROOT, require_complete=True)
        return audit
    except BaseException as exc:
        if state.get("status") == "running":
            state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = now()
        write_json(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Freeze the recovery plan and provenance without optimizer steps")
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()
    with (ROOT / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from audit_provenance import freeze_recovery_anchor
        plan = prepare_plan()
        prepare_manifest(plan)
        freeze_recovery_anchor(ROOT)
        if args.prepare_only:
            print(json.dumps(dict(expected_runs=308, origins=read(ROOT / "manifest.json")["recovery_origin_counts"]), indent=2))
            return
        if not (ROOT / "execution_cpus.json").exists():
            raise ValueError("The recovery CPU governor must be explicitly configured before production")
        audit = execute(plan)
        progress = read(ROOT / "evaluation_progress.json")
        observations = sum(row["observations"] for row in progress["runs"].values())
        write_json(ROOT / "evaluation_completion.json", dict(completed_utc=now(), evaluation_runs=308,
                   observations=observations, audit=audit))
        if not args.no_postprocess:
            subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--no-plots"], check=True)
            subprocess.run([sys.executable, str(ROOT / "report_update.py")], check=True)
            write_json(ROOT / "completion.json", dict(completed_utc=now(), evaluation_runs=308, observations=observations,
                       paper_pdf=str(ROOT.parent / "reports" / "paper_section" / "sframes_ct_section.pdf")))


if __name__ == "__main__":
    main()
