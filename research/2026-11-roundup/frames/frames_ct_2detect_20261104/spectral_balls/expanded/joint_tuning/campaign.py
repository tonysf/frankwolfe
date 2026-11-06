"""Joint development search for the SFRAMES smoothing and step prefactors."""

from run import ROOT, run, sha256, write_json

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from itertools import product
import json
import math
import multiprocessing as mp
import os
import platform
from statistics import median
import sys


ORIGINAL = ROOT.parent
CPUS = (0, 2, 4, 6)
VARIANTS = ("sframes", "sframes_full")
DEV_SLICES = ("slice04300", "slice03150")
TRAINING_VIEWS = (360, 1080)
DEV_BUDGET = 8640
INITIAL_BETA = (0.25, 1.0, 4.0)
INITIAL_GAMMA = (0.05, 0.10, 0.16, 0.25, 0.50, 0.80)
BETA_LIMITS = (0.00390625, 256.0)
GAMMA_LIMITS = (0.0125, 1.0)
SOURCES = ("campaign.py", "run.py", "methods.py", "ct_problem.py",
           "spectral_reference_core.py", "spectral_reference_baseline.py")


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen configuration changed at {path}")
    write_json(path, value)


def number(value):
    return f"{value:.12g}".replace(".", "p")


def initial_grids():
    return {variant: dict(beta_scales=list(INITIAL_BETA), gamma0=list(INITIAL_GAMMA))
            for variant in VARIANTS}


def job(phase, slice_id, n_train, variant, beta_scale, gamma0, seed):
    if phase not in ("development", "evaluation") or variant not in VARIANTS:
        raise ValueError("Joint-search jobs require a supported phase and SFRAMES variant")
    development = phase == "development"
    trial = f"b{number(beta_scale)}_g{number(gamma0)}"
    return dict(
        run_id=f"{phase}_{slice_id}_{n_train}_{variant}_seed{seed}_{trial}",
        phase=phase, slice_id=slice_id, n_train=n_train, method="sframes", variant=variant,
        trial=trial, seed=seed, calibration_seed=110 if development else 201,
        grid=512, rank=64, batch=n_train if variant == "sframes_full" else 64,
        view_budget=DEV_BUDGET if development else 17280,
        beta_scale=float(beta_scale), rate=float(gamma0),
        record_every_iteration=not development,
    )


def development_jobs(grids):
    return [job("development", sid, n, variant, beta, gamma, 110)
            for variant in VARIANTS
            for beta, gamma in product(grids[variant]["beta_scales"], grids[variant]["gamma0"])
            for sid, n in product(DEV_SLICES, TRAINING_VIEWS)]


def validate(config):
    folder = ROOT / "runs" / config["run_id"]
    result = read(folder / "result.json")
    trace = read(folder / "trace.json")
    if result["status"] != "completed" or result["config"] != config or not trace:
        raise ValueError(f"Completed result does not match its configuration: {folder}")
    if trace[0]["iteration"] != 0 or result["final"] != trace[-1]:
        raise ValueError(f"Initial/final observations disagree with the result: {folder}")
    if result["iteration"] != trace[-1]["iteration"]:
        raise ValueError(f"Final iteration differs from the trace: {folder}")
    if trace[-1]["forward_views"] < config["view_budget"]:
        raise ValueError(f"Run stopped below its optimization-work budget: {folder}")
    if config["record_every_iteration"]:
        if [row["iteration"] for row in trace] != list(range(result["iteration"] + 1)):
            raise ValueError(f"Per-iteration confirmation has missing observations: {folder}")
    for row in trace:
        if row["forward_views"] != row["backward_views"]:
            raise ValueError("Forward and adjoint work differ")
        if not all(math.isfinite(row[field]) for field in ("raw_loss", "feasible_loss", "d_distance")):
            raise ValueError("Nonfinite development diagnostic")
        if any("seconds" in field for field in row):
            raise ValueError("Unexpected timing measurement")
    return result, trace


def execute(jobs, state):
    assignments = [dict(config=config, cpu=CPUS[i % len(CPUS)])
                   for i, config in enumerate(jobs)]
    write_json(ROOT / "jobs.json", assignments)
    state.update(status="running", phase="development", expected_runs=len(jobs))
    for row in assignments:
        state["runs"].setdefault(row["config"]["run_id"], dict(status="queued", cpu=row["cpu"]))
    write_json(ROOT / "progress.json", state)
    ctx = mp.get_context("spawn")
    with ExitStack() as stack:
        pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx))
                 for cpu in CPUS}
        pending = {pools[row["cpu"]].submit(run, row["config"], row["cpu"]): row
                   for row in assignments}
        for future in as_completed(pending):
            row = pending[future]
            name = row["config"]["run_id"]
            try:
                final = future.result()
                validate(row["config"])
                item = dict(status="completed", cpu=row["cpu"], iteration=final["iteration"],
                            forward_views=final["forward_views"], loss=final["feasible_loss"])
            except Exception as exc:
                item = dict(status="failed", cpu=row["cpu"], error=repr(exc))
            state["runs"][name] = item
            write_json(ROOT / "progress.json", state)
            print(now(), name, item, flush=True)
    failed = [config["run_id"] for config in jobs
              if state["runs"][config["run_id"]]["status"] != "completed"]
    if failed:
        raise RuntimeError(f"Development cases failed: {failed}")


def score(grids):
    scores, selected = {}, {}
    for variant in VARIANTS:
        trials = []
        for beta, gamma in product(grids[variant]["beta_scales"], grids[variant]["gamma0"]):
            cases = []
            for sid, n in product(DEV_SLICES, TRAINING_VIEWS):
                config = job("development", sid, n, variant, beta, gamma, 110)
                _, trace = validate(config)
                final = next(row for row in reversed(trace)
                             if max(row["forward_views"], row["backward_views"]) <= DEV_BUDGET)
                cases.append(dict(run_id=config["run_id"], slice_id=sid, n_train=n,
                                  forward_views=final["forward_views"],
                                  initial_loss=trace[0]["feasible_loss"],
                                  repaired_loss=final["feasible_loss"],
                                  relative_loss=final["feasible_loss"] / trace[0]["feasible_loss"]))
            trials.append(dict(beta_scale=beta, gamma0=gamma, cases=cases,
                               median_relative_loss=median(row["relative_loss"] for row in cases)))
        scores[variant] = trials
        best = min(trials, key=lambda row: (row["median_relative_loss"], row["beta_scale"], row["gamma0"]))
        selected[variant] = {key: best[key] for key in ("beta_scale", "gamma0", "median_relative_loss")}
    return dict(
        status="development_round_complete", selected=selected, scores=scores, grids=deepcopy(grids),
        development_view_budget=DEV_BUDGET,
        rule="Smallest median repaired training loss relative to initial loss across the two "
             "development slices and both training-view counts, using the last within-budget "
             "observation. Ties favor the smaller beta multiplier, then smaller gamma0.",
    )


def refine(grids, selected):
    updated = deepcopy(grids)
    decisions, boundaries = {}, {}
    for variant in VARIANTS:
        decisions[variant], boundaries[variant] = [], {}
        for axis, parameter, limits, factor in (
            ("beta_scales", "beta_scale", BETA_LIMITS, 4.0),
            ("gamma0", "gamma0", GAMMA_LIMITS, 2.0),
        ):
            values = grids[variant][axis]
            chosen = selected[variant][parameter]
            side = "lower" if chosen == values[0] else "upper" if chosen == values[-1] else "interior"
            addition = None
            if side == "lower" and chosen > limits[0]:
                addition = max(limits[0], chosen / factor)
            if side == "upper" and chosen < limits[1]:
                addition = min(limits[1], chosen * factor)
            boundaries[variant][parameter] = dict(
                side=side, at_limit=side != "interior" and addition is None,
                searched_limits=list(limits),
            )
            if addition is not None:
                updated[variant][axis] = sorted(set(values + [addition]))
                decisions[variant].append(dict(parameter=parameter, side=side, added=addition))
    return updated, decisions, boundaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-only", action="store_true",
                        help="Development is the default and only campaign phase")
    parser.add_argument("--plan", action="store_true", help="Print initial grid counts without running")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(dict(grids=initial_grids(), candidates_per_variant=18,
                              initial_physical_runs=len(development_jobs(initial_grids()))), indent=2))
        return
    with (ROOT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign()


def campaign():
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("The frozen physical CPU assignment is unavailable")
    original = read(ORIGINAL / "manifest.json")
    frozen(ROOT / "baseline_selection.json", read(ORIGINAL / "selection.json"))
    design = deepcopy(original["design"])
    for key in ("evaluation_time_budget_seconds", "timing", "development_trials"):
        design.pop(key, None)
    design.update(
        variants=list(original["design"]["development_trials"]),
        baseline_development_trials={key: value for key, value in
                                     original["design"]["development_trials"].items()
                                     if key not in VARIANTS},
        development_variants=list(VARIANTS), initial_joint_grids=initial_grids(),
        beta_scale_definition="beta0=beta_scale/max(L_U,L_V)",
        refinement=dict(beta_scale_factor=4, beta_scale_limits=list(BETA_LIMITS),
                        gamma0_factor=2, gamma0_limits=list(GAMMA_LIMITS),
                        criterion="Expand a selected grid edge to the stated limit, crossing "
                                  "every value of the other parameter to retain a full rectangle."),
        diagnostics="Development initial/final endpoints, selected evaluation every completed iteration.",
    )
    manifest = dict(
        original_manifest_sha256=sha256(ORIGINAL / "manifest.json"),
        original_selection_sha256=sha256(ORIGINAL / "selection.json"),
        baseline_selection_sha256=sha256(ROOT / "baseline_selection.json"),
        original_sources=original["sources"],
        sources={name: sha256(ROOT / name) for name in SOURCES},
        data=original["data"], data_sidecars=original["data_sidecars"],
        host=platform.node(), python=sys.executable, cpus=list(CPUS), blas_threads=1,
        timing_measurements=False, design=design,
    )
    frozen(ROOT / "manifest.json", manifest)
    search_path = ROOT / "search_state.json"
    search = read(search_path) if search_path.exists() else dict(round=0, grids=initial_grids(), history=[])
    state = read(ROOT / "progress.json") if (ROOT / "progress.json").exists() else dict(runs={})
    state.update(status="running", pid=os.getpid(), started_utc=now())
    try:
        while True:
            write_json(search_path, search)
            state["round"] = search["round"]
            execute(development_jobs(search["grids"]), state)
            selection = score(search["grids"])
            updated, decisions, boundaries = refine(search["grids"], selection["selected"])
            selection["boundaries"] = boundaries
            selection["round"] = search["round"]
            expanded = any(decisions.values())
            selection["status"] = "expanding_grid" if expanded else "selected"
            write_json(ROOT / "selection.json", selection)
            record = dict(round=search["round"], grids=deepcopy(search["grids"]),
                          selected=selection["selected"], expansions=decisions, boundaries=boundaries)
            search["history"].append(record)
            if not expanded:
                search["status"] = "selected"
                write_json(search_path, search)
                state.update(status="completed", phase="development_selected",
                             selected=selection["selected"], boundaries=boundaries)
                break
            search.update(round=search["round"] + 1, grids=updated)
            write_json(search_path, search)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = now()
        write_json(ROOT / "progress.json", state)


if __name__ == "__main__":
    main()
