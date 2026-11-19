"""Retune the three baselines on complete Cartesian development grids."""

from run import ROOT, run, sha256, write_json, method_config

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
from pathlib import Path
import platform
from statistics import median
import sys


PREVIOUS = ROOT.parent / "joint_tuning"
CPUS = tuple(int(value) for value in os.environ.get("FRAMES_CAMPAIGN_CPUS", "0,2,4,6").split(","))
if not CPUS or len(set(CPUS)) != len(CPUS) or not set(CPUS).issubset({0, 2, 4, 6}):
    raise ValueError("FRAMES_CAMPAIGN_CPUS must list distinct supported physical CPUs")
VARIANTS = ("sframes", "sframes_full", "spring_sarah", "prox_sarah", "palm")
BASELINES = ("spring_sarah", "prox_sarah", "palm")
DEV_SLICES = ("slice04300", "slice03150")
DEV_SEEDS = (110, 111, 112)
TRAINING_VIEWS = (360, 1080)
DEV_BUDGET = 34560
EVAL_BUDGET = 69120
RATE_LIMITS = (0.05, 80.0)
SECONDARY = {"spring_sarah": ("refresh_period", (1, 128)),
             "prox_sarah": ("inner_steps", (3, 256))}
SOURCES = ("campaign.py", "run.py", "methods.py", "ct_problem.py",
           "spectral_reference_core.py", "spectral_reference_baseline.py")


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen configuration changed at {path}")
    write_json(path, value)


def number(value):
    return f"{value:.12g}".replace(".", "p")


def initial_grids():
    return {
        "spring_sarah": dict(rate=[0.7, 1.1, 1.6, 2.3, 3.4, 5.0], refresh_period=[8, 16, 32]),
        "prox_sarah": dict(rate=[0.9, 1.5, 2.3, 3.4, 5.0], inner_steps=[6, 12, 24],
                           snapshot_policy=["minibatch", "full"]),
        "palm": dict(rate=[0.6, 1.1, 1.6, 2.3, 3.4, 5.0]),
    }


def trials(variant, grid):
    names = list(grid)
    return [dict(zip(names, values)) for values in product(*(grid[name] for name in names))]


def job(phase, slice_id, n_train, variant, parameters, seed):
    development = phase == "development"
    if phase not in ("development", "evaluation") or variant not in VARIANTS:
        raise ValueError("Unsupported experiment phase or method")
    params = dict(parameters)
    if variant in ("sframes", "sframes_full"):
        beta, gamma = params["beta_scale"], params["gamma0"]
        label = f"b{number(beta)}_g{number(gamma)}"
        params = dict(beta_scale=float(beta), rate=float(gamma))
    else:
        label = f"a{number(params['rate'])}"
        if variant == "spring_sarah":
            label += f"_q{params['refresh_period']}"
        if variant == "prox_sarah":
            label += f"_m{params['inner_steps']}_s{params.get('snapshot_policy', 'minibatch')}"
    return dict(run_id=f"{phase}_{slice_id}_{n_train}_{variant}_seed{seed}_{label}",
                phase=phase, slice_id=slice_id, n_train=n_train,
                method="sframes" if variant in ("sframes", "sframes_full") else variant,
                variant=variant, trial=label, seed=seed, calibration_seed=110 if development else 201,
                grid=512, rank=64, batch=n_train if variant == "sframes_full" else 64,
                view_budget=DEV_BUDGET if development else EVAL_BUDGET,
                record_every_iteration=not development, **params)


def development_jobs(grids):
    return [job("development", sid, n, variant, candidate, seed)
            for variant in BASELINES for candidate in trials(variant, grids[variant])
            for sid, n in product(DEV_SLICES, TRAINING_VIEWS)
            for seed in ((DEV_SEEDS[0],) if variant == "palm" else DEV_SEEDS)]


def validate(config, *, allow_candidate_failure=False):
    folder = ROOT / "runs" / config["run_id"]
    result, metadata, trace = (read(folder / name) for name in
                               ("result.json", "metadata.json", "trace.json"))
    if result["config"] != config or metadata["config"] != config or not trace:
        raise ValueError(f"Result does not match its frozen configuration: {folder}")
    if any(result[field] != value for field, value in metadata.items()):
        raise ValueError(f"Result and metadata disagree: {folder}")
    if result["optimizer"] != method_config(config, result["curvature"]):
        raise ValueError(f"Actual optimizer differs from the configured method: {folder}")
    if result["final"] != trace[-1] or trace[0]["iteration"] != 0:
        raise ValueError(f"Initial/final observations disagree: {folder}")
    for row in trace:
        if row["forward_views"] != row["backward_views"]:
            raise ValueError(f"Forward and adjoint counters differ: {folder}")
        if any("seconds" in field for field in row):
            raise ValueError(f"An optimizer timing field was recorded: {folder}")
        if not all(math.isfinite(row[field]) for field in ("raw_loss", "feasible_loss", "d_distance")):
            raise ValueError(f"Recorded diagnostic is nonfinite: {folder}")
    if allow_candidate_failure and result["status"] == "candidate_failed":
        if result.get("selection_eligible") is not False:
            raise ValueError("A failed development candidate was not excluded from selection")
        return result, trace
    if result["status"] != "completed" or result["iteration"] != trace[-1]["iteration"]:
        raise ValueError(f"Run has not completed: {folder}")
    if trace[-1]["forward_views"] < config["view_budget"]:
        raise ValueError(f"Run stopped below its optimization-work budget: {folder}")
    if config["record_every_iteration"] and [row["iteration"] for row in trace] != list(range(result["iteration"] + 1)):
        raise ValueError(f"Per-iteration history has gaps: {folder}")
    audit = metadata.get("continuation")
    if audit is not None:
        source = Path(audit["source_directory"])
        if sha256(source / "trace.json") != audit["source_trace_sha256"]:
            raise ValueError("The continuation source prefix changed")
        prefix = read(source / "trace.json")
        if trace[:len(prefix)] != prefix:
            raise ValueError("A continued trajectory differs from its immutable native prefix")
    return result, trace


def execute(jobs, state, *, progress_name="progress.json", phase="development"):
    write_json(ROOT / "execution_abort.json", dict(active=False, phase=phase))
    # Diagnostic work dominates dense evaluations; greedy assignment also balances
    # full-batch and minibatch configurations without claiming optimizer timings.
    loads = {cpu: 0 for cpu in CPUS}
    assignments = []
    for config in sorted(jobs, key=lambda row: row["run_id"]):
        cpu = min(loads, key=loads.get)
        dense = config["record_every_iteration"]
        iterations = config["view_budget"] / (config["batch"] if config["method"] == "sframes" else 256)
        loads[cpu] += 2 * config["view_budget"] + (iterations * (2 * config["n_train"] + 360) if dense else 0)
        assignments.append(dict(config=config, cpu=cpu))
    state.update(status="running", phase=phase, expected_runs=len(jobs))
    for row in assignments:
        state["runs"].setdefault(row["config"]["run_id"], dict(status="queued", cpu=row["cpu"]))
    write_json(ROOT / progress_name, state)
    ctx = mp.get_context("spawn")
    with ExitStack() as stack:
        pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx)) for cpu in CPUS}
        pending = {pools[row["cpu"]].submit(run, row["config"], row["cpu"]): row for row in assignments}
        for future in as_completed(pending):
            row = pending[future]
            name = row["config"]["run_id"]
            try:
                future.result()
                result, trace = validate(row["config"], allow_candidate_failure=phase == "development")
                item = dict(status=result["status"], cpu=row["cpu"], iteration=result["iteration"],
                            observations=len(trace), forward_views=trace[-1]["forward_views"],
                            feasible_loss=trace[-1]["feasible_loss"])
                if result["status"] == "candidate_failed":
                    item.update(error=result["error"], error_type=result["error_type"],
                                failure_classification=result["failure_classification"])
            except Exception as exc:
                item = dict(status="failed", cpu=row["cpu"], error=repr(exc))
            state["runs"][name] = item
            write_json(ROOT / progress_name, state)
            print(now(), name, item, flush=True)
            if item["status"] == "failed":
                write_json(ROOT / "execution_abort.json", dict(active=True, phase=phase, failed_run=name, error=item["error"]))
                for outstanding in pending:
                    outstanding.cancel()
                raise RuntimeError(f"Operational experiment failure requires inspection: {name}: {item['error']}")
    failures = [config["run_id"] for config in jobs
                if state["runs"][config["run_id"]]["status"] not in ("completed", "candidate_failed")]
    if failures:
        raise RuntimeError(f"Operational experiment failures require inspection: {failures}")


def score(grids):
    scores, selected = {}, {}
    for variant in BASELINES:
        candidates = []
        for candidate in trials(variant, grids[variant]):
            cases, eligible = [], True
            for sid, n in product(DEV_SLICES, TRAINING_VIEWS):
                seeds = (DEV_SEEDS[0],) if variant == "palm" else DEV_SEEDS
                seed_cases = []
                for seed in seeds:
                    config = job("development", sid, n, variant, candidate, seed)
                    result, trace = validate(config, allow_candidate_failure=True)
                    if result["status"] == "candidate_failed":
                        eligible = False
                        seed_cases.append(dict(run_id=config["run_id"], seed=seed,
                                               status="candidate_failed", error=result["error"]))
                        continue
                    final = next(row for row in reversed(trace) if max(row["forward_views"], row["backward_views"]) <= DEV_BUDGET)
                    seed_cases.append(dict(run_id=config["run_id"], seed=seed, status="completed",
                                           forward_views=final["forward_views"], initial_loss=trace[0]["feasible_loss"],
                                           repaired_loss=final["feasible_loss"],
                                           relative_loss=final["feasible_loss"] / trace[0]["feasible_loss"]))
                case_valid = all(row["status"] == "completed" for row in seed_cases)
                cases.append(dict(slice_id=sid, n_train=n, seed_cases=seed_cases,
                                  status="completed" if case_valid else "candidate_failed",
                                  relative_loss=median(row["relative_loss"] for row in seed_cases) if case_valid else None))
            candidates.append(dict(**candidate, cases=cases, selection_eligible=eligible,
                                   median_relative_loss=median(row["relative_loss"] for row in cases) if eligible else None))
        valid = [row for row in candidates if row["selection_eligible"]]
        if not valid:
            raise RuntimeError(f"No finite development candidate remains for {variant}")
        names = list(grids[variant])
        best = min(valid, key=lambda row: (row["median_relative_loss"], *(row[name] for name in names)))
        scores[variant] = candidates
        selected[variant] = {name: best[name] for name in (*names, "median_relative_loss")}
    selected.update({variant: dict(beta_scale=16.0, gamma0=0.1,
                                  selection_origin="frozen_previous_joint_development_search")
                     for variant in ("sframes", "sframes_full")})
    return dict(status="development_round_complete", selected=selected, scores=scores, grids=deepcopy(grids),
                development_view_budget=DEV_BUDGET,
                rule="Smallest median repaired-to-initial training-loss ratio, taking medians over the three "
                     "development oracle seeds within each slice/training-size case, then over the two original "
                     "development slices and both training-view counts at the last within-budget iterate. PALM "
                     "is deterministic and runs once per case. "
                     "Candidates with nonfinite arithmetic are excluded; operational/projection failures "
                     "halt selection. Ties favor smaller step multiplier, then smaller secondary parameter.")


def refine(grids, selected):
    updated, decisions, boundaries = deepcopy(grids), {}, {}
    for variant in BASELINES:
        decisions[variant], boundaries[variant] = [], {}
        for name, values in grids[variant].items():
            if name == "snapshot_policy":
                boundaries[variant][name] = dict(side="categorical", at_limit=False, values=values)
                continue
            lower, upper = RATE_LIMITS if name == "rate" else SECONDARY[variant][1]
            chosen = selected[variant][name]
            side = "lower" if chosen == values[0] else "upper" if chosen == values[-1] else "interior"
            addition = None
            if side == "lower" and chosen > lower:
                addition = max(lower, chosen / 2)
            if side == "upper" and chosen < upper:
                addition = min(upper, chosen * 2)
            if addition is not None:
                addition = float(f"{addition:.12g}") if name == "rate" else int(addition)
                updated[variant][name] = sorted(set(values + [addition]))
                decisions[variant].append(dict(parameter=name, side=side, added=addition))
            boundaries[variant][name] = dict(side=side, at_limit=side != "interior" and addition is None,
                                              search_limits=[lower, upper])
    return updated, decisions, boundaries


def campaign():
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("The frozen physical CPU assignment is unavailable")
    original = read(PREVIOUS / "manifest.json")
    manifest = dict(previous_manifest_sha256=sha256(PREVIOUS / "manifest.json"),
                    previous_selection_sha256=sha256(PREVIOUS / "selection.json"),
                    sources={name: sha256(ROOT / name) for name in SOURCES},
                    data={sid: sha256(ROOT / "data" / f"{sid}.npz") for sid in DEV_SLICES},
                    host=platform.node(), python=sys.executable, cpus=list(CPUS), blas_threads=1,
                    timing_measurements=False,
                    design=dict(development_slices=list(DEV_SLICES), development_seeds=list(DEV_SEEDS),
                                development_calibration_seed=110, development_training_views=list(TRAINING_VIEWS),
                                development_view_budget=DEV_BUDGET, evaluation_view_budget=EVAL_BUDGET,
                                initial_baseline_grids=initial_grids(), rate_limits=list(RATE_LIMITS),
                                secondary_limits={method: list(value[1]) for method, value in SECONDARY.items()},
                                frozen_sframes_parameters=dict(beta_scale=16.0, gamma0=0.1),
                                refinement="Expand each selected boundary by a factor of two to its predefined guard, "
                                           "retaining complete Cartesian grids; no evaluation data enter selection."))
    frozen(ROOT / "development_manifest.json", manifest)
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
            selection.update(boundaries=boundaries, round=search["round"],
                             status="expanding_grid" if any(decisions.values()) else "selected")
            write_json(ROOT / "selection.json", selection)
            search["history"].append(dict(round=search["round"], grids=deepcopy(search["grids"]),
                                          selected=selection["selected"], expansions=decisions, boundaries=boundaries))
            if not any(decisions.values()):
                search["status"] = "selected"
                write_json(search_path, search)
                write_json(ROOT / "baseline_selection.json", selection)
                state.update(status="completed", phase="development_selected", selected=selection["selected"], boundaries=boundaries)
                break
            search.update(round=search["round"] + 1, grids=updated)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = now()
        write_json(ROOT / "progress.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--cpus", help="Comma-separated physical CPUs, e.g. 4,6; also FRAMES_CAMPAIGN_CPUS")
    args = parser.parse_args()
    if args.cpus:
        global CPUS
        CPUS = tuple(int(value) for value in args.cpus.split(","))
        if not CPUS or len(set(CPUS)) != len(CPUS) or not set(CPUS).issubset({0, 2, 4, 6}):
            raise ValueError("--cpus must list distinct supported physical CPUs")
        os.environ["FRAMES_CAMPAIGN_CPUS"] = args.cpus
    if args.plan:
        print(json.dumps(dict(initial_grids=initial_grids(), initial_runs=len(development_jobs(initial_grids())),
                              development_view_budget=DEV_BUDGET, evaluation_view_budget=EVAL_BUDGET), indent=2))
        return
    with (ROOT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign()


if __name__ == "__main__":
    main()
