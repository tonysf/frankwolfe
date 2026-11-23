"""Frozen development grids and paired evaluation on fresh 2DeteCT slices."""

from run import ROOT, prepare_calibration, run, sha256, write_json

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
import platform
import random
from statistics import median
import subprocess
import sys
from time import monotonic, sleep


CPUS = (0, 2, 4, 6)
GRID, RANK = 512, 64
N_TRAIN = (360, 1080)
DEV_SLICES = ("slice04300", "slice03150")
EVAL_MIXES = {
    "slice03041": "Mix 2", "slice03381": "Mix 2", "slice03611": "Mix 2",
    "slice03791": "Mix 3", "slice04131": "Mix 3", "slice04931": "Mix 3",
}
EVAL_SEEDS = (202, 203, 204)
DETERMINISTIC = ("sframes_full", "palm")
DEV_BUDGET, EVAL_BUDGET, TIME_CAP = 8640, 17280, 75

TUNING = {
    "sframes": tuple(dict(rate=rate) for rate in (.05, .10, .16, .25)),
    "sframes_full": tuple(dict(rate=rate) for rate in (.10, .25, .50, .80)),
    "spring_sarah": tuple(dict(rate=rate, refresh_period=period) for rate, period in (
        (.70, 16), (1.10, 16), (1.60, 16), (1.10, 8), (1.10, 32))),
    "prox_sarah": tuple(dict(rate=rate, inner_steps=period) for rate, period in (
        (.90, 12), (1.50, 12), (2.30, 12), (1.50, 6), (1.50, 24))),
    "palm": tuple(dict(rate=rate) for rate in (.60, 1.10, 1.60, 2.30)),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def job(phase, slice_id, n_train, variant, seed, trial, params):
    development = phase == "development"
    method = "sframes" if variant == "sframes_full" else variant
    return dict(run_id=f"{phase}_{slice_id}_{n_train}_{variant}_seed{seed}_trial{trial}",
                phase=phase, slice_id=slice_id, n_train=n_train, method=method, variant=variant,
                trial=trial, seed=seed, calibration_seed=110 if development else 201,
                grid=GRID, rank=RANK, batch=n_train if variant in DETERMINISTIC else 64,
                view_budget=DEV_BUDGET if development else EVAL_BUDGET,
                probe_every=2160 if development else 4320,
                time_budget_seconds=None if development else TIME_CAP, **params)


def read_json(path):
    return json.loads(path.read_text())


def write_frozen(path, data):
    if path.exists() and read_json(path) != json.loads(json.dumps(data)):
        raise ValueError(f"Frozen configuration has changed: {path}")
    write_json(path, data)


def wait_for_inputs(slice_ids, state, manifest):
    """Accept prepared acquisitions when their data and provenance both exist."""
    deadline = monotonic() + 90 * 60
    while True:
        missing = [s for s in slice_ids if any(
            not (ROOT / "data" / f"{s}.{suffix}").is_file() for suffix in ("npz", "json"))]
        if not missing:
            break
        if monotonic() >= deadline:
            raise TimeoutError(f"Still waiting for prepared slices: {', '.join(missing)}")
        state.update(phase="waiting_for_data", pending_slices=missing)
        write_json(ROOT / "progress.json", state)
        sleep(30)
    state.pop("pending_slices", None)
    for slice_id in slice_ids:
        sidecar = ROOT / "data" / f"{slice_id}.json"
        info = read_json(sidecar)
        if (info["slice_number"] != int(slice_id[5:]) or info["measurement_mode"] != 1
                or info["reference_mode"] != 2 or info["measured_shape"] != [3600, 956]):
            raise ValueError(f"Unexpected data provenance for {slice_id}")
        for key, path in (("data", sidecar.with_suffix(".npz")), ("data_sidecars", sidecar)):
            digest = sha256(path)
            if manifest[key].setdefault(slice_id, digest) != digest:
                raise ValueError(f"Registered data changed: {path}")
    write_json(ROOT / "manifest.json", manifest)


def pools(stack):
    ctx = mp.get_context("spawn")
    return [stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx))
            for _ in CPUS]


def execute(jobs, phase, state):
    # Stable assignment for resume, shuffled to distribute methods and cases across cores.
    queue = list(jobs)
    random.Random(f"ct-expanded-{phase}").shuffle(queue)
    state["phase"] = phase
    state["runs"].update({task["run_id"]: {"status": "queued", "cpu": CPUS[i % len(CPUS)]}
                          for i, task in enumerate(queue)})
    write_json(ROOT / "progress.json", state)
    with ExitStack() as stack:
        executors = pools(stack)
        submitted = {}
        for i, task in enumerate(queue):
            lane = i % len(CPUS)
            future = executors[lane].submit(run, task, CPUS[lane])
            submitted[future] = task["run_id"], CPUS[lane]
        for future in as_completed(submitted):
            name, cpu = submitted[future]
            try:
                row = future.result()
                item = dict(status="completed", cpu=cpu, iteration=row["iteration"],
                            forward_views=row["forward_views"], loss=row["feasible_loss"])
            except Exception as exc:
                item = dict(status="failed", cpu=cpu, error=repr(exc))
            state["runs"][name] = item
            write_json(ROOT / "progress.json", state)
            print(name, item, flush=True)


def calibrate(state, cases):
    state["phase"] = "calibration"
    write_json(ROOT / "progress.json", state)
    with ExitStack() as stack:
        executors = pools(stack)
        submitted = {}
        for i, (slice_id, seed) in enumerate(cases):
            lane = i % len(CPUS)
            submitted[executors[lane].submit(
                prepare_calibration, slice_id, seed, GRID, RANK, CPUS[lane])] = slice_id
        for future in as_completed(submitted):
            state["calibration_sha256"][submitted[future]] = future.result()
            write_json(ROOT / "progress.json", state)
    path = ROOT / "calibration.json"
    prior = read_json(path) if path.exists() else {}
    for slice_id, digest in state["calibration_sha256"].items():
        if prior.setdefault(slice_id, digest) != digest:
            raise ValueError(f"Calibration changed for {slice_id}")
    write_json(path, prior)


def select(jobs, state):
    scores = {variant: [list() for _ in grid] for variant, grid in TUNING.items()}
    for task in jobs:
        if state["runs"][task["run_id"]]["status"] != "completed":
            continue
        trace = read_json(ROOT / "runs" / task["run_id"] / "trace.json")
        row = next(probe for probe in reversed(trace) if probe["forward_views"] <= DEV_BUDGET)
        scores[task["variant"]][task["trial"]].append(dict(
            run_id=task["run_id"], forward_views=row["forward_views"],
            initial_loss=trace[0]["feasible_loss"], repaired_loss=row["feasible_loss"],
            relative_loss=row["feasible_loss"]/trace[0]["feasible_loss"]))
    selected, audit = {}, {}
    for variant, cases in scores.items():
        audit[variant] = [dict(trial=i, params=TUNING[variant][i], cases=items,
                              median_relative_loss=median(row["relative_loss"] for row in items)
                              if len(items) == len(DEV_SLICES)*len(N_TRAIN) else None)
                          for i, items in enumerate(cases)]
        complete = [trial for trial in audit[variant] if trial["median_relative_loss"] is not None]
        if not complete:
            raise RuntimeError(f"No candidate completed all development cases for {variant}")
        selected[variant] = min(complete, key=lambda trial: trial["median_relative_loss"])["trial"]
    write_frozen(ROOT / "selection.json", dict(selected=selected, scores=audit,
        rule="Smallest median repaired training loss relative to the common initial loss, "
             "at the last iterate within 8640 forward views, across both slices and view counts."))
    return selected


def verify_evaluation(jobs, state):
    cohorts = {}
    for task in jobs:
        name = task["run_id"]
        if state["runs"][name]["status"] != "completed":
            continue
        folder = ROOT / "runs" / name
        result = read_json(folder / "result.json")
        trace = read_json(folder / "trace.json")
        first = trace[0]
        keys = ("data_sha256", "calibration_sha256", "caps", "curvature",
                "train_view_ids", "validation_view_ids")
        signature = ([result[k] for k in keys], first["raw_loss"], first["feasible_loss"])
        case = task["slice_id"], task["n_train"]
        if cohorts.setdefault(case, signature) != signature:
            raise ValueError(f"Initial state or view split differs in {case}")
        if max(row["c_violation"] for row in trace) > 1e-10:
            raise ValueError(f"Spectral cap violation: {name}")


def main():
    lock = (ROOT / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Requested physical CPU affinities are unavailable")
    names = ("campaign.py", "run.py", "ct_problem.py", "methods.py",
             "spectral_reference_core.py", "spectral_reference_baseline.py")
    manifest = dict(
        sources={name: sha256(ROOT / name) for name in names},
        data={}, data_sidecars={},
        spectral_pilot_manifest_sha256=sha256(ROOT.parent / "manifest.json"),
        host=platform.node(), python=sys.executable, cpus=CPUS, blas_threads=1,
        design=dict(development_slices=DEV_SLICES, evaluation_slices=tuple(EVAL_MIXES),
                    evaluation_slice_mixes=EVAL_MIXES, evaluation_training_views=N_TRAIN,
                    evaluation_seeds=EVAL_SEEDS, deterministic_variants=DETERMINISTIC,
                    development_calibration_seed=110, evaluation_calibration_seed=201,
                    development_oracle_seed=110, grid=GRID, rank=RANK,
                    development_view_budget=DEV_BUDGET, evaluation_view_budget=EVAL_BUDGET,
                    evaluation_time_budget_seconds=TIME_CAP, stochastic_batch=64,
                    development_trials=TUNING, heldout_views=360,
                    constraint_geometry="Separate spectral-norm balls and nonnegative factors.",
                    full_batch="SFRAMES with all training views and separately tuned gamma0. "
                               "The decay exponents and smoothing scale are shared.",
                    inference="Fresh slices from separated acquisition batches of two sample mixes. "
                              "Slices in a mix share a sample tube; summarize paired effects descriptively.",
                    timing="Optimizer time only, four concurrent single-threaded physical-core workers.",
                    selection="No evaluation slice, held-out view, or reference image enters rate selection."))
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        for key in ("data", "data_sidecars"):
            manifest[key] = previous[key]
    write_frozen(manifest_path, manifest)
    state = dict(status="running", pid=os.getpid(), started_utc=now(), runs={}, calibration_sha256={})
    try:
        development = [job("development", s, n, variant, 110, i, params)
                       for s in DEV_SLICES for n in N_TRAIN for variant, trials in TUNING.items()
                       for i, params in enumerate(trials)]
        for slice_id in DEV_SLICES:
            wait_for_inputs([slice_id], state, manifest)
            calibrate(state, [(slice_id, 110)])
            execute([task for task in development if task["slice_id"] == slice_id],
                    f"development_{slice_id}", state)
        chosen = select(development, state)
        wait_for_inputs(EVAL_MIXES, state, manifest)
        calibrate(state, [(slice_id, 201) for slice_id in EVAL_MIXES])
        evaluation = [job("evaluation", s, n, variant, seed, i, TUNING[variant][i])
                      for s in EVAL_MIXES for n in N_TRAIN for variant, i in chosen.items()
                      for seed in (EVAL_SEEDS[:1] if variant in DETERMINISTIC else EVAL_SEEDS)]
        execute(evaluation, "evaluation", state)
        verify_evaluation(evaluation, state)
        state["phase"] = "report"
        write_json(ROOT / "progress.json", state)
        subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--root", str(ROOT)], check=True)
        subprocess.run([sys.executable, str(ROOT / "report_latex.py"), "--root", str(ROOT)], check=True)
        state["status"] = "completed" if all(
            item["status"] == "completed" for item in state["runs"].values()) else "completed_with_failures"
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(ROOT / "progress.json", dict(state, finished_utc=now()))


if __name__ == "__main__":
    main()
