"""Paired 2DeteCT pilot with development-only rate selection."""

from run import ROOT, run, sha256, write_json

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys


METHOD_RATES = {
    "sframes": (0.10, 0.25),
    "spring_sarah": (0.30, 0.70),
    "prox_sarah": (0.40, 0.90),
    "palm": (0.60, 1.10),
}
DEV_SLICE = "slice04300"
EVAL_SLICES = ("slice04531", "slice04800")
DEV_BUDGET = 5760
EVAL_BUDGET = 21600
CPUS = (0, 2, 4, 6)


def now():
    return datetime.now(timezone.utc).isoformat()


def config(method, rate, phase, n_train, slice_id):
    development = phase == "development"
    return dict(run_id=f"{phase}_{slice_id}_{n_train}_{method}_{rate:g}",
                method=method, rate=rate, phase=phase, slice_id=slice_id,
                n_train=n_train, grid=512, rank=64, batch=64 if method != "palm" else n_train,
                seed=110 if development else 201,
                view_budget=DEV_BUDGET if development else EVAL_BUDGET,
                probe_every=720 if development else 2160)


def worker(job):
    cpu = CPUS[(mp.current_process()._identity[-1] - 1) % len(CPUS)]
    return run(job, cpu)


def execute(jobs, phase, state):
    state.update(phase=phase, runs={job["run_id"]: {"status": "queued"} for job in jobs})
    write_json(ROOT / "progress.json", state)
    with ProcessPoolExecutor(max_workers=len(CPUS), mp_context=mp.get_context("spawn")) as executor:
        submitted = {executor.submit(worker, job): job for job in jobs}
        for future in as_completed(submitted):
            job = submitted[future]
            name = job["run_id"]
            try:
                row = future.result()
                state["runs"][name] = dict(status="completed", iteration=row["iteration"],
                                           forward_views=row["forward_views"], loss=row["feasible_loss"])
            except Exception as exc:
                state["runs"][name] = dict(status="failed", error=repr(exc))
            print(name, state["runs"][name], flush=True)
            write_json(ROOT / "progress.json", state)
    if any(item["status"] == "failed" for item in state["runs"].values()):
        raise RuntimeError(f"At least one {phase} arm failed")


def select(development):
    scores = {}
    for job in development:
        path = ROOT / "runs" / job["run_id"] / "trace.json"
        trace = json.loads(path.read_text())
        probe = [row for row in trace if row["forward_views"] <= DEV_BUDGET][-1]
        scores.setdefault(job["method"], []).append(dict(rate=job["rate"],
            feasible_loss=probe["feasible_loss"], forward_views=probe["forward_views"]))
    chosen = {name: min(options, key=lambda row: row["feasible_loss"])["rate"]
              for name, options in scores.items()}
    write_json(ROOT / "selection.json", dict(selected=chosen, scores=scores,
               rule="Minimum repaired training loss at the last saved probe within 5760 forward views."))
    return chosen


def main():
    lock = (ROOT / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Requested physical CPU affinities are unavailable")
    names = ("campaign.py", "run.py", "ct_problem.py", "methods.py", "report.py")
    manifest = dict(
        sources={name: sha256(ROOT / name) for name in names},
        data={name: sha256(ROOT / "data" / f"{name}.npz") for name in (DEV_SLICE, *EVAL_SLICES)},
        host=platform.node(), python=sys.executable, cpus=CPUS, blas_threads=1,
        design=dict(development_slice=DEV_SLICE, evaluation_slices=EVAL_SLICES,
                    grid=512, rank=64, development_view_budget=DEV_BUDGET,
                    evaluation_view_budget=EVAL_BUDGET, evaluation_training_views=[360, 1080],
                    stochastic_batch=64, heldout_views=360, development_rates=METHOD_RATES,
                    stop="First completed update reaching the forward-view budget.",
                    timing="Optimizer work only. Arms share the host, one BLAS thread per physical core.",
                    spring="Alternating SPRING-SARAH, practical tuned fixed steps, independent 1/16 full refresh per block.",
                    proxsarah="Joint Algorithm 1 with minibatch snapshots, empirical tuned steps.",
                    constraints="Nonnegative factors with separate Frobenius caps.",
                    reference="Mode 2 TIFF reflected into DeepInv convention and converted to per-mm attenuation.",
                    inference="One development slice and two held-out evaluation slices. Pilot."))
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("Campaign provenance changed")
    write_json(manifest_path, manifest)
    state = dict(status="running", started_utc=now(), pid=os.getpid())
    development = [config(method, rate, "development", 360, DEV_SLICE)
                   for method, rates in METHOD_RATES.items() for rate in rates]
    try:
        execute(development, "development", state)
        chosen = select(development)
        evaluation = [config(method, rate, "evaluation", n, slice_id)
                      for slice_id in EVAL_SLICES for n in (360, 1080)
                      for method, rate in chosen.items()]
        execute(evaluation, "evaluation", state)
        subprocess.run([sys.executable, str(ROOT / "report.py"), "--root", str(ROOT)], check=True)
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(ROOT / "progress.json", dict(state, finished_utc=now()))


if __name__ == "__main__":
    main()
