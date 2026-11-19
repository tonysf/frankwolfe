"""Finite, checkpointed Elite campaign with no external monitoring loop."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from run import ROOT, atomic_json, now, run, source_hashes
from setup_problem import make_problem_config


def pipeline_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("campaign.py", "control_runner.py", "build_plots.py", "prepare_data.py")}


def prepare():
    bases = {}
    for dataset, rank in [("orl64", 25), ("yaleb32", 49)]:
        data_path = f"data/processed/{dataset}_unitnorm.npz"
        with np.load(ROOT / data_path, allow_pickle=False) as data:
            A = data["A"]
        geometry = make_problem_config(A, rank)
        base = dict(geometry, dataset=dataset, data_path=data_path,
                    mean_data_square=float(np.mean(A*A)),
                    batch=max(1, int(np.ceil(.05 * A.shape[1]))),
                    probe_every=100, checkpoint_every=250)
        bases[dataset] = base
        atomic_json(ROOT / "configs" / f"{dataset}_problem.json", base)
    grid = [dict(candidate=f"f{i:02d}", gamma0=g, rho0=r, mu_scale=b)
            for i, (g, r, b) in enumerate(itertools.product([.3, 1.], [.1, 1.], [.1, 1., 10.]))]
    manifest = dict(schema_version=1, created_utc=now(), bases=bases, grid=grid,
                    development_seeds=[110], confirmation_seeds=[201, 202, 203],
                    development_iterations=20000, confirmation_iterations=50000,
                    selection="Per dataset: minimum FINAL feasible diagnostic loss after 20000 iterations on development seed110; no early elimination. Confirm winner on three fresh seeds.",
                    cpu_affinity=[0, 2, 4, 6], workers=4, blas_threads_per_worker=1,
                    objective="Half mean squared pixel reconstruction loss; each image is normalized to unit Euclidean norm as in the author code.",
                    constraints="U spectral ball and nonnegative; V spectral ball and entrywise l1 budget, V may have either sign.",
                    oracle="Uniform image columns without replacement within a minibatch, independently redrawn each iteration/block; no random-sign sketch.",
                    sfr_frames_parameters="gamma=gamma0*(k+1)^(-3/4),rho=rho0*(k+1)^(-1/2),mu=(mu_scale/L_full_bound)*(k+1)^(-1/4); first sample initializes momentum; exact penalty added after momentum.",
                    baseline="SPRING-SARAH Corollary3.3 parameters and loopless estimator, with certified numerical Dykstra block projections. Finite floating-point oracle errors are reported; not claimed to equal exact mathematical proxes.",
                    publication_control="Separate author-code sparse-l0 NMF control has different constraints; its losses must not be overlaid as same-problem competitors.",
                    publication_control_gate="Before main runs, each original-code control must finish and reduce its own objective by at least20%; failed gates stop automatic continuation.",
                    publication_control_horizons=dict(orl64_source_epochs=1000, yaleb32_source_epochs=250),
                    publication_control_workers="Two independent runs, each using two physical cores and two BLAS threads.",
                    reference="Unconstrained rank-r SVD lower bound is a lower bound only; not a certified optimum for our constraints.",
                    theorem_statistic="Probe metrics only. No claim to measure the theorem's all-iterate averages or random-output expectation.",
                    normalization_excludes_initialization_cost=True,
                    source_sha256=source_hashes(),
                    pipeline_sha256=pipeline_hashes(),
                    python=sys.version, executable=sys.executable, platform=platform.platform())
    atomic_json(ROOT / "campaign_config.json", manifest)
    return manifest


def worker(task):
    kind, name, config = task
    cpu = [0, 2, 4, 6][(mp.current_process()._identity[0]-1) % 4]
    out = ROOT / "runs" / name
    if kind == "control":
        control_cpus = {0, 2} if config["dataset"] == "orl64" else {4, 6}
        os.sched_setaffinity(0, control_cpus)
        out.mkdir(parents=True, exist_ok=True)
        status_file = out / "status.json"
        if not (status_file.exists() and json.loads(status_file.read_text()).get("status") == "completed"):
            command = [sys.executable, str(ROOT / "control_runner.py"), "--dataset", config["dataset"],
                       "--source-epochs", str(config["source_epochs"]), "--seed", "0", "--threads", "2",
                       "--checkpoint-every", "5", "--output", str(out)]
            if (out / "checkpoint.npz").exists():
                command.append("--resume")
            with (out / "stdout.log").open("a") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        with (out / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        initial = float(rows[0]["loss_half_squared_frobenius"])
        final = float(rows[-1]["loss_half_squared_frobenius"])
        result = dict(config=config, status="completed", iteration=int(rows[-1]["outer_cycle"]),
                      initial_loss=initial, final=dict(feasible_loss=final), loss_reduction=1-final/initial,
                      scope="Separate original-author-code l0 NMF control, not same constrained problem.")
        atomic_json(out / "control_summary.json", result)
        if not np.isfinite(final) or final >= .8 * initial:
            raise RuntimeError(f"Published-code control did not clear the preregistered 20% decrease gate: {out}")
        return result
    if kind == "adapted":
        if (out / "result.json").exists():
            result = json.loads((out / "result.json").read_text())
            if result["status"] != "completed" or result["config"] != config:
                raise RuntimeError(f"Invalid existing result {out}")
            if result["source_sha256"] != source_hashes():
                raise RuntimeError(f"Code changed after completed run {out}")
            return result
        return run(config, out, cpu)
    raise ValueError(kind)


def execute(tasks, phase):
    results, failures = [], []
    atomic_json(ROOT / "progress.json", dict(phase=phase, completed=0, total=len(tasks), updated_utc=now()))
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(worker, task): task[1] for task in tasks}
        for future in as_completed(pending):
            name = pending[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(dict(event="run_completed", run=name, iteration=result["iteration"],
                                      feasible_loss=result["final"]["feasible_loss"], utc=now())), flush=True)
            except Exception as exc:
                failures.append(dict(run=name, error=repr(exc)))
                print(json.dumps(dict(event="run_failed", run=name, error=repr(exc), utc=now())), flush=True)
            atomic_json(ROOT / "progress.json", dict(phase=phase, completed=len(results), failed=len(failures),
                        total=len(tasks), latest_run=name, failures=failures, updated_utc=now()))
    atomic_json(ROOT / "reports" / f"{phase}_status.json", dict(completed=len(results), failures=failures))
    if failures:
        # Independent tasks complete, but no selection/confirmation uses incomplete development results.
        raise RuntimeError(f"{phase}: {len(failures)} failures, see reports/{phase}_status.json")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
        return
    lock_file = (ROOT / "campaign.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((ROOT / "campaign_config.json").read_text())
    if manifest["source_sha256"] != source_hashes():
        raise RuntimeError("Source changed since campaign configuration was frozen")
    if manifest["pipeline_sha256"] != pipeline_hashes():
        raise RuntimeError("Pipeline changed since campaign configuration was frozen")
    started = time.perf_counter()
    atomic_json(ROOT / "launch.json", dict(pid=os.getpid(), started_utc=now(), hostname=platform.node(),
                                           workers=4, physical_cpu_affinity=[0, 2, 4, 6]))
    controls = [("control", f"original_spring_{dataset}_seed0", dict(dataset=dataset, source_epochs=epochs))
                for dataset, epochs in [("orl64", 1000), ("yaleb32", 250)]]
    execute(controls, "published_code_control")
    tasks = []
    for dataset, base in manifest["bases"].items():
        for candidate in manifest["grid"]:
            for seed in manifest["development_seeds"]:
                config = dict(base, **candidate, method="sframes", seed=seed,
                              mu0=candidate["mu_scale"] / base["L_full_bound"],
                              iterations=manifest["development_iterations"], phase="development")
                name = f"{dataset}/development/{candidate['candidate']}_seed{seed}"
                atomic_json(ROOT / "configs" / f"{dataset}_{candidate['candidate']}_seed{seed}.json", config)
                tasks.append(("adapted", name, config))
        # The baseline is not tuned. Its development run is a visible control.
        config = dict(base, method="spring_sarah", baseline_mode="theorem", candidate="spring_theorem",
                      seed=110, iterations=manifest["development_iterations"], phase="development")
        tasks.insert(0, ("adapted", f"{dataset}/development/spring_theorem_seed110", config))
    tuned = execute(tasks, "development")
    selected, scores = {}, {}
    for dataset in manifest["bases"]:
        candidates = [r for r in tuned if r["config"]["dataset"] == dataset and r["config"]["method"] == "sframes"]
        best = min(candidates, key=lambda r:r["final"]["feasible_loss"])
        selected[dataset] = {k:best["config"][k] for k in ("candidate", "gamma0", "rho0", "mu_scale", "mu0")}
        scores[dataset] = {r["config"]["candidate"]:r["final"]["feasible_loss"] for r in candidates}
    atomic_json(ROOT / "selection.json", dict(selected=selected, scores=scores, rule=manifest["selection"]))
    subprocess.run([sys.executable, str(ROOT / "build_plots.py")], cwd=ROOT, check=True)
    tasks = []
    for dataset, base in manifest["bases"].items():
        for seed in manifest["confirmation_seeds"]:
            for method in ("sframes", "spring_sarah"):
                config = dict(base, method=method, seed=seed, iterations=manifest["confirmation_iterations"], phase="confirmation")
                if method == "sframes":
                    config.update(selected[dataset])
                else:
                    config.update(baseline_mode="theorem", candidate="spring_theorem")
                tasks.append(("adapted", f"{dataset}/confirmation/{method}_seed{seed}", config))
    confirmed = execute(tasks, "confirmation")
    atomic_json(ROOT / "completed.json", dict(status="completed", finished_utc=now(),
                elapsed_seconds=time.perf_counter()-started, development_runs=len(tuned),
                confirmation_runs=len(confirmed), selected=selected))
    subprocess.run([sys.executable, str(ROOT / "build_plots.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
