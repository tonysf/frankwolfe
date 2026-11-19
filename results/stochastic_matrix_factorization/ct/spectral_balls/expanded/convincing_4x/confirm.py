"""Evaluate frozen settings on the expanded cohort at four times the work budget."""

from campaign import (BASELINES, CPUS, DEV_BUDGET, DEV_SEEDS, DEV_SLICES, EVAL_BUDGET,
                      PREVIOUS, SOURCES, TRAINING_VIEWS, VARIANTS, execute, frozen,
                      job, now, read, trials, validate)
from run import ROOT, NUMERICAL_SOURCES, method_config, prepare_calibration, sha256, write_json
from audit_provenance import audit_development, audit_evaluation, freeze_evaluation_anchor

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import fcntl
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def prepare(cohort):
    pairs = [(sid, 201) for sid in cohort["evaluation_slices"]]
    ctx = mp.get_context("spawn")
    with ExitStack() as stack:
        pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx)) for cpu in CPUS}
        pending = {pools[CPUS[index % len(CPUS)]].submit(prepare_calibration, sid, seed, 512, 64,
                                                       CPUS[index % len(CPUS)]): (sid, seed)
                   for index, (sid, seed) in enumerate(pairs)}
        for future in as_completed(pending):
            sid, seed = pending[future]
            print(now(), "calibration", sid, seed, future.result(), flush=True)


def previous_runs():
    return [(path.parent.resolve(), read(path)) for path in sorted((PREVIOUS / "runs").glob("evaluation_*/metadata.json"))]


def prior_prefix(config, prior, data_hash, calibration_hash, numeric_hashes):
    scientific = ("slice_id", "n_train", "method", "variant", "seed", "calibration_seed", "grid", "rank", "batch")
    matches = []
    for folder, metadata in prior:
        if any(metadata["config"][key] != config[key] for key in scientific):
            continue
        if metadata["data_sha256"] != data_hash or metadata["calibration_sha256"] != calibration_hash:
            raise ValueError(f"An original evaluation problem changed: {config['slice_id']}")
        if method_config(config, metadata["curvature"]) != metadata["optimizer"]:
            continue
        if any(metadata["source_sha256"][name] != numeric_hashes[name] for name in NUMERICAL_SOURCES):
            raise ValueError(f"Exact continuation would change a numerical helper: {folder}")
        matches.append(folder)
    if len(matches) > 1:
        raise ValueError(f"Multiple previous trajectories match the selected evaluation arm: {config['run_id']}")
    return str(matches[0]) if matches else None


def plan(selection, cohort):
    if selection["status"] != "selected":
        raise ValueError("Baseline development choices have not been frozen")
    prior = previous_runs()
    numeric_hashes = {name: sha256(ROOT / name) for name in NUMERICAL_SOURCES}
    data = {sid: sha256(ROOT / "data" / f"{sid}.npz") for sid in cohort["evaluation_slices"]}
    calibrations = {sid: sha256(ROOT / "calibration" / f"{sid}_seed201.npz") for sid in cohort["evaluation_slices"]}
    jobs = []
    for sid, n, variant in product(cohort["evaluation_slices"], TRAINING_VIEWS, VARIANTS):
        seeds = (cohort["evaluation_seeds"][0],) if variant in cohort["deterministic_variants"] else cohort["evaluation_seeds"]
        parameters = {name: value for name, value in selection["selected"][variant].items()
                      if name in ("rate", "refresh_period", "inner_steps", "snapshot_policy", "gamma0", "beta_scale")}
        for seed in seeds:
            config = job("evaluation", sid, n, variant, parameters, seed)
            prefix = prior_prefix(config, prior, data[sid], calibrations[sid], numeric_hashes)
            if prefix:
                config["continuation_directory"] = prefix
            jobs.append(config)
    expected = len(cohort["evaluation_slices"]) * len(TRAINING_VIEWS) * (
        len(cohort["deterministic_variants"]) + (len(VARIANTS) - len(cohort["deterministic_variants"])) * len(cohort["evaluation_seeds"]))
    if len(jobs) != expected or len({row["run_id"] for row in jobs}) != expected:
        raise ValueError("The expanded evaluation plan has duplicate or missing arms")
    return jobs


def manifest(selection, cohort, jobs):
    previous = read(PREVIOUS / "manifest.json")
    design = deepcopy(previous["design"])
    for name in ("development_trials", "development_variants", "initial_joint_grids", "refinement"):
        design.pop(name, None)
    design.update(
        variants=list(VARIANTS), development_slices=list(DEV_SLICES), development_seeds=list(DEV_SEEDS),
        development_calibration_seed=110, development_training_views=list(TRAINING_VIEWS),
        development_view_budget=DEV_BUDGET, evaluation_view_budget=EVAL_BUDGET,
        evaluation_slices=cohort["evaluation_slices"], evaluation_seeds=cohort["evaluation_seeds"],
        evaluation_training_views=list(TRAINING_VIEWS), deterministic_variants=cohort["deterministic_variants"],
        evaluation_slice_groups=cohort["evaluation_slice_groups"], specimen_group_labels=cohort["specimen_group_labels"],
        baseline_development_trials={variant: trials(variant, selection["grids"][variant]) for variant in BASELINES},
        baseline_development_grids=selection["grids"], fixed_sframes_parameters=dict(beta_scale=16.0, gamma0=0.1),
        diagnostics="Evaluation records initialization and every native completed optimizer update, including continued prefixes.",
        grouping="Median over seeds within slice, then slices within specimen group, then across groups.",
    )
    slices = list(DEV_SLICES) + cohort["evaluation_slices"]
    sidecars = {sid: sha256(ROOT / "data" / f"{sid}.json") for sid in slices if (ROOT / "data" / f"{sid}.json").exists()}
    result = dict(
        original_manifest_sha256=sha256(PREVIOUS / "manifest.json"),
        original_selection_sha256=sha256(PREVIOUS / "selection.json"),
        baseline_selection_sha256=sha256(ROOT / "baseline_selection.json"),
        original_sources=previous["sources"], sources={name: sha256(ROOT / name) for name in SOURCES},
        data={sid: sha256(ROOT / "data" / f"{sid}.npz") for sid in slices}, data_sidecars=sidecars,
        calibration={sid: sha256(ROOT / "calibration" / f"{sid}_seed201.npz") for sid in cohort["evaluation_slices"]},
        cohort_sha256=sha256(ROOT / "cohort.json"), targets_sha256=sha256(ROOT / "targets.json"),
        protocol_sha256=sha256(ROOT / "protocol_frozen.json"), selection_sha256=sha256(ROOT / "selection.json"),
        development_manifest_sha256=sha256(ROOT / "development_manifest.json"),
        host=platform.node(), python=sys.executable, cpus=list(CPUS), blas_threads=1,
        timing_measurements=False, design=design,
        physical_arm_origins=dict(continued_from_previous=sum("continuation_directory" in row for row in jobs),
                                  fresh=sum("continuation_directory" not in row for row in jobs)),
    )
    frozen(ROOT / "manifest.json", result)
    return result


def evaluate(*, postprocess=True):
    selection, cohort = read(ROOT / "selection.json"), read(ROOT / "cohort.json")
    if selection["status"] != "selected":
        raise ValueError("Baseline development search is not complete")
    # This external gate is intentionally outside the frozen numerical runner.
    # It anchors every development record to the preproduction inputs before
    # any expanded-cohort optimizer or newly generated calibration is used.
    print(now(), "development provenance", audit_development(ROOT, require_complete=True), flush=True)
    write_json(ROOT / "execution_abort.json", dict(active=False, phase="calibration"))
    prepare(cohort)
    jobs = plan(selection, cohort)
    manifest(selection, cohort, jobs)
    frozen(ROOT / "evaluation_plan.json", dict(selected_sha256=sha256(ROOT / "selection.json"),
                                               cohort_sha256=sha256(ROOT / "cohort.json"),
                                               source_sha256={name: sha256(ROOT / name) for name in SOURCES},
                                               orchestrator_sha256=sha256(Path(__file__)),
                                               expected_runs=len(jobs), runs=jobs))
    freeze_evaluation_anchor(ROOT)
    print(now(), "evaluation input provenance", audit_evaluation(ROOT), flush=True)
    state_path = ROOT / "evaluation_progress.json"
    state = read(state_path) if state_path.exists() else dict(runs={})
    state.update(status="running", phase="evaluation", started_utc=now(), pid=os.getpid(),
                 selected=selection["selected"], expected_runs=len(jobs))
    try:
        execute(jobs, state, progress_name="evaluation_progress.json", phase="evaluation")
        # A final full validation also verifies unchanged native prefix observations.
        details = [validate(config) for config in jobs]
        print(now(), "completed evaluation provenance", audit_evaluation(ROOT, require_complete=True), flush=True)
        state.update(status="completed", phase="evaluation_validated", failed_runs=[])
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = now()
        write_json(state_path, state)
    audit = dict(completed_utc=now(), selected=selection["selected"], evaluation_runs=len(jobs),
                 continued_runs=sum("continuation_directory" in row for row in jobs),
                 fresh_runs=sum("continuation_directory" not in row for row in jobs),
                 observations=sum(len(trace) for _, trace in details),
                 source_sha256={name: sha256(ROOT / name) for name in SOURCES})
    write_json(ROOT / "evaluation_completion.json", audit)
    if postprocess:
        subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--no-plots"], check=True)
        subprocess.run([sys.executable, str(ROOT / "report_update.py")], check=True)
        audit["paper_pdf"] = str(ROOT.parent / "reports" / "paper_section" / "sframes_ct_section.pdf")
        write_json(ROOT / "completion.json", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Wait for the development process and its lock")
    parser.add_argument("--no-postprocess", action="store_true")
    args = parser.parse_args()
    with (ROOT / "confirmation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            path = ROOT / "progress.json"
            state = read(path) if path.exists() else {}
            if state.get("phase") == "development_selected" and state.get("status") == "completed":
                break
            if state.get("status") == "failed":
                raise RuntimeError(f"Development failed: {state.get('error')}")
            if not args.wait:
                raise RuntimeError("Development selection is not complete")
            time.sleep(30)
        with (ROOT / "campaign.lock").open("a") as development:
            fcntl.flock(development, fcntl.LOCK_EX)
            evaluate(postprocess=not args.no_postprocess)


if __name__ == "__main__":
    main()
