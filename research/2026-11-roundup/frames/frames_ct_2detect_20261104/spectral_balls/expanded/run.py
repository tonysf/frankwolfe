"""Run one checkpointed arm of the expanded spectral-ball CT comparison."""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter

import astra
import numpy as np

from ct_problem import CTProblem
from methods import make_solver, op_norm, repair_factor


ROOT = Path(__file__).resolve().parent
COUNTS = ("forward_views", "backward_views", "gradient_calls", "projection_calls",
          "projection_inner_iterations", "projection_dykstra_calls",
          "projection_floor_count", "projection_seconds")


def atomic_write(path, content):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def write_json(path, value):
    atomic_write(path, (json.dumps(value, allow_nan=False, indent=2) + "\n").encode())


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def method_config(config, curvature):
    u, v = curvature
    method, rate, batch = config["method"], config["rate"], config["batch"]
    if method == "sframes":
        return dict(batch=batch, gamma0=rate, rho0=1.0, mu0=1 / max(u, v))
    if method == "palm":
        return dict(eta_u=rate/u, eta_v=rate/v)
    if method == "spring_sarah":
        return dict(batch=batch, eta_u=rate/u, eta_v=rate/v,
                    refresh_period=config["refresh_period"])
    return dict(batch=batch, eta=rate/(u+v), relaxation=0.8, inner_steps=config["inner_steps"],
                snapshot_batch=min(config["n_train"], 4*batch))


def observe(problem, U, V, caps, counters, iteration):
    fixed_u, fixed_v = (repair_factor(matrix, cap) for matrix, cap in zip((U, V), caps))
    du = float(np.linalg.norm(U-fixed_u))
    dv = float(np.linalg.norm(V-fixed_v))
    image = fixed_u @ fixed_v.T
    repaired = problem.image_loss(image, problem.train_ids)
    raw = repaired if du + dv == 0 else problem.image_loss(U @ V.T, problem.train_ids)
    c_violation = max(op_norm(U) - caps[0], op_norm(V) - caps[1], 0.0)
    d_distance = float(np.hypot(np.linalg.norm(np.minimum(U, 0)),
                                np.linalg.norm(np.minimum(V, 0))))
    violation = max(c_violation, float(-U.min()), float(-V.min()))
    reference = problem.reference
    error = None if reference is None else float(np.linalg.norm(image-reference) / np.linalg.norm(reference))
    return dict(iteration=iteration, **counters, raw_loss=raw, feasible_loss=repaired,
                validation_loss=problem.image_loss(image, problem.validation_ids),
                reference_relative_error=error, max_constraint_violation=violation,
                repair_distance=float(np.hypot(du, dv)), c_violation=c_violation,
                d_distance=d_distance)


def calibration_path(slice_id, seed):
    return ROOT / "calibration" / f"{slice_id}_seed{seed}.npz"


def prepare_calibration(slice_id, seed, grid, rank, cpu):
    os.sched_setaffinity(0, {cpu})
    path = calibration_path(slice_id, seed)
    data_file = ROOT / "data" / f"{slice_id}.npz"
    data_hash = sha256(data_file)
    problem_hash = sha256(ROOT / "ct_problem.py")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (str(saved["data_sha256"]) != data_hash or str(saved["problem_sha256"]) != problem_hash
                    or saved["U"].shape != (grid, rank) or saved["V"].shape != (grid, rank)):
                raise ValueError(f"Calibration provenance changed: {path}")
        return sha256(path)
    problem = CTProblem(data_file, grid, 360)
    try:
        U, V, _, curvature = problem.calibration(rank, seed)
        caps = (2 * op_norm(U), 2 * op_norm(V))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez(stream, U=U, V=V, caps=caps, curvature=curvature,
                     data_sha256=data_hash, problem_sha256=problem_hash)
        temp.replace(path)
        return sha256(path)
    finally:
        problem.close()


def run(config, cpu=None, runs_dir=None):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    folder = (ROOT / "runs" if runs_dir is None else Path(runs_dir)) / config["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    data_file = ROOT / "data" / f'{config["slice_id"]}.npz'
    source = {name: sha256(ROOT / name) for name in (
        "run.py", "ct_problem.py", "methods.py", "spectral_reference_core.py",
        "spectral_reference_baseline.py")}
    problem = CTProblem(data_file, config["grid"], config["n_train"])
    try:
        calibration = calibration_path(config["slice_id"], config["calibration_seed"])
        data_hash = sha256(data_file)
        with np.load(calibration, allow_pickle=False) as saved:
            if str(saved["data_sha256"]) != data_hash or str(saved["problem_sha256"]) != source["ct_problem.py"]:
                raise ValueError(f"Calibration does not match current data and projector: {calibration}")
            U, V = saved["U"], saved["V"]
            caps, curvature = tuple(saved["caps"].tolist()), tuple(saved["curvature"].tolist())
        params = method_config(config, curvature)
        solver = make_solver(config["method"], problem, U, V, caps,
                             seed=config["seed"] + 100_000, config=params)
        metadata = dict(config=config, optimizer=params, caps=caps, curvature=curvature,
                        source_sha256=source, data_sha256=data_hash,
                        calibration_sha256=sha256(calibration),
                        projector="ASTRA CPU line_fanflat FP32", factor_dtype="float64",
                        numpy=np.__version__, astra=astra.__version__, cpu=cpu,
                        train_view_ids=problem.train_ids.tolist(),
                        validation_view_ids=problem.validation_ids.tolist(),
                        initialization=dict(name="warm", seed=config["calibration_seed"]),
                        constraint_geometry="separate spectral norm balls and entrywise nonnegativity")
        final_path = folder / "result.json"
        if final_path.exists():
            result = json.loads(final_path.read_text())
            if any(result[key] != metadata[key] for key in (
                    "config", "source_sha256", "data_sha256", "calibration_sha256")):
                raise ValueError("completed run provenance changed")
            return result["final"]
        checkpoint = folder / "checkpoint.pkl"
        counters = {key: 0 for key in COUNTS}
        counters["optimizer_seconds"] = 0.0
        counters["max_projection_distance_bound"] = 0.0
        counters["max_lmo_support_gap_bound"] = 0.0
        trace = []
        if checkpoint.exists():
            state = pickle.loads(checkpoint.read_bytes())
            if state["metadata"] != metadata:
                raise ValueError("checkpoint provenance changed")
            solver.load_state_dict(state["solver"])
            counters, trace = state["counters"], state["trace"]
        write_json(folder / "metadata.json", metadata)
        view_budget, probe_every = config["view_budget"], config["probe_every"]
        time_budget = config.get("time_budget_seconds")

        def save(status, probe=True):
            if probe and (not trace or trace[-1]["iteration"] != solver.k):
                trace.append(observe(problem, solver.U, solver.V, caps, counters, solver.k))
            atomic_write(checkpoint, pickle.dumps(dict(metadata=metadata, solver=solver.state_dict(),
                                                        counters=counters, trace=trace), protocol=5))
            write_json(folder / "trace.json", trace)
            write_json(folder / "progress.json", dict(status=status, iteration=solver.k, **counters))

        save("running")
        next_probe = (counters["forward_views"] // probe_every + 1) * probe_every
        try:
            while counters["forward_views"] < view_budget:
                old_u, old_v, previous = solver.U, solver.V, counters.copy()
                start = perf_counter()
                step = solver.step()
                counters["optimizer_seconds"] += perf_counter()-start
                for key in COUNTS:
                    counters[key] += step.get(key, 0)
                for key, field in (("max_projection_distance_bound", "projection_max_distance_bound"),
                                   ("max_lmo_support_gap_bound", "lmo_support_gap_bound")):
                    counters[key] = max(counters[key], step.get(field, 0.0))
                # All solvers replace their factor arrays on update, so the
                # previous factors can be probed without copying optimizer state.
                crosses_view_cap = previous["forward_views"] < view_budget < counters["forward_views"]
                crosses_time_cap = (time_budget is not None and
                                    previous["optimizer_seconds"] <= time_budget < counters["optimizer_seconds"])
                if (crosses_view_cap or crosses_time_cap) and trace[-1]["iteration"] < solver.k-1:
                    trace.append(observe(problem, old_u, old_v, caps, previous, solver.k-1))
                if solver.k == 1 or counters["forward_views"] >= next_probe or crosses_time_cap:
                    save("running")
                    next_probe = (counters["forward_views"] // probe_every + 1) * probe_every
            save("completed")
            write_json(final_path, dict(metadata, status="completed", iteration=solver.k, final=trace[-1]))
        except Exception as exc:
            # Keep the last valid checkpoint if a block fails partway through an update.
            write_json(folder / "progress.json", dict(status="failed", iteration=solver.k,
                                                       error=repr(exc), **counters))
            raise
        return trace[-1]
    finally:
        problem.close()
