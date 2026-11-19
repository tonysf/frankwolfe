"""Run retuned spectral CT experiments, with audited exact continuations."""

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import fcntl
import hashlib
import json
import pickle
from pathlib import Path
import time

import astra
import numpy as np

from ct_problem import CTProblem
from methods import make_solver, op_norm, repair_factor


ROOT = Path(__file__).resolve().parent
COUNTS = ("forward_views", "backward_views", "gradient_calls", "projection_calls",
          "projection_inner_iterations", "projection_dykstra_calls",
          "projection_floor_count")
NUMERICAL_SOURCES = ("ct_problem.py", "methods.py", "spectral_reference_core.py",
                     "spectral_reference_baseline.py")
PROBLEM_FIELDS = ("data_sha256", "calibration_sha256", "caps", "curvature",
                  "train_view_ids", "validation_view_ids", "initialization", "constraint_geometry")


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
        return dict(batch=batch, gamma0=rate, rho0=1.0,
                    mu0=float(config.get("beta_scale", 1.0)) / max(u, v))
    if method == "palm":
        return dict(eta_u=rate/u, eta_v=rate/v)
    if method == "spring_sarah":
        return dict(batch=batch, eta_u=rate/u, eta_v=rate/v,
                    refresh_period=config["refresh_period"])
    policy = config.get("snapshot_policy", "minibatch")
    if policy not in ("minibatch", "full"):
        raise ValueError("Unsupported ProxSARAH snapshot policy")
    snapshot = config["n_train"] if policy == "full" else min(config["n_train"], 4*batch)
    return dict(batch=batch, eta=rate/(u+v), relaxation=0.8, inner_steps=config["inner_steps"],
                snapshot_batch=snapshot)


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
    result = dict(iteration=iteration, **counters, raw_loss=raw, feasible_loss=repaired,
                validation_loss=problem.image_loss(image, problem.validation_ids),
                reference_relative_error=error, max_constraint_violation=violation,
                repair_distance=float(np.hypot(du, dv)), c_violation=c_violation,
                d_distance=d_distance)
    if any(not np.isfinite(result[key]) for key in
           ("raw_loss", "feasible_loss", "validation_loss", "d_distance", "repair_distance")):
        raise FloatingPointError("nonfinite optimizer diagnostic")
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def continuation(config, metadata, solver):
    """Read an immutable completed prefix and validate it before importing state."""
    source = config.get("continuation_directory")
    if source is None:
        return None
    folder = Path(source).resolve()
    if folder == (ROOT / "runs" / config["run_id"]).resolve():
        raise ValueError("A run cannot continue itself")
    result = read_json(folder / "result.json")
    previous = read_json(folder / "metadata.json")
    trace = read_json(folder / "trace.json")
    state = pickle.loads((folder / "checkpoint.pkl").read_bytes())
    if result.get("status") != "completed" or not trace or result["final"] != trace[-1]:
        raise ValueError("Continuation source is not a completed observed run")
    if json.loads(json.dumps(state["metadata"])) != previous or state["trace"] != trace:
        raise ValueError("Continuation checkpoint and recorded prefix disagree")
    if any(result[field] != value for field, value in previous.items()):
        raise ValueError("Continuation result and metadata disagree")
    if state["solver"]["k"] != result["iteration"] or trace[-1]["iteration"] != result["iteration"]:
        raise ValueError("Continuation checkpoint is not at the recorded final iteration")
    if config.get("record_every_iteration", True) and [row["iteration"] for row in trace] != list(range(result["iteration"] + 1)):
        raise ValueError("A dense confirmation requires a natively dense prefix")
    for field in PROBLEM_FIELDS:
        if previous[field] != metadata[field]:
            raise ValueError(f"Continuation changes the original problem on {field}")
    for name in NUMERICAL_SOURCES:
        if previous["source_sha256"][name] != metadata["source_sha256"][name]:
            raise ValueError(f"Continuation changes numerical code on {name}")
    for field in ("optimizer", "numpy", "astra", "factor_dtype", "projector"):
        if previous[field] != metadata[field]:
            raise ValueError(f"Continuation changes optimizer or environment on {field}")
    for field in ("slice_id", "n_train", "method", "variant", "seed", "calibration_seed", "grid", "rank", "batch"):
        if previous["config"][field] != config[field]:
            raise ValueError(f"Continuation changes scientific configuration on {field}")
    if state["counters"]["forward_views"] >= config["view_budget"]:
        raise ValueError("Continuation source has already consumed the requested new budget")
    if any(state["counters"][key] != trace[-1][key] for key in COUNTS):
        raise ValueError("Continuation counters disagree with the observed prefix")
    # load_state_dict validates factor feasibility, train IDs, optimizer state and RNG.
    solver.load_state_dict(state["solver"])
    audit = dict(source_directory=str(folder), source_checkpoint_sha256=sha256(folder / "checkpoint.pkl"),
                 source_trace_sha256=sha256(folder / "trace.json"),
                 source_prefix_sha256=sha256(folder / "trace.json"),
                 source_result_sha256=sha256(folder / "result.json"),
                 source_final_iteration=result["iteration"],
                 source_forward_views=state["counters"]["forward_views"],
                 source_backward_views=state["counters"]["backward_views"],
                 source_sha256=previous["source_sha256"], source_config=previous["config"],
                 preserves_optimizer_state_and_rng=True, source_prefix_observations=len(trace))
    return audit, state["counters"], trace


def scientific_failure(exc):
    return (isinstance(exc, FloatingPointError)
            or isinstance(exc, ValueError) and str(exc) == "projection expects a finite matrix")


def calibration_path(slice_id, seed):
    return ROOT / "calibration" / f"{slice_id}_seed{seed}.npz"


def execution_gate(cpu):
    """Resource-only governor, checked before loading any problem or mutating runs."""
    check_abort()
    if cpu is None:
        return
    path = ROOT / "execution_cpus.json"
    while path.exists():
        check_abort()
        allowed = read_json(path)["allowed_cpus"]
        if not isinstance(allowed, list) or not allowed or not set(allowed).issubset({0, 2, 4, 6}):
            raise ValueError("execution_cpus.json must name supported physical CPUs")
        if cpu in allowed:
            return
        time.sleep(30)


def check_abort():
    path = ROOT / "execution_abort.json"
    if path.exists() and read_json(path).get("active"):
        raise RuntimeError("Campaign halted after an operational failure; retained checkpoints are resumable")


def prepare_calibration(slice_id, seed, grid, rank, cpu):
    execution_gate(cpu)
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
    execution_gate(cpu)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    folder = (ROOT / "runs" if runs_dir is None else Path(runs_dir)) / config["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(config, cpu, folder)


def _run(config, cpu, folder):
    every_iteration = config.get("record_every_iteration", True)
    if not isinstance(every_iteration, bool):
        raise ValueError("record_every_iteration must be a boolean")
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
                        initial_factor_frobenius_norm=float(np.hypot(np.linalg.norm(U), np.linalg.norm(V))),
                        initial_factor_norm=float(np.hypot(np.linalg.norm(U), np.linalg.norm(V))),
                        source_sha256=source, data_sha256=data_hash,
                        calibration_sha256=sha256(calibration),
                        projector="ASTRA CPU line_fanflat FP32", factor_dtype="float64",
                        numpy=np.__version__, astra=astra.__version__, cpu=cpu,
                        train_view_ids=problem.train_ids.tolist(),
                        validation_view_ids=problem.validation_ids.tolist(),
                        initialization=dict(name="warm", seed=config["calibration_seed"]),
                        diagnostics=dict(cadence="every_completed_iteration" if every_iteration else
                                         "initial_final_and_within_budget_endpoint",
                                         includes_iteration_zero=True,
                                         optimizer_work_counters_only=True),
                        constraint_geometry="separate spectral norm balls and entrywise nonnegativity")
        metadata = json.loads(json.dumps(metadata))
        imported = continuation(config, metadata, solver)
        metadata["continuation"] = None if imported is None else imported[0]
        final_path = folder / "result.json"
        if final_path.exists():
            result = json.loads(final_path.read_text())
            if any(result[key] != metadata[key] for key in (
                    "config", "source_sha256", "data_sha256", "calibration_sha256", "optimizer", "continuation")):
                raise ValueError("completed run provenance changed")
            return result["final"]
        checkpoint = folder / "checkpoint.pkl"
        counters = {key: 0 for key in COUNTS}
        counters["max_projection_distance_bound"] = 0.0
        counters["max_lmo_support_gap_bound"] = 0.0
        trace = []
        if imported is not None:
            _, counters, trace = imported
        if checkpoint.exists():
            state = pickle.loads(checkpoint.read_bytes())
            if state["metadata"] != metadata:
                raise ValueError("checkpoint provenance changed")
            solver.load_state_dict(state["solver"])
            counters, trace = state["counters"], state["trace"]
        write_json(folder / "metadata.json", metadata)
        view_budget = config["view_budget"]

        def save(status, probe=True):
            if probe and (not trace or trace[-1]["iteration"] != solver.k):
                trace.append(observe(problem, solver.U, solver.V, caps, counters, solver.k))
            atomic_write(checkpoint, pickle.dumps(dict(metadata=metadata, solver=solver.state_dict(),
                                                        counters=counters, trace=trace), protocol=5))
            write_json(folder / "trace.json", trace)
            write_json(folder / "progress.json", dict(status=status, iteration=solver.k, **counters))

        save("running", probe=every_iteration or solver.k == 0)
        try:
            while counters["forward_views"] < view_budget:
                check_abort()
                old_u, old_v, previous = solver.U, solver.V, counters.copy()
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    step = solver.step()
                for key in COUNTS:
                    counters[key] += step.get(key, 0)
                for key, field in (("max_projection_distance_bound", "projection_max_distance_bound"),
                                   ("max_lmo_support_gap_bound", "lmo_support_gap_bound")):
                    counters[key] = max(counters[key], step.get(field, 0.0))
                if (not every_iteration and
                        previous["forward_views"] < view_budget < counters["forward_views"] and
                        trace[-1]["iteration"] < solver.k - 1):
                    trace.append(observe(problem, old_u, old_v, caps, previous, solver.k - 1))
                save("running", probe=every_iteration)
            save("completed")
            write_json(final_path, dict(metadata, status="completed", iteration=solver.k, final=trace[-1]))
        except Exception as exc:
            # Keep the last valid checkpoint if a block fails partway through an update.
            candidate = config.get("phase") == "development" and scientific_failure(exc)
            status = "candidate_failed" if candidate else "failed"
            details = dict(status=status, iteration=solver.k, error=repr(exc),
                           error_type=type(exc).__name__, **counters)
            write_json(folder / "progress.json", details)
            if candidate:
                write_json(final_path, dict(metadata, **details, final=trace[-1],
                                           failure_classification="nonfinite_arithmetic_or_diagnostic",
                                           selection_eligible=False))
                return trace[-1]
            raise
        return trace[-1]
    finally:
        problem.close()
