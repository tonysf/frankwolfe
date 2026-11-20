"""Run one checkpointed CT optimizer comparison arm."""

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
from methods import ball_lmo, make_solver, project_factor


ROOT = Path(__file__).resolve().parent
COUNTS = ("forward_views", "backward_views", "gradient_calls")


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
        return dict(batch=batch, eta_u=rate/u, eta_v=rate/v, refresh_period=16)
    return dict(batch=batch, eta=rate/(u+v), relaxation=0.8, inner_steps=12,
                snapshot_batch=min(config["n_train"], 4*batch))


def observe(problem, U, V, caps, counters, iteration):
    fixed_u, fixed_v = (project_factor(matrix, cap) for matrix, cap in zip((U, V), caps))
    du = float(np.linalg.norm(U-fixed_u))
    dv = float(np.linalg.norm(V-fixed_v))
    image = fixed_u @ fixed_v.T
    repaired = problem.image_loss(image, problem.train_ids)
    raw = repaired if du + dv == 0 else problem.image_loss(U @ V.T, problem.train_ids)
    c_violation = max(float(np.linalg.norm(U)) - caps[0], float(np.linalg.norm(V)) - caps[1], 0.0)
    violation = max(c_violation, float(-U.min()), float(-V.min()))
    negative_u = float(np.linalg.norm(np.minimum(U, 0.0)))
    negative_v = float(np.linalg.norm(np.minimum(V, 0.0)))
    reference = problem.reference
    error = None if reference is None else float(np.linalg.norm(image-reference) / np.linalg.norm(reference))
    return dict(iteration=iteration, **counters, raw_loss=raw, feasible_loss=repaired,
                validation_loss=problem.image_loss(image, problem.validation_ids),
                reference_relative_error=error, max_constraint_violation=violation,
                repair_distance=float(np.hypot(du, dv)), c_violation=c_violation,
                d_distance=float(np.hypot(negative_u, negative_v)),
                u_negative_norm=negative_u, v_negative_norm=negative_v)


def initialize_lmo(solver, seed):
    """Replace the calibrated factors with one signed Frobenius atom per block."""
    rng = np.random.default_rng(seed)
    state = solver.state_dict()
    norms, negative_norms = [], []
    for name, cap in (("U", solver.ru), ("V", solver.rv)):
        atom = ball_lmo(rng.standard_normal(state[name].shape), cap)
        norm = float(np.linalg.norm(atom))
        negative_norm = float(np.linalg.norm(np.minimum(atom, 0.0)))
        if not np.isclose(norm, cap, atol=0, rtol=1e-12) or negative_norm <= 0:
            raise ValueError("Expected an LMO atom on the Frobenius cap and outside the orthant")
        state[name] = atom
        norms.append(norm)
        negative_norms.append(negative_norm)
    # Reuse the optimizer's validated state path. Momentum and draw state are
    # still those of the fresh solver, which has not taken an optimizer step.
    solver.load_state_dict(state)
    return dict(name="Gaussian-direction Frobenius LMO", seed=seed,
                factor_norms=norms, negative_norms=negative_norms)


def run(config, cpu=None):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    folder = ROOT / "runs" / config["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    data_file = ROOT / "data" / f'{config["slice_id"]}.npz'
    source = {name: sha256(ROOT / name) for name in ("run.py", "ct_problem.py", "methods.py")}
    problem = CTProblem(data_file, config["grid"], config["n_train"])
    try:
        U, V, caps, curvature = problem.calibration(config["rank"], config["seed"])
        params = method_config(config, curvature)
        if config["method"] != "sframes":
            raise ValueError("This initialization ablation supports SFRAMES only")
        solver = make_solver(config["method"], problem, U, V, caps,
                             seed=config["seed"] + 100_000, config=params)
        initialization = initialize_lmo(solver, config["seed"] + 200_000)
        metadata = dict(config=config, optimizer=params, caps=caps, curvature=curvature,
                        source_sha256=source, data_sha256=sha256(data_file),
                        projector="ASTRA CPU line_fanflat FP32", factor_dtype="float64",
                        numpy=np.__version__, astra=astra.__version__, cpu=cpu,
                        train_view_ids=problem.train_ids.tolist(),
                        validation_view_ids=problem.validation_ids.tolist(),
                        initialization=initialization)
        original_path = ROOT.parent / "runs" / config["run_id"] / "result.json"
        original = json.loads(original_path.read_text())
        for key in ("config", "optimizer", "caps", "curvature", "data_sha256",
                    "train_view_ids", "validation_view_ids", "numpy", "astra", "projector"):
            if original[key] != json.loads(json.dumps(metadata[key])):
                raise ValueError(f"Paired original run differs in {key}")
        final_path = folder / "result.json"
        if final_path.exists():
            result = json.loads(final_path.read_text())
            if any(result[key] != metadata[key] for key in ("config", "source_sha256", "data_sha256")):
                raise ValueError("completed run provenance changed")
            return result["final"]
        checkpoint = folder / "checkpoint.pkl"
        counters = {key: 0 for key in COUNTS}
        counters["optimizer_seconds"] = 0.0
        trace = []
        if checkpoint.exists():
            state = pickle.loads(checkpoint.read_bytes())
            if state["metadata"] != metadata:
                raise ValueError("checkpoint provenance changed")
            solver.load_state_dict(state["solver"])
            counters, trace = state["counters"], state["trace"]
        write_json(folder / "metadata.json", metadata)
        view_budget, probe_every = config["view_budget"], config["probe_every"]

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
                start = perf_counter()
                step = solver.step()
                counters["optimizer_seconds"] += perf_counter()-start
                for key in COUNTS:
                    counters[key] += step[key]
                if solver.k == 1 or counters["forward_views"] >= next_probe:
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
