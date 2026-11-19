"""Record budgeted nonnegative CT baselines from the common frozen factors.

Every accepted factor update is observed. A terminal observation can repeat
the last accepted iteration to account for charged, unaccepted line-search
work. Spectral norms are diagnostics against the historical reference radii;
they never affect the optimizer or the repaired image.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import pickle
import platform
import traceback

import numpy as np
import scipy

try:
    from .nonnegative_solvers import make_solver
except ImportError:
    from nonnegative_solvers import make_solver


SOURCE_ROOT = Path(__file__).resolve().parent
SOURCES = ("nonnegative_run.py", "nonnegative_solvers.py", "ct_problem.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def source_hashes():
    return {name: sha256(SOURCE_ROOT / name) for name in SOURCES}


def read(path):
    return json.loads(Path(path).read_text())


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def plain(value):
    return json.loads(json.dumps(value, default=json_scalar, allow_nan=False))


def atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write(path, value):
    atomic_bytes(path, (json.dumps(plain(value), indent=2, allow_nan=False) + "\n").encode())


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def make_problem(config):
    try:
        from .ct_problem import CTProblem
    except ImportError:
        from ct_problem import CTProblem
    return CTProblem(config["data_file"], config["grid"], config["n_train"])


def observe(problem, solver, reference_caps, event):
    U, V = solver.U, solver.V
    if not np.isfinite(U).all() or not np.isfinite(V).all():
        raise FloatingPointError("Nonfinite accepted factors")
    positive_u, positive_v = np.maximum(U, 0.0), np.maximum(V, 0.0)
    distance = float(np.hypot(np.linalg.norm(U - positive_u), np.linalg.norm(V - positive_v)))
    raw_image = U @ V.T
    image = positive_u @ positive_v.T
    raw_loss = problem.image_loss(raw_image, problem.train_ids)
    feasible_loss = raw_loss if distance == 0 else problem.image_loss(image, problem.train_ids)
    reference = problem.reference
    reference_error = None if reference is None else float(np.linalg.norm(image - reference) / np.linalg.norm(reference))
    norms = [float(np.linalg.norm(factor, 2)) for factor in (U, V)]
    counters = plain(solver.counters)
    row = dict(iteration=int(solver.iteration), observation_type=event["observation_type"],
        completed_blocks=int(solver.completed_blocks), outer_sweeps=int(solver.outer_sweeps),
        **counters, processed_views=max(counters["forward_views"], counters["backward_views"]),
        raw_loss=raw_loss, feasible_loss=feasible_loss,
        validation_loss=problem.image_loss(image, problem.validation_ids),
        reference_relative_error=reference_error, d_distance=distance,
        repair_distance=distance, max_constraint_violation=max(0.0, float(-U.min()), float(-V.min())),
        spectral_norm_u=norms[0], spectral_norm_v=norms[1],
        reference_spectral_ratio_u=norms[0] / reference_caps[0],
        reference_spectral_ratio_v=norms[1] / reference_caps[1],
        factor_u_sha256=array_hash(U), factor_v_sha256=array_hash(V), last_step=plain(event))
    if any(not np.isfinite(row[name]) for name in
           ("raw_loss", "feasible_loss", "validation_loss", "d_distance", "repair_distance",
            "spectral_norm_u", "spectral_norm_v")):
        raise FloatingPointError("Nonfinite factor diagnostic")
    if distance != 0:
        raise ValueError("A nonnegative baseline produced an infeasible accepted factor")
    return plain(row)


def check_abort(root):
    path = root / "execution_abort.json"
    if path.exists() and read(path).get("active"):
        raise RuntimeError("Campaign halted; retained checkpoints require review")


def run(config, cpu=None, root=None, output=None, problem_factory=None):
    """Run or resume one immutable configuration, returning its full result."""
    if root is None:
        raise ValueError("An explicit campaign root is required")
    root = Path(root).resolve()
    folder = Path(output).resolve() if output is not None else root / "runs" / config["run_id"]
    folder.mkdir(parents=True, exist_ok=True)
    if cpu is not None:
        os.sched_setaffinity(0, {int(cpu)})
    with (folder / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(plain(config), cpu, root, folder, problem_factory or make_problem)


def _run(config, cpu, root, folder, problem_factory):
    if (config["method"] not in ("anls", "mu")
            or config["phase"] not in ("development", "evaluation")
            or config["record_every_iteration"] is not True
            or config["view_budget"] <= 0):
        raise ValueError("Invalid nonnegative baseline configuration")
    sources = source_hashes()
    protocol = read(root / "protocol.json")
    expected = protocol["numerical_sources"]
    if any(expected.get(name) != value for name, value in sources.items()):
        raise ValueError("Numerical code differs from frozen campaign sources")
    for field in ("data", "calibration"):
        if sha256(config[field + "_file"]) != config[field + "_sha256"]:
            raise ValueError("Frozen input changed: " + field)
    with np.load(config["calibration_file"], allow_pickle=False) as saved:
        if (str(saved["data_sha256"]) != config["data_sha256"]
                or str(saved["problem_sha256"]) != sources["ct_problem.py"]):
            raise ValueError("Calibration does not match the frozen CT problem")
        U, V = saved["U"].copy(), saved["V"].copy()
        caps = tuple(float(value) for value in saved["caps"])
    if (U.shape != (config["grid"], config["rank"]) or V.shape != U.shape
            or not np.isfinite(U).all() or not np.isfinite(V).all()
            or np.any(U < 0) or np.any(V < 0)):
        raise ValueError("Invalid common nonnegative initialization")
    problem = problem_factory(config)
    try:
        solver = make_solver(config["method"], problem, U, V, config=config["parameters"])
        metadata = plain(dict(config=config, optimizer=config["parameters"], source_sha256=sources,
            protocol_sha256=sha256(root / "protocol.json"), cpu=cpu,
            data_sha256=config["data_sha256"], calibration_sha256=config["calibration_sha256"],
            geometry="nonnegative", constraint_geometry="entrywise nonnegative factors; no spectral constraints",
            reference_spectral_caps=caps, spectral_caps_are_constraints=False,
            initial_factor_frobenius_norm=float(np.hypot(np.linalg.norm(U), np.linalg.norm(V))),
            initial_u_sha256=array_hash(U), initial_v_sha256=array_hash(V),
            initialization=dict(name="shared frozen warm start", seed=config["calibration_seed"]),
            train_view_ids=problem.train_ids.tolist(), validation_view_ids=problem.validation_ids.tolist(),
            numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
            projector="ASTRA CPU line_fanflat FP32", factor_dtype="float64", deterministic=True,
            diagnostics=dict(cadence="every accepted factor update and terminal charged-work state",
                repair="orthant only; no radial or spectral repair",
                optimizer_timing=False, optimizer_work_counters_only=True,
                view_cap="separate forward and adjoint caps; every inner-solver call and MU RHS adjoint charged",
                oracle_calls="completed forward/adjoint pairs plus standalone MU RHS adjoint",
                terminal_observation="may repeat the last accepted iteration after unaccepted charged work")))
        result_file = folder / "result.json"
        if result_file.exists():
            result = read(result_file)
            if result.get("metadata") != metadata or result["status"] != "completed":
                raise ValueError("Completed run provenance changed")
            return result
        checkpoint = folder / "checkpoint.pkl"
        if checkpoint.exists():
            receipt = read(folder / "checkpoint.sha256.json")
            if sha256(checkpoint) not in (receipt["sha256"], receipt.get("previous_sha256")):
                raise ValueError("Checkpoint does not match its recorded digest")
            state = pickle.loads(checkpoint.read_bytes())
            if state["metadata"] != metadata:
                raise ValueError("Checkpoint provenance changed")
            solver.load_state_dict(state["solver"])
            trace = state["trace"]
            last_event = state.get("last_event", trace[-1]["last_step"])
            if solver.iteration != trace[-1]["iteration"]:
                if (solver.iteration != trace[-1]["iteration"] + 1
                        or last_event.get("observation_type") != "accepted_update"
                        or last_event.get("iteration") != solver.iteration):
                    raise ValueError("Checkpoint cannot reconstruct its missing accepted observation")
                trace.append(observe(problem, solver, caps, last_event))
            elif (last_event.get("observation_type") in ("budget_stop", "converged")
                    and trace[-1]["observation_type"] not in ("budget_stop", "converged")):
                # Terminal diagnostic failure does not advance the iteration;
                # recover its original stopping/KKT details and charged work.
                if last_event.get("iteration") != solver.iteration:
                    raise ValueError("Checkpoint terminal event has the wrong iteration")
                trace.append(observe(problem, solver, caps, last_event))
        else:
            trace = [observe(problem, solver, caps, dict(observation_type="initial"))]
            last_event = trace[-1]["last_step"]
        write(folder / "metadata.json", metadata)

        def save(status, termination=None):
            # A receipt names both the pending and previous complete payload.
            # A crash between receipt publication and rename is recoverable.
            pending = checkpoint.with_suffix(".next.pkl")
            atomic_bytes(pending, pickle.dumps(dict(metadata=metadata,
                solver=solver.state_dict(), trace=trace, last_event=last_event), protocol=5))
            write(folder / "checkpoint.sha256.json", dict(sha256=sha256(pending),
                previous_sha256=sha256(checkpoint) if checkpoint.exists() else None))
            pending.replace(checkpoint)
            write(folder / "trace.json", trace)
            write(folder / "progress.json", dict(status=status, termination=termination,
                iteration=solver.iteration, observations=len(trace), **plain(solver.counters)))

        def finish(kind):
            result = dict(status="completed", termination=kind, metadata=metadata,
                iteration=solver.iteration, observations=len(trace), initial=trace[0], final=trace[-1])
            write(result_file, result)
            return result

        if trace[-1]["observation_type"] in ("budget_stop", "converged"):
            save("completed", trace[-1]["observation_type"])
            return finish(trace[-1]["observation_type"])
        save("running")
        try:
            while True:
                check_abort(root)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    event = plain(solver.step(view_budget=config["view_budget"]))
                last_event = event
                if max(solver.counters["forward_views"], solver.counters["backward_views"]) > config["view_budget"]:
                    raise ValueError("An optimizer call exceeded its directional work budget")
                kind = event["observation_type"]
                if kind == "stalled":
                    raise RuntimeError("Inner solver stalled: " + json.dumps(event))
                if kind not in ("accepted_update", "budget_stop", "converged"):
                    raise ValueError("Unexpected solver event: " + kind)
                row = observe(problem, solver, caps, event)
                if row["iteration"] < trace[-1]["iteration"]:
                    raise ValueError("Accepted iteration counter decreased")
                trace.append(row)
                terminal = kind in ("budget_stop", "converged")
                save("completed" if terminal else "running", kind if terminal else None)
                if terminal:
                    return finish(kind)
        except Exception as exc:
            # Store the solver's latest trial state too; accepted U,V remain intact.
            save("failed", type(exc).__name__)
            write(folder / "failure.json", dict(error=repr(exc), traceback=traceback.format_exc(),
                iteration=solver.iteration, counters=plain(solver.counters)))
            raise
    finally:
        problem.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    result = run(read(args.config), cpu=args.cpu, root=args.root, output=args.output)
    print(json.dumps(dict(status=result["status"], termination=result["termination"],
                         iteration=result["iteration"], final=result["final"])))


if __name__ == "__main__":
    main()
