"""Resumable FP64 factorization runs; optimizer and diagnostics are timed separately."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import pickle
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from core import FactorizationProblem
from setup_problem import initialize

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_pickle(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        pickle.dump(value, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / n for n in ("core.py", "baseline.py", "setup_problem.py", "run.py")]}


class SFrames:
    def __init__(self, problem, U, V, config):
        if not (0 < config["gamma0"] <= 1 and 0 < config["rho0"] <= 1 and config["mu0"] > 0):
            raise ValueError("SFRAMES requires gamma0,rho0 in (0,1] and mu0>0")
        if not 1 <= config["batch"] <= problem.n:
            raise ValueError("Image batch must be between one and n_images")
        self.problem = problem
        self.U, self.V = U.copy(), V.copy()
        self.mU, self.mV = None, None
        self.k = 0
        self.config = config
        self.rng = np.random.default_rng(config["seed"] + 100000)

    def step(self):
        c, p, k = self.config, self.problem, self.k
        cols = self.rng.choice(p.n, size=c["batch"], replace=False)
        qU, qV = p.sampled_grad_columns(self.U, self.V, cols)
        rho = c["rho0"] / (k + 1)**0.5
        if self.mU is None:
            self.mU, self.mV = qU, qV
        else:
            self.mU *= 1 - rho
            self.mV *= 1 - rho
            self.mU += rho * qU
            self.mV += rho * qV
        mu = c["mu0"] / (k + 1)**0.25
        gU, gV = p.smoothed_gradient(self.U, self.V, mu, gradient=(self.mU, self.mV))
        sU, sV, lmo_certificate = p.lmo_C(gU, gV, return_certificate=True)
        gamma = c["gamma0"] / (k + 1)**0.75
        self.U += gamma * (sU - self.U)
        self.V += gamma * (sV - self.V)
        self.k += 1
        return {"component_gradient_calls": c["batch"], "sampled_images": c["batch"],
                "gamma": gamma, "rho": rho, "mu": mu, "prox_iterations": 0,
                "lmo_support_gap_bound": lmo_certificate["support_gap_bound"]}

    def state_dict(self):
        return {"U": self.U, "V": self.V, "mU": self.mU, "mV": self.mV,
                "k": self.k, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        for name in ("U", "V", "mU", "mV", "k"):
            setattr(self, name, state[name])
        self.rng.bit_generator.state = state["rng"]


def make_solver(problem, U, V, config):
    if config["method"] == "sframes":
        return SFrames(problem, U, V, config)
    from baseline import SpringSARAH
    return SpringSARAH(
        U, V, full_gradient=problem.full_grad,
        batch_gradient=problem.sampled_grad_columns,
        sample_indices=lambda rng, b: rng.choice(problem.n, b, replace=False),
        n_components=problem.n, radius_u=problem.RU, radius_v=problem.RV,
        l1_radius=problem.l1_radius,
        component_lipschitz=problem.certificates()["image_L_bound"],
        block_lipschitz_u=problem.RV**2 / problem.m,
        block_lipschitz_v=problem.RU**2 / problem.m,
        batch_size=config["batch"], mode=config.get("baseline_mode", "theorem"),
        seed=config["seed"] + 100000, local_curvature_seed=config.get("local_curvature_seed"),
    )


def load_problem(config):
    with np.load(ROOT / config["data_path"], allow_pickle=False) as data:
        A = np.array(data["A"], dtype=np.float64, order="C")
    return FactorizationProblem(A, config["rank"], config["RU"], config["RV"], config["l1_radius"])


def diagnostic(problem, solver, config, k, counters, last_step):
    U, V = solver.U, solver.V
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(V))):
        raise FloatingPointError("Nonfinite iterate")
    feasible_U, feasible_V = problem.repair(U, V)
    raw_loss = problem.objective(U, V)
    feasible_loss = problem.objective(feasible_U, feasible_V)
    row = {"iteration": k, "raw_loss": raw_loss, "feasible_loss": feasible_loss,
           "relative_reconstruction_error": float(np.sqrt(2 * feasible_loss / config["mean_data_square"])),
           "raw_relative_reconstruction_error": float(np.sqrt(2 * raw_loss / config["mean_data_square"])),
           "repair_distance": float(np.sqrt(np.sum((U-feasible_U)**2) + np.sum((V-feasible_V)**2))),
           "rank_r_lower_bound": config["rank_r_loss_lower_bound"],
           "gap_to_unconstrained_rank_r_lower_bound": feasible_loss - config["rank_r_loss_lower_bound"],
           **problem.feasibility(U, V), **counters, "utc": now()}
    if config["method"] == "sframes":
        mu = config["mu0"] / (k + 1)**0.25
        row["true_smoothed_fw_gap"] = problem.smoothed_fw_gap(U, V, mu)
        row["mu"] = mu
    for name in ("prox_iterations", "prox_max_distance_bound", "prox_max_dual_gap", "step_u", "step_v"):
        if name in last_step:
            row[name] = last_step[name]
    return row


def run(config, output, cpu=None, stop_at=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    hashes = source_hashes()
    data_hash = hashlib.sha256((ROOT / config["data_path"]).read_bytes()).hexdigest()
    problem = load_problem(config)
    U, V = initialize(problem, config["seed"])
    solver = make_solver(problem, U, V, config)
    counters = dict(optimizer_seconds=0.0, diagnostic_seconds=0.0, checkpoint_seconds=0.0,
                    component_gradient_calls=0, sampled_images=0, cumulative_prox_iterations=0,
                    max_lmo_support_gap_bound=0.0, max_prox_distance_bound=0.0,
                    prox_numerical_floor_count=0)
    trace, k, last_step = [], 0, {}
    checkpoint_path = output / "checkpoint.pkl"
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            saved = pickle.load(stream)
        assert saved["config"] == config, "Refusing mismatched checkpoint configuration"
        assert saved["source_sha256"] == hashes, "Refusing checkpoint from a different implementation"
        assert saved["data_sha256"] == data_hash, "Refusing checkpoint for different data"
        solver.load_state_dict(saved["solver"])
        counters, trace, k = saved["counters"], saved["trace"], saved["iteration"]
        last_step = saved.get("last_step", {})
    stopped = [False]
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    metadata = dict(config=config, source_sha256=hashes, data_sha256=data_hash,
                    cpu=cpu, pid=os.getpid(), started_utc=now(), precision="float64",
                    diagnostics="Repaired feasible loss and exact gradient gaps at probes only; probes are not the theorem's all-iterate average.",
                    oracle="Uniform image-column finite sum, batch sampled without replacement, fresh between iterations.",
                    feasible_repair="Diagnostic only; never fed back into the optimizer.",
                    timing="Optimizer includes gradients, variance reduction, curvature, projection and updates. Diagnostics, initialization and checkpoint IO excluded.")
    atomic_json(output / "metadata.json", metadata)

    def save(status):
        t = time.perf_counter()
        atomic_pickle(checkpoint_path, dict(config=config, source_sha256=hashes, data_sha256=data_hash,
            solver=solver.state_dict(), counters=counters, trace=trace, iteration=k, last_step=last_step))
        counters["checkpoint_seconds"] += time.perf_counter()-t
        atomic_json(output / "progress.json", {"status": status, "iteration": k,
            "target_iterations": config["iterations"], "updated_utc": now(), **counters,
            "latest_diagnostic": trace[-1] if trace else None})
        atomic_json(output / "trace.json", trace)

    def probe():
        t = time.perf_counter()
        row = diagnostic(problem, solver, config, k, counters, last_step)
        counters["diagnostic_seconds"] += time.perf_counter()-t
        row["diagnostic_seconds"] = counters["diagnostic_seconds"]
        trace.append(row)

    last_checkpoint = time.perf_counter()
    try:
        if not trace or trace[-1]["iteration"] != k:
            probe()
        save("running")
        while k < config["iterations"]:
            if stopped[0] or (stop_at is not None and k >= stop_at):
                save("checkpointed")
                return dict(status="checkpointed", output=str(output), iteration=k)
            t = time.perf_counter()
            last_step = solver.step()
            counters["optimizer_seconds"] += time.perf_counter()-t
            if config["method"] != "sframes":
                last_step["component_gradient_calls"] = last_step["step_oracle_calls"]
                last_step["sampled_images"] = sum(problem.n if last_step["refresh_"+block]
                                                 else config["batch"] for block in ("u", "v"))
                certs = [last_step["projection_u"], last_step["projection_v"]]
                last_step["prox_iterations"] = sum(c["iterations"] for c in certs)
                last_step["prox_max_distance_bound"] = max(c["attained_distance_bound"] for c in certs)
                last_step["prox_max_dual_gap"] = max(c["duality_gap"] for c in certs)
                counters["max_prox_distance_bound"] = max(counters["max_prox_distance_bound"], last_step["prox_max_distance_bound"])
                counters["prox_numerical_floor_count"] += sum(c["status"] == "numerical_floor" for c in certs)
            else:
                counters["max_lmo_support_gap_bound"] = max(counters["max_lmo_support_gap_bound"], last_step["lmo_support_gap_bound"])
            k += 1
            counters["component_gradient_calls"] += int(last_step.get("component_gradient_calls", 0))
            counters["sampled_images"] += int(last_step.get("sampled_images", 0))
            counters["cumulative_prox_iterations"] += int(last_step.get("prox_iterations", 0))
            if k in (1, 2, 5, 10, 20, 50) or k % config.get("probe_every", 100) == 0 or k == config["iterations"]:
                probe()
            if k % config.get("checkpoint_every", 250) == 0 or time.perf_counter()-last_checkpoint > 120:
                save("running")
                last_checkpoint = time.perf_counter()
        save("completed")
        result = dict(metadata, status="completed", iteration=k, finished_utc=now(),
                      final=trace[-1], **counters)
        atomic_json(output / "result.json", result)
        return result
    except Exception as exc:
        # A failed block can already have consumed random draws. Retain the
        # last successful checkpoint so resume replays the entire failed step.
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        atomic_json(output / "trace.json", trace)
        atomic_json(progress_path, dict(progress, status="failed", failed_at_iteration=k,
                                        updated_utc=now()))
        atomic_json(output / "failure.json", {"iteration": k, "error": repr(exc), "utc": now()})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.config.read_text()), args.output, args.cpu), indent=2))
