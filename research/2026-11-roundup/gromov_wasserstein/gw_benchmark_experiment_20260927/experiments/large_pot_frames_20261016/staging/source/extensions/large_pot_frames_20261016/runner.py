"""FP64 row-sharded POT-step controls and fixed-exponent stochastic FraMES.

All methods use the original distance matrices and the same product start.
The POT update minimizes <G,Q> + epsilon*h(Q) on the transport polytope.
FraMES retains gamma/rho/mu exponents 3/4, 1/2, 1/4. Only prefactors vary.
Every stochastic update draws fresh iid coupling pairs; the initial gradient
is reused as the initial momentum. No rounding enters either optimizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
for folder in ("shareseq_25000_20260930", "published_stochastic_20261002", "pair_sampling_20261011"):
    sys.path.insert(0, str(ROOT / "extensions" / folder))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sharded_gw import RowShardedGW, row_partition
from oracle import pair_gradient, sample_pairs
from smd import ProximalSolveError
from checkpoint import CheckpointStore, atomic_json, sha256

SOURCE_FILES = (
    "extensions/large_pot_frames_20261016/runner.py",
    "extensions/large_pot_frames_20261016/checkpoint.py",
    "extensions/large_pot_frames_20261016/prox.py",
    "extensions/shareseq_25000_20260930/sharded_gw.py",
    "extensions/published_stochastic_20261002/certified_prox.py",
    "extensions/published_stochastic_20261002/smd.py",
    "extensions/pair_sampling_20261011/oracle.py",
)
INNER_TOL = 1e-7


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def frames_schedule(iteration, m, gamma0, rho0, mu_scale):
    if (iteration < 0 or m < 1 or not 0 < gamma0 <= 1 or not 0 < rho0 <= 1
            or not math.isfinite(mu_scale) or mu_scale <= 0):
        raise ValueError("Invalid FraMES iteration, dimension or prefactors")
    k = iteration + 1
    return gamma0 / k ** .75, rho0 / k ** .5, mu_scale / m / k ** .25


def product_log_reference(problem):
    # Uniform archived marginals are common; expand keeps this constant
    # reference a scalar view rather than a dense extra multi-GB allocation.
    a, b = problem.a_local, problem.b
    if bool((a == a[0]).all() and (b == b[0]).all()):
        return (a[0].log() + b[0].log()).expand(problem.local_rows, problem.m)
    return a.log()[:, None] + b.log()[None, :]


@torch.no_grad()
def pot_step(problem, log_reference, gradient, prox, *, epsilon=.001, inner_max=100000):
    problem._require(math.isfinite(epsilon) and epsilon > 0, "Positive finite epsilon required")
    plan, log_plan, info = prox(log_reference, gradient, eta=1. / epsilon,
                                tol=INNER_TOL, max_iter=inner_max, check_every=10)
    problem._validate_plan(plan)
    problem.require_nonnegative(plan)
    problem.require_finite(log_plan)
    residual = float(prox.marginal_residuals(plan)["both_l1"])
    required = ("primal_objective", "dual_objective", "primal_dual_gap",
                "normalized_primal_dual_gap", "primal_witness_marginal_l1",
                "stable_primal_dual_gap", "gap_roundoff_tolerance")
    try:
        primal, dual, gap, normalized, witness, stable, roundoff = (float(info[key]) for key in required)
        scale = max(1., abs(primal), abs(dual))
        upper = (max(gap, stable) + roundoff) / scale
        valid = (all(math.isfinite(x) for x in (residual, primal, dual, gap, normalized, witness, stable, roundoff))
                 and info.get("converged") is True and info.get("proximal_optimality_certified") is True
                 and 0 <= residual <= INNER_TOL and 0 <= witness <= 1e-12
                 and 0 <= roundoff / scale <= INNER_TOL
                 and min(gap, stable) >= -roundoff and abs(gap - stable) <= roundoff
                 and abs(primal - dual - gap) <= 1e-12 * scale
                 and abs(normalized - gap / scale) <= 1e-12 and upper <= INNER_TOL)
    except (KeyError, TypeError, ValueError):
        valid, upper = False, math.inf
    flag = torch.tensor(int(valid), device=problem.device)
    problem._reduce(flag, dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise ProximalSolveError("POT transport subproblem failed actual-feasibility or guarded-gap checks", certificate=info)
    info = dict(info, eta=1. / epsilon, marginal_l1=residual,
                normalized_gap_upper_bound=upper, normalized_gap_tolerance=INNER_TOL,
                fixed_reference="product_marginals")
    del log_plan
    return plan, info


def _broadcast_record(record, rank, world):
    if world > 1:
        values = [record]
        dist.broadcast_object_list(values, src=0)
        return values[0]
    return record


def _rank_zero_io(action, rank, world):
    result, error = None, None
    if rank == 0:
        try:
            result = action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    result, error = _broadcast_record((result,error), rank, world)
    if error:
        raise RuntimeError(error)
    return result


def load_problem(instance, *, device, rank=0, world_size=1, exact_block_rows=128):
    instance = Path(instance)
    manifest_path = instance / "manifest.json"
    manifest = _rank_zero_io(lambda: json.loads(manifest_path.read_text()), rank, world_size)
    expected = manifest.get("array_files_sha256", {})
    record = None
    if rank == 0:
        try:
            hashes = {name + ".npy": sha256(instance / (name + ".npy")) for name in ("A", "B", "a", "b")}
            if set(hashes) - expected.keys() or any(expected[k] != v for k, v in hashes.items()):
                raise ValueError("Manifest does not match original array hashes")
            record = {"hashes": hashes, "manifest_hash": sha256(manifest_path)}
        except Exception as exc:
            record = {"error": str(exc)}
    record = _broadcast_record(record, rank, world_size)
    if "error" in record:
        raise ValueError(record["error"])
    arrays = {name: np.load(instance / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ("A", "B", "a", "b")}
    n, m = len(arrays["a"]), len(arrays["b"])
    start, stop = row_partition(n, world_size, rank)
    if stop <= start:
        raise ValueError("Every process must own at least one row")
    for name in ("a", "b"):
        if arrays[name].dtype != np.float64:
            raise ValueError("Archived marginals must already be FP64; no silent renormalization")
    for name, array in arrays.items():
        if array.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"Unsupported archived dtype for {name}")
    dtype_record = {name: str(array.dtype) for name, array in arrays.items()}
    # Copy only this A shard, and B once. Promotion from FP32 is lossless.
    def tensor(value):
        return torch.from_numpy(np.array(value, copy=True)).to(device=device, dtype=torch.float64)
    A, B = tensor(arrays["A"][start:stop]), tensor(arrays["B"])
    a, b = tensor(arrays["a"]), tensor(arrays["b"])
    tolerance = 512 * torch.finfo(torch.float64).eps
    if max(abs(a.sum().item()-1), abs(b.sum().item()-1), abs(a.sum().item()-b.sum().item())) > tolerance:
        raise ValueError("Archived marginal masses must be equal and unit to FP64 audit tolerance")
    problem = RowShardedGW(A, B, a, b, exact_block_rows=exact_block_rows)
    if (n, m) != (int(manifest["n"]), int(manifest["m"])):
        raise ValueError("Manifest dimensions disagree")
    return problem, dict(array_files_sha256=record["hashes"], manifest_sha256=record["manifest_hash"],
                         original_array_dtypes=dtype_record, arithmetic_dtype="float64",
                         input_transform="lossless float32 distance promotion; unchanged archived float64 marginals")


@torch.no_grad()
def _entropy(problem, plan):
    value = plan.new_zeros(())
    for start in range(0, problem.local_rows, problem.exact_block_rows):
        block = plan[start:start+problem.exact_block_rows]
        value.add_((torch.xlogy(block, block)-block).sum())
    return problem._reduce(value).item()


@torch.no_grad()
def diagnose(problem, plan, epsilon, iteration, *, momentum=None, prefactors=None):
    # Sequential full evaluations; no dense coupling gather or saved copies.
    raw_loss = problem.loss(plan).item()
    raw_entropy = _entropy(problem, plan)
    raw_residuals = problem.marginal_residuals(plan)
    rounded = problem.round_transport(plan)
    rounded_loss = problem.loss(rounded).item()
    rounded_entropy = _entropy(problem, rounded)
    rounded_residuals = problem.marginal_residuals(rounded)
    repair = plan.new_zeros(())
    for start in range(0, problem.local_rows, problem.exact_block_rows):
        block = slice(start, start+problem.exact_block_rows)
        repair.add_((rounded[block]-plan[block]).abs().sum())
    repair_l1 = problem._reduce(repair).item()
    del rounded
    problem._require(rounded_residuals["both_l1"] <= 512*torch.finfo(torch.float64).eps,
                     "Rounded diagnostic witness exceeds FP64 feasibility tolerance")
    values = (raw_loss, rounded_loss, raw_entropy, rounded_entropy)
    problem._require(all(math.isfinite(x) for x in values), "Nonfinite diagnostic objective")
    result = dict(iteration=int(iteration), raw_loss=raw_loss, rounded_loss=rounded_loss,
                raw_entropy=raw_entropy, rounded_entropy=rounded_entropy,
                raw_regularized_loss=raw_loss+epsilon*raw_entropy,
                rounded_regularized_loss=rounded_loss+epsilon*rounded_entropy,
                raw_residuals=raw_residuals, rounded_residuals=rounded_residuals,
                rounding_repair_l1=repair_l1)
    if prefactors is not None:
        gamma, rho, mu = frames_schedule(max(0, iteration-1), problem.m, *prefactors)
        result.update(gamma=gamma, rho=rho, mu=mu, schedule_iteration=max(0,iteration-1))
        if momentum is not None:
            error = problem._reduce(plan.sum(0))-problem.b
            penalty = error/mu
            centered_squares, changed_mass = plan.new_zeros(()), plan.new_zeros(())
            for start in range(0, problem.local_rows, problem.exact_block_rows):
                block = slice(start, start+problem.exact_block_rows)
                d = momentum[block]
                centered_squares.add_((d-d.mean(1,keepdim=True)).square().sum())
                changed = d.argmin(1) != (d+penalty[None,:]).argmin(1)
                changed_mass.add_((changed*problem.a_local[block]).sum())
            dnorm = problem._reduce(centered_squares).sqrt().item()
            pnorm = (problem.n*penalty.square().sum()).sqrt().item()
            result.update(penalty_frobenius_norm=pnorm,
                          row_centered_momentum_frobenius_norm=dnorm,
                          penalty_to_momentum_norm_ratio=pnorm/dnorm if dnorm>0 else None,
                          lmo_changed_source_mass=problem._reduce(changed_mass).item(),
                          lmo_diagnostic_timing="current plan and stored momentum; previous update schedule")
    return result


def immutable_config(args):
    return {name: getattr(args, name) for name in ("method", "seed", "iterations", "epsilon", "pairs",
            "gamma0", "rho0", "mu_scale", "dtype", "device", "inner_max", "diagnostic_every", "exact_block_rows")}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--method", choices=("pot-full", "pot-pair", "frames"), required=True)
    p.add_argument("--seed", type=int, default=811)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--chunk-updates", type=int, default=1000000000)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--checkpoint-seconds", type=float, default=1800.)
    p.add_argument("--max-seconds", type=float, default=79200)
    p.add_argument("--epsilon", type=float, default=.001)
    p.add_argument("--pairs", type=int, default=1024)
    p.add_argument("--gamma0", type=float, default=1.)
    p.add_argument("--rho0", type=float, default=1.)
    p.add_argument("--mu-scale", type=float, default=10.)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("float64",), default="float64")
    p.add_argument("--inner-max", type=int, default=100000)
    p.add_argument("--diagnostic-every", type=int, default=500)
    p.add_argument("--exact-block-rows", type=int, default=128)
    p.add_argument("--retain-checkpoint", action="store_true")
    return p


def _memory(problem):
    if problem.device.type == "cuda":
        reserved = torch.cuda.max_memory_reserved(problem.device)
        total = torch.cuda.get_device_properties(problem.device).total_memory
    else:
        reserved, total = 0, 0
    values = torch.tensor([reserved, total], dtype=torch.int64, device=problem.device)
    if problem.world_size > 1:
        gathered = [torch.empty_like(values) for _ in range(problem.world_size)]
        dist.all_gather(gathered, values)
    else:
        gathered = [values]
    records = [v.tolist() for v in gathered]
    return dict(max_memory_reserved_bytes_per_rank=[v[0] for v in records],
                memory_total_bytes_per_rank=[v[1] for v in records],
                max_memory_reserved_fraction=max((a/b if b else 0) for a,b in records))


@torch.no_grad()
def _run(args, lock_handles):
    for name in ("iterations", "chunk_updates", "checkpoint_every", "pairs", "inner_max", "diagnostic_every", "exact_block_rows"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if (not math.isfinite(args.max_seconds) or args.max_seconds <= 0
            or not math.isfinite(args.epsilon) or args.epsilon <= 0
            or not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds <= 0):
        raise ValueError("Time limits and epsilon must be positive finite")
    frames_schedule(0, 1, args.gamma0, args.rho0, args.mu_scale)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    out = Path(args.out).resolve()
    def acquire_lock():
        out.mkdir(parents=True, exist_ok=True)
        handle = (out/"run.lock").open("a+")
        lock_handles.append(handle)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    _rank_zero_io(acquire_lock, rank, world)
    previous = _rank_zero_io(lambda: json.loads((out/"progress.json").read_text())
                            if (out/"progress.json").exists() else None, rank, world)
    if previous is not None and previous.get("status") == "completed":
        raise ValueError("Completed output already exists; refusing duplicate run")
    started = time.monotonic()
    problem, data = load_problem(args.instance, device=device, rank=rank, world_size=world,
                                  exact_block_rows=args.exact_block_rows)
    sources = source_hashes()
    identity = dict(config=immutable_config(args), source_file_sha256=sources,
                    array_files_sha256=data["array_files_sha256"], manifest_sha256=data["manifest_sha256"],
                    world_size=world, n=problem.n, m=problem.m,
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    torch_cuda_version=torch.version.cuda, num_threads=torch.get_num_threads(),
                    cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    device_model=torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
                    cuda_capability=list(torch.cuda.get_device_capability(device)) if device.type=="cuda" else None)
    store = CheckpointStore(args.checkpoint_dir or out/"checkpoints", rank=rank, world_size=world)
    state, checkpoint_record = store.load(identity=identity, map_location=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    if state is None:
        if previous is not None:
            raise ValueError("Prior run exists without a committed checkpoint; refusing silent restart")
        plan = problem.a_local[:, None] * problem.b[None, :]
        momentum, iteration, trace, inner_trace = None, 0, [], []
        counters = dict(full_gradient_calls=0, stochastic_gradient_calls=0, pair_samples=0,
                        total_inner_iterations=0, updates_with_underflow=0)
        elapsed_prior = 0.
    else:
        plan, momentum = state["plan"], state["momentum"]
        iteration, trace, inner_trace, counters = (state[k] for k in ("iteration", "trace", "inner_trace", "counters"))
        generator.set_state(state["generator_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device)
        elapsed_prior = state["elapsed_seconds"]
        if iteration != checkpoint_record["iteration"] or not 0 <= iteration <= args.iterations:
            raise ValueError("Checkpoint iteration is inconsistent")
    # Loading maps the entire state dictionary to this rank's device. Drop
    # that container now so its old plan cannot stay alive after an update.
    del state
    problem._validate_plan(plan)
    if momentum is not None:
        problem._validate_plan(momentum)
    start_iteration = iteration
    last_checkpoint_time = time.monotonic()
    stop_requested = [False]
    def stop_signal(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGUSR1, stop_signal)
    if args.method.startswith("pot-"):
        from prox import DistributedCertifiedPOTProx
        prox = DistributedCertifiedPOTProx(problem, gap_tol=INNER_TOL,
                                          certificate_block_rows=args.exact_block_rows)
        log_reference = product_log_reference(problem)
    else:
        prox, log_reference = None, None

    def receipt(status):
        counters["completed_updates"] = iteration
        record = dict(status=status, updated_at=utc_now(), completed_updates=iteration,
                      method=args.method, seed=args.seed,
                      target_updates=args.iterations, config=immutable_config(args), identity=identity,
                      source_file_sha256=sources, **data, counters=counters, trace=trace,
                      inner_trace=inner_trace, elapsed_seconds=elapsed_prior+time.monotonic()-started,
                      schedule=dict(gamma_exponent=.75, rho_exponent=.5, mu_exponent=.25,
                                    eta=1/args.epsilon, inner_tolerance=INNER_TOL, gap_tolerance=INNER_TOL,
                                    scope="POT-step empirical control; FraMES theoretical exponents fixed"),
                      dimensions=dict(n=problem.n,m=problem.m), world_size=world,
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"), **_memory(problem))
        if status == "completed":
            record["final_loss"] = trace[-1]["rounded_loss"]
            record["final_raw_loss"] = trace[-1]["raw_loss"]
            record["final_regularized_loss"] = trace[-1]["rounded_regularized_loss"]
        return record

    def save():
        nonlocal checkpoint_record, last_checkpoint_time
        if source_hashes() != sources:
            raise ValueError("Numerical source changed during a run")
        payload = dict(plan=plan, momentum=momentum, iteration=iteration, trace=trace,
                       inner_trace=inner_trace, counters=counters,
                       generator_state=generator.get_state(), torch_rng_state=torch.get_rng_state(),
                       cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                       elapsed_seconds=elapsed_prior+time.monotonic()-started)
        checkpoint_record = store.save(payload, identity=identity, iteration=iteration)
        last_checkpoint_time = time.monotonic()

    try:
        def diagnostic():
            return diagnose(problem, plan, args.epsilon, iteration, momentum=momentum,
                            prefactors=(args.gamma0,args.rho0,args.mu_scale) if args.method=="frames" else None)
        if not trace:
            trace.append(diagnostic())
        if checkpoint_record is None:
            save()
        record = receipt("running")
        if rank == 0:
            atomic_json(out/"progress.json", record)
        while iteration < args.iterations:
            if args.method == "pot-full":
                gradient = problem.gradient(plan)
                counters["full_gradient_calls"] += 1
            else:
                sampled = sample_pairs(problem, plan, args.pairs, generator=generator)
                gradient = pair_gradient(problem, sampled)
                del sampled
                counters["stochastic_gradient_calls"] += 1
                counters["pair_samples"] += args.pairs
            if args.method == "frames":
                gamma, rho, mu = frames_schedule(iteration, problem.m, args.gamma0, args.rho0, args.mu_scale)
                if momentum is None:
                    momentum = gradient
                else:
                    momentum.mul_(1-rho).add_(gradient, alpha=rho)
                del gradient
                plan = problem.fw_step(plan, momentum, gamma=gamma, mu=mu)
            else:
                plan, info = pot_step(problem, log_reference, gradient, prox,
                                       epsilon=args.epsilon, inner_max=args.inner_max)
                del gradient
                info["iteration"] = iteration
                inner_trace.append(info)
                counters["total_inner_iterations"] += info["inner_iterations"]
                counters["updates_with_underflow"] += int(info.get("underflow_entries", 0) > 0)
            iteration += 1
            if iteration in (1,2,5,10,20,50,100,200) or iteration % args.diagnostic_every == 0 or iteration == args.iterations:
                trace.append(diagnostic())
            stop = (stop_requested[0] or time.monotonic()-started >= args.max_seconds
                    or iteration-start_iteration >= args.chunk_updates)
            flag = torch.tensor(int(stop), device=device)
            problem._reduce(flag, dist.ReduceOp.MAX)
            stop = bool(flag.item())
            checkpoint_due = (iteration % args.checkpoint_every == 0
                              or time.monotonic()-last_checkpoint_time >= args.checkpoint_seconds)
            due_flag = torch.tensor(int(checkpoint_due), device=device)
            problem._reduce(due_flag, dist.ReduceOp.MAX)
            if bool(due_flag.item()) or stop or iteration == args.iterations:
                save()
                record = receipt("completed" if iteration == args.iterations else "checkpointed" if stop else "running")
                record["checkpoint"] = checkpoint_record
                if rank == 0:
                    atomic_json(out/"progress.json", record)
                if stop:
                    break
        status = "completed" if iteration == args.iterations else "checkpointed"
        record = receipt(status)
        record["checkpoint"] = checkpoint_record
        record["retained_checkpoint"] = args.retain_checkpoint or status != "completed"
        if rank == 0:
            atomic_json(out/"metadata.json", record)
            atomic_json(out/"progress.json", record)
        if status == "completed" and not args.retain_checkpoint:
            store.remove_completed_state()
        return record
    except Exception as exc:
        if rank == 0:
            atomic_json(out/"failure.json", dict(status="failed", updated_at=utc_now(),
                completed_updates=iteration, error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(), certificate=getattr(exc,"certificate",None)))
        raise


def run(args):
    # Keep the lock across initialization, updates, checkpoint transactions
    # and final receipt publication. Release also on validation exceptions.
    lock_handles = []
    try:
        return _run(args, lock_handles)
    finally:
        for handle in lock_handles:
            handle.close()


def main():
    run(parser().parse_args())
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
