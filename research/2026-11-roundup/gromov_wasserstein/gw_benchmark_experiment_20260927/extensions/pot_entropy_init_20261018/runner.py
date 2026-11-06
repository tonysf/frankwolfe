"""FP64 full-gradient POT entropic-GW diagnostics with reproducible starts.

The data, loss, exact gradient and certified entropic transport subproblem
are reused from the frozen large-data comparison. Only epsilon and the
initial coupling change. The stopping diagnostic measures consecutive
outer-update L1 norms; it is not a global-optimality certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
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
SHARED = ROOT / "extensions/large_pot_frames_20261016"
spec = importlib.util.spec_from_file_location("_entropy_init_shared_runner", SHARED / "runner.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sys.path.insert(0, str(SHARED))
from checkpoint import CheckpointStore, atomic_json, sha256
from prox import DistributedCertifiedPOTProx

RowShardedGW = shared.RowShardedGW
load_problem = shared.load_problem
pot_step = shared.pot_step
product_log_reference = shared.product_log_reference
INNER_TOL = shared.INNER_TOL
SOURCE_FILES = ("extensions/pot_entropy_init_20261018/runner.py",) + shared.SOURCE_FILES


class SoftDeadlineReached(RuntimeError):
    """Discard the unfinished transport solve and checkpoint its input plan."""


class DeadlineCertifiedPOTProx(DistributedCertifiedPOTProx):
    """Interruptible single-rank wrapper; numerical acceptance stays unchanged."""
    def __init__(self, *args, deadline, stop_requested, **kwargs):
        super().__init__(*args, **kwargs)
        if self.problem.world_size != 1:
            raise ValueError("Deadline-aware entropy diagnostic requires one rank")
        self.deadline, self.stop_requested = deadline, stop_requested

    def _check_deadline(self):
        if self.stop_requested() or time.monotonic() >= self.deadline:
            raise SoftDeadlineReached("Inner solve interrupted before accepting an update")

    def _blocks(self):
        # Covers the positive kernel, its certificate, and log-domain fallback.
        for block in super()._blocks():
            self._check_deadline()
            yield block

    def _scaling_iteration(self, *args, **kwargs):
        self._check_deadline()
        return super()._scaling_iteration(*args, **kwargs)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


@torch.no_grad()
def initialize(problem, initialization, *, seed, mixture_weight=.9):
    """Build a feasible positive start, without labels or geometry matching."""
    if initialization not in ("product", "permutation-mixture"):
        raise ValueError("Unknown initialization")
    if not math.isfinite(mixture_weight) or not 0 < mixture_weight < 1:
        raise ValueError("The permutation mixture weight must lie strictly between zero and one")
    plan = problem.a_local[:, None] * problem.b[None, :]
    record = dict(initialization=initialization, seed=seed,
                  formula="a b^T", uses_labels=False)
    if initialization == "permutation-mixture":
        tolerance = 512 * torch.finfo(torch.float64).eps
        valid = (problem.n == problem.m
                 and bool((problem.a == problem.a[0]).all())
                 and bool((problem.b == problem.b[0]).all())
                 and abs(problem.a[0].item()-1/problem.n) <= tolerance
                 and abs(problem.b[0].item()-1/problem.m) <= tolerance)
        problem._require(valid, "Permutation starts require equal-size uniform marginals")
        # A CPU generator gives the identical permutation on every rank/device.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(problem.n, generator=generator)
        record.update(mixture_weight=mixture_weight,
                      formula="(1-w) a b^T + w Permutation(seed)/n",
                      permutation_generator="torch.randperm with seeded CPU generator",
                      permutation_sha256=hashlib.sha256(permutation.numpy().tobytes()).hexdigest())
        columns = permutation[problem.row_start:problem.row_stop].to(problem.device)
        rows = torch.arange(problem.local_rows, device=problem.device)
        plan.mul_(1-mixture_weight)
        plan[rows, columns] += mixture_weight/problem.n
    problem._validate_plan(plan)
    problem.require_nonnegative(plan)
    residual = problem.marginal_residuals(plan)["both_l1"]
    problem._require(residual <= 512*torch.finfo(torch.float64).eps,
                     "Initialization violates the archived marginals")
    record["marginal_l1"] = residual
    return plan, record


@torch.no_grad()
def outer_update_l1(problem, old, new):
    value = old.new_zeros(())
    for start in range(0, problem.local_rows, problem.exact_block_rows):
        block = slice(start, start+problem.exact_block_rows)
        value.add_((new[block]-old[block]).abs().sum())
    result = problem._reduce(value).item()
    problem._require(math.isfinite(result), "Nonfinite outer update norm")
    return result


def stopping_update(iteration, delta, stable_updates, *, tolerance, minimum, consecutive):
    stable_updates = stable_updates+1 if delta <= tolerance else 0
    return stable_updates, iteration >= minimum and stable_updates >= consecutive


def immutable_config(args):
    return {name: getattr(args, name) for name in (
        "method", "seed", "iterations", "epsilon", "initialization", "init_mixture_weight",
        "dtype", "device", "inner_max", "diagnostic_every", "exact_block_rows",
        "outer_tolerance", "outer_min_iterations", "outer_stable_updates")}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--method", choices=("pot-full",), default="pot-full")
    p.add_argument("--seed", type=int, default=811)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--initialization", choices=("product", "permutation-mixture"), default="product")
    p.add_argument("--init-mixture-weight", type=float, default=.9)
    p.add_argument("--outer-tolerance", type=float, default=1e-6)
    p.add_argument("--outer-min-iterations", type=int, default=50)
    p.add_argument("--outer-stable-updates", type=int, default=10)
    p.add_argument("--chunk-updates", type=int, default=1000000000)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--checkpoint-seconds", type=float, default=1800.)
    p.add_argument("--max-seconds", type=float, default=79200)
    p.add_argument("--epsilon", type=float, default=.001)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--dtype", choices=("float64",), default="float64")
    p.add_argument("--inner-max", type=int, default=100000)
    p.add_argument("--diagnostic-every", type=int, default=25)
    p.add_argument("--exact-block-rows", type=int, default=128)
    p.add_argument("--retain-checkpoint", action="store_true")
    return p


@torch.no_grad()
def _run(args, lock_handles):
    for name in ("iterations", "chunk_updates", "checkpoint_every", "inner_max", "diagnostic_every",
                 "exact_block_rows", "outer_min_iterations", "outer_stable_updates"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_seconds", "epsilon", "checkpoint_seconds", "outer_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive finite")
    if not math.isfinite(args.init_mixture_weight) or not 0 < args.init_mixture_weight < 1:
        raise ValueError("init_mixture_weight must lie strictly between zero and one")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world != 1 or rank != 0:
        raise ValueError("Entropy diagnostic currently requires one process and one GPU")
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
    shared._rank_zero_io(acquire_lock, rank, world)
    previous = shared._rank_zero_io(lambda: json.loads((out/"progress.json").read_text())
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
    if state is None:
        if previous is not None:
            raise ValueError("Prior run exists without a committed checkpoint; refusing silent restart")
        plan, initialization_record = initialize(problem, args.initialization, seed=args.seed,
                                                mixture_weight=args.init_mixture_weight)
        iteration, stable_updates = 0, 0
        trace, inner_trace, outer_trace = [], [], []
        counters = dict(full_gradient_calls=0, total_inner_iterations=0, updates_with_underflow=0,
                        interrupted_inner_attempts=0)
        elapsed_prior, stopping_reason = 0., None
    else:
        plan, initialization_record = state["plan"], state["initialization_record"]
        iteration, stable_updates, trace, inner_trace, outer_trace, counters = (
            state[k] for k in ("iteration", "stable_updates", "trace", "inner_trace", "outer_trace", "counters"))
        stopping_reason = state["stopping_reason"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device)
        elapsed_prior = state["elapsed_seconds"]
        if iteration != checkpoint_record["iteration"] or not 0 <= iteration <= args.iterations:
            raise ValueError("Checkpoint iteration is inconsistent")
    del state
    problem._validate_plan(plan)
    problem.require_nonnegative(plan)
    start_iteration = iteration
    last_checkpoint_time = time.monotonic()
    stop_requested = [False]
    def stop_signal(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGUSR1, stop_signal)
    prox = DeadlineCertifiedPOTProx(problem, gap_tol=INNER_TOL,
                                   certificate_block_rows=args.exact_block_rows,
                                   deadline=started+args.max_seconds, stop_requested=lambda: stop_requested[0])
    log_reference = product_log_reference(problem)

    def receipt(status):
        counters["completed_updates"] = iteration
        record = dict(status=status, updated_at=utc_now(), completed_updates=iteration,
                      method=args.method, seed=args.seed, target_updates=args.iterations,
                      config=immutable_config(args), identity=identity, source_file_sha256=sources,
                      **data, counters=counters, trace=trace, inner_trace=inner_trace, outer_trace=outer_trace,
                      initialization=initialization_record, stopping_reason=stopping_reason,
                      stable_updates=stable_updates,
                      last_outer_update_l1=outer_trace[-1]["update_l1"] if outer_trace else None,
                      stopping_criterion=dict(norm="L1(P_next-P_previous)", tolerance=args.outer_tolerance,
                                              minimum_updates=args.outer_min_iterations,
                                              consecutive_updates=args.outer_stable_updates,
                                              global_optimality_certificate=False),
                      elapsed_seconds=elapsed_prior+time.monotonic()-started,
                      schedule=dict(eta=1/args.epsilon, epsilon=args.epsilon,
                                    inner_tolerance=INNER_TOL, gap_tolerance=INNER_TOL,
                                    scope="Full-gradient POT entropic GW PGD; fixed entropy coefficient"),
                      dimensions=dict(n=problem.n,m=problem.m), world_size=world,
                      slurm_job_id=os.environ.get("SLURM_JOB_ID"), **shared._memory(problem))
        if status == "completed":
            record.update(final_loss=trace[-1]["rounded_loss"], final_raw_loss=trace[-1]["raw_loss"],
                          final_regularized_loss=trace[-1]["rounded_regularized_loss"])
        return record

    def save():
        nonlocal checkpoint_record, last_checkpoint_time
        if source_hashes() != sources:
            raise ValueError("Numerical source changed during a run")
        payload = dict(plan=plan, initialization_record=initialization_record,
                       iteration=iteration, stable_updates=stable_updates, stopping_reason=stopping_reason,
                       trace=trace, inner_trace=inner_trace, outer_trace=outer_trace, counters=counters,
                       torch_rng_state=torch.get_rng_state(),
                       cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                       elapsed_seconds=elapsed_prior+time.monotonic()-started)
        checkpoint_record = store.save(payload, identity=identity, iteration=iteration)
        last_checkpoint_time = time.monotonic()

    def diagnostic():
        record = shared.diagnose(problem, plan, args.epsilon, iteration)
        record.update(outer_update_l1=outer_trace[-1]["update_l1"] if outer_trace else None,
                      stable_updates=stable_updates)
        return record

    try:
        if not trace:
            trace.append(diagnostic())
        if checkpoint_record is None:
            save()
        record = receipt("running")
        if rank == 0:
            atomic_json(out/"progress.json", record)
        while iteration < args.iterations and stopping_reason is None:
            gradient = problem.gradient(plan)
            try:
                new_plan, info = pot_step(problem, log_reference, gradient, prox,
                                          epsilon=args.epsilon, inner_max=args.inner_max)
            except SoftDeadlineReached:
                # The live plan and numerical iteration state are unchanged.
                # Recompute this deterministic gradient/subproblem after resume.
                del gradient
                counters["interrupted_inner_attempts"] += 1
                save()
                break
            del gradient
            counters["full_gradient_calls"] += 1
            delta = outer_update_l1(problem, plan, new_plan)
            plan = new_plan
            del new_plan
            iteration += 1
            info["iteration"] = iteration-1  # Preserve the reused runner's update indexing.
            inner_trace.append(info)
            counters["total_inner_iterations"] += info["inner_iterations"]
            counters["updates_with_underflow"] += int(info.get("underflow_entries", 0) > 0)
            stable_updates, converged = stopping_update(
                iteration, delta, stable_updates, tolerance=args.outer_tolerance,
                minimum=args.outer_min_iterations, consecutive=args.outer_stable_updates)
            outer_trace.append(dict(iteration=iteration, update_l1=delta, stable_updates=stable_updates))
            if converged:
                stopping_reason = "outer_update_l1_tolerance"
            elif iteration == args.iterations:
                stopping_reason = "iteration_budget"
            if (iteration <= 10 or iteration in (15,20) or iteration % args.diagnostic_every == 0
                    or stopping_reason is not None):
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
            if bool(due_flag.item()) or stop or stopping_reason is not None:
                save()
                record = receipt("completed" if stopping_reason is not None else "checkpointed" if stop else "running")
                record["checkpoint"] = checkpoint_record
                if rank == 0:
                    atomic_json(out/"progress.json", record)
                if stop:
                    break
        status = "completed" if stopping_reason is not None else "checkpointed"
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
        numerical_failure = isinstance(exc, shared.ProximalSolveError)
        failure = dict(status="failed", updated_at=utc_now(), completed_updates=iteration,
                       error=f"{type(exc).__name__}: {exc}",
                       failure_kind="inner_certificate_not_met" if numerical_failure else "runtime_error",
                       identity=identity, config=immutable_config(args), source_file_sha256=sources,
                       traceback=traceback.format_exc(), certificate=getattr(exc,"certificate",None),
                       trace=trace, inner_trace=inner_trace, outer_trace=outer_trace, counters=counters,
                       initialization=initialization_record, stable_updates=stable_updates,
                       checkpoint=checkpoint_record, checkpoint_removed=False)
        if rank == 0:
            atomic_json(out/"failure.json", failure)
        if numerical_failure:
            # This is a terminal unsuccessful trial, not a resumable timeout.
            # Preserve its certificate, accepted trajectory and checkpoint
            # manifest before deleting only this trial's bulky tensor state.
            store.remove_completed_state()
            failure["checkpoint_removed"] = True
            if rank == 0:
                atomic_json(out/"failure.json", failure)
        raise


def run(args):
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
