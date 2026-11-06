"""FP64 CPU stochastic and full-gradient POT-scale entropic GW updates.

Each outer update solves <G_k,Q> + epsilon*h(Q) over U(a,b), where G_k
is either the full GW gradient or a fresh unbiased coupling-pair estimate.
This is the POT PGD update, equivalently entropic mirror descent on
f_GW+epsilon*h at fixed eta=1/epsilon. This experiment does not claim the
conservative stochastic theorem permits that step size.

A fixed product reference R=a*b is used in the numerical KL subproblem:
<G_k,Q> + epsilon*KL(Q||R). On U(a,b) its difference from the target
subproblem is a constant, since log(R_ij)=log(a_i)+log(b_j). Its unit
mass avoids an enormous constant inflating the normalized gap scale.
The current P only enters the GW gradient oracle; log(P) is never needed.
Physical underflow entries are allowed, counted, and given 0*log(0)=0
in diagnostics. No momentum, tuning, or adaptive horizon is used here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

# Set before importing numerical libraries; the process owns one CPU thread.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _directory in ("shareseq_25000_20260930", "published_stochastic_20261002",
                   "pair_sampling_20261011"):
    sys.path.insert(0, str(ROOT / "extensions" / _directory))
from sharded_gw import RowShardedGW
from certified_prox import CertifiedKLProx
import smd
from oracle import pair_gradient, sample_pairs
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_prox import FastCertifiedKLProx

INNER_TOL = 1e-7
INNER_GAP_TOL = 1e-7
INNER_MAX = 10000
CHECK_EVERY = 10
DEFAULT_CHECKPOINTS = "0,1,2,5,10,20,50,100,200,400,600,800,1000,1250,1500,1750,2000"
SOURCE_FILES = (
    "extensions/scot_pot_step_20261015/run.py",
    "extensions/scot_pot_step_20261015/fast_prox.py",
    "extensions/shareseq_25000_20260930/sharded_gw.py",
    "extensions/published_stochastic_20261002/certified_prox.py",
    "extensions/published_stochastic_20261002/smd.py",
    "extensions/pair_sampling_20261011/oracle.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def atomic_json(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def entropy(plan):
    """h(P), including the continuous 0*log(0)=0 convention."""
    return (torch.xlogy(plan, plan) - plan).sum()


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


@torch.no_grad()
def pot_step(log_reference, fresh_gradient, *, iteration, epsilon, prox,
             inner_max=INNER_MAX, inner_tol=INNER_TOL,
             gap_tol=INNER_GAP_TOL, check_every=CHECK_EVERY):
    """One fixed-POT-scale update with the audited numerical acceptance checks.

    The reference must be log(a[:,None]*b[None,:]), fixed across updates.
    Consequently this has the same minimizer as POT's entropic OT solve.
    Unlike a current-plan KL update, zero entries in the current P cannot
    trap later iterates: P is absent from this subproblem except via G.
    Numerical acceptance checks reproduce the audited smd_step checks, but
    eta is explicitly 1/epsilon and no theorem step helper is called.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    eta = 1. / epsilon
    if not math.isfinite(eta):
        raise ValueError("POT step size must be finite")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    tolerance, gap_tolerance = float(inner_tol), float(gap_tol)
    if not all(math.isfinite(v) and v > 0 for v in (tolerance, gap_tolerance)):
        raise ValueError("inner_tol and gap_tol must be positive and finite")
    for value, name in ((inner_max, "inner_max"), (check_every, "check_every")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    maximum, interval = inner_max, check_every
    _require(isinstance(log_reference, torch.Tensor) and isinstance(fresh_gradient, torch.Tensor),
            "The log-plan and fresh gradient must be tensors.")
    _require(log_reference.ndim == 2 and log_reference.shape == fresh_gradient.shape
            and log_reference.dtype in (torch.float32, torch.float64)
            and fresh_gradient.dtype in (torch.float32, torch.float64)
            and torch.finfo(log_reference.dtype).bits >= torch.finfo(fresh_gradient.dtype).bits
            and fresh_gradient.device == log_reference.device,
            "The log-plan and gradient must be compatible floating-point matrices.")
    _require(bool(torch.isfinite(log_reference).all() and torch.isfinite(fresh_gradient).all()),
            "The fixed log-reference and fresh gradient must be finite.")
    plan, log_next, metadata = prox(log_reference, fresh_gradient, eta=eta, tol=tolerance,
                                    max_iter=maximum, check_every=interval)
    _require(isinstance(plan, torch.Tensor) and isinstance(log_next, torch.Tensor),
            "The proximal solver must return tensors.")
    _require(plan.shape == log_reference.shape and log_next.shape == log_reference.shape
            and plan.dtype == fresh_gradient.dtype and log_next.dtype == torch.float64
            and plan.device == log_reference.device and log_next.device == log_reference.device,
            "The proximal solver returned incompatible tensors.")
    _require(bool(torch.isfinite(plan).all() and torch.isfinite(log_next).all()
                 and (plan >= 0).all()), "The proximal solve returned invalid values.")
    audited = float(prox.marginal_residuals(plan)["both_l1"])
    if not math.isfinite(audited) or audited < 0 or audited > tolerance:
        raise smd.ProximalSolveError(
            f"Actual summed marginal residual {audited:.17g} exceeds tolerance {tolerance:.17g}.",
            certificate=metadata)
    if not metadata.get("converged", False):
        raise smd.ProximalSolveError("The proximal solver reported nonconvergence.", certificate=metadata)
    if metadata.get("proximal_optimality_certified") is not True:
        raise smd.ProximalSolveError("Missing numerical primal-dual optimality certificate.", certificate=metadata)
    required = ("primal_objective", "dual_objective", "primal_dual_gap",
                "normalized_primal_dual_gap", "primal_witness_marginal_l1",
                "stable_primal_dual_gap", "gap_roundoff_tolerance")
    try:
        primal, dual, gap, normalized, witness_residual, stable_gap, roundoff = (
            float(metadata[key]) for key in required)
    except (KeyError, TypeError, ValueError) as exc:
        raise smd.ProximalSolveError("Incomplete numerical primal-dual certificate.", certificate=metadata) from exc
    if not all(math.isfinite(value) for value in
               (primal, dual, gap, normalized, witness_residual, stable_gap, roundoff)):
        raise smd.ProximalSolveError("Nonfinite numerical primal-dual certificate.", certificate=metadata)
    scale = max(1., abs(primal), abs(dual))
    identity_tolerance = 1e-12 * scale
    if witness_residual < 0 or witness_residual > 1e-12:
        raise smd.ProximalSolveError("The primal witness is not feasible to FP64 audit tolerance.", certificate=metadata)
    if (roundoff < 0 or roundoff / scale > gap_tolerance
            or min(gap, stable_gap) < -roundoff
            or abs(gap - stable_gap) > roundoff
            or abs((primal - dual) - gap) > identity_tolerance):
        raise smd.ProximalSolveError("Invalid primal-dual gap or inconsistent objective values.", certificate=metadata)
    if abs(normalized - gap / scale) > 1e-12:
        raise smd.ProximalSolveError("Inconsistent normalized primal-dual gap.", certificate=metadata)
    # Independently enforce the conservative upper gap used by the kernel.
    # Cancellation may make raw J-D slightly negative; the reported FP64
    # roundoff allowance must itself fit within the selected tolerance.
    upper_normalized = (max(gap, stable_gap) + roundoff) / scale
    if upper_normalized > gap_tolerance:
        raise smd.ProximalSolveError(
            f"Normalized primal-dual gap upper bound {upper_normalized:.17g} exceeds {gap_tolerance:.17g}.",
            certificate=metadata)
    info = dict(metadata)
    info.update(iteration=int(iteration), eta=eta, fixed_reference="product_marginals",
                marginal_l1=audited, inner_tolerance=tolerance,
                normalized_gap_tolerance=gap_tolerance,
                normalized_gap_upper_bound=upper_normalized, proximal_optimality_certified=True)
    return plan, log_next, info


def load_problem(instance):
    instance = Path(instance)
    manifest_path = instance / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["array_files_sha256"]
    hashes, arrays = {}, {}
    for name in ("A", "B", "a", "b"):
        filename = name + ".npy"
        path = instance / filename
        hashes[filename] = sha256(path)
        if hashes[filename] != expected[filename]:
            raise ValueError(f"Input hash mismatch: {filename}")
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.float64:
            raise ValueError(f"Input {filename} is not original FP64")
        arrays[name] = torch.from_numpy(value.copy())
    problem = RowShardedGW(**{("A_local" if k == "A" else k): v
                             for k, v in arrays.items()}, exact_block_rows=256)
    if (problem.n, problem.m) != (manifest["n"], manifest["m"]):
        raise ValueError("Manifest dimensions disagree with actual arrays")
    cmax = smd.cost_bound_from_distances(problem.A, problem.B)
    return problem, cmax, hashes, sha256(manifest_path)


def _save_plan(out, filename, plan):
    path = out / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, plan.detach().cpu().numpy(), allow_pickle=False)
    return {"file": filename, "file_sha256": sha256(path)}


@torch.no_grad()
def diagnose(problem, plan, epsilon):
    rounded = problem.round_transport(plan)
    raw_residuals = problem.marginal_residuals(plan)
    rounded_residuals = problem.marginal_residuals(rounded)
    if rounded_residuals["both_l1"] > 100 * torch.finfo(torch.float64).eps:
        raise FloatingPointError("Saved rounded plan exceeds strict FP64 feasibility bound")
    raw_loss, rounded_loss = problem.loss(plan).item(), problem.loss(rounded).item()
    raw_entropy, rounded_entropy = entropy(plan).item(), entropy(rounded).item()
    values = (raw_loss, rounded_loss, raw_entropy, rounded_entropy)
    if not all(math.isfinite(x) for x in values):
        raise FloatingPointError("Nonfinite diagnostic objective")
    return rounded, dict(
        raw_loss=raw_loss, rounded_loss=rounded_loss,
        raw_entropy=raw_entropy, rounded_entropy=rounded_entropy,
        raw_regularized_loss=raw_loss + epsilon * raw_entropy,
        rounded_regularized_loss=rounded_loss + epsilon * rounded_entropy,
        raw_residuals=raw_residuals, rounded_residuals=rounded_residuals,
        raw_zero_entries=int((plan == 0).sum().item()),
        rounded_zero_entries=int((rounded == 0).sum().item()))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--instance", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--oracle", choices=("full", "pair"), required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--iterations", type=int, default=2000)
    result.add_argument("--epsilon", type=float, default=.001)
    result.add_argument("--pairs", type=int, default=1024)
    result.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS)
    result.add_argument("--progress-every", type=int, default=100)
    result.add_argument("--inner-max", type=int, default=INNER_MAX)
    return result


def validate_args(args):
    if args.iterations < 1 or args.pairs < 1 or args.progress_every < 1 or args.inner_max < 1:
        raise ValueError("Iterations, pair count, progress interval and inner cap must be positive")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    checkpoints = {int(x) for x in args.checkpoints.split(",") if x.strip()}
    if any(x < 0 for x in checkpoints):
        raise ValueError("Checkpoint indices must be nonnegative")
    # The common defaults may exceed a short smoke horizon.
    return {0, args.iterations} | {x for x in checkpoints if x <= args.iterations}


@torch.no_grad()
def run(args):
    checkpoints = validate_args(args)
    out = Path(args.out)
    # Deliberately outside the failure handler: never overwrite an existing run.
    out.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    start = time.perf_counter()
    counters = dict(completed_updates=0, full_gradient_calls=0,
                    stochastic_gradient_calls=0, pair_samples=0,
                    inner_stopping_failures=0, inner_iterations_total=0,
                    certificate_checks=0, exact_diagnostic_loss_calls=0,
                    updates_with_underflow=0, max_underflow_entries=0)
    trace, inner_trace, saved = [], [], {}
    timing = dict(started_at=started, solver_seconds=0., diagnostic_seconds=0.,
                  serialization_seconds=0.)
    try:
        problem, cmax, hashes, manifest_sha = load_problem(args.instance)
        prox = FastCertifiedKLProx(problem, gap_tol=INNER_GAP_TOL, certificate_block_rows=256)
        plan = problem.a[:, None] * problem.b[None, :]
        log_reference = plan.log()
        oracle_rng = torch.Generator(device="cpu").manual_seed(args.seed)
        output_seed = args.seed + 20000
        output_rng = torch.Generator(device="cpu").manual_seed(output_seed)
        selected_index = int(torch.randint(args.iterations, (1,), generator=output_rng).item())
        selected_probability = 1. / args.iterations
        selected_plan = None
        record = dict(
            status="running", config=dict(
                **vars(args), output_seed=output_seed, inner_tol=INNER_TOL,
                inner_gap_tol=INNER_GAP_TOL,
                inner_check_every=CHECK_EVERY, check_every=CHECK_EVERY),
            source_file_sha256=source_hashes(), array_files_sha256=hashes,
            manifest_sha256=manifest_sha, dimensions={"n": problem.n, "m": problem.m},
            environment=dict(hostname=platform.node(), platform=platform.platform(),
                             python=sys.version, numpy=np.__version__, torch=torch.__version__,
                             dtype="float64", device="cpu", distributed=False,
                             deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                             torch_threads=torch.get_num_threads(),
                             torch_interop_threads=torch.get_num_interop_threads(),
                             cpu_affinity=sorted(os.sched_getaffinity(0)),
                             thread_environment={k: os.environ.get(k) for k in
                                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                  "NUMEXPR_NUM_THREADS")}),
            counters=counters, trace=trace, inner_trace=inner_trace, saved_plans=saved,
            schedule=dict(kind="pot-fixed", cmax=cmax, epsilon=args.epsilon,
                          formula="eta_k=1/epsilon", eta0=1./args.epsilon,
                          eta_last=1./args.epsilon,
                          theory_scope="empirical, not claimed theorem-admissible",
                          inner_tol=INNER_TOL, inner_gap_tol=INNER_GAP_TOL,
                          inner_max=args.inner_max, inner_check_every=CHECK_EVERY,
                          inner_solver="FP64 matrix-vector Sinkhorn with original-log-kernel certificate and log-domain fallback",
                          output_distribution="uniform P_0,...,P_(T-1); diagnostic only",
                          fixed_reference="product_marginals",
                          certificate_objective="<G,Q> + epsilon*KL(Q||a*b); differs from <G,Q>+epsilon*h(Q) by a fixed feasible-set constant",
                          objective="full_squared_GW_no_half + epsilon*sum(P*(log(P)-1))"),
            selected_output=dict(index=selected_index, output_seed=output_seed,
                                 probability=selected_probability, diagnostic_only=True,
                                 rounded_selected_is_diagnostic=True), timing=timing)
        timing["preparation_seconds"] = time.perf_counter() - start

        def checkpoint(iteration):
            mark = time.perf_counter()
            rounded, values = diagnose(problem, plan, args.epsilon)
            counters["exact_diagnostic_loss_calls"] += 2
            timing["diagnostic_seconds"] += time.perf_counter() - mark
            mark = time.perf_counter()
            item = _save_plan(out, f"checkpoints/rounded_{iteration:06d}.npy", rounded)
            trace.append(dict(iteration=iteration, **values,
                              plan_file=item["file"], plan_sha256=item["file_sha256"]))
            timing["serialization_seconds"] += time.perf_counter() - mark
            return rounded

        def progress(status):
            atomic_json(out / "progress.json", dict(
                status=status, updated_at=utc_now(), counters=counters.copy(),
                last_trace=trace[-1] if trace else None,
                elapsed_seconds=time.perf_counter()-start))

        checkpoint(0)
        atomic_json(out / "metadata.json", record)
        progress("running")
        for iteration in range(args.iterations):
            if iteration == selected_index:
                selected_plan = plan.clone()
            mark = time.perf_counter()
            if args.oracle == "full":
                gw_gradient = problem.gradient(plan)
                counters["full_gradient_calls"] += 1
            else:
                sample = sample_pairs(problem, plan, args.pairs, generator=oracle_rng)
                gw_gradient = pair_gradient(problem, sample)
                counters["stochastic_gradient_calls"] += 1
                counters["pair_samples"] += args.pairs
            try:
                plan, log_plan, info = pot_step(
                    log_reference, gw_gradient, iteration=iteration,
                    epsilon=args.epsilon, prox=prox, inner_max=args.inner_max)
            except smd.ProximalSolveError:
                counters["inner_stopping_failures"] += 1
                raise
            underflow = int((plan == 0).sum().item())
            if underflow != info["underflow_entries"]:
                raise RuntimeError("Plan underflow count disagrees with numerical certificate")
            counters["updates_with_underflow"] += int(underflow > 0)
            counters["max_underflow_entries"] = max(counters["max_underflow_entries"], underflow)
            counters["completed_updates"] += 1
            counters["inner_iterations_total"] += info["inner_iterations"]
            counters["certificate_checks"] += info["certificate_checks"]
            inner_trace.append(info)
            timing["solver_seconds"] += time.perf_counter() - mark
            completed = iteration + 1
            if completed in checkpoints:
                checkpoint(completed)
            if completed in checkpoints or completed % args.progress_every == 0:
                progress("running")
                print(json.dumps(dict(completed_updates=completed, oracle=args.oracle,
                                      last_diagnostic_iteration=trace[-1]["iteration"],
                                      last_rounded_loss=trace[-1]["rounded_loss"])), flush=True)

        if selected_plan is None:
            raise RuntimeError("The fixed-horizon selected iterate was not captured")
        mark = time.perf_counter()
        last_rounded, last_values = diagnose(problem, plan, args.epsilon)
        selected_rounded, selected_values = diagnose(problem, selected_plan, args.epsilon)
        counters["exact_diagnostic_loss_calls"] += 4
        timing["diagnostic_seconds"] += time.perf_counter()-mark
        mark = time.perf_counter()
        for name, value in (("last_raw", plan), ("last_rounded", last_rounded),
                            ("selected_raw", selected_plan), ("selected_rounded", selected_rounded)):
            saved[name] = _save_plan(out, name + ".npy", value)
        record["selected_output"].update(iteration=selected_index, **selected_values)
        record["last_output"] = dict(iteration=args.iterations, **last_values)
        timing["serialization_seconds"] += time.perf_counter()-mark
        if source_hashes() != record["source_file_sha256"]:
            raise RuntimeError("Numerical source files changed during the run")
        timing.update(completed_at=utc_now(), total_seconds=time.perf_counter()-start)
        record["status"] = "completed"
        atomic_json(out / "metadata.json", record)
        progress("completed")
        return record
    except Exception as exc:
        failure = dict(status="failed", started_at=started, failed_at=utc_now(),
                       config=vars(args), counters=counters, trace=trace,
                       inner_trace=inner_trace, error_type=type(exc).__name__,
                       error=str(exc), traceback=traceback.format_exc(),
                       certificate=getattr(exc, "certificate", None))
        atomic_json(out / "failure.json", failure)
        atomic_json(out / "progress.json", dict(
            status="failed", updated_at=utc_now(), counters=counters,
            last_trace=trace[-1] if trace else None, error=str(exc)))
        raise


def main():
    args = parser().parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args)


if __name__ == "__main__":
    main()
