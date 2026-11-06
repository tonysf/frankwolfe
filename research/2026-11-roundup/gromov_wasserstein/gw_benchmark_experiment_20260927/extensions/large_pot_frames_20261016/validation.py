"""Small scheduled CPU/CUDA gates for the large-data POT-step implementation.

Run directly for one process or with torchrun for four row shards. Dense NumPy
reference algebra is independent of the distributed implementation; only tiny
fixtures are created here. Dataset-size/memory and resume gates belong to the
campaign controller and runner tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, ROOT / "extensions/shareseq_25000_20260930",
                  ROOT / "extensions/pair_sampling_20261011"):
    sys.path.insert(0, str(directory))
from prox import DistributedCertifiedPOTProx
from sharded_gw import RowShardedGW, row_partition
from oracle import PairSample, pair_gradient

SOURCE_FILES = (
    "extensions/large_pot_frames_20261016/prox.py",
    "extensions/large_pot_frames_20261016/validation.py",
    "extensions/published_stochastic_20261002/certified_prox.py",
    "extensions/shareseq_25000_20260930/sharded_gw.py",
    "extensions/pair_sampling_20261011/oracle.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def fixture():
    rng = np.random.default_rng(785)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    A = np.linalg.norm(x[:, None] - x[None, :], axis=-1) * .035
    B = np.linalg.norm(y[:, None] - y[None, :], axis=-1) * .03
    a, b = np.arange(1., 8.), np.arange(1., 6.)
    a, b = a / a.sum(), b / b.sum()
    return A, B, a, b


def dense_gradient(A, B, plan):
    costs = (A[:, :, None, None] - B[None, None, :, :]) ** 2
    return 2 * np.einsum("ikjl,kl->ij", costs, plan)


def dense_sinkhorn(gradient, a, b, eta=1000., tolerance=1e-13):
    """Independent tiny log-domain entropic OT reference (no product kernel)."""
    log_kernel = -eta * gradient
    loga, logb = np.log(a), np.log(b)
    v = np.zeros_like(b)
    for iteration in range(100000):
        u = loga - np.logaddexp.reduce(log_kernel + v[None, :], axis=1)
        v = logb - np.logaddexp.reduce(log_kernel + u[:, None], axis=0)
        if (iteration + 1) % 10 == 0:
            Q = np.exp(log_kernel + u[:, None] + v[None, :])
            residual = np.abs(Q.sum(1) - a).sum() + np.abs(Q.sum(0) - b).sum()
            if residual <= tolerance:
                return Q
    raise RuntimeError("Independent tiny reference failed to converge")


def _audit(prox, Q, logQ, info):
    problem = prox.problem
    scale = max(1., abs(info["primal_objective"]), abs(info["dual_objective"]))
    upper = (max(info["primal_dual_gap"], info["stable_primal_dual_gap"])
             + info["gap_roundoff_tolerance"]) / scale
    actual = prox.marginal_residuals(Q)["both_l1"]
    problem._require(info["converged"] and info["certificate_valid"]
                     and info["proximal_optimality_certified"]
                     and actual <= 1e-12 and upper <= 1e-11,
                     "The validation solve failed its numerical acceptance tests.")
    problem._require(torch.allclose(logQ.exp(), Q, atol=0., rtol=0.),
                     "Returned log-plan and physical plan disagree.")
    return dict(actual_marginal_l1=actual, normalized_gap_upper_bound=upper,
                inner_iterations=info["inner_iterations"],
                sinkhorn_path=info["sinkhorn_path"],
                underflow_entries=info["underflow_entries"])


@torch.no_grad()
def run_validation(device="cpu"):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    A, B, a, b = fixture()
    start, stop = row_partition(len(a), world, rank)
    tensors = [torch.tensor(value, dtype=torch.float64, device=device)
               for value in (A[start:stop], B, a, b)]
    problem = RowShardedGW(*tensors, exact_block_rows=2)
    prox = DistributedCertifiedPOTProx(problem, gap_tol=1e-11,
                                      certificate_block_rows=2)
    log_reference = (problem.a_local.log()[:, None] + problem.b.log()[None, :])
    reference = np.outer(a, b)
    plan = torch.tensor(reference[start:stop], dtype=torch.float64, device=device)
    checks = []
    for iteration in range(3):
        gradient = problem.gradient(plan)
        expected_gradient = dense_gradient(A, B, reference)
        error = problem.global_max((gradient - torch.tensor(
            expected_gradient[start:stop], device=device)).abs()).item()
        problem._require(error < 1e-12, "Full gradient differs from direct four-index reference.")
        plan, log_plan, info = prox(log_reference, gradient, eta=1000., tol=1e-12,
                                    check_every=1)
        reference = dense_sinkhorn(expected_gradient, a, b)
        error = problem.global_max((plan - torch.tensor(reference[start:stop],
                                    device=device)).abs()).item()
        problem._require(error < 2e-10, "Full POT-step trajectory differs from dense reference.")
        checks.append(dict(name=f"full_gradient_update_{iteration + 1}",
                           max_plan_error=error, **_audit(prox, plan, log_plan, info)))

    K = np.array([0, 6, 2, 4, 1, 5, 3, 0, 2], dtype=np.int64)
    L = np.array([4, 1, 3, 0, 2, 4, 1, 2, 0], dtype=np.int64)
    sample = PairSample(torch.tensor(K, device=device), torch.tensor(L, device=device),
                        torch.tensor(1., dtype=torch.float64, device=device))
    gradient = pair_gradient(problem, sample)
    expected_gradient = 2 * ((A[:, K, None] - B.T[L][None, :, :]) ** 2).mean(1)
    error = problem.global_max((gradient - torch.tensor(expected_gradient[start:stop],
                                device=device)).abs()).item()
    problem._require(error < 1e-12, "Pair oracle differs from direct sampled-cost average.")
    plan, log_plan, info = prox(log_reference, gradient, eta=1000., tol=1e-12,
                                check_every=1)
    reference = dense_sinkhorn(expected_gradient, a, b)
    error = problem.global_max((plan - torch.tensor(reference[start:stop],
                                device=device)).abs()).item()
    problem._require(error < 2e-10, "Pair POT step differs from dense reference.")
    checks.append(dict(name="specified_pairs_update", max_plan_error=error,
                       **_audit(prox, plan, log_plan, info)))

    # Only the owner of global row zero initially sees a zero kernel entry.
    # Every rank must take the same fallback without mismatched collectives.
    gradient = torch.zeros_like(plan)
    if start == 0 and stop > 0:
        gradient[0, 0] = 1.
    plan, log_plan, info = prox(log_reference, gradient, eta=1000., tol=1e-12,
                                check_every=1)
    problem._require(info["sinkhorn_path"] == "original-log-domain-fallback"
                     and info["fast_scaling_iterations"] == 0,
                     "One-rank kernel underflow did not trigger collective fallback.")
    checks.append(dict(name="single_rank_underflow_collective_fallback",
                       **_audit(prox, plan, log_plan, info)))

    result = dict(status="passed", device=torch.device(device).type, world_size=world,
                  checks=checks, source_file_sha256=source_hashes(),
                  dtype="float64", eta=1000., epsilon=.001,
                  fixture_shape=[len(a), len(b)],
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  backend=str(dist.get_backend()) if dist.is_initialized() else "none")
    if torch.device(device).type == "cuda":
        result["gpu"] = torch.cuda.get_device_name()
        result["peak_allocated_bytes_per_rank_max"] = int(problem._reduce(
            torch.tensor(torch.cuda.max_memory_allocated(), device=device),
            dist.ReduceOp.MAX).item())
        result["peak_reserved_bytes_per_rank_max"] = int(problem._reduce(
            torch.tensor(torch.cuda.max_memory_reserved(), device=device),
            dist.ReduceOp.MAX).item())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = "cpu"
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    try:
        result = run_validation(device)
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=True)
            destination = args.out / "checks.json"
            with destination.open("x") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            print(json.dumps({"status": result["status"], "world_size": world,
                              "report": str(destination)}, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
