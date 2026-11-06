"""Instrumented small-instance Moreau FW / row-projection diagnostics.

Every pre-update iterate is evaluated, so the reported arithmetic mean uses all
P_0,...,P_(T-1). This intentionally expensive runner is for Stage A/B diagnostics,
not large-instance throughput comparisons or a claim of theorem verification.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from gw_core import GWProblem, marginal_residuals, original_fw_gap, round_transport
from smoothing_fw import MomentumConfig, initialize_state, step

ROOT = Path(__file__).resolve().parent
AVERAGED = ("true_smoothed_gap", "row_l1", "col_l1", "col_l2", "col_l2_squared", "momentum_error_l2", "oracle_error_l2")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    result = {"python": sys.version, "platform": platform.platform(), "hostname": platform.node(),
              "versions": {key: importlib.metadata.version(key) for key in ("numpy", "scipy", "torch")},
              "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
              "git_status_at_start": status.stdout.splitlines() if status.returncode == 0 else None,
              "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
              "thread_environment": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "cuda_runtime": torch.version.cuda, "amp": False,
              "scheduler": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK", "SLURM_GPUS", "CUDA_VISIBLE_DEVICES")}}
    cpu = Path("/proc/cpuinfo")
    if cpu.exists():
        result["cpu_model"] = next((line.split(":", 1)[1].strip() for line in cpu.read_text().splitlines() if line.startswith("model name")), None)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                         "compute_capability": [properties.major, properties.minor]}
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            result["nvidia_driver_versions"] = sorted(set(driver.stdout.splitlines())) if driver.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_driver_versions"] = None
    return result


def memory_usage(device):
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": maximum if sys.platform == "darwin" else maximum * 1024,
              "memory_scope": "process high-water RSS and GPU allocated/reserved peaks include diagnostics and final validation"}
    if device.type == "cuda":
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def final_float64_evaluation(raw, actual_rounded, canonical, mu, gap_max_n):
    """Promote the actual emitted output; separately audit a new float64 repair."""
    problem = GWProblem(*(torch.as_tensor(canonical[key], dtype=torch.float64) for key in ("A", "B", "a", "b")))
    P = torch.as_tensor(raw, dtype=torch.float64)
    actual = torch.as_tensor(actual_rounded, dtype=torch.float64)
    R = round_transport(P, problem.a, problem.b)
    G = problem.gradient(P)
    H = G + (P.sum(0) - problem.b)[None, :] / mu
    actual_tolerance = 1e-12 if np.asarray(actual_rounded).dtype == np.float64 else 1e-5
    result = {"raw_smooth_loss": problem.loss(P).item(), "rounded_feasible_loss": problem.loss(actual).item(),
              "raw_residuals": marginal_residuals(P, problem.a, problem.b),
              "rounded_residuals": marginal_residuals(actual, problem.a, problem.b),
              "repair_entrywise_l1": (actual-P).abs().sum().item(),
              "true_smoothed_gap": ((H*P).sum() - (problem.a*H.min(1).values).sum()).item(),
              "mu_at_terminal_point": mu,
              "original_fw_gap_at_rounded": original_fw_gap(problem, actual) if max(P.shape) <= gap_max_n else None,
              "rounded_source": "Actual solver-dtype rounded output saved as P_rounded, promoted to float64 without further repair",
              "rounded_source_dtype": str(np.asarray(actual_rounded).dtype),
              "rounded_feasibility_tolerance": actual_tolerance,
              "recomputed_float64_repair": {
                  "rounded_feasible_loss": problem.loss(R).item(),
                  "rounded_residuals": marginal_residuals(R, problem.a, problem.b),
                  "repair_entrywise_l1": (R-P).abs().sum().item(),
                  "difference_from_actual_rounded_entrywise_l1": (R-actual).abs().sum().item(),
                  "original_fw_gap_at_rounded": original_fw_gap(problem, R) if max(P.shape) <= gap_max_n else None,
                  "note": "Separate numerical audit: reapply rounding to the promoted raw iterate; saved as P_rounded_float64"},
              "note": "CPU float64 evaluation of the saved raw and actual rounded iterates and frozen stored costs; no solver refinement"}
    if not torch.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("The actual rounded output must be finite and nonnegative.")
    assert result["rounded_residuals"]["both_l1"] <= actual_tolerance
    assert result["recomputed_float64_repair"]["rounded_residuals"]["both_l1"] <= 1e-12
    return result, R.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--oracle", choices=["exact", "sketch", "anchor"], default="sketch")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mu-scaled", type=float, default=1., help="m*mu0; the operator T is not rescaled")
    parser.add_argument("--update", choices=["fw", "row_projection"], default="fw")
    parser.add_argument("--row-eta0", type=float)
    parser.add_argument("--row-eta-power", type=float)
    parser.add_argument("--gap-max-n", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1 or args.gap_max_n < 1:
        parser.error("Positive steps, threads and gap-max-n are required.")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Output directory is nonempty; choose a new run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable.")
        torch.cuda.reset_peak_memory_stats(device)
    instance = Path(args.instance).resolve()
    with np.load(instance) as data:
        canonical = {key: np.array(data[key], copy=True) for key in ("A", "B", "a", "b")}
    hashes = {key + "_sha256": hashlib.sha256(canonical[key].tobytes()).hexdigest() for key in ("A", "B")}
    manifest = json.loads(instance.with_suffix(".json").read_text())
    if any(manifest[key] != value for key, value in hashes.items()):
        raise ValueError("Frozen cost hashes do not match the instance manifest.")
    n, m = len(canonical["a"]), len(canonical["b"])
    if n != m:
        raise ValueError("This initial experiment requires square uniform instances.")
    problem = GWProblem(*(torch.as_tensor(canonical[key], dtype=dtype, device=device) for key in ("A", "B", "a", "b")))
    config = MomentumConfig(alpha=.75, beta=.5, delta=.25, gamma0=1., rho0=1.,
                            mu0=args.mu_scaled/m, oracle=args.oracle, batch=args.batch,
                            update=args.update, eta0=args.row_eta0, eta_exponent=args.row_eta_power,
                            theory_id="supplied draft Proposition 2.11 (open-loop), equations 65-67",
                            output_statistic="arithmetic mean of true smoothed FW gap at every pre-update P_k, k=0..T-1")
    source_files = ("gw_core.py", "smoothing_fw.py", "run_smoothing.py", "PROTOCOL.md")
    metadata = {"arguments": vars(args), "started_utc": datetime.now(timezone.utc).isoformat(),
                "instance": str(instance), "instance_sha256": sha256(instance), "instance_manifest": manifest,
                **hashes, "shape": [n, m], "schedules": config.metadata(), "environment": environment(device),
                "source_sha256": {name: sha256(ROOT/name) for name in source_files},
                "theory_mapping": {"p_step": .75, "r_momentum": .5, "q_smoothing": .25,
                    "mu0": config.mu0, "m_mu0": args.mu_scaled, "beta0_theorem": config.mu0/n,
                    "operator_norm_squared": n,
                    "constraint_mapping": "Dtilde={P: P^T1=b}; dist_F(P,Dtilde)^2=||P^T1-b||_2^2/n; beta_k=mu_k/n",
                    "penalty_sign": "positive T*(T(P)-b)/mu; follows Moreau definition and protocol; corrects reversed draft Algorithm 1 sign",
                    "momentum_initialization": "d0 is first sampled smooth gradient; exact current penalty added afterward",
                    "scope": "theory-consistency diagnostics; not proof verification or last-iterate guarantees",
                    "exact_oracle_ablation": "retains stochastic-track momentum exponent .5; does not invoke deterministic Proposition 2.12"},
                "checkpoint_semantics": "After update k+1 completes, evaluate its retained pre-update P_k and matching d_k. Cumulative solver time includes that completed update; no time is backdated.",
                "terminal_semantics": "P_T gets a separate terminal evaluation; no d_T is sampled, and terminal metrics are excluded from the T-point trajectory averages.",
                "evaluation_rng": "none; all reported evaluation metrics are exact",
                "peak_memory_scope": "includes diagnostics; no solver-only peak claim",
                "timing_scope": "main() wall time excludes Python imports; GPU synchronized at all timing boundaries"}
    manuscript = ROOT / "provenance/manuscript.txt"
    if manuscript.exists():
        metadata["manuscript_sha256"] = sha256(manuscript)
    sync(device)
    timing = {"preparation_cache_seconds": time.perf_counter()-started, "solver_seconds": 0.,
              "repair_seconds": 0., "diagnostic_seconds": 0., "serialization_seconds": 0.,
              "final_float64_validation_seconds": 0.}
    serialization_started = time.perf_counter()
    write_json(out/"config.json", metadata)
    timing["serialization_seconds"] += time.perf_counter()-serialization_started
    sync(device)
    init_started = time.perf_counter()
    state = initialize_state(problem, config, seed=args.seed)
    sync(device)
    timing["solver_seconds"] = time.perf_counter()-init_started
    initialization_seconds = timing["solver_seconds"]
    sums = {key: 0. for key in AVERAGED}
    diagnostic_counts = {"exact_gradient_calls": 0, "full_loss_calls": 0, "assignment_lmo_calls": 0, "row_gap_evaluations": 0, "rounding_calls": 0}
    kappa = max((float(canonical["A"].max())-float(canonical["B"].min()))**2,
                (float(canonical["B"].max())-float(canonical["A"].min()))**2)
    records = []

    def evaluate(P, k, mu, info=None):
        sync(device)
        began = time.perf_counter()
        rounded = round_transport(P, problem.a, problem.b)
        sync(device)
        repair_time = time.perf_counter()-began
        timing["repair_seconds"] += repair_time
        diagnostic_counts["rounding_calls"] += 1
        began = time.perf_counter()
        G = problem.gradient(P)
        H = G + (P.sum(0)-problem.b)[None, :]/mu
        residuals = marginal_residuals(P, problem.a, problem.b)
        rounded_residuals = marginal_residuals(rounded, problem.a, problem.b)
        true_gap = ((H*P).sum()-(problem.a*H.min(1).values).sum()).item()
        original_gap = original_fw_gap(problem, rounded) if max(n, m) <= args.gap_max_n else None
        diagnostic_counts["exact_gradient_calls"] += 1 + int(original_gap is not None)
        diagnostic_counts["assignment_lmo_calls"] += int(original_gap is not None)
        diagnostic_counts["full_loss_calls"] += 2
        diagnostic_counts["row_gap_evaluations"] += 1
        row = {"point_iteration": k, "completed_updates_at_checkpoint": state.iteration,
               "checkpoint": "pre_update" if info is not None else "terminal",
               "trajectory_average_included": info is not None,
               "mu": mu, "m_mu": m*mu, "beta_theorem": mu/n,
               "gamma": info.gamma if info else None, "rho": info.rho if info else None,
               "eta": info.eta if info else None,
               "solver_seconds_cumulative": timing["solver_seconds"],
               "repair_seconds_current": repair_time, "repair_seconds_cumulative": timing["repair_seconds"],
               "time_to_this_feasible_output": timing["solver_seconds"]+repair_time,
               "raw_smooth_loss": problem.loss(P).item(), "rounded_feasible_loss": problem.loss(rounded).item(),
               **residuals, "col_l2_squared": residuals["col_l2"]**2,
               "rounded_both_l1": rounded_residuals["both_l1"], "rounded_min_entry": rounded_residuals["min_entry"],
               "repair_entrywise_l1": (rounded-P).abs().sum().item(),
               "true_smoothed_gap": true_gap, "original_fw_gap_at_rounded": original_gap,
               "certificate_rhs": true_gap+6*kappa*residuals["col_l1"],
               "certificate_excess": original_gap-true_gap-6*kappa*residuals["col_l1"] if original_gap is not None else None,
               "momentum_error_l2": (info.d_used-G).norm().item() if info else None,
               "oracle_error_l2": (info.gradient_estimate-G).norm().item() if info else None,
               "gradient_l2": G.norm().item(), "raw_zero_entries": int((P == 0).sum().item()),
               **{"solver_"+key: value for key, value in asdict(state.counts).items()}}
        if info is not None:
            for key in AVERAGED:
                sums[key] += row[key]
        points_averaged = k+1 if info is not None else k
        row["trajectory_average_points"] = points_averaged
        row.update({"mean_"+key: value/points_averaged for key, value in sums.items()})
        sync(device)
        timing["diagnostic_seconds"] += time.perf_counter()-began
        row["diagnostic_seconds_cumulative"] = timing["diagnostic_seconds"]
        row["serialization_seconds_before_checkpoint"] = timing["serialization_seconds"]
        row["instrumented_seconds_at_checkpoint"] = time.perf_counter()-started
        return row, rounded

    try:
        with (out/"trace.csv").open("w", newline="") as handle:
            writer = None
            for _ in range(args.steps):
                sync(device)
                step_started = time.perf_counter()
                info = step(problem, state)
                sync(device)
                timing["solver_seconds"] += time.perf_counter()-step_started
                row, _ = evaluate(info.P_before, info.update_index, info.mu, info)
                records.append(row)
                serialization_started = time.perf_counter()
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                timing["serialization_seconds"] += time.perf_counter()-serialization_started
            final_row, final_rounded = evaluate(state.P, state.iteration, config.schedule(state.iteration)["mu"])
            serialization_started = time.perf_counter()
            writer.writerow(final_row)
            handle.flush()
            timing["serialization_seconds"] += time.perf_counter()-serialization_started
        sync(device)
        validation_started = time.perf_counter()
        raw_numpy = state.P.detach().cpu().numpy()
        rounded_numpy = final_rounded.detach().cpu().numpy()
        float64_result, rounded64 = final_float64_evaluation(raw_numpy, rounded_numpy, canonical, config.schedule(state.iteration)["mu"], args.gap_max_n)
        timing["final_float64_validation_seconds"] = time.perf_counter()-validation_started
        serialization_started = time.perf_counter()
        np.savez(out/"final.npz", P_raw=raw_numpy, P_rounded=rounded_numpy, P_rounded_float64=rounded64)
        write_json(out/"final_float64_evaluation.json", float64_result)
        timing["serialization_seconds"] += time.perf_counter()-serialization_started
        result = {"status": "completed", "completed_updates": state.iteration,
                  "trajectory_average_points": len(records), "trajectory_average_index_range": [0, args.steps-1],
                  "trajectory_means": {key: value/args.steps for key, value in sums.items()},
                  "solver_counts": asdict(state.counts), "trajectory_diagnostic_counts": diagnostic_counts,
                  "final_cpu_float64_validation_counts": {"exact_gradient_calls": 1+2*int(max(n,m)<=args.gap_max_n), "full_loss_calls": 3, "assignment_lmo_calls": 2*int(max(n,m)<=args.gap_max_n), "rounding_calls": 1},
                  "initialization_solver_seconds": initialization_seconds, "timing": dict(timing),
                  "total_instrumented_seconds_before_summary_write": time.perf_counter()-started,
                  "memory": memory_usage(device), "final_checkpoint": final_row,
                  "final_float64": float64_result,
                  "max_row_l1_in_preupdate_trajectory": max(row["row_l1"] for row in records),
                  "max_certificate_excess": max((row["certificate_excess"] for row in records if row["certificate_excess"] is not None), default=None),
                  "claim_scope": "Every-iterate theory-consistency diagnostics only; no proof, last-iterate theorem guarantee, tuning winner or algorithmic speedup claim"}
        write_json(out/"summary.json", result)
        print(json.dumps({"output": str(out), "status": result["status"], "updates": state.iteration,
                          "mean_true_smoothed_gap": result["trajectory_means"]["true_smoothed_gap"],
                          "terminal_raw_col_l1": float64_result["raw_residuals"]["col_l1"],
                          "terminal_rounded_loss": float64_result["rounded_feasible_loss"]}, indent=2))
    except Exception:
        write_json(out/"failure.json", {"completed_updates": state.iteration, "solver_counts": asdict(state.counts),
                                       "timing": timing, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
