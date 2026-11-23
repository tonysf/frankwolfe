"""Fixed-plan preliminary diagnostics for PROTOCOL.md section 6.

Uses the supplied reference oracle kernels without changing their estimators.
CPU results are not A100 measurements. This does not freeze an oracle. The planted
permutation is deliberately never loaded. No optimization is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

# Set these before importing numerical libraries; all three kernels get one thread.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from gw_core import GWProblem, marginal_residuals, rademacher, row_lmo

PLAN_NAMES = ("product", "mixed_permutation", "concentrated_permutation")
ORACLE_NAMES = ("anchor", "trace_sketch")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def source_dirty_status(root: Path) -> list[str] | None:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True)
        return status.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def derived_seed(base: int, plan_id: int, oracle_id: int, q: int, replicate: int) -> int:
    # All entries are positive and the returned seed is accepted by torch.Generator.
    state = np.random.SeedSequence([base, plan_id, oracle_id, q, replicate]).generate_state(2)
    return (int(state[0]) | (int(state[1]) << 32)) & ((1 << 63) - 1)


def distribution(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)),
            "min": float(x.min()), "max": float(x.max()),
            "std_sample": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def draw_sample(P: torch.Tensor, oracle: str, q: int, generator: torch.Generator) -> torch.Tensor:
    if oracle == "anchor":
        # Uniform flat anchors give independent uniform k,l and replacement sampling.
        flat = torch.randint(P.numel(), (q,), generator=generator, device=P.device)
        return torch.stack((flat // P.shape[1], flat % P.shape[1]), dim=1)
    return rademacher(P.shape[1], q, like=P, generator=generator)


def oracle_value(problem: GWProblem, P: torch.Tensor, oracle: str, sample: torch.Tensor) -> torch.Tensor:
    if oracle == "anchor":
        return problem.anchor_gradient(P, sample)
    return problem.sketch_gradient(P, sample)


def make_plans(problem: GWProblem, seed: int) -> tuple[dict[str, torch.Tensor], dict]:
    n, m = problem.A.shape[0], problem.B.shape[0]
    if n != m or not torch.allclose(problem.a, torch.full_like(problem.a, 1 / n)) or not torch.allclose(problem.b, torch.full_like(problem.b, 1 / n)):
        raise ValueError("The diagnostic plan construction needs uniform square marginals.")
    product = torch.outer(problem.a, problem.b)
    metadata = {}
    permutations = []
    for plan_id in (1, 2):
        init_seed = derived_seed(seed, plan_id, 0, 0, 0)
        rng = np.random.default_rng(init_seed)
        permutation = rng.permutation(n)
        R = torch.zeros_like(product)
        R[torch.arange(n, device=product.device), torch.from_numpy(permutation).to(product.device)] = 1 / n
        permutations.append(R)
        metadata[PLAN_NAMES[plan_id]] = {"initialization_seed": init_seed,
                                        "independent_permutation": permutation.tolist()}
    return {"product": product, "mixed_permutation": 0.9 * product + 0.1 * permutations[0],
            "concentrated_permutation": permutations[1]}, metadata


def plot_results(summaries: list[dict], exact: dict, out: Path, device: str, dtype: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    metrics = [
        ("relative_error_frobenius", "Relative Frobenius error", "relative_error"),
        ("fw_decision_error", "True-gradient row-LMO decision error", "decision_error"),
        ("sampling_inclusive_seconds", "Oracle wall time, sampling included (ms)", "timing"),
    ]
    for key, ylabel, suffix in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), constrained_layout=True)
        for ax, plan in zip(axes, PLAN_NAMES):
            for oracle, marker in zip(ORACLE_NAMES, ("o", "s")):
                group = sorted((s for s in summaries if s["plan"] == plan and s["oracle"] == oracle), key=lambda s: s["batch_size"])
                q = np.array([s["batch_size"] for s in group])
                scale = 1000 if suffix == "timing" else 1
                median = np.array([s[key]["median"] for s in group]) * scale
                low = np.array([s[key]["q25"] for s in group]) * scale
                high = np.array([s[key]["q75"] for s in group]) * scale
                ax.plot(q, median, marker=marker, label=oracle)
                ax.fill_between(q, low, high, alpha=0.18)
            if suffix == "timing":
                ax.axhline(exact[plan]["seconds"]["median"] * 1000, linestyle="--", color="black", label="exact full gradient")
                ax.set_yscale("log")
            if suffix == "relative_error":
                ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 32, 128], ["8", "32", "128"])
            ax.set_xlabel("Batch size q")
            ax.set_title(plan.replace("_", " "))
            ax.grid(alpha=0.2)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"{device.upper()} preliminary · {dtype} · one CPU thread · median and interquartile range", fontsize=10)
        for extension in ("png", "pdf"):
            path = out / f"oracle_{device}_{suffix}.{extension}"
            fig.savefig(path, dpi=180)
            saved.append(str(path))
        plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--label", default="oracle_cpu_n128_s1000")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--out", type=Path, help="Output root containing runs, reports, figures; default is script directory.")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--initialization-seed", type=int, default=3000)
    parser.add_argument("--stochastic-seed", type=int, default=0)
    parser.add_argument("--order-seed", type=int, default=9000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 2 or args.warmup < 1 or any(q < 1 for q in args.batches):
        parser.error("At least two repetitions, one warmup and positive batch sizes are required.")
    if len(set(args.batches)) != len(args.batches):
        parser.error("Batch sizes must be distinct.")
    if any(s < 0 for s in (args.initialization_seed, args.stochastic_seed, args.order_seed)):
        parser.error("Seeds must be nonnegative.")
    if Path(args.label).name != args.label or not args.label.startswith("oracle_"):
        parser.error("Use a simple label beginning with oracle_.")
    root = Path(__file__).resolve().parent
    output_root = args.out.resolve() if args.out else root
    run_dir, report_dir = output_root / "runs" / args.label, output_root / "reports"
    figure_dir = output_root / "figures" / args.label
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback.")
    # Preserve previous evidence: reruns need a distinct label.
    run_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    instance = args.instance.resolve()
    with np.load(instance, allow_pickle=False) as frozen:
        # Only A,B,a,b enter diagnostics; never read the planted correspondence.
        arrays = {key: frozen[key].copy() for key in ("A", "B", "a", "b")}
    manifest_path = instance.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    hashes = {key: hashlib.sha256(arrays[key].tobytes()).hexdigest() for key in ("A", "B")}
    for key in ("A", "B"):
        if hashes[key] != manifest[f"{key}_sha256"]:
            raise ValueError(f"Frozen {key} hash differs from its manifest.")
    problem = GWProblem(**{key: torch.from_numpy(value).to(device=device, dtype=dtype) for key, value in arrays.items()})
    plans, plan_init = make_plans(problem, args.initialization_seed)
    true_gradients = {name: problem.gradient(P) for name, P in plans.items()}
    true_norms = {name: float(G.norm()) for name, G in true_gradients.items()}
    true_lmo_values = {name: float((problem.a * G.min(1).values).sum()) for name, G in true_gradients.items()}
    feasibility = {name: marginal_residuals(P, problem.a, problem.b) for name, P in plans.items()}
    feasibility_tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    if any(v["both_l1"] > feasibility_tolerance or v["min_entry"] < 0 for v in feasibility.values()):
        raise ValueError("Diagnostic initialization is not feasible to the recorded dtype tolerance.")
    groups = {}
    tasks = []
    exact_rows = []
    for plan_id, plan in enumerate(PLAN_NAMES):
        for rep in range(args.repetitions):
            tasks.append((plan_id, -1, 0, rep))
        for oracle_id, oracle in enumerate(ORACLE_NAMES):
            for q in args.batches:
                groups[(plan, oracle, q)] = {"count": 0, "mean": torch.zeros_like(plans[plan]),
                                            "m2": 0.0, "rows": []}
                for rep in range(args.repetitions):
                    tasks.append((plan_id, oracle_id, q, rep))
    np.random.default_rng(args.order_seed).shuffle(tasks)
    synchronize(device)
    preparation_seconds = time.perf_counter() - start
    warmup_start = time.perf_counter()
    for plan_id, plan in enumerate(PLAN_NAMES):
        for rep in range(args.warmup):
            problem.gradient(plans[plan])
            for oracle_id, oracle in enumerate(ORACLE_NAMES):
                for q in args.batches:
                    generator = torch.Generator(device=device).manual_seed(derived_seed(args.stochastic_seed, plan_id, oracle_id, q, args.repetitions + rep))
                    oracle_value(problem, plans[plan], oracle, draw_sample(plans[plan], oracle, q, generator))
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start
    rows = []
    kernel_start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    kernel_end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    diagnostic_seconds = 0.0
    for order, (plan_id, oracle_id, q, rep) in enumerate(tasks):
        plan = PLAN_NAMES[plan_id]
        P = plans[plan]
        if oracle_id == -1:
            synchronize(device)
            timer = time.perf_counter_ns()
            exact_gradient = problem.gradient(P)
            synchronize(device)
            elapsed = (time.perf_counter_ns() - timer) * 1e-9
            if not torch.equal(exact_gradient, true_gradients[plan]):
                raise RuntimeError("Exact full-gradient repeat was not deterministic.")
            exact_rows.append({"plan": plan, "replicate": rep, "execution_order": order, "seconds": elapsed})
            continue
        oracle = ORACLE_NAMES[oracle_id]
        seed = derived_seed(args.stochastic_seed, plan_id, oracle_id, q, rep)
        # Generator construction is seed bookkeeping, excluded consistently from
        # the sampled evaluation. Drawing the samples is inside the main timing.
        generator = torch.Generator(device=device).manual_seed(seed)
        synchronize(device)
        inclusive_start = time.perf_counter_ns()
        sample = draw_sample(P, oracle, q, generator)
        if kernel_start_event is not None:
            kernel_start_event.record()
        kernel_start = time.perf_counter_ns()
        estimate = oracle_value(problem, P, oracle, sample)
        if kernel_end_event is not None:
            kernel_end_event.record()
        synchronize(device)
        finish = time.perf_counter_ns()
        inclusive_seconds = (finish - inclusive_start) * 1e-9
        kernel_seconds = (kernel_start_event.elapsed_time(kernel_end_event) * 1e-3
                          if kernel_start_event is not None else (finish - kernel_start) * 1e-9)
        diagnostic_start = time.perf_counter()
        if not torch.isfinite(estimate).all():
            raise RuntimeError(f"Nonfinite gradient in {plan}/{oracle}/q={q}/rep={rep}.")
        error = float((estimate - true_gradients[plan]).norm())
        decision_error = float((true_gradients[plan] * row_lmo(estimate, problem.a)).sum()) - true_lmo_values[plan]
        decision_tolerance = 20 * torch.finfo(dtype).eps * max(1.0, abs(true_lmo_values[plan]))
        if decision_error < -decision_tolerance:
            raise RuntimeError("Negative decision regret exceeds roundoff tolerance.")
        record = {"plan": plan, "oracle": oracle, "batch_size": q, "replicate": rep,
                  "sample_seed": seed, "execution_order": order,
                  "absolute_error_frobenius": error,
                  "relative_error_frobenius": error / true_norms[plan],
                  "fw_decision_error": decision_error,
                  "sampling_inclusive_seconds": inclusive_seconds, "kernel_seconds": kernel_seconds}
        group = groups[(plan, oracle, q)]
        group["count"] += 1
        delta = estimate - group["mean"]
        group["mean"] += delta / group["count"]
        group["m2"] += float((delta * (estimate - group["mean"])).sum())
        group["rows"].append(record)
        rows.append(record)
        diagnostic_seconds += time.perf_counter() - diagnostic_start
    summaries, mean_arrays = [], {}
    for (plan, oracle, q), group in groups.items():
        count = group["count"]
        bias = float((group["mean"] - true_gradients[plan]).norm())
        variance = group["m2"] / (count - 1)
        summary = {"plan": plan, "oracle": oracle, "batch_size": q, "evaluations": count,
                   "true_gradient_frobenius": true_norms[plan],
                   "empirical_mean_bias_frobenius": bias,
                   "empirical_mean_relative_bias_frobenius": bias / true_norms[plan],
                   "sample_variance_frobenius": variance,
                   "sample_variance_mean_per_entry": variance / plans[plan].numel(),
                   "empirical_mean_noise_rms_frobenius": (variance / count) ** 0.5}
        for key in ("absolute_error_frobenius", "relative_error_frobenius", "fw_decision_error", "sampling_inclusive_seconds", "kernel_seconds"):
            summary[key] = distribution([row[key] for row in group["rows"]])
        summaries.append(summary)
        mean_arrays[f"{plan}__{oracle}__q{q}"] = group["mean"].cpu().numpy()
    exact = {name: {"evaluations": args.repetitions, "seconds": distribution([row["seconds"] for row in exact_rows if row["plan"] == name])} for name in PLAN_NAMES}
    serialization_start = time.perf_counter()
    for path, records in ((run_dir / "evaluations.csv", rows), (run_dir / "exact_gradient_timings.csv", exact_rows)):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    np.savez_compressed(run_dir / "mean_estimates.npz", **mean_arrays,
                        **{f"true__{name}": G.cpu().numpy() for name, G in true_gradients.items()})
    serialization_seconds = time.perf_counter() - serialization_start
    plot_start = time.perf_counter()
    figure_paths = [] if args.no_plots else plot_results(summaries, exact, figure_dir, args.device, args.dtype)
    figure_seconds = time.perf_counter() - plot_start
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")), cpu_model)
    report = {
        "status": f"completed_{args.device}_preliminary",
        "a100_validation": False,
        "a100_validation_note": "Fixed-plan diagnostics alone do not validate a complete A100 optimizer experiment.",
        "oracle_selection_frozen": False,
        "scope": "Protocol section 6 fixed-plan oracle diagnostics; no optimizer or theory schedule.",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label, "command": sys.argv,
        "instance": {"path": str(instance), "npz_sha256": sha256(instance), "manifest": manifest,
                     "verified_array_sha256": hashes, "planted_permutation_loaded": False},
        "settings": {"dtype": args.dtype, "canonical_cost_dtype": str(arrays["A"].dtype),
                     "device": args.device, "feasibility_tolerance": feasibility_tolerance,
                     "amp": False, "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "threads": torch.get_num_threads(),
                     "interop_threads": torch.get_num_interop_threads(),
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "matmul_precision": torch.get_float32_matmul_precision(),
                     "batches": args.batches, "evaluations_per_configuration": args.repetitions,
                     "warmup_per_configuration": args.warmup,
                     "initialization_seed": args.initialization_seed,
                     "stochastic_seed": args.stochastic_seed, "execution_order_seed": args.order_seed},
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "hostname": platform.node(), "cpu_model": cpu_model,
                        "cuda_runtime": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "thread_environment": {v: os.environ[v] for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "source": {"git_commit_at_run": source_revision(root),
                   "git_porcelain_at_run": source_dirty_status(root),
                   "sha256": {name: sha256(root / name) for name in ("oracle_diagnostics.py", "gw_core.py", "PROTOCOL.md")}},
        "plans": {"definition": {"product": "a b^T", "mixed_permutation": "0.9 a b^T + 0.1 R/n, independent permutation R", "concentrated_permutation": "independent scaled permutation R/n"},
                  "initializations": plan_init, "feasibility": feasibility},
        "methodology": {
            "seed_derivation": "SeedSequence([stochastic_seed, plan_id, oracle_id, batch_size, replicate]); two uint32 words combined and masked to 63 bits. Each evaluation uses a fresh torch.Generator on the selected device.",
            "timing": "perf_counter_ns with CUDA synchronization before sampling and after the complete oracle when using CUDA; interleaved shuffled evaluations after per-configuration warmup. Sampling-inclusive includes RNG draws and oracle, excludes per-evaluation generator setup, diagnostics and file I/O. Kernel timing reuses the just-drawn sample: CPU perf_counter_ns or CUDA stream event elapsed time. CUDA event markers add small instrumentation overhead to inclusive time but no intermediate sampling barrier is inserted. Exact gradient uses identical P and cached A^2/B^2 on same dtype/backend/thread policy.",
            "variance": "sum ||Ghat - sample_mean||_F^2 / (repetitions-1), via Welford; this is an unbiased sample estimator of total Frobenius variance.",
            "mean_bias": "||sample_mean - G||_F is empirical finite-sample mean error, not evidence of theoretical bias. sqrt(sample_variance/repetitions) records its sampling-noise scale.",
            "decision_error": "<G,row_lmo(Ghat,a)> - sum_i a_i min_j G_ij; argmin uses the first index on ties. Values are not clipped.",
            "anchor_caveat": "Supplied Python-loop reference anchor implementation; timing is not a production implementation speedup comparison.",
            "independence": "Plan permutations, oracle draws and evaluation ordering use separate seed streams. Frozen planted correspondence never enters plan construction or evaluation.",
        },
        "time_breakdown_seconds": {"preparation_and_initial_exact_diagnostics": preparation_seconds,
                                   "warmup": warmup_seconds,
                                   "sampling_inclusive_oracles": sum(row["sampling_inclusive_seconds"] for row in rows),
                                   "timed_exact_gradients": sum(row["seconds"] for row in exact_rows),
                                   "error_and_moment_diagnostics": diagnostic_seconds,
                                   "raw_serialization": serialization_seconds,
                                   "figures": figure_seconds,
                                   "total_before_summary_serialization": time.perf_counter() - start},
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 ** 2),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "exact_gradient": exact, "configurations": summaries,
        "artifacts": {"raw_evaluations": str(run_dir / "evaluations.csv"),
                      "raw_exact_timings": str(run_dir / "exact_gradient_timings.csv"),
                      "mean_estimates": str(run_dir / "mean_estimates.npz"), "figures": figure_paths},
    }
    report_path = report_dir / f"{args.label}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [f"# {args.device.upper()} preliminary oracle diagnostics", "", f"Instance: `{instance.name}`; n={len(problem.a)}; {args.dtype}; one CPU thread; {args.repetitions} independent evaluations per configuration.", "", "These fixed-plan diagnostics do not validate a complete A100 optimizer experiment, select a frozen default oracle, or implement the missing theorem-specific optimizer.", "", "| Plan | Oracle | q | Mean relative error | Mean absolute error | Empirical mean bias | Frobenius variance | Median decision error | Median time (ms) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['plan']} | {s['oracle']} | {s['batch_size']} | {s['relative_error_frobenius']['mean']:.4g} | {s['absolute_error_frobenius']['mean']:.4g} | {s['empirical_mean_bias_frobenius']:.4g} | {s['sample_variance_frobenius']:.4g} | {s['fw_decision_error']['median']:.4g} | {1000*s['sampling_inclusive_seconds']['median']:.4g} |")
    lines += ["", "Exact-gradient median time (ms): " + "; ".join(f"{name}: {1000*entry['seconds']['median']:.4g}" for name, entry in exact.items()) + ".", "", "Empirical mean bias is the norm of a finite Monte Carlo mean error, not a demonstration of estimator bias. The JSON includes its sampling-noise scale, full mean/median/IQR distributions, plan seeds, source hashes and timing definitions. Anchor timings are for the supplied Python-loop correctness reference. Only saved A/B/a/b are loaded; the planted matching is not used.", "", f"Raw evaluations and reproducible evaluation seeds: `runs/{args.label}/evaluations.csv`.", f"Structured report: `reports/{args.label}.json`."]
    (report_dir / f"{args.label}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(report_path), "stochastic_evaluations": len(rows),
                      "exact_timing_evaluations": len(exact_rows),
                      "elapsed_seconds": time.perf_counter() - start,
                      "status": report["status"]}, indent=2))


if __name__ == "__main__":
    with torch.no_grad():
        main()
