"""Reproducible CPU/float64 Stage A baseline checks; NOT a timing benchmark.

Run after creating data/toy16_s1000.npz with make_instances.py. All seven
solvers use the same frozen costs and product initialization. No unpublished
method, parameter selection, theorem claim, or performance claim is implied.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone

# Set before numerical-library imports, including imports in child processes.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import ot
import torch
from gw_core import GWProblem, marginal_residuals, original_fw_gap, round_transport

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serializable(obj), indent=2, allow_nan=False) + "\n")


def archive_pot(provenance):
    functions = {
        "entropic_gromov_wasserstein": ot.gromov.entropic_gromov_wasserstein,
        "sampled_gromov_wasserstein": ot.gromov.sampled_gromov_wasserstein,
        "gromov_wasserstein": ot.gromov.gromov_wasserstein,
        "init_matrix": ot.gromov.init_matrix,
        "gwloss": ot.gromov.gwloss,
        "gwggrad": ot.gromov.gwggrad,
        "sinkhorn": ot.bregman.sinkhorn,
    }
    result = {"pot_version": ot.__version__, "signatures": {}, "source_modules": {}}
    for name, function in functions.items():
        result["signatures"][name] = str(inspect.signature(function))
        source = Path(inspect.getsourcefile(function))
        module = function.__module__
        if module not in result["source_modules"]:
            target = provenance / ("pot_" + module.replace(".", "_") + ".py")
            shutil.copyfile(source, target)
            result["source_modules"][module] = {
                "installed_path": str(source), "archived_path": str(target.relative_to(ROOT)),
                "sha256": sha256(target),
            }
    write_json(provenance / "pot_installed_api.json", result)
    return result


def independent_evaluation(problem, raw):
    """Use actual marginals for both loss and gradient; repair a copy only."""
    P = torch.as_tensor(np.array(raw, copy=True), dtype=torch.float64)
    if not bool(torch.isfinite(P).all()) or bool((P < 0).any()):
        raise ValueError("Nonfinite or negative output: rejected without clamping.")
    raw_copy = P.clone()
    started = time.perf_counter()
    rounded = round_transport(P, problem.a, problem.b)
    repair_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics = {
        "raw_smooth_loss": problem.loss(P).item(),
        "rounded_feasible_loss": problem.loss(rounded).item(),
        "raw_residuals": marginal_residuals(P, problem.a, problem.b),
        "rounded_residuals": marginal_residuals(rounded, problem.a, problem.b),
        "repair_entrywise_l1": (rounded - P).abs().sum().item(),
        "original_fw_gap_at_rounded": original_fw_gap(problem, rounded),
        "zero_entries_raw": int((P == 0).sum().item()),
        "repair_seconds": repair_seconds,
    }
    metrics["diagnostic_seconds"] = time.perf_counter() - started
    assert torch.equal(P, raw_copy), "The evaluator mutated its raw input."
    assert metrics["rounded_residuals"]["both_l1"] <= 1e-12
    assert metrics["original_fw_gap_at_rounded"] >= -1e-10
    assert metrics["raw_smooth_loss"] >= -1e-10
    assert metrics["rounded_feasible_loss"] >= -1e-10
    return metrics, rounded.numpy()


def check_pot_conventions(problem, arrays):
    A, B, a, b = (arrays[key] for key in ("A", "B", "a", "b"))
    n = len(a)
    rng = np.random.default_rng(901000)  # evaluation-only, never the planted permutation
    assignment = np.zeros((n, n))
    assignment[np.arange(n), rng.permutation(n)] = 1 / n
    product = np.outer(a, b)
    plans = {"product": product, "independent_mixed_permutation": .9 * product + .1 * assignment,
             "independent_concentrated_permutation": assignment}
    constC, hC1, hC2 = ot.gromov.init_matrix(A, B, a, b, loss_fun="square_loss")
    rows = []
    for name, P in plans.items():
        tensor = torch.as_tensor(P, dtype=torch.float64)
        own_loss = problem.loss(tensor).item()
        own_gradient = problem.gradient(tensor).numpy()
        pot_loss = float(ot.gromov.gwloss(constC, hC1, hC2, P))
        pot_gradient = ot.gromov.gwggrad(constC, hC1, hC2, P)
        np.testing.assert_allclose(own_loss, pot_loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(own_gradient, pot_gradient, rtol=1e-12, atol=1e-12)
        rows.append({"plan": name, "full_loss": own_loss, "pot_square_loss": pot_loss,
                     "loss_absolute_difference": abs(own_loss - pot_loss),
                     "gradient_max_absolute_difference": float(np.max(np.abs(own_gradient - pot_gradient))),
                     "residuals": marginal_residuals(tensor, problem.a, problem.b)})
    return {"passed": True, "convention": "full squared-loss GW; no half, root, entropy, or scale adjustment",
            "note": "POT feasible-marginal shortcuts are checked only at feasible plans; all solver outputs use GWProblem.",
            "plans": rows}


def run_reference(method, eta, instance, problem, out, common):
    run_dir = out / ("reference_" + method.replace("-", "_"))
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "run_reference.py"), "--instance", str(instance),
               "--method", method, "--steps", "5", "--seed", "0", "--batch", "32",
               "--inner-tol", "1e-9", "--inner-max", "5000", "--init", "product",
               "--device", "cpu", "--dtype", "float64", "--threads", "1", "--eval-every", "1",
               "--out", str(run_dir)]
    if eta is not None:
        command += ["--eta", str(eta)]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    elapsed = time.perf_counter() - started
    (run_dir / "stdout.log").write_text(proc.stdout)
    (run_dir / "stderr.log").write_text(proc.stderr)
    config = dict(common, method=method, implementation="provided run_reference.py", command=command,
                  eta=eta, eta_power=0, outer_steps_requested=5, inner_tolerance=1e-9, inner_max=5000,
                  wall_seconds_including_process_startup=elapsed, returncode=proc.returncode,
                  smoothing=None, momentum=None,
                  torch_deterministic_algorithms_in_child="default False; CPU float64 single-thread smoke check")
    write_json(run_dir / "validation_config.json", config)
    if proc.returncode:
        raise RuntimeError(f"{method}: child exit {proc.returncode}; see {run_dir}/stderr.log")
    with np.load(run_dir / "final.npz") as data:
        raw, stored_rounded = data["P_raw"], data["P_rounded"]
    metrics, rounded = independent_evaluation(problem, raw)
    np.testing.assert_allclose(rounded, stored_rounded, atol=1e-13, rtol=1e-13)
    with (run_dir / "trace.csv").open() as handle:
        trace = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in trace] == list(range(6))
    final = trace[-1]
    for trace_name, metric_name in (("raw_loss", "raw_smooth_loss"), ("rounded_loss", "rounded_feasible_loss"),
                                     ("original_fw_gap", "original_fw_gap_at_rounded")):
        np.testing.assert_allclose(float(final[trace_name]), metrics[metric_name], atol=1e-12, rtol=1e-12)
    inner_failures = sum(row["inner_converged"] != "True" for row in trace[1:])
    metrics.update(inner_stopping_failures=inner_failures,
                   total_logged_inner_iterations=sum(int(row["inner_iterations"]) for row in trace[1:]),
                   stderr_nonempty=bool(proc.stderr.strip()),
                   underflow_entries_reported=[int(row["underflow_entries"]) for row in trace[1:]])
    write_json(run_dir / "independent_evaluation.json", metrics)
    assert inner_failures == 0, f"{method}: {inner_failures} inner stopping failures"
    return {"method": run_dir.name, "status": "passed", "metrics": metrics,
            "warnings": "see stderr.log" if proc.stderr.strip() else [], "output": str(run_dir.relative_to(ROOT))}


def run_native(name, function, kwargs, problem, out, common):
    run_dir = out / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(common, method=name, implementation="installed POT native implementation",
                  parameters={k: ("elementwise (x-y)**2" if callable(v) else v)
                              for k, v in kwargs.items() if k not in ("C1", "C2", "p", "q", "G0")},
                  G0="native product initialization" if name == "pot_sampled" else "explicit outer(a,b)",
                  outer_steps_requested=5, smoothing=None, momentum=None,
                  native_per_iteration_counts="not instrumented; this smoke check does not claim full timing/accounting coverage")
    write_json(run_dir / "config.json", config)
    stdout, stderr = io.StringIO(), io.StringIO()
    captured = []
    try:
        with warnings.catch_warnings(record=True) as captured, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            warnings.simplefilter("always")
            started = time.perf_counter()
            raw, native_log = function(**kwargs)
            solver_seconds = time.perf_counter() - started
    finally:
        (run_dir / "stdout.log").write_text(stdout.getvalue())
        (run_dir / "stderr.log").write_text(stderr.getvalue())
        warning_records = [{"category": item.category.__name__, "message": str(item.message),
                            "filename": item.filename, "lineno": item.lineno} for item in captured]
        write_json(run_dir / "warnings.json", warning_records)
    metrics, rounded = independent_evaluation(problem, raw)
    np.savez(run_dir / "final.npz", P_raw=np.asarray(raw), P_rounded=rounded)
    write_json(run_dir / "native_log.json", native_log)
    metrics.update(solver_call_seconds=solver_seconds,
                   native_log_warning="Native loss estimates and prescribed-marginal losses are not independent evaluation.")
    write_json(run_dir / "independent_evaluation.json", metrics)
    return {"method": name, "status": "passed", "metrics": metrics, "warnings": warning_records,
            "output": str(run_dir.relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="data/toy16_s1000.npz")
    args = parser.parse_args()
    instance = Path(args.instance).resolve()
    out, reports, provenance = ROOT / "runs/stage_a", ROOT / "reports", ROOT / "provenance"
    for directory in (out, reports, provenance):
        directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    with np.load(instance) as loaded:
        arrays = {key: np.array(loaded[key], dtype=np.float64) for key in ("A", "B", "a", "b")}
        hashes = {key + "_sha256": hashlib.sha256(loaded[key].tobytes()).hexdigest() for key in ("A", "B")}
    assert arrays["A"].shape == (16, 16) and arrays["B"].shape == (16, 16), "This bounded Stage A script requires n=m=16."
    manifest = json.loads(instance.with_suffix(".json").read_text())
    assert manifest["seed"] == 1000 and manifest["noise"] == 0
    assert all(manifest[key] == value for key, value in hashes.items())
    problem = GWProblem(*(torch.as_tensor(arrays[key], dtype=torch.float64) for key in ("A", "B", "a", "b")))
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    common = {"source_revision": git_revision.stdout.strip() if git_revision.returncode == 0 else None,
              "source_tree_status_at_start": git_status.stdout.splitlines() if git_status.returncode == 0 else None,
              "instance": str(instance.relative_to(ROOT)), "instance_sha256": sha256(instance), **hashes,
              "problem_seed": 1000, "stochastic_seed": 0, "initialization": "product outer(a,b)",
              "initialization_seed": None, "device": "cpu", "dtype": "float64", "threads": 1,
              "batch": 32, "objective": "full_squared_discrepancy_no_half_no_entropy",
              "peak_memory": "not measured in correctness-only smoke checks", "performance_claim": False}
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(), "stage": "A, baseline portion only",
               "scope": "CPU n=16, five outer steps; settings prespecified, not selected winners; no performance conclusion",
               "common": common,
               "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "POT")},
               "python": sys.version, "platform": platform.platform(), "pot_api": archive_pot(provenance),
               "source_sha256": {name: sha256(ROOT / name) for name in ("gw_core.py", "run_reference.py", "validate_baselines.py", "PROTOCOL.md")},
               "limits": ["Own-method and theorem-specific checks are outside this baseline-only script.",
                          "Native POT solvers use their own stopping rules and may stop before the cap of five steps.",
                          "POT Sampled GW uses its native conditional estimator; its sample count and epsilon differ from stochastic KL.",
                          "Complete per-oracle accounting, peak memory, synchronized GPU timing and target-device validation await the benchmark harness."],
               "methods": []}
    try:
        summary["pot_conventions"] = check_pot_conventions(problem, arrays)
    except Exception:
        summary["pot_conventions"] = {"passed": False, "traceback": traceback.format_exc()}
    A, B, a, b = (arrays[key] for key in ("A", "B", "a", "b"))
    native_shared = dict(C1=A, C2=B, p=a, q=b, max_iter=5, log=True)
    jobs = [("reference_" + method.replace("-", "_"), lambda method=method, eta=eta: run_reference(method, eta, instance, problem, out, common))
            for method, eta in (("fw", None), ("kl", 10.), ("stochastic-kl", 1.), ("pg", .01))]
    jobs += [
        ("pot_ppa", lambda: run_native("pot_ppa", ot.gromov.entropic_gromov_wasserstein,
             dict(native_shared, loss_fun="square_loss", symmetric=True, G0=np.outer(a, b), solver="PPA",
                  epsilon=.1, tol=1e-12, numItermax=1000, stopThr=1e-9), problem, out, common)),
        ("pot_sampled", lambda: run_native("pot_sampled", ot.gromov.sampled_gromov_wasserstein,
             dict(native_shared, loss_fun=lambda x, y: (x - y) ** 2, nb_samples_grad=32, epsilon=.1,
                  random_state=0), problem, out, common)),
        ("pot_unregularized", lambda: run_native("pot_unregularized", ot.gromov.gromov_wasserstein,
             dict(native_shared, loss_fun="square_loss", symmetric=True, G0=np.outer(a, b),
                  armijo=False, tol_rel=1e-12, tol_abs=1e-12), problem, out, common)),
    ]
    for name, job in jobs:
        try:
            result = job()
        except Exception:
            result = {"method": name, "status": "failed", "traceback": traceback.format_exc()}
        summary["methods"].append(result)
        print(name + ": " + result["status"], flush=True)
        write_json(reports / "baseline_validation.json", summary)
    summary["all_baseline_checks_passed"] = summary["pot_conventions"]["passed"] and all(row["status"] == "passed" for row in summary["methods"])
    summary["elapsed_seconds_including_diagnostics_and_io"] = time.perf_counter() - started
    write_json(reports / "baseline_validation.json", summary)
    lines = ["# Stage A baseline validation", "", summary["scope"], "",
             "Loss and gradients agree with the installed POT squared-loss conventions on three feasible plans: " + str(summary["pot_conventions"]["passed"]), "",
             "| Method | Status | Raw smooth loss | Rounded feasible loss | Raw marginal L1 | Rounded marginal L1 | Original FW gap after rounding |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["methods"]:
        if row["status"] != "passed":
            lines.append(f"| {row['method']} | failed | — | — | — | — | — |")
        else:
            m = row["metrics"]
            lines.append(f"| {row['method']} | passed | {m['raw_smooth_loss']:.9g} | {m['rounded_feasible_loss']:.9g} | {m['raw_residuals']['both_l1']:.3g} | {m['rounded_residuals']['both_l1']:.3g} | {m['original_fw_gap_at_rounded']:.9g} |")
    lines += ["", "These short CPU checks do not rank methods, select parameters, validate GPU behavior, or establish convergence rates.", "",
              "Every output is independently evaluated with actual marginals. Raw and rounded arrays, configurations, native logs, warnings, and installed POT sources are retained.", "",
              "The proposed method and theorem-specific checks are outside this baseline-only validation; see the separate method validation records.", ""]
    (reports / "baseline_validation.md").write_text("\n".join(lines))
    return 0 if summary["all_baseline_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
