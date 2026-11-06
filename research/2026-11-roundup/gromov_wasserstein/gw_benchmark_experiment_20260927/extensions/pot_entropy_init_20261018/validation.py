"""Scheduled numerical gates for entropy and initialization sensitivity.

CPU: tiny independent gradient/transport checks at every entropy and both starts.
CUDA: the same checks and restart; optional actual-size memory/restart check.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module("entropy_sweep_runner_validation", HERE / "runner.py")
reference = module("entropy_independent_reference", ROOT / "extensions/large_pot_frames_20261016/validation.py")
from prox import DistributedCertifiedPOTProx
from checkpoint import CheckpointStore, atomic_json, sha256


def write_fixture(path):
    path.mkdir(parents=True)
    rng = np.random.default_rng(849)
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    arrays = dict(A=(.005*np.linalg.norm(x[:, None]-x[None, :], axis=-1)).astype("float32"),
                  B=(.005*np.linalg.norm(y[:, None]-y[None, :], axis=-1)).astype("float32"),
                  a=np.full(8, 1/8, dtype="float64"), b=np.full(8, 1/8, dtype="float64"))
    for name, value in arrays.items():
        np.save(path / (name + ".npy"), value)
    atomic_json(path / "manifest.json", dict(n=8, m=8,
        array_files_sha256={name+".npy": sha256(path/(name+".npy")) for name in arrays}))


def initial(problem, kind, seed):
    # Construct independently of the runner to test its initial iterate receipt.
    plan = torch.outer(problem.a_local, problem.b)
    if kind == "permutation-mixture":
        plan.mul_(.1)
        permutation = torch.randperm(problem.n, generator=torch.Generator().manual_seed(seed))
        plan[torch.arange(problem.n, device=problem.device), permutation.to(problem.device)] += .9 / problem.n
    return plan


@torch.no_grad()
def tiny_checks(instance, device):
    p, _ = runner.load_problem(instance, device=device, exact_block_rows=4)
    checks = []
    A, B, a, b = [v.cpu().numpy() for v in (p.A, p.B, p.a, p.b)]
    for epsilon in (.001, .0003, .0001, .00003):
        for kind in ("product", "permutation-mixture"):
            plan = initial(p, kind, 901)
            expected = plan.cpu().numpy().copy()
            prox = DistributedCertifiedPOTProx(p, gap_tol=1e-11, certificate_block_rows=4)
            maximum_error = 0.
            for _ in range(3):
                gradient = p.gradient(plan)
                direct = reference.dense_gradient(A, B, expected)
                # Plan errors from preceding transport solves also enter the gradient.
                assert np.max(np.abs(gradient.cpu().numpy()-direct)) <= 1e-11
                plan, _, info = prox(runner.product_log_reference(p), gradient,
                                     eta=1/epsilon, tol=1e-12, check_every=1)
                assert info["converged"] and info["proximal_optimality_certified"]
                expected = reference.dense_sinkhorn(direct, a, b, eta=1/epsilon)
                error = float(np.max(np.abs(plan.cpu().numpy()-expected)))
                assert error <= 2e-9, (epsilon, kind, error)
                maximum_error = max(maximum_error, error)
            checks.append(dict(name="independent_full_gradient_transport", epsilon=epsilon,
                               initialization=kind, max_plan_error=maximum_error))
    # Explicitly exercise low-epsilon log-domain fallback without kernel clipping.
    gradient = torch.zeros((p.n, p.m), device=p.device, dtype=torch.float64)
    gradient[0, :] = 1.
    gradient[:, 0] += 1.
    prox = DistributedCertifiedPOTProx(p, certificate_block_rows=4)
    plan, info = runner.pot_step(p, runner.product_log_reference(p), gradient, prox, epsilon=.00003)
    assert info["sinkhorn_path"] == "original-log-domain-fallback"
    assert torch.max(torch.abs(plan-torch.outer(p.a, p.b))).item() < 1e-10
    checks.append(dict(name="low_entropy_underflow_log_fallback", epsilon=.00003,
                       marginal_l1=info["marginal_l1"],
                       normalized_gap_upper_bound=info["normalized_gap_upper_bound"]))
    return checks


def run_fixture(instance, out, device, *, actual=False):
    """Exercise the actual CLI and check that a two-part trajectory is identical."""
    epsilon = .001 if actual else .00003
    iterations = 2 if actual else 4
    base = [sys.executable, str(HERE / "runner.py"), "--instance", str(instance),
            "--device", device, "--initialization", "permutation-mixture", "--seed", "901",
            "--epsilon", str(epsilon), "--iterations", str(iterations),
            "--checkpoint-every", "1", "--diagnostic-every", "1", "--max-seconds", "10800"]
    split = out / ("actual_restart" if actual else "tiny_restart")
    subprocess.run(base + ["--out", str(split), "--chunk-updates", "1"], check=True)
    part = json.loads((split / "progress.json").read_text())
    assert part["status"] == "checkpointed" and part["completed_updates"] == 1
    subprocess.run(base + ["--out", str(split)], check=True)
    result = json.loads((split / "metadata.json").read_text())
    assert result["status"] == "completed" and result["completed_updates"] == iterations
    assert result["max_memory_reserved_fraction"] < .9
    assert result["trace"][0]["raw_residuals"]["both_l1"] < 1e-12
    if not actual:
        full = out / "tiny_uninterrupted"
        subprocess.run(base + ["--out", str(full)], check=True)
        original = json.loads((full / "metadata.json").read_text())
        for name in ("trace", "outer_trace", "inner_trace", "counters", "stopping_reason"):
            assert original[name] == result[name], name
        p, _ = runner.load_problem(instance, device="cpu")
        start = initial(p, "permutation-mixture", 901)
        assert abs(result["trace"][0]["raw_loss"]-p.loss(start).item()) < 1e-13
    return dict(name="actual_size_restart" if actual else "exact_tiny_restart",
                epsilon=epsilon, iterations=iterations,
                max_memory_reserved_fraction=result["max_memory_reserved_fraction"],
                final_loss=result["final_loss"], dimensions=result["dimensions"])


def low_entropy_probe(instance, out, device, initialization):
    """Bound conditioning probes; an unfinished solve never becomes a result.

These probes diagnose the new parameter regime. Their numerical or time
limits are reported explicitly and do not discard other sweep settings.
"""
    folder = out / ("low_entropy_probe_" + initialization)
    command = [sys.executable, str(HERE / "runner.py"), "--instance", str(instance),
               "--out", str(folder), "--device", device, "--initialization", initialization,
               "--seed", "901", "--epsilon", ".00003", "--iterations", "2",
               "--diagnostic-every", "1", "--checkpoint-every", "1", "--max-seconds", "900"]
    executed = subprocess.run(command)
    if executed.returncode:
        failure = json.loads((folder / "failure.json").read_text())
        assert failure["failure_kind"] == "inner_certificate_not_met", failure["error"]
        return dict(name="actual_low_entropy_probe", initialization=initialization,
                    epsilon=.00003, status="inner_certificate_not_met",
                    completed_updates=failure["completed_updates"], certificate=failure["certificate"])
    result = json.loads((folder / "metadata.json").read_text())
    assert result["status"] in ("completed", "checkpointed")
    assert result["max_memory_reserved_fraction"] < .9
    record = dict(name="actual_low_entropy_probe", initialization=initialization,
                  epsilon=.00003,
                  status="completed" if result["status"] == "completed" else "time_budget_exhausted",
                  completed_updates=result["completed_updates"], elapsed_seconds=result["elapsed_seconds"],
                  max_memory_reserved_fraction=result["max_memory_reserved_fraction"],
                  inner_trace=result["inner_trace"], counters=result["counters"],
                  checkpoint=result["checkpoint"])
    # Probes are terminal; their accepted traces/identity/manifest remain in
    # metadata.json, but their matrices are not needed for production.
    atomic_json(folder / "probe_receipt.json", record)
    if result["status"] == "checkpointed":
        CheckpointStore(folder / "checkpoints").remove_completed_state()
        record["probe_checkpoint_removed"] = True
        atomic_json(folder / "probe_receipt.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == "cuda":
        torch.cuda.set_device(0)
        assert "A100" in torch.cuda.get_device_name(), "Gate requires target A100 hardware"
    fixture = args.out / "tiny_input"
    write_fixture(fixture)
    checks = tiny_checks(fixture, args.device)
    checks.append(run_fixture(fixture, args.out, args.device))
    if args.instance:
        checks.append(run_fixture(args.instance, args.out, args.device, actual=True))
        for kind in ("product", "permutation-mixture"):
            checks.append(low_entropy_probe(args.instance, args.out, args.device, kind))
    atomic_json(args.out / "checks.json", dict(status="passed", world_size=1,
        device=args.device, dtype="float64", checks=checks,
        source_file_sha256=runner.source_hashes(),
        validation_sha256=sha256(Path(__file__)),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None))


if __name__ == "__main__":
    main()
