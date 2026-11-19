#!/usr/bin/env python3
"""Port of the author's SPRING-SARAH sparse-NMF CODE, a reproduction control.

This deliberately preserves empirical code details that differ from the
theorem: shuffled fixed blocks, deterministic outer refresh, one warmup outer
cycle, and the source hard-threshold/clipping order. It is NOT the comparator
for the spectral-ball/convex-sparsity problem.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

if __name__ == "__main__":
    # BLAS reads these at NumPy import, so resolve this option first.
    _thread_parser = argparse.ArgumentParser(add_help=False)
    _thread_parser.add_argument("--threads", type=int, default=1)
    _thread_args, _ = _thread_parser.parse_known_args()
    for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_name] = str(_thread_args.threads)

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "e756ade993f8331ae805fa115a844a77bcdb0bad"


def power_method(matrix: np.ndarray, rng: np.random.Generator, steps: int = 5) -> float:
    """Exactly the source power iteration algebra, with NumPy RNG draws."""
    vector = rng.standard_normal(matrix.shape[1])
    vector /= np.linalg.norm(vector)
    for _ in range(steps):
        product = matrix.T @ (matrix @ vector)
        norm = np.linalg.norm(product)
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Original power iteration has zero/nonfinite norm.")
        vector = product / norm
    value = float(np.linalg.norm(matrix.T @ (matrix @ vector)))
    if not np.isfinite(value) or value <= 0:
        raise FloatingPointError("Invalid Lipschitz estimate in original-code control.")
    return value


def source_outer_threshold(U: np.ndarray, sparsity: int) -> np.ndarray:
    """Refresh step: retain largest raw values, then clip at zero."""
    result = U.copy()
    order = np.argsort(result, axis=0, kind="stable")
    np.put_along_axis(result, order[: U.shape[0] - sparsity], 0.0, axis=0)
    return np.maximum(result, 0.0)


def source_inner_threshold(U: np.ndarray, sparsity: int) -> np.ndarray:
    """MATLAB wthresh(...,'h',tau-th absolute value) uses STRICT >."""
    threshold = np.sort(np.abs(U), axis=0)[-sparsity]
    return np.maximum(U * (np.abs(U) > threshold[None, :]), 0.0)


def fixed_blocks(length: int, count: int) -> list[slice]:
    width = length // count
    if width == 0:
        raise ValueError("Need at least one coordinate per source block.")
    return [slice(i * width, length if i == count - 1 else (i + 1) * width) for i in range(count)]


def outer_cycle(A: np.ndarray, U: np.ndarray, V: np.ndarray, cycle: int,
                rng: np.random.Generator, sparsity: int, blocks: int = 20):
    """One complete author-code outer cycle; cycle is one-based.

    V is stored as images x rank externally; H=V.T follows author algebra.
    Counts include every new/old/refresh residual used for partial gradients.
    """
    U = U.copy()
    H = V.T.copy()
    previous_U = U.copy()
    previous_H = H.copy()
    outer_U = U.copy()
    outer_H = H.copy()
    rows, columns = A.shape
    residual_entries = 0
    gradient_calls = 0
    power_calls = 0
    if cycle > 1:
        grad_H = outer_U.T @ (outer_U @ outer_H - A) / blocks
        estimate = power_method(outer_U, rng) / blocks
        power_calls += 1
        previous_H = outer_H.copy()
        H = np.maximum(outer_H - (0.5 / estimate) * grad_H, 0.0)
        grad_U = (outer_U @ H - A) @ H.T / blocks
        estimate = power_method(H, rng) / blocks
        power_calls += 1
        previous_U = outer_U.copy()
        U = source_outer_threshold(outer_U - (0.5 / estimate) * grad_U, sparsity)
        residual_entries += 2 * rows * columns
        gradient_calls += 2

    row_order = rng.permutation(blocks)
    column_order = rng.permutation(blocks)
    # The original code computes these two unused estimates. Retain the draws.
    power_method(outer_U, rng)
    power_method(outer_H, rng)
    power_calls += 2
    row_blocks = fixed_blocks(rows, blocks)
    column_blocks = fixed_blocks(columns, blocks)
    for row_id, column_id in zip(row_order, column_order):
        index = row_blocks[int(row_id)]
        current = U[index, :]
        previous = previous_U[index, :]
        observations = A[index, :]
        estimate = power_method(current, rng)
        power_calls += 1
        new_gradient = current.T @ (current @ H - observations)
        if cycle == 1:
            grad_H = new_gradient
            multiplier = 1
        else:
            old_gradient = previous.T @ (previous @ previous_H - observations)
            grad_H = new_gradient - old_gradient + grad_H
            multiplier = 2
        residual_entries += multiplier * observations.size
        gradient_calls += multiplier
        previous_H = H.copy()
        H = np.maximum(H - (0.5 / estimate) * grad_H, 0.0)

        index = column_blocks[int(column_id)]
        current = H[:, index]
        previous = previous_H[:, index]
        observations = A[:, index]
        estimate = power_method(current, rng)
        power_calls += 1
        new_gradient = (U @ current - observations) @ current.T
        if cycle == 1:
            grad_U = new_gradient
            multiplier = 1
        else:
            old_gradient = (previous_U @ previous - observations) @ previous.T
            grad_U = new_gradient - old_gradient + grad_U
            multiplier = 2
        residual_entries += multiplier * observations.size
        gradient_calls += multiplier
        previous_U = U.copy()
        U = source_inner_threshold(U - (0.5 / estimate) * grad_U, sparsity)
    if not np.isfinite(U).all() or not np.isfinite(H).all():
        raise FloatingPointError("Nonfinite factor in original-code control.")
    counts = dict(residual_entries=residual_entries, partial_gradient_calls=gradient_calls,
                  power_method_calls=power_calls)
    return U, np.ascontiguousarray(H.T), counts


def diagnostic(A, U, V, cycle, elapsed, counts):
    residual = U @ V.T - A
    loss = 0.5 * float(np.sum(residual * residual))
    return {
        "outer_cycle": cycle,
        "source_reported_epochs": 2 * cycle,
        "optimizer_seconds": elapsed,
        "loss_half_squared_frobenius": loss,
        "log10_loss": float(np.log10(loss)),
        "relative_reconstruction_error": float(np.sqrt(2 * loss / np.sum(A * A))),
        "residual_entries_evaluated": counts["residual_entries"],
        "equivalent_full_residual_passes": counts["residual_entries"] / A.size,
        "partial_gradient_calls": counts["partial_gradient_calls"],
        "power_method_calls": counts["power_method_calls"],
        "dictionary_max_nonzeros_per_column": int(np.count_nonzero(U, axis=0).max()),
        "negative_entries": int(np.count_nonzero(U < 0) + np.count_nonzero(V < 0)),
    }


def atomic_json(path: Path, payload: dict):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(path)


def save_checkpoint(path, U, V, state):
    partial = path.with_suffix(".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, U=U, V=V, state=json.dumps(state))
    partial.replace(path)


def run(args):
    data_path = ROOT / "data" / "processed" / f"{args.dataset}_unitnorm.npz"
    data = np.load(data_path, allow_pickle=False)
    A = data["A"]
    dataset_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_cycles = args.source_epochs // 2
    sparsity = A.shape[0] // 4
    config = {
        "method": "SPRING_SARAH_original_author_code_control",
        "dataset": args.dataset,
        "data_path": str(data_path),
        "data_sha256": dataset_sha,
        "source_commit": SOURCE_COMMIT,
        "seed": args.seed,
        "rng": "NumPy PCG64; fixed author initial factors, not MATLAB bitwise RNG reproduction",
        "source_epochs": args.source_epochs,
        "target_outer_cycles": target_cycles,
        "row_column_blocks": 20,
        "sparsity_limit": sparsity,
        "rank": int(data["U0"].shape[1]),
        "threads": args.threads,
        "theorem_matched": False,
        "scope": "Author-code control only: l0 nonnegative NMF, no spectral bounds.",
        "source_differences": [
            "Cast bundled uint8 pixels to FP64 before the intended per-image normalization; the MATLAB demo omits this required cast.",
            "Different PRNG draws from MATLAB, same sampling operations and saved initial factors.",
            "Atomic resumable checkpoints and diagnostic evaluations outside optimizer timing.",
            "Explicit residual-entry work accounting includes refresh and difference gradients.",
        ],
        "source_details_preserved": [
            "Each image divided by its L2 norm.",
            "One plain-SGD outer cycle, then one deterministic full refresh per outer cycle.",
            "20 shuffled contiguous row blocks and 20 independently shuffled contiguous column blocks.",
            "5 power iterations, including source unused estimate calls; steps 0.5/L.",
            "Inner threshold uses absolute tau-th magnitude and strict greater-than, then nonnegative clipping.",
            "Refresh threshold keeps tau largest raw values, then nonnegative clipping.",
            "Reported source epoch coordinate equals twice the outer-cycle count.",
        ],
    }
    checkpoint = output / "checkpoint.npz"
    csv_path = output / "metrics.csv"
    rng = np.random.default_rng(args.seed)
    if args.resume:
        loaded = np.load(checkpoint, allow_pickle=False)
        state = json.loads(str(loaded["state"]))
        for key in ("dataset", "data_sha256", "seed", "source_commit"):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume mismatch: {key}")
        U, V = loaded["U"], loaded["V"]
        rng.bit_generator.state = state["rng_state"]
        completed = state["outer_cycle"]
        elapsed = state["optimizer_seconds"]
        counts = state["counts"]
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows = [row for row in rows if int(row["outer_cycle"]) <= completed]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        if checkpoint.exists() or csv_path.exists():
            raise FileExistsError("Output exists; use --resume or choose another output directory.")
        U, V = data["U0"].copy(), data["V0"].copy()
        completed, elapsed = 0, 0.0
        counts = dict(residual_entries=0, partial_gradient_calls=0, power_method_calls=0)
        first = diagnostic(A, U, V, completed, elapsed, counts)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(first))
            writer.writeheader()
            writer.writerow(first)
    atomic_json(output / "config.json", config)
    start = time.perf_counter()
    last = diagnostic(A, U, V, completed, elapsed, counts)
    state = dict(config=config, outer_cycle=completed, optimizer_seconds=elapsed,
                 counts=counts, rng_state=rng.bit_generator.state)
    save_checkpoint(checkpoint, U, V, state)
    atomic_json(output / "status.json", dict(status="running", **last))
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(last))
        for cycle in range(completed + 1, target_cycles + 1):
            before = time.perf_counter()
            U, V, new_counts = outer_cycle(A, U, V, cycle, rng, sparsity)
            elapsed += time.perf_counter() - before
            for name in counts:
                counts[name] += new_counts[name]
            completed = cycle
            last = diagnostic(A, U, V, cycle, elapsed, counts)
            writer.writerow(last)
            handle.flush()
            state = dict(config=config, outer_cycle=completed, optimizer_seconds=elapsed,
                         counts=counts, rng_state=rng.bit_generator.state)
            stop = args.max_seconds is not None and time.perf_counter() - start >= args.max_seconds
            if cycle % args.checkpoint_every == 0 or cycle == target_cycles or stop:
                save_checkpoint(checkpoint, U, V, state)
                atomic_json(output / "status.json", dict(status="running", **last))
                print(json.dumps(last), flush=True)
            if stop:
                break
    status = "completed" if completed >= target_cycles else "checkpointed"
    save_checkpoint(checkpoint, U, V, state)
    atomic_json(output / "status.json", dict(status=status, **last))
    print(json.dumps(dict(status=status, **last)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["orl64", "yaleb32"], required=True)
    parser.add_argument("--source-epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.source_epochs < 2 or args.source_epochs % 2:
        parser.error("--source-epochs must be a positive even number (2 per outer cycle).")
    if args.checkpoint_every < 1 or args.threads < 1:
        parser.error("Checkpoint interval and thread count must be positive.")
    run(args)


if __name__ == "__main__":
    main()
