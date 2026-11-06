"""Independent CPU float64 check of the noiseless development control only.

This script never reads held-out instances or planted permutations, never runs
an optimizer trajectory, and does not import any benchmark numerical kernel.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Set thread controls before loading NumPy/SciPy; no extra runtime dependency.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_ONLY = tuple(
    f"metric{n}_s{seed}.npz" for n in (128, 512) for seed in (1000, 1001, 1002)
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(path):
    if path.name not in DEVELOPMENT_ONLY or path.parent.resolve() != (ROOT / "data").resolve():
        raise ValueError("Only the six explicitly whitelisted noiseless development arrays may be read.")
    with np.load(path, allow_pickle=False) as archive:
        # Deliberately never access the planted permutation/true_target fields.
        A, B, a, b = (np.array(archive[key], dtype=np.float64) for key in ("A", "B", "a", "b"))
    n = len(a)
    assert A.shape == B.shape == (n, n)
    assert np.array_equal(A, A.T) and np.array_equal(B, B.T)
    assert np.array_equal(a, np.full(n, 1 / n)) and np.array_equal(b, a)
    x, y = A @ a, B @ b
    sorted_x, sorted_y = np.argsort(x, kind="stable"), np.argsort(y, kind="stable")
    rank_map = np.empty(n, dtype=int)
    rank_map[sorted_x] = sorted_y
    rank_cost = -np.outer(x, y)
    G = 2 * ((A * A @ a)[:, None] + (B * B @ b)[None, :]) + 4 * rank_cost
    ri, cj = linear_sum_assignment(G)
    assignment_map = np.empty(n, dtype=int)
    assignment_map[ri] = cj

    def discrepancy(mapping):
        delta = A - B[np.ix_(mapping, mapping)]
        return {"direct_squared_gw_loss": float(np.square(delta).mean()),
                "max_matched_cost_error": float(np.abs(delta).max())}

    f0 = float(a @ (A * A) @ a + b @ (B * B) @ b - 2 * (a @ A @ a) * (b @ B @ b))
    fs = discrepancy(assignment_map)["direct_squared_gw_loss"]
    cross = float(a @ (A * A) @ a + b @ (B * B) @ b - 2 * np.mean(x * y[assignment_map]))
    linear = 2 * (cross - f0)
    quadratic = f0 - 2 * cross + fs
    candidates = [(0., f0), (1., fs)]
    if quadratic > 0:
        gamma = -linear / (2 * quadratic)
        if 0 < gamma < 1:
            candidates.append((gamma, (1-gamma)**2*f0 + 2*gamma*(1-gamma)*cross + gamma**2*fs))
    chosen = min(candidates, key=lambda item: item[1])
    degree_gap = np.diff(x[sorted_x])
    # NumPy float32 means are a precision diagnostic, not a CUDA reduction replica.
    x32 = A.astype(np.float32).mean(axis=1, dtype=np.float32)
    return {"instance": path.name, "instance_sha256": sha(path), "n": n,
            "source_storage_dtype": "float32", "computation_dtype": "float64",
            "degree_equal_pairs_float64": int(np.count_nonzero(degree_gap == 0)),
            "minimum_degree_spacing_float64": float(degree_gap.min()),
            "degree_equal_pairs_numpy_float32": int(n - np.unique(x32).size),
            "maximum_sorted_degree_mismatch_float64": float(np.max(np.abs(x[sorted_x]-y[sorted_y]))),
            "rank_assignment": discrepancy(rank_map),
            "full_gradient_assignment": discrepancy(assignment_map),
            "assignment_matches_rank_sorting": bool(np.array_equal(assignment_map, rank_map)),
            "product_loss_float64": f0,
            "segment_bilinear_cross_term_float64": cross,
            "segment_linear_coefficient_float64": linear,
            "segment_quadratic_coefficient_float64": quadratic,
            "independent_global_segment_minimizer": chosen[0],
            "selected_segment_loss_float64": chosen[1]}


def main():
    records = [check(ROOT / "data" / name) for name in DEVELOPMENT_ONLY]
    result = {"scope": "Independent CPU float64 analytical control; only noiseless development seeds1000–1002 at n128/512; no held-out arrays or planted labels read; no benchmark timing claims",
              "script_sha256": sha(Path(__file__).resolve()), "numpy": np.__version__, "scipy": scipy.__version__,
              "blas_threads": 1, "records": records}
    target = ROOT / "reports" / "noiseless_control.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"record": str(target), "instances": len(records),
                      "all_full_gradient_assignments_match_costs_exactly": all(r["full_gradient_assignment"]["direct_squared_gw_loss"] == 0 for r in records),
                      "all_segment_minimizers_one": all(r["independent_global_segment_minimizer"] == 1 for r in records)}, indent=2))


if __name__ == "__main__":
    main()
