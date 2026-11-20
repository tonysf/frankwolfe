"""Frozen matrix-completion instance and independently certified convex reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import cvxpy as cp
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_data():
    path = ROOT / "data.npz"
    if path.exists():
        return
    rng = np.random.default_rng(20261104)
    m = n = 50
    rank = 5
    signal = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
    signal *= .8 / np.max(np.abs(signal))
    noise = rng.normal(scale=.6, size=(m, n))
    Y = signal + noise
    indices = rng.permutation(m * n)
    mask = np.zeros(m * n, dtype=bool)
    mask[indices[:1000]] = True
    mask = mask.reshape(m, n)
    assert mask.any(axis=0).all() and mask.any(axis=1).all()
    np.savez(path, Y=Y, signal=signal, mask=mask, test_mask=~mask,
             tau=np.array(20.), lower=np.array(-1.), upper=np.array(1.),
             data_seed=np.array(20261104), signal_rank=np.array(rank),
             noise_sigma=np.array(.6))


def feasible_scale(x, tau):
    """Scale towards the strictly feasible zero point to satisfy both constraints."""
    nuclear = np.linalg.svd(x, compute_uv=False).sum()
    factor = max(1., nuclear / tau, np.max(np.abs(x))) * (1. + 1e-13)
    return x / factor, factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="SCS", choices=["SCS", "CLARABEL"])
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()
    create_data()
    data = dict(np.load(ROOT / "data.npz"))
    Y, mask, tau = data["Y"], data["mask"], float(data["tau"])
    N = int(mask.sum())
    X = cp.Variable(Y.shape)
    constraints = [cp.normNuc(X) <= tau, X >= -1, X <= 1]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(X[mask] - Y[mask]) / (2 * N)), constraints)
    start = perf_counter()
    if args.solver == "SCS":
        problem.solve(solver=cp.SCS, eps=args.eps, max_iters=100000, verbose=False)
    else:
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=args.eps, tol_gap_rel=args.eps,
                      tol_feas=args.eps, max_iter=300, verbose=False)
    elapsed = perf_counter() - start
    if X.value is None:
        raise RuntimeError(f"Reference solve did not produce a matrix: {problem.status}")
    raw = np.asarray(X.value)
    feasible, scale = feasible_scale(raw, tau)
    def loss(x):
        return float(np.square(x[mask] - Y[mask]).sum() / (2 * N))
    gradient = np.zeros_like(raw)
    gradient[mask] = (raw[mask] - Y[mask]) / N
    Z = np.asarray(constraints[2].dual_value) - np.asarray(constraints[1].dual_value)
    opnorm = float(np.linalg.svd(gradient + Z, compute_uv=False)[0])
    lower_bound = loss(raw) - float(np.sum(gradient * raw)) - tau * opnorm - float(np.abs(Z).sum())
    upper_bound = loss(feasible)
    assert np.linalg.svd(feasible, compute_uv=False).sum() <= tau + 1e-10
    assert np.abs(feasible).max() <= 1 + 1e-12
    assert lower_bound <= upper_bound + 1e-10
    np.save(ROOT / "Xreference.npy", feasible)
    np.save(ROOT / "Xreference_raw.npy", raw)
    np.save(ROOT / "reference_box_dual.npy", Z)
    results = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status=problem.status, solver=args.solver, requested_tolerance=args.eps,
        shape=list(Y.shape), train_entries=N, test_entries=int((~mask).sum()),
        data_seed=int(data["data_seed"]), signal_rank=int(data["signal_rank"]),
        noise_sigma=float(data["noise_sigma"]), tau=tau, lower=-1., upper=1.,
        initial_loss=loss(np.zeros_like(Y)), solver_reported_objective=float(problem.value),
        raw_objective=loss(raw), feasible_objective=upper_bound,
        certified_lower_bound=lower_bound, certified_upper_bound=upper_bound,
        certificate_gap=upper_bound-lower_bound,
        certificate_formula="f(Xr)-<gradient f(Xr),Xr>-tau*operator_norm(gradient f(Xr)+Z)-sum(abs(Z)); Z=dual_upper-dual_lower",
        certificate_scope="Convex tangent lower bound using independently evaluated spectral norm and box support; numerical floating-point certificate, not interval arithmetic.",
        raw_nuclear_norm=float(np.linalg.svd(raw, compute_uv=False).sum()),
        feasible_nuclear_norm=float(np.linalg.svd(feasible, compute_uv=False).sum()),
        raw_max_abs=float(np.abs(raw).max()), feasible_max_abs=float(np.abs(feasible).max()),
        raw_box_violation_frobenius=float(np.linalg.norm(raw - np.clip(raw,-1,1))),
        feasible_scaling_factor=scale,
        nuclear_dual=float(constraints[0].dual_value),
        active_box_entries_at_1e6=int((np.abs(np.abs(raw)-1) <= 1e-6).sum()),
        largest_box_dual=float(np.abs(Z).max()),
        zero_is_strictly_feasible=True,
        signal_nuclear_norm=float(np.linalg.svd(data["signal"],compute_uv=False).sum()),
        signal_max_abs=float(np.abs(data["signal"]).max()),
        observed_values_outside_box=int((np.abs(Y[mask])>1).sum()),
        solve_seconds=elapsed, solver_iterations=int(problem.solver_stats.num_iters),
        versions=dict(numpy=np.__version__,scipy=scipy.__version__,cvxpy=cp.__version__),
        hashes=dict(data=sha(ROOT/"data.npz"),reference_code=sha(__file__),
                    feasible_matrix=sha(ROOT/"Xreference.npy"),box_dual=sha(ROOT/"reference_box_dual.npy")),
        feasible_repair="For X in the nuclear ball, X/max(1,maxabs(X)) is in its intersection with the symmetric box. Reference additionally accounts for small solver nuclear-norm residual.",
        dataset_scope="Synthetic noisy rank-five signal; all observed values include noise, some outside the prescribed fitted-value box. Unobserved values are excluded from optimization.")
    (ROOT/"reference.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2),flush=True)


if __name__ == "__main__":
    main()
