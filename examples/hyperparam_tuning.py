"""
Hyperparameter tuning for SCAD and MCP on the trend-filtered matrix
factorization problem.

Sweeps over:
  - lambda (penalty strength)
  - a (SCAD shape, must be > 2) / gamma (MCP shape, must be > 1)
  - beta_0 (smoothing initialization)

Reports: final relative error, min smoothed gap, final smoothed gap.
Saves a summary table and a heatmap figure.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np

matplotlib.use("Agg")
from itertools import product as cartesian

import matplotlib.pyplot as plt
from frank_wolfe import Frames
from tqdm import tqdm

from examples.trend_filtering_mf import (
    TrendFilteredMFObjective,
    create_spectral_ball_product_lmo,
    finite_diff_matrix,
    generate_piecewise_constant_mf,
    l1_minimal_norm,
    l1_prox,
)

# ─── SCAD and MCP implementations ────────────────────────────────────────────


def make_scad(lam, a):
    assert a > 2
    rho = 1.0 / (a - 1)

    def prox(z, beta):
        assert beta < a - 1, f"Need beta < {a - 1}, got {beta}"
        y = np.abs(z)
        r = np.zeros_like(z)
        m1 = y <= lam * (1 + beta)
        r[m1] = np.maximum(y[m1] - beta * lam, 0)
        m2 = (y > lam * (1 + beta)) & (y <= a * lam)
        r[m2] = ((a - 1) * y[m2] - beta * a * lam) / (a - 1 - beta)
        m3 = y > a * lam
        r[m3] = y[m3]
        return np.sign(z) * r

    def deriv(z):
        t = np.abs(z)
        s = np.sign(z)
        r = np.zeros_like(z)
        m1 = (t > 0) & (t <= lam)
        r[m1] = lam * s[m1]
        m2 = (t > lam) & (t <= a * lam)
        r[m2] = s[m2] * (a * lam - t[m2]) / (a - 1)
        return r

    return prox, deriv, rho


def make_mcp(lam, gamma):
    assert gamma > 1
    rho = 1.0 / gamma

    def prox(z, beta):
        assert beta < gamma, f"Need beta < {gamma}, got {beta}"
        y = np.abs(z)
        r = np.zeros_like(z)
        m1 = (y > beta * lam) & (y <= gamma * lam)
        r[m1] = (y[m1] - beta * lam) / (1 - beta / gamma)
        m2 = y > gamma * lam
        r[m2] = y[m2]
        return np.sign(z) * r

    def deriv(z):
        t = np.abs(z)
        s = np.sign(z)
        r = np.zeros_like(z)
        m1 = (t > 0) & (t <= gamma * lam)
        r[m1] = s[m1] * (lam - t[m1] / gamma)
        return r

    return prox, deriv, rho


# ─── Problem setup ────────────────────────────────────────────────────────────

m, n, r = 100, 100, 50
n_steps = 2000
margin = 1.05
seed = 42

X_star, U_star, V_star, tau_U, tau_V = generate_piecewise_constant_mf(
    m, n, r, n_blocks=5, margin=margin, seed=seed
)
X_star_fro = np.linalg.norm(X_star, "fro")
D = finite_diff_matrix(m)
DU_star = D @ U_star

lmo = create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r)
x0 = np.zeros(m * r + n * r)

print(f"Problem: m={m}, n={n}, r={r}")
print(f"||X*||_F = {X_star_fro:.2f}")
print(f"||D U*||_1 = {np.sum(np.abs(DU_star)):.4f}")
print(f"nnz(D U*) = {np.count_nonzero(DU_star)} / {DU_star.size}")
print()


# ─── Hyperparameter grids ────────────────────────────────────────────────────

lam_values = [0.5]
beta0_values = [1.0]

# SCAD: a > 2, and we need beta_k < a-1 for all k.
# beta_k = beta_0 / (k+1)^{1/4}, so beta_max = beta_0.
# Constraint: beta_0 < a - 1.
scad_a_values = np.linspace(2.1, 100, 32)

# MCP: gamma > 1, and we need beta_k < gamma for all k.
# Constraint: beta_0 < gamma.
mcp_gamma_values = np.linspace(1.1, 100, 32)


# ─── Run sweeps ──────────────────────────────────────────────────────────────


def run_single(prox_fn, deriv_fn, beta0, n_steps):
    """Run FRAMES and return metrics."""
    obj = TrendFilteredMFObjective(X_star, m, n, r)
    obj.minimal_norm_selection = deriv_fn
    try:
        frames = Frames(obj, lmo, prox_fn, "lipschitz")
        frames.run(x0, beta0=beta0, n_steps=n_steps)
    except (AssertionError, Exception) as e:
        return None
    avg_func_vals = np.cumsum(frames.func_vals) / np.arange(1, n_steps + 1)
    rel_err = np.sqrt(2.0 * max(avg_func_vals[-1], 0)) / X_star_fro
    min_gap = np.min(frames.gaps)
    final_gap = frames.gaps[-1]
    return dict(rel_err=rel_err, min_gap=min_gap, final_gap=final_gap)


# ── L1 baseline ──
print("=" * 70)
print("L1 baseline")
print("=" * 70)
print(f"{'beta0':>8s}  {'rel_err':>10s}  {'min_gap':>12s}  {'final_gap':>12s}")
print("-" * 50)
l1_results = {}
for beta0 in beta0_values:
    res = run_single(l1_prox, l1_minimal_norm, beta0, n_steps)
    if res:
        l1_results[beta0] = res
        print(
            f"{beta0:8.2f}  {res['rel_err']:10.4f}  {res['min_gap']:12.2e}  {res['final_gap']:12.2e}"
        )

# ── SCAD sweep ──
print()
print("=" * 70)
print("SCAD sweep")
print("=" * 70)
print(
    f"{'lam':>6s}  {'a':>6s}  {'beta0':>6s}  {'rho':>8s}  "
    f"{'rel_err':>10s}  {'min_gap':>12s}  {'final_gap':>12s}  {'status':>8s}"
)
print("-" * 80)

scad_results = []
for lam, a, beta0 in cartesian(lam_values, scad_a_values, beta0_values):
    # Feasibility check: beta_0 < a - 1
    if beta0 >= a - 1:
        continue
    prox_fn, deriv_fn, rho = make_scad(lam, a)
    res = run_single(prox_fn, deriv_fn, beta0, n_steps)
    status = "OK" if res else "FAIL"
    row = dict(lam=lam, a=a, beta0=beta0, rho=rho, status=status)
    if res:
        row.update(res)
        print(
            f"{lam:6.2f}  {a:6.1f}  {beta0:6.2f}  {rho:8.4f}  "
            f"{res['rel_err']:10.4f}  {res['min_gap']:12.2e}  {res['final_gap']:12.2e}  {status:>8s}"
        )
    else:
        row.update(dict(rel_err=np.inf, min_gap=np.inf, final_gap=np.inf))
    scad_results.append(row)

# ── MCP sweep ──
print()
print("=" * 70)
print("MCP sweep")
print("=" * 70)
print(
    f"{'lam':>6s}  {'gamma':>6s}  {'beta0':>6s}  {'rho':>8s}  "
    f"{'rel_err':>10s}  {'min_gap':>12s}  {'final_gap':>12s}  {'status':>8s}"
)
print("-" * 80)

mcp_results = []
for lam, gamma, beta0 in cartesian(lam_values, mcp_gamma_values, beta0_values):
    # Feasibility check: beta_0 < gamma
    if beta0 >= gamma:
        continue
    prox_fn, deriv_fn, rho = make_mcp(lam, gamma)
    res = run_single(prox_fn, deriv_fn, beta0, n_steps)
    status = "OK" if res else "FAIL"
    row = dict(lam=lam, gamma=gamma, beta0=beta0, rho=rho, status=status)
    if res:
        row.update(res)
        print(
            f"{lam:6.2f}  {gamma:6.1f}  {beta0:6.2f}  {rho:8.4f}  "
            f"{res['rel_err']:10.4f}  {res['min_gap']:12.2e}  {res['final_gap']:12.2e}  {status:>8s}"
        )
    else:
        row.update(dict(rel_err=np.inf, min_gap=np.inf, final_gap=np.inf))
    mcp_results.append(row)


# ─── Summary: best configs ───────────────────────────────────────────────────

print()
print("=" * 70)
print("Best configurations (by min smoothed gap)")
print("=" * 70)

# Best L1
best_l1 = min(l1_results.items(), key=lambda x: x[1]["min_gap"])
print(
    f"\nL1:   beta0={best_l1[0]:.2f}  "
    f"rel_err={best_l1[1]['rel_err']:.4f}  min_gap={best_l1[1]['min_gap']:.2e}"
)

# Best SCAD
valid_scad = [r for r in scad_results if r["status"] == "OK"]
if valid_scad:
    best_scad = min(valid_scad, key=lambda r: r["min_gap"])
    print(
        f"SCAD: lam={best_scad['lam']:.2f}  a={best_scad['a']:.1f}  "
        f"beta0={best_scad['beta0']:.2f}  rho={best_scad['rho']:.4f}  "
        f"rel_err={best_scad['rel_err']:.4f}  min_gap={best_scad['min_gap']:.2e}"
    )

# Best MCP
valid_mcp = [r for r in mcp_results if r["status"] == "OK"]
if valid_mcp:
    best_mcp = min(valid_mcp, key=lambda r: r["min_gap"])
    print(
        f"MCP:  lam={best_mcp['lam']:.2f}  gamma={best_mcp['gamma']:.1f}  "
        f"beta0={best_mcp['beta0']:.2f}  rho={best_mcp['rho']:.4f}  "
        f"rel_err={best_mcp['rel_err']:.4f}  min_gap={best_mcp['min_gap']:.2e}"
    )

print()
print("=" * 70)
print("Best configurations (by relative error)")
print("=" * 70)

best_l1_err = min(l1_results.items(), key=lambda x: x[1]["rel_err"])
print(
    f"\nL1:   beta0={best_l1_err[0]:.2f}  "
    f"rel_err={best_l1_err[1]['rel_err']:.4f}  min_gap={best_l1_err[1]['min_gap']:.2e}"
)

if valid_scad:
    best_scad_err = min(valid_scad, key=lambda r: r["rel_err"])
    print(
        f"SCAD: lam={best_scad_err['lam']:.2f}  a={best_scad_err['a']:.1f}  "
        f"beta0={best_scad_err['beta0']:.2f}  rho={best_scad_err['rho']:.4f}  "
        f"rel_err={best_scad_err['rel_err']:.4f}  min_gap={best_scad_err['min_gap']:.2e}"
    )

if valid_mcp:
    best_mcp_err = min(valid_mcp, key=lambda r: r["rel_err"])
    print(
        f"MCP:  lam={best_mcp_err['lam']:.2f}  gamma={best_mcp_err['gamma']:.1f}  "
        f"beta0={best_mcp_err['beta0']:.2f}  rho={best_mcp_err['rho']:.4f}  "
        f"rel_err={best_mcp_err['rel_err']:.4f}  min_gap={best_mcp_err['min_gap']:.2e}"
    )


# ─── Heatmaps: rel error as function of (lam, shape) at best beta0 ──────────


def make_heatmap(results, shape_key, shape_vals, lam_vals, beta0_vals, title, ax):
    """For each (lam, shape), pick the best beta0 by rel_err and plot."""
    grid = np.full((len(shape_vals), len(lam_vals)), np.nan)
    for r in results:
        if r["status"] != "OK":
            continue
        shape_index = {v: i for i, v in enumerate(shape_vals)}
        i = shape_index.get(r[shape_key], -1)
        # i = shape_vals.index(r[shape_key]) if r[shape_key] in shape_vals else -1
        lam_index = {v: i for i, v in enumerate(lam_vals)}
        j = lam_index.get(r["lam"], -1)
        # j = lam_vals.index(r["lam"]) if r["lam"] in lam_vals else -1
        if i < 0 or j < 0:
            continue
        if np.isnan(grid[i, j]) or r["rel_err"] < grid[i, j]:
            grid[i, j] = r["rel_err"]

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[-0.5, len(lam_vals) - 0.5, -0.5, len(shape_vals) - 0.5],
    )
    ax.set_xticks(range(len(lam_vals)))
    ax.set_xticklabels([f"{v}" for v in lam_vals], fontsize=7)
    ax.set_yticks(range(len(shape_vals)))
    ax.set_yticklabels([f"{v}" for v in shape_vals], fontsize=7)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(shape_key if shape_key != "gamma" else r"$\gamma$")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Rel. error")

    # Annotate cells
    for i in range(len(shape_vals)):
        for j in range(len(lam_vals)):
            if not np.isnan(grid[i, j]):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if grid[i, j] > np.nanmedian(grid) else "black",
                )


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
make_heatmap(
    scad_results,
    "a",
    scad_a_values,
    lam_values,
    beta0_values,
    "SCAD: best rel. error over $\\beta_0$",
    ax1,
)
make_heatmap(
    mcp_results,
    "gamma",
    mcp_gamma_values,
    lam_values,
    beta0_values,
    "MCP: best rel. error over $\\beta_0$",
    ax2,
)
fig.suptitle("Hyperparameter tuning: relative reconstruction error", fontsize=13)
plt.tight_layout()
outpath = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hyperparam_tuning.pdf"
)
fig.savefig(outpath, dpi=150)
plt.close(fig)
print(f"\nSaved heatmap to {outpath}")
