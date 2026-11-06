#!/usr/bin/env python3
"""Stochastic FRAMES: 1024-dimensional l1-constrained least squares.

Run with a Python environment containing numpy and matplotlib:
    python3 run_demo.py
One uniform sample from two dense least-squares blocks per iteration.
The independent noise draws have a rank-one covariance; see metadata.json.
"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
DIM = 1024
STEPS = 32768
SEEDS = list(range(8))
BETA0 = 1.0
EXACT_HORIZON = 10
SOURCE = Path("/home/tony/gw_benchmark_experiment_20260927/provenance/manuscript.txt")
A = np.r_[np.full(64, 1/32), np.full(64, -3/128), np.full(896, 1/128)]
B = np.where(np.arange(DIM) % 2 == 0, 1.0, -1.0)/64
XSTAR = np.r_[np.full(64, 1/64), np.zeros(960)]
FSTAR = 91/512
GRAD_BOUND = np.sqrt(311)/16


def loss(x):
    """Mean of the two block losses, including the constant noise term."""
    return .5*np.sum((x-A)**2, axis=-1) + 1/8


def smoothed_solution(beta):
    result = np.zeros(DIM)
    result[:64] = (2+3*beta)/(128*(1+2*beta))
    result[64:128] = -beta/(128*(1+2*beta))
    return result


def metrics(x, beta, momentum=None):
    gradient = x-A
    negative = np.minimum(x, 0.0)
    h = gradient+negative/beta
    true_gap = np.sum(h*x, axis=-1)+np.max(np.abs(h), axis=-1)
    original = np.sum(gradient*x, axis=-1)+np.maximum(0.0, -np.min(gradient, axis=-1))
    feasibility = np.linalg.norm(negative, axis=-1)
    projected_gap = loss(np.maximum(x, 0.0))-FSTAR
    distance = np.linalg.norm(x-XSTAR, axis=-1)
    output = (true_gap, original, feasibility, projected_gap, distance)
    if momentum is not None:
        output += (np.linalg.norm(momentum-gradient, axis=-1),)
    return output


def step(x, momentum, beta, gamma):
    """Exact l1-ball LMO, with the first maximizing coordinate breaking ties."""
    h = momentum+np.minimum(x, 0.0)/beta
    j = np.argmax(np.abs(h), axis=1)
    sign = np.sign(h[np.arange(len(x)), j])
    result = (1-gamma)*x
    result[np.arange(len(x)), j] -= gamma*sign
    return result


def exact_expected_output(n):
    """Full probability-tree sum, not a Monte Carlo or per-run selector.

    At depth k, x_k depends on k random component draws. Momentum contains
    the previous estimate, until the next draw is branched into both options.
    """
    x = np.zeros((1, DIM))
    momentum = np.zeros_like(x)
    expected_gap, expected_feasibility, expected_magnitude, expected_signed = [], [], [], []
    for k in range(n+1):
        beta = BETA0*(k+1)**(-.25)
        gap, original, delta, _, _ = metrics(x, beta)
        expected_gap.append(float(gap.mean()))
        expected_feasibility.append(float(delta.mean()))
        expected_magnitude.append(float(np.abs(original).mean()))
        expected_signed.append(float(original.mean()))
        if k == n:
            break
        x = np.repeat(x, 2, axis=0)
        momentum = np.repeat(momentum, 2, axis=0)
        noise = np.tile([-1., 1.], len(x)//2)
        sample = x-A+noise[:, None]*B
        alpha = (k+1)**(-.5)
        momentum = sample if k == 0 else (1-alpha)*momentum+alpha*sample
        x = step(x, momentum, beta, (k+1)**(-.75))
    lo = (n+1)//2
    selected = lo+int(np.argmin(expected_gap[lo:n+1]))
    return {"horizon": n, "last_half_inclusive": [lo, n],
            "selected_index": selected, "histories_at_horizon": 2**n,
            "expected_true_gap": expected_gap,
            "expected_feasibility": expected_feasibility,
            "expected_abs_original_gap": expected_magnitude,
            "expected_signed_original_gap": expected_signed,
            "method": "All equally weighted binary histories; exact expectation up to floating point arithmetic."}


def verify_static_math():
    rng = np.random.default_rng(1024)
    x = rng.normal(size=(100, DIM))
    x *= rng.uniform(size=(100, 1))/np.sum(np.abs(x), axis=1)[:, None]
    qminus, qplus = x-A-B, x-A+B
    np.testing.assert_allclose((qminus+qplus)/2, x-A, atol=1e-16)
    variance = .5*(np.sum((qminus-(x-A))**2, axis=1)+np.sum((qplus-(x-A))**2, axis=1))
    np.testing.assert_allclose(variance, .25, atol=1e-15)
    np.testing.assert_allclose(np.sum(A*A), 39/256)
    np.testing.assert_allclose(np.sum(B*B), .25)
    assert np.max(np.sum(np.maximum(x, 0), axis=1)) <= 1
    # Positive-simplex projection KKT: threshold tau=1/64.
    np.testing.assert_allclose(XSTAR, np.maximum(A-1/64, 0), atol=1e-16)
    np.testing.assert_allclose(XSTAR.sum(), 1)
    np.testing.assert_allclose(loss(XSTAR), FSTAR)
    for beta in [1., .1, .01]:
        xb = smoothed_solution(beta)
        np.testing.assert_allclose(np.linalg.norm(xb, 1), 1)
        gap, original, delta, _, _ = metrics(xb[None, :], beta)
        np.testing.assert_allclose(gap, 0, atol=1e-15)
        np.testing.assert_allclose(delta, beta/(16*(1+2*beta)), atol=1e-15)
        np.testing.assert_allclose(original, -beta/(256*(1+2*beta)**2), atol=1e-15)
        np.testing.assert_allclose(loss(xb)+delta**2/(2*beta), FSTAR-beta/(512*(1+2*beta)), atol=1e-15)
        h = x-A+np.minimum(x, 0)/beta
        vertex = step(np.zeros_like(x), h, beta, 1.)
        np.testing.assert_allclose(np.sum(h*vertex, axis=1), -np.max(np.abs(h), axis=1))
        gap, original, delta, _, _ = metrics(x, beta)
        assert np.all(delta <= GRAD_BOUND*beta+np.sqrt(beta*np.maximum(gap, 0))+1e-13)
        assert np.all(np.abs(original) <= gap+GRAD_BOUND*delta+1e-13)
    return ["Two-component mean and conditional variance identities",
            "Exact l1-ball LMO support-function identity",
            "Orthant projection remains in the l1 ball (regularity constant 1)",
            "Unique original solution verified by positive-simplex KKT",
            "Exact smoothed minimizer, zero gap, feasibility, original gap and objective",
            "Pointwise gap and feasibility transfer inequalities"]


def main():
    started = time.perf_counter()
    checks = verify_static_math()
    k = np.arange(STEPS)
    beta = BETA0*(k+1)**(-.25)
    alpha = (k+1)**(-.5)
    gamma = (k+1)**(-.75)
    noise = np.stack([2*np.random.default_rng(seed).integers(0, 2, STEPS)-1 for seed in SEEDS], axis=1)
    checkpoint_k = np.unique(np.r_[0, np.geomspace(1, STEPS, 200).astype(int), EXACT_HORIZON])
    checkpoint_lookup = {int(v): j for j, v in enumerate(checkpoint_k)}
    checkpoints = np.empty((len(checkpoint_k), len(SEEDS), DIM), dtype=np.float32)
    values = np.empty((6, STEPS, len(SEEDS)))
    x = np.zeros((len(SEEDS), DIM))
    momentum = np.zeros_like(x)
    max_l1_norm = 0.
    for j in range(STEPS):
        if j in checkpoint_lookup:
            checkpoints[checkpoint_lookup[j]] = x
        sample = x-A+noise[j, :, None]*B
        momentum = sample if j == 0 else (1-alpha[j])*momentum+alpha[j]*sample
        values[:, j] = metrics(x, beta[j], momentum)
        max_l1_norm = max(max_l1_norm, float(np.max(np.sum(np.abs(x), axis=1))))
        x = step(x, momentum, beta[j], gamma[j])
    checkpoints[checkpoint_lookup[STEPS]] = x
    optimizer_seconds = time.perf_counter()-started
    gap, signed, delta, projected, distance, error = values
    cumulative = np.cumsum(gap, axis=0)/(k+1)[:, None]
    mean_gap = cumulative.mean(axis=1)
    beta_average = np.cumsum(beta)/(k+1)
    delta_average = np.cumsum(delta, axis=0)/(k+1)[:, None]
    magnitude_average = np.cumsum(np.abs(signed), axis=0)/(k+1)[:, None]
    assert max_l1_norm <= 1+1e-12
    assert float(gap.min()) >= -1e-12
    assert float(projected.min()) >= -1e-12
    assert np.all(delta <= GRAD_BOUND*beta[:, None]+np.sqrt(beta[:, None]*np.maximum(gap, 0))+1e-12)
    assert np.all(np.abs(signed) <= gap+GRAD_BOUND*delta+1e-12)
    delta_bound = GRAD_BOUND*beta_average+np.sqrt(beta_average*mean_gap)
    magnitude_bound = mean_gap+GRAD_BOUND*delta_average.mean(axis=1)
    assert np.all(delta_average.mean(axis=1) <= delta_bound+1e-12)
    assert np.all(magnitude_average.mean(axis=1) <= magnitude_bound+1e-12)
    checks += ["Every iterate remains in the l1 ball",
               "Every true smoothed gap and projected feasible objective gap is nonnegative",
               "Pointwise and cumulative transfer inequalities on every saved metric"]
    exact = exact_expected_output(EXACT_HORIZON)
    # The exact expectation selects one common index for all stochastic runs.
    # Replaying only these few updates stores its actual outputs in float64.
    selected_x = np.zeros((len(SEEDS), DIM))
    selected_momentum = np.zeros_like(selected_x)
    for j in range(exact["selected_index"]):
        sample = selected_x-A+noise[j, :, None]*B
        selected_momentum = sample if j == 0 else (1-alpha[j])*selected_momentum+alpha[j]*sample
        selected_x = step(selected_x, selected_momentum, beta[j], gamma[j])
    checks.append("Full expectation-based output selector at horizon 10 (1024 histories)")
    horizons = [64, 256, 1024, 4096, 16384, 32768]
    table = []
    for n in horizons:
        table.append({"updates": n,
                      "mean_cumulative_true_gap": float(cumulative[n-1].mean()),
                      "seed_standard_error": float(cumulative[n-1].std(ddof=1)/np.sqrt(len(SEEDS))),
                      "mean_cumulative_feasibility": float(delta_average[n-1].mean()),
                      "mean_cumulative_abs_original_gap": float(magnitude_average[n-1].mean()),
                      "mean_last_projected_objective_gap": float(projected[n-1].mean()),
                      "mean_last_distance_to_solution": float(distance[n-1].mean()),
                      "mean_last_momentum_error": float(error[n-1].mean())})
    with (OUT/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    use = np.array([1024, 2048, 4096, 8192, 16384, 32768])
    slope = float(np.polyfit(np.log(use), np.log(mean_gap[use-1]), 1)[0])
    np.savez_compressed(OUT/"results.npz", a=A, b=B, solution=XSTAR, final_x=x,
                        seeds=SEEDS, sample_noise=noise.astype(np.int8), beta=beta,
                        alpha=alpha, gamma=gamma, true_smoothed_gap=gap,
                        signed_original_gap=signed, feasibility=delta,
                        projected_objective_gap=projected, distance_to_solution=distance,
                        momentum_error=error, checkpoint_k=checkpoint_k,
                        x_checkpoints=checkpoints,
                        exact_output_index=exact["selected_index"],
                        exact_selected_seed_outputs=selected_x)
    metadata = {"dimension": DIM, "steps": STEPS, "seeds": SEEDS,
                "source_manuscript": str(SOURCE),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest() if SOURCE.exists() else None,
                "source_theorem": "Proposition 2.11 (stochastic fixed-schedule result)",
                "problem": "min_{||x||_1<=1} .25*(||x-a-b||_2^2+||x-a+b||_2^2) + indicator_R_+^1024(x)",
                "a_groups": [[0, 64, 1/32], [64, 128, -3/128], [128, 1024, 1/128]],
                "b": "b_j=(-1)^j/64, zero-based j; ||b||_2^2=1/4",
                "oracle": "One independent uniform draw of the two full least-squares-block gradients per iteration; noise covariance bb^T has rank one. Not individual scalar-row sampling.",
                "x0": "Zero vector", "dtype": "float64; vector checkpoints stored as float32",
                "schedules": {"gamma": "(k+1)^(-3/4)", "alpha": "(k+1)^(-1/2)", "beta": "(k+1)^(-1/4)"},
                "tie_rule": "First maximum-absolute-gradient coordinate, with zero vertex if the full LMO input is zero",
                "exact_solution": "First64 entries1/64, remaining960 zero", "f_star": FSTAR,
                "constants": {"smoothness": 1., "diameter_C": 2., "operator_norm_T": 1.,
                              "regularity_kappa": 1., "gradient_bound_on_C": GRAD_BOUND,
                              "oracle_variance": .25, "oracle_second_moment_bound": 375/256,
                              "common_draft_safe_Lf": 1.25},
                "optimizer_seconds_including_diagnostics": optimizer_seconds,
                "total_seconds_before_plotting": time.perf_counter()-started,
                "empirical_cumulative_gap_slope_1024_to_32768": slope,
                "slope_note": "Descriptive fixed-range fit; not a theorem validation or a guaranteed pointwise slope",
                "exact_output_demonstration": exact,
                "exact_selected_seed_outputs": "results.npz: exact_selected_seed_outputs (float64), all8 seeds at the same exact expectation-selected index",
                "summary_table_indexing": "Cumulative statistics average x_0 through x_(N-1); every mean_last_* column is evaluated at the pre-update x_(N-1), not final_x=x_N",
                "long_horizon_output": "No last-iterate or empirical-minimum substitute for the expectation-defined theorem index; report cumulative Eq65 and diagnostic trajectories",
                "projected_objective_note": "f(max(x,0))-f_star is a diagnostic at a feasible projected point; projection does not change the algorithm's iterates",
                "checks": checks, "summary": table,
                "python_executable": sys.executable, "numpy_version": np.__version__,
                "matplotlib_version": matplotlib.__version__,
                "draft_caveats": ["Use +(x-proj_D(x))/beta, consistent with the smoothed objective; printed Algorithm1 has the opposite penalty sign",
                                  "Raw oracle conditional unbiasedness and bounded variance are supplied explicitly; momentum is not conditionally unbiased",
                                  "Use Proposition2.11 fixed schedules, not adaptive-penalty Algorithm1"]}
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    colors = plt.cm.viridis(np.linspace(.07, .9, len(SEEDS)))
    display = np.unique(np.r_[np.arange(1, 80), np.geomspace(80, STEPS, 600).astype(int)])-1
    n = k+1
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for s in range(len(SEEDS)):
        axes[0].loglog(n[display], cumulative[display, s], color=colors[s], alpha=.55, lw=.9)
    sem = cumulative.std(axis=1, ddof=1)/np.sqrt(len(SEEDS))
    axes[0].loglog(n[display], mean_gap[display], "k", lw=2, label="8-seed mean")
    axes[0].fill_between(n[display], np.maximum(mean_gap[display]-2*sem[display], 1e-16), mean_gap[display]+2*sem[display], color="black", alpha=.12, label=r"$\pm2$ seed SE (descriptive)")
    guide = mean_gap[1023]*(n/1024)**(-.25)
    axes[0].loglog(n[display], guide[display], "--", color="#d95f02", lw=2, label=r"$N^{-1/4}$ slope guide")
    axes[0].set(xlabel="Sampled block gradients / updates N", ylabel=r"$N^{-1}\sum_{k<N}G_k$", title="1024D: theorem's cumulative true gap")
    axes[0].legend(fontsize=8)
    window = np.arange(STEPS-1000, STEPS)
    for s in range(4):
        axes[1].plot(window, gap[window, s], color=colors[2*s], lw=.7, alpha=.6, label=f"Seed{s}")
    axes[1].set(xlabel="Iteration k", ylabel=r"$G_k$ (true gradient)", title="Instantaneous gaps need not be monotone")
    axes[1].legend(fontsize=8)
    axes[2].loglog(n[display], delta_average[display].mean(axis=1), color="#2166ac", lw=2, label=r"Mean cumulative $\delta_k$")
    axes[2].loglog(n[display], magnitude_average[display].mean(axis=1), color="#b35806", lw=2, label=r"Mean cumulative $|H_k|$")
    axes[2].loglog(n[display], delta_bound[display], "--", color="#2166ac", alpha=.7, label=r"$M\bar\beta+\sqrt{\bar\beta\bar G}$ bound")
    axes[2].loglog(n[display], magnitude_bound[display], "--", color="#b35806", alpha=.7, label=r"$\bar G+M\bar\delta$ bound")
    axes[2].set(xlabel="Updates N", ylabel="Cumulative averages", title="Original problem: gap and feasibility")
    axes[2].legend(fontsize=8)
    fig.savefig(OUT/"stationarity_1024.png", dpi=170)
    fig.savefig(OUT/"stationarity_1024.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for s in range(len(SEEDS)):
        axes[0].loglog(n[display], distance[display, s], color=colors[s], lw=.8, alpha=.7)
        axes[1].loglog(n[display], np.maximum(projected[display, s], 1e-16), color=colors[s], lw=.8, alpha=.7)
        axes[2].loglog(n[display], np.maximum(error[display, s], 1e-16), color=colors[s], lw=.8, alpha=.7)
    bias_distance = np.sqrt(128)*beta/(128*(1+2*beta))
    v = beta/(128*(1+2*beta))
    axes[0].loglog(n[display], bias_distance[display], "k--", lw=1.7, label=r"Exact $\|x_{\beta_k}-x^*\|_2$")
    axes[1].loglog(n[display], (v+32*v*v)[display], "k--", lw=1.7, label=r"$f((x_{\beta_k})_+)-f^*$")
    axes[0].set(xlabel="Iteration k+1", ylabel=r"$\|x_k-x^*\|_2$", title="All 8 trajectories: distance to solution")
    axes[1].set(xlabel="Iteration k+1", ylabel=r"$f((x_k)_+)-f^*$", title="Objective at a feasible diagnostic point")
    axes[2].set(xlabel="Iteration k+1", ylabel=r"$\|m_k-\nabla f(x_k)\|_2$", title="Momentum tracking error")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/"trajectories_1024.png", dpi=170)
    fig.savefig(OUT/"trajectories_1024.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for s in range(len(SEEDS)):
        axes[0].plot(np.arange(DIM), x[s], color=colors[s], lw=.7, alpha=.6)
        axes[1].plot(np.arange(160), x[s, :160], color=colors[s], lw=.7, alpha=.6)
    xb = smoothed_solution(BETA0*(STEPS+1)**(-.25))
    for axis, length in [(axes[0], DIM), (axes[1], 160)]:
        axis.plot(np.arange(length), XSTAR[:length], color="black", lw=2, label=r"Exact solution $x^*$")
        axis.plot(np.arange(length), xb[:length], color="#d95f02", ls="--", lw=1.5, label=r"Exact smoothed solution $x_{\beta_N}$")
        axis.axvline(63.5, color=".65", lw=.8)
        axis.axvline(127.5, color=".65", lw=.8)
        axis.set(xlabel="Coordinate (zero based)", ylabel=r"Final coefficient $x_{N,j}$")
    axes[0].set_title("1024 coefficients, all 8 seeds")
    axes[1].set_title("Zoom: nonzero solution and active orthant blocks")
    axes[0].legend(fontsize=8)
    fig.savefig(OUT/"coefficient_profile_1024.png", dpi=170)
    fig.savefig(OUT/"coefficient_profile_1024.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ks = np.arange(EXACT_HORIZON+1)
    axis.plot(ks, exact["expected_true_gap"], "o-", color="#2166ac", label="Full sum over binary histories")
    lo = exact["last_half_inclusive"][0]
    axis.axvspan(lo, EXACT_HORIZON, color="#dfebf4", alpha=.5, label="Theorem's last-half window")
    chosen = exact["selected_index"]
    axis.scatter([chosen], [exact["expected_true_gap"][chosen]], color="#d95f02", s=170, marker="*", zorder=5, label=fr"Exact $k_{{10}}={chosen}$")
    axis.set(xlabel="Iteration k", ylabel=r"$\mathbb{E}[G_k]$", title="1024D: exact expectation-based output at horizon10", xticks=ks)
    axis.legend(fontsize=8)
    fig.savefig(OUT/"exact_output_rule_1024.png", dpi=170)
    plt.close(fig)
    landscape = OUT/"plot_landscape.py"
    if landscape.exists():
        subprocess.run([sys.executable, str(landscape)], check=True)
    print(json.dumps({"optimizer_seconds": optimizer_seconds,
                      "total_seconds": time.perf_counter()-started,
                      "results_archive_bytes": (OUT/"results.npz").stat().st_size,
                      "empirical_cumulative_gap_slope": slope,
                      "exact_output_index_n10": chosen,
                      "summary": table, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
