#!/usr/bin/env python3
"""A scalar, two-component demonstration of stochastic FRAMES.

Reproduce: python run_demo.py  (requires numpy and matplotlib).
Source: Nonsmooth_FW-7.pdf (2026-09-27), Proposition 2.11, pp. 15--16.
No benchmarking, external data, inner optimization, or parameter search.
"""
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
STEPS = 32768
SEEDS = list(range(16))
BETA0 = 1.0
X0 = 0.75
EXACT_HORIZON = 18
SOURCE = Path("SOURCE_MANUSCRIPT/Nonsmooth_FW-7.pdf")


def f(x):
    return 0.5*x


def grad(x):
    return np.full_like(x, 0.5, dtype=float)


def moreau_grad(x, beta):
    return np.minimum(x, 0.0)/beta


def true_gap(x, beta):
    h = grad(x) + moreau_grad(x, beta)
    return x*h + np.abs(h)


def original_gap(x):
    h = grad(x)
    return x*h - np.minimum(h, 0.0)  # LMO over C intersect D = [0,1]


def lmo(h):
    # Any minimizer is allowed at zero; fix the tie convention explicitly.
    return np.where(h > 0.0, -1.0, np.where(h < 0.0, 1.0, 0.0))


def exact_expected_output(n):
    """Enumerate every noise history: actual expectation-based theorem index.

    x_k depends on k noise draws. Equal-weight states at depth k represent all
    2**k histories. This uses no Monte Carlo approximation or per-run selector.
    At k=0 the first gradient sample is the momentum initialization.
    """
    x = np.array([X0])
    d = np.zeros(1)
    means = []
    feasibility = []
    magnitude = []
    signed = []
    for k in range(n+1):
        beta = BETA0/(k+1)**0.25
        means.append(float(np.mean(true_gap(x, beta))))
        feasibility.append(float(np.mean(np.maximum(-x, 0.0))))
        magnitude.append(float(np.mean(np.abs(original_gap(x)))))
        signed.append(float(np.mean(original_gap(x))))
        if k == n:
            break
        x = np.repeat(x, 2)
        d = np.repeat(d, 2)
        noise = np.tile(np.array([-1.0, 1.0]), x.size//2)
        sample = grad(x) + noise
        alpha = (k+1)**-0.5
        d = sample if k == 0 else (1-alpha)*d + alpha*sample
        x += (k+1)**-0.75*(lmo(d + moreau_grad(x, beta)) - x)
    lo = (n+1)//2
    selected = lo + int(np.argmin(means[lo:n+1]))
    return {
        "horizon": n, "last_half_inclusive": [lo, n],
        "selected_index": selected, "number_of_histories_at_horizon": 2**n,
        "expected_gap_by_index": means,
        "expected_feasibility_by_index": feasibility,
        "expected_abs_original_gap_by_index": magnitude,
        "expected_signed_original_gap_by_index": signed,
        "note": "Full finite probability-tree sum, up to floating point roundoff; not Monte Carlo.",
    }


def verify_math(x, beta, gaps, error, noise, exact):
    grid = np.linspace(-1, 1, 10001)
    sample_minus, sample_plus = grad(grid)-1, grad(grid)+1
    np.testing.assert_allclose((sample_minus+sample_plus)/2, grad(grid), atol=1e-15)
    np.testing.assert_allclose(((sample_minus-grad(grid))**2 + (sample_plus-grad(grid))**2)/2, 1)
    assert np.max(np.abs(x)) <= 1+1e-14
    assert gaps.min() >= -1e-14
    for b in [1.0, 0.1, 0.01]:
        h = grad(grid) + moreau_grad(grid, b)
        # Independent endpoint maximization over C, not the noisy LMO.
        np.testing.assert_allclose(true_gap(grid, b), np.maximum(h*(grid+1), h*(grid-1)), atol=1e-14)
        xb = -b/2
        assert abs(float(true_gap(np.array(xb), b))) < 1e-13
        delta = np.maximum(-grid, 0)
        # Direct scalar transfer inequalities with exact coefficient 1/2.
        assert np.all(delta <= b/2 + np.sqrt(np.maximum(b*true_gap(grid, b), 0)) + 1e-13)
        assert np.all(np.abs(original_gap(grid)) <= true_gap(grid, b) + delta/2 + 1e-13)
        points = np.array([-0.9, -0.2, 0.2, 0.9])
        eps = 1e-6
        phi = lambda z: f(z) + np.minimum(z, 0)**2/(2*b)
        numeric = (phi(points+eps)-phi(points-eps))/(2*eps)
        np.testing.assert_allclose(numeric, grad(points)+moreau_grad(points, b), rtol=1e-8, atol=1e-8)
    assert np.all(np.isin(noise, [-1, 1]))
    np.testing.assert_allclose(np.abs(error[0]), 1.0)
    assert exact["selected_index"] in range(9, 19)
    # Gamma_0=1: the first random gradient selects one endpoint exactly.
    np.testing.assert_allclose(x[1], -noise[0])
    return ["Exact finite-sum unbiasedness and variance identities",
            "Feasibility in C for every iterate and seed",
            "True gap checked against independent endpoint maximization",
            "Closed-form smoothed stationary point has zero true gap",
            "Direct feasibility and original-gap transfer inequalities on 10001 points",
            "Finite-difference penalty-gradient sign and magnitude",
            "Full first-sample momentum initialization and first update"]


def main():
    started = time.perf_counter()
    k = np.arange(STEPS)
    beta = BETA0/(k+1)**0.25
    alpha = (k+1)**-0.5
    gamma = (k+1)**-0.75
    # Independent PCG64 streams; sampling is uniform, with replacement.
    noise = np.stack([2*np.random.default_rng(seed).integers(0, 2, STEPS)-1
                      for seed in SEEDS], axis=1)
    xs = np.empty((STEPS+1, len(SEEDS)))
    ds = np.empty((STEPS, len(SEEDS)))
    xs[0] = X0
    d = np.zeros(len(SEEDS))
    for j in range(STEPS):
        sample = grad(xs[j]) + noise[j]
        d = sample if j == 0 else (1-alpha[j])*d + alpha[j]*sample
        ds[j] = d
        h_est = d + moreau_grad(xs[j], beta[j])
        xs[j+1] = xs[j] + gamma[j]*(lmo(h_est)-xs[j])
    optimizer_seconds = time.perf_counter()-started
    x = xs[:-1]
    gaps = true_gap(x, beta[:, None])
    avg_gaps = np.cumsum(gaps, axis=0)/(k+1)[:, None]
    delta = np.maximum(-x, 0.0)
    signed = original_gap(x)
    abs_original = np.abs(signed)
    avg_delta = np.cumsum(delta, axis=0)/(k+1)[:, None]
    avg_abs_original = np.cumsum(abs_original, axis=0)/(k+1)[:, None]
    estimator_error = ds-grad(x)
    exact = exact_expected_output(EXACT_HORIZON)
    checks = verify_math(xs, beta, gaps, estimator_error, noise, exact)

    # Save EVERY pre-update gap. Logarithmic downsampling is for display only.
    np.savez_compressed(OUT/"results.npz", seeds=SEEDS, x=xs, d=ds,
        sample_noise=noise, beta=beta, alpha=alpha, gamma=gamma,
        true_smoothed_gap=gaps, cumulative_true_gap=avg_gaps,
        signed_original_gap=signed, feasibility=delta,
        exact_output_index=exact["selected_index"])
    horizons = [64, 256, 1024, 4096, 16384, 32768]
    table = []
    for n in horizons:
        table.append({"steps": n,
            "mean_cumulative_true_gap": float(avg_gaps[n-1].mean()),
            "seed_standard_error": float(avg_gaps[n-1].std(ddof=1)/np.sqrt(len(SEEDS))),
            "mean_cumulative_feasibility": float(avg_delta[n-1].mean()),
            "mean_cumulative_abs_original_gap": float(avg_abs_original[n-1].mean()),
            "mean_last_x": float(xs[n].mean()),
        })
    with (OUT/"summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=table[0].keys())
        writer.writeheader(); writer.writerows(table)
    # Descriptive slope only, for the predeclared range 1024--32768.
    use = np.array([1024, 2048, 4096, 8192, 16384, 32768])
    slope = float(np.polyfit(np.log(use), np.log(avg_gaps[use-1].mean(axis=1)), 1)[0])
    selected_outputs = xs[exact["selected_index"]]
    metadata = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_result": "Proposition 2.11, PDF pp. 15-16; Corollary 2.10; Lemmas 2.4 and 2.7",
        "problem": "min_{x in [-1,1]} (f_minus(x)+f_plus(x))/2 + indicator_[0,infinity)(x), T=Id",
        "components": ["-0.5*x", "1.5*x"],
        "x0": X0, "seeds": SEEDS, "steps": STEPS, "dtype": "float64",
        "schedules": {"gamma": "(k+1)^(-3/4)", "alpha": "(k+1)^(-1/2)", "beta": "(k+1)^(-1/4)"},
        "optimizer_seconds_including_rng": optimizer_seconds,
        "empirical_log_log_slope_1024_to_32768": slope,
        "checks": checks, "table": table,
        "exact_expectation_output_demonstration": exact,
        "exact_rule_seed_outputs_at_horizon_18": selected_outputs.tolist(),
        "long_horizon_output": "No surrogate for expectation-defined k_n is used; long runs report Eq.65 and trajectories only.",
        "draft_caveats": ["Use + (x-proj_D(x))/beta as in Section 2; Algorithm 1 line 6 has opposite sign.",
                          "Conditional unbiasedness and bounded variance are supplied proof-level hypotheses; sigma squared is undefined in Eq.61.",
                          "The fixed schedules of Proposition 2.11, not Algorithm 1 adaptive penalty update, are used."],
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
    }
    (OUT/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    colors = plt.cm.viridis(np.linspace(.08, .9, len(SEEDS)))
    display = np.unique(np.r_[np.arange(1, 80), np.geomspace(80, STEPS, 600).astype(int)])-1

    fig, ax = plt.subplots(1, 3, figsize=(14.8, 4.2), constrained_layout=True)
    z = np.linspace(-1, 1, 1601)
    ax[0].axvspan(-1, 0, color="#eeeef3", zorder=-5)
    ax[0].plot(z, f(z), color=".5", ls=":", label=r"$f(x)$ (ignores $D$)")
    zz = z[z >= 0]
    ax[0].plot(zz, f(zz), color="black", lw=2.7, label=r"$F(x)$ on $[0,1]$")
    for b, color in [(1, "#ef8a62"), (.25, "#67a9cf"), (.0625, "#2166ac")]:
        ax[0].plot(z, f(z)+np.minimum(z, 0)**2/(2*b), color=color, label=fr"$\Phi_\beta$, $\beta={b:g}$")
        xb = -b/2
        ax[0].scatter([xb], [f(xb)+xb*xb/(2*b)], color=color, s=22)
    ax[0].scatter([0], [0], marker="*", s=95, color="black", zorder=6)
    ax[0].text(-.95, 1.08, r"$F(x)=+\infty$ for $x<0$", fontsize=9)
    ax[0].set(xlim=(-1, 1), ylim=(-.17, 1.25), xlabel="$x$", ylabel="Objective", title="Landscape and smoothing bias")
    ax[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(.02, .87))

    for s in range(len(SEEDS)):
        ax[1].plot(k[display]+1, x[display, s], color=colors[s], lw=.8, alpha=.65)
    xb = -beta/2
    ax[1].plot(k[display]+1, xb[display], color="black", ls="--", lw=1.6, label=r"Exact $x_{\beta_k}$")
    ax[1].axhline(0, color="black", lw=.8, label=r"Solution $x^*=0$")
    ax[1].set(xscale="log", xlabel="$k+1$", ylabel="$x_k$", title="All 16 sampled trajectories")
    ax[1].legend(fontsize=8)
    for s in range(len(SEEDS)):
        ax[2].plot(k[display]+1, f(x[display, s])+np.minimum(x[display, s], 0)**2/(2*beta[display]), color=colors[s], lw=.8, alpha=.65)
    ax[2].plot(k[display]+1, -beta[display]/8, "k--", lw=1.5, label=r"$\min_x\Phi_k=-\beta_k/8$")
    ax[2].set(xscale="log", xlabel="$k+1$", ylabel=r"$\Phi_k(x_k)$", ylim=(-.14, .2), title="Smoothed objective along trajectories")
    ax[2].text(.04, .91, "Early values above 0.2 are clipped", transform=ax[2].transAxes, fontsize=8)
    ax[2].legend(fontsize=8, loc="lower right")
    fig.savefig(OUT/"landscape_trajectories.png", dpi=170)
    fig.savefig(OUT/"landscape_trajectories.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14.8, 4.2), constrained_layout=True)
    n = k+1
    for s in range(len(SEEDS)):
        ax[0].loglog(n[display], avg_gaps[display, s], color=colors[s], lw=.8, alpha=.55)
    mean = avg_gaps.mean(axis=1)
    sem = avg_gaps.std(axis=1, ddof=1)/np.sqrt(len(SEEDS))
    ax[0].loglog(n[display], mean[display], "k", lw=2, label="16-seed mean")
    ax[0].fill_between(n[display], np.maximum(mean[display]-2*sem[display], 1e-12), mean[display]+2*sem[display], color="black", alpha=.12, label=r"$\pm 2$ seed SE (descriptive)")
    anchor = 1024
    ref = mean[anchor-1]*(n/anchor)**-.25
    ax[0].loglog(n[display], ref[display], "--", color="#d95f02", lw=2, label=r"$N^{-1/4}$ slope guide")
    ax[0].set(xlabel="Updates / sampled gradients $N$", ylabel=r"$N^{-1}\sum_{k<N}G_k$", title="The theorem's primary statistic (Eq. 65)")
    ax[0].legend(fontsize=8)
    # A contiguous late window avoids aliasing oscillatory instantaneous gaps.
    window = np.arange(STEPS-1200, STEPS)
    for s in range(4):
        ax[1].plot(window, gaps[window, s], color=colors[4*s], alpha=.55, lw=.6, label=f"Seed {s}")
    ax[1].set(xlabel="Iteration $k$", ylabel=r"$G_k$ (true gradient)", title="Instantaneous gaps can oscillate")
    ax[1].legend(fontsize=8)
    ax[2].loglog(n[display], avg_delta[display].mean(axis=1), lw=2, color="#2166ac", label=r"Mean cumulative $\delta_k$")
    ax[2].loglog(n[display], avg_abs_original[display].mean(axis=1), lw=2, color="#b35806", label=r"Mean cumulative $|H_k|$")
    # The direct scalar transfer gives an average bound, explicitly a corollary.
    bbar = np.cumsum(beta)/n
    transfer_delta = .5*bbar+np.sqrt(bbar*mean)
    ax[2].loglog(n[display], transfer_delta[display], "--", color="#2166ac", alpha=.7, label=r"$\bar\beta/2+\sqrt{\bar\beta\,\bar G}$ bound")
    ax[2].loglog(n[display], (mean+.5*transfer_delta)[display], "--", color="#b35806", alpha=.7, label=r"$\bar G+\frac{1}{2}$ feasibility bound")
    ax[2].set(xlabel="Updates $N$", ylabel="Cumulative averages", title="Original problem: two-part diagnostic")
    ax[2].legend(fontsize=8)
    fig.savefig(OUT/"stationarity.png", dpi=170)
    fig.savefig(OUT/"stationarity.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    ks = np.arange(EXACT_HORIZON+1)
    ax.plot(ks, exact["expected_gap_by_index"], "o-", color="#2166ac", label="Exact expectation over all histories")
    ax.axvspan(9, 18, color="#dfebf4", alpha=.5, label="Theorem's last-half window")
    sel = exact["selected_index"]
    ax.scatter([sel], [exact["expected_gap_by_index"][sel]], marker="*", s=180, color="#d95f02", zorder=5, label=fr"Exact $k_{{18}}={sel}$")
    ax.set(xlabel="$k$", ylabel=r"$\mathbb{E}[G_k]$", title="Exact output-rule demonstration (262,144 noise histories)", xticks=np.arange(0, 19, 2))
    ax.legend(fontsize=8)
    fig.savefig(OUT/"exact_output_rule.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"optimizer_seconds": optimizer_seconds, "empirical_slope": slope,
        "summary": table, "exact_selected_index_n18": sel,
        "exact_expected_G_at_output": exact["expected_gap_by_index"][sel],
        "exact_expected_feasibility_at_output": exact["expected_feasibility_by_index"][sel],
        "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
