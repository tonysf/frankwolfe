"""
Generate the main paper figures.

Figure 1: Nonneg MF (indicator) -- beta_0 sweep including 1/L, 2x2
Figure 3: Nonconvex splitting -- 3x2: avg/min metrics, power vs log schedules
"""

import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(THIS_DIR / ".mplconfig"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)
from frank_wolfe import Frames
from matplotlib.lines import Line2D
from tqdm import tqdm

legendfontsize = 12
titlefontsize = 20
labelfontsize = 18
suptitlefontsize = 24


def estimate_lipschitz(obj, lmo, dim, n_samples=500, seed=99):
    rng = np.random.default_rng(seed)
    pool = np.array([lmo(rng.standard_normal(dim)) for _ in range(20)])
    L_est = 0.0
    for _ in range(n_samples):
        w1, w2 = rng.dirichlet(np.ones(len(pool))), rng.dirichlet(np.ones(len(pool)))
        z1, z2 = w1 @ pool, w2 @ pool
        diff = np.linalg.norm(z1 - z2)
        if diff < 1e-12:
            continue
        L_est = max(L_est, np.linalg.norm(obj.gradient(z1) - obj.gradient(z2)) / diff)
    print("L_est is " + str(L_est))
    return L_est


def figure1():
    from paper.experiments.nonnegative_matrix_factorization import (
        MatrixFactorizationObjective,
        create_spectral_ball_product_lmo,
        generate_nonneg_mf_problem,
        nonneg_prox,
    )

    m, n, r = 100, 100, 20
    n_steps = 50000
    margin = 1.05
    seed = 42
    X_star, U_star, V_star, tau_U, tau_V = generate_nonneg_mf_problem(
        m, n, r, margin=margin, seed=seed
    )
    X_star_fro = np.linalg.norm(X_star, "fro")
    obj = MatrixFactorizationObjective(X_star, m, n, r)
    lmo = create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r)
    x0 = np.zeros(m * r + n * r)
    L = estimate_lipschitz(obj, lmo, m * r + n * r)
    inv_L = 1.0 / L
    print(f"Estimated L_{{nabla f}} approx {L:.2f},  1/L approx {inv_L:.6f}")
    beta0_values = [0.2, 0.5, 1.0, 2.0, 5.0, inv_L]
    results = {}
    for beta0 in beta0_values:
        frames = Frames(obj, lmo, nonneg_prox, "indicator")
        frames.run(x0, beta0=beta0, n_steps=n_steps)
        rel_err = np.sqrt(2.0 * np.maximum(frames.func_vals, 0)) / X_star_fro
        results[beta0] = dict(
            gaps=frames.gaps.copy(),
            ns_gaps=frames.ns_gaps.copy(),
            func_vals=frames.func_vals.copy(),
            rel_error=rel_err,
        )
        lab_str = f"1/L={inv_L:.4f}" if beta0 == inv_L else f"{beta0}"
        print(
            f"  beta0={lab_str:>12s}  rel_err={rel_err[-1]:.4f}  dist_D={np.sqrt(2 * frames.ns_gaps[-1]):.4e}"
        )
    iters = np.arange(1, n_steps + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plot_order = sorted(beta0_values, reverse=True)
    colors = plt.cm.hsv(np.linspace(0, 0.9, len(plot_order)))
    plot_stride = 10
    plot_idx = np.arange(plot_stride - 1, n_steps, plot_stride)
    plot_iters = iters[plot_idx]

    def b0_label(b):
        if b == inv_L:
            return rf"$\beta_0 = 1/L_{{\nabla f}}$"
        elif b >= 1 and b == int(b):
            return rf"$\beta_0 = {int(b)}$"
        else:
            return rf"$\beta_0 = {b}$"

    for idx, beta0 in enumerate(plot_order):
        c = colors[idx]
        R = results[beta0]
        lab = b0_label(beta0)
        lw = 2.0 if beta0 == inv_L else 1.2

        avg_g = np.cumsum(R["gaps"]) / iters
        min_g = np.minimum.accumulate(R["gaps"])

        axs[0, 0].semilogy(
            plot_iters, avg_g[plot_idx], color=c, label=lab, alpha=1.0, linewidth=lw
        )
        axs[0, 1].semilogy(
            plot_iters, min_g[plot_idx], color=c, label=lab, alpha=1.0, linewidth=lw
        )
        axs[1, 0].semilogy(
            plot_iters,
            np.sqrt(2.0 * R["ns_gaps"][plot_idx]),
            color=c,
            label=lab,
            alpha=1.0,
            linewidth=lw,
        )
        axs[1, 1].semilogy(
            plot_iters,
            R["rel_error"][plot_idx],
            color=c,
            label=lab,
            alpha=1.0,
            linewidth=lw,
        )

    # --- legends and axes ---
    # 6 beta entries -> ncol=3 gives 2 rows
    leg_kw = dict(fontsize=legendfontsize, loc="lower left", ncol=1)
    # 7 entries (6 beta + reference) -> ncol=4 gives 2 rows
    leg_kw_feas = dict(fontsize=legendfontsize, loc="lower left", ncol=1)

    # Avg gaps
    axs[0, 0].set_title("Average smoothed gaps", fontsize=titlefontsize)
    axs[0, 0].set_xlabel("Iteration $k$", fontsize=labelfontsize)
    axs[0, 0].set_ylabel(
        r"$\frac{1}{k}\sum\limits_{j=0}^{k-1}\mathrm{gap}^{\beta_j}((U_j,V_j))$",
        fontsize=labelfontsize,
    )
    axs[0, 0].legend(**leg_kw)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_ylim(2.5e4, 3e5)

    # Min gaps
    axs[0, 1].set_title("Minimum smoothed gaps", fontsize=titlefontsize)
    axs[0, 1].set_xlabel("Iteration $k$", fontsize=labelfontsize)
    axs[0, 1].set_ylabel(
        r"$\min\limits_{0\leq j\leq k-1}\mathrm{gap}^{\beta_j}((U_j,V_j))$",
        fontsize=labelfontsize,
    )
    axs[0, 1].legend(**leg_kw)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_ylim(5.75e3, 1.5e5)

    # Feasibility
    axs[1, 0].set_title("Feasibility", fontsize=titlefontsize)
    axs[1, 0].set_xlabel("Iteration $k$", fontsize=labelfontsize)
    axs[1, 0].set_ylabel(r"$\mathrm{dist}_D((U_k,V_k))$", fontsize=labelfontsize)
    axs[1, 0].legend(**leg_kw_feas)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_ylim(1.4e-1, 2e0)

    # Reconstruction
    axs[1, 1].set_title("Relative reconstruction error", fontsize=titlefontsize)
    axs[1, 1].set_xlabel("Iteration $k$", fontsize=labelfontsize)
    axs[1, 1].set_ylabel(
        r"$\|U_k V_k^\top - X^\star\|_F / \|X^\star\|_F$", fontsize=labelfontsize
    )
    axs[1, 1].legend(**leg_kw)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_ylim(3e-3, 2e-2)

    fig.suptitle(
        f"Nonnegative matrix factorization,  $m={m},\\; n={n},\\; r={r}$",
        fontsize=suptitlefontsize,
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "paper_fig1_nonneg_mf.pdf"), dpi=150
    )
    plt.close(fig)
    print("Saved paper_fig1_nonneg_mf.pdf")


def figure3():
    from paper.experiments.l1_splitting_nonconvex import (
        IndefiniteQuadraticSplitting,
        create_product_l1_lmo,
        generate_indefinite_quadratic,
        nonsmooth_gap,
        run_frames_splitting,
        run_frames_splitting_log,
        zero_prox,
    )

    nn = 50
    n_steps = 50000
    seed = 42
    c1 = np.zeros(nn)
    c1[0] = 1.0
    c2 = np.zeros(nn)
    c2[0] = -1.0
    radius = 2.0
    Q, b, eigvals = generate_indefinite_quadratic(nn, frac_neg=0.3, seed=seed)
    Q_op = np.linalg.norm(Q, ord=2)
    L_product = 0.5 * Q_op
    inv_L = 1.0 / L_product
    n_pos = np.sum(eigvals > 0)
    n_neg = np.sum(eigvals < 0)
    print(f"Splitting: n={nn}, {n_pos} pos / {n_neg} neg eigs")
    print(f"||Q||_op={Q_op:.2f}, L_product=||Q||/2={L_product:.2f}, 1/L={inv_L:.4f}")
    obj = IndefiniteQuadraticSplitting(Q, b, nn)
    lmo = create_product_l1_lmo(c1, c2, radius, nn)
    beta0_values = [0.25, 0.5, 1.0, 2.0, 4.0, inv_L]

    # Power schedule
    results = {}
    for beta0 in beta0_values:
        _, fvals, sgaps, nsgaps, feas = run_frames_splitting(
            obj, lmo, n_steps, beta0=beta0
        )
        results[beta0] = dict(
            func_vals=fvals, smoothed_gaps=sgaps, ns_gaps=nsgaps, feasibility=feas
        )
        lab_str = f"1/L={inv_L:.4f}" if beta0 == inv_L else f"{beta0}"
        print(
            f"  [power] beta0={lab_str:>12s}  f={fvals[-1]:.4f}  ||x1-x2||={np.sqrt(feas[-1]):.4e}  ns_gap={nsgaps[-1]:.4e}"
        )

    # Log schedule
    results_log = {}
    for beta0_log in beta0_values:
        _, fvals_log, sgaps_log, nsgaps_log, feas_log = run_frames_splitting_log(
            obj, lmo, n_steps, beta0=beta0_log
        )
        results_log[beta0_log] = dict(
            func_vals=fvals_log,
            smoothed_gaps=sgaps_log,
            ns_gaps=nsgaps_log,
            feasibility=feas_log,
        )
        print(
            f"  [log]   beta0={beta0_log}  f={fvals_log[-1]:.4f}  ||x1-x2||={np.sqrt(feas_log[-1]):.4e}  ns_gap={nsgaps_log[-1]:.4e}"
        )

    iters = np.arange(1, n_steps + 1)
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    plot_order = sorted(beta0_values, reverse=True)
    colors = plt.cm.hsv(np.linspace(0, 0.9, len(plot_order)))
    plot_stride = 10
    plot_idx = np.arange(plot_stride - 1, n_steps, plot_stride)
    plot_iters = iters[plot_idx]

    def b0_label(b0):
        if b0 == inv_L:
            return rf"$\beta_0 = 1/L_{{\nabla f}}$"
        else:
            return rf"$\beta_0 = {b0}$"

    metric_specs = [
        (
            "Smoothed gaps",
            r"$\frac{1}{k}\sum\limits_{j=0}^{k-1}\mathrm{gap}^{\beta_j}(\mathbf{x}_j)$",
            r"$\min\limits_{0\leq j\leq k-1}\mathrm{gap}^{\beta_j}(\mathbf{x}_j)$",
            lambda R: R["smoothed_gaps"],
        ),
        (
            "Feasibility",
            r"$\frac{1}{k}\sum\limits_{j=0}^{k-1}\mathrm{dist}_{\mathcal{D}}(T\mathbf{x}_j)$",
            r"$\min\limits_{0\leq j\leq k-1}\mathrm{dist}_{\mathcal{D}}(T\mathbf{x}_j)$",
            lambda R: np.sqrt(np.maximum(R["feasibility"], 0.0)),
        ),
        (
            "Nonsmooth gaps",
            r"$\frac{1}{k}\sum\limits_{j=0}^{k-1}|\widetilde{\mathrm{gap}}(\mathbf{x}_j)|$",
            r"$\min\limits_{0\leq j\leq k-1}|\widetilde{\mathrm{gap}}(\mathbf{x}_j)|$",
            lambda R: np.abs(R["ns_gaps"]),
        ),
    ]

    for idx, beta0 in enumerate(plot_order):
        c = colors[idx]
        lab = b0_label(beta0)
        lw = 2.0 if beta0 == inv_L else 1.2
        zorder = 10 if beta0 == 4.0 else 2

        for row, (_, _, _, metric_fn) in enumerate(metric_specs):
            y_power = metric_fn(results[beta0])
            y_log = metric_fn(results_log[beta0])

            avg_power = np.cumsum(y_power) / iters
            avg_log = np.cumsum(y_log) / iters
            min_power = np.minimum.accumulate(y_power)
            min_log = np.minimum.accumulate(y_log)

            axs[row, 0].semilogy(
                plot_iters,
                avg_power[plot_idx],
                color=c,
                label=lab,
                alpha=1.0,
                linewidth=lw,
                zorder=zorder,
            )
            axs[row, 0].semilogy(
                plot_iters,
                avg_log[plot_idx],
                color=c,
                ls="--",
                alpha=1.0,
                linewidth=lw,
                zorder=zorder,
            )
            axs[row, 1].semilogy(
                plot_iters,
                min_power[plot_idx],
                color=c,
                label=lab,
                alpha=1.0,
                linewidth=lw,
                zorder=zorder,
            )
            axs[row, 1].semilogy(
                plot_iters,
                min_log[plot_idx],
                color=c,
                ls="--",
                alpha=1.0,
                linewidth=lw,
                zorder=zorder,
            )

    for row, (title, avg_ylabel, min_ylabel, _) in enumerate(metric_specs):
        axs[row, 0].set_title(f"Average {title.lower()}", fontsize=titlefontsize)
        axs[row, 1].set_title(f"Minimum {title.lower()}", fontsize=titlefontsize)
        axs[row, 0].set_ylabel(avg_ylabel, fontsize=labelfontsize)
        axs[row, 1].set_ylabel(min_ylabel, fontsize=labelfontsize)

        for col in range(2):
            axs[row, col].set_xlabel("Iteration $k$", fontsize=labelfontsize)
            axs[row, col].grid(True, alpha=0.3)
            handles, _ = axs[row, col].get_legend_handles_labels()
            axs[row, col].legend(
                handles=handles,
                fontsize=8,
                loc="lower left",
                ncol=1,
            )

    axs[0, 0].set_ylim(6e-2, 2e0)
    axs[0, 1].set_ylim(8e-3, 6e-1)
    axs[1, 0].set_ylim(6e-2, 9e-1)
    axs[1, 1].set_ylim(1e-2, 9e-1)
    axs[2, 0].set_ylim(8e-2, 1.5e0)
    axs[2, 1].set_ylim(5e-7, 2e-1)

    fig.suptitle(
        f"Nonconvex splitting,  $n={nn},\\; {n_neg}$ neg.\\ eigs",
        fontsize=suptitlefontsize,
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(os.path.dirname(__file__), "paper_fig3_splitting.pdf"), dpi=150
    )
    plt.close(fig)
    print("Saved paper_fig3_splitting.pdf")


def figure2():
    figure3()


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 1: Nonneg MF")
    print("=" * 60)
    figure1()
    print()
    print("=" * 60)
    print("Figure 3: Nonconvex splitting")
    print("=" * 60)
    figure3()
