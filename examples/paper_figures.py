"""
Generate all figures for Section 4 of the paper.

Figure 1: Nonneg MF (indicator) — beta_0 sweep + schedule comparison
Figure 2: Trend filtering (Lipschitz) — beta_0 sweep + recovered U
Figure 3: Nonconvex splitting — smoothed gaps, feasibility, nonsmooth gap
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from frank_wolfe.algorithms.nono import NoNoFrankWolfe


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: estimate Lipschitz constant of gradient on constraint set
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_lipschitz(obj, lmo, dim, n_samples=500, seed=99):
    rng = np.random.default_rng(seed)
    pool = np.array([lmo(rng.standard_normal(dim)) for _ in range(20)])
    L_est = 0.0
    for _ in range(n_samples):
        w1, w2 = rng.dirichlet(np.ones(len(pool))), rng.dirichlet(np.ones(len(pool)))
        z1, z2 = w1 @ pool, w2 @ pool
        diff = np.linalg.norm(z1 - z2)
        if diff < 1e-12: continue
        L_est = max(L_est, np.linalg.norm(obj.gradient(z1) - obj.gradient(z2)) / diff)
    return L_est


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Nonneg MF
# ═══════════════════════════════════════════════════════════════════════════════

def figure1():
    from examples.nonneg_matrix_factorization import (
        MatrixFactorizationObjective,
        create_spectral_ball_product_lmo,
        nonneg_prox,
        generate_nonneg_mf_problem,
    )
    from frank_wolfe.algorithms.base import FrankWolfe

    # --- Log schedule variant ---
    class NoNoFW_Log(FrankWolfe):
        def __init__(self, obj, lmo, prox, otype):
            super().__init__(obj, lmo)
            self.prox = prox; self.objective_type = otype; self.ns_gaps = None
        def run(self, x0, beta0=1.0, n_steps=100):
            self.x = self.lmo(self.objective.gradient(x0))
            self.func_vals = np.zeros(n_steps); self.gaps = np.zeros(n_steps)
            self.ns_gaps = np.zeros(n_steps); self.num_oracles = np.zeros(n_steps)
            for i in tqdm(range(n_steps), desc="NSFW (log)"):
                beta = beta0 / np.log(i + 2); step_size = 2.0 / np.sqrt(i + 2)
                grad = self.objective.gradient(self.x)
                Tx = self.objective.linear_operator(self.x)
                mg = self.objective.linear_operator_adjoint(Tx - self.prox(Tx, beta)) / beta
                cg = grad + mg; direction = self.lmo(cg); self.num_oracles[i] += 1
                self.gaps[i] = np.sum(cg * (self.x - direction))
                self.func_vals[i] = self.objective.evaluate(self.x)
                self.ns_gaps[i] = 0.5 * np.linalg.norm((Tx - self.prox(Tx, beta)).flatten()) ** 2
                self.x = (1 - step_size) * self.x + step_size * direction
            self.num_oracles = np.cumsum(self.num_oracles)

    m, n, r = 100, 100, 20
    n_steps = 2000
    margin = 1.05
    seed = 42

    X_star, U_star, V_star, tau_U, tau_V = generate_nonneg_mf_problem(
        m, n, r, margin=margin, seed=seed
    )
    X_star_fro = np.linalg.norm(X_star, "fro")
    obj = MatrixFactorizationObjective(X_star, m, n, r)
    lmo = create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r)
    x0 = np.zeros(m * r + n * r)

    # Estimate L_{nabla f}
    L = estimate_lipschitz(obj, lmo, m * r + n * r)
    print(f"Estimated L_{{nabla f}} = {L:.2f}")

    beta0_values = [0.2, 0.5, 1.0, 2.0, 5.0, L]
    results = {}
    for beta0 in beta0_values:
        nsfw = NoNoFrankWolfe(obj, lmo, nonneg_prox, "indicator")
        nsfw.run(x0, beta0=beta0, n_steps=n_steps)
        rel_err = np.sqrt(2.0 * np.maximum(nsfw.func_vals, 0)) / X_star_fro
        results[beta0] = dict(gaps=nsfw.gaps.copy(), ns_gaps=nsfw.ns_gaps.copy(),
                              func_vals=nsfw.func_vals.copy(), rel_error=rel_err)
        U_f, V_f = obj._unpack(nsfw.x)
        lab_str = f"L={L:.0f}" if beta0 == L else f"{beta0}"
        print(f"  beta0={lab_str:>8s}  rel_err={rel_err[-1]:.4f}  "
              f"dist_D^2={2*nsfw.ns_gaps[-1]:.4e}")

    # Schedule comparison at beta0=1
    beta0_shared = 1.0
    nsfw_log = NoNoFW_Log(obj, lmo, nonneg_prox, "indicator")
    nsfw_log.run(x0, beta0=beta0_shared, n_steps=n_steps)
    res_log = dict(gaps=nsfw_log.gaps.copy(), ns_gaps=nsfw_log.ns_gaps.copy())

    # --- Plot ---
    iters = np.arange(1, n_steps + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Plot from largest beta0 to smallest (so smallest is on top)
    plot_order = sorted(beta0_values, reverse=True)
    n_vals = len(plot_order)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_vals))

    def b0_label(b):
        if b == L:
            return rf"$\beta_0 = L_{{\nabla f}} \approx {L:.0f}$"
        elif b >= 1 and b == int(b):
            return rf"$\beta_0 = {int(b)}$"
        else:
            return rf"$\beta_0 = {b}$"

    for idx, beta0 in enumerate(plot_order):
        c = colors[idx]
        R = results[beta0]
        lab = b0_label(beta0)
        lw = 2.0 if beta0 == L else 1.2

        # [0,0] Rel error
        axs[0, 0].semilogy(iters, R["rel_error"], color=c, label=lab, alpha=0.5, linewidth=lw)

        # [0,1] Feasibility
        axs[0, 1].semilogy(iters, 2.0 * R["ns_gaps"], color=c, label=lab, alpha=0.5, linewidth=lw)

        # [1,0] Min and avg smoothed gaps
        min_g = np.minimum.accumulate(R["gaps"])
        avg_g = np.cumsum(R["gaps"]) / iters
        axs[1, 0].semilogy(iters, min_g, color=c, label=lab, alpha=0.5, linewidth=lw)
        axs[1, 0].semilogy(iters, avg_g, color=c, linestyle="--", alpha=0.3, linewidth=lw)

    # Reference line
    C_ref = (2.0 * results[1.0]["ns_gaps"][n_steps // 4]) * (n_steps // 4 + 1) ** 0.25
    axs[0, 1].semilogy(iters, C_ref / iters**0.25, "--", color="gray",
                        label=r"$O(k^{-1/4})$", linewidth=2)

    axs[0, 0].set_title("Relative reconstruction error")
    axs[0, 0].set_xlabel("Iteration $k$")
    axs[0, 0].set_ylabel(r"$\|U_k V_k^\top - X^\star\|_F / \|X^\star\|_F$")
    axs[0, 0].legend(fontsize=7)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].set_title(r"Feasibility: $\mathrm{dist}_D^2(x_k)$")
    axs[0, 1].set_xlabel("Iteration $k$")
    axs[0, 1].set_ylabel(r"$\mathrm{dist}_D^2(x_k)$")
    axs[0, 1].legend(fontsize=7)
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].set_title("Min and avg smoothed gaps")
    axs[1, 0].set_xlabel("Iteration $k$")
    axs[1, 0].set_ylabel("Smoothed gap")
    handles, _ = axs[1, 0].get_legend_handles_labels()
    handles += [Line2D([0], [0], color="k", ls="-", label="solid = min"),
                Line2D([0], [0], color="k", ls="--", label="dashed = avg")]
    axs[1, 0].legend(handles=handles, fontsize=6, ncol=2)
    axs[1, 0].grid(True, alpha=0.3)

    # [1,1] Schedule comparison
    res_pow = results[beta0_shared]
    avg_pow = np.cumsum(res_pow["gaps"]) / iters
    min_pow = np.minimum.accumulate(res_pow["gaps"])
    avg_log = np.cumsum(res_log["gaps"]) / iters
    min_log = np.minimum.accumulate(res_log["gaps"])

    axs[1, 1].semilogy(iters, avg_pow, color="C0",
                        label=rf"$\beta_k = {beta0_shared}/(k\!+\!1)^{{1/4}}$ (avg)")
    axs[1, 1].semilogy(iters, min_pow, color="C0", ls="--",
                        label=rf"$\beta_k = {beta0_shared}/(k\!+\!1)^{{1/4}}$ (min)")
    axs[1, 1].semilogy(iters, avg_log, color="C3",
                        label=rf"$\beta_k = {beta0_shared}/\log(k\!+\!2)$ (avg)")
    axs[1, 1].semilogy(iters, min_log, color="C3", ls="--",
                        label=rf"$\beta_k = {beta0_shared}/\log(k\!+\!2)$ (min)")
    C_ref2 = avg_pow[n_steps // 4] * (n_steps // 4 + 1) ** 0.25
    axs[1, 1].semilogy(iters, C_ref2 / iters**0.25, "--", color="gray", lw=2,
                        label=r"$O(k^{-1/4})$")
    axs[1, 1].set_title(rf"Schedule comparison ($\beta_0 = {beta0_shared}$)")
    axs[1, 1].set_xlabel("Iteration $k$")
    axs[1, 1].set_ylabel("Smoothed gap")
    axs[1, 1].legend(fontsize=7)
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Nonneg matrix factorization (Assumption 1.4(I))\n"
        f"$m={m},\\; n={n},\\; r={r},\\; N={n_steps}$",
        fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "paper_fig1_nonneg_mf.pdf"), dpi=150)
    plt.close(fig)
    print("Saved paper_fig1_nonneg_mf.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Trend filtering
# ═══════════════════════════════════════════════════════════════════════════════

def figure2():
    from examples.trend_filtering_mf import (
        TrendFilteredMFObjective,
        create_spectral_ball_product_lmo,
        l1_prox,
        l1_minimal_norm,
        generate_piecewise_constant_mf,
        finite_diff_matrix,
    )

    # ── SCAD penalty ──
    def make_scad(lam, a):
        assert a > 2
        rho = 1.0 / (a - 1)
        def prox(z, beta):
            assert beta < a - 1
            y = np.abs(z); r = np.zeros_like(z)
            m1 = y <= lam * (1 + beta)
            r[m1] = np.maximum(y[m1] - beta * lam, 0)
            m2 = (y > lam * (1 + beta)) & (y <= a * lam)
            r[m2] = ((a - 1) * y[m2] - beta * a * lam) / (a - 1 - beta)
            m3 = y > a * lam
            r[m3] = y[m3]
            return np.sign(z) * r
        def deriv(z):
            t = np.abs(z); s = np.sign(z); r = np.zeros_like(z)
            m1 = (t > 0) & (t <= lam); r[m1] = lam * s[m1]
            m2 = (t > lam) & (t <= a * lam)
            r[m2] = s[m2] * (a * lam - t[m2]) / (a - 1)
            return r
        return prox, deriv, rho

    # ── MCP penalty ──
    def make_mcp(lam, gamma):
        assert gamma > 1
        rho = 1.0 / gamma
        def prox(z, beta):
            assert beta < gamma
            y = np.abs(z); r = np.zeros_like(z)
            m1 = (y > beta * lam) & (y <= gamma * lam)
            r[m1] = (y[m1] - beta * lam) / (1 - beta / gamma)
            m2 = y > gamma * lam
            r[m2] = y[m2]
            return np.sign(z) * r
        def deriv(z):
            t = np.abs(z); s = np.sign(z); r = np.zeros_like(z)
            m1 = (t > 0) & (t <= gamma * lam)
            r[m1] = s[m1] * (lam - t[m1] / gamma)
            return r
        return prox, deriv, rho

    m, n, r = 100, 100, 5
    n_steps = 2000
    margin = 1.05
    seed = 42
    beta0 = 1.0
    scad_lam, scad_a = 0.5, 3.7
    mcp_lam, mcp_gamma = 0.5, 3.0

    X_star, U_star, V_star, tau_U, tau_V = generate_piecewise_constant_mf(
        m, n, r, n_blocks=5, margin=margin, seed=seed
    )
    X_star_fro = np.linalg.norm(X_star, "fro")

    lmo = create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r)
    x0 = np.zeros(m * r + n * r)

    scad_prox, scad_deriv, scad_rho = make_scad(scad_lam, scad_a)
    mcp_prox, mcp_deriv, mcp_rho = make_mcp(mcp_lam, mcp_gamma)

    penalties = {
        r"$\ell_1$": (l1_prox, l1_minimal_norm, 0.0),
        rf"SCAD ($a={scad_a}$)": (scad_prox, scad_deriv, scad_rho),
        rf"MCP ($\gamma={mcp_gamma}$)": (mcp_prox, mcp_deriv, mcp_rho),
    }

    results = {}
    for name, (prox_fn, deriv_fn, rho) in penalties.items():
        obj = TrendFilteredMFObjective(X_star, m, n, r)
        obj.minimal_norm_selection = deriv_fn
        nsfw = NoNoFrankWolfe(obj, lmo, prox_fn, "lipschitz")
        nsfw.run(x0, beta0=beta0, n_steps=n_steps)
        rel_err = np.sqrt(2.0 * np.maximum(nsfw.func_vals, 0)) / X_star_fro
        results[name] = dict(gaps=nsfw.gaps.copy(), rel_error=rel_err)
        print(f"  {name:20s}  rho={rho:.4f}  rel_err={rel_err[-1]:.4f}  "
              f"min_gap={np.min(nsfw.gaps):.2e}")

    iters = np.arange(1, n_steps + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["C0", "C1", "C3"]

    # [0] Smoothed gaps
    for idx, (name, R) in enumerate(results.items()):
        c = colors[idx]
        min_g = np.minimum.accumulate(R["gaps"])
        avg_g = np.cumsum(R["gaps"]) / iters
        axs[0].semilogy(iters, min_g, color=c, label=name)
        axs[0].semilogy(iters, avg_g, color=c, ls="--", alpha=0.5)

    first_name = list(results.keys())[0]
    C_ref = (np.cumsum(results[first_name]["gaps"]) / iters)[n_steps // 4] \
            * (n_steps // 4 + 1) ** 0.25
    axs[0].semilogy(iters, C_ref / iters**0.25, "--", color="gray", lw=2,
                    label=r"$O(k^{-1/4})$")
    handles, _ = axs[0].get_legend_handles_labels()
    handles += [Line2D([0], [0], color="k", ls="-", label="solid = min"),
                Line2D([0], [0], color="k", ls="--", label="dashed = avg")]
    axs[0].legend(handles=handles, fontsize=7)
    axs[0].set_title("Smoothed gaps")
    axs[0].set_xlabel("Iteration $k$")
    axs[0].set_ylabel("Smoothed gap")
    axs[0].grid(True, alpha=0.3)

    # [1] Relative error
    for idx, (name, R) in enumerate(results.items()):
        c = colors[idx]
        axs[1].semilogy(iters, R["rel_error"], color=c, label=name, alpha=0.8)
    axs[1].set_title("Relative reconstruction error")
    axs[1].set_xlabel("Iteration $k$")
    axs[1].set_ylabel(r"$\|U_k V_k^\top - X^\star\|_F / \|X^\star\|_F$")
    axs[1].legend(fontsize=7)
    axs[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Trend-filtered matrix factorization (Assumption 1.4(II))\n"
        f"$m={m},\\; n={n},\\; r={r},\\; N={n_steps},\\;"
        f"\\beta_0={beta0},\\; \\lambda={scad_lam}$",
        fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "paper_fig2_trend_filter.pdf"), dpi=150)
    plt.close(fig)
    print("Saved paper_fig2_trend_filter.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Nonconvex splitting
# ═══════════════════════════════════════════════════════════════════════════════

def figure3():
    from examples.l1_splitting_nonconvex import (
        IndefiniteQuadraticSplitting,
        create_product_l1_lmo,
        zero_prox,
        nonsmooth_gap,
        generate_indefinite_quadratic,
        run_nsfw_splitting,
    )

    nn = 50
    n_steps = 5000
    seed = 42

    c1 = np.zeros(nn); c1[0] = 1.0
    c2 = np.zeros(nn); c2[0] = -1.0
    radius = 2.0

    Q, b, eigvals = generate_indefinite_quadratic(nn, frac_neg=0.3, seed=seed)
    L = np.linalg.norm(Q, ord=2)

    n_pos = np.sum(eigvals > 0)
    n_neg = np.sum(eigvals < 0)
    print(f"Splitting: n={nn}, {n_pos} pos / {n_neg} neg eigenvalues, "
          f"||Q||_op = L = {L:.2f}")

    obj = IndefiniteQuadraticSplitting(Q, b, nn)
    lmo = create_product_l1_lmo(c1, c2, radius, nn)

    beta0_values = [0.5, 1.0, 2.0, L]
    results = {}
    for beta0 in beta0_values:
        x_final, fvals, sgaps, nsgaps, feas = run_nsfw_splitting(
            obj, lmo, n_steps, beta0=beta0
        )
        results[beta0] = dict(
            func_vals=fvals, smoothed_gaps=sgaps,
            ns_gaps=nsgaps, feasibility=feas,
        )
        lab_str = f"L={L:.2f}" if beta0 == L else f"{beta0}"
        print(f"  beta0={lab_str:>8s}  f={fvals[-1]:.4f}  "
              f"||x1-x2||^2={feas[-1]:.4e}  ns_gap={nsgaps[-1]:.4e}")

    iters = np.arange(1, n_steps + 1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

    # Plot from largest to smallest beta0
    plot_order = sorted(beta0_values, reverse=True)
    n_vals = len(plot_order)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_vals))

    def b0_label(b0):
        if b0 == L:
            return rf"$\beta_0 = L_{{\nabla f}} \approx {L:.1f}$"
        else:
            return rf"$\beta_0 = {b0}$"

    for idx, beta0 in enumerate(plot_order):
        c = colors[idx]
        R = results[beta0]
        lab = b0_label(beta0)
        lw = 2.0 if beta0 == L else 1.2

        # [0] Smoothed gaps
        min_g = np.minimum.accumulate(R["smoothed_gaps"])
        avg_g = np.cumsum(R["smoothed_gaps"]) / iters
        axs[0].semilogy(iters, min_g, color=c, label=lab, alpha=0.5, linewidth=lw)
        axs[0].semilogy(iters, avg_g, color=c, ls="--", alpha=0.3, linewidth=lw)

        # [1] Feasibility
        axs[1].semilogy(iters, R["feasibility"], color=c, label=lab, alpha=0.5, linewidth=lw)

        # [2] Nonsmooth gap: just plot |gap|
        abs_ns = np.abs(R["ns_gaps"])
        axs[2].semilogy(iters, abs_ns, color=c, alpha=0.3, linewidth=0.5)
        min_abs = np.minimum.accumulate(abs_ns)
        axs[2].semilogy(iters, min_abs, color=c, label=lab, linewidth=lw, alpha=0.8)

    # Reference lines
    C_ref = np.minimum.accumulate(results[1.0]["smoothed_gaps"])[n_steps // 4] \
            * (n_steps // 4 + 1) ** 0.25
    axs[0].semilogy(iters, C_ref / iters**0.25, "--", color="gray", lw=2,
                    label=r"$O(k^{-1/4})$")

    C_feas = results[1.0]["feasibility"][n_steps // 4] * (n_steps // 4 + 1) ** 0.25
    axs[1].semilogy(iters, C_feas / iters**0.25, "--", color="gray", lw=2,
                    label=r"$O(k^{-1/4})$")

    handles0, _ = axs[0].get_legend_handles_labels()
    handles0 += [Line2D([0], [0], color="k", ls="-", label="solid = min"),
                 Line2D([0], [0], color="k", ls="--", label="dashed = avg")]
    axs[0].legend(handles=handles0, fontsize=7)
    axs[0].set_title("Smoothed gaps")
    axs[0].set_xlabel("Iteration $k$")
    axs[0].set_ylabel("Smoothed gap")
    axs[0].grid(True, alpha=0.3)

    axs[1].set_title(r"Feasibility: $\|x_{1,k} - x_{2,k}\|^2$")
    axs[1].set_xlabel("Iteration $k$")
    axs[1].set_ylabel(r"$\|x_1 - x_2\|^2$")
    axs[1].legend(fontsize=7)
    axs[1].grid(True, alpha=0.3)

    axs[2].set_title(
        r"Nonsmooth gap $|\widetilde{\mathrm{gap}}(\mathbf{x}_k)|$")
    axs[2].set_xlabel("Iteration $k$")
    axs[2].set_ylabel(r"$|\widetilde{\mathrm{gap}}|$")
    axs[2].legend(fontsize=7)
    axs[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Nonconvex splitting (Assumption 1.4(I))\n"
        f"$n={nn},\\; N={n_steps},\\; "
        f"{n_neg}$ negative eigenvalues,  "
        f"$L_{{\\nabla f}} = \\|Q\\|_{{\\mathrm{{op}}}} = {L:.2f}$",
        fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), "paper_fig3_splitting.pdf"), dpi=150)
    plt.close(fig)
    print("Saved paper_fig3_splitting.pdf")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Figure 1: Nonneg MF")
    print("=" * 60)
    figure1()
    print()
    print("=" * 60)
    print("Figure 2: Trend filtering")
    print("=" * 60)
    figure2()
    print()
    print("=" * 60)
    print("Figure 3: Nonconvex splitting")
    print("=" * 60)
    figure3()
