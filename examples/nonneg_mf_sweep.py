"""
Sweep over beta0 values and compare smoothing schedules.

Layout:
  [0,0] Relative reconstruction error vs beta0
  [0,1] Feasibility dist_D^2 vs beta0
  [1,0] Min and avg smoothed gaps vs beta0 (combined)
  [1,1] Schedule comparison: beta_k = beta0/(k+1)^{1/4} vs beta0/log(k+2)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from examples.nonneg_matrix_factorization import (
    MatrixFactorizationObjective,
    create_spectral_ball_product_lmo,
    generate_nonneg_mf_problem,
    nonneg_prox,
)
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.algorithms.nono import NoNoFrankWolfe
from tqdm import tqdm


class NoNoFrankWolfe_LogSchedule(FrankWolfe):
    """
    Same as NoNoFrankWolfe but with the 'bad' smoothing schedule
    beta_k = beta_0 / log(k+2) instead of beta_0 / (k+1)^{1/4}.
    Step size also uses the old 2/sqrt(k+2).
    """

    def __init__(self, objective_fn, lmo_fn, prox_fn, objective_type):
        super().__init__(objective_fn, lmo_fn)
        self.prox = prox_fn
        self.objective_type = objective_type
        self.ns_gaps = None

    def run(self, x0, beta0=1.0, n_steps=int(1e2)):
        self.x = self.lmo(self.objective.gradient(x0))
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.ns_gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)

        for i in tqdm(range(n_steps), desc="NSFW (log schedule) Progress"):
            # "Bad" schedules from old nono.py
            beta = beta0 / np.log(i + 2)
            step_size = 2.0 / np.sqrt(i + 2)

            grad = self.objective.gradient(self.x)
            Tx = self.objective.linear_operator(self.x)
            moreau_grad = (
                self.objective.linear_operator_adjoint(Tx - self.prox(Tx, beta)) / beta
            )
            combined_grad = grad + moreau_grad

            direction = self.lmo(combined_grad)
            self.num_oracles[i] += 1

            gap = np.sum(combined_grad * (self.x - direction))
            self.gaps[i] = gap
            self.func_vals[i] = self.objective.evaluate(self.x)

            if self.objective_type == "indicator":
                ns_gap = 0.5 * np.linalg.norm((Tx - self.prox(Tx, beta)).flatten()) ** 2
            elif self.objective_type == "lipschitz":
                ns_grad = self.objective.linear_operator_adjoint(
                    self.objective.minimal_norm_selection(Tx)
                )
                combined_ns_grad = grad + ns_grad
                ns_direction = self.lmo(combined_ns_grad)
                ns_gap = np.sum(combined_ns_grad * (self.x - ns_direction))
            else:
                raise ValueError(f"Unknown objective type: {self.objective_type}")

            self.ns_gaps[i] = ns_gap
            self.x = (1 - step_size) * self.x + step_size * direction

        self.num_oracles = np.cumsum(self.num_oracles)


# ─── Problem setup ────────────────────────────────────────────────────────────

m, n, r = 100, 100, 20
n_steps = 2000
seed = 42
margin = 1.05

X_star, U_star, V_star, tau_U, tau_V = generate_nonneg_mf_problem(
    m, n, r, margin=margin, seed=seed
)
X_star_fro = np.linalg.norm(X_star, "fro")

svs = np.linalg.svd(X_star, compute_uv=False)
print(f"Nonzero singular values of X*: {svs[:r].round(2)}")
print(f"Condition number (rank-r): {svs[0] / svs[r - 1]:.2f}")
print(f"||X*||_F = {X_star_fro:.2f}")
print(f"tau_U = {tau_U:.4f},  tau_V = {tau_V:.4f}")
print()

obj = MatrixFactorizationObjective(X_star, m, n, r)
lmo = create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r)
x0 = np.zeros(m * r + n * r)

# ─── Part 1: beta0 sweep (correct schedule) ──────────────────────────────────

beta0_values = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0]
results = {}

for beta0 in beta0_values:
    nsfw = NoNoFrankWolfe(obj, lmo, nonneg_prox, "indicator")
    nsfw.run(x0, beta0=beta0, n_steps=n_steps)
    rel_err = np.sqrt(2.0 * np.maximum(nsfw.func_vals, 0)) / X_star_fro
    results[beta0] = {
        "gaps": nsfw.gaps.copy(),
        "ns_gaps": nsfw.ns_gaps.copy(),
        "func_vals": nsfw.func_vals.copy(),
        "rel_error": rel_err,
    }
    U_f, V_f = obj._unpack(nsfw.x)
    print(
        f"[power]  beta0={beta0:10.4f}  |  rel_err={rel_err[-1]:.4f}  "
        f"dist_D^2={2 * nsfw.ns_gaps[-1]:.4e}  "
        f"neg_frac_U={np.mean(U_f < 0):.3f}  neg_frac_V={np.mean(V_f < 0):.3f}"
    )

# ─── Part 2: schedule comparison at shared beta0 ─────────────────────────────

beta0_shared = 1.0

# Correct schedule (already computed)
res_power = results[beta0_shared]

# Log schedule
nsfw_log = NoNoFrankWolfe_LogSchedule(obj, lmo, nonneg_prox, "indicator")
nsfw_log.run(x0, beta0=beta0_shared, n_steps=n_steps)
rel_err_log = np.sqrt(2.0 * np.maximum(nsfw_log.func_vals, 0)) / X_star_fro
res_log = {
    "gaps": nsfw_log.gaps.copy(),
    "ns_gaps": nsfw_log.ns_gaps.copy(),
    "func_vals": nsfw_log.func_vals.copy(),
    "rel_error": rel_err_log,
}
U_f, V_f = obj._unpack(nsfw_log.x)
print(
    f"[log]    beta0={beta0_shared:10.4f}  |  rel_err={rel_err_log[-1]:.4f}  "
    f"dist_D^2={2 * nsfw_log.ns_gaps[-1]:.4e}  "
    f"neg_frac_U={np.mean(U_f < 0):.3f}  neg_frac_V={np.mean(V_f < 0):.3f}"
)

# ─── Plotting ─────────────────────────────────────────────────────────────────

iters = np.arange(1, n_steps + 1)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(beta0_values)))

for idx, beta0 in enumerate(beta0_values):
    c = colors[idx]
    if beta0 < 0.01:
        lab = rf"$\beta_0 = 10^{{{int(np.log10(beta0))}}}$"
    elif beta0 >= 1 and beta0 == int(beta0):
        lab = rf"$\beta_0 = {int(beta0)}$"
    else:
        lab = rf"$\beta_0 = {beta0}$"
    R = results[beta0]

    # [0,0] Relative reconstruction error
    axs[0, 0].semilogy(iters, R["rel_error"], color=c, label=lab, alpha=0.8)

    # [0,1] Feasibility
    axs[0, 1].semilogy(iters, 2.0 * R["ns_gaps"], color=c, label=lab, alpha=0.8)

    # [1,0] Min and avg smoothed gaps (combined)
    min_gaps = np.minimum.accumulate(R["gaps"])
    avg_gaps = np.cumsum(R["gaps"]) / iters
    axs[1, 0].semilogy(iters, min_gaps, color=c, label=lab, alpha=0.8)
    axs[1, 0].semilogy(
        iters,
        avg_gaps,
        color=c,
        linestyle="--",
        alpha=0.5,
    )

# Reference lines
C_feas = (2.0 * results[1.0]["ns_gaps"][n_steps // 4]) * (n_steps // 4 + 1) ** 0.25
axs[0, 1].semilogy(
    iters, C_feas / iters**0.25, "--", color="gray", label=r"$O(k^{-1/4})$"
)

C_gap = (
    np.minimum.accumulate(results[1.0]["gaps"])[n_steps // 4]
    * (n_steps // 4 + 1) ** 0.25
)
axs[1, 0].semilogy(
    iters, C_gap / iters**0.25, "--", color="gray", linewidth=2, label=r"$O(k^{-1/4})$"
)

# ─── [1,1] Schedule comparison ────────────────────────────────────────────────

avg_power = np.cumsum(res_power["gaps"]) / iters
avg_log = np.cumsum(res_log["gaps"]) / iters
min_power = np.minimum.accumulate(res_power["gaps"])
min_log = np.minimum.accumulate(res_log["gaps"])

axs[1, 1].semilogy(
    iters,
    avg_power,
    color="C0",
    label=rf"$\beta_k = {beta0_shared}/(k\!+\!1)^{{1/4}}$ (avg)",
)
axs[1, 1].semilogy(
    iters,
    min_power,
    color="C0",
    linestyle="--",
    label=rf"$\beta_k = {beta0_shared}/(k\!+\!1)^{{1/4}}$ (min)",
)
axs[1, 1].semilogy(
    iters,
    avg_log,
    color="C3",
    label=rf"$\beta_k = {beta0_shared}/\log(k\!+\!2)$ (avg)",
)
axs[1, 1].semilogy(
    iters,
    min_log,
    color="C3",
    linestyle="--",
    label=rf"$\beta_k = {beta0_shared}/\log(k\!+\!2)$ (min)",
)

# Reference O(k^{-1/4})
C_ref = avg_power[n_steps // 4] * (n_steps // 4 + 1) ** 0.25
axs[1, 1].semilogy(
    iters,
    C_ref / iters**0.25,
    "--",
    color="gray",
    linewidth=2,
    label=r"$O(k^{-1/4})$",
)

# ─── Axis labels and titles ───────────────────────────────────────────────────

axs[0, 0].set_title("Relative reconstruction error")
axs[0, 0].set_xlabel("Iteration $k$")
axs[0, 0].set_ylabel(r"$\|U_k V_k^\top - X^*\|_F \;/\; \|X^*\|_F$")
axs[0, 0].legend(fontsize=6)
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].set_title(r"Feasibility: $\mathrm{dist}_D^2(x_k)$")
axs[0, 1].set_xlabel("Iteration $k$")
axs[0, 1].set_ylabel(r"$\mathrm{dist}_D^2(x_k)$")
axs[0, 1].legend(fontsize=6)
axs[0, 1].grid(True, alpha=0.3)

axs[1, 0].set_title("Min and avg smoothed gaps")
axs[1, 0].set_xlabel("Iteration $k$")
axs[1, 0].set_ylabel("Smoothed gap")
# Add manual style entries for solid=min, dashed=avg
from matplotlib.lines import Line2D

style_handles = [
    Line2D([0], [0], color="black", linestyle="-", label="solid = min"),
    Line2D([0], [0], color="black", linestyle="--", label="dashed = avg"),
]
handles, labels = axs[1, 0].get_legend_handles_labels()
axs[1, 0].legend(handles=handles + style_handles, fontsize=6, ncol=2)
axs[1, 0].grid(True, alpha=0.3)

axs[1, 1].set_title(rf"Schedule comparison ($\beta_0 = {beta0_shared}$)")
axs[1, 1].set_xlabel("Iteration $k$")
axs[1, 1].set_ylabel("Smoothed gap")
axs[1, 1].legend(fontsize=7)
axs[1, 1].grid(True, alpha=0.3)

fig.suptitle(
    f"NSFW for Nonneg Matrix Factorization\n"
    f"$m={m},\\; n={n},\\; r={r},\\; N={n_steps},\\; "
    f"\\mathrm{{margin}}={margin}$",
    fontsize=13,
)
plt.tight_layout()
fig.savefig("/home/claude/experiments/nonneg_mf_beta_sweep.png", dpi=150)
plt.close(fig)
print("\nSaved to experiments/nonneg_mf_beta_sweep.png")
