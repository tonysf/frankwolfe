"""
Nonnegative low-rank matrix factorization via NSFW (Algorithm 1).

Problem:
    min_{U, V}  f(U, V) + g(U, V)
    s.t.        ||U||_op <= tau_U,  ||V||_op <= tau_V

where:
    f(U, V) = 0.5 * ||U V^T - X*||_F^2
    g(U, V) = iota_{>= 0}(U, V)    (indicator of componentwise nonnegativity)

This is an instance of (P) with T = Id, C = spectral-norm-ball product,
and g = iota_D with D = {(U,V) : U >= 0, V >= 0}.

The NSFW algorithm smooths g via its Moreau envelope (here: squared distance
to the nonneg orthant, scaled by 1/(2 beta)) and applies one FW step per
iteration with schedules gamma_k = 1/(k+1)^{1/2}, beta_k = beta0/(k+1)^{1/4}.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frank_wolfe import NoNoFrankWolfe
from frank_wolfe.core.objective import ObjectiveFunction


# ─── Objective ────────────────────────────────────────────────────────────────

class MatrixFactorizationObjective(ObjectiveFunction):
    """
    f(U, V) = 0.5 * ||U V^T - X*||_F^2

    Decision variable: x = [vec(U); vec(V)] in R^{(m+n)*r}.
    Linear operator T = Id (nonsmooth term acts directly on (U, V)).
    """

    def __init__(self, X_star, m, n, r):
        super().__init__()
        self.X_star = X_star
        self.m = m
        self.n = n
        self.r = r

    def _unpack(self, x):
        U = x[: self.m * self.r].reshape(self.m, self.r)
        V = x[self.m * self.r :].reshape(self.n, self.r)
        return U, V

    def _pack(self, A, B):
        return np.concatenate([A.ravel(), B.ravel()])

    def evaluate(self, x):
        U, V = self._unpack(x)
        R = U @ V.T - self.X_star
        return 0.5 * np.linalg.norm(R, "fro") ** 2

    def gradient(self, x):
        U, V = self._unpack(x)
        R = U @ V.T - self.X_star
        return self._pack(R @ V, R.T @ U)

    def linear_operator(self, x):
        return x

    def linear_operator_adjoint(self, x):
        return x


# ─── LMO for product of spectral-norm balls ──────────────────────────────────

def create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r):
    """
    LMO over C = { (U,V) : ||U||_op <= tau_U, ||V||_op <= tau_V }.

    For a single spectral-norm ball {M : ||M||_op <= tau}, the LMO is
        argmin_{||S||_op <= tau}  <G, S>  =  -tau * sign(G)
    where sign(G) = U_G @ V_G^T from the (thin) SVD G = U_G Sigma V_G^T,
    i.e., all nonzero singular values are set to 1.

    The product structure means we apply this independently to the U- and V-blocks.
    """

    def lmo(gradient):
        grad_U = gradient[: m * r].reshape(m, r)
        grad_V = gradient[m * r :].reshape(n, r)

        # spectral-norm ball LMO: -tau * matrix_sign(G)
        Uu, _, Vtu = np.linalg.svd(grad_U, full_matrices=False)
        s_U = -tau_U * (Uu @ Vtu)

        Uv, _, Vtv = np.linalg.svd(grad_V, full_matrices=False)
        s_V = -tau_V * (Uv @ Vtv)

        return np.concatenate([s_U.ravel(), s_V.ravel()])

    return lmo


# ─── Prox for nonnegativity indicator ─────────────────────────────────────────

def nonneg_prox(x, beta):
    """prox_{beta * iota_{>= 0}}(x) = max(x, 0), independent of beta."""
    return np.maximum(x, 0)


# ─── Data generation ──────────────────────────────────────────────────────────

def generate_nonneg_mf_problem(m, n, r, margin=1.0, seed=None):
    """
    Generate a nonneg matrix factorization instance.

    Ground truth: X* = U* V*^T  with U*, V* >= 0.
    Constraint radii: tau_U = margin * ||U*||_op, tau_V = margin * ||V*||_op.

    Parameters
    ----------
    m, n : int
        Matrix dimensions.
    r : int
        Rank.
    margin : float
        Multiplicative slack on the spectral-norm constraint.
        margin = 1.0 puts the ground truth on the boundary.
    seed : int or None
        Random seed.

    Returns
    -------
    X_star, U_star, V_star, tau_U, tau_V
    """
    rng = np.random.default_rng(seed)
    U_star = np.abs(rng.standard_normal((m, r)))
    V_star = np.abs(rng.standard_normal((n, r)))
    X_star = U_star @ V_star.T
    tau_U = margin * np.linalg.norm(U_star, ord=2)
    tau_V = margin * np.linalg.norm(V_star, ord=2)
    return X_star, U_star, V_star, tau_U, tau_V


# ─── Experiment ───────────────────────────────────────────────────────────────

def run_experiment(
    m=100, n=100, r=5, margin=1.05, beta0=1.0, n_steps=500, seed=42
):
    # Generate problem
    X_star, U_star, V_star, tau_U, tau_V = generate_nonneg_mf_problem(
        m, n, r, margin=margin, seed=seed
    )
    X_star_fro = np.linalg.norm(X_star, "fro")

    print(f"Problem: m={m}, n={n}, r={r}")
    print(f"||X*||_F = {X_star_fro:.4f}")
    print(f"tau_U = {tau_U:.4f},  tau_V = {tau_V:.4f}")
    print(f"rank(X*) = {np.linalg.matrix_rank(X_star)}")
    print(f"cond(X*) = {np.linalg.cond(X_star):.2f}")
    print(f"beta_0 = {beta0},  n_steps = {n_steps}")
    print()

    # Build components
    obj = MatrixFactorizationObjective(X_star, m, n, r)
    lmo = create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r)
    x0 = np.zeros(m * r + n * r)

    # Run NSFW
    nsfw = NoNoFrankWolfe(obj, lmo, nonneg_prox, "indicator")
    nsfw.run(x0, beta0=beta0, n_steps=n_steps)

    # Derived quantities
    rel_error = np.sqrt(2.0 * np.maximum(nsfw.func_vals, 0)) / X_star_fro
    iters = np.arange(1, n_steps + 1)

    # ── Compute running min and running average of smoothed gaps ──
    min_gaps = np.minimum.accumulate(nsfw.gaps)
    avg_gaps = np.cumsum(nsfw.gaps) / iters

    # ── Negative-entry fraction per iteration (from stored final x only;
    #    we approximate the trajectory via the gap/feasibility arrays) ──

    # ── Print summary ──
    U_final, V_final = obj._unpack(nsfw.x)
    X_rec = U_final @ V_final.T
    final_rel = np.linalg.norm(X_rec - X_star, "fro") / X_star_fro
    frac_neg_U = np.mean(U_final < 0)
    frac_neg_V = np.mean(V_final < 0)

    print(f"Final relative error ||UV^T - X*||_F / ||X*||_F = {final_rel:.6f}")
    print(f"Final f(x) = {nsfw.func_vals[-1]:.6f}")
    print(f"Final smoothed gap = {nsfw.gaps[-1]:.6f}")
    print(f"Final dist_D^2 = {2 * nsfw.ns_gaps[-1]:.6e}")
    print(f"Fraction negative entries: U {frac_neg_U:.4f}, V {frac_neg_V:.4f}")
    print()

    # ── Plots ──
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Relative reconstruction error
    axs[0, 0].semilogy(iters, rel_error)
    axs[0, 0].set_xlabel("Iteration $k$")
    axs[0, 0].set_ylabel(r"$\|U_k V_k^\top - X^*\|_F \;/\; \|X^*\|_F$")
    axs[0, 0].set_title("Relative reconstruction error")
    axs[0, 0].grid(True, alpha=0.3)

    # (0,1) Feasibility: dist_D^2(x_k) = 2 * ns_gaps
    axs[0, 1].semilogy(iters, 2.0 * nsfw.ns_gaps)
    # Reference rate O(beta_k) = O(k^{-1/4})
    C_feas = (2.0 * nsfw.ns_gaps[n_steps // 4]) * (n_steps // 4 + 1) ** 0.25
    axs[0, 1].semilogy(
        iters,
        C_feas / iters ** 0.25,
        "--",
        color="gray",
        label=r"$O(k^{-1/4})$",
    )
    axs[0, 1].set_xlabel("Iteration $k$")
    axs[0, 1].set_ylabel(r"$\mathrm{dist}_D^2(x_k)$")
    axs[0, 1].set_title(r"Squared distance to $D = \{(U,V) \geq 0\}$")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # (1,0) Smoothed gaps with min and avg
    axs[1, 0].semilogy(iters, nsfw.gaps, alpha=0.5, label="Smoothed gap")
    axs[1, 0].semilogy(iters, min_gaps, label="Min smoothed gap")
    axs[1, 0].semilogy(iters, avg_gaps, label="Avg smoothed gap")
    # Reference rate O(k^{-1/4})
    C_gap = avg_gaps[n_steps // 4] * (n_steps // 4 + 1) ** 0.25
    axs[1, 0].semilogy(
        iters,
        C_gap / iters ** 0.25,
        "--",
        color="gray",
        label=r"$O(k^{-1/4})$",
    )
    axs[1, 0].set_xlabel("Iteration $k$")
    axs[1, 0].set_ylabel("Smoothed gap")
    axs[1, 0].set_title(
        r"$\mathrm{gap}_{\beta_k}(x_k) "
        r"= \max_{s \in C}\;\langle \nabla\Phi_k(x_k),\, x_k - s\rangle$"
    )
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True, alpha=0.3)

    # (1,1) Functional values f(x_k)
    axs[1, 1].plot(iters, nsfw.func_vals)
    axs[1, 1].set_xlabel("Iteration $k$")
    axs[1, 1].set_ylabel(r"$f(x_k)$")
    axs[1, 1].set_title(
        r"Objective $f(U_k, V_k) = \frac{1}{2}\|U_k V_k^\top - X^*\|_F^2$"
    )
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"NSFW for Nonneg Matrix Factorization\n"
        f"$m={m},\\; n={n},\\; r={r},\\; "
        f"\\beta_0={beta0},\\; \\mathrm{{margin}}={margin}$",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig("/home/claude/experiments/nonneg_mf_results.png", dpi=150)
    plt.close(fig)
    print("Saved plot to experiments/nonneg_mf_results.png")

    return nsfw, obj


if __name__ == "__main__":
    run_experiment()
