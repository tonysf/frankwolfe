"""
Low-rank matrix factorization with trend filtering via FRAMES.

Problem:
    min_{U, V}  f(U, V) + g(T U)
    s.t.        ||U||_op <= tau_U,  ||V||_op <= tau_V

where:
    f(U, V) = 0.5 * ||U V^T - X*||_F^2
    T = D_row  (row-wise first-order finite difference on U)
    g = ||·||_1           (convex, Lipschitz)          — Assumption 1.4(II) with rho=0
    g = sum SCAD(·)       (weakly convex, Lipschitz)   — Assumption 1.4(II) with rho>0
    g = sum MCP(·)        (weakly convex, Lipschitz)   — Assumption 1.4(II) with rho>0

The linear operator T acts only on U:  T(U,V) = D_row @ U  where
D_row in R^{(m-1) x m} is the first-order difference matrix,
so (TU)_{i,j} = U_{i+1,j} - U_{i,j}.  This promotes piecewise-constant
structure in the rows of U.

Ground truth: U* is piecewise constant (blockwise), V* is random,
X* = U* V*^T.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frank_wolfe import Frames
from frank_wolfe.core.objective import ObjectiveFunction


# ─── Finite difference operator ───────────────────────────────────────────────

def finite_diff_matrix(m):
    """
    First-order finite difference matrix D in R^{(m-1) x m}.
    D[i, i] = -1,  D[i, i+1] = 1.
    """
    D = np.zeros((m - 1, m))
    for i in range(m - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    return D


# ─── Objective with trend filtering on U ──────────────────────────────────────

class TrendFilteredMFObjective(ObjectiveFunction):
    """
    f(U, V) = 0.5 * ||U V^T - X*||_F^2

    Linear operator T(U, V) = D @ U  where D is the row-wise finite
    difference matrix.  The adjoint is T^*(Z) = (D^T Z, 0_V).

    Decision variable: x = [vec(U); vec(V)].
    """

    def __init__(self, X_star, m, n, r):
        super().__init__()
        self.X_star = X_star
        self.m = m
        self.n = n
        self.r = r
        self.D = finite_diff_matrix(m)       # (m-1) x m
        self.DT = self.D.T                    # m x (m-1)

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
        R = U @ V.T - self.X_star          # m x n
        return self._pack(R @ V, R.T @ U)  # (grad_U, grad_V)

    def linear_operator(self, x):
        """T(U, V) = D @ U,  returned as a flat vector."""
        U, V = self._unpack(x)
        return (self.D @ U).ravel()         # (m-1)*r

    def linear_operator_adjoint(self, z):
        """T^*(z) = (D^T @ Z, 0_V),  where Z = reshape(z, (m-1, r))."""
        Z = z.reshape(self.m - 1, self.r)
        adj_U = self.DT @ Z                # m x r
        adj_V = np.zeros((self.n, self.r))
        return self._pack(adj_U, adj_V)

    def minimal_norm_selection(self, z):
        """
        Minimal norm element of the subdifferential of g at z.
        For L1: sign(z) with sign(0) = 0  (this is np.sign).
        For SCAD: the (unique a.e.) derivative, see scad_minimal_norm_selection.
        
        This method is set externally depending on the choice of g.
        """
        raise NotImplementedError("Set via objective.minimal_norm_selection = ...")


# ─── LMO for product of spectral-norm balls ──────────────────────────────────

def create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r):
    """
    LMO over C = {(U,V) : ||U||_op <= tau_U, ||V||_op <= tau_V}.
    Returns -tau * sign(G) for each block independently.
    """

    def lmo(gradient):
        grad_U = gradient[: m * r].reshape(m, r)
        grad_V = gradient[m * r :].reshape(n, r)

        Uu, _, Vtu = np.linalg.svd(grad_U, full_matrices=False)
        s_U = -tau_U * (Uu @ Vtu)

        Uv, _, Vtv = np.linalg.svd(grad_V, full_matrices=False)
        s_V = -tau_V * (Uv @ Vtv)

        return np.concatenate([s_U.ravel(), s_V.ravel()])

    return lmo


# ─── Prox and subdifferential for L1 ─────────────────────────────────────────

def l1_prox(z, beta):
    """prox_{beta * ||·||_1}(z) = soft_thresh(z, beta)."""
    return np.sign(z) * np.maximum(np.abs(z) - beta, 0)


def l1_minimal_norm(z):
    """Minimal norm selection of subdifferential of ||·||_1 at z."""
    return np.sign(z)


# ─── SCAD penalty, prox, and subdifferential ─────────────────────────────────

def make_scad_functions(lam, a):
    """
    Returns (scad_prox, scad_minimal_norm, scad_eval, rho) for the
    elementwise SCAD penalty with parameters (lam, a).

    SCAD(t) for t >= 0:
        lam * t                                     if t <= lam
        -(t^2 - 2*a*lam*t + lam^2) / (2*(a-1))     if lam < t <= a*lam
        (a+1)*lam^2 / 2                             if t > a*lam

    Weak convexity modulus: rho = 1/(a-1).
    Lipschitz constant (elementwise): lam.
    Requires a > 2.
    """
    assert a > 2, "SCAD requires a > 2"
    rho = 1.0 / (a - 1)

    def scad_eval_scalar(t):
        t = abs(t)
        if t <= lam:
            return lam * t
        elif t <= a * lam:
            return -(t**2 - 2 * a * lam * t + lam**2) / (2 * (a - 1))
        else:
            return (a + 1) * lam**2 / 2

    def scad_eval(z):
        return np.sum(np.vectorize(scad_eval_scalar)(z))

    def scad_prox(z, beta):
        """
        prox_{beta * sum SCAD(·)}(z), applied elementwise.
        Requires beta < a - 1 = 1/rho.

        For y = |z_ij| >= 0:
            y <= beta*lam            :  0
            beta*lam < y <= lam*(1+beta)  :  y - beta*lam
            lam*(1+beta) < y <= a*lam     :  ((a-1)*y - beta*a*lam) / (a-1-beta)
            y > a*lam                     :  y
        """
        assert beta < a - 1, f"Need beta < a-1 = {a-1}, got beta = {beta}"
        y = np.abs(z)
        result = np.zeros_like(z)

        # Region 1: soft threshold (L1-like region)
        mask1 = y <= lam * (1 + beta)
        result[mask1] = np.maximum(y[mask1] - beta * lam, 0)

        # Region 2: quadratic region
        mask2 = (y > lam * (1 + beta)) & (y <= a * lam)
        result[mask2] = ((a - 1) * y[mask2] - beta * a * lam) / (a - 1 - beta)

        # Region 3: identity (no penalty gradient)
        mask3 = y > a * lam
        result[mask3] = y[mask3]

        return np.sign(z) * result

    def scad_deriv(z):
        """
        Derivative of SCAD applied elementwise (= minimal norm subgradient).
        For t = |z| > 0:
            t <= lam          :  lam * sign(z)
            lam < t <= a*lam  :  sign(z) * (a*lam - t) / (a - 1)
            t > a*lam         :  0
        At z = 0: minimal norm selection = 0.
        """
        t = np.abs(z)
        s = np.sign(z)
        result = np.zeros_like(z)

        mask1 = (t > 0) & (t <= lam)
        result[mask1] = lam * s[mask1]

        mask2 = (t > lam) & (t <= a * lam)
        result[mask2] = s[mask2] * (a * lam - t[mask2]) / (a - 1)

        # mask3 (t > a*lam) and t == 0: result stays 0
        return result

    return scad_prox, scad_deriv, scad_eval, rho


def make_mcp_functions(lam, gamma):
    """
    Returns (mcp_prox, mcp_minimal_norm, mcp_eval, rho) for the elementwise MCP
    penalty with parameters (lam, gamma).
    """
    assert gamma > 1, "MCP requires gamma > 1"
    rho = 1.0 / gamma

    def mcp_eval_scalar(t):
        t = abs(t)
        if t <= gamma * lam:
            return lam * t - t**2 / (2 * gamma)
        return gamma * lam**2 / 2

    def mcp_eval(z):
        return np.sum(np.vectorize(mcp_eval_scalar)(z))

    def mcp_prox(z, beta):
        assert beta < gamma, f"Need beta < gamma = {gamma}, got beta = {beta}"
        y = np.abs(z)
        result = np.zeros_like(z)

        mask1 = (y > beta * lam) & (y <= gamma * lam)
        result[mask1] = (y[mask1] - beta * lam) / (1 - beta / gamma)

        mask2 = y > gamma * lam
        result[mask2] = y[mask2]

        return np.sign(z) * result

    def mcp_deriv(z):
        t = np.abs(z)
        s = np.sign(z)
        result = np.zeros_like(z)

        mask = (t > 0) & (t <= gamma * lam)
        result[mask] = s[mask] * (lam - t[mask] / gamma)
        return result

    return mcp_prox, mcp_deriv, mcp_eval, rho


# ─── Data generation ──────────────────────────────────────────────────────────

def generate_piecewise_constant_mf(m, n, r, n_blocks=5, margin=1.05, seed=None):
    """
    Generate a matrix factorization problem where U* is piecewise constant
    along its rows (each column of U* is a step function with n_blocks levels).

    Parameters
    ----------
    m, n, r : int
        Dimensions.
    n_blocks : int
        Number of constant blocks per column of U*.
    margin : float
        Multiplicative slack on spectral-norm constraints.
    seed : int or None

    Returns
    -------
    X_star, U_star, V_star, tau_U, tau_V
    """
    rng = np.random.default_rng(seed)

    # U*: piecewise constant, nonneg, with n_blocks levels per column
    U_star = np.zeros((m, r))
    block_boundaries = np.sort(rng.choice(range(1, m), size=n_blocks - 1, replace=False))
    block_boundaries = np.concatenate([[0], block_boundaries, [m]])
    for j in range(r):
        levels = rng.uniform(0.5, 3.0, size=n_blocks)
        for b in range(n_blocks):
            U_star[block_boundaries[b]:block_boundaries[b + 1], j] = levels[b]

    # V*: random positive
    V_star = np.abs(rng.standard_normal((n, r)))

    X_star = U_star @ V_star.T
    tau_U = margin * np.linalg.norm(U_star, ord=2)
    tau_V = margin * np.linalg.norm(V_star, ord=2)

    return X_star, U_star, V_star, tau_U, tau_V


# ─── Run one FRAMES instance ─────────────────────────────────────────────────

def run_frames(obj, lmo, prox_fn, objective_type, x0, beta0, n_steps):
    frames = Frames(obj, lmo, prox_fn, objective_type)
    frames.run(x0, beta0=beta0, n_steps=n_steps)
    return frames


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(
    m=100, n=100, r=5, n_blocks=5,
    scad_a=3.7, scad_lam=0.5,
    beta0=1.0, n_steps=2000, margin=1.05, seed=42,
):
    # ── Problem setup ──
    X_star, U_star, V_star, tau_U, tau_V = generate_piecewise_constant_mf(
        m, n, r, n_blocks=n_blocks, margin=margin, seed=seed
    )
    X_star_fro = np.linalg.norm(X_star, "fro")
    D = finite_diff_matrix(m)
    DU_star = D @ U_star

    print(f"Problem: m={m}, n={n}, r={r}, n_blocks={n_blocks}")
    print(f"||X*||_F = {X_star_fro:.2f}")
    print(f"tau_U = {tau_U:.4f},  tau_V = {tau_V:.4f}")
    print(f"||D @ U*||_1 = {np.sum(np.abs(DU_star)):.4f}  "
          f"(nnz = {np.count_nonzero(DU_star)} / {DU_star.size})")
    print(f"SCAD params: a={scad_a}, lam={scad_lam}, rho={1/(scad_a-1):.4f}")
    print(f"beta_0 = {beta0}, n_steps = {n_steps}")
    print()

    lmo = create_spectral_ball_product_lmo(tau_U, tau_V, m, n, r)
    x0 = np.zeros(m * r + n * r)

    # ── L1 experiment ──
    obj_l1 = TrendFilteredMFObjective(X_star, m, n, r)
    obj_l1.minimal_norm_selection = l1_minimal_norm

    print("Running FRAMES with g = ||·||_1 ...")
    frames_l1 = run_frames(obj_l1, lmo, l1_prox, "lipschitz", x0, beta0, n_steps)

    # ── SCAD experiment ──
    scad_prox, scad_deriv, scad_eval, rho = make_scad_functions(scad_lam, scad_a)
    obj_scad = TrendFilteredMFObjective(X_star, m, n, r)
    obj_scad.minimal_norm_selection = scad_deriv

    print("Running FRAMES with g = SCAD ...")
    frames_scad = run_frames(obj_scad, lmo, scad_prox, "lipschitz", x0, beta0, n_steps)

    # ── Extract results ──
    iters = np.arange(1, n_steps + 1)

    def get_metrics(frames, obj):
        U_final, V_final = obj._unpack(frames.x)
        rel_err = np.sqrt(2.0 * np.maximum(frames.func_vals, 0)) / X_star_fro
        DU_final = D @ U_final
        return {
            "rel_error": rel_err,
            "func_vals": frames.func_vals,
            "gaps": frames.gaps,
            "ns_gaps": frames.ns_gaps,
            "min_gaps": np.minimum.accumulate(frames.gaps),
            "avg_gaps": np.cumsum(frames.gaps) / iters,
            "min_ns_gaps": np.minimum.accumulate(frames.ns_gaps),
            "avg_ns_gaps": np.cumsum(frames.ns_gaps) / iters,
            "U_final": U_final,
            "DU_nnz_final": np.sum(np.abs(DU_final) > 1e-6),
            "DU_l1_final": np.sum(np.abs(DU_final)),
        }

    m_l1 = get_metrics(frames_l1, obj_l1)
    m_scad = get_metrics(frames_scad, obj_scad)

    print(f"\n{'':30s} {'L1':>12s} {'SCAD':>12s}")
    print(f"{'─'*56}")
    print(f"{'Final rel error':30s} {m_l1['rel_error'][-1]:12.4f} {m_scad['rel_error'][-1]:12.4f}")
    print(f"{'Final f(x)':30s} {m_l1['func_vals'][-1]:12.2f} {m_scad['func_vals'][-1]:12.2f}")
    print(f"{'||D U_final||_1':30s} {m_l1['DU_l1_final']:12.2f} {m_scad['DU_l1_final']:12.2f}")
    print(f"{'nnz(D U_final)':30s} {m_l1['DU_nnz_final']:12d} {m_scad['DU_nnz_final']:12d}")
    print(f"{'nnz(D U*) (ground truth)':30s} {np.count_nonzero(DU_star):12d}")
    print()

    # ── Plots ──
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # [0,0] Relative reconstruction error
    axs[0, 0].semilogy(iters, m_l1["rel_error"], label=r"$g = \|\cdot\|_1$", alpha=0.8)
    axs[0, 0].semilogy(iters, m_scad["rel_error"], label=r"$g = \mathrm{SCAD}$", alpha=0.8)
    axs[0, 0].set_title("Relative reconstruction error")
    axs[0, 0].set_xlabel("Iteration $k$")
    axs[0, 0].set_ylabel(r"$\|U_k V_k^\top - X^*\|_F / \|X^*\|_F$")
    axs[0, 0].legend(fontsize=8)
    axs[0, 0].grid(True, alpha=0.3)

    # [0,1] Smoothed gaps: min and avg
    axs[0, 1].semilogy(iters, m_l1["min_gaps"], color="C0", label=r"$\|\cdot\|_1$ (min)")
    axs[0, 1].semilogy(iters, m_l1["avg_gaps"], color="C0", linestyle="--", alpha=0.6, label=r"$\|\cdot\|_1$ (avg)")
    axs[0, 1].semilogy(iters, m_scad["min_gaps"], color="C1", label=r"SCAD (min)")
    axs[0, 1].semilogy(iters, m_scad["avg_gaps"], color="C1", linestyle="--", alpha=0.6, label=r"SCAD (avg)")
    C_ref = m_l1["avg_gaps"][n_steps // 4] * (n_steps // 4 + 1) ** 0.25
    axs[0, 1].semilogy(iters, C_ref / iters**0.25, "--", color="gray", linewidth=2, label=r"$O(k^{-1/4})$")
    axs[0, 1].set_title("Smoothed gaps")
    axs[0, 1].set_xlabel("Iteration $k$")
    axs[0, 1].set_ylabel(r"$\mathrm{gap}_{\beta_k}(x_k)$")
    axs[0, 1].legend(fontsize=7)
    axs[0, 1].grid(True, alpha=0.3)

    # [0,2] Nonsmooth gaps: min and avg
    axs[0, 2].semilogy(iters, m_l1["min_ns_gaps"], color="C0", label=r"$\|\cdot\|_1$ (min)")
    axs[0, 2].semilogy(iters, m_l1["avg_ns_gaps"], color="C0", linestyle="--", alpha=0.6, label=r"$\|\cdot\|_1$ (avg)")
    axs[0, 2].semilogy(iters, m_scad["min_ns_gaps"], color="C1", label=r"SCAD (min)")
    axs[0, 2].semilogy(iters, m_scad["avg_ns_gaps"], color="C1", linestyle="--", alpha=0.6, label=r"SCAD (avg)")
    axs[0, 2].set_title("Nonsmooth gaps")
    axs[0, 2].set_xlabel("Iteration $k$")
    axs[0, 2].set_ylabel(r"$\mathrm{gap}(x_k; \xi_k)$")
    axs[0, 2].legend(fontsize=7)
    axs[0, 2].grid(True, alpha=0.3)

    # [1,0] and [1,1]: Recovered U vs U* (one column each, for L1 and SCAD)
    col_to_show = 0
    rows = np.arange(m)

    axs[1, 0].step(rows, U_star[:, col_to_show], where="mid", color="black",
                    linewidth=2, label=r"$U^\star$", zorder=3)
    axs[1, 0].plot(rows, m_l1["U_final"][:, col_to_show], alpha=0.8,
                   label=r"$U_k$ ($\ell_1$)")
    axs[1, 0].set_title(rf"Recovered $U$ (column {col_to_show}): $\ell_1$")
    axs[1, 0].set_xlabel("Row index $i$")
    axs[1, 0].set_ylabel(rf"$U_{{i,{col_to_show}}}$")
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].step(rows, U_star[:, col_to_show], where="mid", color="black",
                    linewidth=2, label=r"$U^\star$", zorder=3)
    axs[1, 1].plot(rows, m_scad["U_final"][:, col_to_show], alpha=0.8,
                   color="C1", label=r"$U_k$ (SCAD)")
    axs[1, 1].set_title(rf"Recovered $U$ (column {col_to_show}): SCAD")
    axs[1, 1].set_xlabel("Row index $i$")
    axs[1, 1].set_ylabel(rf"$U_{{i,{col_to_show}}}$")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].grid(True, alpha=0.3)

    # [1,2] Objective values
    axs[1, 2].plot(iters, m_l1["func_vals"], label=r"$g = \|\cdot\|_1$")
    axs[1, 2].plot(iters, m_scad["func_vals"], label=r"$g = \mathrm{SCAD}$")
    axs[1, 2].set_title(r"Objective $f(U_k, V_k)$")
    axs[1, 2].set_xlabel("Iteration $k$")
    axs[1, 2].set_ylabel(r"$f(x_k)$")
    axs[1, 2].legend(fontsize=8)
    axs[1, 2].grid(True, alpha=0.3)

    fig.suptitle(
        f"FRAMES for Trend-Filtered Matrix Factorization\n"
        f"$m={m},\\; n={n},\\; r={r},\\; "
        f"\\beta_0={beta0},\\; N={n_steps},\\; "
        f"\\lambda_{{\\mathrm{{SCAD}}}}={scad_lam},\\; "
        f"a_{{\\mathrm{{SCAD}}}}={scad_a}$",
        fontsize=13,
    )
    plt.tight_layout()
    outpath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "trend_filtering_results.png",
    )
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {outpath}")

    return frames_l1, frames_scad


if __name__ == "__main__":
    run_experiment()
