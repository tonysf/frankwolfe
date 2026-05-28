"""
L1 splitting with nonconvex (indefinite) quadratic via FRAMES.

Problem (original):
    min_{x in C_1 ∩ C_2}  f(x) = 0.5 * x^T Q x - b^T x

where Q is symmetric indefinite (has positive and negative eigenvalues),
C_1 = {x : ||x - c_1||_1 <= 2},  C_2 = {x : ||x - c_2||_1 <= 2},
c_1 = e_1, c_2 = -e_1.  The intersection C_1 ∩ C_2 = B_1 = {x : ||x||_1 <= 1}.

Splitting formulation (instance of (P)):
    min_{(x_1, x_2) in C_1 x C_2}  f(w_1 x_1 + w_2 x_2)  +  iota_{0}(x_1 - x_2)

with T(x_1,x_2) = x_1 - x_2, g = iota_{0}, D = {0}.
This is Assumption 1.4(I) with f nonconvex smooth.

Nonsmooth gap (computable in closed form since C_1 ∩ C_2 = B_1):
    gap_tilde(x) = <Q x_bar - b, x_bar> + ||Q x_bar - b||_inf
where x_bar = w_1 x_1 + w_2 x_2 and ||·||_inf is the support function of B_1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frank_wolfe.core.objective import ObjectiveFunction
from tqdm import tqdm

# ─── Objective ────────────────────────────────────────────────────────────────


class IndefiniteQuadraticSplitting(ObjectiveFunction):
    """
    f(x_1, x_2) = 0.5 * x_bar^T Q x_bar - b^T x_bar

    where x_bar = w1*x1 + w2*x2.

    Decision variable: x = [x_1; x_2] in R^{2n}.
    T(x) = x_1 - x_2,  T^*(z) = (z, -z).
    """

    def __init__(self, Q, b, n, w1=0.5, w2=0.5):
        super().__init__()
        self.Q = Q
        self.b = b
        self.n = n
        self.w1 = w1
        self.w2 = w2
        self.lipschitz = np.linalg.norm(Q, ord=2)

    def _unpack(self, x):
        return x[: self.n], x[self.n :]

    def _pack(self, x1, x2):
        return np.concatenate([x1, x2])

    def _xbar(self, x):
        x1, x2 = self._unpack(x)
        return self.w1 * x1 + self.w2 * x2

    def evaluate(self, x):
        xb = self._xbar(x)
        return 0.5 * xb @ self.Q @ xb - self.b @ xb

    def gradient(self, x):
        xb = self._xbar(x)
        g = self.Q @ xb - self.b
        return self._pack(self.w1 * g, self.w2 * g)

    def linear_operator(self, x):
        x1, x2 = self._unpack(x)
        return x1 - x2

    def linear_operator_adjoint(self, z):
        return self._pack(z, -z)


# ─── LMO for C_1 x C_2 ───────────────────────────────────────────────────────


def create_product_l1_lmo(c1, c2, radius, n):
    """
    LMO over C_1 x C_2 where C_i = {x : ||x - c_i||_1 <= radius}.
    The LMO of {x : ||x - c||_1 <= r} at gradient g is:
        c - r * sign(g_{j*}) * e_{j*}   where j* = argmax |g_j|.
    """

    def single_lmo(grad, center, rad):
        j = np.argmax(np.abs(grad))
        s = center.copy()
        s[j] -= rad * np.sign(grad[j])
        return s

    def lmo(gradient):
        g1 = gradient[:n]
        g2 = gradient[n:]
        s1 = single_lmo(g1, c1, radius)
        s2 = single_lmo(g2, c2, radius)
        return np.concatenate([s1, s2])

    return lmo


# ─── Prox for g = iota_{0} ───────────────────────────────────────────────────


def zero_prox(z, beta):
    return np.zeros_like(z)


# ─── Nonsmooth gap ────────────────────────────────────────────────────────────


def nonsmooth_gap(x_bar, Q, b):
    """
    gap_{C_1 ∩ C_2}(x_bar) for f(x) = 0.5 x^T Q x - b^T x.

    = max_{s in B_1} <Qx_bar - b, x_bar - s>
    = <Qx_bar - b, x_bar> + ||Qx_bar - b||_inf

    where ||·||_inf is the support function of B_1.
    """
    g = Q @ x_bar - b
    return g @ x_bar + np.linalg.norm(g, ord=np.inf)


# ─── Generate indefinite Q ───────────────────────────────────────────────────


def generate_indefinite_quadratic(n, frac_neg=0.3, eigval_range=(0.5, 5.0), seed=None):
    """
    Generate a symmetric indefinite Q in R^{n x n} and a vector b.

    Parameters
    ----------
    n : int
    frac_neg : float
        Fraction of eigenvalues that are negative.
    eigval_range : tuple
        Magnitudes of eigenvalues are drawn uniformly from this range.
    seed : int or None

    Returns
    -------
    Q, b, eigenvalues
    """
    rng = np.random.default_rng(seed)

    # Random orthogonal basis
    M = rng.standard_normal((n, n))
    V, _ = np.linalg.qr(M)

    # Eigenvalues: some positive, some negative
    eigvals = rng.uniform(*eigval_range, size=n)
    n_neg = int(frac_neg * n)
    signs = np.ones(n)
    neg_idx = rng.choice(n, size=n_neg, replace=False)
    signs[neg_idx] = -1
    eigvals = signs * eigvals

    Q = V @ np.diag(eigvals) @ V.T
    Q = 0.5 * (Q + Q.T)  # ensure exact symmetry

    # b: random
    b = rng.standard_normal(n)

    return Q, b, eigvals


# ─── FRAMES loop ──────────────────────────────────────────────────────────────


def run_frames_splitting(obj, lmo, n_steps, beta0=1.0):
    n = obj.n
    x0 = np.zeros(2 * n)
    x = lmo(obj.gradient(x0))

    func_vals = np.zeros(n_steps)
    smoothed_gaps = np.zeros(n_steps)
    ns_gaps = np.zeros(n_steps)
    feasibility = np.zeros(n_steps)

    for i in tqdm(range(n_steps), desc="FRAMES Splitting"):
        beta = beta0 / (i + 1) ** 0.25
        step_size = 1.0 / (i + 1) ** 0.5

        grad = obj.gradient(x)
        Tx = obj.linear_operator(x)
        moreau_grad = obj.linear_operator_adjoint(Tx - zero_prox(Tx, beta)) / beta
        combined_grad = grad + moreau_grad

        direction = lmo(combined_grad)

        smoothed_gaps[i] = np.sum(combined_grad * (x - direction))
        func_vals[i] = obj.evaluate(x)
        feasibility[i] = np.linalg.norm(Tx) ** 2

        x_bar = obj._xbar(x)
        ns_gaps[i] = nonsmooth_gap(x_bar, obj.Q, obj.b)

        x = (1 - step_size) * x + step_size * direction

    return x, func_vals, smoothed_gaps, ns_gaps, feasibility


def run_frames_splitting_log(obj, lmo, n_steps, beta0=1.0):
    n = obj.n
    x0 = np.zeros(2 * n)
    x = lmo(obj.gradient(x0))

    func_vals = np.zeros(n_steps)
    smoothed_gaps = np.zeros(n_steps)
    ns_gaps = np.zeros(n_steps)
    feasibility = np.zeros(n_steps)

    for i in tqdm(range(n_steps), desc="FRAMES Splitting"):
        beta = beta0 / np.log(i + 2)
        step_size = 1.0 / (i + 1) ** 0.5

        grad = obj.gradient(x)
        Tx = obj.linear_operator(x)
        moreau_grad = obj.linear_operator_adjoint(Tx - zero_prox(Tx, beta)) / beta
        combined_grad = grad + moreau_grad

        direction = lmo(combined_grad)

        smoothed_gaps[i] = np.sum(combined_grad * (x - direction))
        func_vals[i] = obj.evaluate(x)
        feasibility[i] = np.linalg.norm(Tx) ** 2

        x_bar = obj._xbar(x)
        ns_gaps[i] = nonsmooth_gap(x_bar, obj.Q, obj.b)

        x = (1 - step_size) * x + step_size * direction

    return x, func_vals, smoothed_gaps, ns_gaps, feasibility


# ─── Experiment ───────────────────────────────────────────────────────────────


def run_experiment(n=50, n_steps=5000, frac_neg=0.3, seed=42):
    rng = np.random.default_rng(seed)

    # Problem data
    c1 = np.zeros(n)
    c1[0] = 1.0
    c2 = np.zeros(n)
    c2[0] = -1.0
    radius = 2.0

    Q, b, eigvals = generate_indefinite_quadratic(n, frac_neg=frac_neg, seed=seed)

    n_pos = np.sum(eigvals > 0)
    n_neg = np.sum(eigvals < 0)
    print(f"Problem: n={n}, n_steps={n_steps}")
    print(f"Q: {n_pos} positive eigenvalues, {n_neg} negative eigenvalues")
    print(f"Eigenvalue range: [{eigvals.min():.3f}, {eigvals.max():.3f}]")
    print(f"||Q||_op = {np.linalg.norm(Q, ord=2):.3f}")
    print()

    obj = IndefiniteQuadraticSplitting(Q, b, n)
    lmo = create_product_l1_lmo(c1, c2, radius, n)

    beta0_values = [0.01, 0.1, 1.0, 10.0]
    results = {}

    for beta0 in beta0_values:
        x_final, fvals, sgaps, nsgaps, feas = run_frames_splitting(
            obj, lmo, n_steps, beta0=beta0
        )
        x_bar_final = obj._xbar(x_final)
        results[beta0] = {
            "func_vals": fvals,
            "smoothed_gaps": sgaps,
            "ns_gaps": nsgaps,
            "feasibility": feas,
            "x_bar_final": x_bar_final,
        }
        print(
            f"beta0={beta0:6.2f}  |  f={fvals[-1]:.4f}  "
            f"||x1-x2||={np.sqrt(feas[-1]):.4e}  "
            f"ns_gap={nsgaps[-1]:.4e}  "
            f"||x_bar||_1={np.linalg.norm(x_bar_final, ord=1):.4f}"
        )

    print()

    # ── Plots ──
    iters = np.arange(1, n_steps + 1)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["C0", "C1", "C2", "C3"]

    for idx, beta0 in enumerate(beta0_values):
        c = colors[idx]
        R = results[beta0]
        lab = rf"$\beta_0 = {beta0}$"

        # [0,0] Min gaps: smoothed vs nonsmooth
        min_sgaps = np.minimum.accumulate(R["smoothed_gaps"])
        # Nonsmooth gap can be negative for nonconvex f when x_bar not in C_tilde;
        # plot absolute value for log scale
        abs_nsgaps = np.abs(R["ns_gaps"])
        # Running min of |ns_gap| is not quite right; track the gap itself
        # and plot it with sign awareness
        axs[0, 0].semilogy(iters, min_sgaps, color=c, label=lab + " (smoothed)")
        axs[0, 0].semilogy(
            iters,
            np.minimum.accumulate(abs_nsgaps),
            color=c,
            linestyle="--",
            label=lab + " (|nonsmooth|)",
        )

        # [0,1] Feasibility
        axs[0, 1].semilogy(
            iters,
            np.sqrt(np.maximum(R["feasibility"], 0.0)),
            color=c,
            label=lab,
            alpha=0.7,
        )

        # [1,0] Functional values (not gap, since we don't know f* for nonconvex)
        axs[1, 0].plot(iters, R["func_vals"], color=c, label=lab, alpha=0.7)

        # [1,1] Avg smoothed gap
        avg_sgaps = np.cumsum(R["smoothed_gaps"]) / iters
        axs[1, 1].semilogy(iters, avg_sgaps, color=c, label=lab + " (smoothed)")
        avg_abs_nsgaps = np.cumsum(abs_nsgaps) / iters
        axs[1, 1].semilogy(
            iters, avg_abs_nsgaps, color=c, linestyle="--", label=lab + " (|nonsmooth|)"
        )

    # Reference rates
    C_ref = (
        np.minimum.accumulate(results[1.0]["smoothed_gaps"])[n_steps // 4]
        * (n_steps // 4 + 1) ** 0.25
    )
    axs[0, 0].semilogy(
        iters,
        C_ref / iters**0.25,
        "--",
        color="gray",
        linewidth=2,
        label=r"$O(k^{-1/4})$",
    )

    feas_dist = np.sqrt(np.maximum(results[1.0]["feasibility"], 0.0))
    C_feas = feas_dist[n_steps // 4] * (n_steps // 4 + 1) ** 0.125
    axs[0, 1].semilogy(
        iters,
        C_feas / iters**0.125,
        "--",
        color="gray",
        linewidth=2,
        label=r"$O(k^{-1/8})$",
    )

    C_avg = (np.cumsum(results[1.0]["smoothed_gaps"]) / iters)[n_steps // 4] * (
        n_steps // 4 + 1
    ) ** 0.25
    axs[1, 1].semilogy(
        iters,
        C_avg / iters**0.25,
        "--",
        color="gray",
        linewidth=2,
        label=r"$O(k^{-1/4})$",
    )

    # Labels
    axs[0, 0].set_title("Min gaps: smoothed vs $|$nonsmooth$|$")
    axs[0, 0].set_xlabel("Iteration $k$")
    axs[0, 0].set_ylabel("Gap")
    axs[0, 0].legend(fontsize=6, ncol=2)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].set_title(r"Feasibility: $\|x_{1,k} - x_{2,k}\|$")
    axs[0, 1].set_xlabel("Iteration $k$")
    axs[0, 1].set_ylabel(r"$\|x_1 - x_2\|$")
    axs[0, 1].legend(fontsize=7)
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].set_title(
        r"Objective $f(\bar{x}_k) = \frac{1}{2}\bar{x}_k^T Q \bar{x}_k - b^T \bar{x}_k$"
    )
    axs[1, 0].set_xlabel("Iteration $k$")
    axs[1, 0].set_ylabel(r"$f(\bar{x}_k)$")
    axs[1, 0].legend(fontsize=7)
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].set_title("Avg gaps: smoothed vs $|$nonsmooth$|$")
    axs[1, 1].set_xlabel("Iteration $k$")
    axs[1, 1].set_ylabel("Gap")
    axs[1, 1].legend(fontsize=6, ncol=2)
    axs[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"FRAMES for $\\ell_1$ Splitting (nonconvex $f$):  "
        f"$n={n},\\; N={n_steps},\\; "
        f"{n_neg}$ negative eigenvalues",
        fontsize=13,
    )
    plt.tight_layout()
    outpath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "l1_splitting_nonconvex_results.png",
    )
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {outpath}")


if __name__ == "__main__":
    run_experiment()
