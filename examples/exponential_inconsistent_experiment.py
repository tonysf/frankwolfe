"""
Exponentially flat inconsistent indicator experiment.

This example shows that definability in an o-minimal structure does not, by
itself, give a polynomial distance modulus for the best-approximation problem.

    C = {(x1, x2): -a <= x1 <= a, exp(-1/x1^2) <= x2 <= 1},
    T(x1, x2) = x2,
    D = {-1}.

The boundary is completed continuously at x1 = 0 with exp(-1/x1^2) = 0.  For
a < sqrt(2/3), this lower boundary is convex on [-a, a].  Hence C is compact
and convex.  Moreover C^\dagger = {(0, 0)}, but along the lower boundary
h(x)-h^* is proportional to exp(-1/dist_{C^\dagger}(x)^2), which is flatter
than every polynomial in dist_{C^\dagger}(x).
"""

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "exponential_inconsistent_matplotlib"
_font_cache = _cache_root / "fontconfig"
_font_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg-cache"))
os.environ.setdefault("FC_CACHEDIR", str(_font_cache))

import matplotlib
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

matplotlib.use("Agg")
import matplotlib.pyplot as plt

legendfontsize = 12
titlefontsize = 20
labelfontsize = 18
suptitlefontsize = 24

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTFILE = THIS_DIR / "paper_fig_exponential_inconsistent.pdf"
DOMAIN_RADIUS = 0.7
DEFAULT_CENTER = np.array([DOMAIN_RADIUS, 0.0], dtype=float)
DEFAULT_INITIAL_POINT = np.array([DOMAIN_RADIUS, 1.0], dtype=float)
DELTA = 1.0


@dataclass
class Trace:
    center: np.ndarray
    iterations: np.ndarray
    points: np.ndarray
    h_gap: np.ndarray
    distance_residual: np.ndarray
    cdagger_distance: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the definable exponentially flat inconsistent experiment."
    )
    parser.add_argument("--n-steps", type=int, default=500000000)
    parser.add_argument("--beta0", type=float, default=3.0)
    parser.add_argument("--step-shift", type=float, default=100.0)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--outfile", type=Path, default=DEFAULT_OUTFILE)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable text.usetex for quick local previews. The paper default uses TeX.",
    )
    return parser.parse_args()


def configure_matplotlib(use_tex=True):
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
        }
    )


def lower_boundary(x1):
    x1 = np.asarray(x1, dtype=float)
    values = np.zeros_like(x1)
    mask = x1 != 0.0
    values[mask] = np.exp(-1.0 / x1[mask] ** 2)
    return values


def lower_boundary_derivative_scalar(x1):
    if x1 == 0.0:
        return 0.0
    return 2.0 * np.exp(-1.0 / x1**2) / x1**3


@njit
def lower_boundary_scalar_numba(x1):
    if x1 == 0.0:
        return 0.0
    return np.exp(-1.0 / (x1 * x1))


@njit
def lower_boundary_derivative_scalar_numba(x1):
    if x1 == 0.0:
        return 0.0
    return 2.0 * np.exp(-1.0 / (x1 * x1)) / (x1 * x1 * x1)


@njit
def exponential_cusp_lmo_numba(grad_x, grad_y):
    radius = DOMAIN_RADIUS

    if grad_y < 0.0:
        if grad_x >= 0.0:
            return -radius, 1.0
        return radius, 1.0

    if grad_y == 0.0:
        if grad_x >= 0.0:
            x1 = -radius
        else:
            x1 = radius
        return x1, lower_boundary_scalar_numba(x1)

    left_deriv = grad_x + grad_y * lower_boundary_derivative_scalar_numba(-radius)
    right_deriv = grad_x + grad_y * lower_boundary_derivative_scalar_numba(radius)

    if left_deriv >= 0.0:
        x1 = -radius
    elif right_deriv <= 0.0:
        x1 = radius
    else:
        lo = -radius
        hi = radius
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            deriv = grad_x + grad_y * lower_boundary_derivative_scalar_numba(mid)
            if deriv < 0.0:
                lo = mid
            else:
                hi = mid
        x1 = 0.5 * (lo + hi)

    return x1, lower_boundary_scalar_numba(x1)


@njit
def run_trace_samples_numba(
    n_steps,
    beta0,
    step_shift,
    center_x,
    center_y,
    initial_x,
    initial_y,
    sample_iterations,
):
    points = np.empty((sample_iterations.shape[0], 2), dtype=np.float64)
    x1 = initial_x
    x2 = initial_y
    sample_pos = 0

    if sample_iterations[0] == 0:
        points[0, 0] = x1
        points[0, 1] = x2
        sample_pos = 1

    for k in range(n_steps):
        beta = beta0 / (k + 1.0) ** 0.25
        step_size = 1.0 / (k + step_shift) ** 0.5

        grad_x = 2.0 * (x1 - center_x)
        grad_y = 2.0 * (x2 - center_y) + (x2 + 1.0) / beta
        direction_x, direction_y = exponential_cusp_lmo_numba(grad_x, grad_y)
        x1 = (1.0 - step_size) * x1 + step_size * direction_x
        x2 = (1.0 - step_size) * x2 + step_size * direction_y

        iteration = k + 1
        while (
            sample_pos < sample_iterations.shape[0]
            and sample_iterations[sample_pos] == iteration
        ):
            points[sample_pos, 0] = x1
            points[sample_pos, 1] = x2
            sample_pos += 1

    return points


def exponential_cusp_lmo(gradient):
    """
    Linear minimization oracle over C.

    On the lower boundary, the objective is one-dimensional and convex because
    exp(-1/x^2) is convex on [-a, a] for a < sqrt(2/3).
    """
    grad_x, grad_y = gradient
    radius = DOMAIN_RADIUS

    if grad_y < 0.0:
        x1 = -radius if grad_x >= 0.0 else radius
        return np.array([x1, 1.0], dtype=float)

    if grad_y == 0.0:
        x1 = -radius if grad_x >= 0.0 else radius
        return np.array([x1, lower_boundary([x1])[0]], dtype=float)

    left_deriv = grad_x + grad_y * lower_boundary_derivative_scalar(-radius)
    right_deriv = grad_x + grad_y * lower_boundary_derivative_scalar(radius)

    if left_deriv >= 0.0:
        x1 = -radius
    elif right_deriv <= 0.0:
        x1 = radius
    else:
        lo, hi = -radius, radius
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            deriv = grad_x + grad_y * lower_boundary_derivative_scalar(mid)
            if deriv < 0.0:
                lo = mid
            else:
                hi = mid
        x1 = 0.5 * (lo + hi)

    return np.array([x1, lower_boundary([x1])[0]], dtype=float)


def compute_residuals(points):
    x2 = points[:, 1]
    dist_d = x2 + 1.0
    h_gap = x2 + 0.5 * x2**2
    distance_residual = dist_d - DELTA
    cdagger_distance = np.linalg.norm(points, axis=1)
    return h_gap, distance_residual, cdagger_distance


def build_sample_iterations(n_steps, max_samples):
    if n_steps <= max_samples:
        return np.arange(n_steps + 1, dtype=np.int64)

    linear = np.linspace(0, min(n_steps, 2000), min(n_steps, 2000) + 1)
    geometric = np.geomspace(1, n_steps, max_samples)
    return np.unique(np.concatenate([linear, geometric]).round().astype(np.int64))


def run_trace(n_steps, beta0, step_shift, max_samples, center=None, x_initial=None):
    if center is None:
        center = DEFAULT_CENTER
    if x_initial is None:
        x_initial = DEFAULT_INITIAL_POINT

    sample_iterations = build_sample_iterations(n_steps, max_samples)
    points = run_trace_samples_numba(
        n_steps,
        beta0,
        step_shift,
        float(center[0]),
        float(center[1]),
        float(x_initial[0]),
        float(x_initial[1]),
        sample_iterations,
    )

    h_gap, distance_residual, cdagger_distance = compute_residuals(points)
    return Trace(
        center=np.asarray(center, dtype=float),
        iterations=sample_iterations,
        points=points,
        h_gap=h_gap,
        distance_residual=distance_residual,
        cdagger_distance=cdagger_distance,
    )


def semilogy_with_floor(ax, x_values, y_values, **kwargs):
    ax.semilogy(x_values, np.maximum(y_values, 1e-300), **kwargs)


def draw_geometry(ax, trace):
    radius = DOMAIN_RADIUS
    x_grid = np.linspace(-radius, radius, 600)
    y_lower = lower_boundary(x_grid)

    ax.fill_between(
        x_grid,
        y_lower,
        1.0,
        color="0.92",
        edgecolor="none",
        label=r"$\mathcal{C}$",
        zorder=0,
    )
    ax.plot(
        x_grid,
        y_lower,
        color="black",
        linewidth=2.0,
        label=r"$x_2=e^{-1/x_1^2}$",
        zorder=2,
    )
    ax.axhline(
        -1.0,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label=r"$T^{-1}(\mathcal{D})$",
        zorder=1,
    )
    ax.plot(
        trace.points[:, 0],
        trace.points[:, 1],
        color="tab:blue",
        linewidth=1.7,
        alpha=0.9,
        label="trajectory",
        zorder=3,
    )

    marker_idx = np.unique(
        np.round(np.geomspace(1, trace.points.shape[0] - 1, 24)).astype(int)
    )
    ax.scatter(
        trace.points[marker_idx, 0],
        trace.points[marker_idx, 1],
        color="tab:blue",
        edgecolor="white",
        linewidth=0.25,
        s=16,
        zorder=4,
    )
    ax.scatter(
        [trace.points[0, 0]],
        [trace.points[0, 1]],
        color="black",
        marker="o",
        s=45,
        label=r"$x_0$",
        zorder=5,
    )
    ax.scatter(
        [trace.center[0]],
        [trace.center[1]],
        marker="x",
        s=80,
        linewidths=2.2,
        color="tab:orange",
        label=r"$x_f$",
        zorder=5,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=150,
        color="tab:green",
        edgecolor="black",
        linewidth=0.5,
        label=r"$\mathcal{C}^\dagger$",
        zorder=6,
    )

    ax.set_title("Definable", fontsize=titlefontsize)
    ax.set_xlabel(r"$x_1$", fontsize=labelfontsize)
    ax.set_ylabel(r"$x_2$", fontsize=labelfontsize)
    ax.set_xlim(-0.78, 0.78)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legendfontsize, loc="lower center", ncol=2)


def draw_residuals(ax, trace):
    semilogy_with_floor(
        ax,
        trace.iterations,
        trace.h_gap,
        color="tab:blue",
        linewidth=2.0,
        label=r"$h(x_k)-h^\star$",
    )
    semilogy_with_floor(
        ax,
        trace.iterations,
        trace.distance_residual,
        color="tab:orange",
        linewidth=1.8,
        label=r"$\mathrm{dist}_{\mathcal{D}}(Tx_k)-\delta$",
    )
    semilogy_with_floor(
        ax,
        trace.iterations,
        trace.cdagger_distance,
        color="tab:green",
        linewidth=2.0,
        label=r"$\mathrm{dist}_{\mathcal{C}^\dagger}(x_k)$",
    )
    ax.set_title("Residuals", fontsize=titlefontsize)
    ax.set_xlabel(r"Iteration $k$", fontsize=labelfontsize)
    ax.set_ylabel("residual", fontsize=labelfontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legendfontsize)
    ax.set_xlim(0, trace.iterations[-1])


def draw_growth(ax, trace):
    radius = DOMAIN_RADIUS
    r_grid = np.geomspace(0.07, radius, 500)
    boundary_points = np.column_stack([r_grid, lower_boundary(r_grid)])
    boundary_h_gap, _, boundary_dist = compute_residuals(boundary_points)

    positive = trace.h_gap > 0
    ax.loglog(
        boundary_dist,
        boundary_h_gap,
        color="black",
        linewidth=2.2,
        label=r"boundary $x_2=e^{-1/x_1^2}$",
    )
    ax.loglog(
        trace.cdagger_distance[positive],
        trace.h_gap[positive],
        color="tab:blue",
        linewidth=1.4,
        alpha=0.85,
        label="trajectory",
    )

    ref = np.geomspace(0.07, radius, 120)
    ax.loglog(ref, ref**2, "--", color="0.55", linewidth=1.6, label=r"$r^2$")
    ax.loglog(ref, ref**8, ":", color="0.35", linewidth=2.0, label=r"$r^8$")
    ax.loglog(
        ref,
        np.exp(-1.0 / ref**2),
        "-.",
        color="0.15",
        linewidth=1.8,
        label=r"$e^{-1/r^2}$",
    )

    ax.set_title("Non-Polynomial Growth", fontsize=titlefontsize)
    ax.set_xlabel(r"$r=\mathrm{dist}_{\mathcal{C}^\dagger}(x)$", fontsize=labelfontsize)
    ax.set_ylabel(r"$h(x)-h^\star$", fontsize=labelfontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=legendfontsize)


def save_figure(trace, beta0, step_shift, outfile, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    draw_geometry(axes[0], trace)
    draw_residuals(axes[1], trace)
    draw_growth(axes[2], trace)
    fig.suptitle(
        rf"Definable Inconsistent Geometry; "
        rf"$\beta_0={beta0:g}$, $\beta_k=\beta_0(k+1)^{{-1/4}}$, "
        rf"$\gamma_k=(k+{step_shift:g})^{{-1/2}}$",
        fontsize=suptitlefontsize,
    )
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outfile


def print_diagnostics(trace, n_steps, beta0, step_shift, outfile):
    final_point = trace.points[-1]
    print("Definable exponentially flat inconsistent experiment")
    print(r"  C = {(x_1, x_2): -a <= x_1 <= a, exp(-1/x_1^2) <= x_2 <= 1}")
    print(f"  a = {DOMAIN_RADIUS:g}")
    print(r"  T(x_1, x_2) = x_2, D = {-1}, C^\dagger = {(0, 0)}")
    print(f"  n_steps = {n_steps}, beta0 = {beta0:g}, step_shift = {step_shift:g}")
    print(
        "  "
        f"x_N=({final_point[0]: .6f}, {final_point[1]: .6f})  "
        f"h(x_N)-h^*={trace.h_gap[-1]:.12e}  "
        f"dist_Cdag(x_N)={trace.cdagger_distance[-1]:.12e}"
    )
    if trace.cdagger_distance[-1] > 0.0:
        model = np.exp(-1.0 / trace.cdagger_distance[-1] ** 2)
        print(f"  exp(-1/dist_Cdag^2)={model:.12e}")
    print(f"Wrote {outfile}")


def main():
    args = parse_args()
    configure_matplotlib(use_tex=not args.no_tex)
    trace = run_trace(args.n_steps, args.beta0, args.step_shift, args.max_samples)
    outfile = save_figure(trace, args.beta0, args.step_shift, args.outfile, args.dpi)
    print_diagnostics(trace, args.n_steps, args.beta0, args.step_shift, outfile)


if __name__ == "__main__":
    main()
