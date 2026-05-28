"""
Semialgebraic inconsistent indicator experiment.

This example shows a semialgebraic inconsistent system where the
best-approximation value residual controls distance to the closest-point set
only through a higher-order polynomial modulus.

    C = {(x1, x2): -1 <= x1 <= 1, x1^8 <= x2 <= 1},
    T(x1, x2) = x2,
    D = {-1}.

Then T(C) = [0, 1] is disjoint from D, delta = 1, and
C^\dagger = {(0, 0)}.  Along the lower boundary x2 = x1^8,
h(x)-h^* is proportional to dist_{C^\dagger}(x)^8.
"""

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "semialgebraic_inconsistent_matplotlib"
_font_cache = _cache_root / "fontconfig"
_font_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg-cache"))
os.environ.setdefault("FC_CACHEDIR", str(_font_cache))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

legendfontsize = 12
titlefontsize = 20
labelfontsize = 18
suptitlefontsize = 24

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTFILE = THIS_DIR / "paper_fig_semialgebraic_inconsistent.pdf"
DEFAULT_CENTER = np.array([1.0, 0.0], dtype=float)
DEFAULT_INITIAL_POINT = np.array([1.0, 1.0], dtype=float)
POWER = 8
DELTA = 1.0


@dataclass
class Trace:
    center: np.ndarray
    points: np.ndarray
    h_gap: np.ndarray
    distance_residual: np.ndarray
    cdagger_distance: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the semialgebraic inconsistent experiment."
    )
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--beta0", type=float, default=3.0)
    parser.add_argument("--step-shift", type=float, default=100.0)
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
    return x1**POWER


def semialgebraic_lmo(gradient):
    """
    Linear minimization oracle over C = {x1^8 <= x2 <= 1, |x1| <= 1}.
    """
    grad_x, grad_y = gradient

    if grad_y < 0.0:
        x1 = -1.0 if grad_x >= 0.0 else 1.0
        return np.array([x1, 1.0], dtype=float)

    if grad_y == 0.0:
        x1 = -1.0 if grad_x >= 0.0 else 1.0
        return np.array([x1, lower_boundary(x1)], dtype=float)

    if grad_x == 0.0:
        x1 = 0.0
    else:
        x1 = -np.sign(grad_x) * (abs(grad_x) / (POWER * grad_y)) ** (1.0 / (POWER - 1))
        x1 = np.clip(x1, -1.0, 1.0)
    return np.array([x1, lower_boundary(x1)], dtype=float)


def compute_residuals(points):
    x2 = points[:, 1]
    dist_d = x2 + 1.0
    h_gap = x2 + 0.5 * x2**2
    distance_residual = dist_d - DELTA
    cdagger_distance = np.linalg.norm(points, axis=1)
    return h_gap, distance_residual, cdagger_distance


def run_trace(n_steps, beta0, step_shift, center=None, x_initial=None):
    if center is None:
        center = DEFAULT_CENTER
    if x_initial is None:
        x_initial = DEFAULT_INITIAL_POINT

    points = np.empty((n_steps + 1, 2), dtype=float)
    x = np.asarray(x_initial, dtype=float).copy()
    points[0] = x

    for k in range(n_steps):
        beta = beta0 / (k + 1) ** 0.25
        step_size = 1.0 / (k + step_shift) ** 0.5

        grad_f = 2.0 * (x - center)
        moreau_grad = np.array([0.0, (x[1] + 1.0) / beta])
        direction = semialgebraic_lmo(grad_f + moreau_grad)
        x = (1.0 - step_size) * x + step_size * direction
        points[k + 1] = x

    h_gap, distance_residual, cdagger_distance = compute_residuals(points)
    return Trace(
        center=np.asarray(center, dtype=float),
        points=points,
        h_gap=h_gap,
        distance_residual=distance_residual,
        cdagger_distance=cdagger_distance,
    )


def semilogy_with_floor(ax, x_values, y_values, **kwargs):
    ax.semilogy(x_values, np.maximum(y_values, 1e-300), **kwargs)


def draw_geometry(ax, trace):
    x_grid = np.linspace(-1.0, 1.0, 600)
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
        label=r"$x_2=x_1^8$",
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

    ax.set_title("Semialgebraic", fontsize=titlefontsize)
    ax.set_xlabel(r"$x_1$", fontsize=labelfontsize)
    ax.set_ylabel(r"$x_2$", fontsize=labelfontsize)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legendfontsize, loc="lower center", ncol=2)


def draw_residuals(ax, trace):
    iters = np.arange(trace.points.shape[0])
    semilogy_with_floor(
        ax,
        iters,
        trace.h_gap,
        color="tab:blue",
        linewidth=2.0,
        label=r"$h(x_k)-h^\star$",
    )
    semilogy_with_floor(
        ax,
        iters,
        trace.distance_residual,
        color="tab:orange",
        linewidth=1.8,
        label=r"$\mathrm{dist}_{\mathcal{D}}(Tx_k)-\delta$",
    )
    semilogy_with_floor(
        ax,
        iters,
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
    ax.set_xlim(0, iters[-1])


def draw_growth(ax, trace):
    x_grid = np.geomspace(1e-4, 1.0, 500)
    boundary_points = np.column_stack([x_grid, lower_boundary(x_grid)])
    boundary_h_gap, _, boundary_dist = compute_residuals(boundary_points)

    positive = trace.h_gap > 0
    ax.loglog(
        boundary_dist,
        boundary_h_gap,
        color="black",
        linewidth=2.2,
        label=r"boundary $x_2=x_1^8$",
    )
    ax.loglog(
        trace.cdagger_distance[positive],
        trace.h_gap[positive],
        color="tab:blue",
        linewidth=1.4,
        alpha=0.85,
        label="trajectory",
    )

    ref = np.geomspace(1e-3, 1.0, 120)
    ax.loglog(ref, ref**2, "--", color="0.55", linewidth=1.6, label=r"$r^2$")
    ax.loglog(ref, ref**4, ":", color="0.35", linewidth=2.0, label=r"$r^4$")
    ax.loglog(ref, ref**8, "-.", color="0.15", linewidth=1.8, label=r"$r^8$")

    ax.set_title("Polynomial Growth", fontsize=titlefontsize)
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
        rf"Semialgebraic Inconsistent Geometry; "
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
    print("Semialgebraic inconsistent experiment")
    print(r"  C = {(x_1, x_2): -1 <= x_1 <= 1, x_1^8 <= x_2 <= 1}")
    print(r"  T(x_1, x_2) = x_2, D = {-1}, C^\dagger = {(0, 0)}")
    print(f"  n_steps = {n_steps}, beta0 = {beta0:g}, step_shift = {step_shift:g}")
    print(
        "  "
        f"x_N=({final_point[0]: .6f}, {final_point[1]: .6f})  "
        f"h(x_N)-h^*={trace.h_gap[-1]:.12e}  "
        f"dist_Cdag(x_N)={trace.cdagger_distance[-1]:.12e}"
    )
    if trace.h_gap[-1] > 0.0:
        print(
            "  "
            f"dist_Cdag/(h-h*)^(1/8)="
            f"{trace.cdagger_distance[-1] / trace.h_gap[-1] ** (1.0 / POWER):.6f}"
        )
    print(f"Wrote {outfile}")


def main():
    args = parse_args()
    configure_matplotlib(use_tex=not args.no_tex)
    trace = run_trace(args.n_steps, args.beta0, args.step_shift)
    outfile = save_figure(trace, args.beta0, args.step_shift, args.outfile, args.dpi)
    print_diagnostics(trace, args.n_steps, args.beta0, args.step_shift, outfile)


if __name__ == "__main__":
    main()
