"""
2D infeasible FRAMES experiment for inconsistent T(mathcal{C}) and mathcal{D}.

Problem:
    min_{x in mathcal{C}} ||x - x_f||_2^2 + iota_{mathcal{D}}(Tx)

where mathcal{C} = [-1, 1]^2, T = [0 1], so T(x, y) = y, and
mathcal{D} = {2} in the codomain of T. Since T(mathcal{C}) = [-1, 1]
does not intersect mathcal{D}, the algorithm should converge to the minimizer
of ||x - x_f||_2^2 over the closest face

    argmin_{x in mathcal{C}} dist(mathcal{D}, T x)
        = {(x, 1) : x in [-1, 1]}.

In the geometry panel, the dashed line is the primal preimage
T^{-1}(mathcal{D}), not mathcal{D} itself.
"""

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_cache_root = Path(tempfile.gettempdir()) / "nonintersecting_linf_matplotlib"
_font_cache = _cache_root / "fontconfig"
_font_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg-cache"))
os.environ.setdefault("FC_CACHEDIR", str(_font_cache))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

legendfontsize = 12
titlefontsize = 20
labelfontsize = 18
suptitlefontsize = 24

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY_OUTFILE = (
    THIS_DIR / "paper_fig_nonintersecting_linf_trajectories.pdf"
)
DEFAULT_RESIDUALS_OUTFILE = THIS_DIR / "paper_fig_nonintersecting_linf_residuals.pdf"
CENTERS = np.array(
    [
        [-1.5, 0.2],
        [-0.15, 1.75],
        [1.5, 0.25],
    ],
    dtype=float,
)
COLORS = ["tab:blue", "tab:orange", "tab:green"]
DEFAULT_INITIAL_POINT = np.array([-0.2, 0.0], dtype=float)


@dataclass
class Trace:
    center: np.ndarray
    selected_point: np.ndarray
    f_star: float
    points: np.ndarray
    objective_values: np.ndarray
    distance: np.ndarray


@dataclass
class StepSchedule:
    kind: str
    shift: float
    horizon: int

    def step_size(self, iteration):
        if self.kind == "standard":
            return 1.0 / (iteration + 1) ** 0.5
        if self.kind == "shifted":
            return 1.0 / (iteration + self.shift) ** 0.5
        if self.kind == "constant-horizon":
            return 1.0 / (self.horizon + 1) ** 0.5
        raise ValueError(f"Unknown step schedule: {self.kind}")

    def label(self):
        if self.kind == "standard":
            return r"$\gamma_k = (k+1)^{-1/2}$"
        if self.kind == "shifted":
            return rf"$\gamma_k = (k+{self.shift:g})^{{-1/2}}$"
        if self.kind == "constant-horizon":
            return rf"$\gamma_k = (K+1)^{{-1/2}},\; K={self.horizon}$"
        raise ValueError(f"Unknown step schedule: {self.kind}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the inconsistent T(mathcal{C}) experiment and save a paper figure."
        )
    )
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--beta0", type=float, default=3.0)
    parser.add_argument(
        "--trajectory-outfile",
        type=Path,
        default=DEFAULT_TRAJECTORY_OUTFILE,
    )
    parser.add_argument(
        "--residuals-outfile",
        type=Path,
        default=DEFAULT_RESIDUALS_OUTFILE,
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--step-schedule",
        choices=("standard", "shifted", "constant-horizon"),
        default="shifted",
    )
    parser.add_argument("--step-shift", type=float, default=100.0)
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


def linf_lmo(gradient):
    """
    Linear minimization oracle over [-1, 1]^2.

    The zero-gradient tie is sent to -1 to make the trajectories reproducible.
    """
    return np.where(gradient >= 0.0, -1.0, 1.0)


def selected_face_minimizer(center):
    selected = np.array([np.clip(center[0], -1.0, 1.0), 1.0], dtype=float)
    f_star = np.sum((selected - center) ** 2)
    return selected, f_star


def run_trace(center, n_steps, beta0, step_schedule, x_initial=None):
    if x_initial is None:
        x_initial = DEFAULT_INITIAL_POINT

    points = np.empty((n_steps + 1, 2), dtype=float)
    x = np.asarray(x_initial, dtype=float).copy()
    points[0] = x

    for i in range(n_steps):
        beta = beta0 / (i + 1) ** 0.25
        step_size = step_schedule.step_size(i)

        grad_f = 2.0 * (x - center)
        tx_minus_proj = x[1] - 2.0
        moreau_grad = np.array([0.0, tx_minus_proj / beta])
        combined_grad = grad_f + moreau_grad

        direction = linf_lmo(combined_grad)
        x = (1.0 - step_size) * x + step_size * direction
        points[i + 1] = x

    selected, f_star = selected_face_minimizer(center)
    objective_values = np.sum((points - center) ** 2, axis=1)
    distance = np.abs(points[:, 1] - 2.0)
    return Trace(
        center=center,
        selected_point=selected,
        f_star=f_star,
        points=points,
        objective_values=objective_values,
        distance=distance,
    )


def semilogy_with_floor(ax, x_values, y_values, color, label, linewidth=1.8):
    floor = 1e-16
    ax.semilogy(
        x_values,
        np.maximum(y_values, floor),
        color=color,
        label=label,
        linewidth=linewidth,
    )


def trace_label(center):
    return rf"$x_f = ({center[0]:g}, {center[1]:g})$"


def trajectory_marker_indices(n_points):
    if n_points <= 1:
        return np.array([0], dtype=int)

    early = np.arange(0, min(n_points, 20), 2)
    if n_points <= 20:
        return early

    late = np.geomspace(20, n_points - 1, 16)
    return np.unique(np.concatenate([early, np.round(late).astype(int)]))


def draw_geometry(ax, traces):
    box = Rectangle(
        (-1.0, -1.0),
        2.0,
        2.0,
        facecolor="0.96",
        edgecolor="black",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(box)
    ax.plot(
        [-1.0, 1.0],
        [1.0, 1.0],
        color="black",
        linewidth=3.0,
        label=r"$\mathcal{C}^\dagger$",
        zorder=1,
    )
    ax.axhline(
        2.0,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        label=r"$T^{-1}(\mathcal{D})$",
        zorder=1,
    )
    ax.scatter(
        [traces[0].points[0, 0]],
        [traces[0].points[0, 1]],
        marker="o",
        s=45,
        color="black",
        zorder=5,
        label=r"$x_0$",
    )

    center_label_offsets = [
        (0.0, 0.15, "center", "bottom"),
        (0.13, 0.0, "left", "center"),
        (0.0, 0.15, "center", "bottom"),
    ]

    for color, trace, center_label in zip(COLORS, traces, center_label_offsets):
        ax.plot(
            trace.points[:, 0],
            trace.points[:, 1],
            color=color,
            linewidth=1.7,
            alpha=0.85,
            label=trace_label(trace.center),
            zorder=4,
        )
        marker_idx = trajectory_marker_indices(trace.points.shape[0])
        ax.scatter(
            trace.points[marker_idx, 0],
            trace.points[marker_idx, 1],
            marker="o",
            s=15,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.9,
            zorder=5,
        )
        ax.scatter(
            [trace.center[0]],
            [trace.center[1]],
            marker="x",
            s=70,
            linewidths=2.0,
            color=color,
            zorder=6,
        )
        center_dx, center_dy, center_ha, center_va = center_label
        ax.text(
            trace.center[0] + center_dx,
            trace.center[1] + center_dy,
            r"$x_f$",
            color=color,
            fontsize=titlefontsize,
            ha=center_ha,
            va=center_va,
            zorder=8,
        )
        ax.scatter(
            [trace.selected_point[0]],
            [trace.selected_point[1]],
            marker="*",
            s=130,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=7,
        )
        text_offset = np.array([0.08 * np.sign(trace.selected_point[0]), 0.13])
        if abs(trace.selected_point[0]) < 1e-12:
            text_offset = np.array([0.08, 0.13])
        ax.text(
            trace.selected_point[0] + text_offset[0],
            trace.selected_point[1] + text_offset[1],
            r"$x_{\mathcal{C}^\dagger}^\star$",
            color=color,
            fontsize=titlefontsize,
            ha="center",
            va="bottom",
            zorder=8,
        )

    ax.set_title("Inconsistent Constraints", fontsize=titlefontsize)
    ax.set_xlabel(r"$x_1$", fontsize=labelfontsize)
    ax.set_ylabel(r"$x_2$", fontsize=labelfontsize)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.15, 2.15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legendfontsize, loc="lower center", ncol=2)


def title_label(beta0, step_schedule):
    return (
        rf"$\beta_0={beta0:g}$, "
        rf"$\beta_k=\beta_0(k+1)^{{-1/4}}$, {step_schedule.label()}"
    )


def save_trajectory_figure(traces, outfile, dpi):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    draw_geometry(ax, traces)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outfile


def save_residuals_figure(traces, beta0, step_schedule, outfile, dpi):
    iters = np.arange(traces[0].points.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for color, trace in zip(COLORS, traces):
        label = trace_label(trace.center)
        dist_residual = np.abs(trace.distance - 1.0)
        objective_residual = np.abs(trace.objective_values - trace.f_star)

        semilogy_with_floor(axes[0], iters, dist_residual, color=color, label=label)
        semilogy_with_floor(
            axes[1], iters, objective_residual, color=color, label=label
        )

    axes[0].set_title("Distance residual", fontsize=titlefontsize)
    axes[0].set_xlabel(r"Iteration $k$", fontsize=labelfontsize)
    axes[0].set_ylabel(
        r"$|\mathrm{dist}_{\mathcal{D}}(Tx_k) - 1|$",
        fontsize=labelfontsize,
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=legendfontsize)
    axes[0].set_xlim(0, iters[-1])
    axes[0].set_ylim(1e-16, 4e0)

    axes[1].set_title("Objective selection", fontsize=titlefontsize)
    axes[1].set_xlabel(r"Iteration $k$", fontsize=labelfontsize)
    axes[1].set_ylabel(
        r"$|f(x_k) - f(x_{\mathcal{C}^\dagger}^\star)|$",
        fontsize=labelfontsize,
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=legendfontsize)
    axes[1].set_xlim(0, iters[-1])
    axes[1].set_ylim(1e-16, 4e0)

    fig.suptitle(
        rf"Inconsistent $T(\mathcal{{C}})$ and $\mathcal{{D}}$; "
        rf"{title_label(beta0, step_schedule)}",
        fontsize=suptitlefontsize,
    )
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outfile


def print_diagnostics(traces, n_steps, beta0, step_schedule, outfiles):
    print(r"Inconsistent T(\mathcal{C}) experiment")
    print(r"  \mathcal{C} = [-1, 1]^2, T = [0 1], \mathcal{D} = {2} in range(T)")
    print(r"  T^{-1}(\mathcal{D}) = {(x_1, x_2) : x_2 = 2}")
    print(f"  n_steps = {n_steps}, beta0 = {beta0:g}")
    print(f"  step schedule: {step_schedule.label()}")
    print()

    for trace in traces:
        final_point = trace.points[-1]
        final_f = trace.objective_values[-1]
        final_dist = trace.distance[-1]
        objective_residual = abs(final_f - trace.f_star)
        distance_residual = abs(final_dist - 1.0)
        print(
            "  "
            f"x_f=({trace.center[0]: .3f}, {trace.center[1]: .3f})  "
            f"x_N=({final_point[0]: .6f}, {final_point[1]: .6f})  "
            f"x_Cdag^*=({trace.selected_point[0]: .3f}, {trace.selected_point[1]: .3f})"
        )
        print(
            "    "
            f"dist_{{\\mathcal{{D}}}}(Tx_N)={final_dist:.12e}  "
            f"|dist_{{\\mathcal{{D}}}}-1|={distance_residual:.3e}"
        )
        print(
            "    "
            f"f(x_N)={final_f:.12e}  "
            f"f(x_Cdag^*)={trace.f_star:.12e}  "
            f"|f-f(x_Cdag^*)|={objective_residual:.3e}"
        )

    print()
    for outfile in outfiles:
        print(f"Wrote {outfile}")


def main():
    args = parse_args()
    configure_matplotlib(use_tex=not args.no_tex)
    step_schedule = StepSchedule(args.step_schedule, args.step_shift, args.n_steps)
    traces = [
        run_trace(center, args.n_steps, args.beta0, step_schedule) for center in CENTERS
    ]
    trajectory_outfile = save_trajectory_figure(
        traces,
        args.trajectory_outfile,
        args.dpi,
    )
    residuals_outfile = save_residuals_figure(
        traces,
        args.beta0,
        step_schedule,
        args.residuals_outfile,
        args.dpi,
    )
    print_diagnostics(
        traces,
        args.n_steps,
        args.beta0,
        step_schedule,
        [trajectory_outfile, residuals_outfile],
    )


if __name__ == "__main__":
    main()
