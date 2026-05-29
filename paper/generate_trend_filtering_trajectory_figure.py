"""
Generate the trend-filtering trajectory comparison figure.

This reads saved best configurations from hyperparam_sweep_outputs when they
exist, otherwise it uses the default configurations embedded below. It
reconstructs the SCAD and MCP trajectories used in the paper comparison plot
without rerunning the full hyperparameter sweep.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(THIS_DIR / ".mplconfig"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

legendfontsize = 12
titlefontsize = 20
labelfontsize = 18
suptitlefontsize = 24
import numpy as np

from frank_wolfe import Frames
from paper.experiments.trend_filtering_matrix_factorization import (
    TrendFilteredMFObjective,
    create_spectral_ball_product_lmo,
    generate_piecewise_constant_mf,
    make_mcp_functions,
    make_scad_functions,
)

DEFAULT_OUTPUT_DIR = THIS_DIR / "hyperparam_sweep_outputs"
DEFAULT_BEST_CONFIGS = {
    "scad": {
        "lam": 8.254041852680183,
        "beta0": 0.00015848931924611142,
        "a": 5.16227766016838,
    },
    "mcp": {
        "lam": 10.0,
        "beta0": 0.00018836490894898002,
        "gamma": 4.16227766016838,
    },
}
SCAD_COLOR = "tab:orange"
MCP_COLOR = "tab:green"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the SCAD/MCP trend-filtering trajectory figure."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--r", type=int, default=50)
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--margin", type=float, default=1.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--outfile",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "trend_filtering_trajectory_comparison.pdf",
    )
    parser.add_argument("--scad-color", default=SCAD_COLOR)
    parser.add_argument("--mcp-color", default=MCP_COLOR)
    return parser.parse_args()


def parse_float(value):
    if value in (None, ""):
        return None
    return float(value)


def load_best_configs(output_dir):
    path = Path(output_dir) / "best_configs.csv"
    if not path.exists():
        return DEFAULT_BEST_CONFIGS

    rows = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            for key in (
                "lam",
                "beta0",
                "shape_value",
                "shape_delta",
                "a",
                "gamma",
                "rho",
                "tail_rel_err",
                "final_rel_err",
                "cumulative_rel_err",
                "min_gap",
                "final_gap",
            ):
                parsed[key] = parse_float(parsed.get(key))
            rows.append(parsed)
    by_penalty = {row["penalty"]: row for row in rows}
    missing = {"scad", "mcp"} - set(by_penalty)
    if missing:
        raise RuntimeError(f"Missing best config rows for: {sorted(missing)}")
    return by_penalty


def run_trace(args, penalty, config):
    X_star, _, _, tau_U, tau_V = generate_piecewise_constant_mf(
        args.m,
        args.n,
        args.r,
        n_blocks=args.n_blocks,
        margin=args.margin,
        seed=args.seed,
    )
    x0 = np.zeros(args.m * args.r + args.n * args.r)
    lmo = create_spectral_ball_product_lmo(tau_U, tau_V, args.m, args.n, args.r)
    obj = TrendFilteredMFObjective(X_star, args.m, args.n, args.r)

    if penalty == "scad":
        prox_fn, deriv_fn, _, _ = make_scad_functions(config["lam"], config["a"])
        label = (
            r"SCAD "
            f"($\\lambda={config['lam']:.3g},\\ a={config['a']:.3g},\\ "
            f"\\beta_0={config['beta0']:.3g}$)"
        )
    elif penalty == "mcp":
        prox_fn, deriv_fn, _, _ = make_mcp_functions(config["lam"], config["gamma"])
        label = (
            r"MCP "
            f"($\\lambda={config['lam']:.3g},\\ \\gamma={config['gamma']:.3g},\\ "
            f"\\beta_0={config['beta0']:.3g}$)"
        )
    else:
        raise ValueError(f"Unknown penalty: {penalty}")

    obj.minimal_norm_selection = deriv_fn
    frames = Frames(obj, lmo, prox_fn, "lipschitz")
    frames.run(x0, beta0=config["beta0"], n_steps=args.n_steps, show_progress=False)

    x_star_fro = np.linalg.norm(X_star, "fro")
    rel_err = np.sqrt(2.0 * np.maximum(frames.func_vals, 0.0)) / x_star_fro
    iters = np.arange(1, args.n_steps + 1)
    return {
        "penalty": penalty,
        "label": label,
        "iters": iters,
        "rel_err": rel_err,
        "min_gaps": np.minimum.accumulate(frames.gaps),
        "avg_gaps": np.cumsum(frames.gaps) / iters,
    }


def save_plot(args, best_configs):
    traces = [
        run_trace(args, "scad", best_configs["scad"]),
        run_trace(args, "mcp", best_configs["mcp"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_stride = 10

    for trace in traces:
        color = args.scad_color if trace["penalty"] == "scad" else args.mcp_color
        plot_idx = np.arange(plot_stride - 1, len(trace["iters"]), plot_stride)
        plot_iters = trace["iters"][plot_idx]
        axes[0].semilogy(
            plot_iters,
            trace["rel_err"][plot_idx],
            linewidth=2.0,
            color=color,
            label=trace["label"],
        )
        axes[1].semilogy(
            plot_iters,
            trace["min_gaps"][plot_idx],
            linewidth=2.0,
            color=color,
            label=trace["label"] + " min",
        )
        axes[1].semilogy(
            plot_iters,
            trace["avg_gaps"][plot_idx],
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
            color=color,
            label=trace["label"] + " avg",
        )

    axes[0].set_title("Relative reconstruction error", fontsize=titlefontsize)
    axes[0].set_xlabel(r"Iteration $k$", fontsize=labelfontsize)
    axes[0].set_ylabel(
        r"$\|U_k V_k^{\top} - X^{\star}\|_F / \|X^{\star}\|_F$", fontsize=labelfontsize
    )
    axes[0].set_ylim(3e-3, 2e-2)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=legendfontsize)

    axes[1].set_title("Smoothed gaps", fontsize=titlefontsize)
    axes[1].set_xlabel(r"Iteration $k$", fontsize=labelfontsize)
    axes[1].set_ylabel(r"$\mathrm{gap}^{\beta_k}$", fontsize=labelfontsize)
    axes[1].set_ylim(9e5, 4e6)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=legendfontsize)
    fig.suptitle(
        f"Trend filtering matrix factorization,  $m={args.m},\\; n={args.n},\\; r={args.r}$",
        fontsize=suptitlefontsize,
    )
    fig.tight_layout()
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile = args.outfile
    if outfile.suffix == "":
        outfile = outfile.with_suffix(".pdf")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main():
    args = parse_args()
    best_configs = load_best_configs(args.output_dir)
    outfile = save_plot(args, best_configs)
    print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
