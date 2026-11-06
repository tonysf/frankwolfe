"""Plot the observed development scores of the joint smoothing/step search."""

import argparse
from itertools import product
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = {"sframes": "Minibatch SFRAMES", "sframes_full": "Full-batch SFRAMES"}
PAIR_FIELDS = ("data_sha256", "calibration_sha256", "caps", "curvature",
               "train_view_ids", "initialization", "constraint_geometry")


def read(path):
    return json.loads(path.read_text())


def agree(actual, expected, description):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"Saved development values disagree for {description}")


def observed_scores(root):
    """Recompute every cell from its four completed development endpoints."""
    selection = read(root / "selection.json")
    if selection.get("status") != "selected":
        raise ValueError("The adaptive development search has not finalized its grid")
    design = read(root / "manifest.json")["design"]
    expected_cases = set(product(design["development_slices"],
                                 design["evaluation_training_views"]))
    budget = selection["development_view_budget"]
    reference = {}
    panels = {}
    for variant in METHODS:
        grid = selection["grids"][variant]
        betas, gammas = sorted(grid["beta_scales"]), sorted(grid["gamma0"])
        expected_pairs = set(product(betas, gammas))
        records = selection["scores"][variant]
        pairs = [(record["beta_scale"], record["gamma0"]) for record in records]
        if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
            raise ValueError(f"The final {variant} grid is not a complete rectangle")
        scores = {}
        for record in records:
            beta_scale, gamma0 = record["beta_scale"], record["gamma0"]
            cases = record["cases"]
            case_keys = [(case["slice_id"], case["n_train"]) for case in cases]
            if len(case_keys) != len(set(case_keys)) or set(case_keys) != expected_cases:
                raise ValueError(f"Incomplete development cases for {variant}, {beta_scale}, {gamma0}")
            ratios = []
            for case in cases:
                folder = root / "runs" / case["run_id"]
                metadata = read(folder / "metadata.json")
                result = read(folder / "result.json")
                trace = read(folder / "trace.json")
                config = metadata["config"]
                if (config["phase"] != "development" or config["variant"] != variant
                        or config["beta_scale"] != beta_scale or config["rate"] != gamma0
                        or config["slice_id"] != case["slice_id"]
                        or config["n_train"] != case["n_train"]
                        or config["view_budget"] != budget):
                    raise ValueError(f"Development configuration disagrees in {folder}")
                if (result["status"] != "completed" or result["config"] != config
                        or not trace or trace[0]["iteration"] != 0
                        or result["final"] != trace[-1]
                        or result["iteration"] != trace[-1]["iteration"]):
                    raise ValueError(f"Development result is incomplete in {folder}")
                final = trace[-1]
                if final["forward_views"] != budget or final["backward_views"] != budget:
                    raise ValueError(f"Development endpoint misses the common data budget in {folder}")
                signature = [metadata[field] for field in PAIR_FIELDS]
                key = case["slice_id"], case["n_train"]
                if reference.setdefault(key, signature) != signature:
                    raise ValueError(f"Development data or calibration differs for {key}")
                curvature = max(metadata["curvature"])
                agree(metadata["optimizer"]["mu0"], beta_scale / curvature, case["run_id"])
                agree(metadata["optimizer"]["gamma0"], gamma0, case["run_id"])
                initial = trace[0]["feasible_loss"]
                value = final["feasible_loss"]
                if not math.isfinite(value) or initial <= 0 or value <= 0:
                    raise ValueError(f"Invalid relative training loss in {folder}")
                ratio = value / initial
                agree(case["initial_loss"], initial, case["run_id"])
                agree(case["repaired_loss"], value, case["run_id"])
                agree(case["relative_loss"], ratio, case["run_id"])
                if case["forward_views"] != budget:
                    raise ValueError(f"Selection uses different work for {case['run_id']}")
                ratios.append(ratio)
            score = median(ratios)
            agree(record["median_relative_loss"], score, f"{variant} score")
            scores[beta_scale, gamma0] = score
        chosen = selection["selected"][variant]
        pair = chosen["beta_scale"], chosen["gamma0"]
        if pair not in scores:
            raise ValueError(f"Selected pair lies outside the final {variant} grid")
        agree(chosen["median_relative_loss"], scores[pair], f"{variant} selected score")
        agree(scores[pair], min(scores.values()), f"{variant} selected minimum")
        panels[variant] = dict(betas=betas, gammas=gammas, scores=scores, selected=pair)
    return panels, budget


def plot(panels, output):
    """Display tested parameter values as discrete cells with a common color scale."""
    values = [value for panel in panels.values() for value in panel["scores"].values()]
    low, high = min(values), max(values)
    norm = LogNorm(vmin=low, vmax=max(high, low * (1 + 1e-6)))
    height = 2.9 + 0.24 * max(0, max(len(p["betas"]) for p in panels.values()) - 3)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, height), layout="constrained")
    for ax, (variant, panel) in zip(axes, panels.items()):
        betas, gammas = panel["betas"], panel["gammas"]
        matrix = np.asarray([[panel["scores"][beta, gamma] for gamma in gammas]
                             for beta in betas])
        image = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="none",
                          cmap="cividis_r", norm=norm)
        ax.set_title(METHODS[variant], weight="bold")
        ax.set_xlabel(r"Step prefactor $\gamma_0$")
        ax.set_ylabel(r"Smoothing multiplier $c_\beta$")
        ax.set_xticks(range(len(gammas)), [f"{value:g}" for value in gammas])
        ax.set_yticks(range(len(betas)), [f"{value:.4g}" for value in betas])
        ax.set_xticks(np.arange(len(gammas) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(betas) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6, alpha=0.65)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i, beta in enumerate(betas):
            for j, gamma in enumerate(gammas):
                score = matrix[i, j]
                selected = (beta, gamma) == panel["selected"]
                ax.text(j, i, f"{score:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm(score) > 0.55 else "black",
                        weight="bold" if selected else "normal")
                if selected:
                    ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                           fill=False, edgecolor="black", linewidth=1.5))
    colorbar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.025)
    colorbar.set_label("Repaired training loss / initial loss")
    colorbar.set_ticks(np.geomspace(low, max(high, low * (1 + 1e-6)), 4))
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
    colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
    colorbar.ax.tick_params(labelsize=7)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"joint_selection.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    panels, budget = observed_scores(root)
    output = args.output.resolve() if args.output else root / "reports"
    plot(panels, output)
    print(f"Plotted complete observed development grids at {budget:,} views per direction")


if __name__ == "__main__":
    main()
