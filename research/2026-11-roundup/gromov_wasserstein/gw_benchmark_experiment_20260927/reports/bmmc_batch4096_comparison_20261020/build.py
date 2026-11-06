"""Plot measured feasible BMMC loss, with a separately labelled POT endpoint.

All trajectories retain their actual diagnostic iterations. Lines connect
those diagnostics without smoothing; no POT updates after termination exist.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old = ROOT / "reports/wallclock_comparison_20261018/input/bmmc6224/confirmation"
    specs = [
        ("FRAMES, 4,096 pairs, CPU, seed 811", "#842b8d", "-", 2.5,
         ROOT / "experiments/bmmc_cpu_batch4096_20261018/runs/frames_q4096_seed811/metadata.json"),
        ("FRAMES, 1,024 pairs, A100, seed 811", "#0072b2", "--", 1.6, old / "c08_seed811/metadata.json"),
        ("FRAMES, 1,024 pairs, A100, seed 812", "#56b4e9", "-.", 1.6, old / "c08_seed812/metadata.json"),
        ("FRAMES, 1,024 pairs, A100, seed 813", "#009e73", "-", 1.6, old / "c08_seed813/metadata.json"),
        (r"POT update, full gradient, $\epsilon=10^{-4}$, best start", "#d55e00", "-", 2.2,
         OUT / "input/best_pot_metadata.json"),
    ]
    selection = json.loads((OUT / "input/best_pot_export.json").read_text())
    assert sha256(specs[-1][-1]) == selection["raw_metadata_sha256"]
    records = []
    for label, color, style, width, path in specs:
        meta = json.loads(path.read_text())
        assert meta["status"] == "completed"
        trace = meta["trace"]
        x = [p["iteration"] for p in trace]
        y = [p["rounded_loss"] for p in trace]
        assert x[0] == 0 and all(b > a for a, b in zip(x, x[1:]))
        assert x[-1] == meta["completed_updates"] and y[-1] == meta["final_loss"]
        assert meta["config"]["dtype"] == "float64"
        assert max(p["rounded_residuals"]["both_l1"] for p in trace) < 1e-12
        if records:
            assert meta["array_files_sha256"] == records[0]["meta"]["array_files_sha256"]
            assert meta["manifest_sha256"] == records[0]["meta"]["manifest_sha256"]
        records.append(dict(label=label, color=color, style=style, width=width,
                            path=path, sha256=sha256(path), meta=meta, x=x, y=y))
    for row in records[:-1]:
        c = row["meta"]["config"]
        assert (c["gamma0"], c["rho0"], c["mu_scale"]) == (.3, 1., 10.)
        assert row["x"][-1] == 20000

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.25, 1, 1.1]})
    fig.subplots_adjust(left=.065, right=.985, bottom=.245, top=.72, wspace=.30)
    fig.suptitle("BMMC: feasible GW loss versus optimizer iterations", x=.065, y=.97,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(.065, .915, "6,224 × 6,224 coupling  •  FP64  •  Lower loss is better", color="#444444", fontsize=11)
    for ax in axes:
        # Draw the new run last so the close previous trajectories remain legible.
        for r in records[1:] + records[:1]:
            ax.plot(r["x"], r["y"], label=r["label"], color=r["color"],
                    linestyle=r["style"], linewidth=r["width"], marker="o",
                    markersize=2.5 if r is not records[-1] else 3,
                    markeredgewidth=0, zorder=4 if r is records[0] else 3)
        ax.grid(True, color="#e5e7eb", linewidth=.7)
        ax.set_xlabel("Optimizer iterations")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
        ax.set_axisbelow(True)
    pot = records[-1]
    pot_end, pot_loss = pot["x"][-1], pot["y"][-1]
    for ax in (axes[0], axes[2]):
        ax.hlines(pot_loss, pot_end, 20000, color=pot["color"], linewidth=1.2,
                  linestyle=(0, (3, 3)), alpha=.8, zorder=2)
        ax.scatter([pot_end], [pot_loss], color=pot["color"], s=30, zorder=5)
    axes[0].set(title="Full trajectories", xlim=(0, 20000), ylim=(.1429, .1617))
    axes[0].set_ylabel("Feasible unregularized GW loss")
    axes[0].set_yticks([.144, .148, .152, .156, .160])
    axes[0].xaxis.set_major_locator(MaxNLocator(5))
    axes[0].annotate(f"POT final reference: {pot_loss:.6f}", xy=(13000, pot_loss),
                     xytext=(6000, pot_loss+.00045), fontsize=9, color=pot["color"],
                     arrowprops=dict(arrowstyle="-", color=pot["color"], linewidth=.8))
    axes[1].set(title="First 250 iterations", xlim=(0, 250), ylim=(.1429, .1617))
    axes[1].set_yticks([.144, .148, .152, .156, .160])
    axes[1].annotate(f"Stops at {pot_end} updates", xy=(pot_end, pot_loss),
                     xytext=(85, .1462), fontsize=9, color=pot["color"],
                     arrowprops=dict(arrowstyle="->", color=pot["color"], linewidth=.9))
    axes[2].set(title="Last 10,000 iterations (zoom)", xlim=(10000, 20000), ylim=(.1432, .1462))
    axes[2].xaxis.set_major_locator(MaxNLocator(3))
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
    for r in records[:-1]:
        axes[2].scatter([20000], [r["y"][-1]], color=r["color"], s=20, zorder=6, clip_on=False)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend([by_label[r["label"]] for r in records], [r["label"] for r in records],
               loc="upper left", bbox_to_anchor=(.06, .875), ncol=2, frameon=False,
               fontsize=10, columnspacing=2.3, handlelength=3)
    notes = [
        "All FRAMES runs: product initialization; identical prefactors and theoretical exponents. The 4,096-pair run is a single CPU replicate.",
        f"POT: best of {selection['completed_candidates']} completed entropy/initialization trials; permutation-mixture start, seed 903. Each optimizer update has a different computational cost.",
        f"Dots are measured diagnostics; connecting lines are unsmoothed. The dashed POT line is its final-loss reference, not additional iterations.",
        "CPU and CUDA random streams differ, including when the numerical seed is the same; this comparison does not isolate the effect of batch size.",
    ]
    for y, note in zip((.165, .13, .095, .06), notes):
        fig.text(.065, y, note, fontsize=9, color="#444444")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"bmmc_loss_vs_iterations.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)

    with (OUT / "curves.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("method", "pairs", "device", "seed", "iteration", "feasible_gw_loss"))
        writer.writeheader()
        for r in records:
            c = r["meta"]["config"]
            for iteration, loss in zip(r["x"], r["y"]):
                writer.writerow(dict(method=c["method"], pairs=c.get("pairs") if c["method"] == "frames" else None,
                                     device=c["device"], seed=c["seed"], iteration=iteration, feasible_gw_loss=loss))
    summary = dict(pot_selection={k:v for k,v in selection.items() if k != "raw_metadata"},
                   plotted_metric="rounded_loss: full feasible unregularized squared-distance GW loss",
                   notes=notes, runs=[dict(label=r["label"], source=str(r["path"]), sha256=r["sha256"],
                          config=r["meta"]["config"], iterations=r["x"][-1], initial_loss=r["y"][0],
                          final_loss=r["y"][-1], best_recorded_loss=min(r["y"]), diagnostics=len(r["x"])) for r in records])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"png": str(OUT / "bmmc_loss_vs_iterations.png"),
                      "final_losses": {r["label"]:r["y"][-1] for r in records}}, indent=2))


if __name__ == "__main__":
    main()
