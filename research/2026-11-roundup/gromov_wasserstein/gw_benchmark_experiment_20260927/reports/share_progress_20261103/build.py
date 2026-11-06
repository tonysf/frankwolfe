"""Plot saved SHARE-seq observations without inventing diagnostic timestamps."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
INPUT = OUT / "input/traces.json"
DATA = json.loads(INPUT.read_text())
RUNS = DATA["runs"]
FULL = next(r for r in RUNS if r["method"] == "pot-full")
PAIR = [r for r in RUNS if r["method"] == "pot-pair"]
FRAMES = [r for r in RUNS if r["method"] == "frames"]
GROUPS = {}
for r in FRAMES:
    GROUPS.setdefault(r["run_id"].split("/")[-1].split("_seed")[0], []).append(r)
COMPLETE_GROUPS = {k: rs for k, rs in GROUPS.items()
                   if len(rs) == 2 and all(r["status"] == "completed" for r in rs)}
BEST_ID = min(COMPLETE_GROUPS, key=lambda k: np.median([r["final_loss"] for r in COMPLETE_GROUPS[k]]))
BEST = COMPLETE_GROUPS[BEST_ID]
PURPLE, BLUE, ORANGE, GREY = "#7b3294", "#0072b2", "#d55e00", "#9ca3af"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def series(r):
    return (np.array([t["iteration"] for t in r["trace"]]),
            np.array([t["rounded_loss"] for t in r["trace"]]))


def verify():
    assert len(RUNS) == 24 and len(FRAMES) == 20
    for r in RUNS:
        x, y = series(r)
        assert r["dimensions"] == {"n": 31890, "m": 31890}
        assert r["config"]["dtype"] == "float64"
        assert r["array_files_sha256"] == FULL["array_files_sha256"]
        assert r["manifest_sha256"] == FULL["manifest_sha256"]
        assert x[0] == 0 and np.all(np.diff(x) > 0) and np.isfinite(y).all()
        assert y[0] == FULL["trace"][0]["rounded_loss"]
        assert max(t["rounded_residuals"]["both_l1"] for t in r["trace"]) < 1e-12
        assert x[-1] <= r["completed_updates"]
        if r["status"] == "completed":
            assert x[-1] == r["completed_updates"] and y[-1] == r["final_loss"]
        assert r["elapsed_seconds"] > 0
    assert BEST_ID == "c06"


def style(ax):
    ax.grid(color="#e5e7eb", linewidth=.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_ylabel("Feasible unregularized GW loss ↓")


def band(ax, rs, color, label):
    x = series(rs[0])[0]
    assert all(np.array_equal(series(r)[0], x) for r in rs)
    ys = np.array([series(r)[1] for r in rs])
    ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=.17)
    ax.plot(x, np.median(ys, axis=0), color=color, lw=2.3, label=label, zorder=4)
    ax.scatter(x[-1], np.median(ys, axis=0)[-1], color=color, s=25, zorder=5)


def save(fig, name, pdf):
    fig.savefig(OUT / f"{name}.png", dpi=170, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def iteration_plot(pdf):
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.3))
    fig.subplots_adjust(left=.075, right=.98, top=.72, bottom=.23, wspace=.25)
    fig.suptitle("SHARE-seq: loss versus iterations", x=.075, y=.965,
                 ha="left", fontsize=19, weight="bold")
    fig.text(.075, .915, "31,890 cells per modality · FP64 · Product initialization · Observations saved by 3 November 2026", fontsize=11)
    for ax in axes:
        for i, r in enumerate(r for r in FRAMES if r not in BEST):
            x, y = series(r)
            ax.plot(x, y, color=GREY, alpha=.45, lw=1,
                    label="Other FRAMES runs (18, including 1 partial)" if i == 0 else None)
        band(ax, BEST, PURPLE, "Best completed FRAMES setting: median + range of 2 seeds")
        band(ax, PAIR, ORANGE, "Pair-sampled POT: median + range of 3 seeds")
        x, y = series(FULL)
        ax.plot(x, y, color=BLUE, lw=2.2, label="Full-gradient POT: 1 run", zorder=5)
        ax.scatter(x[-1], y[-1], color=BLUE, s=25, zorder=6)
        ax.plot([2000, 20000], [y[-1]] * 2, color=BLUE, lw=1.1, ls=":")
        pfinal = float(np.median([r["final_loss"] for r in PAIR]))
        ax.plot([2000, 20000], [pfinal] * 2, color=ORANGE, lw=1.1, ls=":")
        style(ax)
        ax.set_xlabel("Completed outer iterations")
    axes[0].set(xlim=(0, 20500), ylim=(.1015, .1188), title="Full trajectory and all available FRAMES runs")
    axes[0].set_xticks([0, 5000, 10000, 15000, 20000])
    axes[1].set(xlim=(5000, 20500), ylim=(.1019, .1042), title="Later progress: enlarged loss scale")
    axes[1].set_xticks([5000, 10000, 15000, 20000])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.067, .875), ncol=2,
               frameon=False, fontsize=10, columnspacing=2)
    for y, text in zip([.15, .11, .07, .03], [
        "FRAMES: 1,024 sampled pairs/update; best setting γ₀=0.3, ρ₀=0.1, smoothing scale=10. Theoretical exponents fixed.",
        "POT: ε=10⁻³, step=1/ε; sampled version uses 1,024 pairs/update. POT trajectories end at 2,000 updates.",
        "Dotted segments beyond 2,000 are final POT loss references, not additional measured iterations. Bands show seed ranges.",
        "FRAMES setting selected on these tuning seeds; fresh-seed confirmation is pending. Lines connect saved diagnostics.",
    ]):
        fig.text(.075, y, text, fontsize=9.4, color="#444444")
    save(fig, "loss_vs_iterations", pdf)


def grid_plot(pdf):
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.16, hspace=.32, wspace=.20)
    fig.suptitle("SHARE-seq: every available FRAMES setting", x=.075, y=.965,
                 ha="left", fontsize=19, weight="bold")
    fig.text(.075, .925, "19 completed runs + 1 partial · γ₀=0.3 throughout · 1,024 sampled pairs/update · Same loss scale in every panel", fontsize=10.8)
    colors = {.1: "#009e73", .3: "#cc79a7", 1.: "#e69f00"}
    handles = [Line2D([0], [0], color=c, lw=2, label=f"ρ₀={rho:g}") for rho, c in colors.items()]
    handles += [Line2D([0], [0], color="#555555", lw=1.5, ls=ls, label=f"Seed {seed}")
                for seed, ls in [(810, "-"), (814, "--")]]
    handles += [Line2D([0], [0], color=BLUE, ls=":", label="Full-gradient POT final loss")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.067, .89), ncol=6,
               frameon=False, fontsize=9.5, columnspacing=1.3)
    for ax, scale in zip(axes.flat, [.1, 1., 10., 100.]):
        rs = [r for r in FRAMES if r["config"]["mu_scale"] == scale]
        for r in rs:
            x, y = series(r)
            ax.plot(x, y, color=colors[r["config"]["rho0"]], lw=1.7,
                    ls="-" if r["seed"] == 810 else "--")
            if r["status"] != "completed":
                ax.scatter(x[-1], y[-1], color=colors[r["config"]["rho0"]], marker="x", s=50)
                ax.annotate("last diagnostic: 17,000", (x[-1], y[-1]), xytext=(-4, 14),
                            textcoords="offset points", ha="right", fontsize=9)
        ax.axhline(FULL["final_loss"], color=BLUE, ls=":", lw=1.2)
        style(ax)
        ax.set(xlim=(0, 20500), ylim=(.1015, .1188), title=f"Smoothing scale = {scale:g}  ({len(rs)} runs)")
        ax.set_xticks([0, 5000, 10000, 15000, 20000])
        ax.set_xlabel("Completed outer iterations")
    fig.text(.075, .10, "All six runs at smoothing scale 10 beat the full-gradient POT endpoint; the other observed scales remain worse.", fontsize=10)
    fig.text(.075, .065, "Scale 100 has only ρ₀=0.1 available; one run is saved at 17,451 updates, with its last loss diagnostic at 17,000.", fontsize=9.6)
    fig.text(.075, .03, "The γ₀=1 grid and remaining γ₀=0.3 settings have not produced results. No smoothing or loss values are extrapolated.", fontsize=9.6)
    save(fig, "all_frames_settings", pdf)


def endpoint_plot(pdf):
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    fig.subplots_adjust(left=.10, right=.97, top=.73, bottom=.24)
    fig.suptitle("SHARE-seq: final loss versus measured total runtime", x=.10, y=.965,
                 ha="left", fontsize=17, weight="bold")
    fig.text(.10, .915, "Completed runs only · Four A100 GPUs per run · Endpoints, not loss-versus-time trajectories", fontsize=10.5)
    other = [r for r in FRAMES if r not in BEST and r["status"] == "completed"]
    ax.scatter([r["elapsed_seconds"] / 60 for r in other], [r["final_loss"] for r in other],
               color=GREY, s=40, alpha=.75, label="Other FRAMES settings: 20,000 updates")
    ax.scatter([r["elapsed_seconds"] / 60 for r in BEST], [r["final_loss"] for r in BEST],
               color=PURPLE, s=65, marker="s", label="Best FRAMES setting: 20,000 updates", zorder=5)
    ax.scatter([r["elapsed_seconds"] / 60 for r in PAIR], [r["final_loss"] for r in PAIR],
               color=ORANGE, s=65, marker="^", label="Pair-sampled POT: 2,000 updates", zorder=5)
    ax.scatter(FULL["elapsed_seconds"] / 60, FULL["final_loss"], color=BLUE, s=65,
               marker="D", label="Full-gradient POT: 2,000 updates", zorder=5)
    ax.annotate("≈48 min", (BEST[0]["elapsed_seconds"] / 60, np.mean([r["final_loss"] for r in BEST])),
                xytext=(-12, -20), textcoords="offset points", ha="right", color=PURPLE, fontsize=10)
    ax.annotate("≈71 min", (PAIR[0]["elapsed_seconds"] / 60, np.mean([r["final_loss"] for r in PAIR])),
                xytext=(6, 12), textcoords="offset points", color=ORANGE, fontsize=10)
    ax.annotate("≈151 min", (FULL["elapsed_seconds"] / 60, FULL["final_loss"]),
                xytext=(-6, 12), textcoords="offset points", ha="right", color=BLUE, fontsize=10)
    style(ax)
    ax.set(xlim=(0, 165), ylim=(.1010, .1142), xlabel="Measured total runtime per completed run (minutes)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.09, .88), ncol=2,
               frameon=False, fontsize=9.6)
    for y, text in zip([.16, .115, .07, .025], [
        "Timing includes loading, optimization, feasible rounding, loss diagnostics and checkpoint work; queueing and tuning are excluded.",
        "Per-diagnostic timestamps were not recorded. No interpolation of iteration counts into time is used.",
        "These endpoints do not establish time to a target loss: full-gradient POT was already near its final loss by 50–200 updates.",
        "All runs used fixed iteration budgets. FRAMES selection used the same tuning seeds shown here; confirmation is pending.",
    ]):
        fig.text(.10, y, text, fontsize=9.1, color="#444444")
    save(fig, "final_loss_vs_runtime", pdf)


def export():
    fields = ["run_id", "method", "seed", "status", "gamma0", "rho0", "mu_scale", "iteration",
              "rounded_loss", "raw_loss", "run_elapsed_seconds", "completed_updates"]
    with (OUT / "curves.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in RUNS:
            for t in r["trace"]:
                row = {k: r[k] for k in ["run_id", "method", "seed", "status", "completed_updates"]}
                row.update({k: r["config"][k] for k in ["gamma0", "rho0", "mu_scale"]})
                row.update({k: t[k] for k in ["iteration", "rounded_loss", "raw_loss"]})
                row["run_elapsed_seconds"] = r["elapsed_seconds"]
                w.writerow(row)
    initial = FULL["trace"][0]["rounded_loss"]
    bestfinal = float(np.median([r["final_loss"] for r in BEST]))
    last5k = float(np.median([next(t["rounded_loss"] for t in r["trace"] if t["iteration"] == 15000) for r in BEST]))
    summary = dict(input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        fetched_utc=DATA["fetched_utc"], n=31890, completed_runs=sum(r["status"] == "completed" for r in RUNS),
        partial_runs=sum(r["status"] != "completed" for r in RUNS), best_completed_setting=BEST_ID,
        best_config=BEST[0]["config"], initial_loss=initial, best_frames_median=bestfinal,
        full_pot_loss=FULL["final_loss"], sampled_pot_median=float(np.median([r["final_loss"] for r in PAIR])),
        frames_reduction_percent=100*(initial-bestfinal)/initial,
        full_pot_reduction_percent=100*(initial-FULL["final_loss"])/initial,
        relative_loss_advantage_percent=100*(FULL["final_loss"]-bestfinal)/FULL["final_loss"],
        best_frames_last_5000_loss_decrease=last5k-bestfinal,
        first_recorded_iteration_below_full_pot={str(r["seed"]): next(t["iteration"] for t in r["trace"] if t["rounded_loss"] < FULL["final_loss"]) for r in BEST},
        per_diagnostic_times_available=False,
        timing_scope="Total per-run time includes diagnostics, rounding, loading and checkpoints; no per-loss timestamps.",
        selection_scope="Best completed two-seed tuning setting; no fresh-seed confirmation yet.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    verify()
    with PdfPages(OUT / "share_progress.pdf") as pdf:
        iteration_plot(pdf)
        grid_plot(pdf)
        endpoint_plot(pdf)
    export()
