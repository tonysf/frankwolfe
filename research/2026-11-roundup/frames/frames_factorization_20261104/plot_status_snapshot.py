"""Read saved development probes once and render an immutable status snapshot."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
from datetime import datetime, timezone
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent


def main():
    stamp = datetime.now(timezone.utc)
    output = ROOT / "reports" / ("status_" + stamp.strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True)
    snapshot = {"created_utc": stamp.isoformat(), "datasets": {},
                "campaign_progress": json.loads((ROOT / "progress.json").read_text())}
    for dataset in ("orl64", "yaleb32"):
        records = []
        for path in sorted((ROOT / "runs" / dataset / "development").glob("*/metadata.json")):
            trace_path = path.with_name("trace.json")
            if not trace_path.exists():
                continue
            config = json.loads(path.read_text())["config"]
            trace = json.loads(trace_path.read_text())
            if trace:
                records.append({"run": str(path.parent.relative_to(ROOT)), "config": config,
                                "trace": trace, "completed": path.with_name("result.json").exists()})
        snapshot["datasets"][dataset] = records
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .17, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    summary = {}
    for ax, (dataset, records) in zip(axes, snapshot["datasets"].items()):
        sframes = [r for r in records if r["config"]["method"] == "sframes"]
        spring = next(r for r in records if r["config"]["method"] == "spring_sarah")
        all_sframes_complete = len(sframes) == 12 and all(r["completed"] for r in sframes)
        best = min(sframes, key=lambda r: r["trace"][-1]["feasible_loss"]) if all_sframes_complete else None
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
        other_label = False
        for i, record in enumerate(sframes):
            config, trace = record["config"], record["trace"]
            if best:
                highlighted = record is best
                color, alpha, width = ("#0072B2", 1., 2.8) if highlighted else ("#66A8CB", .45, 1.)
                label = "SFRAMES: best of 12 completed settings" if highlighted else ("SFRAMES: other 11 settings" if not other_label else None)
                if not highlighted:
                    other_label = True
            else:
                color, alpha, width = palette[i % len(palette)], 1., 2.
                label = (f"SFRAMES: ρ₀={config['rho0']:g}, μ₀L={config['mu_scale']:g}"
                         + (" (complete)" if record["completed"] else " (running)"))
            ax.plot([r["iteration"] for r in trace], [r["feasible_loss"] for r in trace],
                    color=color, alpha=alpha, lw=width, label=label,
                    zorder=4 if record is best else 2)
            if not record["completed"]:
                ax.plot(trace[-1]["iteration"], trace[-1]["feasible_loss"], "o", color=color, ms=4)
        trace = spring["trace"]
        ax.plot([r["iteration"] for r in trace], [r["feasible_loss"] for r in trace],
                color="#292929", lw=2.4, label="SPRING-SARAH: theorem parameters")
        floor = records[0]["config"]["rank_r_loss_lower_bound"]
        ax.axhline(floor, color="#888888", ls=":", lw=1.4,
                   label="Unconstrained rank-r lower bound")
        name = "ORL · rank 25" if dataset == "orl64" else "Extended Yale-B · rank 49"
        gammas = ", ".join(f"{g:g}" for g in sorted({r["config"]["gamma0"] for r in sframes}))
        subtitle = ("All 12 SFRAMES settings complete" if all_sframes_complete
                    else f"{len(sframes)} of 12 SFRAMES settings started; γ₀={gammas}")
        ax.set(title=name + "\n" + subtitle, xlabel="Optimizer iterations",
               ylabel="Feasible reconstruction loss (log scale)", yscale="log", xlim=(0, 20000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.legend(loc="upper right", fontsize=8, framealpha=.95)
        common = min(r["trace"][-1]["iteration"] for r in records)
        common = min(row["iteration"] for row in [max((t for t in r["trace"] if t["iteration"] <= common), key=lambda t:t["iteration"]) for r in records])
        summary[dataset] = {"common_iteration": common, "runs": []}
        for record in records:
            initial, final = record["trace"][0], record["trace"][-1]
            row = next(t for t in record["trace"] if t["iteration"] == common)
            summary[dataset]["runs"].append({"candidate": record["config"]["candidate"],
                "config": record["config"], "completed": record["completed"],
                "initial_loss": initial["feasible_loss"], "last_iteration": final["iteration"],
                "final_loss": final["feasible_loss"], "final_loss_reduction": 1-final["feasible_loss"]/initial["feasible_loss"],
                "common_loss": row["feasible_loss"], "common_loss_reduction": 1-row["feasible_loss"]/initial["feasible_loss"]})
    fig.suptitle("SFRAMES and SPRING-SARAH — preliminary development runs (seed 110)", fontsize=14)
    fig.text(.055, .065, "Loss = ‖UVᵀ−A‖² / (2 × pixels × images), evaluated after diagnostic feasibility repair. Lower is better.", fontsize=9)
    fig.text(.055, .027, "L is the full-gradient Lipschitz bound on C; μ₀L denotes the smoothing prefactor scale. Fresh-seed confirmation is pending.", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("loss_vs_iterations." + suffix), dpi=180)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output)
    for dataset, result in summary.items():
        print(dataset, "common iteration", result["common_iteration"])
        for run in result["runs"]:
            print(run["candidate"], "last", run["last_iteration"], "loss", run["final_loss"],
                  "common reduction", run["common_loss_reduction"])


if __name__ == "__main__":
    main()
