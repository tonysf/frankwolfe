"""Scientific plots from saved probes; no optimization or background polling."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent


def collect(dataset):
    records = []
    for metadata in (ROOT / "runs" / dataset).glob("*/*/metadata.json"):
        trace_path = metadata.with_name("trace.json")
        if not trace_path.exists():
            continue
        meta = json.loads(metadata.read_text())
        trace = json.loads(trace_path.read_text())
        if trace:
            records.append((meta["config"], trace))
    confirmed = [r for r in records if r[0]["phase"] == "confirmation"]
    return confirmed if confirmed else records


def main():
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    summary = {}
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    with PdfPages(report / "comparison_plots.pdf") as pdf:
        for dataset in ("orl64", "yaleb32"):
            records = collect(dataset)
            if not records:
                continue
            grouped = {}
            for config, trace in records:
                key = config["candidate"]
                grouped.setdefault(key, []).append((config, trace))
            summary[dataset] = {}
            for xkey, xlabel, suffix in [
                    ("iteration", "Optimizer iterations", "iterations"),
                    ("optimizer_seconds", "Optimizer time (seconds; diagnostics excluded)", "optimizer_time"),
                    ("component_gradient_calls", "Image-component gradient evaluations (including refreshes)", "oracle_calls")]:
                fig, ax = plt.subplots(figsize=(8, 5))
                for index, (key, runs) in enumerate(sorted(grouped.items())):
                    config = runs[0][0]
                    if config["method"] == "spring_sarah":
                        label = "SPRING-SARAH: theorem parameters, numerical prox"
                        color = "black"
                    else:
                        label = (f"SFRAMES γ₀={config['gamma0']:g}, ρ₀={config['rho0']:g}, "
                                 f"μ scale={config['mu_scale']:g}")
                        color = plt.cm.tab20(index % 20)
                    xmax = min(r[1][-1][xkey] for r in runs)
                    xs = np.linspace(0, xmax, 300) if xmax else np.array([0.])
                    ys = np.array([np.interp(xs, [r[xkey] for r in tr], [r["feasible_loss"] for r in tr])
                                   for _, tr in runs])
                    median = np.median(ys, axis=0)
                    ax.plot(xs, median, label=label + (f" (median {len(runs)})" if len(runs)>1 else ""), color=color)
                    if len(runs)>1:
                        ax.fill_between(xs, np.min(ys,axis=0), np.max(ys,axis=0), color=color, alpha=.12)
                    if xkey == "iteration":
                        summary[dataset][key] = dict(method=config["method"], runs=len(runs),
                            final_feasible_losses=[tr[-1]["feasible_loss"] for _, tr in runs],
                            final_iterations=[tr[-1]["iteration"] for _, tr in runs],
                            median_final_loss=float(np.median([tr[-1]["feasible_loss"] for _, tr in runs])))
                floor = records[0][0]["rank_r_loss_lower_bound"]
                ax.axhline(floor, ls=":", color="grey", label="Unconstrained rank-r lower bound")
                phase = records[0][0]["phase"]
                ax.set(title=f"{dataset}: {phase}", xlabel=xlabel, ylabel="Feasible half mean squared reconstruction loss")
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
                ax.legend(fontsize=7, loc="upper right")
                fig.text(.01, .01, "Feasible repair is diagnostic only. Time includes LMO/prox; excludes metrics and checkpoint IO.", fontsize=7)
                fig.tight_layout(rect=(0,.035,1,1))
                fig.savefig(report / f"{dataset}_loss_vs_{suffix}.png", dpi=170)
                pdf.savefig(fig)
                plt.close(fig)
            fig, axes = plt.subplots(1,2,figsize=(10,4))
            for index, (key, runs) in enumerate(sorted(grouped.items())):
                for run_index, (config, trace) in enumerate(runs):
                    xs = [r["iteration"] for r in trace]
                    axes[0].plot(xs, [max(r["D_distance"],1e-16) for r in trace], label=key if run_index==0 else None)
                    axes[1].plot(xs, [r["relative_reconstruction_error"] for r in trace])
            axes[0].set(yscale="log", xlabel="Iterations", ylabel="Distance to positivity / l1 constraints", title=dataset)
            axes[1].set(xlabel="Iterations", ylabel="Feasible relative reconstruction error", title="Lower is better")
            axes[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(report / f"{dataset}_feasibility_and_reconstruction.png", dpi=170)
            pdf.savefig(fig)
            plt.close(fig)
    (report / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
