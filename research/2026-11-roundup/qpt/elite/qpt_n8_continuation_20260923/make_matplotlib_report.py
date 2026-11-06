"""Create the final five-page continuation PDF from verified scalar JSON."""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "n8_continue_200k_20260923_01"
PARENT_PATH = Path("/home/tony/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/n8_step0p5_rho0p6/summary.json")
OUT = ROOT / "qpt_n8_continuation_20260923.pdf"


def read(path):
    return json.loads(Path(path).read_text())


parent = read(PARENT_PATH)
summary = read(RUN / "summary.json")
verification = read(RUN / "verification.json")
status = read(RUN / "status.json")
provenance = read(RUN / "provenance.json")
collection = read(ROOT / "collection_verification.json")

assert status["status"] == "completed"
assert verification["status"] == "verified" and verification["primary_target_met"]
assert collection["status"] == "verified_and_backed_up"
old = parent["trajectory"]
new = summary["trajectory"]
p0, p1 = old[-1], new[-1]
meta = summary["metadata"]

INK = "#253343"
BLUE = "#176B9A"
ORANGE = "#BF651D"
MUTED = "#617183"
LINE = "#DCE4EA"
PALE = "#EBF4F9"
STRIPE = "#F5F7F9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#B7C3CE",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": "#E0E6EB",
    "grid.linewidth": 0.7,
    "pdf.fonttype": 42,
})


def canvas(page, title, kicker):
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.lines.append(Line2D([0.045, 0.955], [0.955, 0.955], transform=fig.transFigure, color=LINE, lw=0.8))
    fig.text(0.045, 0.966, "STOCHASTIC-FRAMES / QUANTUM PROCESS TOMOGRAPHY", fontsize=7.1, color=MUTED)
    fig.text(0.955, 0.966, "Eight-qubit exact continuation | 24 September 2026", ha="right", fontsize=7.1, color=MUTED)
    fig.text(0.055, 0.915, title, fontsize=20, weight="bold", color=INK)
    fig.text(0.055, 0.88, kicker, fontsize=8.5, color=MUTED)
    fig.lines.append(Line2D([0.045, 0.955], [0.052, 0.052], transform=fig.transFigure, color=LINE, lw=0.8))
    fig.text(0.045, 0.032, "One fixed dataset / initialization / sampling seed; complete-factor fidelity and TP", fontsize=7.1, color=MUTED)
    fig.text(0.955, 0.032, str(page), ha="right", fontsize=7.1, color=MUTED)
    return fig


def wrapped(fig, x, y, text, width=115, size=9.2, color=INK, weight="normal", linespacing=1.35):
    fig.text(x, y, textwrap.fill(text, width), fontsize=size, color=color, weight=weight,
             va="top", linespacing=linespacing)


def table(ax, headers, rows, widths=None, font=8.1, scale_y=1.35):
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                   loc="upper left", cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(LINE)
        cell.set_linewidth(0.45)
        cell.PAD = 0.08
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor(STRIPE)
        else:
            cell.set_facecolor("white")
    tbl.scale(1, scale_y)
    return tbl


def plot_panels(fig, bounds, late=False):
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.96, bottom=bounds[0], top=bounds[1], wspace=0.33)
    metrics = [
        ("fidelity", "Fidelity", "Factor fidelity (%)"),
        ("measurement_loss", "Sampled measurement loss", "Least-squares loss"),
        ("tp_violation", "TP residual (log scale)", "Frobenius residual"),
    ]
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, (key, title, ylabel) in zip(axes, metrics):
        for trace, color, label in ((old, BLUE, "Original winning run"), (new, ORANGE, "Exact continuation")):
            rows = [row for row in trace if (row["step"] >= 100000 or not late)]
            x = [row["step"] / 1000 for row in rows]
            y = [row[key] * (100 if key == "fidelity" else 1) for row in rows]
            ax.plot(x, y, color=color, lw=2.1, ls="-", label=label)
            ax.plot(x[-1], y[-1], "o", color=color, ms=3.3)
        ax.set_title(title, fontsize=11.3, weight="bold", pad=9)
        ax.set_xlabel("Completed iterations (thousands)", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlim(100 if late else 0, 202)
        if key == "fidelity":
            ax.axhline(99, color=MUTED, lw=0.9, ls=":")
            ax.set_ylim((96.5, 100.25) if late else (0, 103))
        elif key == "measurement_loss":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            ax.yaxis.set_major_locator(MaxNLocator(5))
        else:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
    handles = [Line2D([0], [0], color=BLUE, lw=2.3, label="Original winning run: 0–148k"),
               Line2D([0], [0], color=ORANGE, lw=2.3, label="Exact continuation: 148k–200k")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, bounds[1] + 0.035),
               ncol=2, frameon=False, fontsize=9)


with PdfPages(OUT) as pdf:
    # Page 1
    fig = canvas(1, "Eight qubits: exact continuation to 200,000 steps",
                 "Usual noise σ=0.05 | batch 65,536 | requested exponents 3/4, 1/4, 1/2")
    fig.text(0.055, 0.83,
             "The trajectory reaches 99% at saved step 174,000 and finishes at 99.13929%.",
             fontsize=13, weight="bold", color=BLUE)
    ax = fig.add_axes([0.055, 0.64, 0.89, 0.145])
    table(ax, ["Checkpoint", "Fidelity", "Sampled loss", "TP residual"], [
        ["Parent: 148,000", f"{100*p0['fidelity']:.5f}%", f"{p0['measurement_loss']:.9g}", f"{p0['tp_violation']:.7g}"],
        ["Continuation: 200,000", f"{100*p1['fidelity']:.5f}%", f"{p1['measurement_loss']:.9g}", f"{p1['tp_violation']:.7g}"],
    ], [0.31, 0.20, 0.25, 0.24], font=8.8)
    wrapped(fig, 0.055, 0.59,
            "Answer: more iterations alone were sufficient for this requested-exponent trajectory to cross 99%. Every saved checkpoint from 191,000 through 200,000 remained above 99%, so the endpoint is not a lone saved-threshold excursion.", 120, 10)
    wrapped(fig, 0.055, 0.47,
            f"From 148,000 to 200,000 steps, fidelity improves by {100*(p1['fidelity']-p0['fidelity']):.4f} percentage points. The fixed-sample loss changes from {p0['measurement_loss']:.9g} to {p1['measurement_loss']:.9g}; the complete-factor TP residual improves from {p0['tp_violation']:.7g} to {p1['tp_violation']:.7g}. TP remains separate: fidelity is not a physical-feasibility certificate.", 120, 9.2)
    fig.text(0.055, 0.345, "Fixed scientific scope", fontsize=12, weight="bold", color=BLUE)
    wrapped(fig, 0.055, 0.315,
            "One exact restart, not a new sweep: the parent factor, momentum estimate, PCG64 sampling state, global schedule index, dataset, frozen source snapshot and all optimizer settings are preserved. The segment adds 52,000 updates and 3,407,872,000 optimizer-row draws, ending at 13,107,200,000 total draws. No fidelity early stopping was used.", 120, 9.2)
    wrapped(fig, 0.055, 0.19,
            "This result concerns one Haar-unitary channel, one noisy dataset and one seed. It does not establish a success probability, uncertainty interval or universal optimum.", 120, 8.4, MUTED)
    pdf.savefig(fig); plt.close(fig)

    # Page 2
    fig = canvas(2, "Combined saved trajectory",
                 "Solid colored trajectories; saved evidence at 1,000-step cadence; no smoothing or fitted curves")
    plot_panels(fig, (0.28, 0.79), late=False)
    wrapped(fig, 0.065, 0.22,
            "Fidelity is |c†u|²/(||c||²||u||²), computed from complete factors and invariant to nonzero phase/scale. Loss uses the fixed 512-row metric sample with seed 12345. TP is ||T(uu†)−I||F from the complete factor and uses a logarithmic axis. The dotted line is the 99% reference.", 132, 8.5, MUTED)
    pdf.savefig(fig); plt.close(fig)

    # Page 3
    fig = canvas(3, "Late trajectory and threshold persistence",
                 "Zoomed to 100,000–200,000 steps; scientific traces remain unsmoothed")
    plot_panels(fig, (0.47, 0.79), late=True)
    ax = fig.add_axes([0.055, 0.10, 0.61, 0.31])
    final_rows = summary["final_checkpoints_191k_200k"]
    table(ax, ["Step", "Fidelity", "Sampled loss", "TP residual"], [
        [f"{row['step']:,}", f"{100*row['fidelity']:.5f}%", f"{row['measurement_loss']:.9g}", f"{row['tp_violation']:.7g}"]
        for row in final_rows
    ], [0.19, 0.22, 0.31, 0.28], font=7.0, scale_y=1.05)
    fig.text(0.70, 0.39, "Observed threshold behavior", fontsize=11.5, weight="bold", color=BLUE)
    wrapped(fig, 0.70, 0.35,
            "First saved F ≥ 99%: 174,000 steps.\n\nAll saved checkpoints 191,000–200,000 ≥ 99%: yes.\n\nThe prior descriptive projection suggested 170–175k; the observed saved crossing is 174k. This does not make the earlier fit a confidence interval.\n\nTP improves from 0.1976805 at restart to 0.1706184 at the endpoint.",
            48, 8.6)
    pdf.savefig(fig); plt.close(fig)

    # Page 4
    fig = canvas(4, "Exact schedules, budgets, and runtime definitions",
                 "Global schedule index is preserved across the restart")
    ax = fig.add_axes([0.055, 0.56, 0.89, 0.27])
    table(ax, ["Quantity", "Exact value", "Meaning"], [
        ["Step γₖ", "min(1, 0.5/(k+10)^(3/4))", "Frank-Wolfe atom weight"],
        ["Smoothing βₖ", "10⁷/(k+1)^(1/4); no cap", "TP penalty denominator"],
        ["Momentum ρₖ", "min(1, 0.6/(k+4)^(1/2))", "New batch-gradient weight"],
        ["Steps", "148,000 → 200,000", "52,000 additional updates"],
        ["Batch / rank / τ", "65,536 / 1 / 20", "64-bit product-state backend"],
        ["Metric", "512 rows every 1,000 steps", "Fixed seed 12345"],
    ], [0.20, 0.34, 0.46], font=8.0, scale_y=1.16)
    ax = fig.add_axes([0.055, 0.19, 0.89, 0.29])
    table(ax, ["Timer", "Measured", "Definition"], [
        ["Slurm elapsed", "12 min 17 s", "Whole allocation; COMPLETED, exit 0:0"],
        ["Runner", f"{meta['wall_seconds']:.3f} s ({meta['wall_seconds']/60:.2f} min)", "After setup; includes compilation, preparation, transfers, metrics, restart writes"],
        ["Optimizer", f"{meta['optimizer_seconds']:.3f} s ({meta['optimizer_seconds']/60:.2f} min)", "GPU execution for 52,000 updates"],
        ["Host preparation", f"{meta['host_preparation_seconds']:.3f} s", "Background time; overlaps GPU work"],
        ["Metric evaluation", f"{meta['metric_seconds']:.3f} s", "Saved-checkpoint metrics"],
    ], [0.20, 0.25, 0.55], font=7.6, scale_y=1.15)
    wrapped(fig, 0.055, 0.13,
            "Phase timers overlap and must not be added. Virtual M=24⁸=110,075,314,176 rows is sampled with replacement and never materialized. The expected noisy-truth loss σ²/2=0.00125 is not a measured convergence floor.", 128, 8.2, MUTED)
    pdf.savefig(fig); plt.close(fig)

    # Page 5
    fig = canvas(5, "Independent verification, provenance, and limitations",
                 "Complete saved factors; frozen source_fast; byte-for-byte Elite backup")
    ax = fig.add_axes([0.055, 0.58, 0.89, 0.25])
    table(ax, ["Check", "Result"], [
        ["Scheduler", "COMPLETED on ruche-gpu15; exit 0:0"],
        ["Restart continuity", "Resumed factor hash and initial fidelity/loss/TP match parent endpoint"],
        ["Exact schedules", "All 52,000 applied scalar values match scales, offsets, exponents and caps"],
        ["Independent factor metrics", "Verified; primary ≥99% target met"],
        ["Persistence", "All saved checkpoints 191,000–200,000 ≥99%"],
        ["Elite backup", "All 11 copied files matched Ruche SHA-256 at collection"],
    ], [0.27, 0.73], font=7.8, scale_y=1.1)
    fig.text(0.055, 0.53, "Independent numerical agreement", fontsize=11.5, weight="bold", color=BLUE)
    wrapped(fig, 0.055, 0.50,
            f"Saved fidelity {verification['saved_fidelity']:.16g}; NumPy factor form {verification['fidelity_numpy']:.16g}; independent matrix form {verification['fidelity_matrix_numpy']:.16g}. Saved TP {verification['saved_tp_violation']:.16g}; independently computed TP {verification['tp_violation_numpy']:.16g}; truth TP residual {verification['truth_tp_violation']:.3g}.",
            125, 8.5)
    fig.text(0.055, 0.39, "Integrity note", fontsize=11.5, weight="bold", color=BLUE)
    wrapped(fig, 0.055, 0.36,
            "The wrapper’s internal manifest matches every scientific artifact. Its Slurm stdout entry records the empty-file hash because checksums were computed before Slurm appended the final COMPLETE line. The finalized stdout was separately matched between Ruche and Elite; preserved evidence was not rewritten.",
            125, 8.5)
    fig.text(0.055, 0.26, "Scope and limitations", fontsize=11.5, weight="bold", color=BLUE)
    wrapped(fig, 0.055, 0.23,
            "One channel, noisy dataset, initialization and optimizer sampling seed. The result answers the bounded continuation question for this trajectory only. Fidelity is normalized overlap, not f in f+g and not a TP certificate; f is measurement least squares and g is the TP-constraint indicator.",
            125, 8.5)
    fig.text(0.055, 0.125, "Frozen source", fontsize=7.5, weight="bold", color=MUTED)
    fig.text(0.16, 0.125, provenance["source"], fontsize=7.2, family="DejaVu Sans Mono", color=MUTED)
    fig.text(0.055, 0.098, "Final SHA-256", fontsize=7.5, weight="bold", color=MUTED)
    fig.text(0.16, 0.098, verification["result_sha256"], fontsize=7.2, family="DejaVu Sans Mono", color=MUTED)
    fig.text(0.055, 0.071, "Resume factor", fontsize=7.5, weight="bold", color=MUTED)
    fig.text(0.16, 0.071, summary["resumed_factor_sha256"], fontsize=7.2, family="DejaVu Sans Mono", color=MUTED)
    pdf.savefig(fig); plt.close(fig)

print(OUT)
