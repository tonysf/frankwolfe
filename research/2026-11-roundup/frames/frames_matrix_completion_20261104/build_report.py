"""Build the completed matrix-completion comparison and reuse the archived toy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
HIST = Path("/home/tony/frames_stochastic_rows_1024_20261014")
LABELS = ["sframes_tuned", "shcgm", "sframes_unit"]
NAMES = {"sframes_tuned": "SFRAMES schedule: tuned prefactors", "shcgm": "SHCGM schedule: published prescription",
         "sframes_unit": "SFRAMES schedule: unit step/momentum"}
COLORS = {"sframes_tuned": "#7b3294", "shcgm": "#0072b2", "sframes_unit": "#d55e00"}
BATCH_COLORS = {16: "#009e73", 64: "#7b3294", 256: "#e69f00"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex_escape(value):
    value = str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        value = value.replace(old, new)
    return value


def load_results():
    complete = read(ROOT / "completed.json")
    assert complete["status"] == "completed", "Campaign must finish before final reporting."
    config, selected, ref = [read(ROOT / f) for f in ["campaign_config.json", "selection.json", "reference.json"]]
    runs = []
    for p in sorted((ROOT / "runs/confirmation").glob("*/result.json")):
        r = read(p)
        r["result_path"] = str(p.relative_to(ROOT))
        runs.append(r)
    expected = len(config["confirmation_batches"]) * len(config["confirmation_seeds"]) * len(LABELS)
    assert len(runs) == expected == complete["confirmation_runs"]
    data_hash = digest(ROOT / "data.npz")
    assert ref["hashes"]["data"] == config["data_sha256"] == data_hash
    seen = set()
    for r in runs:
        c, t = r["config"], r["trace"]
        key = (c["label"], c["batch"], c["seed"])
        assert key not in seen
        seen.add(key)
        assert r["status"] == "completed" and r["completed_updates"] == config["iterations"]
        assert r["data_sha256"] == data_hash
        assert r["source_sha256"] == {k: config["source_sha256"][k] for k in ["core.py", "run.py"]}
        assert t[0]["iteration"] == 0 and t[-1]["iteration"] == config["iterations"]
        assert np.all(np.diff([z["iteration"] for z in t]) > 0)
        assert np.all(np.diff([z["optimizer_seconds"] for z in t]) > 0)
        for z in t:
            assert z["sampled_entries"] == z["iteration"] * c["batch"]
            assert z["feasible_nuclear_norm"] <= ref["tau"] + 1e-8
            assert z["feasible_max_abs"] <= 1 + 1e-12
            assert z["feasible_loss"] >= ref["certified_lower_bound"] - 1e-8
            assert z["mean_true_smoothed_gap"] is None if z["iteration"] == 0 else z["mean_true_smoothed_gap"] >= -1e-9
    for q in config["confirmation_batches"]:
        for seed in config["confirmation_seeds"]:
            group = [r for r in runs if r["config"]["batch"] == q and r["config"]["seed"] == seed]
            assert len(group) == 3 and len({r["sampled_stream_sha256"] for r in group}) == 1
    same = all(selected["selected"]["sframes_tuned"][key] == selected["selected"]["sframes_unit"][key]
               for key in ["gamma0", "rho0", "beta0"])
    if same:
        NAMES["sframes_tuned"] = "SFRAMES: tuned = unit prefactors"
        for q in config["confirmation_batches"]:
            for seed in config["confirmation_seeds"]:
                a = next(r for r in runs if r["config"]["label"] == "sframes_tuned" and r["config"]["batch"] == q and r["config"]["seed"] == seed)
                b = next(r for r in runs if r["config"]["label"] == "sframes_unit" and r["config"]["batch"] == q and r["config"]["seed"] == seed)
                assert [t["feasible_loss"] for t in a["trace"]] == [t["feasible_loss"] for t in b["trace"]]
    return config, selected, ref, runs, same


def subset(runs, label, batch=64):
    return sorted([r for r in runs if r["config"]["label"] == label and r["config"]["batch"] == batch],
                  key=lambda r: r["config"]["seed"])


def values(r, field):
    return np.array([p[field] for p in r["trace"]], dtype=float)


def draw(ax, rs, xfield, yfield, color, label, *, subtract=0., statistic="median", logx=False):
    if xfield == "optimizer_seconds":
        lo = max(values(r, xfield)[1] for r in rs)
        hi = min(values(r, xfield)[-1] for r in rs)
        assert hi > lo
        x = np.unique(np.concatenate(([lo, hi], *[values(r, xfield)[(values(r, xfield) >= lo) & (values(r, xfield) <= hi)] for r in rs])))
        ys = np.array([np.interp(x, values(r, xfield), values(r, yfield)) for r in rs]) - subtract
    else:
        x = values(rs[0], xfield)
        assert all(np.array_equal(values(r, xfield), x) for r in rs)
        ys = np.array([values(r, yfield) for r in rs]) - subtract
        keep = x > 0 if logx else np.ones(len(x), dtype=bool)
        keep &= np.isfinite(ys).all(axis=0)
        x, ys = x[keep], ys[:, keep]
    center = np.mean(ys, axis=0) if statistic == "mean" else np.median(ys, axis=0)
    ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=.16)
    ax.plot(x, center, color=color, lw=2., label=label)
    if logx:
        ax.set_xscale("log")
    ax.grid(color="#e5e7eb", lw=.65)
    ax.set_axisbelow(True)
    return x, ys


def save(fig, name, pdf):
    fig.savefig(OUT / f"{name}.png", dpi=175)
    fig.savefig(OUT / f"{name}.pdf")
    pdf.savefig(fig)
    plt.close(fig)


def main_plot(runs, ref, same, pdf):
    labels = LABELS[:2] if same else LABELS
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.5))
    fig.subplots_adjust(left=.085, right=.98, top=.80, bottom=.13, wspace=.25, hspace=.35)
    fig.suptitle("Matrix completion: schedule comparison", x=.085, y=.97, ha="left", fontsize=18, weight="bold")
    fig.text(.085, .925, "50 × 50 · FP64 · Batch 64 · Median and range of five fresh seeds · Current feasible outputs", fontsize=10.5)
    for label in labels:
        rs = subset(runs, label)
        draw(axes[0, 0], rs, "iteration", "feasible_loss", COLORS[label], NAMES[label])
        draw(axes[0, 1], rs, "iteration", "feasible_loss", COLORS[label], NAMES[label],
             subtract=ref["certified_lower_bound"], logx=True)
        draw(axes[1, 0], rs, "iteration", "box_violation_fro", COLORS[label], NAMES[label], logx=True)
        draw(axes[1, 1], rs, "optimizer_seconds", "feasible_loss", COLORS[label], NAMES[label],
             subtract=ref["certified_lower_bound"], logx=True)
    axes[0, 0].axhline(ref["certified_upper_bound"], color="#333333", ls=":", lw=1.2, label="Convex reference")
    axes[0, 0].set(xlabel="Completed updates", ylabel="Feasible training loss", title="Objective decrease")
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[0, 1].set(xlabel="Completed updates", ylabel="Feasible suboptimality upper bound", title="Error relative to the optimum", yscale="log")
    axes[1, 0].set(xlabel="Completed updates", ylabel="Raw box violation, Frobenius norm", title="Constraint violation", yscale="symlog")
    axes[1, 0].set_yscale("symlog", linthresh=1e-5)
    axes[1, 1].set(xlabel="Optimizer time per run (seconds)", ylabel="Feasible suboptimality upper bound", title="Recorded optimizer time", yscale="log")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(.078, .895), frameon=False, ncol=2, fontsize=9)
    fig.text(.085, .065, "Suboptimality uses feasible loss minus the certified lower bound; reference-bracket width is less than 3 × 10⁻¹⁰.", fontsize=9)
    fig.text(.085, .032, "Timing excludes diagnostics and I/O. Time bands interpolate only within shared observed support; four pinned CPU workers ran concurrently.", fontsize=8.8)
    save(fig, "comparison_batch64", pdf)


def batch_plot(runs, ref, same, pdf):
    labels = LABELS[:2] if same else LABELS
    fig, axes = plt.subplots(1, len(labels), figsize=(12, 5.7), squeeze=False, sharey=True)
    fig.subplots_adjust(left=.08, right=.98, top=.73, bottom=.24, wspace=.24)
    fig.suptitle("Matrix completion: batch size and sampled observations", x=.08, y=.965,
                 ha="left", fontsize=17, weight="bold")
    fig.text(.08, .91, "Each point counts actual sampled training entries, including repeats · Five fresh seeds per batch", fontsize=10.5)
    for ax, label in zip(axes.flat, labels):
        for q in [16, 64, 256]:
            draw(ax, subset(runs, label, q), "sampled_entries", "feasible_loss", BATCH_COLORS[q], f"Batch {q}",
                 subtract=ref["certified_lower_bound"], logx=True)
        ax.set(xlabel="Sampled entries", yscale="log", title={"sframes_tuned":"SFRAMES schedule: tuned prefactors", "shcgm":"SHCGM schedule: published prescription", "sframes_unit":"SFRAMES: unit step/momentum"}[label])
    axes[0, 0].set_ylabel("Feasible suboptimality upper bound")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(.073, .865), frameon=False, ncol=3)
    fig.text(.08, .14, "All runs end at 20,000 updates: batch 16 = 320,000 draws; batch 64 = 1,280,000; batch 256 = 5,120,000.", fontsize=9.6)
    fig.text(.08, .095, "Prefactors were selected at batch 64 and reused unchanged at 16 and 256. Curves stop at their actual observation budgets.", fontsize=9.6)
    fig.text(.08, .05, "Shaded regions are observed seed ranges, not confidence intervals. Loss is evaluated after radial feasibility repair, used only for diagnostics.", fontsize=9.3)
    save(fig, "batch_sample_comparison", pdf)


def historical_summary():
    meta = read(HIST / "metadata.json")
    with (HIST / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    with np.load(HIST / "results.npz") as d:
        x = d["final_x"].copy()
    repaired = np.maximum(x, 0.)
    a = np.full(1024, 1/128)
    a[:64] = 1/32
    a[64:128] = -3/128
    u = np.zeros(1024)
    u[:64:2], u[1:64:2] = 1/8, -1/8
    losses = .5 * np.square(repaired - a).sum(axis=1) - 5/8 * np.square(repaired @ u)
    assert np.abs(repaired).sum(axis=1).max() <= 1 + 1e-10
    summary = dict(dimension=1024, steps=meta["steps"], seeds=len(meta["seeds"]), batch=meta["batch_size"],
        optimum=13/256, feasible_final_loss_median=float(np.median(losses)),
        feasible_final_loss_range=[float(losses.min()), float(losses.max())],
        current_feasible_gap_median=float(np.median(losses)-13/256),
        cumulative_gap_1024=float(next(r for r in rows if int(r["updates"]) == 1024)["mean_cumulative_true_gap"]),
        cumulative_gap_final=float(rows[-1]["mean_cumulative_true_gap"]),
        source_directory=str(HIST), source_hashes={f:digest(HIST/f) for f in ["metadata.json","summary.csv","mathematics.json","results.npz"]})
    original = HIST / "gaps_feasibility_zoom.png"
    if original.exists():
        shutil.copy2(original, OUT / "historical_row_sampling_original.png")
        summary["original_plot_sha256"] = digest(original)
    return rows, summary


def theorem_plot(runs, history_rows, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6))
    fig.subplots_adjust(left=.085, right=.98, top=.73, bottom=.23, wspace=.27)
    fig.suptitle("Theorem statistic: true averaged smoothed FW gap", x=.085, y=.965,
                 ha="left", fontsize=17, weight="bold")
    fig.text(.085, .91, "Every pre-update iterate contributes its own exact gap · Expectations estimated across independent seeds", fontsize=10.3)
    for q in [16, 64, 256]:
        draw(axes[0], subset(runs, "sframes_unit", q), "iteration", "mean_true_smoothed_gap", BATCH_COLORS[q],
             f"Batch {q}: mean and range of 5 seeds", statistic="mean", logx=True)
    axes[0].set(xlabel="Completed updates", ylabel="Mean cumulative true smoothed gap", yscale="log",
                title="New matrix-completion problem")
    x = np.array([int(r["updates"]) for r in history_rows])
    y = np.array([float(r["mean_cumulative_true_gap"]) for r in history_rows])
    se = np.array([float(r["seed_standard_error_true_gap"]) for r in history_rows])
    axes[1].fill_between(x, y-se, y+se, color="#444444", alpha=.2)
    axes[1].plot(x, y, color="#444444", lw=2, marker="o", ms=3,
                 label="Archived row sampling: mean ± 1 SE, 100 seeds")
    axes[1].set(xlabel="Completed updates", ylabel="Mean cumulative true smoothed gap", yscale="log", xscale="log",
                title="Existing nonconvex row-sampling example")
    axes[1].grid(color="#e5e7eb", lw=.65)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels(); handles += h; labels += l
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.078, .87), ncol=2, frameon=False, fontsize=9)
    fig.text(.085, .135, "Left: SFRAMES with γ₀=ρ₀=1 and selected β₀; mean of per-run arithmetic averages over all k=0,…,K−1.", fontsize=9.4)
    fig.text(.085, .09, "Right: archived 1,024-variable construction, batch1. Its global optimum and full gradient are available analytically.", fontsize=9.4)
    fig.text(.085, .045, "These are different problems and are not a solver ranking. Declining empirical gaps illustrate the theorem; they do not prove its asymptotic rate.", fontsize=9.1)
    save(fig, "theorem_statistic", pdf)


def endpoint_summary(runs, ref):
    out = []
    for label in LABELS:
        for q in [16,64,256]:
            rs = subset(runs,label,q)
            loss=np.array([r["trace"][-1]["feasible_loss"] for r in rs])
            raw=np.array([r["trace"][-1]["raw_loss"] for r in rs])
            opt=np.array([r["optimizer_seconds"] for r in rs])
            diag=np.array([r["diagnostic_seconds"] for r in rs])
            box=np.array([r["trace"][-1]["box_violation_fro"] for r in rs])
            out.append(dict(label=label,batch=q,seeds=len(rs),loss_median=float(np.median(loss)),
                loss_min=float(loss.min()),loss_max=float(loss.max()),
                suboptimality_lower=float(np.median(loss)-ref["certified_upper_bound"]),
                suboptimality_upper=float(np.median(loss)-ref["certified_lower_bound"]),
                raw_loss_median=float(np.median(raw)),box_violation_median=float(np.median(box)),
                optimizer_seconds_median=float(np.median(opt)),diagnostic_seconds_median=float(np.median(diag)),
                initial_gap_closed_percent=float(100*(ref["initial_loss"]-np.median(loss))/(ref["initial_loss"]-ref["certified_upper_bound"]))))
    return out


def make_tex(config, selection, ref, endpoints, hist, same):
    active = LABELS[:2] if same else LABELS
    short={"sframes_tuned":"SFRAMES tuned", "sframes_unit":"SFRAMES unit", "shcgm":"SHCGM"}
    param_rows=[]
    for label in active:
        c=selection["selected"][label]
        gamma, rho = (f"{c['gamma0']:g}", f"{c['rho0']:g}") if label != "shcgm" else ("fixed", "fixed")
        param_rows.append(f"{short[label]} & {gamma} & {rho} & {c['beta0']:g} \\\\")
    endpoint_rows=[]
    for e in endpoints:
        if e["batch"]==64 and e["label"] in active:
            endpoint_rows.append(f"{short[e['label']]} & {e['loss_median']:.6f} & [{e['loss_min']:.6f}, {e['loss_max']:.6f}] & {e['initial_gap_closed_percent']:.1f}\\% \\\\")
    batch_rows=[]
    for e in endpoints:
        if e["label"] in active:
            batch_rows.append(f"{short[e['label']]} & {e['batch']} & {e['suboptimality_upper']:.4g} & {e['box_violation_median']:.3g} & {e['optimizer_seconds_median']:.2f} & {e['diagnostic_seconds_median']:.2f} \\\\")
    dup = ("The selected tuned and unit-prefactor SFRAMES configurations coincide. Their repeated executions use the same five seeds and produce identical trajectories; they are not additional independent replicates. Performance plots show this configuration once."
           if same else "The unit-prefactor SFRAMES control fixes both step and momentum prefactors to one; only its smoothing prefactor is selected. Its five fresh seeds are shared with the other configurations.")
    best=min([e for e in endpoints if e['batch']==64 and e['label'] in active],key=lambda e:e['loss_median'])
    text=r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\newcommand{\pageheading}[1]{{\Large\bfseries #1}\par\vspace{3mm}}
\begin{document}
\pageheading{Smoothing Frank--Wolfe: convex schedule comparison}
\textbf{4 November 2026.} Small synthetic CPU benchmark with a certified convex reference.

\textbf{Scope.} This is a schedule and prefactor comparison within one shared stochastic momentum--Moreau smoothing--Frank--Wolfe update. The problem is convex. It supplies an implementation check and a schedule ablation; it does not demonstrate an algorithmic advantage or the contribution of the nonconvex analysis.

Let $\Omega$ contain $N=1000$ observed entries of a $50\times50$ matrix $Y$. We solve
\[
\min_{X\in C} f(X)+g(TX),\qquad
f(X)=\frac1{2N}\sum_{(i,j)\in\Omega}(X_{ij}-Y_{ij})^2,
\]
\[
C=\{X:\|X\|_*\le40\},\qquad g=\iota_D,\quad
D=[-1,1]^{50\times50},\quad T=I.
\]
The nuclear norm is the sum of singular values. The linear minimization oracle is $-40uv^\top$, where $(u,v)$ is a leading singular-vector pair of the search direction. This experiment uses a full SVD to evaluate that oracle accurately. The projection onto $D$ clips every entry to $[-1,1]$.

\textbf{Stochastic oracle.} Draw $(i,j)$ uniformly from $\Omega$ and return
\[
Q(X;i,j)=(X_{ij}-Y_{ij})e_ie_j^\top,\qquad
\mathbb E[Q(X;i,j)\mid X]=\nabla f(X).
\]
A batch averages $q$ independent draws with replacement. Repeated indices are accumulated. The finite-sum objective is globally smooth; $C$ is compact and convex; $D$ is closed and convex; zero lies in the interior of both sets. Component gradients have uniformly bounded variance on $C$. Thus the stochastic oracle and geometry meet the stated problem assumptions. Convexity makes this a special case of the manuscript's nonconvex setting.

\textbf{Data and reference.} A fixed seed, 20261104, generates a rank-five signal scaled to maximum absolute entry 0.8, plus Gaussian noise with standard deviation 0.6. Exactly 40\% of entries are observed; every row and column has observations. The other 1500 entries are excluded from optimization. Some noisy observations exceed the fitted-value box. Radius 20 made the box inactive at the independently solved optimum; the documented pre-campaign design check selected radius 40, the first proposed candidate with both constraints active. No optimizer comparison informed that choice.

The independently evaluated primal/dual reference bracket is
\[
@LOW@\ \le f^\star\le\ @UP@,
\qquad U-L=@WIDTH@.
\]
The feasible primal point supplies $U$. With CVXPY box multiplier $Z$, the lower bound is computed independently as
\[
L=f(X_r)-\langle\nabla f(X_r),X_r\rangle
-40\|\nabla f(X_r)+Z\|_{\mathrm{op}}-\|Z\|_1.
\]
This is a floating-point convex certificate, not interval arithmetic. Both the nuclear bound and two box entries are active. Initial loss at zero is @INITIAL@.

\textbf{Meaning of the plotted loss.} Each raw iterate stays in $C$. For reporting only, form $\widehat X=X/\max(1,\max_{ij}|X_{ij}|)$, which also lies in $D$. This radial repair never feeds back into the optimizer. The true feasible objective error lies in $[f(\widehat X)-U,f(\widehat X)-L]$; plots use its upper endpoint. Raw loss and box violation are saved separately.

\newpage
\pageheading{Fresh-seed performance under different schedules}
For zero-based update $k\ge0$, SFRAMES uses
\[
\gamma_k=\gamma_0(k+1)^{-3/4},\quad
\rho_k=\rho_0(k+1)^{-1/2},\quad
\beta_k=\beta_0(k+1)^{-1/4}.
\]
Its momentum starts at the first sampled smooth gradient. SHCGM uses the published Algorithm 1 schedules, written with the same zero-based index:
\[
\gamma_k=\frac9{k+9},\quad \rho_k=\frac4{(k+8)^{2/3}},\quad
\beta_k=\frac{\beta_0}{\sqrt{k+9}}.
\]
Both use the same momentum, Moreau-penalty gradient, LMO and convex-combination update. SHCGM starts from zero momentum with first weight one, so its first estimate is also the first sampled gradient. The schedules and prefactors distinguish the tested configurations. No line search is used.

\begin{center}\small
\begin{tabular}{lrrr}\toprule
Selected configuration & $\gamma_0$ & $\rho_0$ & $\beta_0$\\\midrule
@PARAMS@
\bottomrule\end{tabular}
\end{center}
\includegraphics[width=\linewidth]{comparison_batch64.pdf}
\vspace{-3mm}
\begin{center}\small
\begin{tabular}{lrrr}\toprule
Batch64; five fresh seeds & Median feasible loss & Seed range & Initial gap closed\\\midrule
@ENDPOINTS@
\bottomrule\end{tabular}
\end{center}
@DUPLICATES@

\newpage
\pageheading{Sample budgets, parameter selection, and timing}
Each method receives 16 candidate configurations, two development seeds (101,102), batch 64, and 20000 updates per run. SFRAMES searches $\gamma_0,\rho_0\in\{0.3,1\}$ and $\beta_0/N\in\{0.1,1,10,100\}$. SHCGM keeps its published step/momentum schedules and searches 16 logarithmically spaced values of $\beta_0/N$ from0.01 to1000. Each method selects the lowest median final feasible training loss. The unit-prefactor SFRAMES control selects among its four smoothing values. The selected SFRAMES step, momentum and smoothing prefactors lie at the lower tested grid boundaries; this is the best tested configuration in a fixed grid, not evidence that tuning is globally optimal. The grid was not expanded after observing confirmation results.

Five fresh seeds (201--205) confirm each selection. The same selected parameters are reused at batches 16, 64, 256; these batches are not retuned. For each seed and batch, all methods receive identical random index streams. Their sampled gradients differ because their iterates differ. Each run uses FP64, starts from zero, and completes 20000 updates.

\includegraphics[width=\linewidth]{batch_sample_comparison.pdf}

\begin{center}\small
\begin{tabular}{lrrrrr}\toprule
Method & Batch & Gap upper bound & Box violation & Optimizer(s) & Diagnostics(s)\\\midrule
@BATCH_ROWS@
\bottomrule\end{tabular}
\end{center}
Table entries are medians across five fresh seeds at the final update. The gap is evaluated at the feasible diagnostic output; box violation is evaluated at the raw iterate.

\textbf{Timing.} Four processes run concurrently on four pinned physical cores of an AMD Ryzen 5 PRO 3400G, one numerical-library thread per process. Optimizer timing includes sampling, momentum, penalty gradient, the full-SVD LMO, and the update. Exact-gap diagnostics, objective/feasibility evaluations, feasible repair, and file output are excluded. Diagnostic time is recorded separately. Time curves interpolate only within the common observed time support across seeds and do not extrapolate endpoints. Concurrent CPU measurements do not provide uncontended latency estimates, and tuning cost is excluded from per-run curves.

\textbf{Interpretation.} At batch 64, the @BEST@ schedule has the lowest median final feasible loss in this finite grid. Both named prescriptions instantiate the same update family on this convex problem. The observed difference concerns schedules and constants, not a distinct algorithmic mechanism. SHCGM uses convexity in its schedule analysis; these finite-horizon results do not overturn that theory. SFRAMES also received more kinds of tunable constants (step, momentum and smoothing, versus smoothing alone), despite equal candidate counts. This experiment cannot support a claim of algorithmic superiority or demonstrate the value of the nonconvex extension.

\newpage
\pageheading{The theorem statistic and the existing nonconvex example}
For the new experiment, let $H_k=\nabla f(X_k)+(X_k-\operatorname{clip}(X_k))/\beta_k$. The exact smoothed Frank--Wolfe gap is
\[
G_k=\langle H_k,X_k\rangle+40\|H_k\|_{\mathrm{op}}.
\]
The reported theorem statistic is $K^{-1}\sum_{k=0}^{K-1}G_k$, computed at \emph{every} pre-update iterate, then averaged across seeds. It is neither a sparse-checkpoint average nor a gap computed from the noisy momentum estimator. The SFRAMES unit-prefactor control follows the displayed manuscript prescription; its smoothing prefactor remains positive and tuned. The expected averaged gap has the stated $O(K^{-1/4})$ guarantee. These finite runs illustrate behavior and cannot establish an asymptotic rate.

\includegraphics[width=\linewidth]{theorem_statistic.pdf}

\textbf{Reused experiment, not rerun.} The archived 1024-variable problem is
\[
\min_{\|x\|_1\le1}\ \|Ax-b\|^2-\frac58(u^\top x)^2+\iota_{\mathbb R_+^{1024}}(x),\qquad A^\top A=I/2.
\]
It uses genuine independent uniform row sampling, batch 1, 100 seeds, and 131072 updates. Its true averaged smoothed gap decreases from @HIST_GAP_START@ at 1024 updates to @HIST_GAP_END@ at 131072. Its global optimum is $13/256=0.05078125$, while the final feasible loss has median @HIST_LOSS@, leaving median objective gap @HIST_ERROR@. Thus the stationarity trend does not imply near-optimal objective values. The orthogonal synthetic design also makes the full gradient cheap; it is a theorem illustration, not practical efficiency evidence. Original artifacts are preserved; the original gap plot is copied beside this report.

\textbf{Sources and reproducibility.} SHCGM: Locatello, Yurtsever, Fercoq and Cevher, \href{https://proceedings.neurips.cc/paper/2019/file/bf40d1cbb2ba9fdad19821fc140fa50c-Paper.pdf}{Stochastic Frank--Wolfe for Composite Convex Minimization}, NeurIPS 2019, Algorithm 1 and Section 4.3. The present normalized synthetic problem uses that algorithm, but does not claim the published MovieLens objective values. SFRAMES: supplied \texttt{Nonsmooth\_FW-7.pdf}, Corollary 2.10 and Proposition 2.11. The implementation follows the positive Moreau-penalty gradient from its definition; the draft's opposite sign in Algorithm 1 is recorded in the audit.

Run configurations, selected parameters, data/source checksums, reference duals, complete diagnostic curves, optimizer and diagnostic timing, and validation receipts accompany this report. See \texttt{summary.json}, \texttt{curves.csv}, \texttt{endpoints.csv}, \texttt{campaign\_config.json}, \texttt{selection.json}, and \texttt{provenance/} in the experiment directory. Historical source: \texttt{/home/tony/frames\_stochastic\_rows\_1024\_20261014}.
\end{document}
"""
    replacements={"LOW":f"{ref['certified_lower_bound']:.12f}","UP":f"{ref['certified_upper_bound']:.12f}",
        "WIDTH":f"{ref['certificate_gap']:.3g}".replace("e-",r"\times10^{-") + "}" if "e-" in f"{ref['certificate_gap']:.3g}" else f"{ref['certificate_gap']:.3g}",
        "INITIAL":f"{ref['initial_loss']:.9f}","PARAMS":"\n".join(param_rows),"ENDPOINTS":"\n".join(endpoint_rows),
        "DUPLICATES":dup,"BATCH_ROWS":"\n".join(batch_rows),"BEST":short[best['label']],
        "HIST_GAP_START":f"{hist['cumulative_gap_1024']:.6f}","HIST_GAP_END":f"{hist['cumulative_gap_final']:.6f}",
        "HIST_LOSS":f"{hist['feasible_final_loss_median']:.8f}","HIST_ERROR":f"{hist['current_feasible_gap_median']:.6f}"}
    for key,value in replacements.items():
        text=text.replace("@"+key+"@",value)
    assert "@" not in text
    (OUT/"report.tex").write_text(text)


def export(config, selected, ref, runs, same, hist):
    endpoints=endpoint_summary(runs,ref)
    with (OUT/"endpoints.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(endpoints[0]));w.writeheader();w.writerows(endpoints)
    fields=["label","batch","seed","iteration","sampled_entries","feasible_loss","raw_loss","box_violation_fro", "mean_true_smoothed_gap","true_smoothed_gap","optimizer_seconds","diagnostic_seconds","suboptimality_lower","suboptimality_upper"]
    with (OUT/"curves.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in runs:
            for t in r["trace"]:
                row={k:r["config"][k] for k in ["label","batch","seed"]}
                row.update({k:t[k] for k in fields if k in t})
                row.update(suboptimality_lower=t["feasible_loss"]-ref["certified_upper_bound"],
                           suboptimality_upper=t["feasible_loss"]-ref["certified_lower_bound"])
                w.writerow(row)
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),status="complete_validated",
        confirmation_runs=len(runs),selection=selected,reference=ref,endpoints=endpoints,
        interpretation="Convex schedule/prefactor ablation within one shared momentum-Moreau-LMO update family; not evidence of distinct algorithmic superiority or the nonconvex contribution",
        tuned_and_unit_sframes_same_configuration=same,
        duplicate_interpretation="When selected prefactors coincide, matched-seed executions are duplicate trajectories, not additional independent replicates.",
        historical_row_sampling=hist,
        plotting=dict(main="median and min/max of five fresh seeds",theorem="mean and min/max of five fresh seeds, all-iterate prefix statistic",
                      time="linear interpolation only within shared observed optimizer-time support; no diagnostic or I/O time",sample_work="batch times completed accepted updates"),
        inputs_sha256={p.name:digest(p) for p in [ROOT/"campaign_config.json",ROOT/"selection.json",ROOT/"completed.json",ROOT/"reference.json",ROOT/"data.npz"]},
        result_files_sha256={r["result_path"]:digest(ROOT/r["result_path"]) for r in runs},
        builder_sha256=digest(Path(__file__)))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    make_tex(config,selected,ref,endpoints,hist,same)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--no-latex",action="store_true")
    args=parser.parse_args()
    config,selected,ref,runs,same=load_results()
    OUT.mkdir(exist_ok=True)
    rows,hist=historical_summary()
    with PdfPages(OUT/"plots.pdf") as pdf:
        main_plot(runs,ref,same,pdf)
        batch_plot(runs,ref,same,pdf)
        theorem_plot(runs,rows,pdf)
    summary=export(config,selected,ref,runs,same,hist)
    if not args.no_latex:
        result=subprocess.run(["latexmk","-xelatex","-interaction=nonstopmode","-halt-on-error","report.tex"],cwd=OUT,text=True,capture_output=True)
        (OUT/"build.log").write_text(result.stdout+"\n"+result.stderr)
        if result.returncode:
            raise RuntimeError(f"LaTeX failed; see {OUT/'build.log'}")
        assert (OUT/"report.pdf").exists()
    print(json.dumps(dict(report=str(OUT/"report.pdf"),plots=str(OUT/"plots.pdf"),
        confirmation_runs=summary["confirmation_runs"],same_sframes_configuration=same),indent=2))


if __name__=="__main__":
    main()
