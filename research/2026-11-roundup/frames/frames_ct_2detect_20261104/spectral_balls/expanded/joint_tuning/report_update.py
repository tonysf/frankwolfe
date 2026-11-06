"""Publish the paper section after the joint search and dense confirmations finish."""

import fcntl
import hashlib
import importlib.util
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / "reports" / "paper_section"
BODY = "ct_experiments.tex"
STEM = "sframes_ct_section"
METHODS = {
    "sframes": "Minibatch SFRAMES", "sframes_full": "Full-batch SFRAMES",
    "spring_sarah": "SPRING--SARAH", "prox_sarah": "ProxSARAH", "palm": "PALM",
}
SFRAMES = ("sframes", "sframes_full")
METRICS = (
    ("raw_loss", "Raw", ".5f"), ("feasible_loss", "Repaired", ".5f"),
    ("validation_loss", "Held-out", ".5f"),
    ("reference_relative_error", "Ref. error", ".4f"), ("d_distance", r"$d_D$", ".4g"),
)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integer(value):
    return f"{int(value):,}".replace(",", "{,}")


def parameter(value):
    return f"{value:.12g}"


def numeric(value, style, bold=False):
    token = format(value, style)
    if "e" in token:
        base, exponent = token.split("e")
        token = rf"{base}\times10^{{{int(exponent)}}}"
    if bold:
        token = rf"\boldsymbol{{{token}}}"
    return f"${token}$"


def table(columns, headings, rows, caption, label):
    return "\n".join((
        r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
        rf"\label{{{label}}}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        " & ".join(headings) + r" \\", r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ))


def load_inputs(root):
    selection = read(root / "selection.json")
    if selection.get("status") != "selected":
        raise ValueError("The joint development search has not finalized its selection")
    summary = read(root / "reports" / "summary.json")
    design = read(root / "manifest.json")["design"]
    analysis = load_module("joint_report_analysis", root / "analyze.py")
    runs = analysis.collect(root)
    seeds = sorted(design["evaluation_seeds"])
    deterministic = set(design["deterministic_variants"])
    expected = {
        (analysis.slice_id(sid), seed, n, variant)
        for sid, seed, n, variant in product(
            design["evaluation_slices"], seeds, design["evaluation_training_views"], METHODS)
        if variant not in deterministic or seed == seeds[0]
    }
    if len(expected) != 132 or set(runs) != expected:
        raise ValueError("The paper requires the complete 132-arm evaluation cohort")
    budget = design["evaluation_view_budget"]
    for key, run in runs.items():
        trace = run["trace"]
        if run["status"] != "completed" or not trace:
            raise ValueError(f"Evaluation is incomplete for {run['config']['run_id']}")
        if [row["iteration"] for row in trace] != list(range(trace[-1]["iteration"] + 1)):
            raise ValueError(f"Missing per-iteration diagnostics for {run['config']['run_id']}")
        if trace[-1]["forward_views"] < budget:
            raise ValueError("An evaluation run stopped below its data budget")
        if key[3] in SFRAMES:
            chosen = selection["selected"][key[3]]
            optimizer = run["meta"]["optimizer"]
            expected_beta = chosen["beta_scale"] / max(run["meta"]["curvature"])
            if (optimizer["gamma0"] != chosen["gamma0"]
                    or not math.isclose(optimizer["mu0"], expected_beta, rel_tol=1e-12)
                    or run["config"].get("beta_scale", 1.0) != chosen["beta_scale"]):
                raise ValueError("A SFRAMES confirmation uses a different selected pair")
    fresh = analysis.analyze(root, design, budget)
    if fresh["coverage"]["physical_arm_status_counts"] != {"completed": 132}:
        raise ValueError("The fresh analysis does not cover 132 completed runs")
    if fresh["cohorts"] != summary["cohorts"] or summary["selection"] != selection:
        raise ValueError("The saved report summary is stale")
    baseline = read(root / "baseline_selection.json")
    for key, run in runs.items():
        variant = key[3]
        if variant not in SFRAMES:
            trials = design["baseline_development_trials"][variant]
            chosen = trials[baseline["selected"][variant]]
            if any(run["config"].get(field) != value for field, value in chosen.items()):
                raise ValueError("A baseline uses different frozen development settings")
    return selection, summary, runs, baseline


def setup(selection):
    chosen = selection["selected"]
    mini = chosen["sframes"]
    full = chosen["sframes_full"]
    choices = (
        rf"A joint development search selects \((c_\beta,\gamma_0)="
        rf"({parameter(mini['beta_scale'])},{parameter(mini['gamma0'])})\) for minibatch SFRAMES "
        rf"and \(({parameter(full['beta_scale'])},{parameter(full['gamma0'])})\) "
        r"for full-batch SFRAMES."
    )
    if (mini["beta_scale"], mini["gamma0"]) == (full["beta_scale"], full["gamma0"]):
        choices = (rf"The joint development searches select \((c_\beta,\gamma_0)="
                   rf"({parameter(mini['beta_scale'])},{parameter(mini['gamma0'])})\) "
                   r"for both SFRAMES variants.")
    return r"""\paragraph{Initialization and SFRAMES parameters.}
All methods start from common nonnegative rank-\(64\) factors \(U_0,V_0\),
obtained by an NNDSVD-style factorization of a sensitivity-normalized
backprojection of \(120\) training views, followed by a small positive
perturbation and a scale fit. Set \(R_U=2\|U_0\|_{\rm op}\) and
\(R_V=2\|V_0\|_{\rm op}\). Ten power iterations on the stacked calibration
operator \(P_J\) estimate \(\widehat\lambda\approx\|P_J\|_{\rm op}^2/(120d)\).
We scale the smoothing prefactor by
\[
 L_U=\widehat\lambda\|V_0\|_{\rm op}^2,\qquad
 L_V=\widehat\lambda\|U_0\|_{\rm op}^2,\qquad
 \beta_0=c_\beta/\max(L_U,L_V).
\]
These initial block-curvature estimates provide a heuristic search scale.
""" + choices + r"""
The selection criterion is repaired training loss relative to its initial
value. Minibatch SFRAMES draws \(64\) training views uniformly without
replacement at each update, independently across updates. Full-batch SFRAMES
uses all \(N\) views. Both use
\[
 \gamma_k=\gamma_0(k+1)^{-3/4},\qquad
 \alpha_k=(k+1)^{-1/2},\qquad
 \beta_k=\beta_0(k+1)^{-1/4}.
\]

"""


def quality_table(summary):
    rows = []
    for n, cohort in summary["cohorts"].items():
        marginal = cohort["view_cap"]["matched_all_methods"]["median_of_slice_medians"]
        minima = {key: min(marginal[variant][key] for variant in METHODS)
                  for key, _, _ in METRICS}
        for variant, name in METHODS.items():
            values = marginal[variant]
            rows.append((integer(n), name, *(
                numeric(values[key], style, values[key] == minima[key])
                for key, _, style in METRICS)))
    return table(
        "llrrrrr", (r"$N$", "Method", *(name for _, name, _ in METRICS)), rows,
        r"Reconstruction quality at the saved within-budget iterates. Entries take medians "
        r"over seeds within each slice, then across six slices. Raw and repaired are training "
        r"losses. Held-out loss and relative reference error use repaired factors. Lower is "
        r"better. Bold entries give the best value within each training-view count, including ties.",
        "tab:ct-results",
    )


def boundary_notes(selection):
    notes = []
    for variant in SFRAMES:
        chosen = selection["selected"][variant]
        for key, symbol in (("beta_scale", r"c_\beta"), ("gamma0", r"\gamma_0")):
            boundary = selection["boundaries"][variant][key]
            if boundary["side"] != "interior":
                side = boundary["side"]
                value = parameter(chosen[key])
                kind = "admissible" if key == "gamma0" and chosen[key] == 1 else "searched"
                notes.append(rf"{METHODS[variant]} selects the {side} {kind} limit "
                             rf"\({symbol}={value}\).")
    if not notes:
        notes.append("Both selected SFRAMES pairs are interior to their final development grids.")
    return " ".join(notes)


def results(summary, selection, runs):
    budget = integer(summary["budgets"]["forward_view_cap"])
    sizes = list(summary["cohorts"])
    comparisons = []
    all_wins = True
    for comparator in ("sframes_full", "spring_sarah", "prox_sarah", "palm"):
        effects = []
        for n in sizes:
            stats = summary["cohorts"][n]["view_cap"]["sframes_vs"][comparator]["metrics"]["feasible_loss"]["percent_difference"]
            effects.append(rf"\({stats['median_across_slices']:+.1f}\%\) for \(N={integer(n)}\)")
            all_wins = all_wins and stats["slices_favoring_sframes"] == stats["slices"]
        name = METHODS[comparator].replace("Full-batch", "full-batch")
        comparisons.append(f"Relative to {name}, the median paired change in "
                           f"repaired training loss is {' and '.join(effects)}.")
    wins = ("For every comparator and training-view count, all six slices favor minibatch "
            "SFRAMES in median paired repaired loss." if all_wins else "")
    sf_runs = [run for key, run in runs.items() if key[3] in SFRAMES]
    base_runs = [run for key, run in runs.items() if key[3] not in SFRAMES]
    endpoints = [next(row for row in reversed(run["trace"])
                      if max(row["forward_views"], row["backward_views"]) <= summary["budgets"]["forward_view_cap"])
                 for run in sf_runs]
    repair_gap = max(100 * abs(row["feasible_loss"] - row["raw_loss"]) / row["feasible_loss"]
                     for row in endpoints)
    leaves_and_remains = all(run["trace"][1]["d_distance"] > 0 for run in sf_runs)
    leaves_and_remains = leaves_and_remains and all(row["d_distance"] > 0 for row in endpoints)
    sf_feasibility = (
        r"Both SFRAMES variants leave \(D\) after their first update and retain nonzero "
        r"orthant distance at the work cap." if leaves_and_remains else
        r"The figures show the orthant distance of the raw SFRAMES factors alongside their repaired losses."
    )
    base_feasibility = ""
    if all(row["d_distance"] == 0 for run in base_runs for row in run["trace"]):
        base_feasibility = r"The projection baselines remain in \(D\) throughout."
    return rf"""\subsection{{Results}}
\label{{sec:ct-results}}

The comparison contains \(132\) complete evaluation runs, with a common
budget of \({budget}\) forward and \({budget}\) adjoint views.
Table~\ref{{tab:ct-results}} reports reconstruction quality and feasibility.
{" ".join(comparisons)}
Negative changes favor minibatch SFRAMES. {wins}
Paired percentages use the comparator as denominator and can differ from
ratios of the marginal medians in the table.

{quality_table(summary)}

Figures~\ref{{fig:ct-trajectories-360}} and~\ref{{fig:ct-trajectories-1080}}
show the observed losses and orthant distances against iterations,
gradient-oracle calls and processed views. {sf_feasibility}
Across the SFRAMES runs, the largest absolute raw--repaired endpoint-loss
difference is \({repair_gap:.2f}\%\) of the repaired loss. {base_feasibility}

These descriptive comparisons use six slices from two specimen tubes.
The baseline step multipliers are retained from their development searches.
{boundary_notes(selection)} The observed comparisons apply to the stated
parameter grids and data budget.

"""


def settings_table(selection, design, baseline):
    rows = []
    for variant in SFRAMES:
        grid = selection["grids"][variant]
        betas = ", ".join(map(parameter, grid["beta_scales"]))
        gammas = ", ".join(map(parameter, grid["gamma0"]))
        chosen = selection["selected"][variant]
        name = rf"\shortstack[l]{{{METHODS[variant]}\\ $(c_\beta,\gamma_0)$}}"
        candidates = rf"\shortstack[l]{{$c_\beta\in\{{{betas}\}}$\\ $\gamma_0\in\{{{gammas}\}}$}}"
        selected = rf"$({parameter(chosen['beta_scale'])},{parameter(chosen['gamma0'])})$"
        rows.append((name, candidates, selected))
    for variant in ("spring_sarah", "prox_sarah", "palm"):
        trials = design["baseline_development_trials"][variant]
        chosen = trials[baseline["selected"][variant]]
        second = "refresh_period" if variant == "spring_sarah" else "inner_steps"
        symbol = r"(a,q)" if variant == "spring_sarah" else r"(a,m)"
        if variant == "palm":
            symbol = "a"
            candidates = ", ".join(parameter(trial["rate"]) for trial in trials)
            selected = parameter(chosen["rate"])
        else:
            candidates = ", ".join(f"({parameter(trial['rate'])},{trial[second]})" for trial in trials)
            selected = f"({parameter(chosen['rate'])},{chosen[second]})"
        rows.append((rf"{METHODS[variant]}, ${symbol}$", candidates, selected))
    return table(
        "lll", ("Method and parameter", "Values tested", "Selected"), rows,
        r"Development hyperparameters. The two SFRAMES rows list complete Cartesian grids "
        r"for the joint smoothing multiplier and step prefactor, with the selected pair in "
        r"the final column. Baseline settings are retained from their original development searches.",
        "tab:ct-settings",
    )


def protocol_tail(summary, selection):
    return rf"""
We jointly select \((c_\beta,\gamma_0)\) for each SFRAMES variant by the
smallest median repaired-to-initial training-loss ratio over the two
development slices and both training-view counts at
\({integer(selection['development_view_budget'])}\) views per direction.
When a selected pair lies on a grid edge, that dimension is extended while
retaining the full Cartesian product. Table~\ref{{tab:ct-settings}} gives the
final tested grids, and Figure~\ref{{fig:ct-joint-selection}} shows their
observed scores. Held-out views and reference images do not enter selection.
The initial calibration fixed \(c_\beta=1\) by this curvature-scaling
heuristic and varied only \(\gamma_0\). All SFRAMES evaluation results
reported here use the joint selection.
The SPRING--SARAH, ProxSARAH and PALM step multipliers are at the upper
edges of their original grids.

Evaluation permits \({integer(summary['budgets']['forward_view_cap'])}\)
forward and adjoint views each. Counters include all gradient evaluations,
refreshes, snapshots and both terms of recursive differences. A gradient call
returns a joint or block gradient. Each iteration updates both factors.
Data processed is \(W=\sum_{{\ell=1}}^G|B_\ell|\), counting the views in
each of the \(G\) gradient calls. One processed view uses one forward and
one adjoint over all \(956\) bins. Diagnostic forward projections are
evaluated separately. Evaluation records the initialization and every
optimizer update, and tables use each run's last iterate within budget.
Paired percentage changes are computed within each seed pair, then summarized
by medians over seeds within slices and across slices.

\begin{{figure}}[!htbp]
 \centering
 \includegraphics[width=\linewidth]{{joint_selection.pdf}}
 \caption{{Joint development search. Each cell gives the median repaired
 training loss relative to initialization across the two development slices
 and both training-view counts. Here
 \(\beta_0=c_\beta/\max(L_U,L_V)\). Boxes mark the selected pairs, and lower
 values are better. The color scale is shared and logarithmic.}}
 \label{{fig:ct-joint-selection}}
\end{{figure}}
"""


def render_source(template, selection, summary, runs, baseline, design):
    start = template.index(r"\paragraph{Initialization and SFRAMES parameters.}")
    end = template.index(r"\paragraph{Evaluation criteria.}", start)
    source = template[:start] + setup(selection) + template[end:]
    start = source.index(r"\subsection{Results}")
    end = source.index(r"\begin{figure}", start)
    source = source[:start] + results(summary, selection, runs) + source[end:]
    label = source.index(r"\label{tab:ct-settings}")
    start = source.rfind(r"\begin{table}", 0, label)
    source = source[:start] + settings_table(selection, design, baseline) + protocol_tail(summary, selection)
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    references = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source))
    if references - labels:
        raise ValueError(f"Paper references undefined labels: {references - labels}")
    if r"d_C" in source or "tab:ct-work" in source or "comparison.pdf" in source:
        raise ValueError("A removed table or plot has returned to the paper")
    if source.index(r"\subsection{Experimental protocol}") < source.index(r"\label{fig:ct-trajectories-1080}"):
        raise ValueError("The detailed protocol must remain after the results and figures")
    return source


def atomic_copy(source, destination):
    pending = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, pending)
    pending.replace(destination)


def publish(root, paper):
    selection, summary, runs, baseline = load_inputs(root)
    design = read(root / "manifest.json")["design"]
    source = render_source((paper / BODY).read_text(), selection, summary, runs, baseline, design)
    plot_script = paper / "plot_trajectories.py"
    plot_source = plot_script.read_text()
    old_default = 'ROOT = EXPANDED / "dense_trajectories"'
    new_default = 'ROOT = EXPANDED / "joint_tuning"'
    if old_default not in plot_source and new_default not in plot_source:
        raise ValueError("The paper plotting script has an unexpected default data root")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT.parent.parent.parent / "deps")
    figures = [f"trajectories_{n}.{suffix}" for n in design["evaluation_training_views"]
               for suffix in ("pdf", "png")]
    figures += ["joint_selection.pdf", "joint_selection.png"]
    with TemporaryDirectory(prefix="ct-joint-paper-") as temporary:
        stage = Path(temporary)
        (stage / BODY).write_text(source)
        shutil.copyfile(paper / f"{STEM}.tex", stage / f"{STEM}.tex")
        subprocess.run([sys.executable, str(plot_script), "--root", str(root),
                        "--output", str(stage)], env=env, check=True)
        subprocess.run([sys.executable, str(root / "plot_selection.py"),
                        "--root", str(root), "--output", str(stage)], env=env, check=True)
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        f"{STEM}.tex"], cwd=stage, check=True)
        subprocess.run(["pdftotext", str(stage / f"{STEM}.pdf"), str(stage / "paper.txt")], check=True)
        if "??" in (stage / "paper.txt").read_text():
            raise ValueError("The compiled paper contains an unresolved reference")
        for name in figures + [BODY, f"{STEM}.pdf"]:
            atomic_copy(stage / name, paper / name)
    pending = plot_script.with_name(plot_script.name + ".tmp")
    pending.write_text(plot_source.replace(old_default, new_default))
    pending.replace(plot_script)
    audit = dict(
        selection_sha256=digest(root / "selection.json"),
        summary_sha256=digest(root / "reports" / "summary.json"),
        report_source_sha256=digest(paper / BODY),
        report_pdf_sha256=digest(paper / f"{STEM}.pdf"),
        plotting_source_sha256=digest(plot_script),
        report_builder_sha256=digest(Path(__file__)),
        report_pdf=str(paper / f"{STEM}.pdf"),
        evaluation_runs=len(runs), native_iteration_observations=sum(len(run["trace"]) for run in runs.values()),
    )
    pending = root / "reports" / "report_update.json.tmp"
    pending.write_text(json.dumps(audit, indent=2) + "\n")
    pending.replace(root / "reports" / "report_update.json")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    with (ROOT / "report_update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(ROOT, PAPER)
