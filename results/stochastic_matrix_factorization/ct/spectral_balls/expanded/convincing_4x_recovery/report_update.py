"""Publish expanded, retuned, four-budget results in the existing paper section.

No figure or table is added: the two trajectory figures, quality table and
parameter table are regenerated, while the previous SFRAMES search figure is
retained and explicitly identified as the original development search.
"""

import argparse
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
PREVIOUS = ROOT.parent / "joint_tuning"
BODY = "ct_experiments.tex"
STEM = "sframes_ct_section"
METHODS = {
    "sframes": "Minibatch SFRAMES", "sframes_full": "Full-batch SFRAMES",
    "spring_sarah": "SPRING--SARAH", "prox_sarah": "ProxSARAH", "palm": "PALM",
}
SFRAMES = ("sframes", "sframes_full")
BASELINES = ("spring_sarah", "prox_sarah", "palm")
METRICS = (("raw_loss", "Raw", ".5g"), ("feasible_loss", "Repaired", ".5g"),
           ("validation_loss", "Held-out", ".5g"),
           ("reference_relative_error", "Ref. error", ".4f"), ("d_distance", r"$d_D$", ".4g"))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integer(value):
    return f"{int(value):,}".replace(",", "{,}")


def parameter(value):
    return f"{value:.12g}" if isinstance(value, (int, float)) else str(value)


def numeric(value, style, bold=False):
    token = format(value, style)
    if "e" in token:
        base, exponent = token.split("e")
        token = rf"{base}\times10^{{{int(exponent)}}}"
    if bold:
        token = rf"\boldsymbol{{{token}}}"
    return f"${token}$"


def table(columns, headings, rows, caption, label):
    return "\n".join((r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
                       rf"\label{{{label}}}", r"\small", r"\setlength{\tabcolsep}{4pt}",
                       rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
                       " & ".join(headings) + r" \\", r"\midrule",
                       *(" & ".join(row) + r" \\" for row in rows),
                       r"\bottomrule", r"\end{tabular}", r"\end{table}"))


def labels(design):
    return design.get("specimen_group_labels", design.get("group_labels", {}))


def validate_optimizer(run, chosen):
    config, meta = run["config"], run["meta"]
    variant = config["variant"]
    u, v = meta["curvature"]
    if variant in SFRAMES:
        gamma = chosen.get("gamma0", chosen.get("rate"))
        scale = chosen["beta_scale"]
        expected = dict(batch=config["batch"], gamma0=gamma, rho0=1.0, mu0=scale / max(u, v))
        if config["rate"] != gamma or config["beta_scale"] != scale:
            raise ValueError("A SFRAMES arm does not use its frozen parameter pair")
    elif variant == "palm":
        expected = dict(eta_u=chosen["rate"] / u, eta_v=chosen["rate"] / v)
    elif variant == "spring_sarah":
        expected = dict(batch=config["batch"], eta_u=chosen["rate"] / u,
                        eta_v=chosen["rate"] / v, refresh_period=chosen["refresh_period"])
    else:
        snapshot_batch = (config["n_train"] if chosen.get("snapshot_policy", "minibatch") == "full"
                          else min(config["n_train"], 4 * config["batch"]))
        expected = dict(batch=config["batch"], eta=chosen["rate"] / (u + v), relaxation=0.8,
                        inner_steps=chosen["inner_steps"], snapshot_batch=snapshot_batch)
    if meta["optimizer"] != expected:
        raise ValueError(f"Actual optimizer differs from the selected settings: {config['run_id']}")
    if variant in BASELINES:
        for field in ("rate", "refresh_period", "inner_steps", "snapshot_policy"):
            if field in chosen and config.get(field) != chosen[field]:
                raise ValueError(f"A baseline uses a different selected {field}")


def load_inputs(root):
    if not (root / "audit_provenance.py").is_file() or not (root / "input_anchor.json").is_file():
        raise ValueError("Publication requires the frozen-input provenance audit gate and anchor")
    selection = read(root / "selection.json")
    if selection.get("status") != "selected":
        raise ValueError("Baseline development selection is not final")
    summary = read(root / "reports" / "summary.json")
    manifest = read(root / "manifest.json")
    design = manifest["design"]
    for field, filename in (("cohort_sha256", "cohort.json"), ("targets_sha256", "targets.json"),
                            ("protocol_sha256", "protocol_frozen.json"), ("selection_sha256", "selection.json"),
                            ("baseline_selection_sha256", "baseline_selection.json")):
        if manifest[field] != digest(root / filename):
            raise ValueError(f"The frozen evaluation input changed: {filename}")
    if manifest["original_selection_sha256"] != digest(PREVIOUS / "selection.json"):
        raise ValueError("The original SFRAMES selection changed after the new campaign was frozen")
    analysis = load_module("expanded_ct_report_analysis", root / "analyze.py")
    auditor = load_module("recovered_ct_report_provenance", root / "audit_provenance.py")
    auditor.audit_evaluation(root, require_complete=True)
    smoke_anchor = read(root / "recovery_smoke_anchor.json")
    if smoke_anchor["recovery_input_anchor_sha256"] != digest(root / "recovery_input_anchor.json"):
        raise ValueError("The recovery smoke validation uses a different input anchor")
    smoke_path = (root / smoke_anchor["engine_smoke_path"]).resolve()
    if not smoke_path.is_relative_to(root) or digest(smoke_path) != smoke_anchor["engine_smoke_sha256"]:
        raise ValueError("The actual recovery-runner smoke record changed")
    runs = analysis.collect(root)
    groups = analysis.group_map(design)
    seeds = sorted(design["evaluation_seeds"])
    deterministic = set(design["deterministic_variants"])
    expected = {(analysis.slice_id(sid), seed, n, variant)
                for sid, seed, n, variant in product(design["evaluation_slices"], seeds,
                                                    design["evaluation_training_views"], METHODS)
                if variant not in deterministic or seed == seeds[0]}
    if set(runs) != expected:
        raise ValueError("The paper requires the complete frozen evaluation cohort")
    budget = design["evaluation_view_budget"]
    original_cap = design.get("original_evaluation_view_budget", 17280)
    if budget != 4 * original_cap:
        raise ValueError("The expanded evaluation must use four times the original work cap")
    for key, run in runs.items():
        if run["status"] != "completed" or not run["trace"]:
            raise ValueError(f"Evaluation is incomplete: {run['config']['run_id']}")
        if run["trace"][-1]["forward_views"] < budget:
            raise ValueError("An evaluation arm stopped below the extended cap")
        validate_optimizer(run, selection["selected"][key[3]])
        analysis.initial_factor_norm(run["meta"])
    targets = analysis.load_targets(root)
    if targets is None or len(targets["targets"]) != 4:
        raise ValueError("The report requires all four prespecified joint accuracy targets")
    fresh = analysis.analyze(root, design, budget)
    if fresh["coverage"]["physical_arm_status_counts"] != {"completed": len(expected)}:
        raise ValueError("The complete evaluation cohort did not pass validation")
    if (fresh["cohorts"] != summary["cohorts"] or summary["selection"] != selection
            or summary["targets"] != targets):
        raise ValueError("The saved summary is stale")
    cohort = read(root / "cohort.json")
    if (set(groups) != set(cohort["evaluation_slices"])
            or groups != cohort["evaluation_slice_groups"]):
        raise ValueError("The evaluation groups changed after the cohort was frozen")
    previous_path = root / "sframes_selection.json"
    previous = read(previous_path if previous_path.exists() else PREVIOUS / "selection.json")
    if previous != read(PREVIOUS / "selection.json"):
        raise ValueError("The retained SFRAMES selection copy differs from its frozen source")
    for variant in SFRAMES:
        for field in ("beta_scale", "gamma0"):
            if selection["selected"][variant][field] != previous["selected"][variant][field]:
                raise ValueError("The retained SFRAMES search figure has different selected settings")
    preparation = read(root / "data_preparation_manifest.json")
    if preparation["status"] != "complete":
        raise ValueError("The expanded data preparation manifest is incomplete")
    for run in runs.values():
        sid = run["config"]["slice_id"]
        if run["meta"]["data_sha256"] != preparation["records"][sid]["sha256"]:
            raise ValueError(f"An evaluation arm used unregistered data for {sid}")
    return selection, summary, runs, design, previous, preparation


def setup(selection):
    chosen = selection["selected"]["sframes"]
    full = selection["selected"]["sframes_full"]
    if (chosen["beta_scale"], chosen["gamma0"]) != (full["beta_scale"], full["gamma0"]):
        raise ValueError("The concise setup assumes the retained common SFRAMES pair")
    return r"""\paragraph{Initialization and SFRAMES parameters.}
All methods start from common nonnegative rank-\(64\) factors \(U_0,V_0\),
obtained by an NNDSVD-style factorization of a sensitivity-normalized
backprojection of \(120\) training views, followed by a small positive
perturbation and a scale fit. Set \(R_U=2\|U_0\|_{\rm op}\) and
\(R_V=2\|V_0\|_{\rm op}\). Ten power iterations on the stacked calibration
operator \(P_J\) estimate \(\widehat\lambda\approx\|P_J\|_{\rm op}^2/(120d)\).
We use the initial block-curvature estimates to scale the smoothing prefactor:
\[
 L_U=\widehat\lambda\|V_0\|_{\rm op}^2,\qquad
 L_V=\widehat\lambda\|U_0\|_{\rm op}^2,\qquad
 \beta_0=c_\beta/\max(L_U,L_V).
\]
""" + rf"Both SFRAMES variants retain \((c_\beta,\gamma_0)=({parameter(chosen['beta_scale'])}," \
        rf"{parameter(chosen['gamma0'])})\) from the original joint development search." + r"""
Minibatch SFRAMES draws \(64\) training views uniformly without replacement
at each update, independently across updates; full-batch SFRAMES uses all
\(N\) views. Both use
\[
 \gamma_k=\gamma_0(k+1)^{-3/4},\qquad
 \alpha_k=(k+1)^{-1/2},\qquad
 \beta_k=\beta_0(k+1)^{-1/4}.
\]

"""


def quality_table(summary):
    rows = []
    for n, cohort in summary["cohorts"].items():
        marginal = cohort["view_cap"]["matched_all_methods"]["median_of_group_medians"]
        minima = {field: min(marginal[v][field] for v in METHODS) for field, _, _ in METRICS}
        for variant, name in METHODS.items():
            values = marginal[variant]
            rows.append((integer(n), name, *(numeric(values[field], style, values[field] == minima[field])
                                            for field, _, style in METRICS)))
    return table("llrrrrr", (r"$N$", "Method", *(name for _, name, _ in METRICS)), rows,
                 r"Reconstruction quality at the last recorded iterate within the extended budget. "
                 r"Entries take medians over seeds within slices, slices within specimen/material groups, "
                 r"then across groups. Raw and repaired are training losses; held-out loss and relative "
                 r"reference error use repaired factors. Lower is better. Bold gives the best value "
                 r"within each training-view count, including ties.", "tab:ct-results")


def target_results(summary):
    targets = summary["targets"]["targets"]
    sizes = list(summary["cohorts"])
    target_order = ",".join(rf"({parameter(t['relative_repaired_loss'])},{parameter(t['max_relative_d_distance'])})"
                            for t in targets)
    vectors = []
    for variant, name in METHODS.items():
        rates = []
        for target in targets:
            values = [summary["cohorts"][n]["accuracy_targets"][target["name"]]["methods"][variant]
                      ["mean_group_attainment"] for n in sizes]
            rates.append(100 * sum(values) / len(values))
        vector = ",".join(f"{rate:.0f}" for rate in rates)
        vectors.append(rf"\(({vector})\) for {name.replace('Full-batch', 'full-batch')}")
    ratio_values, coverages, group_coverages, available, total = [], [], [], 0, 0
    for n in sizes:
        for target in targets:
            stats = summary["cohorts"][n]["accuracy_targets"][target["name"]]
            for variant in METHODS:
                if variant == "sframes":
                    continue
                total += 1
                pair = stats["sframes_vs"][variant]
                value = pair["first_views_ratio_when_both_reached"]["median_across_groups"]
                if value is not None:
                    ratio_values.append(value)
                    counts = pair["attainment_pair_counts"]
                    coverages.append((counts.get("both_reached", 0), sum(counts.values())))
                    group_coverages.append(pair["first_views_ratio_when_both_reached"]["groups"])
                    available += 1
    ratios = "No comparator/target/view-count combination has a common attained pair."
    if ratio_values:
        denominators = sorted(set(d for _, d in coverages))
        if len(denominators) != 1:
            raise ValueError("Accuracy-target ratio coverage needs a common paired-case denominator")
        lower, upper = min(n for n, _ in coverages), max(n for n, _ in coverages)
        count = str(lower) if lower == upper else rf"{lower}\text{{--}}{upper}"
        group_count = (str(group_coverages[0]) if min(group_coverages) == max(group_coverages)
                       else rf"{min(group_coverages)}\text{{--}}{max(group_coverages)}")
        ratios = (rf"Conditional on both methods attaining a target, specimen-balanced median "
                  rf"SFRAMES/comparator work ratios range from \({min(ratio_values):.2f}\) to "
                  rf"\({max(ratio_values):.2f}\) across \({available}/{total}\) comparator/target/view-count "
                  rf"combinations, each using \({count}\) of \({denominators[0]}\) paired seed cases "
                  rf"from \({group_count}\) of \({summary['coverage']['specimen_groups']}\) groups.")
    return (rf"For the four joint targets, ordered as \({target_order}\), the specimen-balanced "
            r"attainment percentages, averaged equally over the two view counts, are "
            + "; ".join(vectors) + ". "
            + r"Unreached targets remain right censored. " + ratios)


def baseline_boundary_notes(selection):
    notes = []
    for variant in BASELINES:
        for field, boundary in selection.get("boundaries", {}).get(variant, {}).items():
            if boundary["side"] not in ("interior", "categorical"):
                notes.append(rf"{METHODS[variant]} selects the {boundary['side']} predefined search limit "
                             rf"for \({'a' if field == 'rate' else 'q' if field == 'refresh_period' else 'm'}\).")
    return " ".join(notes) if notes else "All retuned numerical baseline parameters are interior to their final grids."


def results(summary, selection, runs):
    coverage = summary["coverage"]
    budget = integer(summary["budgets"]["forward_view_cap"])
    comparators = ("sframes_full", "spring_sarah", "prox_sarah", "palm")
    effects, group_wins, group_changes = [], [], []
    for n, cohort in summary["cohorts"].items():
        changes = []
        for variant in comparators:
            stats = cohort["view_cap"]["sframes_vs"][variant]["metrics"]["feasible_loss"]["percent_difference"]
            changes.append(f"{stats['median_across_groups']:+.1f}")
            group_wins.append(stats["groups_favoring_sframes"])
            group_changes.extend(stats["group_medians"].values())
        effects.append(rf"\(({','.join(changes)})\%\) at \(N={integer(n)}\)")
    endpoints = {key: next(row for row in reversed(run["trace"])
                          if max(row["forward_views"], row["backward_views"]) <= summary["budgets"]["forward_view_cap"])
                 for key, run in runs.items()}
    sf = [row for key, row in endpoints.items() if key[3] in SFRAMES]
    gap = max(100 * abs(row["feasible_loss"] - row["raw_loss"]) / row["feasible_loss"] for row in sf)
    sf_distance = "The raw SFRAMES factors retain nonzero orthant distance at the cap."
    if not all(row["d_distance"] > 0 for row in sf):
        sf_distance = "The trajectories include raw SFRAMES orthant distances alongside repaired losses."
    loosest_tolerance = max(target["max_relative_d_distance"] for target in summary["targets"]["targets"])
    if all(row["d_distance"] / runs[key]["meta"]["initial_factor_frobenius_norm"] > loosest_tolerance
           for key, row in endpoints.items() if key[3] in SFRAMES):
        sf_distance = (rf"At the cap, every raw SFRAMES orthant distance exceeds "
                       rf"\({100 * loosest_tolerance:g}\%\) of the initial joint Frobenius norm.")
    wins_text = (rf"all \({coverage['specimen_groups']}\) groups" if min(group_wins) == coverage["specimen_groups"]
                 else rf"\({min(group_wins)}\)--\({max(group_wins)}\) of \({coverage['specimen_groups']}\) groups")
    base = [run for key, run in runs.items() if key[3] in BASELINES]
    baseline_feasibility = (r"The projection baselines remain in \(D\) throughout."
                            if all(row["d_distance"] == 0 for run in base for row in run["trace"]) else "")
    return rf"""\subsection{{Results}}
\label{{sec:ct-results}}

The comparison contains \({coverage['expected_physical_arms']}\) complete evaluation runs on
\({coverage['evaluation_slices']}\) slices from \({coverage['specimen_groups']}\) specimen/material groups,
with a common cap of \({budget}\) forward and \({budget}\) adjoint views,
four times the original budget. Table~\ref{{tab:ct-results}} reports quality and feasibility.
Median paired changes in repaired training loss relative to full-batch SFRAMES,
SPRING--SARAH, ProxSARAH and PALM, in that order, are
{' and '.join(effects)}. Negative changes favor minibatch SFRAMES.
Paired changes are computed before aggregation and can differ from ratios of the table entries.
Across comparator/view-count combinations, minibatch SFRAMES is favored in
{wins_text};
individual group changes range from \({min(group_changes):+.1f}\%\) to \({max(group_changes):+.1f}\%\).

{quality_table(summary)}

Figures~\ref{{fig:ct-trajectories-360}} and~\ref{{fig:ct-trajectories-1080}}
show every recorded update against iterations, oracle calls and processed views.
The grey work-axis marker identifies the original \({integer(summary['budgets']['original_forward_view_cap'])}\)-view cap.
{sf_distance} The largest absolute raw--repaired SFRAMES endpoint-loss gap is
\({gap:.2f}\%\) of repaired loss. {baseline_feasibility}

{target_results(summary)}

These are descriptive comparisons across the prespecified groups; slices and
oracle seeds within a group are repeated measurements. {baseline_boundary_notes(selection)}

"""


def candidate_lists(grid, symbols):
    parts = []
    for field, values in grid.items():
        if field == "snapshot_policy":
            entries = ",".join(rf"\mathrm{{{'mini' if value == 'minibatch' else value}}}" for value in values)
        else:
            entries = ",".join(parameter(value) for value in values)
        parts.append(rf"${symbols[field]}\in\{{{entries}\}}$")
    joined = r" \\ ".join(parts)
    return rf"\shortstack[l]{{{joined}}}"


def settings_table(selection, previous):
    rows = []
    for variant in SFRAMES:
        grid = previous["grids"][variant]
        chosen = selection["selected"][variant]
        candidates = candidate_lists({"beta_scale": grid["beta_scales"], "gamma0": grid["gamma0"]},
                                     {"beta_scale": r"c_\beta", "gamma0": r"\gamma_0"})
        rows.append((rf"\shortstack[l]{{{METHODS[variant]}\\$(c_\beta,\gamma_0)$}}", candidates,
                     rf"$({parameter(chosen['beta_scale'])},{parameter(chosen['gamma0'])})$"))
    for variant in BASELINES:
        grid, chosen = selection["grids"][variant], selection["selected"][variant]
        secondary = "refresh_period" if variant == "spring_sarah" else "inner_steps"
        symbol = r"(a,q)" if variant == "spring_sarah" else r"(a,m,s)"
        if variant == "palm":
            symbol = "a"
            selected = parameter(chosen["rate"])
        else:
            selected = f"({parameter(chosen['rate'])},{chosen[secondary]})"
            if variant == "prox_sarah":
                policy = "mini" if chosen.get("snapshot_policy", "minibatch") == "minibatch" else "full"
                selected = rf"({parameter(chosen['rate'])},{chosen[secondary]},\mathrm{{{policy}}})"
        rows.append((rf"{METHODS[variant]}, ${symbol}$",
                     candidate_lists(grid, {"rate": "a", "refresh_period": "q", "inner_steps": "m", "snapshot_policy": "s"}),
                     f"${selected}$"))
    return table("lll", ("Method and parameter", "Values tested", "Selected"), rows,
                 r"Hyperparameter searches. Every row denotes the full Cartesian product of its "
                 r"listed values. SFRAMES settings are retained from the original joint search; "
                 r"the three baselines are retuned at the longer development horizon.", "tab:ct-settings")


def preprocessing_text(design, preparation):
    records = preparation["records"]
    evaluation = set(design["evaluation_slices"])
    clamped = [(sid, records[sid].get("clamped_count", 0)) for sid in evaluation
               if records[sid].get("clamped_count", 0)]
    if not clamped:
        floor = "The transmission floor is inactive for every evaluation acquisition."
    else:
        pairs = ", ".join(rf"\({integer(count)}\) in \texttt{{{sid.removeprefix('slice')}}}"
                          for sid, count in sorted(clamped))
        floor = f"The floor affects {pairs}; it is inactive in the other evaluation acquisitions."
    return r"""We discard the repeated final angle, sum adjacent detector cells, subtract
the similarly binned dark field, and divide by the mean flat field after
dark correction. Detector~1 acquisitions receive the published one-index
linear transmission shift with extrapolation before taking the negative
logarithm with floor \(10^{-6}\); Detector~2 uses zero shift. Reversing the
detector axis gives \(3{,}600\times956\) sinograms in the DeepInverse convention
\cite{deepinv}. """ + floor + r"""
ASTRA~2.5.0 applies the FP32 CPU \texttt{line\_fanflat} projector \cite{astra},
while factor arithmetic is FP64. All acquisitions use source-to-origin and
origin-to-detector distances \(431.019989\) and \(97.980499\)~mm, detector
spacing \(0.1496\)~mm, and a square field of width \(116.528143\)~mm centered
at \((-0.113797,-0.113797)\)~mm. View \(i\) has angle \(\pi-2\pi i/3600\).
We vertically flip the \(1024\times1024\) reference, divide by its native
voxel width \(0.113797\)~mm, and average \(2\times2\) blocks.
"""


def protocol(summary, selection, previous, design, preparation):
    groups = load_module("expanded_ct_protocol_analysis", ROOT / "analyze.py").group_map(design)
    names = labels(design)
    cohort = []
    for group in sorted(set(groups.values()), key=lambda key: (not key.startswith("Mix"), names.get(key, key))):
        slices = ", ".join(rf"\texttt{{{sid.removeprefix('slice')}}}" for sid in sorted(groups)
                           if groups[sid] == group)
        cohort.append(f"{names.get(group, group)} ({slices})")
    development_seeds = design.get("development_seeds", [110, 111, 112])
    devseeds = ", ".join(map(str, development_seeds))
    sf_budget = integer(previous["development_view_budget"])
    baseline_budget = integer(selection["development_view_budget"])
    target_losses = sorted({target["relative_repaired_loss"] for target in summary["targets"]["targets"]}, reverse=True)
    target_distances = sorted({target["max_relative_d_distance"] for target in summary["targets"]["targets"]}, reverse=True)
    return rf"""\clearpage
\subsection{{Experimental protocol}}
\label{{subsec:ct-protocol}}

We use Mode~1 low-dose fan-beam measurements from 2DeteCT \cite{{detect}};
the published Mode~2 high-dose reconstruction is the reference. Development
retains slices \texttt{{04300}} and \texttt{{03150}}. The prespecified evaluation
groups are {'; '.join(cohort)}. Slices within each group are sections of the
same specimen/material acquisition, rather than independent specimens.

{preprocessing_text(design, preparation)}

For \(i\in\{{0,\ldots,3599\}}\), training selects \(i\bmod10=0\) for
\(N=360\) and \(i\bmod10\in\{{0,3,6\}}\) for \(N=1{{,}}080\).
Both reserve the \(360\) views with \(i\bmod10=9\). Calibration uses
the \(120\) training views with \(i\bmod30=0\), shared across both sizes.
Evaluation uses paired oracle seed labels \(202,203,204\), offset by
\(100{{,}}000\) for generators. PALM and full-batch SFRAMES run once per
slice/size and serve as shared deterministic references. Baseline development
uses labels \({devseeds}\), with PALM once per case. Calibration uses seed
\(110\) in development and \(201\) in evaluation.

All methods start from common calibrated factors in \(C\cap D\). We divide
\((P_J^\ast y_J)_+\) entrywise by \(P_J^\ast P_J\mathbf1\), with denominator
floor \(10^{{-3}}\) times its maximum. A nonnegative SVD-based factorization
gives rank-\(64\) factors, perturbed by \(10^{{-3}}\) times each factor's RMS
times uniform \([0,1)\) entries. A least-squares amplitude fit on calibration
views, bounded below by \(10^{{-6}}\), rescales both factors by its square root.
The numerical LMO retains singular values above
\(\epsilon_{{\rm mach}}\max(\operatorname{{shape}}G)\sigma_{{\max}}(G)\).

SPRING--SARAH \cite{{spring}} alternates factor updates with steps
\(a/L_U,a/L_V\). Each block independently refreshes with probability \(1/q\),
starting from a full gradient; otherwise it uses a paired \(64\)-view SARAH
difference. ProxSARAH \cite{{proxsarah}} uses joint steps \(a/(L_U+L_V)\),
snapshots of \(\min(N,256)\) views for \(s=\mathrm{{mini}}\) or all \(N\)
views for \(s=\mathrm{{full}}\), \(m\) recursive \(64\)-view updates, and relaxation
\(0.8\). Both terms of each recursive difference share sampled views.
PALM \cite{{palm}} uses full gradients and alternating steps \(a/L_U,a/L_V\).
Proximal steps project onto each spectral ball intersected with the orthant.
A feasible single-set projection is accepted directly; otherwise Dykstra uses
a primal--dual distance certificate, tolerance \(10^{{-8}}(k+1)^{{-1.1}}\),
at most \(2{{,}}000\) iterations, and checks at the first, every fifth and
final iteration, with floating-point allowance.
Feasibility rescaling tests the norm before division, including the zero-matrix case.

{settings_table(selection, previous)}

SFRAMES retains its joint \((c_\beta,\gamma_0)\) selection from
\({sf_budget}\) views per direction and oracle label \(110\).
Figure~\ref{{fig:ct-joint-selection}} records that original search.
The baselines are retuned at \({baseline_budget}\) views per direction,
using complete Cartesian grids and extending selected edges to predefined
guards. Selection minimizes repaired-to-initial training-loss ratio, taking
seed medians within each development slice/size, then a median over the
four cases. Nonfinite development candidates are ineligible; operational
or projection failures are investigated. Evaluation groups, held-out views
and reference images do not enter parameter selection.

Evaluation permits \({integer(summary['budgets']['forward_view_cap'])}\)
forward and adjoint views each. Counters include every gradient evaluation,
refresh, snapshot and both terms of recursive differences. A gradient call
returns a joint or block gradient; each iteration updates both factors.
Processed views are \(W=\sum_{{\ell=1}}^G|B_\ell|\), where one view uses
one forward and one adjoint over all \(956\) bins. Diagnostic projections
are evaluated separately. Every update is recorded, and tables use the last
recorded iterate within budget. Existing trajectories continue from exact
optimizer/RNG checkpoints when parameters agree; new settings start from
the shared calibrated factors. Saved prefixes are preserved exactly.

Writing \(\mathcal R(x)\) for the pair of repaired factors, we fix joint
targets before expanded evaluation:
\[
 \frac{{f(\mathcal R(x_k))}}{{f(x_0)}}\le\rho,
 \qquad \frac{{d_D(x_k)}}{{\sqrt{{\|U_0\|_F^2+\|V_0\|_F^2}}}}\le\varepsilon,
 \quad \rho\in\{{{','.join(map(parameter, target_losses))}\}},\quad
 \varepsilon\in\{{{','.join(map(parameter, target_distances))}\}}.
\]
Both conditions must hold at the same recorded iterate. First-hit work uses
native observations, without interpolated crossings. Unreached targets are
right censored; work ratios use common attained pairs with coverage stated.
Attainment is averaged over seeds within slices and slices within groups,
then equally over groups and the two view counts. Quality and paired-effect
summaries instead take medians at each level, with groups equally weighted.

\begin{{figure}}[!htbp]
 \centering
 \includegraphics[width=0.9\linewidth]{{joint_selection.pdf}}
 \caption{{Original SFRAMES joint development search at \({sf_budget}\) views
 per direction. Cells give median repaired training loss relative to
 initialization over two development slices and both training-view counts,
 using oracle label \(110\). Here \(\beta_0=c_\beta/\max(L_U,L_V)\).
 Boxes mark the retained pairs; lower is better.}}
 \label{{fig:ct-joint-selection}}
\end{{figure}}
"""


def render_source(template, selection, summary, runs, design, previous, preparation):
    analysis = load_module("expanded_ct_source_analysis", ROOT / "analyze.py")
    groups = analysis.group_map(design)
    start = template.index(r"\paragraph{Initialization and SFRAMES parameters.}")
    end = template.index(r"\paragraph{Evaluation criteria.}", start)
    source = template[:start] + setup(selection) + template[end:]
    source = re.sub(r"Evaluation uses .*?held-out views per slice\.",
                    lambda _: (rf"Evaluation uses \({len(design['evaluation_slices'])}\) 2DeteCT slices from "
                               rf"\({len(set(groups.values()))}\) specimen/material groups, with "
                               r"\(N\in\{360,1{,}080\}\) training views and \(360\) held-out views per slice."),
                    source, count=1, flags=re.S)
    start = source.index(r"\subsection{Results}")
    end = source.index(r"\begin{figure}", start)
    source = source[:start] + results(summary, selection, runs) + source[end:]
    start = source.index(r"\clearpage", source.index(r"\label{fig:ct-trajectories-1080}"))
    source = source[:start] + protocol(summary, selection, previous, design, preparation)
    source = source.replace("and then across six slices. Bands show the middle",
                            "then over slices within groups and across groups. Bands show the middle")
    source = source.replace("then across six slices. Bands show the middle",
                            "then over slices within groups and across groups. Bands show the middle")
    source = source.replace("slice medians\n for SFRAMES", "group medians\n for SFRAMES")
    source = source.replace("slice medians for SFRAMES", "group medians for SFRAMES")
    source = source.replace(" All loss axes are logarithmic.", "")
    source = re.sub(r"Training losses are\s+normalized by their initial values\.",
                    "Training losses are normalized by their initial values. All loss axes are logarithmic.", source)
    all_labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    references = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source))
    if references - all_labels:
        raise ValueError(f"Undefined paper references: {references - all_labels}")
    if source.count(r"\begin{figure}") != 3 or source.count(r"\begin{table}") != 2:
        raise ValueError("The update must keep the existing three figures and two tables")
    if any(token in source for token in (r"d_C", "tab:ct-work", "comparison.pdf")):
        raise ValueError("A removed plot or table has returned")
    if source.index(r"\subsection{Experimental protocol}") < source.index(r"\label{fig:ct-trajectories-1080}"):
        raise ValueError("The detailed protocol must remain at the end")
    if "across six slices" in source or "six slices from two" in source:
        raise ValueError("An obsolete aggregation or cohort claim remains")
    return source


def atomic_copy(source, destination):
    pending = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, pending)
    pending.replace(destination)


def publish(root, paper):
    selection, summary, runs, design, previous, preparation = load_inputs(root)
    source = render_source((paper / BODY).read_text(), selection, summary, runs, design, previous, preparation)
    figures = [f"trajectories_{n}.{suffix}" for n in design["evaluation_training_views"]
               for suffix in ("pdf", "png")] + ["joint_selection.pdf", "joint_selection.png"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root.parent.parent.parent / "deps")
    with TemporaryDirectory(prefix="ct-expanded-paper-") as temporary:
        stage = Path(temporary)
        (stage / BODY).write_text(source)
        shutil.copyfile(paper / f"{STEM}.tex", stage / f"{STEM}.tex")
        subprocess.run([sys.executable, str(root / "plot_trajectories.py"), "--root", str(root),
                        "--output", str(stage)], env=env, check=True)
        for suffix in ("pdf", "png"):
            shutil.copyfile(PREVIOUS / "reports" / f"joint_selection.{suffix}", stage / f"joint_selection.{suffix}")
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
                       cwd=stage, check=True)
        subprocess.run(["pdftotext", str(stage / f"{STEM}.pdf"), str(stage / "paper.txt")], check=True)
        if "??" in (stage / "paper.txt").read_text():
            raise ValueError("The compiled paper contains unresolved references")
        for name in figures + [BODY]:
            atomic_copy(stage / name, paper / name)
        atomic_copy(root / "plot_trajectories.py", paper / "plot_trajectories.py")
        # Publish the PDF last, once all of its source assets have been validated.
        atomic_copy(stage / f"{STEM}.pdf", paper / f"{STEM}.pdf")
    audit = dict(selection_sha256=digest(root / "selection.json"), targets_sha256=digest(root / "targets.json"),
                 cohort_sha256=digest(root / "cohort.json"), summary_sha256=digest(root / "reports" / "summary.json"),
                 report_source_sha256=digest(paper / BODY), report_pdf_sha256=digest(paper / f"{STEM}.pdf"),
                 plotting_source_sha256=digest(root / "plot_trajectories.py"),
                 report_builder_sha256=digest(Path(__file__)), report_pdf=str(paper / f"{STEM}.pdf"),
                 evaluation_runs=len(runs), native_iteration_observations=sum(len(r["trace"]) for r in runs.values()),
                 specimen_groups=summary["coverage"]["specimen_groups"],
                 figures=3, tables=2)
    audit["input_anchor_sha256"] = digest(root / "input_anchor.json")
    audit["provenance_gate_sha256"] = digest(root / "audit_provenance.py")
    audit["recovery_input_anchor_sha256"] = digest(root / "recovery_input_anchor.json")
    audit["compatibility_manifest_sha256"] = digest(root / "compatibility_manifest.json")
    audit["recovery_smoke_anchor_sha256"] = digest(root / "recovery_smoke_anchor.json")
    audit["recovery_smoke_anchor"] = read(root / "recovery_smoke_anchor.json")
    previous_path = root / "sframes_selection.json"
    audit["sframes_selection_sha256"] = digest(previous_path if previous_path.exists() else PREVIOUS / "selection.json")
    pending = root / "reports" / "report_update.json.tmp"
    pending.write_text(json.dumps(audit, indent=2) + "\n")
    pending.replace(root / "reports" / "report_update.json")
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--paper", type=Path, default=PAPER)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "report_update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(root, args.paper.resolve())


if __name__ == "__main__":
    main()
