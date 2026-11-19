"""Add validated nonnegative-only baselines to the existing CT paper.

The runtime root owns all stages, backups and publication journals. Numerical
traces and the previously reported summaries remain immutable. This module
does not modify the repository's captured November 19 report bundle.
"""

import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
from statistics import median
import subprocess
import sys
import tempfile
from types import SimpleNamespace


CAP = 69120
METHODS = ("anls", "mu")
VARIANTS = {
    "sframes": ("SFRAMES (loss tune)", "#0072B2"),
    "tuned_sframes": ("SFRAMES (joint tune)", "#333333"),
    "sframes_full": ("SFRAMES (full batch)", "#E69F00"),
    "spring_sarah": ("SPRING-SARAH", "#D55E00"),
    "prox_sarah": ("ProxSARAH-v3", "#009E73"),
    "palm": ("PALM", "#AA4499"),
    "anls": ("Alternating NNLS", "#56B4E9"),
    "mu": ("Multiplicative updates", "#882255"),
}
METRICS = ("raw_loss", "feasible_loss", "validation_loss", "reference_relative_error", "relative_d_distance")
START, END = "<!-- CT_NONNEGATIVE_BASELINES_START -->", "<!-- CT_NONNEGATIVE_BASELINES_END -->"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def comparison_row(row, method):
    """Keep original counters; add explicit comparison coordinates."""
    value = dict(row)
    if method in METHODS:
        value["comparison_views"] = max(row["forward_views"], row["backward_views"])
        value["comparison_oracle_calls"] = row["oracle_calls"]
        require(row["d_distance"] == 0 and row["feasible_loss"] == row["raw_loss"],
                "A nonnegative baseline was repaired or became infeasible")
    else:
        value["comparison_views"] = max(row[d + "_views"] - row.get("curvature_" + d + "_views", 0)
                                        for d in ("forward", "backward"))
        value["comparison_oracle_calls"] = row["gradient_calls"]
    require(value["comparison_views"] >= 0, "Negative comparison work")
    return value


def prepare_run(trace, meta, method, source_directory):
    converted = [comparison_row(row, method) for row in trace]
    selected = [row for row in converted if row["comparison_views"] <= CAP]
    require(selected and selected[0]["iteration"] == selected[0]["comparison_views"] == 0,
            "Missing zero-work initialization")
    require(all(a["comparison_views"] <= b["comparison_views"] for a, b in zip(converted, converted[1:])),
            "Recorded comparison work decreases")
    norm = meta.get("initial_factor_frobenius_norm", meta.get("initial_factor_norm"))
    require(norm and math.isfinite(norm) and norm > 0, "Missing initial factor-pair norm")
    return dict(trace=selected, meta=meta, method=method, source_directory=str(source_directory), initial_norm=norm)


def _anchor_map(value):
    """Accept the runtime's simple path-to-SHA map or explicit records."""
    if "files" in value:
        value = value["files"]
    elif "assets" in value:
        value = value["assets"]
    if isinstance(value, list):
        return {record["path"]: record["sha256"] for record in value}
    return {name: record["sha256"] if isinstance(record, dict) else record for name, record in value.items()}


def retained_runs(root):
    expanded = root.parent
    prior = read(root / "predecessor_publication.json")
    previous = read(prior["gradient_comparison_summary"])
    baseline = read(expanded / "baseline_step_rules/report_update.json")
    joint = read(baseline["corrected_joint_summary"])
    frozen = _anchor_map(read(root / "retained_report_inputs.json"))
    runs, seen = {}, set()

    def add(case, method, record):
        folder = Path(record["source_directory"])
        if (str(folder), method) in seen:
            return
        seen.add((str(folder), method))
        for name in ("trace.json", "metadata.json"):
            path = folder / name
            require(str(path) in frozen and digest(path) == frozen[str(path)], "A retained report input changed")
        meta, trace = read(folder / "metadata.json"), read(folder / "trace.json")
        seed = int(meta["config"]["seed"])
        key = case["slice_id"], seed, int(case["n_train"]), method
        require(key not in runs, "Duplicate physical retained run")
        runs[key] = prepare_run(trace, meta, method, folder)

    for case in previous["cases"]:
        for method, record in case["arms"].items():
            add(case, method, record)
    for case in joint["cases"]:
        add(case, "tuned_sframes", case["runs"]["tuned_sframes"])
    require(len(runs) == 448, "The retained six-arm cohort changed")
    return runs, previous, joint


def added_runs(root, validation, plan, protocol):
    campaign = module("nonnegative_report_campaign", root / "frozen_sources/nonnegative_campaign.py")
    campaign.verify_prepared(root=root)
    configs = {cfg["run_id"]: cfg for cfg in plan["runs"]}
    runs = {}
    for case in validation["cases"]:
        cfg = configs[case["run_id"]]
        checked = campaign.validate_result(cfg, plan["assignments"][case["run_id"]], plan, protocol, root=root)
        require(checked == case, "A completed baseline changed after validation")
        folder = Path(case["source_directory"])
        require(digest(folder / "trace.json") == case["trace_sha256"], "A baseline trace changed")
        meta, trace = read(folder / "metadata.json"), read(folder / "trace.json")
        key = case["slice_id"], int(cfg["calibration_seed"]), int(case["n_train"]), case["method"]
        require(key not in runs and key[3] in METHODS, "Duplicate or unexpected baseline")
        runs[key] = prepare_run(trace, meta, key[3], folder)
    require(len(runs) == 56, "Publication requires all 56 deterministic baselines")
    return runs


def grouped(records, groups):
    """Nested medians over repetitions, slices within groups, then groups."""
    per_slice = {}
    for slice_id, value in records:
        require(math.isfinite(value), "Nonfinite endpoint statistic")
        per_slice.setdefault(slice_id, []).append(value)
    per_group = {}
    for slice_id, values in per_slice.items():
        per_group.setdefault(groups[slice_id], []).append(median(values))
    return dict(median_across_groups=median(median(values) for values in per_group.values()),
                by_group={g: median(v) for g, v in per_group.items()},
                by_slice={sid: median(v) for sid, v in per_slice.items()})


def summarize(runs, slices, groups, anchors=None):
    require(len(runs) == 504, "The eight-arm comparison must contain 504 physical runs")
    result = dict(schema_version=1, status="completed_validated", budget_per_direction=CAP,
        budget_axis="maximum directional solver views; prior curvature-estimation work excluded",
        physical_runs=len(runs), added_runs=56, figures=2, tables=1, anchors=anchors or {},
        domains={m: "nonnegative factors only" if m in METHODS else "spectral balls and nonnegative factors"
                 for m in VARIANTS},
        evaluation_repair={m: "none; actual nonnegative factors" if m in METHODS else
                           "clip each factor, then radial spectral-ball repair if necessary" for m in VARIANTS},
        cohorts={}, cases=[])
    for (sid, seed, size, method), run in sorted(runs.items()):
        endpoint = dict(run["trace"][-1])
        endpoint["relative_d_distance"] = endpoint["d_distance"] / run["initial_norm"]
        result["cases"].append(dict(slice_id=sid, seed=seed, n_train=size, method=method,
            source_directory=run["source_directory"], initial_factor_frobenius_norm=run["initial_norm"],
            endpoint=endpoint, observations=len(run["trace"])))
    for size in (360, 1080):
        cohort, paired = {}, {}
        for method in VARIANTS:
            chosen = [c for c in result["cases"] if c["n_train"] == size and c["method"] == method]
            require({c["slice_id"] for c in chosen} == set(slices), "Incomplete specimen cohort")
            cohort[method] = {metric: grouped([(c["slice_id"], c["endpoint"][metric]) for c in chosen], groups)
                              for metric in METRICS}
            for metric in ("iteration", "comparison_views", "comparison_oracle_calls", "forward_views", "backward_views"):
                cohort[method][metric] = grouped([(c["slice_id"], c["endpoint"][metric]) for c in chosen], groups)
        for baseline in METHODS:
            reference = {c["slice_id"]: c for c in result["cases"] if c["method"] == baseline and c["n_train"] == size}
            paired[baseline] = {}
            for method in ("sframes", "tuned_sframes"):
                chosen = [c for c in result["cases"] if c["method"] == method and c["n_train"] == size]
                paired[baseline][method] = {metric: grouped([(c["slice_id"],
                    100 * (c["endpoint"][metric] / reference[c["slice_id"]]["endpoint"][metric] - 1))
                    for c in chosen], groups) for metric in ("raw_loss", "feasible_loss")}
        result["cohorts"][str(size)] = dict(methods=cohort, sframes_vs_nonnegative=paired)
    return result


def quality_table(summary):
    text = r"""\begin{table}[!htbp]
\centering
\caption{Endpoint quality at the common processed-view cap. Domains differ: the original methods target \(C\cap D\); NNLS and multiplicative updates target only \(D\). Losses are multiplied by \(10^3\). Eval., held-out loss and reference error use the evaluated nonnegative image; see the repair convention above. Raw infeasibility is relative to the initial factor-pair norm, in percent. Bold marks the best value per view count across both domains.}
\label{tab:ct-results}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lllrrrrr@{}}
\toprule
$N$ & Method & Domain & Raw & Eval. & Held-out & Ref. error & Infeas.\ (\%) \\
\midrule
"""
    for size in (360, 1080):
        methods = summary["cohorts"][str(size)]["methods"]
        best = {metric: min(methods[m][metric]["median_across_groups"] for m in VARIANTS) for metric in METRICS}
        for method, (label, _) in VARIANTS.items():
            label = label.replace("SPRING-SARAH", "SPRING--SARAH").replace("Multiplicative updates", "Multiplicative")
            cells = []
            for metric in METRICS:
                value = methods[method][metric]["median_across_groups"]
                scale = 1000 if metric.endswith("loss") else 100 if metric == "relative_d_distance" else 1
                formatted = f"{value * scale:.3f}" if metric == "reference_relative_error" else f"{value * scale:.2f}"
                if math.isclose(value, best[metric], rel_tol=1e-10, abs_tol=1e-14):
                    formatted = r"\boldsymbol{" + formatted + "}"
                cells.append("$" + formatted + "$")
            domain = "$D$" if method in METHODS else r"$C\cap D$"
            text += ("360" if size == 360 else "1{,}080") + " & " + label + " & " + domain + " & " + " & ".join(cells) + r" \\" + "\n"
    return text + "\\bottomrule\n\\end{tabular}\n\\end{table}"


def replace_once(source, old, new):
    require(source.count(old) == 1, "The predecessor paper changed: " + old[:70])
    return source.replace(old, new)


def source_text(source, summary, selected_inner):
    source = replace_once(source, "equal gradient-view budgets.",
        "matched processed-view budgets. We also test alternating nonnegative\nleast squares (NNLS) and multiplicative updates without spectral bounds.")
    source = replace_once(source, "Uniformly sampling views gives an unbiased estimator of", 
        "The additional NNLS and multiplicative baselines minimize the same loss\nover \\(D\\) alone, so they solve a less constrained application problem.\nUniformly sampling views gives an unbiased estimator of")
    source = replace_once(source, "We use three stochastic repetitions per slice and view\ncount, with a common initialization across methods.",
        "We use three stochastic repetitions per slice and view\ncount; deterministic methods are run once. All methods share the same\ninitial factors.")
    old = r"""We compare the last iterate within \(69{,}120\) gradient forward and
adjoint views per direction, including snapshots and recursive differences.
Curvature-estimation views are recorded separately and excluded from this cap.
A gradient call evaluates one joint or block gradient on a batch."""
    new = r"""We compare the last recorded state within \(69{,}120\) processed views
in each direction. The original methods count gradient views; NNLS and
multiplicative updates count every solver forward/adjoint application,
including inner trials and cached-data initialization. The original methods'
curvature-estimation work remains separate. An oracle call is a paired
forward/adjoint evaluation, or the multiplicative method's startup adjoint."""
    source = replace_once(source, old, new)
    source = replace_once(source, "To assess a feasible reconstruction, we clip each factor to\nnonnegativity and rescale it radially into its spectral ball if needed.",
        "For the original methods, evaluation clips each factor to nonnegativity\nand rescales it radially into its spectral ball if needed. NNLS and\nmultiplicative updates are evaluated at their actual nonnegative factors,\nwithout any spectral rescaling. Both conventions yield a nonnegative image;\nonly the first enforces the stronger spectral constraints.")
    source = source.replace("This comparison uses equal gradient work.",
        "The original six-arm comparison uses equal gradient work; the added\nnonnegative baselines charge all their solver operator work.")
    source = replace_once(source, "and~\\ref{fig:ct-trajectories-1080} show progress versus iterations, gradient calls,\nand gradient views.",
        "and~\\ref{fig:ct-trajectories-1080} show progress versus iterations, oracle calls,\nand processed views.")
    findings = []
    for size in (360, 1080):
        methods = summary["cohorts"][str(size)]["methods"]
        values = [methods[m]["raw_loss"]["median_across_groups"] * 1000 for m in ("sframes", "anls", "mu")]
        formatted_size = "360" if size == 360 else "1{,}080"
        findings.append(rf"At \(N={formatted_size}\), raw losses (\(\times10^3\)) are "
            + rf"\({values[0]:.2f}\), \({values[1]:.2f}\), and \({values[2]:.2f}\) for loss-tuned SFRAMES, NNLS, and multiplicative updates.")
    note = "\n".join(findings) + "\nBoth added baselines have zero nonnegativity violation. They provide\nnonnegative-only application benchmarks; differences also reflect the\noptimization domain.\n\n"
    source = source.replace(r"\begin{table}[!htbp]", note + r"\begin{table}[!htbp]", 1)
    source, count = re.subn(r"\\begin\{table\}.*?\\end\{table\}", lambda _: quality_table(summary), source, count=1, flags=re.S)
    require(count == 1, "Missing endpoint table")
    source = replace_once(source, "with the nonnegative orthant. The \\CTReproducibility{} specifies the data split,",
        "with the nonnegative orthant. Alternating NNLS approximately solves each\nnonnegative factor subproblem with L-BFGS-B, using up to " + str(selected_inner) + " accepted\ninner steps per block, selected from \\(5,10,20\\) on the development slices.\nMultiplicative updates use the signed-right-hand-side majorization rule,\nwithout a tuned step size. Neither additional baseline has a spectral cap.\nThe \\CTReproducibility{} specifies the data split,")
    source = source.replace("Gradient-work summaries use the latest recorded iterate", "Work summaries use the latest recorded state")
    source = source.replace("one quarter of the gradient-view budget.",
        "one quarter of the processed-view budget. Iterations count native updates:\nan accepted inner factor step for NNLS and a factor update for multiplicative\nupdates; iteration costs differ between methods.")
    source = source.replace("The three projection baselines\n coincide at zero orthant distance.",
        "The three projection baselines and both additional nonnegative methods\n coincide at zero orthant distance.")
    require(source.count(r"\begin{figure}") == 2 and source.count(r"\begin{table}") == 1, "Figure/table count changed")
    return source


def appendix_text(source, summary, selection):
    require(START not in source, "The nonnegative appendix already exists")
    source = source.replace("The current paper uses the gradient-only view budget described in the final section below.",
        "The current paper adds nonnegative-only baselines at the processed-view budget described in the final section below.")
    extra = ["", START, "## Nonnegative-only baselines", "",
        "The existing six methods and all their recorded values are retained. Alternating NNLS and multiplicative updates minimize the same training loss over nonnegative factors without spectral bounds. Their actual factors are used for raw/evaluated training loss, held-out loss and image-reference error. Former spectral radii are diagnostics only. For the original methods, the established clipping plus optional radial spectral repair remains the evaluation convention: it produces a valid nonnegative candidate with the additional spectral restriction. Raw loss and raw orthant distance always use actual factors. Thus this is an application comparison across different feasible domains, not an isolated solver comparison on the same domain.", "",
        "There are 504 physical evaluation runs: the existing 448 plus 28 deterministic NNLS and 28 deterministic multiplicative runs (14 slices at each view count). The same calibrated initial factors are reused. Deterministic runs are not duplicated to manufacture stochastic repetitions. Each group, its slices, and each slice's physical repetitions retain equal nested weights for means and descriptive standard deviations; endpoint summaries retain nested medians.", "",
        "### Algorithms and selection", "",
        "Alternating NNLS uses SciPy L-BFGS-B for each convex factor subproblem, with nonnegative bounds, memory 10, relative projected-KKT tolerance 1e-3, objective tolerance 1e-9, and at most 20 line-search trials. The accepted-step cap per block is selected from 5, 10, 20 using only the two development slices and both view counts, with 34,560 processed views per direction. Selection minimizes nested-median raw training loss relative to initialization; held-out loss and reference images do not enter selection. The exact selection record is linked below.", "",
        "Multiplicative updates use the signed-right-hand-side majorization rule and alternate the two factors. They have no step-size tuning. Initialization, positivity safeguards, normalizations and stopping details are in the frozen method implementation and campaign notes.", "",
        "### Work and recorded states", "",
        "The cap is 69,120 forward views and 69,120 adjoint views per evaluation run. Every new solver application counts, including NNLS inner evaluations, line-search trials, and the cached adjoint of data for multiplicative updates. Evaluation diagnostics are excluded. Previously reported methods retain gradient views with curvature-estimation work recorded separately; this comparison therefore does not equalize total operator work across all eight methods. The displayed processed-view coordinate is the maximum of the two directional charged counts. Oracle calls count paired forward/adjoint evaluations plus the multiplicative method's standalone startup adjoint; prior arms use their gradient-call count, excluding curvature calls.", "",
        "Native iterations count accepted updates: an accepted inner factor step for NNLS, a factor update for multiplicative updates, and the original update convention for each retained method. This axis does not assign equal computational meaning to an iteration. Cost plots retain all native charged-work observations, including a terminal budget-stop record when unsuccessful trials spend additional work without changing factors. At duplicate coordinates the latest state is used. Mean/SD summaries use last-observation step alignment on fixed method cohorts; no metric interpolation or extrapolation is performed.", "",
        "| Training views | Method | Domain | Raw loss | Evaluated loss | Held-out loss | Raw orthant distance |", "| ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for size in (360, 1080):
        for method in METHODS:
            stats = summary["cohorts"][str(size)]["methods"][method]
            values = [stats[m]["median_across_groups"] for m in ("raw_loss", "feasible_loss", "validation_loss", "relative_d_distance")]
            extra.append(f"| {size} | {VARIANTS[method][0]} | Nonnegative only | " + " | ".join(f"{v:.7g}" for v in values) + " |")
    extra.extend(["", "The [campaign notes](../../nonnegative_baselines/README.md), [selection](../../nonnegative_baselines/selection.json), [completion audit](../../nonnegative_baselines/validation_summary.json), and [publication audit](../../nonnegative_baselines/report_update.json) record all configurations, hashes, work counters and comparison statistics. The preceding gradient-budget publication is preserved in the publication backup; all earlier numerical traces and summaries remain unchanged.", END, ""])
    return source.rstrip() + "\n" + "\n".join(extra)


def destinations(root, report, baseline):
    e = root.parent
    return {**baseline.destinations(report, e / "reports/paper_section/revised", e / "spring_localcurvature", e / "feasibility_retune"),
            **{name + "_report_update.json": e / folder / "report_update.json" for name, folder in
               (("baseline", "baseline_step_rules"), ("trajectory", "trajectory_mean_sd"), ("gradient", "gradient_budget"))}}


def validate_inputs(root, dest):
    validation = read(root / "validation_summary.json")
    plan, protocol = read(root / "evaluation_plan.json"), read(root / "protocol.json")
    require(validation["status"] == "completed_validated" and len(validation["cases"]) == len(plan["runs"]) == 56,
            "Publication requires all 56 validated evaluation runs")
    for name in ("protocol", "evaluation_protocol", "evaluation_plan", "selection"):
        require(validation[name + "_sha256"] == digest(root / (name + ".json")), "Baseline validation anchor changed")
    prior = read(root / "predecessor_publication.json")
    require(read(root.parent / "gradient_budget/report_update.json") == prior,
            "The prepared gradient-budget publication changed")
    frozen = _anchor_map(read(root / "publication_predecessor.json"))
    require(all(digest(path) == sha for path, sha in frozen.items()), "A prepared publication input changed")
    require(all(str(path) in frozen for path in dest.values()), "The predecessor snapshot omits a publication destination")
    fixture = read(root / "report_fixture_validation.json")
    require(fixture["status"] == "passed" and fixture["updater_sha256"] == digest(Path(__file__))
            and fixture["plotting_sha256"] == digest(Path(__file__).with_name("nonnegative_plots.py")), "Report fixture is stale")
    anchors = {str(root / name): digest(root / name) for name in (
        "protocol.json", "evaluation_protocol.json", "evaluation_plan.json", "selection.json", "validation_summary.json",
        "predecessor_publication.json", "publication_predecessor.json", "retained_report_inputs.json", "report_fixture_validation.json")}
    anchors.update({str(Path(__file__)): digest(Path(__file__)),
                    str(Path(__file__).with_name("nonnegative_plots.py")): digest(Path(__file__).with_name("nonnegative_plots.py"))})
    anchors.update(_anchor_map(read(root / "retained_report_inputs.json")))
    anchors.update({path: sha for path, sha in frozen.items() if Path(path) not in dest.values()})
    for case in validation["cases"]:
        for name in ("metadata", "trace", "result"):
            anchors[str(Path(case["source_directory"]) / (name + ".json"))] = case[name + "_sha256"]
    for path in (root.parent / "feasibility_retune/report.py", root.parent / "baseline_step_rules/update_report.py",
                 root.parent / "gradient_budget/plot_trajectories.py"):
        anchors[str(path)] = digest(path)
    return validation, plan, protocol, anchors


def render(runs, slices, groups, output):
    plotting = module("nonnegative_report_plots", Path(__file__).with_name("nonnegative_plots.py"))
    plotting.plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.edgecolor": "0.55", "axes.linewidth": .6, "pdf.fonttype": 42})
    for size in (360, 1080):
        plotting.plot_size(SimpleNamespace(VARIANTS=VARIANTS), runs, slices, groups, size, CAP, CAP // 4, output)


def chain_audits(root, output, report, successor, pages):
    for name, folder in (("feasibility", "feasibility_retune"), ("spring", "spring_localcurvature"),
                         ("baseline", "baseline_step_rules"), ("trajectory", "trajectory_mean_sd"), ("gradient", "gradient_budget")):
        prior = copy.deepcopy(read(root.parent / folder / "report_update.json"))
        prior.update(nonnegative_successor=successor, pages=pages)
        prior["published_sha256"] = {key: digest(output / key) for key in prior["published_sha256"]}
        if name == "feasibility":
            prior["staged_sha256"] = {asset: digest(output / asset) for asset in report.ASSETS}
        write(output / (name + "_report_update.json"), prior)


def stage(root, report, baseline):
    dest = destinations(root, report, baseline)
    validation, plan, protocol, anchors = validate_inputs(root, dest)
    before = {name: digest(path) for name, path in dest.items()}
    runs, previous, joint = retained_runs(root)
    added = added_runs(root, validation, plan, protocol)
    require(not (set(runs) & set(added)), "Added runs overwrite retained methods")
    runs.update(added)
    slices, groups = previous["design"]["evaluation_slices"], previous["design"]["evaluation_slice_groups"]
    summary = summarize(runs, slices, groups, anchors)
    selection = read(root / "selection.json")
    selected_inner = selection["selected_inner_maxiter"]
    output = Path(tempfile.mkdtemp(prefix="report_stage_", dir=root))
    for name in report.ASSETS:
        shutil.copyfile(dest[name], output / name)
    write(output / "nonnegative_comparison_summary.json", summary)
    (output / report.BODY).write_text(source_text(dest[report.BODY].read_text(), summary, selected_inner))
    (output / report.APPENDIX).write_text(appendix_text(dest[report.APPENDIX].read_text(), summary, selection))
    render(runs, slices, groups, output)
    pages = baseline.compile_pdf(output, report)
    successor = dict(publication_audit=str(root / "report_update.json"),
        comparison_summary=str(output / "nonnegative_comparison_summary.json"),
        comparison_summary_sha256=digest(output / "nonnegative_comparison_summary.json"),
        scope="Add nonnegative-only alternating NNLS and multiplicative updates; preserve original six methods.")
    chain_audits(root, output, report, successor, pages)
    require(all(digest(path) == sha for path, sha in anchors.items()), "Report inputs changed while staging")
    require(all(digest(path) == before[name] for name, path in dest.items()), "Live report changed while staging")
    audit = dict(status="staged_validated", generated_utc=datetime.now(timezone.utc).isoformat(),
        stage=str(output), backup=str(root / "report_previous"), report_pdf=str(dest[report.STEM + ".pdf"]),
        physical_runs=504, added_runs=56, figures=2, tables=1, pages=pages, anchors=anchors,
        git_sync_status="pending_export_and_reviewed_commit",
        statistics={f"statistics_{n}.json": digest(output / f"statistics_{n}.json") for n in (360, 1080)},
        original_sha256=before, published_sha256={name: digest(output / name) for name in dest},
        **{key: value for key, value in successor.items() if key != "publication_audit"})
    write(output / "report_validation.json", audit)
    write(root / "staged_report.json", audit)
    return audit


def finish(root, report, audit, dest, pending):
    output, backup = Path(audit["stage"]), Path(audit["backup"])
    require(output.parent == root and output.name.startswith("report_stage_") and backup == root / "report_previous", "Unexpected publication path")
    require(set(dest) == set(audit["published_sha256"]), "Publication destinations changed")
    for name, path in dest.items():
        require(digest(output / name) == audit["published_sha256"][name] and digest(backup / name) == audit["original_sha256"][name], "Stage or backup changed")
        require(digest(path) in (audit["original_sha256"][name], audit["published_sha256"][name]), "Unrelated live edit prevents recovery")
    try:
        for name, path in dest.items():
            report.atomic_copy(output / name, path)
        require(all(digest(path) == audit["published_sha256"][name] for name, path in dest.items()), "Publication mismatch")
        report.atomic_copy(pending, root / "report_update.json")
    except BaseException:
        for name, path in dest.items():
            report.atomic_copy(backup / name, path)
        raise
    pending.unlink(missing_ok=True)
    return audit


def publish(root, report, baseline):
    dest = destinations(root, report, baseline)
    final, pending = root / "report_update.json", root / "report_update.pending.json"
    if final.exists():
        audit = read(final)
        require(all(digest(path) == audit["published_sha256"][name] for name, path in dest.items()), "Published report changed")
        require(digest(audit["comparison_summary"]) == audit["comparison_summary_sha256"], "Published comparison changed")
        return audit
    if pending.exists():
        return finish(root, report, read(pending), dest, pending)
    audit = stage(root, report, baseline)
    backup = root / "report_previous"
    backup.mkdir(exist_ok=True)
    for name, path in dest.items():
        if not (backup / name).exists():
            report.atomic_copy(path, backup / name)
        require(digest(backup / name) == audit["original_sha256"][name], "Backup belongs to another publication")
    require(all(digest(path) == sha for path, sha in audit["anchors"].items()), "Publication input changed")
    audit.update(status="completed_published", published_utc=datetime.now(timezone.utc).isoformat())
    write(pending, audit)
    return finish(root, report, audit, dest, pending)


def fixture_runs():
    """A bounded 504-run fixture, deliberately unrelated to experimental values."""
    slices = [f"fixture_slice_{i:02d}" for i in range(14)]
    groups = {sid: f"group_{min(i // 2, 5)}" for i, sid in enumerate(slices)}
    runs = {}
    for size in (360, 1080):
        for method_index, method in enumerate(VARIANTS):
            seeds = (201,) if method in (*METHODS, "sframes_full") else (202, 203, 204)
            for slice_index, sid in enumerate(slices):
                for seed in seeds:
                    trace = []
                    for step in range(4):
                        accepted = min(step, 2) if method in METHODS else step
                        loss = 0.04 / (1 + accepted * (0.25 + .02 * method_index)) * (1 + slice_index / 70)
                        nonnegative = method in (*METHODS, "palm", "spring_sarah", "prox_sarah")
                        row = dict(iteration=accepted, raw_loss=loss, feasible_loss=loss if nonnegative else loss * 1.05,
                            validation_loss=loss * 1.1, reference_relative_error=.6 / (1 + accepted),
                            d_distance=0. if nonnegative else .1 * accepted, forward_views=step * size,
                            backward_views=step * size, gradient_calls=step, oracle_calls=step)
                        trace.append(row)
                    meta = dict(initial_factor_frobenius_norm=2., config=dict(seed=seed))
                    runs[sid, seed, size, method] = prepare_run(trace, meta, method, "fixture-only")
    return runs, slices, groups


def fixture(root, report, baseline):
    """Render a small synthetic cohort without touching published assets."""
    runs, slices, groups = fixture_runs()
    summary = summarize(runs, slices, groups)
    output = Path(tempfile.mkdtemp(prefix="report_fixture_", dir=root))
    paper = root.parent / "reports/paper_section/revised"
    before = {name: digest(paper / name) for name in report.ASSETS}
    for name in report.ASSETS:
        shutil.copyfile(paper / name, output / name)
    (output / report.BODY).write_text(source_text((paper / report.BODY).read_text(), summary, 10))
    (output / report.APPENDIX).write_text(appendix_text((paper.parent / report.APPENDIX).read_text(), summary, {}))
    render(runs, slices, groups, output)
    pages = baseline.compile_pdf(output, report)
    require(all(digest(paper / name) == sha for name, sha in before.items()), "Fixture changed published assets")
    result = dict(status="passed", fixture_only=True, stage=str(output), report_pdf=str(output / (report.STEM + ".pdf")),
        figures=2, tables=1, pages=pages, updater_sha256=digest(Path(__file__)),
        plotting_sha256=digest(Path(__file__).with_name("nonnegative_plots.py")),
        physical_fixture_runs=len(runs), fixture_observations=sum(len(r["trace"]) for r in runs.values()),
        original_report_sha256=before,
        renderer_predecessor_sha256=digest(root.parent / "gradient_budget/plot_trajectories.py"))
    write(root / "report_fixture_validation.json", result)
    return result


def export_if_configured(root):
    """Run the separately frozen, idempotent repository exporter after publication."""
    descriptor = root / "git_export_helper.json"
    if not descriptor.exists():
        return
    config = read(descriptor)
    require(config.get("dependencies") and isinstance(config.get("command"), list), "Invalid Git-export descriptor")
    require(all(digest(path) == sha for path, sha in config["dependencies"].items()), "Git exporter dependency changed")
    subprocess.run(config["command"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--fixture", action="store_true", help="build a bounded synthetic report without publishing")
    args = parser.parse_args()
    root = args.root.resolve()
    with ExitStack() as stack:
        for folder in (root, *(root.parent / name for name in ("gradient_budget", "trajectory_mean_sd", "baseline_step_rules", "spring_localcurvature", "feasibility_retune"))):
            lock = stack.enter_context((folder / "report_update.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = module("nonnegative_base_report", root.parent / "feasibility_retune/report.py")
        baseline = module("nonnegative_baseline_report", root.parent / "baseline_step_rules/update_report.py")
        require(not (args.fixture and args.publish), "A synthetic fixture cannot be published")
        audit = fixture(root, report, baseline) if args.fixture else publish(root, report, baseline) if args.publish else stage(root, report, baseline)
        if args.publish:
            export_if_configured(root)
        print(json.dumps({key: audit[key] for key in ("status", "stage", "report_pdf", "figures", "tables", "pages")}, indent=2))


if __name__ == "__main__":
    main()
