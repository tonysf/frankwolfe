"""Add validated joint L-BFGS-B runs to the existing CT paper and figures.

Reuse the prior grouping, work alignment, and transactional publisher. Current
editorial prose is the source; predecessor numerical evidence is never rewritten.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import tempfile

try:
    from . import nonnegative_report as base
    from . import nonnegative_plots as plotting
except ImportError:
    import nonnegative_report as base
    import nonnegative_plots as plotting

read, write, digest, require, module = base.read, base.write, base.digest, base.require, base.module
CAP = base.CAP
METHOD = "joint_lbfgsb"
METHODS = (*base.METHODS, METHOD)
VARIANTS = dict(base.VARIANTS, joint_lbfgsb=("Joint L-BFGS-B", "#666600"))
METRICS = base.METRICS
BODY, STEM, APPENDIX = "ct_experiments.tex", "sframes_ct_section", "CT_REPRODUCIBILITY.md"
ASSETS = (BODY, STEM + ".tex", "ct_reproducibility_link.tex",
          "trajectories_360.pdf", "trajectories_360.png",
          "trajectories_1080.pdf", "trajectories_1080.png", STEM + ".pdf")
START, END = "<!-- CT_JOINT_LBFGSB_START -->", "<!-- CT_JOINT_LBFGSB_END -->"


def prepare_run(trace, meta, method, source_directory):
    # The joint method has the same all-operator accounting and exact orthant
    # feasibility as ANLS; no change to the historical helper's globals.
    run = base.prepare_run(trace, meta, "anls" if method == METHOD else method, source_directory)
    run["method"] = method
    return run


def retained_runs(root):
    prior = read(root / "predecessor_publication.json")
    summary_path = Path(prior["comparison_summary"])
    require(digest(summary_path) == prior["comparison_summary_sha256"], "Predecessor summary changed")
    previous = read(summary_path)
    frozen = base._anchor_map(read(root / "retained_report_inputs.json"))
    runs = {}
    for case in previous["cases"]:
        directory = Path(case["source_directory"])
        for name in ("metadata.json", "trace.json"):
            p = directory / name
            require(frozen.get(str(p)) == digest(p), "Retained trajectory changed")
        key = case["slice_id"], int(case["seed"]), int(case["n_train"]), case["method"]
        require(key not in runs and key[3] in base.VARIANTS, "Unexpected retained physical run")
        runs[key] = prepare_run(read(directory / "trace.json"), read(directory / "metadata.json"), key[3], directory)
    require(len(runs) == 504, "Expected 504 retained physical runs")
    return runs, previous


def added_runs(root, validation, plan, protocol):
    campaign = module("joint_report_campaign", root / "frozen_sources/joint_lbfgsb_campaign.py")
    campaign.verify_prepared(root=root)
    configs = {cfg["run_id"]: cfg for cfg in plan["runs"]}
    runs = {}
    for case in validation["cases"]:
        cfg = configs[case["run_id"]]
        checked = campaign.validate_result(cfg, plan["assignments"][case["run_id"]], plan, protocol, root=root)
        require(checked == case, "Joint run changed after validation")
        key = case["slice_id"], int(cfg["calibration_seed"]), int(case["n_train"]), case["method"]
        require(key not in runs and key[3] == METHOD, "Unexpected added physical run")
        directory = Path(case["source_directory"])
        runs[key] = prepare_run(read(directory / "trace.json"), read(directory / "metadata.json"), METHOD, directory)
    require(len(runs) == 28, "Expected 28 validated joint evaluations")
    return runs


def summarize(runs, slices, groups, anchors=None):
    require(len(runs) == 532, "The nine-arm comparison requires 532 physical runs")
    result = dict(schema_version=1, status="completed_validated", budget_per_direction=CAP,
        budget_axis="maximum directional solver views; prior curvature-estimation work excluded",
        physical_runs=len(runs), added_runs=28, figures=2, tables=1, anchors=anchors or {},
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
        cohort = {}
        for method in VARIANTS:
            chosen = [c for c in result["cases"] if c["n_train"] == size and c["method"] == method]
            require({c["slice_id"] for c in chosen} == set(slices), "Incomplete specimen cohort")
            expected = len(slices) if method in (*METHODS, "sframes_full") else 3 * len(slices)
            require(len(chosen) == expected, "Physical repetitions changed")
            metrics = (*METRICS, "iteration", "comparison_views", "comparison_oracle_calls", "forward_views", "backward_views")
            cohort[method] = {metric: base.grouped([(c["slice_id"], c["endpoint"][metric]) for c in chosen], groups)
                              for metric in metrics}
        references = {c["slice_id"]: c for c in result["cases"] if c["method"] == METHOD and c["n_train"] == size}
        paired = {}
        for method in ("sframes", "tuned_sframes", "anls", "mu"):
            chosen = [c for c in result["cases"] if c["method"] == method and c["n_train"] == size]
            paired[method] = {metric: base.grouped([(c["slice_id"],
                100 * (c["endpoint"][metric] / references[c["slice_id"]]["endpoint"][metric] - 1))
                for c in chosen], groups) for metric in ("raw_loss", "feasible_loss")}
        result["cohorts"][str(size)] = dict(methods=cohort, methods_vs_joint_lbfgsb=paired)
    return result


def quality_table(summary):
    # Retain the previous table formatting while supplying the new method set.
    text = r"""\begin{table}[!htbp]
\centering
\caption{Endpoint quality at the common processed-view cap. Domains differ: the original methods target \(C\cap D\); NNLS, multiplicative updates and joint L-BFGS-B target only \(D\). Losses are multiplied by \(10^3\). Eval., held-out loss and reference error use the evaluated nonnegative image; see the repair convention above. Raw infeasibility is relative to the initial factor-pair norm, in percent. Bold marks the best value per view count across both domains.}
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
        best = {m: min(methods[k][m]["median_across_groups"] for k in VARIANTS) for m in METRICS}
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


def source_text(source, summary, selected_memory):
    gradient = re.search(r"\\paragraph\{Stochastic gradient estimator\.\}.*?(?=\\paragraph\{Evaluation\.\})", source, re.S)
    require(gradient is not None, "The current gradient-estimator explanation is missing")
    original_gradient = gradient.group()
    source = base.replace_once(source, "least squares (NNLS) and multiplicative updates without spectral bounds.",
        "least squares (NNLS), multiplicative updates, and joint L-BFGS-B without spectral bounds.")
    source = base.replace_once(source, "The additional NNLS and multiplicative baselines minimize the same loss",
        "These three additional baselines minimize the same loss")
    source = base.replace_once(source, "The original methods count gradient views; NNLS and\nmultiplicative updates count every solver forward/adjoint application,",
        "The original methods count gradient views; all three additional\nnonnegative baselines count every solver forward/adjoint application,")
    source = base.replace_once(source, "and rescales it radially into its spectral ball if needed. NNLS and\nmultiplicative updates are evaluated at their actual nonnegative factors,",
        "and rescales it radially into its spectral ball if needed. The three\nnonnegative-only baselines are evaluated at their actual factors,")
    values = [summary["cohorts"][str(size)]["methods"][METHOD]["raw_loss"]["median_across_groups"] * 1000
              for size in (360, 1080)]
    pattern = r"At \\\(N=360\\\), raw losses.*?optimization domain\."
    replacement = (rf"Joint L-BFGS-B gives raw/evaluated losses of \({values[0]:.2f}\) and \({values[1]:.2f}\) "
        + r"(\(\times10^3\)) at \(N=360,1{,}080\), respectively. Like NNLS and multiplicative"
        + "\nupdates, it remains nonnegative. These methods optimize over \\(D\\) alone,\nso quality differences also reflect the optimization domain.")
    source, count = re.subn(pattern, lambda _: replacement, source, count=1, flags=re.S)
    require(count == 1, "The predecessor nonnegative findings changed")
    source, count = re.subn(r"\\begin\{table\}.*?\\end\{table\}", lambda _: quality_table(summary), source, count=1, flags=re.S)
    require(count == 1, "Missing endpoint table")
    source = base.replace_once(source, "without a tuned step size. Neither additional baseline has a spectral cap.",
        "without a tuned step size. Joint L-BFGS-B updates both factors together,\nretaining one curvature history with memory " + str(selected_memory)
        + ", selected from \\(5,10,20\\)\non the same development slices and view budgets. None of these three\nadditional baselines has a spectral cap.")
    source = base.replace_once(source, "an accepted inner factor step for NNLS and a factor update for multiplicative\nupdates; iteration costs differ between methods.",
        "an accepted inner factor step for NNLS, a factor update for multiplicative\nupdates, or a joint factor update for L-BFGS-B; iteration costs differ between methods.")
    source = base.replace_once(source, "The three projection baselines and both additional nonnegative methods",
        "The three projection baselines and all three additional nonnegative methods")
    require(original_gradient in source, "Gradient-estimator prose changed")
    require(source.count(r"\begin{figure}") == 2 and source.count(r"\begin{table}") == 1, "Figure/table count changed")
    return source


def appendix_text(source, summary, selection):
    require(START not in source, "Joint appendix already exists")
    extra = ["", START, "## Joint nonnegative L-BFGS-B", "",
        "The current comparison contains 532 physical trajectories: all previous 504 plus 28 deterministic joint L-BFGS-B runs, one for each of the 14 evaluation slices and two view counts. Its feasible set is nonnegative factors only. The same frozen rank-64 initial factors, data and view split are reused; there is no spectral bound, radial repair, factor balancing or regularizer. All prior numerical evidence remains unchanged.", "",
        "SciPy 1.18.1 L-BFGS-B operates on the concatenated U,V vector and retains its limited-memory state across joint accepted updates. This differs from alternating NNLS, whose optimizer is restarted for each factor subproblem. Each joint evaluation uses one shared residual/backprojection to obtain both factor gradients. Every trial forward/adjoint pair counts, including unsuccessful line-search trials. Loss and feasibility are recorded at every accepted state, with terminal unaccepted work retained as a repeated native iteration. Checkpoints preserve the complete reverse-communication state.", "",
        "The memory is selected from {5,10,20} by twelve development runs (two slices, two view counts) at 34,560 views per direction. The selection criterion is the median relative raw endpoint loss across slices at each view count, then the median over view counts; exact ties favor lower memory. Fixed settings are relative projected-gradient tolerance 1e-3, objective tolerance 1e-9, and at most 20 line-search trials. The resulting setting is frozen before evaluation; held-out images are not used for selection.", "",
        "Evaluation stops within 69,120 views separately in the forward and adjoint directions. Only numerical stopping conditions or the budget can end the joint solve; there is no inner-iteration restart cap. Diagnostics remain outside solver work. Earlier spectral-constrained methods retain their gradient-only budget and separate curvature accounting. Consequently the comparison does not equalize total operator work across all nine methods. Native joint iterations are accepted updates of both factors; oracle calls and views retain their actual charged costs. The existing group-balanced means ±1 descriptive SD and nested-median endpoint statistics are unchanged.", "",
        "| Training views | Raw/evaluated loss | Held-out loss | Relative orthant distance |", "| ---: | ---: | ---: | ---: |"]
    for size in (360, 1080):
        metrics = summary["cohorts"][str(size)]["methods"][METHOD]
        extra.append(f"| {size} | " + " | ".join(f"{metrics[m]['median_across_groups']:.7g}"
            for m in ("raw_loss", "validation_loss", "relative_d_distance")) + " |")
    extra += ["", "The [campaign notes](../../joint_lbfgsb/README.md), [selection](../../joint_lbfgsb/selection.json), [completion audit](../../joint_lbfgsb/validation_summary.json), and [publication audit](../../joint_lbfgsb/report_update.json) retain the exact configuration and results. The previous scientific publication and the subsequent gradient-estimator editorial revision are preserved in the predecessor records and document backup.", END, ""]
    return source.rstrip() + "\n" + "\n".join(extra)


def destinations(root):
    paper = root.parent / "reports/paper_section/revised"
    return {**{name: paper / name for name in ASSETS}, APPENDIX: paper.parent / APPENDIX}


def validate_inputs(root, dest):
    validation = read(root / "validation_summary.json")
    plan, protocol = read(root / "evaluation_plan.json"), read(root / "protocol.json")
    require(validation["status"] == "completed_validated" and len(validation["cases"]) == len(plan["runs"]) == 28,
            "Publication requires all 28 validated joint evaluations")
    for name in ("protocol", "evaluation_protocol", "evaluation_plan", "selection"):
        require(validation[name + "_sha256"] == digest(root / (name + ".json")), "Validation anchor changed")
    frozen = base._anchor_map(read(root / "publication_predecessor.json"))
    require(all(digest(path) == sha for path, sha in frozen.items()), "Prepared publication input changed")
    require(all(str(path) in frozen for path in dest.values()), "Missing predecessor destination")
    fixture = read(root / "report_fixture_validation.json")
    require(fixture["status"] == "passed" and fixture["updater_sha256"] == digest(Path(__file__))
        and fixture["plotting_sha256"] == digest(Path(plotting.__file__))
        and fixture["base_helper_sha256"] == digest(Path(base.__file__)), "Report fixture is stale")
    anchors = {str(root / name): digest(root / name) for name in (
        "protocol.json", "evaluation_protocol.json", "evaluation_plan.json", "selection.json", "validation_summary.json",
        "predecessor_publication.json", "publication_predecessor.json", "retained_report_inputs.json", "report_fixture_validation.json")}
    anchors.update({str(Path(__file__)): digest(Path(__file__)), str(Path(plotting.__file__)): digest(Path(plotting.__file__)),
                    str(Path(base.__file__)): digest(Path(base.__file__))})
    anchors.update(base._anchor_map(read(root / "retained_report_inputs.json")))
    anchors.update({path: sha for path, sha in frozen.items() if Path(path) not in dest.values()})
    for case in validation["cases"]:
        for name in ("metadata", "trace", "result"):
            anchors[str(Path(case["source_directory"]) / (name + ".json"))] = case[name + "_sha256"]
    return validation, plan, protocol, anchors


def render(runs, slices, groups, output):
    plotting.plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.edgecolor": "0.55", "axes.linewidth": .6, "pdf.fonttype": 42})
    for size in (360, 1080):
        plotting.plot_size(SimpleNamespace(VARIANTS=VARIANTS, legend_columns=5), runs, slices, groups, size, CAP, CAP // 4, output)


def stage(root, report, compiler):
    dest = destinations(root)
    validation, plan, protocol, anchors = validate_inputs(root, dest)
    before = {name: digest(path) for name, path in dest.items()}
    runs, previous = retained_runs(root)
    added = added_runs(root, validation, plan, protocol)
    require(not (set(runs) & set(added)), "New method overwrites a retained run")
    runs.update(added)
    cohort = read(root / "cohort.json")
    slices, groups = cohort["evaluation_slices"], cohort["evaluation_slice_groups"]
    summary = summarize(runs, slices, groups, anchors)
    # Verify all existing endpoint aggregates were retained exactly.
    for size in ("360", "1080"):
        for method in base.VARIANTS:
            require(summary["cohorts"][size]["methods"][method] == previous["cohorts"][size]["methods"][method],
                    "A predecessor endpoint statistic changed")
    selection = read(root / "selection.json")
    output = Path(tempfile.mkdtemp(prefix="report_stage_", dir=root))
    for name in ASSETS:
        shutil.copyfile(dest[name], output / name)
    write(output / "joint_lbfgsb_comparison_summary.json", summary)
    (output / BODY).write_text(source_text(dest[BODY].read_text(), summary, selection["selected_memory"]))
    (output / APPENDIX).write_text(appendix_text(dest[APPENDIX].read_text(), summary, selection))
    render(runs, slices, groups, output)
    pages = compiler.compile_pdf(output, report)
    require(all(digest(path) == sha for path, sha in anchors.items()), "Report input changed while staging")
    require(all(digest(path) == before[name] for name, path in dest.items()), "Live document changed while staging")
    audit = dict(status="staged_validated", generated_utc=datetime.now(timezone.utc).isoformat(),
        stage=str(output), backup=str(root / "report_previous"), report_pdf=str(dest[STEM + ".pdf"]),
        physical_runs=532, added_runs=28, figures=2, tables=1, pages=pages, anchors=anchors,
        git_sync_status="pending_export_and_reviewed_commit", comparison_summary=str(output / "joint_lbfgsb_comparison_summary.json"),
        comparison_summary_sha256=digest(output / "joint_lbfgsb_comparison_summary.json"),
        statistics={f"statistics_{n}.json": digest(output / f"statistics_{n}.json") for n in (360, 1080)},
        original_sha256=before, published_sha256={name: digest(output / name) for name in dest},
        scope="Add joint nonnegative L-BFGS-B; retain eight prior methods and the gradient-estimator explanation.")
    write(output / "report_validation.json", audit)
    write(root / "staged_report.json", audit)
    return audit


def publish(root, report, compiler):
    dest = destinations(root)
    final, pending = root / "report_update.json", root / "report_update.pending.json"
    if final.exists():
        audit = read(final)
        require(all(digest(path) == audit["published_sha256"][name] for name, path in dest.items()), "Published document changed")
        require(digest(audit["comparison_summary"]) == audit["comparison_summary_sha256"], "Published comparison changed")
        return audit
    if pending.exists():
        return base.finish(root, report, read(pending), dest, pending)
    audit = stage(root, report, compiler)
    backup = root / "report_previous"
    backup.mkdir(exist_ok=True)
    for name, path in dest.items():
        if not (backup / name).exists():
            report.atomic_copy(path, backup / name)
        require(digest(backup / name) == audit["original_sha256"][name], "Publication backup changed")
    require(all(digest(path) == sha for path, sha in audit["anchors"].items()), "Publication input changed")
    audit.update(status="completed_published", published_utc=datetime.now(timezone.utc).isoformat())
    write(pending, audit)
    return base.finish(root, report, audit, dest, pending)


def fixture_runs():
    runs, slices, groups = base.fixture_runs()
    for (sid, seed, size, method), run in list(runs.items()):
        if method == "anls":
            clone = base.copy.deepcopy(run)
            clone["method"] = METHOD
            clone["source_directory"] = "fixture-only"
            runs[sid, seed, size, METHOD] = clone
    return runs, slices, groups


def fixture(root, report, compiler):
    runs, slices, groups = fixture_runs()
    summary = summarize(runs, slices, groups)
    output = Path(tempfile.mkdtemp(prefix="report_fixture_", dir=root))
    dest = destinations(root)
    before = {name: digest(path) for name, path in dest.items()}
    for name in ASSETS:
        shutil.copyfile(dest[name], output / name)
    (output / BODY).write_text(source_text(dest[BODY].read_text(), summary, 10))
    (output / APPENDIX).write_text(appendix_text(dest[APPENDIX].read_text(), summary, {}))
    render(runs, slices, groups, output)
    pages = compiler.compile_pdf(output, report)
    require(all(digest(path) == before[name] for name, path in dest.items()), "Fixture changed a published document")
    result = dict(status="passed", fixture_only=True, stage=str(output), report_pdf=str(output / (STEM + ".pdf")),
        figures=2, tables=1, pages=pages, physical_fixture_runs=len(runs),
        updater_sha256=digest(Path(__file__)), plotting_sha256=digest(Path(plotting.__file__)),
        base_helper_sha256=digest(Path(base.__file__)), original_report_sha256=before)
    write(root / "report_fixture_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for folder in (root, root.parent / "nonnegative_baselines", root.parent / "reports/paper_section/revised"):
            lock = stack.enter_context((folder / "report_update.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = module("joint_base_report", root.parent / "feasibility_retune/report.py")
        compiler = module("joint_compiler", root.parent / "baseline_step_rules/update_report.py")
        require(not (args.fixture and args.publish), "A fixture cannot be published")
        result = fixture(root, report, compiler) if args.fixture else publish(root, report, compiler) if args.publish else stage(root, report, compiler)
        if args.publish:
            base.export_if_configured(root)
        print(base.json.dumps({k: result[k] for k in ("status", "stage", "report_pdf", "figures", "tables", "pages")}, indent=2))


if __name__ == "__main__":
    main()
