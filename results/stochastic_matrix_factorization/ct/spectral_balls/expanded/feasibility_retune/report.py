"""Build the compact CT paper, then add validated joint tuning results."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "convincing_4x_recovery"
ARCHIVE = ROOT.parent / "reports" / "paper_section"
PAPER = ARCHIVE / "revised"
LINK = "ct_reproducibility_link.tex"
APPENDIX = "CT_REPRODUCIBILITY.md"
BODY = "ct_experiments.tex"
STEM = "sframes_ct_section"
FIGURES = ("trajectories_360", "trajectories_1080", "joint_selection")
ASSETS = (BODY, f"{STEM}.tex", LINK, *(f"{figure}.{suffix}" for figure in FIGURES for suffix in ("pdf", "png")),
          f"{STEM}.pdf")
LABELS = {
    "sframes": "Minibatch SFRAMES (original)",
    "tuned_sframes": "Minibatch SFRAMES (joint tune)",
    "sframes_full": "Full-batch SFRAMES",
    "spring_sarah": "SPRING--SARAH",
    "prox_sarah": "ProxSARAH",
    "palm": "PALM",
}
ORIGINAL = (16, 0.1)
TABLE_METRICS = (
    ("raw_loss", "Raw", ".2f"), ("feasible_loss", "Repaired", ".2f"),
    ("validation_loss", "Held-out", ".2f"),
    ("reference_relative_error", "Ref. error", ".3f"),
    ("relative_d_distance", r"Infeas.\ (\%)", ".2f"),
)


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


def pair(setting):
    return (setting["beta_scale"], setting["gamma0"])


def fmt_pair(params):
    return f"({params[0]:g},{params[1]:g})"


def relative_distances(previous):
    """Normalize each saved endpoint before the nested median aggregation."""
    grouped = {}
    norms = {}
    for case in previous["cases"]:
        for method, arm in case["arms"].items():
            if method not in LABELS:
                continue
            folder = Path(arm["source_directory"])
            if folder not in norms:
                meta = read(folder / "metadata.json")
                norms[folder] = meta.get("initial_factor_frobenius_norm", meta.get("initial_factor_norm"))
                if not norms[folder] or not math.isfinite(norms[folder]):
                    raise ValueError(f"Missing initial factor norm: {folder}")
            value = arm["probes"]["view_cap"]["d_distance"] / norms[folder]
            key = (str(case["n_train"]), method)
            grouped.setdefault(key, {}).setdefault(case["group"], {}).setdefault(
                case["slice_id"], []).append(value)
    return {key: statistics.median(statistics.median(statistics.median(seeds)
                for seeds in slices.values()) for slices in groups.values())
            for key, groups in grouped.items()}


def quality_table(legacy, previous, summary=None):
    rows = []
    distances = relative_distances(previous)
    for n, cohort in previous["cohorts"].items():
        reference = cohort["view_cap"]["matched_all_methods"]["median_of_group_medians"]
        values = {method: dict(metrics, relative_d_distance=distances[str(n), method])
                  for method, metrics in reference.items()}
        if summary is not None:
            matched = summary["by_n"][n]["methods"]
            values["tuned_sframes"] = {
                metric: matched["tuned_sframes"]["endpoint"][metric]["median_across_groups"]
                for metric, _, _ in legacy.METRICS
            }
            values["tuned_sframes"]["relative_d_distance"] = matched["tuned_sframes"]["endpoint"][
                "relative_d_distance"]["median_across_groups"]
            for method in ("sframes", "prox_sarah"):
                for metric, _, _ in legacy.METRICS:
                    recomputed = matched[method]["endpoint"][metric]["median_across_groups"]
                    if not math.isclose(recomputed, reference[method][metric], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError(f"Follow-up references disagree with the original table: {n}, {method}, {metric}")
        for metrics in values.values():
            for name in ("raw_loss", "feasible_loss", "validation_loss"):
                metrics[name] *= 1000
            metrics["relative_d_distance"] *= 100
        minima = {metric: min(values[method][metric] for method in LABELS if method in values)
                  for metric, _, _ in TABLE_METRICS}
        for method, name in LABELS.items():
            if method not in values:
                continue
            if method == "sframes" and summary is None:
                name = "Minibatch SFRAMES"
            rows.append((legacy.integer(n), name, *(legacy.numeric(values[method][metric], style,
                                                                   values[method][metric] == minima[metric])
                                                     for metric, _, style in TABLE_METRICS)))
    return legacy.table(
        "llrrrrr", (r"$N$", "Method", *(name for _, name, _ in TABLE_METRICS)), rows,
        r"Endpoint quality at a common view budget. The three loss columns are multiplied by \(10^3\). "
        r"Held-out loss and reference error use the repaired image. Infeasibility is relative to "
        r"the initial factor-pair norm, in percent. Bold marks the best value within each view count.",
        "tab:ct-results",
    )


def render_source(previous, legacy, selection=None, summary=None):
    setting = "The loss-tuned" if selection is not None else "The"
    tuning = setting + r""" SFRAMES setting \((c_\beta,\gamma_0)=(16,0.1)\) is selected
jointly by repaired training loss on two development slices.
The baseline searches use the same slices and criterion, with
\(34{,}560\) views per run versus \(8{,}640\) for this SFRAMES search."""
    baseline_name = "The loss-tuned minibatch setting" if selection is not None else "Minibatch SFRAMES"
    effects = {m: [previous["cohorts"][n]["view_cap"]["sframes_vs"][m]["metrics"][
                   "feasible_loss"]["percent_difference"]["median_across_groups"]
                  for n in ("360", "1080")] for m in ("prox_sarah", "sframes_full")}
    results = baseline_name + rf""" gives lower repaired training loss than ProxSARAH
in all six group summaries. The median paired reductions are
\({-effects['prox_sarah'][0]:.1f}\%\) for \(N=360\) and
\({-effects['prox_sarah'][1]:.1f}\%\) for \(N=1{{,}}080\);
relative to full-batch SFRAMES they are \({-effects['sframes_full'][0]:.1f}\%\)
and \({-effects['sframes_full'][1]:.1f}\%\), respectively.
The held-out loss also favors the minibatch setting, although at \(N=360\)
the full-batch variant has lower reference image error.

At this loss-tuned setting, relative raw infeasibility for minibatch
SFRAMES is about \(2\%\) at both view counts.
Neither loss-tuned SFRAMES variant finishes below \(1\%\) in any run.
The proximal baselines remain nonnegative throughout.
Thus the lower loss of repaired SFRAMES outputs comes with a persistent
raw-feasibility residual at this budget."""
    if selection is not None:
        if summary is None:
            raise ValueError("A tuned report needs the paired comparison")
        chosen = fmt_pair(pair(selection["selected"]))
        tuning += "\n" + rf"""A second minibatch sweep at \(34{{,}}560\) views selected
\((c_\beta,\gamma_0)={chosen}\) using the joint endpoint target of repaired
loss at most \(30\%\) of its initial value and relative orthant distance
at most \(1\%\). The evaluation cohort did not enter selection."""
        sizes = sorted(summary["by_n"], key=int)
        def rate(n, method, target):
            value = summary["by_n"][n]["methods"][method]["joint_targets"][
                target]["endpoint_group_balanced_fraction"]
            return rf"\({100 * value:.0f}\%\)"

        def effect(n, method):
            value = summary["by_n"][n]["pairs"][method]["repaired_loss_pct"]["median_across_groups"]
            if abs(value) < 0.05:
                return "unchanged to the reported precision"
            return rf"\({abs(value):.1f}\%\) " + ("lower" if value < 0 else "higher")

        results += "\n\n" + rf"""Relative to ProxSARAH, the joint tune's paired repaired loss is
{effect('360', 'prox_sarah')} at \(N=360\) and
{effect('1080', 'prox_sarah')} at \(N=1{{,}}080\)."""
        for n in sizes:
            results += "\n" + rf"""At \(N={legacy.integer(n)}\), the group-balanced fraction of endpoints
meeting both the \(30\%\) relative-loss and \(1\%\) raw-feasibility targets is
{rate(n, 'tuned_sframes', '0.3:0.01')} for the joint tune, compared with
{rate(n, 'sframes', '0.3:0.01')} for the loss-tuned minibatch setting and
{rate(n, 'prox_sarah', '0.3:0.01')} for ProxSARAH."""
        results += "\n" + rf"""At the stricter \(0.1\%\) feasibility tolerance, the joint tune
attains both targets in {rate('360', 'tuned_sframes', '0.3:0.001')} and
{rate('1080', 'tuned_sframes', '0.3:0.001')} of cases after group balancing,
respectively. Complete endpoint and first-hit comparisons are in the appendix."""
    source = Path(__file__).with_name("compact_paper.tex").read_text()
    for token, content in (("@@TUNING_TEXT@@", tuning), ("@@RESULTS_TEXT@@", results),
                           ("@@RESULTS_TABLE@@", quality_table(legacy, previous, summary))):
        if source.count(token) != 1:
            raise ValueError(f"Missing or ambiguous compact paper placeholder: {token}")
        source = source.replace(token, content)
    if source.count(r"\begin{figure}") != 2 or source.count(r"\begin{table}") != 1:
        raise ValueError("The compact paper must have two figures and one table")
    if r"d_C" in source or "@@" in source:
        raise ValueError("The compact paper contains unwanted columns or unresolved placeholders")
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", source))
    if refs - labels:
        raise ValueError(f"Unresolved source references: {refs - labels}")
    return source


def plot_development(selection, stage):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    scores = selection["scores"]
    selected_pair = pair(selection["selected"])
    selected = next(s for s in scores if pair(s) == selected_pair)
    anchor = next(s for s in scores if pair(s) == ORIGINAL)
    n = scores[0]["n_cases"]
    if n != 12 or len(scores) != 13 or any(s["n_cases"] != n for s in scores):
        raise ValueError("Joint development plot requires all 13 settings and 12 paired cases")
    x = lambda s: 100 * s["median_relative_d_distance"]
    y = lambda s: 100 * s["median_relative_loss"]
    fig, ax = plt.subplots(figsize=(5.65, 2.25), layout="constrained")
    cloud = ax.scatter([x(s) for s in scores], [y(s) for s in scores],
                       c=[s["primary_hits"] for s in scores], cmap="viridis", vmin=0, vmax=n,
                       s=31, linewidth=0.4, edgecolor="0.24", zorder=3)
    ax.scatter(x(selected), y(selected), facecolors="none", edgecolors="0.05",
               s=113, linewidth=1.15, marker="o", label=f"Selected {fmt_pair(selected_pair)}", zorder=4)
    ax.scatter(x(anchor), y(anchor), c="0.07", s=59, marker="+", linewidth=1.35,
               label=f"Original {fmt_pair(ORIGINAL)}", zorder=5)
    ax.axvline(1, color="0.40", linewidth=0.65, linestyle="--")
    ax.axhline(30, color="0.40", linewidth=0.65, linestyle="--")
    if min(x(s) for s in scores) > 0:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Raw orthant distance / initial factor norm (%)")
    ax.set_ylabel("Repaired loss / initial loss (%)")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.17)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    fig.colorbar(cloud, ax=ax, pad=0.03, ticks=[0, 3, 6, 9, 12]).set_label("Joint endpoints (of 12)")
    fig.savefig(stage / "joint_selection.pdf")
    fig.savefig(stage / "joint_selection.png", dpi=190)
    plt.close(fig)


def atomic_copy(source, destination):
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                     delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        shutil.copyfile(source, pending)
        pending.chmod(0o644)
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def compile_pdf(stage):
    with (stage / "build.log").open("w") as log:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{STEM}.tex"],
                       cwd=stage, check=True, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(["pdftotext", "-layout", f"{STEM}.pdf", "paper.txt"], cwd=stage, check=True)
    extracted = (stage / "paper.txt").read_text()
    if any(token not in extracted for token in ("ProxSARAH", "Figure 2:", "Table 1:", "reproducibility appendix")):
        raise ValueError("The compiled paper is missing the new comparison or an existing figure/table")
    if "??" in extracted or "Figure 3:" in extracted or "Table 2:" in extracted:
        raise ValueError("Unresolved references or an added figure/table appeared in the compiled paper")
    if r"Overfull \hbox" in (stage / f"{STEM}.log").read_text():
        raise ValueError("The compiled paper has overfull text")
    info = subprocess.check_output(["pdfinfo", str(stage / f"{STEM}.pdf")], text=True)
    return int(re.search(r"^Pages:\s*(\d+)$", info, flags=re.M).group(1))


def archived_inputs():
    legacy = load_module("legacy_ct_report", LEGACY / "report_update.py")
    previous = read(LEGACY / "reports" / "summary.json")
    audit = read(LEGACY / "reports" / "report_update.json")
    inputs = {
        LEGACY / "reports" / "summary.json": audit["summary_sha256"],
        ARCHIVE / BODY: audit["report_source_sha256"],
        ARCHIVE / f"{STEM}.pdf": audit["report_pdf_sha256"],
    }
    if any(digest(path) != value for path, value in inputs.items()):
        raise ValueError("The archived comparison differs from its publication audit")
    return legacy, previous, audit


def build_current(root, paper, publish=False):
    legacy, previous, archive = archived_inputs()
    audit_path = root / "current_report.json"
    if publish and audit_path.exists():
        audit = read(audit_path)
        if (Path(audit["report_pdf"]) != paper / f"{STEM}.pdf"
                or any(digest(paper / name) != sha for name, sha in audit["published_sha256"].items())):
            raise ValueError("The compact edition has changed since its initial publication")
        if (root / "report_update.json").exists():
            raise ValueError("The validated joint tune has already been published")
    if publish and (paper == ARCHIVE or not (paper.parent / APPENDIX).is_file()):
        raise ValueError("Publish alongside the reproducibility appendix, without overwriting the archived edition")
    if publish and not audit_path.exists() and (paper / f"{STEM}.pdf").exists():
        raise ValueError("The destination already contains a PDF")
    stage = Path(tempfile.mkdtemp(prefix="report_stage_", dir=root))
    (stage / BODY).write_text(render_source(previous, legacy))
    driver = (ARCHIVE / f"{STEM}.tex").read_text()
    driver = driver.replace(r"\begin{document}", r"\input{ct_reproducibility_link.tex}" + "\n" + r"\begin{document}", 1)
    driver = driver.replace(r"\clearpage" + "\n" + r"\begin{thebibliography}",
                            r"\begin{thebibliography}", 1)
    driver, removed = re.subn(r"\\bibitem\{deepinv\}.*?(?=\\bibitem\{spring\})", "", driver, flags=re.S)
    if removed != 1:
        raise ValueError("Expected the archived DeepInverse and ASTRA implementation references")
    (stage / f"{STEM}.tex").write_text(driver)
    (stage / LINK).write_text(
        "% At release, replace the relative target with the public GitHub file URL.\n"
        r"\newcommand{\CTReproducibilityURL}{../CT_REPRODUCIBILITY.md}" "\n"
        r"\newcommand{\CTReproducibility}{\href{\CTReproducibilityURL}{reproducibility appendix}}" "\n"
    )
    plotting = load_module("ct_current_trajectories", Path(__file__).with_name("report_trajectories.py"))
    n_trajectories, n_observations = plotting.render(LEGACY, None, stage)
    for suffix in ("pdf", "png"):
        name = f"joint_selection.{suffix}"
        shutil.copyfile(ARCHIVE / name, stage / name)
    pages = compile_pdf(stage)
    outputs = {name: digest(stage / name) for name in ASSETS}
    audit = dict(status="current_results_compact", stage=str(stage), pages=pages, figures=2, tables=1,
                 report_pdf=str(paper / f"{STEM}.pdf"), legacy_summary_sha256=archive["summary_sha256"],
                 plotting_source_sha256=digest(Path(__file__).with_name("report_trajectories.py")),
                 evaluation_runs=n_trajectories, native_iteration_observations=n_observations,
                 published_sha256=outputs)
    (stage / "report_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    if publish:
        paper.mkdir(parents=True, exist_ok=True)
        for name in ASSETS:
            atomic_copy(stage / name, paper / name)
        atomic_copy(stage / "report_validation.json", audit_path)
    return audit


def supplement_results(paper, selection, summary):
    path = paper.parent / APPENDIX
    if not path.is_file():
        return
    source = path.read_text()
    start = "<!-- CT_FEASIBILITY_RETUNE_RESULT_START -->"
    end = "<!-- CT_FEASIBILITY_RETUNE_RESULT_END -->"
    if source.count(start) != 1 or source.count(end) != 1 or source.index(start) > source.index(end):
        raise ValueError("The reproducibility appendix is missing its joint-tune result markers")
    chosen = selection["selected"]
    report = [
        "",
        f"The joint sweep selected **(c_beta, gamma_0) = {fmt_pair(pair(chosen))}**. "
        f"It met the primary endpoint target in {chosen['primary_hits']}/{chosen['n_cases']} development runs.",
        f"The [development tradeoff plot]({paper.name}/joint_selection.png) shows all 13 settings.",
        "",
        "Each entry gives the number of final iterates meeting both targets, followed by the number of "
        "runs meeting both at any recorded iterate. There are 42 paired cases per training-view count.",
        "",
        "| Training views | Repaired loss / initial loss | Relative raw distance | Joint tune | Original SFRAMES | ProxSARAH |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n, by_n in sorted(summary["by_n"].items(), key=lambda item: int(item[0])):
        for key in ("0.3:0.01", "0.3:0.001", "0.4:0.01", "0.4:0.001"):
            loss, distance = map(float, key.split(":"))
            stats = [by_n["methods"][name]["joint_targets"][key]
                     for name in ("tuned_sframes", "sframes", "prox_sarah")]
            values = " | ".join(f"{a['endpoint_hits']} / {a['first_hits']}" for a in stats)
            report.append(f"| {int(n):,} | {loss:.0%} | {100 * distance:g}% | {values} |")
    revised = source[:source.index(start) + len(start)] + "\n".join(report) + "\n\n" + source[source.index(end):]
    if revised != source:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".ct-appendix-", delete=False, mode="w") as tmp:
            tmp.write(revised)
            pending = Path(tmp.name)
        try:
            pending.chmod(0o644)
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def finish_publication(root, paper, audit, pending):
    stage = Path(audit["stage"])
    backup = root / "report_previous"
    if stage.parent != root or not stage.name.startswith("report_stage_"):
        raise ValueError("The staged report location is invalid")
    for name in ASSETS:
        old = audit["original_sha256"][name]
        new = audit["published_sha256"][name]
        if digest(backup / name) != old or digest(stage / name) != new:
            raise ValueError(f"Report backup or stage changed: {name}")
        if digest(paper / name) not in (old, new):
            raise ValueError(f"Live report contains an unexpected change: {name}")
    try:
        for name in ASSETS:  # PDF is deliberately last.
            atomic_copy(stage / name, paper / name)
        if any(digest(paper / name) != sha for name, sha in audit["published_sha256"].items()):
            raise ValueError("Published assets differ from the validated staged output")
        atomic_copy(pending, root / "report_update.json")
    except BaseException:
        for name in ASSETS:
            atomic_copy(backup / name, paper / name)
        raise
    pending.unlink()
    return audit


def build(root, paper, publish=False):
    legacy, previous, previous_audit = archived_inputs()
    summary_path = root / "summary.json"
    selection_path = root / "selection.json"
    summary, selection = read(summary_path), read(selection_path)
    anchors = {"summary_sha256": digest(summary_path), "selection_sha256": digest(selection_path)}
    if (summary["status"] != "completed_validated"
            or summary["provenance"]["selection_sha256"] != anchors["selection_sha256"]
            or summary["provenance"]["old_report_summary_sha256"] != previous_audit["summary_sha256"]):
        raise ValueError("Comparison and tuning selection do not match")
    if (len(summary["cases"]) != 84 or set(summary["by_n"]) != {"360", "1080"}
            or summary["view_budget_per_direction"] != 69120
            or summary["development_view_budget_per_direction"] != 34560):
        raise ValueError("Report needs the complete paired evaluation")
    audit_path = root / "report_update.json"
    pending = root / "report_update.pending.json"
    if publish and audit_path.exists():
        audit = read(audit_path)
        if any(audit[key] != value for key, value in anchors.items()):
            raise ValueError("A different follow-up was already published")
        if any(digest(paper / name) != sha for name, sha in audit["published_sha256"].items()):
            raise ValueError("The published report has changed since publication")
        if pending.exists() and read(pending) == audit:
            pending.unlink()
        supplement_results(paper, selection, summary)
        return audit
    if publish and pending.exists():
        audit = read(pending)
        if (any(audit[key] != value for key, value in anchors.items())
                or Path(audit["report_pdf"]) != paper / f"{STEM}.pdf"):
            raise ValueError("Pending publication belongs to different inputs or a different report")
        audit = finish_publication(root, paper, audit, pending)
        supplement_results(paper, selection, summary)
        return audit

    current = read(root / "current_report.json")
    starting_digests = {name: digest(paper / name) for name in ASSETS}
    if (current["status"] != "current_results_compact"
            or Path(current["report_pdf"]) != paper / f"{STEM}.pdf"
            or current["legacy_summary_sha256"] != previous_audit["summary_sha256"]
            or starting_digests != current["published_sha256"]):
        raise ValueError("The compact edition changed before joint tune integration")
    stage = Path(tempfile.mkdtemp(prefix="report_stage_", dir=root))
    source = render_source(previous, legacy, selection, summary)
    (stage / BODY).write_text(source)
    shutil.copyfile(paper / f"{STEM}.tex", stage / f"{STEM}.tex")
    shutil.copyfile(paper / LINK, stage / LINK)
    plotting = load_module("ct_joint_report_trajectories", Path(__file__).with_name("report_trajectories.py"))
    n_trajectories, n_observations = plotting.render(LEGACY, summary["cases"], stage)
    plot_development(selection, stage)
    pages = compile_pdf(stage)
    output_digests = {name: digest(stage / name) for name in ASSETS}
    audit = dict(**anchors, report_builder_sha256=digest(Path(__file__)),
                 plotting_source_sha256=digest(Path(__file__).with_name("report_trajectories.py")),
                 stage=str(stage), figures=2, tables=1, pages=pages,
                 evaluation_runs=n_trajectories, native_iteration_observations=n_observations,
                 original_sha256=starting_digests, staged_sha256=output_digests)
    (stage / "report_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    if publish:
        if any(digest(paper / name) != sha for name, sha in starting_digests.items()):
            raise ValueError("Original paper changed while the new report was building")
        previous_dir = root / "report_previous"
        previous_dir.mkdir(exist_ok=True)
        for name, sha in starting_digests.items():
            old = previous_dir / name
            if not old.exists():
                atomic_copy(paper / name, old)
            if digest(old) != sha:
                raise ValueError(f"Backup already contains different content: {name}")
        audit.update(report_pdf=str(paper / f"{STEM}.pdf"), published_sha256=output_digests)
        source_audit = stage / "report_update.json"
        source_audit.write_text(json.dumps(audit, indent=2) + "\n")
        atomic_copy(source_audit, pending)
        audit = finish_publication(root, paper, audit, pending)
        supplement_results(paper, selection, summary)
        return audit
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--paper", type=Path, default=PAPER)
    parser.add_argument("--current", action="store_true", help="Use the completed comparison before the joint tune")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "report_update.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder = build_current if args.current else build
        print(json.dumps(builder(root, args.paper.resolve(), args.publish), indent=2))


if __name__ == "__main__":
    main()
