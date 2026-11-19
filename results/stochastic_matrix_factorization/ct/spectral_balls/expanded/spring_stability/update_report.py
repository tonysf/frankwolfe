"""Stage, validate, and optionally publish the corrected SPRING comparison.

No optimizer is run here. Both campaigns must be complete before staging. The
frozen recovery files and feasibility comparison summary are never modified.
The latter's publication audit keeps its summary/selection anchors and records
this correction separately; an exact copy of the original audit is retained.
"""

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parent
FEASIBILITY = ROOT.parent / "feasibility_retune"
PAPER = ROOT.parent / "reports" / "paper_section" / "revised"
METHOD = "spring_sarah"
CAP = 69120
COUNT = 84
CORRECTION_START = "<!-- CT_SPRING_STABILITY_CORRECTION_START -->"
CORRECTION_END = "<!-- CT_SPRING_STABILITY_CORRECTION_END -->"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(case):
    return case["slice_id"], int(case["seed"]), int(case["n_train"])


def inputs(root, feasibility, report):
    """Check immutable completion anchors before loading any report assets."""
    progress = read(root / "progress.json")
    validation = read(root / "validation_summary.json")
    selection = read(root / "selection.json")
    plan = read(root / "evaluation_plan.json")
    require((progress["status"], progress["phase"]) in (("evaluated", "evaluation_validated"),
            ("completed", "report_published")) and validation["status"] == "completed_validated",
            "Corrected SPRING evaluation is not complete and validated")
    require(len(validation["cases"]) == COUNT and len(plan["runs"]) == COUNT,
            "Corrected SPRING publication requires all 84 fresh evaluation runs")
    require(len({key(case) for case in validation["cases"]}) == COUNT,
            "Corrected validation contains duplicate paired cases")
    selected = selection["selected"][METHOD]
    require(selected["rate"] == 1.6 and selected["refresh_period"] == 32,
            "This correction is restricted to the development-selected (a,q)=(1.6,32)")
    require(selected["endpoint_count"] == 12 and selected["max_relative_loss"] <= 0.2964
            and all(selection["unchanged"][name] for name in ("prox_sarah", "palm")),
            "Development screening does not support the correction description")
    rejected = next(row for row in selection["scores"][METHOD]
                    if row["rate"] == 2.3 and row["refresh_period"] == 16)
    require(rejected["failed_endpoint_count"] == 1 and round(rejected["max_relative_loss"], 2) == 33.66,
            "The historical development failure differs from the audited description")
    for name in ("protocol", "selection", "evaluation_plan"):
        require(validation[name + "_sha256"] == digest(root / (name + ".json")),
                f"Corrected validation has stale {name} provenance")
    fp = read(feasibility / "progress.json")
    summary = read(feasibility / "summary.json")
    tune = read(feasibility / "selection.json")
    require(fp["status"] == "completed" and fp["phase"] == "report_published",
            "The joint feasibility retune and its report must finish first")
    require(summary["status"] == "completed_validated" and len(summary["cases"]) == COUNT
            and summary["view_budget_per_direction"] == CAP
            and summary["development_view_budget_per_direction"] == CAP // 2
            and set(summary["by_n"]) == {"360", "1080"},
            "The feasibility retune is missing its complete paired comparison")
    require(summary["provenance"]["selection_sha256"] == digest(feasibility / "selection.json"),
            "Feasibility selection changed after analysis")
    legacy, previous, historical = report.archived_inputs()
    require(summary["provenance"]["old_report_summary_sha256"] == historical["summary_sha256"],
            "Feasibility comparison does not use the frozen historical report")
    files = {
        "corrected_validation": root / "validation_summary.json",
        "corrected_protocol": root / "protocol.json",
        "corrected_selection": root / "selection.json",
        "corrected_plan": root / "evaluation_plan.json",
        "feasibility_summary": feasibility / "summary.json",
        "feasibility_selection": feasibility / "selection.json",
        "historical_summary": report.LEGACY / "reports" / "summary.json",
        "report_builder": Path(__file__).resolve(),
        "base_report_builder": feasibility / "report.py",
        "plotting_source": feasibility / "report_trajectories.py",
        "paper_template": feasibility / "compact_paper.tex",
        "analysis_source": report.LEGACY / "analyze.py",
        "legacy_report_source": report.LEGACY / "report_update.py",
        "legacy_provenance_source": report.LEGACY / "audit_provenance.py",
    }
    anchors = {name: {"path": str(path), "sha256": digest(path)} for name, path in files.items()}
    return validation, selection, plan, summary, tune, legacy, previous, anchors


def corrected_runs(root, validation, plan, analysis, original_runs):
    """Reuse the campaign audit and verify all 84 reporting pairings."""
    campaign = module("spring_report_campaign", root / "campaign.py")
    protocol = campaign.verify_prepared()
    auditor = module("spring_report_sources", analysis.ROOT / "audit_provenance.py")
    expected = {k[:3] for k in original_runs if k[3] == METHOD}
    configs = {config["run_id"]: config for config in plan["runs"]}
    require(len(configs) == COUNT and len(expected) == COUNT, "Incomplete evaluation plan")
    result = {}
    for case in validation["cases"]:
        k = key(case)
        require(k in expected and k not in result, f"Unexpected corrected case {k}")
        config = configs[case["run_id"]]
        checked = campaign.validate_result(config, plan["assignments"][case["run_id"]], plan, protocol)
        require(checked == case, f"Corrected result changed after validation: {k}")
        folder = Path(case["source_directory"])
        meta, trace = read(folder / "metadata.json"), read(folder / "trace.json")
        run = {"config": config, "meta": meta, "trace": trace, "status": "completed",
               "source_directory": str(folder.resolve()),
               "canonical_numerical_source_sha256": {
                   name: auditor.canonical_source_hash(analysis.ROOT, name, meta["source_sha256"][name])
                   for name in analysis.NUMERICAL_SOURCES}}
        old = original_runs[(*k, METHOD)]
        require(plan["matched_run_ids"][case["run_id"]] == old["config"]["run_id"],
                f"Unmatched historical reference: {k}")
        analysis.verify_pair(old, run)
        result[k] = run
    require(set(result) == expected, "Corrected traces do not cover every paired evaluation case")
    return result


def replace_summary(previous, corrected, analysis, anchors):
    """Create a derived summary; do not mutate any frozen input or other arm."""
    updated = copy.deepcopy(previous)
    original_cap = previous["budgets"]["original_forward_view_cap"]
    groups = analysis.group_map(previous["design"])
    for case in updated["cases"]:
        run = corrected[key(case)]
        old = case["arms"][METHOD]
        arm = {
            "status": "completed", "run_id": run["config"]["run_id"],
            "shared_deterministic_reference": False, "origin": "executed_from_initialization",
            "source_directory": run["source_directory"], "source_sha256": run["meta"]["source_sha256"],
            "optimizer": run["meta"]["optimizer"], "continuation": None, "imported_checkpoint": None,
            "canonical_numerical_source_sha256": run["canonical_numerical_source_sha256"],
            "checkpoint_iteration": run["trace"][-1]["iteration"], "error": None,
            "probes": analysis.saved_probes(run["trace"], CAP, original_cap),
            "superseded_run_id": old["run_id"],
        }
        if updated.get("targets"):
            arm["target_hits"] = {target["name"]: analysis.first_target_hit(run, target, CAP)
                                  for target in updated["targets"]["targets"]}
        case["arms"][METHOD] = arm
    for n, cohort in updated["cohorts"].items():
        subset = [case for case in updated["cases"] if case["n_train"] == int(n)]
        for probe in ("original_cap", "view_cap", "endpoint"):
            cohort[probe] = {
                "matched_all_methods": analysis.marginal_summary(subset, probe, groups),
                "sframes_vs": {variant: analysis.paired_summary(subset, variant, probe, groups)
                                for variant in analysis.VARIANTS if variant != analysis.ANCHOR},
            }
        if updated.get("targets"):
            cohort["accuracy_targets"] = {target["name"]: analysis.target_summary(subset, target["name"], groups, CAP)
                                           for target in updated["targets"]["targets"]}
    for before, after in zip(previous["cases"], updated["cases"]):
        require(key(before) == key(after), "Derived summary changed case order")
        require(all(before["arms"][method] == arm for method, arm in after["arms"].items() if method != METHOD),
                "Derived summary changed an unaffected arm")
    # Preserve the frozen selection and coverage as explicitly historical records.
    updated["historical_selection"] = updated.pop("selection", None)
    updated["historical_coverage"] = updated.pop("coverage", None)
    updated["generated_utc"] = datetime.now(timezone.utc).isoformat()
    updated["spring_correction"] = {
        "anchors": anchors, "selected": {"rate": 1.6, "refresh_period": 32},
        "replaced_physical_runs": COUNT, "other_arms": "unchanged frozen reference runs",
        "scope": "Derived report summary; historical campaign files remain unchanged.",
    }
    return updated


def source_text(report, updated, legacy, tune, summary):
    source = report.render_source(updated, legacy, tune, summary)
    old = (r"The baseline searches use the same slices and criterion, with" "\n"
           r"\(34{,}560\) views per run versus \(8{,}640\) for this SFRAMES search.")
    new = (r"The baseline searches use the same slices and criterion, with" "\n"
           r"\(34{,}560\) views per run versus \(8{,}640\) for this SFRAMES search." "\n"
           r"A baseline candidate is excluded if its endpoint loss exceeds its initial" "\n"
           r"loss in any development run; the \CTReproducibility{} details this screen.")
    require(source.count(old) == 1, "The baseline selection paragraph changed; review the correction text")
    return source.replace(old, new)


def appendix_text(source, root, paper, updated):
    require(CORRECTION_START not in source and CORRECTION_END not in source,
            "The appendix already contains a correction; use its publication audit")
    old = "| $(a,q)=(2.3,16)$ |"
    require(source.count(old) == 1, "The historical SPRING development table changed")
    source = source.replace(old, "| $(a,q)=(1.6,32)$ |")
    old = "All selected numerical baseline parameters were interior to their final grids. "
    require(source.count(old) == 1, "The baseline search description changed")
    source = source.replace(old, "The historical median-only selection was subsequently screened as described below; "
                            "the corrected SPRING refresh period is at the largest tested grid value. ")
    old = "Nonfinite candidates were ineligible; operational or projection failures stopped selection."
    require(source.count(old) == 1, "The baseline eligibility description changed")
    source = source.replace(old, "Nonfinite candidates and candidates whose endpoint repaired training loss exceeds "
                            "initial loss in any development run are ineligible; operational or projection failures stop selection.")
    old = "### Original first-hit and paired summaries\n"
    require(source.count(old) == 1, "The historical comparison heading changed")
    source = source.replace(old, old + "\nThe following two tables retain the archived comparison with SPRING "
                            "$(a,q)=(2.3,16)$. They are historical results, superseded for SPRING in the revised "
                            "paper by the audited correction below. The other arms are unchanged.\n")
    relative = Path(os.path.relpath(root, paper.parent))
    lines = ["", CORRECTION_START, "## Audited SPRING–SARAH correction", "",
             "The recursive estimator and alternating query points were checked against the SPRING update. "
             "The failure was in parameter screening: the previous $(a,q)=(2.3,16)$ candidate had one of 12 "
             "development endpoints with repaired loss **33.66 times its initial loss**, hidden by the nested "
             "median selection score. Finite loss alone was therefore an inadequate eligibility screen.", "",
             "All existing baseline development candidates were re-screened before evaluating the correction. "
             "A candidate is excluded if any development run is nonfinite or finishes above its initial "
             "repaired training loss. The remaining candidates retain the original nested-median ranking and "
             "tie breaks. This selects **$(a,q)=(1.6,32)$** for SPRING–SARAH: all 12 development loss ratios are "
             "at most **0.2964**. ProxSARAH and PALM retain their previous selections. The screen does not "
             "guarantee stability on unseen instances; $q=32$ is the largest tested refresh period.", "",
             "The corrected SPRING comparison uses 84 fresh runs from the original calibrated factors, with "
             "the same evaluation slices, oracle seeds, constraints, and 69,120-view budget per direction. "
             "Every iterate is recorded. Endpoints use the last native observation within the cap; an "
             "overshooting final update is excluded. The revised paper replaces only the SPRING arm, "
             "retains all original other arms, and includes the completed joint SFRAMES tune. "
             "SPRING still uses practical fixed initial-curvature steps $1.6/L_U^0,1.6/L_V^0$, not the "
             "published local-curvature step schedule.", "",
             "Corrected paired changes in repaired loss (original minibatch SFRAMES minus SPRING, "
             "divided by SPRING) are:", "",
             "| Training views | Median paired change across groups |", "| ---: | ---: |"]
    for n in ("360", "1080"):
        effect = updated["cohorts"][n]["view_cap"]["sframes_vs"][METHOD]["metrics"]["feasible_loss"][
            "percent_difference"]["median_across_groups"]
        lines.append(f"| {int(n):,} | {effect:+.1f}% |")
    lines.extend(["", f"The [corrected selection]({relative}/selection.json), "
                  f"[completion audit]({relative}/validation_summary.json), and "
                  f"[publication audit]({relative}/report_update.json) retain the source and result hashes. "
                  "The publication audit links the derived summary and exact pre-correction report/audit backup. "
                  "Frozen historical data and manifests were not rewritten.", CORRECTION_END, ""])
    return source.rstrip() + "\n" + "\n".join(lines)


def plot(report, plotting, analysis, runs, corrected, slices, sizes, summary, stage):
    for k, run in corrected.items():
        runs[(*k, METHOD)] = dict(run, trace=[row for row in run["trace"]
                                             if max(row["forward_views"], row["backward_views"]) <= CAP])
    design = read(report.LEGACY / "manifest.json")["design"]
    groups = analysis.group_map(design)
    plotting.add_joint_traces(runs, summary["cases"], groups, CAP)
    variants = analysis.VARIANTS
    analysis.VARIANTS = {
        "sframes": ("SFRAMES (orig.)", variants["sframes"][1]),
        "tuned_sframes": ("SFRAMES (joint)", "#333333"),
        "sframes_full": ("Full-batch SF", variants["sframes_full"][1]),
        **{name: variants[name] for name in (METHOD, "prox_sarah", "palm")},
    }
    plotting.plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
                                 "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.edgecolor": "0.55",
                                 "axes.linewidth": 0.6, "pdf.fonttype": 42})
    for size in sizes:
        plotting.plot_size(analysis, runs, slices, groups, size, CAP,
                           design.get("original_evaluation_view_budget", 17280), stage)
    require(len(runs) == 392, "The corrected report must keep the same six-arm cohort")
    return len(runs), sum(len(run["trace"]) for run in runs.values())


def destinations(report, paper, feasibility):
    return {**{name: paper / name for name in report.ASSETS},
            report.APPENDIX: paper.parent / report.APPENDIX,
            "feasibility_report_update.json": feasibility / "report_update.json"}


def finish(root, report, audit, targets, pending):
    """Replay an audited journal; every live file must be its old or new version."""
    stage, backup = Path(audit["stage"]), Path(audit["backup"])
    require(stage.parent == root and stage.name.startswith("report_stage_")
            and backup == root / "report_previous", "Invalid correction stage or backup")
    require(set(targets) == set(audit["published_sha256"]), "Publication destinations changed")
    for name, target in targets.items():
        old, new = audit["original_sha256"][name], audit["published_sha256"][name]
        require(digest(backup / name) == old and digest(stage / name) == new, f"Stage or backup changed: {name}")
        require(digest(target) in (old, new), f"Live publication contains an unrelated edit: {name}")
    try:
        for name, target in targets.items():  # Base report orders PDF last; compatibility audit follows.
            report.atomic_copy(stage / name, target)
        require(all(digest(target) == audit["published_sha256"][name] for name, target in targets.items()),
                "Published assets do not match the validated stage")
        report.atomic_copy(pending, root / "report_update.json")
    except BaseException:
        for name, target in targets.items():
            report.atomic_copy(backup / name, target)
        raise
    pending.unlink(missing_ok=True)
    return audit


def build(root, feasibility, paper, publish=False, check=False):
    report = module("spring_base_report", feasibility / "report.py")
    validation, selection, plan, summary, tune, legacy, previous, anchors = inputs(root, feasibility, report)
    targets = destinations(report, paper, feasibility)
    final, pending = root / "report_update.json", root / "report_update.pending.json"
    if final.exists():
        audit = read(final)
        require(audit["anchors"] == anchors and audit["report_pdf"] == str(paper / f"{report.STEM}.pdf"),
                "A different correction was already published")
        require(all(digest(path) == audit["published_sha256"][name] for name, path in targets.items()),
                "The corrected publication changed after its audit")
        if publish and pending.exists() and read(pending) == audit:
            pending.unlink()
        return audit
    if pending.exists():
        audit = read(pending)
        require(audit["anchors"] == anchors and audit["report_pdf"] == str(paper / f"{report.STEM}.pdf"),
                "Pending correction has different inputs")
        return finish(root, report, audit, targets, pending) if publish else dict(status="publication_pending", stage=audit["stage"])
    old_audit = read(feasibility / "report_update.json")
    require(old_audit["summary_sha256"] == digest(feasibility / "summary.json")
            and old_audit["selection_sha256"] == digest(feasibility / "selection.json")
            and Path(old_audit["report_pdf"]) == paper / f"{report.STEM}.pdf"
            and "spring_stability" not in old_audit, "The feasibility publication audit does not match")
    require(all(digest(paper / name) == sha for name, sha in old_audit["published_sha256"].items()),
            "The joint-tune report changed since publication")
    before = {name: digest(path) for name, path in targets.items()}
    plotting = module("spring_corrected_trajectories", feasibility / "report_trajectories.py")
    design = read(report.LEGACY / "manifest.json")["design"]
    analysis, runs, slices, sizes = plotting.evaluation_traces(report.LEGACY, design, CAP)
    corrected = corrected_runs(root, validation, plan, analysis, runs)
    require({key(case) for case in previous["cases"]} == set(corrected)
            == {key(case) for case in summary["cases"]}, "The campaigns use different evaluation cases")
    if check:
        return {"status": "ready", "corrected_runs": len(corrected), "anchors": anchors}
    updated = replace_summary(previous, corrected, analysis, anchors)
    stage = Path(tempfile.mkdtemp(prefix="report_stage_", dir=root))
    write(stage / "corrected_summary.json", updated)
    (stage / report.BODY).write_text(source_text(report, updated, legacy, tune, summary))
    for name in (f"{report.STEM}.tex", report.LINK, "joint_selection.pdf", "joint_selection.png"):
        shutil.copyfile(paper / name, stage / name)
    (stage / report.APPENDIX).write_text(appendix_text((paper.parent / report.APPENDIX).read_text(),
                                                    root, paper, updated))
    count, observations = plot(report, plotting, analysis, runs, corrected, slices, sizes, summary, stage)
    pages = report.compile_pdf(stage)
    hashes = {name: digest(stage / name) for name in report.ASSETS}
    corrected_audit = copy.deepcopy(old_audit)
    corrected_audit.update(stage=str(stage), pages=pages, figures=2, tables=1, evaluation_runs=count,
                           native_iteration_observations=observations, staged_sha256=hashes, published_sha256=hashes)
    corrected_audit["spring_stability"] = {
        "source": str(Path(__file__).resolve()), "source_sha256": digest(Path(__file__)),
        "completion_audit": str(root / "validation_summary.json"),
        "completion_sha256": digest(root / "validation_summary.json"),
        "corrected_summary": str(stage / "corrected_summary.json"),
        "corrected_summary_sha256": digest(stage / "corrected_summary.json"),
        "prior_audit": str(root / "report_previous" / "feasibility_report_update.json"),
        "prior_audit_sha256": before["feasibility_report_update.json"],
        "publication_audit": str(root / "report_update.json"),
        "scope": "Only SPRING results replaced; feasibility summary and selection unchanged.",
    }
    write(stage / "feasibility_report_update.json", corrected_audit)
    audit = {"status": "completed_published" if publish else "staged_validated", "anchors": anchors,
             "stage": str(stage), "backup": str(root / "report_previous"),
             "report_pdf": str(paper / f"{report.STEM}.pdf"), "figures": 2, "tables": 1, "pages": pages,
             "evaluation_runs": count, "corrected_spring_runs": len(corrected),
             "native_iteration_observations": observations,
             "corrected_summary": str(stage / "corrected_summary.json"),
             "corrected_summary_sha256": digest(stage / "corrected_summary.json"),
             "original_sha256": before, "published_sha256": {name: digest(stage / name) for name in targets}}
    write(stage / "report_validation.json", audit)
    require(all(digest(path) == before[name] for name, path in targets.items()), "Live report changed during staging")
    require(all(digest(Path(value["path"])) == value["sha256"] for value in anchors.values()),
            "An audited input changed during staging")
    if not publish:
        return audit
    backup = root / "report_previous"
    backup.mkdir(exist_ok=True)
    for name, path in targets.items():
        if not (backup / name).exists():
            report.atomic_copy(path, backup / name)
        require(digest(backup / name) == before[name], f"An existing backup belongs to another publication: {name}")
    report.atomic_copy(stage / "report_validation.json", pending)
    return finish(root, report, audit, targets, pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--feasibility-root", type=Path, default=FEASIBILITY)
    parser.add_argument("--paper", type=Path, default=PAPER)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--check", action="store_true", help="Validate readiness without staging or publishing")
    args = parser.parse_args()
    root, feasibility = args.root.resolve(), args.feasibility_root.resolve()
    with (root / "report_update.lock").open("a") as own_lock, (feasibility / "report_update.lock").open("a") as other_lock:
        for lock in (own_lock, other_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(build(root, feasibility, args.paper.resolve(), args.publish, args.check), indent=2))


if __name__ == "__main__":
    main()
