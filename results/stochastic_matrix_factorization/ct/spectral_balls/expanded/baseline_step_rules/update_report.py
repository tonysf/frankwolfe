"""Publish paper-parameter PALM/ProxSARAH after the completed SPRING correction.

All optimizer runs are external to this module. Immutable predecessor summaries
remain intact. The two upstream publication audits receive successor pointers
and current asset hashes in the same recoverable transaction as the report.
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
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
SPRING = ROOT.parent / "spring_localcurvature"
FEASIBILITY = ROOT.parent / "feasibility_retune"
PAPER = ROOT.parent / "reports" / "paper_section" / "revised"
METHODS = ("palm", "prox_sarah")
CAP, CASES, PHYSICAL_RUNS, FIGURE_RUNS = 69120, 84, 168, 448
START = "<!-- CT_BASELINE_PAPER_PARAMETERS_START -->"
END = "<!-- CT_BASELINE_PAPER_PARAMETERS_END -->"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def key(case):
    return case["slice_id"], int(case["seed"]), int(case["n_train"])


def inputs(root, spring, feasibility, report):
    fixture = read(root / "report_fixture_validation.json")
    require(fixture["status"] == "passed" and fixture["fixture_only"] is True
            and fixture["updater_sha256"] == digest(Path(__file__))
            and fixture["validator_sha256"] == digest(root / "validate_report.py"),
            "The current report builder needs its bounded fixture validation")
    progress = read(root / "progress.json")
    validation = read(root / "validation_summary.json")
    plan = read(root / "evaluation_plan.json")
    require((progress["status"], progress["phase"]) in (
        ("evaluated", "evaluation_validated"), ("completed", "report_published")),
        "The corrected baseline campaign is not complete")
    require(validation["status"] == "completed_validated"
            and len(validation["cases"]) == len(plan["runs"]) == PHYSICAL_RUNS,
            "Publication requires all 84 PALM and 84 ProxSARAH runs")
    for name in ("protocol", "selection", "evaluation_plan", "implementation_manifest"):
        require(validation[name + "_sha256"] == digest(root / (name + ".json")),
                f"Stale baseline validation: {name}")
    for folder, label in ((spring, "SPRING correction"), (feasibility, "joint SFRAMES retune")):
        state = read(folder / "progress.json")
        require((state["status"], state["phase"]) == ("completed", "report_published"),
                f"The {label} must be published first")
    replay = (root / "report_update.json").exists() or (root / "report_update.pending.json").exists()
    spring_audit_path = root / "report_previous" / "spring_report_update.json" if replay else spring / "report_update.json"
    predecessor = read(spring_audit_path)
    # Never reconstruct the starting summary from the archived recovery campaign:
    # that would restore the superseded fixed-step SPRING arm.
    require(predecessor["status"] == "completed_published", "SPRING publication is incomplete")
    summary_path = Path(predecessor["corrected_summary"])
    require(digest(summary_path) == predecessor["corrected_summary_sha256"], "SPRING summary changed")
    require(not predecessor.get("successor"), "A different successor already exists; review its explicit chain")
    previous = read(summary_path)
    require("spring_correction" in previous and len(previous["cases"]) == CASES,
            "Starting summary is not the corrected SPRING comparison")
    require(predecessor["anchors"]["corrected_validation"]["sha256"] == digest(spring / "validation_summary.json"),
            "Retained SPRING completion audit changed")
    joint = read(feasibility / "summary.json")
    tune = read(feasibility / "selection.json")
    require(joint["status"] == "completed_validated" and len(joint["cases"]) == CASES
            and joint["view_budget_per_direction"] == CAP
            and joint["provenance"]["selection_sha256"] == digest(feasibility / "selection.json"),
            "The frozen joint SFRAMES comparison is incomplete or changed")
    legacy, _, historical = report.archived_inputs()
    require(joint["provenance"]["old_report_summary_sha256"] == historical["summary_sha256"],
            "The joint comparison has a different historical anchor")
    paths = {
        "baseline_validation": root / "validation_summary.json", "baseline_plan": root / "evaluation_plan.json",
        "baseline_protocol": root / "protocol.json", "baseline_selection": root / "selection.json",
        "baseline_implementation": root / "implementation_manifest.json", "baseline_readme": root / "README.md",
        "spring_summary": summary_path, "spring_validation": spring / "validation_summary.json",
        "joint_summary": feasibility / "summary.json", "joint_selection": feasibility / "selection.json",
        "report_builder": Path(__file__), "base_report_builder": feasibility / "report.py",
        "joint_analysis": feasibility / "analysis.py", "plotting_source": feasibility / "report_trajectories.py",
        "paper_template": feasibility / "compact_paper.tex", "legacy_analysis": report.LEGACY / "analyze.py",
        "campaign_validator": root / "campaign.py", "legacy_report": report.LEGACY / "report_update.py",
        "legacy_provenance": report.LEGACY / "audit_provenance.py",
        "report_fixture_validation": root / "report_fixture_validation.json", "report_fixture_validator": root / "validate_report.py",
    }
    anchors = {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in paths.items()}
    return validation, plan, previous, joint, tune, legacy, predecessor, anchors


def corrected_runs(root, validation, plan, previous, analysis):
    campaign = module("paper_baseline_campaign", root / "campaign.py")
    protocol = campaign.verify_prepared()
    auditor = module("paper_baseline_source_audit", analysis.ROOT / "audit_provenance.py")
    configs = {cfg["run_id"]: cfg for cfg in plan["runs"]}
    originals = {key(case): case for case in previous["cases"]}
    expected = {(*k, method) for k in originals for method in METHODS}
    require(len(configs) == PHYSICAL_RUNS and len(expected) == PHYSICAL_RUNS, "Duplicate planned baseline runs")
    result = {}
    for case in validation["cases"]:
        config = configs[case["run_id"]]
        k = (*key(config), config["method"])
        require(k in expected and k not in result, f"Unplanned or duplicate baseline case: {k}")
        checked = campaign.validate_result(config, plan["assignments"][case["run_id"]], plan, protocol)
        require(checked == case, f"Result changed after completion audit: {k}")
        folder = Path(case["source_directory"])
        meta, trace = read(folder / "metadata.json"), read(folder / "trace.json")
        run = dict(config=config, meta=meta, trace=trace, status="completed", source_directory=str(folder.resolve()),
                   canonical_numerical_source_sha256={
                       name: auditor.canonical_source_hash(analysis.ROOT, name, meta["source_sha256"][name])
                       for name in analysis.NUMERICAL_SOURCES})
        old_arm = originals[k[:3]]["arms"][k[3]]
        old_folder = Path(old_arm["source_directory"])
        old_meta = read(old_folder / "metadata.json")
        old = dict(config=old_meta["config"], meta=old_meta, trace=read(old_folder / "trace.json"),
                   canonical_numerical_source_sha256=old_arm["canonical_numerical_source_sha256"])
        require(plan["matched_run_ids"][case["run_id"]] == old_arm["run_id"], "Historical pairing changed")
        analysis.verify_pair(old, run)
        require(all(row["d_distance"] == 0 for row in trace), "A proximal baseline is orthant-infeasible")
        result[k] = run
    require(set(result) == expected, "Incomplete corrected baseline cohort")
    return result


def replace_summary(previous, corrected, analysis, anchors):
    updated = copy.deepcopy(previous)
    groups = analysis.group_map(previous["design"])
    for case in updated["cases"]:
        for method in METHODS:
            run = corrected[(*key(case), method)]
            old = case["arms"][method]
            probes = analysis.saved_probes(run["trace"], CAP, previous["budgets"]["original_forward_view_cap"])
            for probe in probes.values():
                native = run["trace"][probe["iteration"]]
                for field in ("oracle_calls", "curvature_operator_calls", "curvature_forward_views", "curvature_backward_views"):
                    probe[field] = native.get(field, native["gradient_calls"] if field == "oracle_calls" else 0)
            arm = dict(status="completed", run_id=run["config"]["run_id"], shared_deterministic_reference=False,
                       origin="executed_from_initialization", source_directory=run["source_directory"],
                       source_sha256=run["meta"]["source_sha256"], optimizer=run["meta"]["optimizer"],
                       continuation=None, imported_checkpoint=None, error=None,
                       canonical_numerical_source_sha256=run["canonical_numerical_source_sha256"],
                       checkpoint_iteration=run["trace"][-1]["iteration"], probes=probes,
                       superseded_run_id=old["run_id"])
            if updated.get("targets"):
                arm["target_hits"] = {t["name"]: analysis.first_target_hit(run, t, CAP)
                                      for t in updated["targets"]["targets"]}
                for hit in arm["target_hits"].values():
                    if hit["reached"]:
                        native = run["trace"][hit["iteration"]]
                        hit["oracle_calls"] = native.get("oracle_calls", native["gradient_calls"])
            case["arms"][method] = arm
    for n, cohort in updated["cohorts"].items():
        subset = [case for case in updated["cases"] if case["n_train"] == int(n)]
        for probe in ("original_cap", "view_cap", "endpoint"):
            cohort[probe] = dict(matched_all_methods=analysis.marginal_summary(subset, probe, groups),
                sframes_vs={v: analysis.paired_summary(subset, v, probe, groups)
                            for v in analysis.VARIANTS if v != analysis.ANCHOR})
        if updated.get("targets"):
            cohort["accuracy_targets"] = {t["name"]: analysis.target_summary(subset, t["name"], groups, CAP)
                                           for t in updated["targets"]["targets"]}
    for before, after in zip(previous["cases"], updated["cases"]):
        require(key(before) == key(after) and all(before["arms"][m] == arm for m, arm in after["arms"].items()
                                                   if m not in METHODS), "An unaffected arm changed")
    updated["generated_utc"] = datetime.now(timezone.utc).isoformat()
    updated["historical_design"] = copy.deepcopy(previous["design"])
    updated["design"]["deterministic_variants"] = [m for m in previous["design"]["deterministic_variants"] if m != "palm"]
    updated["design"]["palm_repetitions"] = "All three seeds executed with independent fresh power directions; 84 physical runs."
    updated["baseline_parameter_correction"] = dict(anchors=anchors, replaced_methods=list(METHODS),
        physical_runs=PHYSICAL_RUNS, preserved_spring_correction=previous["spring_correction"],
        scope="Derived summary; frozen predecessor results and selections remain unchanged.")
    return updated


def joint_summary(previous, corrected, analysis, helpers, anchors):
    """Recompute the joint tune's ProxSARAH comparator without mutating F/summary.json."""
    updated = copy.deepcopy(previous)
    groups = analysis.group_map(previous["cohort"])
    target_keys = [f"{t['relative_repaired_loss']:g}:{t['max_relative_d_distance']:g}" for t in previous["targets"]]
    for case in updated["cases"]:
        full = corrected[(*key(case), "prox_sarah")]
        run = dict(full, trace=[row for row in full["trace"]
                               if max(row["forward_views"], row["backward_views"]) <= CAP])
        require(run["trace"], "No eligible native ProxSARAH iterate")
        case["runs"]["prox_sarah"] = helpers.record_run(run, Path(run["source_directory"]), analysis,
                                                        previous["targets"], CAP)
    by_n = {}
    for n in sorted(previous["by_n"], key=int):
        subset = [case for case in updated["cases"] if case["n_train"] == int(n)]
        methods = {}
        for label in helpers.METHODS:
            fields = (*analysis.WORK, *analysis.METRICS, "relative_raw_loss", "relative_repaired_loss", "relative_d_distance")
            endpoint = {metric: helpers.grouped(subset, groups, analysis,
                        lambda c, m=metric: c["runs"][label]["endpoint"][m]) for metric in fields}
            targets = {t: helpers.joint_summary(subset, groups, analysis, label, t) for t in target_keys}
            methods[label] = dict(endpoint=endpoint, joint_targets=targets)
        by_n[n] = dict(methods=methods, cases=len(subset), pairs={
            label: helpers.compare(subset, groups, analysis, label, target_keys) for label in helpers.COMPARATORS})
    updated["by_n"] = by_n
    updated["generated_utc"] = datetime.now(timezone.utc).isoformat()
    updated["baseline_parameter_correction"] = dict(anchors=anchors, replaced_reference="prox_sarah",
        scope="Recomputed comparison only; original and joint-tuned SFRAMES observations and selection are unchanged.")
    for before, after in zip(previous["cases"], updated["cases"]):
        require(all(before["runs"][m] == after["runs"][m] for m in ("sframes", "tuned_sframes")),
                "The joint comparison changed a SFRAMES observation")
    return updated


def effect(value):
    if abs(value) < .05:
        return "unchanged to the reported precision"
    return rf"\({abs(value):.1f}\%\) " + ("lower" if value < 0 else "higher")


def source_text(report, updated, legacy, tune, joint, selection):
    report.LABELS["prox_sarah"] = "ProxSARAH-v3"
    source = (Path(report.__file__).parent / "compact_paper.tex").read_text()
    chosen = report.fmt_pair(report.pair(tune["selected"]))
    c_l = selection["selected"]["c_L"]
    tuning = rf"""The loss-tuned SFRAMES setting \((c_\beta,\gamma_0)=(16,0.1)\)
is selected jointly by repaired training loss on two development slices.
A second sweep selects \((c_\beta,\gamma_0)={chosen}\) using the joint
endpoint target of repaired loss at most \(30\%\) of its initial value
and relative orthant distance at most \(1\%\).
The two searches use \(8{{,}}640\) and \(34{{,}}560\) views per direction,
respectively. No evaluation slice enters parameter selection."""
    results = []
    for n in ("360", "1080"):
        original = updated["cohorts"][n]["view_cap"]["sframes_vs"]["prox_sarah"]["metrics"][
            "feasible_loss"]["percent_difference"]["median_across_groups"]
        tuned = joint["by_n"][n]["pairs"]["prox_sarah"]["repaired_loss_pct"]["median_across_groups"]
        rates = [100 * joint["by_n"][n]["methods"][m]["joint_targets"]["0.3:0.01"][
            "endpoint_group_balanced_fraction"] for m in ("tuned_sframes", "sframes", "prox_sarah")]
        results.append(rf"""At \(N={legacy.integer(n)}\), paired repaired loss relative to ProxSARAH
is {effect(original)} for loss-tuned SFRAMES and {effect(tuned)} for the joint tune.
The group-balanced fractions meeting both the \(30\%\) relative-loss
and \(1\%\) raw-feasibility targets are \({rates[0]:.0f}\%\),
\({rates[1]:.0f}\%\), and \({rates[2]:.0f}\%\) for the joint tune,
loss-tuned SFRAMES, and ProxSARAH, respectively.""")
    results.append("The proximal baselines remain nonnegative throughout. All curvature and\n"
        "snapshot work is charged to the view budget; PALM completes 13 and 4\n"
        "two-block updates at the two view counts, respectively.")
    for token, value in (("@@RESULTS_TEXT@@", "\n\n".join(results)), ("@@TUNING_TEXT@@", tuning),
                          ("@@RESULTS_TABLE@@", report.quality_table(legacy, updated, joint))):
        require(source.count(token) == 1, "Paper template placeholders changed")
        source = source.replace(token, value)
    old = "SPRING--SARAH alternates block updates with loopless SARAH estimates;"
    start, end = source.index(old), source.index("The \\CTReproducibility{}", source.index(old))
    methods = rf"""SPRING--SARAH uses \(64\)-view differences, independent full-gradient
refreshes with probability \(1/16\), and steps \(1/(2\widetilde L)\).
PALM uses full block gradients and steps \(1/\widehat L\).
Both estimate current block curvature by five power iterations followed
by a final norm evaluation, using minibatches for SPRING and all views for PALM.
ProxSARAH uses the paper's constant-parameter variant v3: full snapshots,
relaxation \(0.99\), and \((b,m)=(27,18)\) or \((48,32)\) at the two
view counts. Its proximal step is \(0.400802/\widehat L\), with
\(\widehat L={c_l:g}(L_U+L_V)\); only this curvature-scale multiplier
is selected on the development slices at \(34{{,}}560\) views per direction.
These practical curvature estimates are not certified Lipschitz bounds.
All three baselines project each factor onto its spectral ball intersected
with the nonnegative orthant. """
    source = source[:start] + methods + source[end:]
    old = "views per direction, including all refresh and recursive-gradient work.\nAn oracle call returns one joint or block gradient on a batch."
    require(source.count(old) == 1, "Paper work-accounting text changed")
    source = source.replace(old, "views per direction, including snapshot, recursive-gradient, and curvature work.\n"
        "An oracle call is one batch gradient or one block-Hessian application.")
    require(source.count(r"\begin{figure}") == 2 and source.count(r"\begin{table}") == 1
            and "@@" not in source and r"d_C" not in source, "Paper layout invariants changed")
    return source


def appendix_text(source, root, paper, joint, selection):
    require(START not in source and END not in source, "The correction is already in the appendix")
    pending = ("Corrected results for these two methods are not yet available; "
               "the separate SPRING correction is already queued.")
    require(source.count(pending) == 1, "Historical audit status changed; review the appendix")
    source = source.replace(pending, "This audit is resolved by the completed paper-parameter reruns below. "
        "The old runs and their parameter audit remain historical evidence; the corrected SPRING arm is retained.")
    marker = "<!-- CT_FEASIBILITY_RETUNE_RESULT_START -->"
    require(source.count(marker) == 1, "Joint-tune appendix marker changed")
    source = source.replace(marker, "The following marked joint-tune table retains its original ProxSARAH reference "
        "for the predecessor publication audit. The current comparison uses the corrected reference below.\n\n" + marker)
    source = source.replace("Full-batch SFRAMES and PALM are deterministic given the calibration",
                            "Full-batch SFRAMES and the archived fixed-step PALM are deterministic given the calibration")
    for name in METHODS:
        label = "PALM" if name == "palm" else "ProxSARAH"
        source = source.replace(f"- **[{label}]", f"- **Archived custom [{label}]", 1)
    spring_heading = "## SPRING with the published practical step rule"
    if spring_heading in source:
        source = source.replace(spring_heading, spring_heading + "\n\nThis section records the predecessor publication. "
            "Its corrected SPRING results are retained; the later correction below replaces its PALM and ProxSARAH references.", 1)
    for name in ("ProxSARAH", "PALM"):
        prefix = f"| {name} |"
        # The first match is the archived tuning table, not a current result.
        index = source.find(prefix)
        require(index >= 0, f"Missing historical {name} tuning row")
        source = source[:index] + source[index:].replace(prefix, f"| {name}, archived custom parameters |", 1)
    rel = Path(os.path.relpath(root, paper.parent))
    c_l = selection["selected"]["c_L"]
    lines = ["", START, "## Completed PALM and ProxSARAH paper-parameter comparison", "",
        "The paper replaces both archived custom-parameter arms with 84 fresh PALM and 84 fresh ProxSARAH runs. "
        "The corrected local-curvature SPRING arm, both original SFRAMES arms, and the completed joint SFRAMES tune "
        "are retained. The data, split, projections, and shared starting factors are unchanged.", "",
        "**PALM.** At each block update, estimate full-data block curvature with a fresh normalized Gaussian direction, "
        "five normalized Hessian applications and a sixth application for the norm. Use $1/\\widehat L$; the second "
        "block uses the updated first factor. Fresh power directions make the numerical implementation random, so "
        "all three seed repetitions are executed instead of sharing one deterministic reference. Twelve curvature "
        "applications plus two block gradients cost $14N$ views per direction per complete iteration: 13 updates "
        "fit the cap for $N=360$ and four for $N=1080$.", "",
        "**ProxSARAH.** Use the constant-parameter v3 experiment in Section 5 of the paper, with "
        "$\\gamma=0.99$, $C=2/(3\\gamma^2)$, $m=\\lfloor\\sqrt N\\rfloor$ and "
        "$b=\\lfloor\\sqrt N/C\\rfloor$. Thus $(b,m)=(27,18)$ or $(48,32)$. "
        "Full snapshots use all $N$ training views; recursive differences use the same batch at the two points. "
        "Apply the normalized-$L=1$ paper rule to $f/\\widehat L$, giving proximal step "
        "$[2/(4+0.99)]/\\widehat L$ and fixed relaxation $0.99$. "
        f"The frozen scale is $\\widehat L={c_l:g}(L_U^0+L_V^0)$. Only $c_L$ is selected, "
        "starting from $2^{-5},2^{-4},\\ldots,2^2$ on the two development slices, both view counts and three seeds, "
        "using 34,560 views per direction. Nonfinite settings and settings ending above initial repaired loss "
        "are ineligible; the nested median repaired-to-initial loss determines selection. A boundary winner extends "
        "the grid by a factor of two on that side, within guards $[2^{-8},2^5]$. A boundary winner at a guard "
        "requires review before evaluation. The selection record lists every tested scale. "
        "The paper's relaxation and normalized step are not independently tuned. "
        "This is a practical curvature scale, not a certified joint average-smoothness bound.", "",
        "Both methods record every native iterate, and all snapshot, difference, and curvature views count toward "
        "69,120 views per direction. An overshooting step is excluded from reported endpoints and first-hit comparisons. "
        "`gradient_calls` counts actual gradients; `oracle_calls` additionally counts Hessian applications. "
        "The same two figures and one table now use the corrected native traces.", "",
        "The joint SFRAMES comparison has been recomputed against the corrected ProxSARAH reference. "
        "Entries below are endpoint hits / first hits among 42 paired cases per view count.", "",
        "| Training views | Relative repaired loss | Relative raw distance | Joint SFRAMES | Original SFRAMES | ProxSARAH |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n, cohort in sorted(joint["by_n"].items(), key=lambda item: int(item[0])):
        for target in ("0.3:0.01", "0.3:0.001", "0.4:0.01", "0.4:0.001"):
            loss, distance = map(float, target.split(":"))
            counts = [cohort["methods"][m]["joint_targets"][target] for m in ("tuned_sframes", "sframes", "prox_sarah")]
            cells = " | ".join(f"{v['endpoint_hits']} / {v['first_hits']}" for v in counts)
            lines.append(f"| {int(n):,} | {loss:.0%} | {distance:.1%} | {cells} |")
    lines += ["", f"The [method notes]({rel}/README.md), [selection]({rel}/selection.json), "
        f"[completion audit]({rel}/validation_summary.json), and [publication audit]({rel}/report_update.json) "
        "record all configurations and checks. The publication audit links the derived all-method and joint-comparison "
        "summaries, exact predecessor backups, and the publication chain. Frozen historical summaries are unchanged.", END, ""]
    return source.rstrip() + "\n" + "\n".join(lines)


def retained_spring_runs(previous, spring):
    audited = {key(c): c for c in read(spring / "validation_summary.json")["cases"]}
    require(len(audited) == CASES, "Incomplete retained SPRING audit")
    result = {}
    for case in previous["cases"]:
        arm, checked = case["arms"]["spring_sarah"], audited[key(case)]
        folder = Path(arm["source_directory"])
        require(arm["run_id"] == checked["run_id"] and str(folder) == checked["source_directory"],
                "Retained SPRING pairing differs from its completion audit")
        raw = (folder / "trace.json").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == checked["trace_sha256"], "A retained SPRING trajectory changed")
        result[(*key(case), "spring_sarah")] = dict(trace=json.loads(raw))
    return result


def plot(report, plotting, analysis, runs, corrected, previous, slices, sizes, joint, stage, spring_runs):
    for k in list(runs):
        if k[3] in (*METHODS, "spring_sarah"):
            del runs[k]
    replacement = {**corrected, **spring_runs}
    for k, run in replacement.items():
        runs[k] = dict(run, trace=[dict(row, oracle_calls=row.get("oracle_calls", row["gradient_calls"]))
                                   for row in run["trace"] if max(row["forward_views"], row["backward_views"]) <= CAP])
    groups = analysis.group_map(previous["design"])
    plotting.add_joint_traces(runs, joint["cases"], groups, CAP)
    variants = analysis.VARIANTS
    analysis.VARIANTS = {"sframes": ("SFRAMES (orig.)", variants["sframes"][1]),
        "tuned_sframes": ("SFRAMES (joint)", "#333333"), "sframes_full": variants["sframes_full"],
        **{m: variants[m] for m in ("spring_sarah", "palm")},
        "prox_sarah": ("ProxSARAH-v3", variants["prox_sarah"][1])}
    for run in runs.values():
        for row in run["trace"]:
            row.setdefault("oracle_calls", row["gradient_calls"])
    plotting.AXES = (("iteration", "Iterations"), ("oracle_calls", "Oracle calls"), ("forward_views", "Views processed"))
    plotting.plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 7, "axes.edgecolor": "0.55", "axes.linewidth": .6, "pdf.fonttype": 42})
    for size in sizes:
        plotting.plot_size(analysis, runs, slices, groups, size, CAP,
                           previous["budgets"]["original_forward_view_cap"], stage)
    require(len(runs) == FIGURE_RUNS, "The corrected figure cohort must contain 448 physical traces")
    return len(runs), sum(len(run["trace"]) for run in runs.values())


def compile_pdf(stage, report):
    """Keep the standard compiler and validate text independent of PDF line breaks."""
    with (stage / "build.log").open("w") as log:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", report.STEM + ".tex"],
                       cwd=stage, check=True, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(["pdftotext", "-layout", report.STEM + ".pdf", "paper.txt"], cwd=stage, check=True)
    extracted = " ".join((stage / "paper.txt").read_text().split())
    require(all(t in extracted for t in ("ProxSARAH", "Figure 2:", "Table 1:", "reproducibility appendix")),
            "The compiled paper is missing required content")
    require(not any(t in extracted for t in ("??", "Figure 3:", "Table 2:")), "Unexpected or unresolved paper references")
    require(r"Overfull \hbox" not in (stage / (report.STEM + ".log")).read_text(), "The paper has overfull text")
    info = subprocess.check_output(["pdfinfo", str(stage / (report.STEM + ".pdf"))], text=True)
    return int(re.search(r"^Pages:\s*(\d+)$", info, re.M).group(1))


def destinations(report, paper, spring, feasibility):
    return {**{name: paper / name for name in report.ASSETS}, report.APPENDIX: paper.parent / report.APPENDIX,
            "feasibility_report_update.json": feasibility / "report_update.json",
            "spring_report_update.json": spring / "report_update.json"}


def finish(root, report, audit, targets, pending):
    stage, backup = Path(audit["stage"]), Path(audit["backup"])
    require(stage.parent == root and stage.name.startswith("report_stage_") and backup == root / "report_previous",
            "Invalid publication stage or backup")
    require(set(targets) == set(audit["published_sha256"]), "Publication destinations changed")
    for name, target in targets.items():
        old, new = audit["original_sha256"][name], audit["published_sha256"][name]
        require(digest(backup / name) == old and digest(stage / name) == new, f"Stage or backup changed: {name}")
        require(digest(target) in (old, new), f"Unrelated live edit prevents journal recovery: {name}")
    try:
        for name, target in targets.items():
            report.atomic_copy(stage / name, target)
        require(all(digest(path) == audit["published_sha256"][name] for name, path in targets.items()),
                "Publication does not match the validated stage")
        report.atomic_copy(pending, root / "report_update.json")
    except BaseException:
        for name, target in targets.items():
            report.atomic_copy(backup / name, target)
        raise
    pending.unlink(missing_ok=True)
    return audit


def build(root, spring, feasibility, paper, publish=False, check=False):
    report = module("baseline_base_report", feasibility / "report.py")
    validation, plan, previous, frozen_joint, tune, legacy, predecessor, anchors = inputs(root, spring, feasibility, report)
    targets = destinations(report, paper, spring, feasibility)
    final, pending = root / "report_update.json", root / "report_update.pending.json"
    if final.exists() or pending.exists():
        audit = read(final if final.exists() else pending)
        require(audit["anchors"] == anchors and audit["report_pdf"] == str(paper / f"{report.STEM}.pdf"),
                "Existing publication has different inputs")
        if final.exists():
            require(all(digest(path) == audit["published_sha256"][name] for name, path in targets.items()),
                    "The completed publication changed")
            for name in ("corrected_summary", "corrected_joint_summary"):
                require(digest(audit[name]) == audit[name + "_sha256"], "A derived summary changed")
            return audit
        return finish(root, report, audit, targets, pending) if publish else dict(status="publication_pending", stage=audit["stage"])
    prior_f = read(feasibility / "report_update.json")
    require(prior_f["summary_sha256"] == digest(feasibility / "summary.json")
            and prior_f["selection_sha256"] == digest(feasibility / "selection.json"), "Frozen joint anchors changed")
    spring_targets = {k: v for k, v in targets.items() if k != "spring_report_update.json"}
    require(all(digest(path) == predecessor["published_sha256"][name] for name, path in spring_targets.items()),
            "Live report differs from its SPRING publication audit")
    before = {name: digest(path) for name, path in targets.items()}
    plotting = module("baseline_corrected_plots", feasibility / "report_trajectories.py")
    analysis, runs, slices, sizes = plotting.evaluation_traces(report.LEGACY, previous["design"], CAP)
    corrected = corrected_runs(root, validation, plan, previous, analysis)
    require({key(c) for c in previous["cases"]} == {key(c) for c in frozen_joint["cases"]}, "Cohorts differ")
    if check:
        return dict(status="ready", corrected_runs=len(corrected), anchors=anchors)
    helpers = module("baseline_joint_helpers", feasibility / "analysis.py")
    updated = replace_summary(previous, corrected, analysis, anchors)
    joint = joint_summary(frozen_joint, corrected, analysis, helpers, anchors)
    selection = read(root / "selection.json")
    stage = Path(tempfile.mkdtemp(prefix="report_stage_", dir=root))
    write(stage / "corrected_summary.json", updated)
    write(stage / "corrected_joint_summary.json", joint)
    (stage / report.BODY).write_text(source_text(report, updated, legacy, tune, joint, selection))
    for name in (f"{report.STEM}.tex", report.LINK, "joint_selection.pdf", "joint_selection.png"):
        shutil.copyfile(paper / name, stage / name)
    (stage / report.APPENDIX).write_text(appendix_text((paper.parent / report.APPENDIX).read_text(), root, paper, joint, selection))
    spring_runs = retained_spring_runs(previous, spring)
    count, observations = plot(report, plotting, analysis, runs, corrected, previous, slices, sizes, joint, stage, spring_runs)
    pages = compile_pdf(stage, report)
    assets = {name: digest(stage / name) for name in report.ASSETS}
    successor = dict(publication_audit=str(root / "report_update.json"),
        corrected_summary=str(stage / "corrected_summary.json"), corrected_summary_sha256=digest(stage / "corrected_summary.json"),
        corrected_joint_summary=str(stage / "corrected_joint_summary.json"),
        corrected_joint_summary_sha256=digest(stage / "corrected_joint_summary.json"),
        scope="Replace PALM and ProxSARAH only, recomputing comparisons; preserve corrected SPRING and all SFRAMES arms.")
    next_f = copy.deepcopy(prior_f)
    next_f.update(successor=successor, stage=str(stage), pages=pages, figures=2, tables=1, evaluation_runs=count,
                  native_iteration_observations=observations, staged_sha256=assets, published_sha256=assets)
    write(stage / "feasibility_report_update.json", next_f)
    next_s = copy.deepcopy(predecessor)
    next_s.update(successor=successor, pages=pages, figures=2, tables=1, evaluation_runs=count,
                  native_iteration_observations=observations,
                  published_sha256={name: digest(stage / name) for name in spring_targets})
    write(stage / "spring_report_update.json", next_s)
    audit = dict(status="completed_published" if publish else "staged_validated", anchors=anchors,
        stage=str(stage), backup=str(root / "report_previous"), report_pdf=str(paper / f"{report.STEM}.pdf"),
        figures=2, tables=1, pages=pages, evaluation_runs=count, corrected_baseline_runs=len(corrected),
        native_iteration_observations=observations, original_sha256=before,
        published_sha256={name: digest(stage / name) for name in targets},
        **{k: v for k, v in successor.items() if k != "publication_audit"})
    write(stage / "report_validation.json", audit)
    require(all(digest(path) == before[name] for name, path in targets.items()), "Live report changed during staging")
    require(all(digest(value["path"]) == value["sha256"] for value in anchors.values()), "A scientific input changed during staging")
    if not publish:
        return audit
    backup = root / "report_previous"
    backup.mkdir(exist_ok=True)
    for name, path in targets.items():
        if not (backup / name).exists():
            report.atomic_copy(path, backup / name)
        require(digest(backup / name) == before[name], f"Backup belongs to a different publication: {name}")
    report.atomic_copy(stage / "report_validation.json", pending)
    return finish(root, report, audit, targets, pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--spring-root", type=Path, default=SPRING)
    parser.add_argument("--feasibility-root", type=Path, default=FEASIBILITY)
    parser.add_argument("--paper", type=Path, default=PAPER)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root, spring, feasibility, paper = (p.resolve() for p in (args.root, args.spring_root, args.feasibility_root, args.paper))
    # Descendant first, then its predecessors: all correction publishers use
    # this order, preventing a successor from racing a predecessor publication.
    with (root / "report_update.lock").open("a") as own, (spring / "report_update.lock").open("a") as sp, \
            (feasibility / "report_update.lock").open("a") as fp:
        for lock in (own, sp, fp):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(build(root, spring, feasibility, paper, args.publish, args.check), indent=2))


if __name__ == "__main__":
    main()
