#!/usr/bin/env python3
"""Render a verified, complete 60-case pilot report without running experiments.

Usage: build_resolution_pdf.py --audit-json audit_complete.json \
    --status-json status.json --output NEW_REPORT.pdf [--followup-audit FOLLOWUP.json]

Status fields: checked_utc (offset-bearing ISO timestamp), job_id,
scheduler_state, optional notes, and optional armijo_fix object with status,
tests_passed, tests_skipped, review_status, ruche_run. Missing state is reported
as unavailable. armijo_fix may be omitted while retaining the last explicitly
reported development stage: 556 passed / 55 skipped; independent review approved.
Optional followup fields: status, job_id, scheduler_state, source_commit. These
describe a separate prospective calibration run, never the completed pilot.
All numerical outcomes come only from fully verified audit JSONs. A third page
is emitted only when --followup-audit supplies a complete, successful 12-case
audit matching its frozen plan and this prior 60-case audit. Otherwise two pages.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import textwrap
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
PRIOR_AUDIT = Path("/home/tony/qpt_quiroga_status_20261006_1529/validation_audit.json")
COMMIT = "c7a8c9789b7814ede1c95e53eef13bfe1dd58c63"
FOLLOWUP_COMMIT = "0f7857e94f6b16ac07733be248f68aa6904d287b"
FOLLOWUP_PLAN = Path("/home/tony/qpt_quiroga_followup_20261006_1820/followup_plan.json")
PAPER = "https://arxiv.org/html/2312.01311v1"
INK, MUTED, BLUE, AMBER = "#172b3a", "#4e6070", "#18546b", "#825017"
METHODS = ("printed-adafgd", "frames", "armijo-fgd")
LABELS = {"printed-adafgd": "Quiroga printed adaFGD", "frames": "FRAMES",
          "armijo-fgd": "Armijo diagnostic control"}
DEFAULT_FIX = {
    "status": "Armijo-only stopping fix merged locally on stochastic-frames at 0f7857e; not applied to the old pilot",
    "repository_status": "clean, ahead by 4 commits, not pushed; Ruche uses a separate pinned checkout",
    "tests_passed": 556, "tests_skipped": 55,
    "review_status": "independently approved", "ruche_run": "not yet reported",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display(value):
    if value is None:
        return "not available in this snapshot"
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)


def checked_time(value):
    if not value:
        return "check time not available"
    stamp = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("checked_utc must include its UTC offset")
    return stamp.astimezone(ZoneInfo("Europe/Paris")).strftime("%d %B %Y, %H:%M %Z")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def metric(value):
    return f"{value:.4g}" if finite(value) else "N/A"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_cases(audit, plan, plan_hash):
    require(audit.get("kind") == "qpt_diagnostic_result_audit", "Unexpected audit kind")
    require(audit.get("audit_passed") is True and audit.get("partial_mode") is False,
            "A successful non-partial audit is required")
    require(audit.get("all_results_present") is True
            and audit.get("all_inventory_checksums_verified") is True,
            "All results and inventory checksums must be verified")
    require(audit.get("expected_trials") == audit.get("audited_trials") == 60,
            "The complete pilot must contain exactly 60 audited trials")
    require(not audit.get("missing_trial_ids") and not audit.get("trial_ids_without_inventory_checksum"),
            "Missing trial or inventory checksum evidence")
    require(audit["provenance"]["plan_sha256"] == plan_hash
            and audit["provenance"]["source_commit"] == COMMIT,
            "Audit provenance differs from the frozen source or plan")
    cases = audit["cases"]
    expected = {trial["id"]: trial for trial in plan["trials"]}
    require(len(cases) == 60 and len({case["trial_id"] for case in cases}) == 60,
            "Case count or unique IDs differ")
    require({case["trial_id"] for case in cases} == set(expected), "Frozen trial IDs differ")
    for case in cases:
        require(case["checks"] == "passed" and case["trial"] == expected[case["trial_id"]],
                "Case does not match its frozen trial")
        require(case["status"] in {"complete", "converged", "failed"}, "Unknown terminal status")
        require(not (case["status"] == "failed" and case["gate"]["pass"]),
                "A failed trial cannot pass its terminal gate")
    require(Counter(c["method"] for c in cases) == Counter({"printed-adafgd": 42, "frames": 12, "armijo-fgd": 6}),
            "Method counts differ")
    require(Counter(c["channel"] for c in cases) == Counter({0: 30, 1: 30}), "Channel counts differ")
    return cases



def verified_followup(audit, plan, prior_hash):
    require(audit.get("kind") == "qpt_followup_audit"
            and audit.get("audit_passed") is True and audit.get("partial_mode") is False,
            "The optional follow-up needs a successful complete audit")
    require(audit.get("expected_trials") == audit.get("audited_trials") == 12
            and not audit.get("missing_trial_ids")
            and not audit.get("trial_ids_without_inventory_checksum"),
            "The optional follow-up must contain all 12 verified cases")
    require(audit["provenance"]["source_commit"] == plan["source"]["commit"] == FOLLOWUP_COMMIT
            and audit["provenance"]["plan_sha256"] == digest(FOLLOWUP_PLAN),
            "Follow-up source or frozen plan differs")
    require(audit["prior_audit"]["sha256"] == prior_hash,
            "Follow-up is not linked to the supplied 60-case audit")
    inventories = audit["inventories"]
    require(len(inventories) == 2 and {v["channel"] for v in inventories} == {0, 1}
            and all(v["state"] == "complete" for v in inventories),
            "Both follow-up inventories must be complete")
    cases = audit["cases"]
    expected = {trial["id"]: trial for trial in plan["trials"]}
    require(len(cases) == len({c["trial_id"] for c in cases}) == 12
            and {c["trial_id"] for c in cases} == set(expected), "Follow-up IDs differ")
    for case in cases:
        require(case["trial"] == expected[case["trial_id"]], "Follow-up trial differs")
        require(case["status"] in {"complete", "converged", "failed"}, "Unknown follow-up status")
        require(not (case["status"] == "failed" and case["gate"]["pass"]),
                "A failed follow-up cannot pass its gate")
        if case["status"] == "converged":
            require(case["method"] == "armijo-fgd" and case.get("stopping_audit") is not None,
                    "Converged follow-up needs its independent Armijo stopping check")
    require(Counter(c["method"] for c in cases) ==
            Counter({"printed-adafgd": 4, "frames": 2, "armijo-fgd": 6}), "Follow-up method counts differ")
    require(Counter(c["channel"] for c in cases) == Counter({0: 6, 1: 6}), "Follow-up channel counts differ")
    return cases


class Page:
    def __init__(self, title, subtitle, number, plan_hash, audit_hash, *, total_pages=2, source_commit=COMMIT):
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        self.y = .875
        self.fig.text(.08, .953, title, fontsize=16, weight="bold", color=INK, va="top")
        self.fig.text(.08, .918, subtitle, fontsize=9, color=MUTED, va="top")
        self.fig.text(.08, .049,
                      f"Page {number}/{total_pages} | source {source_commit[:12]} | plan {plan_hash[:12]} | audit {audit_hash[:12]}",
                      fontsize=7.1, color=MUTED)

    def heading(self, title):
        self.fig.text(.08, self.y, title, fontsize=11.4, weight="bold", color=BLUE, va="top")
        self.y -= .027

    def paragraph(self, text, *, color=INK, size=9.4, width=107):
        lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
        self.fig.text(.08, self.y, "\n".join(lines), fontsize=size, color=color,
                      va="top", linespacing=1.26)
        self.y -= len(lines) * size / (11.69 * 72) * 1.26 + .010

    def table(self, headings, rows, widths, *, font=8.2, row_height=.025):
        height = row_height * (len(rows) + 1)
        ax = self.fig.add_axes([.08, self.y-height, .84, height])
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headings, colWidths=widths,
                         bbox=[0, 0, 1, 1], cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(font)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d3dee5")
            cell.set_linewidth(.45)
            cell.PAD = .045
            cell.set_text_props(color=INK)
            if r == 0:
                cell.set_facecolor("#edf4f7")
                cell.set_text_props(weight="bold", color=BLUE)
        self.y -= height + .015

    def finish(self, pdf):
        require(self.y >= .075, f"Report text overflows page: final position {self.y}")
        pdf.savefig(self.fig)
        plt.close(self.fig)


def build(audit_path, status_path, output, followup_audit_path=None):
    audit = json.loads(audit_path.read_text())
    status = json.loads(status_path.read_text())
    require(isinstance(status, dict), "Snapshot must be a JSON object")
    plan_path = ROOT / "diagnostic_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash, audit_hash = digest(plan_path), digest(audit_path)
    cases = verified_cases(audit, plan, plan_hash)
    followup_audit, followup_plan, followup_cases = None, None, None
    if followup_audit_path is not None:
        followup_audit = json.loads(followup_audit_path.read_text())
        followup_plan = json.loads(FOLLOWUP_PLAN.read_text())
        followup_cases = verified_followup(followup_audit, followup_plan, audit_hash)
    page_count = 3 if followup_cases is not None else 2
    indexed = {case["trial_id"]: case for case in cases}
    prior = json.loads(PRIOR_AUDIT.read_text())
    require(prior["status"] == "all_24_archives_verified_against_remote_inventory_hashes",
            "Prior validation audit is not verified")
    q, f = (prior["methods"][name] for name in ("printed-adafgd", "frames"))
    require(not output.exists(), f"Refusing to overwrite {output}")
    subtitle = "Verified snapshot: " + checked_time(status.get("checked_utc", audit.get("created_utc")))
    fix = dict(DEFAULT_FIX)
    require(status.get("armijo_fix") is None or isinstance(status["armijo_fix"], dict),
            "armijo_fix must be an object")
    fix.update(status.get("armijo_fix") or {})
    followup = status.get("followup") or {}
    require(isinstance(followup, dict), "followup must be an object")

    with PdfPages(output, metadata={"Title": "QPT: complete calibration diagnostic pilot",
                                   "Subject": "Two calibration channels; not final configuration selection"}) as pdf:
        page = Page("QPT: latest follow-up complete" if followup_cases is not None
                    else "QPT: 60-case diagnostic pilot complete",
                    subtitle, 1, plan_hash, audit_hash, total_pages=page_count)
        if followup_cases is not None:
            latest = {method: [c for c in followup_cases if c["method"] == method] for method in METHODS}
            passed = {method: sum(c["gate"]["pass"] for c in subset) for method, subset in latest.items()}
            primary_note = ("Neither primary method passes the full gate in this follow-up. "
                            if passed["printed-adafgd"] == passed["frames"] == 0
                            else "Passing calibration cases are not a robust, confirmed comparison. ")
            page.heading("Latest: all 12 follow-up cases checked")
            page.paragraph(
                f"Full-gate passes: printed Quiroga {passed['printed-adafgd']}/4; "
                f"FRAMES {passed['frames']}/2; Armijo control {passed['armijo-fgd']}/6. "
                + primary_note + "A successful Armijo control is not a Quiroga result.")
            def completed_range(method, key):
                values = [c["endpoint_physical"].get(key) for c in latest[method]
                          if c["status"] in {"complete", "converged"}]
                values = [v for v in values if finite(v)]
                return (metric(min(values)) + " to " + metric(max(values))) if values else "N/A"
            page.paragraph("Completed FRAMES RMSE: " + completed_range("frames", "rmse")
                           + "; printed Quiroga RMSE: " + completed_range("printed-adafgd", "rmse")
                           + ". Required: <= 1e-6. These remain two calibration channels, not unseen confirmation.",
                           size=9.1, color=BLUE)
            page.paragraph("Earlier 60-case pilot (table below): the same corrected n = 2, full-128-row, "
                           "noiseless rank-one problem, calibration channels 0 and 1, original initial factors "
                           "and one FRAMES stream. All earlier outcomes remain unchanged.", size=9.1)
        else:
            page.heading("Complete and checked; the comparison is not yet resolved")
            page.paragraph("All 60 frozen cases and both channel inventories are verified. Ruche job "
                           + display(status.get("job_id")) + "; scheduler: "
                           + display(status.get("scheduler_state")) + ". No optimizer equations changed in this pilot.")
            page.paragraph("This remains the corrected n = 2, full-128-row, noiseless rank-one problem. "
                           "Only calibration channels 0 and 1 were used: 21 printed-rule, 6 FRAMES and "
                           "3 Armijo cases per channel, with the original shared initial factors and one "
                           "FRAMES sampling stream. These are diagnostic cases, not independent robustness trials.")
        rows = []
        for method in METHODS:
            subset = [case for case in cases if case["method"] == method]
            good = sum(case["status"] in {"complete", "converged"} for case in subset)
            failed = sum(case["status"] == "failed" for case in subset)
            gate = sum(case["gate"]["pass"] for case in subset)
            rows.append([LABELS[method], str(len(subset)), str(good), str(failed), str(gate)])
        page.table(["Earlier 60-case pilot", "Cases", "Normal", "Failed", "Final gate"], rows,
                   [.48, .13, .13, .13, .13], font=8.6, row_height=.031)
        page.paragraph("The gate still requires fidelity >= 0.99, full-design RMSE <= 1e-6 and normalized "
                       "TP error <= 0.001 together. A failed terminal status is never relabeled as a gate "
                       "pass, even if an earlier or final checkpoint meets the physical thresholds.", size=9.1, color=MUTED)
        page.heading("What the pilot is resolving")
        page.paragraph("No sign/factor discrepancy was found in the reviewed printed update and TP-gradient equations. Quiroga's "
                       "printed rule is scale-sensitive: large fixed TP penalties can make updates stiff, "
                       "while larger step multipliers at low penalties had been underexplored. The expanded "
                       "grid tests that hypothesis without changing the printed update.")
        page.paragraph("For FRAMES, the true channel is feasible with zero measurement and TP loss. "
                       "Averaging boundary atoms can shrink the factor norm while trace-normalized fidelity "
                       "looks excellent. Faster averaging, lower initial penalty and modest radius slack "
                       "were tested separately; slack-radius runs are domain ablations, not identical-domain comparisons.")
        page.paragraph("Armijo is a recoverability control, not Quiroga or a third primary competitor. "
                       "Its actual line-search work is not budget-matched; physically accurate checkpoints "
                       "do not erase later terminal failures.", size=9.1)
        page.heading("Armijo-only stopping fix: development status")
        page.paragraph(display(fix.get("status")) + ". Full suite: "
                       + display(fix.get("tests_passed")) + " passed, " + display(fix.get("tests_skipped"))
                       + " skipped. Review: " + display(fix.get("review_status"))
                       + ". Ruche confirmation: " + display(fix.get("ruche_run")) + ".", color=BLUE, size=9.1)
        page.paragraph("Repository status: " + display(fix.get("repository_status")) + ".", size=8.4, color=MUTED, width=119)
        page.paragraph("This optional stopping change does not alter printed Quiroga or "
                       "FRAMES, and cannot retroactively change any archived pilot status. Earlier source-fixed "
                       f"validation remains unchanged: Quiroga {q['gate_passes']}/{q['unique_outcomes']} and "
                       f"FRAMES {f['gate_passes']}/{f['unique_outcomes']} passed the full gate.", size=9.1)
        page.heading("Separate follow-up: calibration, not new validation")
        page.paragraph("Job " + display(followup.get("job_id"))
                       + ": " + display(followup.get("scheduler_state"))
                       + ". Source " + str(followup.get("source_commit", "unavailable"))[:12] + ".",
                       size=8.7, color=BLUE, width=115)
        page.paragraph("All 12 follow-up cases are audited; full endpoints and work accounting are on page 3. "
                       "Best-for-best selection and fresh confirmation remain unfinished."
                       if followup_cases is not None else display(followup.get("status")),
                       size=8.7, color=MUTED, width=115)
        page.finish(pdf)

        page = Page("Per-channel endpoints, not a robustness claim", subtitle, 2, plan_hash, audit_hash, total_pages=page_count)
        page.heading("FRAMES: every pilot arm, separated by channel")
        arms = [("original-anchor", "Original"), ("fast-average", "Faster averaging"),
                ("lower-alpha", "Lower initial penalty"), ("lower-alpha-fast", "Lower + faster"),
                ("tau2p02", "Radius 2.02"), ("tau2p1", "Radius 2.1")]
        rows = []
        for arm, label in arms:
            pair = [indexed[f"c{channel}-f-{arm}"] for channel in (0, 1)]
            vals = [[metric(c["endpoint_physical"].get("rmse")),
                     metric(c["endpoint_physical"].get("normalized_tp_violation"))] for c in pair]
            gates = "/".join("P" if c["gate"]["pass"] else "F" for c in pair)
            rows.append([label, *vals[0], *vals[1], gates])
        page.table(["Arm", "RMSE c0", "TP c0", "RMSE c1", "TP c1", "Gate 0/1"],
                   rows, [.30, .145, .145, .145, .145, .12], font=7.6, row_height=.024)
        page.paragraph("P/F means terminal full-gate pass/fail, including fidelity and method status. "
                       "Both channels are calibration cases, not unseen confirmation. One stream per channel "
                       "does not resolve sampling variability or establish which arm is robust.", size=8.8, color=MUTED, width=114)
        page.heading("Printed Quiroga: descriptive fidelity maxima")
        rows = []
        for channel in (0, 1):
            eligible = [case for case in cases if case["method"] == "printed-adafgd"
                        and case["channel"] == channel and case["status"] in {"complete", "converged"}
                        and finite(case["endpoint_physical"].get("process_fidelity"))]
            if not eligible:
                rows.append([str(channel), "None complete", "N/A", "N/A", "N/A", "N/A"])
                continue
            best = max(eligible, key=lambda c: c["endpoint_physical"]["process_fidelity"])
            p = best["endpoint_physical"]
            setting = f"a={metric(best['trial']['alpha'])}; e={metric(best['trial']['eta_scale'])}"
            rows.append([str(channel), setting, metric(p.get("process_fidelity")), metric(p.get("rmse")),
                         metric(p.get("normalized_tp_violation")), "P" if best["gate"]["pass"] else "F"])
        page.table(["Channel", "Fixed alpha; multiplier", "Fidelity", "RMSE", "TP", "Gate"], rows,
                   [.105, .34, .145, .145, .145, .12], font=7.5, row_height=.024)
        page.paragraph("These are merely the largest completed endpoint fidelities on each channel, not "
                       "selected configurations or winners. All 42 printed-rule cases, including failures, "
                       "remain in the complete audit.", size=8.6, color=MUTED, width=116)
        page.heading("Armijo control: all endpoints and retained checkpoint evidence")
        rows = []
        for case in sorted((c for c in cases if c["method"] == "armijo-fgd"),
                           key=lambda c: (c["channel"], -c["trial"]["alpha"])):
            p = case["endpoint_physical"]
            first = case.get("first_recorded_physical_gate_step")
            rows.append([f"c{case['channel']} a={metric(case['trial']['alpha'])}", case["status"],
                         metric(p.get("rmse")), metric(p.get("normalized_tp_violation")),
                         "P" if case["gate"]["pass"] else "F", str(first) if first is not None else "None"])
        page.table(["Channel / fixed alpha", "Status", "RMSE", "TP", "Final gate", "First physical"],
                   rows, [.29, .14, .15, .15, .115, .155], font=7.3, row_height=.024)
        page.paragraph("First physical = first archived iteration meeting all physical thresholds, "
                       "retrospectively; it is not an early-stopped result. Late Armijo failures remain failed "
                       "scientific outcomes, not infrastructure failures or Quiroga successes.", size=8.7, color=AMBER, width=114)
        page.paragraph("All previous calibration/validation evidence is preserved. No held-out data or "
                       "validation retuning was used. The original initialization remains; equivalence to the "
                       "paper's Gell-Mann-basis initialization is still open, so this is not an exact "
                       "reproduction of every published choice. Revised configurations require fresh confirmation.",
                       size=8.7, width=114)
        if followup_cases is not None:
            page.paragraph("Follow-up settings (page 3): Quiroga alpha is fixed, with step multiplier 1. "
                           "FRAMES alpha is the initial 5e-5 and grows as (k+1)^0.25; its averaging step "
                           "is 4/(k+4), radius 2 and batch size 32. Armijo uses full-gradient stopping "
                           "atol = 1e-12 and rtol = 1e-10, separately from the physical gate.",
                           size=8.7, width=114)

        page.fig.text(.08, .069, "Full 60-case evidence: " + audit_path.name, fontsize=7.3,
                      color=BLUE, url=str(audit_path.resolve()))
        page.fig.text(.56, .069, "Primary paper: arXiv:2312.01311v1", fontsize=7.3, color=BLUE, url=PAPER)
        page.finish(pdf)

        if followup_cases is not None:
            followup_hash = digest(followup_audit_path)
            page = Page("Longer-horizon follow-up: all 12 cases",
                        "Audit: " + checked_time(followup_audit.get("created_utc")),
                        3, digest(FOLLOWUP_PLAN), followup_hash,
                        total_pages=page_count, source_commit=FOLLOWUP_COMMIT)
            page.heading("Separate calibration evidence, with the new Armijo-only stop")
            page.paragraph("All 12 follow-up cases and both inventories are verified against the new pinned "
                           "source and the original pilot's data and initial factors. Printed Quiroga and "
                           "FRAMES update equations are unchanged. Only the Armijo control opts into the "
                           "reviewed stationarity stop; recovery still requires a separate physical-gate check.",
                           size=9.2)
            for method in METHODS:
                page.heading(LABELS[method])
                subset = sorted((c for c in followup_cases if c["method"] == method),
                                key=lambda c: (c["channel"], -c["trial"]["alpha"]))
                rows = []
                for case in subset:
                    p = case["endpoint_physical"]
                    first = case.get("first_recorded_physical_gate")
                    label = f"c{case['channel']} a={metric(case['trial']['alpha'])}"
                    rows.append([label, str(case["completed_steps"]), case["status"],
                                 metric(p.get("process_fidelity")), metric(p.get("rmse")),
                                 metric(p.get("normalized_tp_violation")),
                                 "P" if case["gate"]["pass"] else "F",
                                 str(first["step"]) if first is not None else "None"])
                page.table(["Channel / alpha", "End step", "Status", "Fidelity", "RMSE", "TP", "Gate", "First step"],
                           rows, [.215, .11, .13, .11, .12, .12, .075, .12],
                           font=7.0, row_height=.025)
            page.paragraph("End step and first step are native optimizer iterations, not equal units of work "
                           "across methods. Printed Quiroga uses up to 100,000 full-data updates; FRAMES uses "
                           "400,000 batch-32 updates (12.8 million nominal rows each when complete). Armijo "
                           "uses at most 10,000 iterations with separately counted line-search and stopping-check work. "
                           "Primary nominal row counts exclude Quiroga spectral-numerator work and checkpoint "
                           "metrics; they are not CPU-time equivalents.",
                           size=8.9, width=112)
            page.paragraph("First step is the first archived checkpoint meeting all physical thresholds; "
                           "it is retrospective and does not replace terminal status. P/F is the terminal full "
                           "gate, including status, fidelity, RMSE and TP. Old failed Armijo results remain failed.",
                           size=8.9, color=MUTED, width=112)
            page.paragraph("More iterations do not promise recovery. Two reused calibration channels and one "
                           "FRAMES stream do not establish a winner or robustness; no held-out data were used. "
                           "Next: reconcile unresolved paper initialization and step conventions, and predeclare "
                           "a Gell-Mann-basis initialization control before much longer Quiroga runs. "
                           "No error in the published method has been established.",
                           size=9.1, color=AMBER, width=111)
            page.fig.text(.08, .069, "Full follow-up evidence: " + followup_audit_path.name,
                          fontsize=7.3, color=BLUE, url=str(followup_audit_path.resolve()))
            page.finish(pdf)
    return {"pdf": str(output), "sha256": digest(output), "pages": page_count,
            "followup_audit_sha256": digest(followup_audit_path) if followup_audit_path is not None else None,
            "plan_sha256": plan_hash, "audit_sha256": audit_hash,
            "snapshot_sha256": digest(status_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--followup-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.audit_json, args.status_json, args.output, args.followup_audit), sort_keys=True))


if __name__ == "__main__":
    main()



