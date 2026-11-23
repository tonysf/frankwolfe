#!/usr/bin/env python3
"""Produce a concise two-page PDF from the frozen selection and actual snapshot.

Status fields: checked_utc, scheduler_state, completed_paired_trials,
method_failure_pairs, first_trial_sidecar (absolute JSON path or null), note.
No optimization or remote operation is performed.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "selection_n2_full" / "selection_manifest.json"
AUDIT = ROOT / "independent_calibration_audit.json"
PLAN = Path("/home/tony/qpt_quiroga_validation_20261005_2136/validation_plan.json")
COMMIT = "c7a8c9789b7814ede1c95e53eef13bfe1dd58c63"
MANIFEST_SHA = "fa9b5d5287f260296d7ee8fa292f303b389494fe6413d897bd6c60afa06d4178"
METHODS = ("printed-adafgd", "frames")
METRICS = ("process_fidelity", "rmse", "normalized_tp_violation")
INK, MUTED, BLUE, LIGHT, AMBER = "#172b3a", "#4e6070", "#18546b", "#edf4f7", "#825017"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, fidelity=False):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "unavailable"
    return f"{value:.9f}" if fidelity else f"{value:.5g}"


def local_time(utc_text):
    value = datetime.datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    return value.astimezone(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d %H:%M:%S %Z") + \
        " (" + value.astimezone(datetime.timezone.utc).strftime("%H:%M:%S UTC") + ")"


def paragraph(fig, text, y, size=10, color=INK, width=99, spacing=1.3):
    lines = []
    for line in str(text).splitlines():
        lines.extend(textwrap.wrap(line, width=width, break_long_words=True,
                                   break_on_hyphens=False) or [""])
    fig.text(0.085, y, "\n".join(lines), ha="left", va="top", fontsize=size,
             color=color, linespacing=spacing)
    return y - len(lines) * size / (11.69 * 72) * spacing - 0.016


def heading(fig, text, y):
    fig.text(0.085, y, text, fontsize=12, weight="bold", color=BLUE, va="top")
    return y - 0.030


def table(fig, headers, rows, y, height, widths):
    ax = fig.add_axes([0.085, y - height, 0.83, height])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                 loc="center", cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(9.4)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor("#d3dee5")
        cell.set_linewidth(0.45)
        cell.PAD = 0.075
        cell.set_text_props(color=INK)
        if r == 0:
            cell.set_facecolor(LIGHT)
            cell.set_text_props(weight="bold", color=BLUE)
    return y - height - 0.019


def page(title, subtitle, index):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.085, 0.954, title, fontsize=17, color=INK, weight="bold", va="top")
    fig.text(0.085, 0.918, subtitle, fontsize=9.3, color=MUTED, va="top")
    fig.text(0.085, 0.035, "QPT • Quiroga-matched sensing • independently tuned TP penalties",
             fontsize=8, color=MUTED)
    fig.text(0.915, 0.035, f"{index} / 2", fontsize=8, color=MUTED, ha="right")
    return fig


def evidence(args):
    manifest, audit, plan, status = (load(args.manifest), load(args.audit),
                                    load(args.plan), load(args.status_json))
    require(sha(args.manifest) == MANIFEST_SHA, "Frozen selection checksum differs")
    require(manifest["source_commit"] == COMMIT == plan["source"]["commit"], "Source pin differs")
    require(plan["selection"]["manifest_sha256"] == MANIFEST_SHA, "Plan selection differs")
    require(audit["status"] == "all_archives_and_repeated_native_outcomes_verified", "Audit incomplete")
    require(audit["paired_counts"] == {"complete": 345, "method_failure_archived": 7}, "Counts differ")
    require(manifest["validation_evidence"]["paired_trial_count"] == 352
            and manifest["validation_evidence"]["unique_native_outcome_count"] == 360, "Inventory incomplete")
    selected, counts = {}, {}
    for method in METHODS:
        rows = manifest["candidate_outcomes_and_ranking"][method]
        require(all(row["gate_pass_rate"] == 0 for row in rows), "Calibration gate results changed")
        candidate_id = manifest["printed_config_id" if method == "printed-adafgd" else "frames_config_id"]
        row = next(row for row in rows if row["config_id"] == candidate_id)
        reference = next(row for row in audit["rankings"][method] if row["config_id"] == candidate_id)
        require(row["rank"] == 1, "Selected candidate differs")
        for index, metric in enumerate(METRICS, 2):
            value = row["physical_ranking_channel_medians"][metric]["value"]
            require(math.isclose(value, reference["score"][index], rel_tol=1e-12), "Independent metrics differ")
        components = {metric: sum(bool(o["gate_components"][metric]) for o in row["outcomes"])
                      for metric in METRICS}
        require(components == reference["component_pass_counts"], "Independent gate-component counts differ")
        selected[method], counts[method] = row, components
    require(plan["channels"] == list(range(100, 108))
            and plan["shared"]["row_epochs"] == 10000, "Validation design changed")
    require(isinstance(status.get("checked_utc"), str) and status["checked_utc"], "Status needs checked_utc")
    for field in ("completed_paired_trials", "method_failure_pairs"):
        if field in status:
            require(isinstance(status[field], int) and 0 <= status[field] <= 24, "Invalid trial count")
    first = None
    if status.get("first_trial_sidecar"):
        first = load(status["first_trial_sidecar"])
        require(first["provenance"]["git"]["commit"] == COMMIT, "Snapshot source differs")
        require(first["comparison"]["split"] == "validation", "Snapshot is not validation")
        selection = first["comparison"]["selection_provenance"]
        require(selection["selection_manifest_sha256"] == MANIFEST_SHA
                and selection["printed_config_id"] == manifest["printed_config_id"]
                and selection["frames_config_id"] == manifest["frames_config_id"], "Snapshot selection differs")
    return manifest, audit, plan, status, selected, counts, first


def calibration_page(manifest, selected, counts, checked):
    fig = page("Calibration complete; validation is separate", "Evidence checked " + checked, 1)
    y = heading(fig, "352 / 352 calibration archives verified", 0.881)
    y = paragraph(fig, "345 paired runs completed normally; 7 numerical-failure pairs were archived and retained. "
                  "After verifying repeated endpoints and deduplicating: 240 Quiroga outcomes (233 complete, "
                  "7 failed) and 120 FRAMES outcomes (all complete). No candidate passed the full calibration gate.", y)
    y = heading(fig, "Independent settings now frozen", y)
    q, f = (manifest["selected_candidates"][m] for m in METHODS)
    y = paragraph(fig, f"Quiroga: fixed alpha_Q = {q['alpha']:g}, lambda_Q = 128 × alpha_Q = {128*q['alpha']:g}; "
                  f"adaptive-step multiplier = {q['eta_scale']:g}.\n"
                  f"FRAMES: alpha_F,0 = {f['alpha0']:g}, beta_0 = {1/(2*f['alpha0']):g}, p = 0.25; "
                  "alpha_F,k = alpha_F,0 × (k + 1)^0.25 increases during optimization.", y)
    y = heading(fig, "Selected-candidate calibration endpoints", y)
    labels = {"process_fidelity": "Process fidelity", "rmse": "Full-design RMSE",
              "normalized_tp_violation": "Normalized TP error"}
    thresholds = {"process_fidelity": "≥ 0.99", "rmse": "≤ 1e-6", "normalized_tp_violation": "≤ 0.001"}
    rows = []
    for metric in METRICS:
        row = [labels[metric]]
        for method in METHODS:
            r = selected[method]
            value = r["physical_ranking_channel_medians"][metric]["value"]
            row.append(number(value, fidelity=metric == "process_fidelity") + "\n" +
                       f"{counts[method][metric]} / {r['unique_native_outcomes']} pass")
        rows.append(row + [thresholds[metric]])
    rows.append(["Full recovery gate", "0 / 8 pass", "0 / 24 pass", "All three"])
    y = table(fig, ["Metric", "Quiroga", "FRAMES", "Gate threshold"], rows, y,
              height=0.205, widths=[0.29, 0.24, 0.24, 0.23])
    y = paragraph(fig, "Medians are over 8 channel means at 2,000 row epochs. FRAMES averages its 3 sampling streams "
                  "within each channel first. Pass counts are native outcomes: 8 deterministic Quiroga runs versus "
                  "24 FRAMES streams—not 24 independent channels. Both use 256,000 nominal row accesses per run; "
                  "this is not an equal-compute comparison.", y, size=9.3, color=MUTED, width=108)
    y = heading(fig, "What the calibration does—and does not—show", y)
    y = paragraph(fig, "The predeclared order was full-gate rate, then number of gate components passed, then fidelity, "
                  "RMSE and TP tie-breakers. With no full-gate passes, Quiroga's TP-feasible settings ranked ahead "
                  "of higher-fidelity, non-TP settings. The selected Quiroga setting is not its maximum-fidelity "
                  "setting: that candidate reached median fidelity 0.105339 but passed no gate components.", y)
    y = paragraph(fig, "FRAMES has high normalized process fidelity here, but its median TP error and RMSE still miss "
                  "the gate. Q alpha is at the upper grid boundary; F alpha0 is at the lower boundary. These are "
                  "best settings within the declared grid—not global optima, successful recovery, or an overall winner. "
                  "Findings are specific to this pinned implementation, sensing, initialization, grid and budget; "
                  "they do not establish that Quiroga's published method fails. Benchmark readiness is not yet established.",

                  y, color=AMBER)
    require(y > 0.060, "Calibration page overflow")
    return fig


def validation_page(args, plan, status, first):
    fig = page("Validation plan and current run snapshot", "Separate channels; frozen choices; no reranking", 2)
    y = heading(fig, "Confirmatory design frozen before validation", 0.881)
    y = paragraph(fig, "24 paired runs: 8 new channels (100–107), one shared initialization per channel and 3 FRAMES "
                  "sampling streams. Data, initial factor, rank and physical gate are shared; TP penalties are not.\n"
                  "Budget: 10,000 row epochs = 1,280,000 nominal row accesses per method; 10,000 Quiroga updates "
                  "and 40,000 FRAMES updates. Checkpoints every 100 row epochs.\n"
                  "No selection or retuning on validation outcomes. Held-out channels remain untouched.", y)
    y = heading(fig, "Scheduler / artifact snapshot", y)
    completed, failures = status.get("completed_paired_trials"), status.get("method_failure_pairs")
    done_text = "not checked" if completed is None else f"{completed} / 24"
    fail_text = "not checked" if failures is None else str(failures)
    y = paragraph(fig, f"Ruche array job: {args.job_id}   |   State: {status.get('scheduler_state', 'not supplied')}\n"
                  f"Checked: {local_time(status['checked_utc'])}\n"
                  f"Archived paired trials: {done_text}   |   Archived numerical-failure pairs: {fail_text}\n"
                  "Per task: cpu_short, 2 CPUs, 1 GB, 20 minutes; array concurrency 1.", y)
    if status.get("note"):
        y = paragraph(fig, status["note"], y, size=9.2, color=MUTED, width=109)
    if first is not None:
        y = heading(fig, "First downloaded validation trial—not an aggregate", y)
        y = paragraph(fig, first["comparison"]["trial_id"], y, size=8.8, color=MUTED, width=110)
        rows = []
        for method, label in zip(METHODS, ("Quiroga", "FRAMES")):
            endpoint, gate = first["methods"][method], first["comparison"]["method_gate"][method]
            rows.append([label, endpoint["status"], number(endpoint["process_fidelity"], fidelity=True),
                         number(gate["metric_values"]["rmse"]), number(endpoint["normalized_tp_violation"]),
                         "pass" if gate["passed"] else "fail"])
        y = table(fig, ["Method", "Status", "Fidelity", "RMSE", "TP error", "Gate"], rows, y,
                  height=0.105, widths=[0.15, 0.16, 0.21, 0.18, 0.18, 0.12])
        q = first["methods"]["printed-adafgd"]
        detail = (f"Quiroga failed numerically after {q['completed_steps']} updates; the failed endpoint and valid "
                  "FRAMES result were preserved. " if q["status"] == "failed" else "")
        f_components = first["comparison"]["method_gate"]["frames"]["component_passes"]
        if (f_components["process_fidelity"] and f_components["normalized_tp_violation"]
                and not f_components["rmse"]):
            detail += "FRAMES passes fidelity and TP, but misses the RMSE threshold. "

        y = paragraph(fig, detail + "A single trial cannot establish the validation pass rate. A Slurm exit 2 "
                      "can mean archived numerical failure, not missing artifacts; failed outcomes remain counted.",
                      y, size=9.2, color=AMBER, width=109)
    else:
        y = paragraph(fig, "No downloaded validation sidecar was supplied. No scientific result is inferred from "
                      "queue state alone.", y, color=AMBER)
    y = heading(fig, "Reproducibility / preserved evidence", y)
    for label, fingerprint, path in (
        ("Source", COMMIT[:12], "/home/tony/frankwolfe"),
        ("Selection", sha(args.manifest)[:16], str(args.manifest)),
        ("Frozen validation plan", sha(args.plan)[:16], str(args.plan)),
        ("Independent audit", sha(args.audit)[:16], str(args.audit))):
        y = paragraph(fig, label + ": " + fingerprint + "\n" + path,
                      y, size=8, color=MUTED, width=116, spacing=1.2)
    y = paragraph(fig, "Earlier aborted runs remain preserved. The c7a8c97 archival repair did not change optimizer "
                  "equations, penalties or gate thresholds. Checks: 9 selector regression tests and 13 validation "
                  "wrapper tests passed, with independent reviews.", y, size=9, color=MUTED, width=112)
    require(y > 0.060, "Validation page overflow")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--plan", type=Path, default=PLAN)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an existing PDF; choose a fresh output path")
    manifest, audit, plan, status, selected, counts, first = evidence(args)
    figures = [calibration_page(manifest, selected, counts, local_time(status["checked_utc"])),
               validation_page(args, plan, status, first)]
    with PdfPages(args.output, metadata={"Title": "Quiroga / FRAMES: calibration and validation update",
                                        "Subject": "Frozen-grid calibration and separate validation"}) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    print(json.dumps({"pdf": str(args.output), "pages": 2, "sha256": sha(args.output),
                      "source_commit": COMMIT, "selection_manifest_sha256": MANIFEST_SHA}, sort_keys=True))


if __name__ == "__main__":
    main()
