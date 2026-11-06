#!/usr/bin/env python3
"""Build a two-page, evidence-checked PDF; never rerun an optimizer.

Optional follow-up audits must be complete and use the frozen configuration.
``--check-only`` renders both pages in memory and checks text bounds without
creating a PDF. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parent
COMMIT = "4ce1108a0692edc43b47ef465dbc74a42397765e"
SCREEN_HASH = "ce5135b4e04535bf90fa143cf7372341428fb15217986c8a8443f5536f356074"
GATE = {"fidelity_min": .99, "rmse_max": 1e-6, "normalized_tp_max": .001}
KEYS = {"method", "alpha", "eta_scale", "n_steps", "metrics_frequency"}
PAPER = "https://arxiv.org/html/2312.01311v1#S3.SS2"
REFERENCE47 = "https://arxiv.org/abs/1203.4481"
INK, MUTED, BLUE, ORANGE = "#142c40", "#4b5e6a", "#087e8b", "#b25c28"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def config(case):
    return {key: case["trial"][key] for key in KEYS}


def deficit(physical):
    return max(physical["rmse"] / 1e-6,
               physical["normalized_tp_violation"] / .001,
               max(0., 1. - physical["process_fidelity"]) / .01)


def physical_pass(case):
    p = case["endpoint_physical"]
    return (case["status"] in {"complete", "converged"}
            and all(p.get(k) is not None and math.isfinite(p[k]) for k in
                    ("rmse", "normalized_tp_violation", "process_fidelity"))
            and p["rmse"] <= 1e-6 and p["normalized_tp_violation"] <= .001
            and p["process_fidelity"] >= .99)


def audit(path, expected_channels, expected_count, phase):
    obj = read(path)
    require(obj["kind"] == "qpt_quiroga_tuning_audit" and obj["audit_passed"]
            and not obj["partial_mode"] and not obj["missing_trial_ids"]
            and not obj["trial_ids_without_inventory_checksum"], "Audit is incomplete: " + str(path))
    require(obj["expected_trials"] == obj["audited_trials"] == len(obj["cases"]) == expected_count,
            "Unexpected audit trial count")
    require({c["channel"] for c in obj["cases"]} == set(expected_channels), "Unexpected channel set")
    require(obj["provenance"]["source_commit"] == COMMIT
            and obj["provenance"]["phase"] == phase, "Audit source/phase differs")
    for case in obj["cases"]:
        require(case["gate"]["thresholds"] == GATE
                and case["gate"]["pass"] == physical_pass(case), "Gate does not match physical endpoint")
        path_ = Path(case["json_path"])
        require(digest(path_) == case["json_sha256"]
                and digest(path_.with_suffix(".npz")) == case["npz_sha256"], "Archived result checksum differs")
    return obj


def checked_inputs(args):
    screen_path = ROOT / "audit_screen.json"
    require(digest(screen_path) == SCREEN_HASH, "Screen audit pin differs")
    screen = audit(screen_path, {0, 1}, 50, "calibration")
    selection = read(ROOT / "selection_screen.json")
    require(selection["audit_sha256"] == SCREEN_HASH and selection["gate"] == GATE,
            "Selection is not pinned to this screen")
    selected = selection["methods"]["exact-linesearch-fgd"]["selected_configuration"]
    require(selected == {"method": "exact-linesearch-fgd", "alpha": 5e-5,
                         "eta_scale": 1., "n_steps": 10000, "metrics_frequency": 100},
            "Unexpected selected configuration")
    require(selection["methods"]["printed-adafgd"]["selected_configuration"] is None,
            "Unexpected successful printed setting")
    counts = Counter((c["method"], c["status"], c["gate"]["pass"]) for c in screen["cases"])
    require(counts == Counter({("printed-adafgd", "complete", False): 14,
                               ("printed-adafgd", "failed", False): 30,
                               ("exact-linesearch-fgd", "converged", True): 6}), "Screen statistics changed")
    chosen = [c for c in screen["cases"] if config(c) == selected]
    literal = [c for c in screen["cases"] if c["method"] == "printed-adafgd"
               and c["trial"]["alpha"] == 5e-5 and c["trial"]["eta_scale"] == 2]
    require(len(chosen) == len(literal) == 2, "Missing paired trajectories")
    scores = []
    for setting in selection["methods"]["exact-linesearch-fgd"]["ranked_settings"]:
        cases = [c for c in screen["cases"] if config(c) == setting["configuration"]]
        score = max(deficit(c["endpoint_physical"]) for c in cases)
        rows = sum(c["work_accounting"]["verified_rows"] for c in cases)
        require(math.isclose(score, setting["worst_channel_deficit"], rel_tol=1e-12)
                and rows == setting["total_charged_rows"], "Selection rank evidence differs")
        scores.append((score, rows, setting["configuration"]["alpha"], setting["configuration"]["eta_scale"]))
    require(scores == sorted(scores), "Selection order is incorrect")
    stages = [("Selected screen", "0–1", chosen)]
    followups = []
    require(args.revalidation is None or args.confirmation is not None,
            "Revalidation reporting also requires its preceding confirmation audit")
    for name, path, channels, phase in (
        ("confirmation", args.confirmation, range(2, 8), "calibration"),
        ("revalidation", args.revalidation, range(100, 108), "validation"),
    ):
        if path is None:
            continue
        obj = audit(path, channels, len(channels), phase)
        require(all(config(c) == selected and c["trial"]["initialization_seed"] == 200000 + 10*c["channel"]
                    for c in obj["cases"]), "Follow-up changed the selected configuration or seed rule")
        plan_path = ROOT / (name + "_stage") / "tuning_plan.json"
        plan = read(plan_path)
        require(digest(plan_path) == obj["provenance"]["plan_sha256"], "Follow-up plan pin differs")
        freeze = plan["frozen_screen_selection"]
        require(freeze["configuration"] == selected and freeze["no_retuning"]
                and freeze["screen_audit_sha256"] == SCREEN_HASH
                and freeze["selection_file_sha256"] == digest(ROOT / "selection_screen.json"),
                "Follow-up selection was not frozen")
        if name == "revalidation":
            require(all(c["gate"]["pass"] for c in followups[0][1]["cases"]),
                    "Revalidation requires successful prior confirmation")
            require(plan["selection"]["calibration_audit_sha256"] == digest(args.confirmation),
                    "Revalidation confirmation-audit pin differs")
        label = "Calibration confirmation" if name == "confirmation" else "Historical revalidation"
        span = "2–7" if name == "confirmation" else "100–107"
        stages.append((label, span, obj["cases"]))
        followups.append((name, obj, path))
    prior_path = Path("/home/tony/qpt_gell_mann_init_20261007_1750/audit_complete.json")
    require(digest(prior_path) == "73ae52860439868eb12fc59be9dc9f4c15d810e9463e0dc17b0811645a32fdd3",
            "Prior GM audit pin differs")
    prior = [c for c in read(prior_path)["cases"] if c["method"] == "printed-adafgd"]
    require(len(prior) == 4 and all(c["completed_steps"] == 100000
            and c["endpoint_physical"]["process_fidelity"] > .99 and not c["gate"]["pass"] for c in prior),
            "Prior 100k context changed")
    submission = read(ROOT / "submission_screen.json")
    require(submission["source_commit"] == COMMIT and submission["job_id"] == "2039775",
            "Screen submission differs")
    require(submission["tests"] == {"full_suite_passed": 597, "full_suite_skipped": 55, "harness_passed": 16},
            "Recorded test counts differ")
    jobs = [("screen", submission["job_id"])]
    for name, _, _ in followups:
        path = ROOT / ("submission_" + name + ".json")
        if path.exists():
            job = read(path)
            require(job["source_commit"] == COMMIT, "Follow-up submission source differs")
            jobs.append((name, str(job["job_id"])))
    return screen, selected, chosen, literal, stages, followups, jobs


class Page:
    def __init__(self, number, title, subtitle):
        self.fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        self.texts = []
        self.text(.07, .955, title, size=16.5 if len(title) > 47 else 19, weight="bold")
        self.text(.07, .918, subtitle, size=9.5, color=MUTED)
        self.text(.07, .027, "QPT adaptive-step tuning • 8 October 2026 • all prior artifacts retained",
                  size=8, color=MUTED)
        self.text(.93, .027, str(number), size=8, color=MUTED, ha="right")

    def text(self, x, y, value, size=10.2, color=INK, **kwargs):
        obj = self.fig.text(x, y, value, va="top", fontsize=size, color=color,
                            family="DejaVu Sans", **kwargs)
        self.texts.append(obj)
        return obj

    def paragraph(self, y, value, size=10.2, width=98, color=INK):
        lines = textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)
        self.text(.07, y, "\n".join(lines), size=size, color=color, linespacing=1.35)
        return y - len(lines) * (size * 1.35 / (11.69 * 72)) - .014

    def heading(self, y, value):
        self.text(.07, y, value, size=11.8, weight="bold")
        return y - .032

    def check(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for obj in self.texts:
            box = obj.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            require(box.x0 >= .02 and box.x1 <= .98 and box.y0 >= .012 and box.y1 <= .99,
                    "Report text outside page bounds: " + obj.get_text()[:70])


def scientific_range(cases, key):
    values = [c["endpoint_physical"][key] for c in cases]
    if any(v is None or not math.isfinite(v) for v in values):
        return "nonfinite endpoint present"
    return f"{min(values):.2e}–{max(values):.2e}"


def make_pages(inputs):
    screen, selected, chosen, literal, stages, followups, jobs = inputs
    first = Page(1, "The modified step works; the printed rule is unresolved",
                 "Fixed-penalty Quiroga-style FGD • audited Ruche experiments • two-qubit diagnostic")
    y = first.paragraph(.870,
        "Exact factor-ray line search recovers the target under the unchanged physical gate. "
        "It changes the adaptive step, not the objective or gradient direction. This is a successful "
        "explicit variant—not a reproduction of published adaFGD.", size=11.1, width=89)
    y = first.heading(y, "What was tested")
    y = first.paragraph(y,
        "Two qubits; rank-one Haar-unitary channels; 128 noiseless Born observations; normalized "
        "Gell-Mann initialization. Same data and starting factor within each channel. Gate: fidelity "
        "≥ 0.99, full-design RMSE ≤ 1e-6, normalized TP violation ≤ 1e-3. No projection or relaxed gate.")
    y = first.paragraph(y,
        "New screen: 50 runs on calibration channels 0/1. Literal rule: four fixed penalties × six "
        "constant step multipliers, excluding two previously archived settings per channel. "
        "Exact-ray variant: three fixed penalties per channel; maximum 10,000 updates for every run.")
    y = first.heading(y, "Screen result: literal and modified methods remain separate")
    for label, detail, color in (
        ("Printed adaptive ratio", "0/44 pass: 14 completed, 30 numerical failures", ORANGE),
        ("Exact factor-ray variant", "6/6 pass and converge; all three penalties succeed on both channels", BLUE),
    ):
        first.text(.07, y, label, size=10.1, weight="bold", color=color)
        first.text(.35, y, detail, size=9.5)
        y -= .027
    y -= .008
    y = first.paragraph(y,
        "Frozen choice: α = 5e-5, λ = 128α = 0.0064, fixed throughout each trial. Selection minimized "
        "the worst-channel normalized gate deficit; it did not select the fastest wall time. "
        f"Selected screen RMSE: {scientific_range(chosen, 'rmse')}; updates: "
        f"{min(c['completed_steps'] for c in chosen)}–{max(c['completed_steps'] for c in chosen)}.")
    y = first.heading(y, "Frozen follow-up evidence")
    first.text(.07, y, "Stage / channels", size=9.2, weight="bold")
    first.text(.44, y, "Gate", size=9.2, weight="bold")
    first.text(.54, y, "Endpoint RMSE range", size=9.2, weight="bold")
    first.text(.82, y, "Updates", size=9.2, weight="bold")
    y -= .025
    for label, span, cases in stages:
        passes = sum(c["gate"]["pass"] for c in cases)
        first.text(.07, y, label + " / " + span, size=9.0)
        first.text(.44, y, f"{passes}/{len(cases)}", size=9.0)
        first.text(.54, y, scientific_range(cases, "rmse"), size=8.7)
        first.text(.82, y, f"{min(c['completed_steps'] for c in cases)}–{max(c['completed_steps'] for c in cases)}", size=9.0)
        y -= .024
    if not followups:
        y = first.paragraph(y - .004, "Additional confirmation/revalidation audits were not supplied; no follow-up success is claimed.", size=9.2, color=MUTED)
    else:
        y -= .009
    frozen_cases = [case for _, _, cases in stages for case in cases]
    finite_fidelity = [c["endpoint_physical"]["process_fidelity"] for c in frozen_cases]
    finite_tp = [c["endpoint_physical"]["normalized_tp_violation"] for c in frozen_cases]
    require(all(v is not None and math.isfinite(v) for v in finite_fidelity + finite_tp),
            "Frozen-setting physical range contains undefined values")
    y = first.paragraph(y,
        f"Frozen setting: {sum(c['gate']['pass'] for c in frozen_cases)}/{len(frozen_cases)} gates pass; "
        f"{sum(c['status'] == 'converged' for c in frozen_cases)}/{len(frozen_cases)} converge. "
        f"Minimum fidelity {min(finite_fidelity):.12g} (rounded); "
        f"maximum normalized TP {max(finite_tp):.2e}.", size=9.4, width=106)
    revalidated = next((obj["cases"] for name, obj, _ in followups if name == "revalidation"), None)
    revalidation_text = (
        "The unchanged setting passes all historical channels 100–107; this is not unseen validation. "
        if revalidated is not None and all(c["gate"]["pass"] for c in revalidated)
        else "Historical revalidation channels 100–107 are not unseen validation. "
        if revalidated is not None else "Historical revalidation 100–107 is not included in this report. "
    )
    y = first.paragraph(y,
        "Channels 2–7 are additional calibration. " + revalidation_text
        + "No retuning; held-out channels 1000–1019 remain untouched.", size=9.5, width=105)
    y = first.heading(y, "Conclusion and remaining boundary")
    y = first.paragraph(y,
        "The fixed-penalty factorized formulation is recoverable here; the tested printed ratio is "
        "the unresolved part. This is not evidence that Quiroga's paper is wrong. No completed "
        "higher-qubit campaign or final best-for-best FRAMES comparison follows from these tests.",
        size=10., width=99)
    require(y > .055, "Page 1 content overlaps footer")

    second = Page(2, "Why the step-size rule is the focus",
                  "Same α = 5e-5 and λ = 0.0064 • paired calibration channels 0/1 • no optimizer reruns")
    ax = second.fig.add_axes([.11, .610, .83, .253])
    for cases, prefix, label, color in (
        (literal, "printed_adafgd", "Printed ratio, c=2", ORANGE),
        (chosen, "exact_linesearch_fgd", "Exact-ray variant", BLUE),
    ):
        for case in sorted(cases, key=lambda c: c["channel"]):
            with np.load(Path(case["json_path"]).with_suffix(".npz"), allow_pickle=False) as z:
                x = z[prefix + "_cumulative_data_row_accesses"] / 1e6
                values = z[prefix + "_rmse"]
            ax.semilogy(x, values, color=color, linestyle="-" if case["channel"] == 0 else "--",
                        linewidth=1.9, label=f"{label}, channel {case['channel']}")
            ax.plot(x[-1], values[-1], "o", color=color, markersize=4)
    ax.axhline(1e-6, color="#4a5964", linestyle=":", linewidth=1.4, label="RMSE gate: 1e-6")
    ax.set(xlabel="Charged optimizer row accesses (millions)", ylabel="Full-design RMSE",
           ylim=(3e-11, .2), xlim=(0, 1.32))
    ax.tick_params(labelsize=9)
    ax.grid(True, which="major", alpha=.18)
    ax.legend(loc="upper right", fontsize=8.4, framealpha=.97)
    ax.spines[["top", "right"]].set_visible(False)
    y = second.paragraph(.553,
        "Iterations do not have equal cost. Exact-ray accounting includes the direction pass, "
        "two polynomial forward passes and each directly evaluated candidate. Charges exclude "
        "TP contractions, spectral-norm work, full-design overhead and checkpoint evaluation; "
        "this is not a wall-time or total-compute equivalence.", size=9.4, width=105)
    y = second.heading(y, "Why no blind 100,000-update extension was selected")
    best = max(c["endpoint_physical"]["rmse"] for c in literal)
    y = second.paragraph(y,
        f"Best jointly stable literal setting: α = 5e-5, c = 2; worst RMSE {best:.5g}, "
        f"still {best/1e-6:,.0f}× the gate after 10,000 updates. Recorded tail slopes imply only "
        "~7e-4 and ~1.1e-3 at 100,000 updates if those rates persisted. These are diagnostic "
        "extrapolations, not guarantees; they do not justify the optional extension.", size=9.6, width=103)
    y = second.paragraph(y,
        "Prior context, not new success: four GM-initialized printed-rule runs already reached "
        "fidelity > 0.99 at 100,000 updates, but all failed the joint RMSE/TP gate. All earlier "
        "results and all numerical failures remain preserved.", size=9.6, width=103)
    y = second.heading(y, "Checks and provenance")
    jobtext = "; ".join(name + " " + job for name, job in jobs)
    y = second.paragraph(y,
        "Source " + COMMIT[:12] + "; full suite: 597 passed, 55 skipped; tuning harness: 16 passed. "
        "Audits verify pins, archives, independent endpoints, gates, initialization, stopping and work. "
        "Screen Slurm exit 2 denotes archived method failures, not missing results. "
        "Ruche jobs: " + jobtext + ".",
        size=9.2, width=108)
    y = second.heading(y, "Relation to the published work")
    # Under 100 words of paper-derived explanation; no verbatim excerpts.
    y = second.paragraph(y,
        "Quiroga and Kyrillidis give an adaptive ratio only up to proportionality. The multiplier, "
        "experimental TP weight and stopping choices are not fully specified for exact reproduction. "
        "Their cited reference [47] treats hard-thresholding matrix methods. Our positive-factor-ray "
        "quartic line search is neither the printed adaFGD step nor Matrix ALPS.",
        size=9.2, width=109)
    second.text(.07, y, "Quiroga & Kyrillidis, arXiv:2312.01311, §III-B", size=8.5, color=BLUE, url=PAPER)
    second.text(.53, y, "[47] Kyrillidis & Cevher, arXiv:1203.4481", size=8.5, color=BLUE, url=REFERENCE47)
    y -= .027
    second.text(.07, y, "Screen audit SHA256: " + SCREEN_HASH, size=6.9, color=MUTED)
    require(y > .055, "Page 2 content overlaps footer")
    return [first, second]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--revalidation", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "QPT_QUIROGA_ADAPTIVE_TUNING_20261008.pdf")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only:
        require(not args.output.exists(), "Refusing to overwrite an existing PDF")
    pages = make_pages(checked_inputs(args))
    for page in pages:
        page.check()
    if args.check_only:
        print(json.dumps({"layout_checked": True, "pages": len(pages), "pdf_created": False}))
    else:
        with args.output.open("xb") as stream:
            with PdfPages(stream, metadata={"Title": "Quiroga adaptive-step tuning: modified-step recovery",
                                           "Author": "QPT experiment audit",
                                           "Subject": "Separate literal-rule tuning and exact-ray variant; not paper-exact reproduction"}) as pdf:
                for page in pages:
                    pdf.savefig(page.fig)
        print(json.dumps({"pdf": str(args.output), "sha256": digest(args.output), "pages": len(pages)}))
    for page in pages:
        plt.close(page.fig)


if __name__ == "__main__":
    main()
