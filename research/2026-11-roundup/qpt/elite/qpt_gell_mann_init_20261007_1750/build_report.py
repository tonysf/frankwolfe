#!/usr/bin/env python3
"""Render a two-page PDF only from a complete, passed eight-case audit.

This report builder does not run optimizers, infer missing endpoints, or alter
the gate.  It accepts the schema emitted by qpt_gell_mann_diagnostic's audited
paired-reference workflow.  Existing output paths are never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAPER_URL = "https://arxiv.org/html/2312.01311v1#S5"
METHODS = ("printed-adafgd", "frames", "armijo-fgd")
LABELS = {"printed-adafgd": "Printed adaFGD", "frames": "FRAMES", "armijo-fgd": "Armijo control"}
EXPECTED_COUNTS = {"printed-adafgd": 4, "frames": 2, "armijo-fgd": 2}
GATE = {"fidelity_min": 0.99, "rmse_max": 1e-6, "normalized_tp_max": 0.001}
INK, BLUE, MUTED = "#172632", "#155b85", "#536571"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _gate_pass(case):
    metrics = case["endpoint_physical"]
    conditions = (("process_fidelity", GATE["fidelity_min"], True),
                  ("rmse", GATE["rmse_max"], False),
                  ("normalized_tp_violation", GATE["normalized_tp_max"], False))
    physical = all(_finite_number(metrics.get(key)) and
                   (metrics[key] >= limit if lower else metrics[key] <= limit)
                   for key, limit, lower in conditions)
    return case["status"] in {"complete", "converged"} and physical


def validate_inputs(audit, plan, plan_hash):
    require(audit.get("kind") == "qpt_gell_mann_audit" and audit.get("schema_version") == 1,
            "Unsupported audit schema")
    require(audit.get("audit_passed") is True and audit.get("expected_trials") == 8
            and audit.get("audited_trials") == 8 and audit.get("partial_mode") is False,
            "A complete passed audit of all eight cases is required")
    require(audit.get("missing_trial_ids") == []
            and audit.get("trial_ids_without_inventory_checksum") == [],
            "Missing or inventory-unverified cases cannot be reported")
    require(plan.get("kind") == "qpt_gell_mann_initialization_diagnostic"
            and plan.get("schema_version") == 1 and plan.get("channels") == [0, 1],
            "Unsupported frozen plan")
    shared = plan["shared"]
    require(shared["gate"] == GATE and shared["n_qubits"] == 2 and shared["rank"] == 1
            and shared["active_rows"] == 128 and shared["batch_size"] == 32
            and shared["observation_mode"] == "noiseless"
            and shared["initialization_basis"] == "gell-mann"
            and shared["frames_penalty_exponent"] == 0.25,
            "Plan differs from this report's frozen design or unchanged gates")
    provenance = audit["provenance"]
    require(provenance["plan_sha256"] == plan_hash
            and provenance["source_commit"] == plan["source"]["commit"]
            and provenance["campaign_id"] == plan["campaign_id"],
            "Audit and plan provenance differ")
    require(audit["prior_audit"]["sha256"] == plan["prior"]["audit_sha256"],
            "Prior audit fingerprint differs")
    inventories = audit["inventories"]
    require(len(inventories) == 2 and {item["channel"] for item in inventories} == {0, 1}
            and all(item["state"] == "complete" for item in inventories),
            "Both channel inventories must be complete")
    expected = {trial["id"]: trial for trial in plan["trials"]}
    cases = audit["cases"]
    require(len(expected) == len(plan["trials"]) == len(cases) == 8
            and {case["trial_id"] for case in cases} == set(expected),
            "Cases do not match the eight frozen trial identities")
    require(Counter(case["method"] for case in cases) == EXPECTED_COUNTS,
            "Method case counts differ from the frozen design")
    for case in cases:
        trial = expected[case["trial_id"]]
        require(case["trial"] == trial and case["method"] == trial["method"]
                and case["channel"] == trial["channel"]
                and case["baseline"]["trial_id"] == case["trial_id"],
                "A paired old/new trial identity differs")
        require(trial["tau"] == 2.0, "A trial differs from the reported norm bound")
        if trial["method"] == "printed-adafgd":
            require(trial["alpha"] in (5e-6, 5e-4) and trial["eta_scale"] == 1.0
                    and trial["row_epochs"] == 100000,
                    "Printed-rule trial differs from the reported design")
        elif trial["method"] == "frames":
            require(trial["alpha"] == 5e-5 and trial["row_epochs"] == 100000
                    and trial["step_schedule"] == {"scale": 4.0, "offset": 4.0, "exponent": 1.0, "cap": 1.0},
                    "FRAMES trial differs from the reported design")
        else:
            require(trial["alpha"] == 5e-4 and trial["row_epochs"] == 10000
                    and trial["armijo_gradient_atol"] == 1e-12
                    and trial["armijo_gradient_rtol"] == 1e-10,
                    "Armijo trial differs from the reported design")
        for endpoint in (case, case["baseline"]):
            require(endpoint["status"] in {"complete", "converged", "failed"},
                    "Unknown endpoint status")
            require(endpoint["gate"]["pass"] is _gate_pass(endpoint),
                    "Archived endpoint gate disagrees with physical metrics/status")
    for method in METHODS:
        selected = [case for case in cases if case["method"] == method]
        recorded = audit["method_counts"][method]
        require(recorded["status_counts"] == dict(Counter(case["status"] for case in selected))
                and recorded["official_endpoint_gate_passes"] == sum(case["gate"]["pass"] for case in selected),
                "Audit method summary differs from endpoints")
    return sorted(cases, key=lambda case: (METHODS.index(case["method"]),
                                          case["trial"]["alpha"], case["channel"]))


def _page(number, subtitle):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.065, 0.963, "QPT / INITIALIZATION SENSITIVITY", fontsize=10,
             color=BLUE, weight="bold")
    fig.text(0.065, 0.930, subtitle, fontsize=19, color=INK, weight="bold")
    fig.text(0.065, 0.033, "Calibration channels 0 and 1 only • archived results preserved",
             fontsize=8, color=MUTED)
    fig.text(0.935, 0.033, str(number) + " / 2", ha="right", fontsize=8, color=MUTED)
    return fig


def _paragraph(fig, text, y, *, size=9.4, width=104, color=INK, weight="normal"):
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    fig.text(0.065, y, "\n".join(lines), va="top", fontsize=size, color=color,
             linespacing=1.45, weight=weight)
    return y - len(lines) * size * 1.45 / (11.69 * 72) - 0.016


def _number(value, fidelity=False):
    if not _finite_number(value):
        return "undefined"
    return f"{value:.7f}" if fidelity else f"{value:.2e}"


def render_pdf(audit, plan, cases, hashes, submission, supplement):
    """Return PDF bytes; caller validates input provenance and publishes once."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    buffer = io.BytesIO()
    with PdfPages(buffer, metadata={"Title": "QPT initializer sensitivity: audited calibration pairs",
                                   "Author": "QPT experiment audit", "Subject": "Exploratory calibration; not unseen validation"}) as pdf:
        fig = _page(1, "Eight audited old/new pairs")
        _paragraph(fig, "Only the initial coefficient basis changed: archived Choi-coordinate draws versus "
                   "the same normalized draws mapped through a unit-norm Gell-Mann basis. "
                   "These are eight new runs paired with eight existing endpoints, not sixteen new runs.", 0.885)
        counts = []
        for method in METHODS:
            selected = [case for case in cases if case["method"] == method]
            old = sum(case["baseline"]["gate"]["pass"] for case in selected)
            new = sum(case["gate"]["pass"] for case in selected)
            counts.append(f"{LABELS[method]}: {old}/{len(selected)} → {new}/{len(selected)}")
        fig.text(0.065, 0.797, "Endpoint gate passes: old Choi → new Gell-Mann", fontsize=10,
                 weight="bold", color=BLUE)
        fig.text(0.065, 0.772, "   |   ".join(counts), fontsize=9, color=INK)
        rows = []
        for case in cases:
            for which, endpoint in (("Choi", case["baseline"]), ("GM", case)):
                metrics = endpoint["endpoint_physical"]
                method = {"printed-adafgd": "Q", "frames": "F", "armijo-fgd": "A"}[case["method"]]
                rows.append([
                    f"{method} / {case['trial']['alpha']:.0e}" if which == "Choi" else "",
                    str(case["channel"]) if which == "Choi" else "", which,
                    _number(metrics["process_fidelity"], fidelity=True),
                    _number(metrics["rmse"]), _number(metrics["normalized_tp_violation"]),
                    "PASS" if endpoint["gate"]["pass"] else "FAIL",
                    {"complete": "complete", "converged": "converged", "failed": "failed"}[endpoint["status"]],
                ])
        ax = fig.add_axes([0.065, 0.285, 0.87, 0.459])
        ax.axis("off")
        table = ax.table(cellText=rows,
                         colLabels=["Method / α", "ch", "Basis", "Fidelity", "RMSE", "norm. TP", "Gate", "Status"],
                         colWidths=[0.175, 0.035, 0.07, 0.14, 0.135, 0.135, 0.08, 0.13],
                         cellLoc="center", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#ffffff")
            if row == 0:
                cell.set_facecolor(BLUE)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#eef3f6" if ((row - 1) // 2) % 2 == 0 else "#fafbfd")
                cell.set_text_props(color=INK)
                if column == 6:
                    cell.set_text_props(color="#116541" if rows[row - 1][6] == "PASS" else "#8a3f2d", weight="bold")
        y = _paragraph(fig, "Q = Quiroga's printed adaptive update; F = FRAMES; A = Armijo diagnostic control. "
                       "α is the fixed TP coefficient per row for Q/A and the initial coefficient for F. "
                       "Values are rounded for display; gates use unrounded values and terminal status.", 0.262,
                       size=8.7, width=114)
        y = _paragraph(fig, "Unchanged full gate: fidelity ≥ 0.99, full-design RMSE ≤ 10⁻⁶, and normalized "
                       "TP violation ≤ 10⁻³; terminal status must be complete or converged. "
                       "Converged alone is not recovery. Failed terminal runs stay failures.", y,
                       size=8.7, width=114)
        _paragraph(fig, "Only one initializer and one FRAMES sampling stream per channel were tested. "
                   "This two-channel calibration probe does not select a winner or establish robustness. "
                   "No validation/held-out channels were used, no gate was relaxed, and no failed result was discarded.",
                   y, size=8.7, width=114, color=BLUE)
        pdf.savefig(fig)
        plt.close(fig)

        fig = _page(2, "What changed — and what did not")
        y = 0.882
        q_pass = sum(c["gate"]["pass"] for c in cases if c["method"] == "printed-adafgd")
        conclusion = ("The initializer change did not rescue the printed adaptive rule in these four runs. "
                      if q_pass == 0 else
                      f"The new initializer passed the gate in {q_pass}/4 printed-rule runs. ")
        y = _paragraph(fig, "Takeaway. " + conclusion +
                       "This does not settle the missing paper conventions or complete the best-for-best comparison. "
                       "Independent calibration and a frozen validation protocol are still needed before scaling up.",
                       y, color=BLUE, size=9.0, width=110)
        verification = submission["verification"]
        y = _paragraph(fig, f"Execution and checks. Ruche array {submission['job_id']}; all eight archives audited. "
                       f"Full suite: {verification['full_pytest_passed']} passed, {verification['full_pytest_skipped']} skipped; "
                       f"{verification['harness_tests_passed']} harness tests passed. Final-factor, native-endpoint and "
                       f"last-checkpoint metrics agree in {supplement['fully_verified_trials']}/8 cases. "
                       "Code merged locally; not pushed.", y, size=8.7, width=114)
        y = _paragraph(fig, "Frozen setup. Two qubits, rank-one unitary truth, 128 noiseless global-design rows; "
                       "calibration channel seeds 0 and 1. Every method on a channel shares the same truth, data "
                       "and mapped initial factor. Old/new pairs share the exact original coefficient draw and "
                       "all method parameters; only its physical-coordinate basis changes.", y)
        y = _paragraph(fig, "Initialization convention. Independent real/imaginary uniform[0,1) coefficients "
                       "are globally normalized to squared Frobenius norm d=4, then mapped without another "
                       "normalization. The Hilbert–Schmidt-unit basis orders identity, lexicographic symmetric "
                       "pairs, lexicographic antisymmetric pairs (-i above, +i below the diagonal), then normalized "
                       "diagonal generators. The implementation avoids a dense basis-change matrix.", y)
        y = _paragraph(fig, "Independent penalties and budgets. Q keeps α fixed (λ=128α), with η scale 1; F starts at α₀=5×10⁻⁵ "
                       "and increases it as α₀(k+1)^0.25. Q runs at most 100,000 full updates; F runs 400,000 "
                       "batch-32 updates, with γ=4/(k+4) and τ=2. Each complete primary run charges 12.8 million "
                       "nominal optimizer rows, excluding spectral work and checkpoints. Armijo keeps α=5×10⁻⁴, "
                       "has at most 10,000 iterations and charges line-search/terminal-gradient work separately; "
                       "it is not a matched-work primary competitor.", y)
        y = _paragraph(fig, "Interpretation limit. Quiroga's printed adaptive step was not replaced by Armijo. "
                       "The paper does not fully specify the uniform support, initializer normalization or basis "
                       "ordering. This explicit Gell-Mann diagnostic therefore is not an exact reproduction. "
                       "The primary protocol, previous archives and gates remain unchanged; channels 100–107 "
                       "and 1000–1019 remain outside this probe.", y)
        fig.text(0.065, y, "Paper, Section V: " + PAPER_URL, fontsize=8.2, color=BLUE, url=PAPER_URL, va="top")
        y -= 0.038
        y = _paragraph(fig, "Audit evidence. Both inventories are complete; all eight archive pairs passed "
                       "source/configuration checks, independent endpoint and gate recomputation, row-accounting "
                       "checks, and an independent dense Gell-Mann mapping check against the original draw. "
                       "Every baseline was read from its preserved archive, not rerun.", y, size=9.0, width=110)
        fig.text(0.065, y, "Provenance / SHA-256", fontsize=10, color=BLUE, weight="bold", va="top")
        y -= 0.025
        entries = [
            ("Source commit", audit["provenance"]["source_commit"]),
            ("Frozen plan", hashes["plan"]),
            ("Complete audit", hashes["audit"]),
        ]
        for label, value in entries:
            fig.text(0.065, y, label, fontsize=7.6, color=MUTED, va="top")
            fig.text(0.065, y - 0.012, value, fontsize=7.3, family="DejaVu Sans Mono", color=INK, va="top")
            y -= 0.037
        _paragraph(fig, "Individual source/runner/archive hashes and all unrounded endpoints are retained in the audit; "
                   "supplementary_endpoint_check.json records the additional trace-integrity checks. "
                   "Audit created: " + audit["created_utc"], y - 0.002, size=7.7, width=126)
        pdf.savefig(fig)
        plt.close(fig)
    return buffer.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Refusing to overwrite an existing PDF")
    audit_bytes, plan_bytes = args.audit.read_bytes(), args.plan.read_bytes()
    audit, plan = json.loads(audit_bytes), json.loads(plan_bytes)
    hashes = {"audit": sha256_bytes(audit_bytes), "plan": sha256_bytes(plan_bytes),
              "builder": sha256_bytes(Path(__file__).read_bytes()),
              "source_map": sha256_bytes(json.dumps(audit["provenance"]["source_sha256"],
                                                     sort_keys=True, separators=(",", ":")).encode())}
    cases = validate_inputs(audit, plan, hashes["plan"])
    submission = json.loads(args.submission.read_bytes())
    supplement = json.loads(args.supplement.read_bytes())
    require(submission["source_commit"] == audit["provenance"]["source_commit"]
            and submission["plan_sha256"] == hashes["plan"], "Submission provenance differs")
    require(supplement["audit_sha256"] == hashes["audit"]
            and supplement["source_commit"] == submission["source_commit"]
            and supplement["all_comparable_checks_passed"] is True
            and supplement["checked_trials"] == supplement["fully_verified_trials"] == 8
            and supplement["undefined_failed_trial_ids"] == [], "Supplementary endpoint checks incomplete")
    pdf_bytes = render_pdf(audit, plan, cases, hashes, submission, supplement)
    with args.output.open("xb") as stream:
        stream.write(pdf_bytes)
    print(json.dumps({"pdf": str(args.output.resolve()), "pages": 2,
                      "audited_trials": len(cases), "sha256": sha256_bytes(pdf_bytes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
