"""Build the n=8 continuation report on Elite without scientific imports."""
from __future__ import print_function

import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "n8_continue_200k_20260923_01"
PARENT = Path("/home/tony/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/n8_step0p5_rho0p6/summary.json")
ASSETS = ROOT / "report_assets"
HTML = ROOT / "qpt_n8_continuation_20260923.html"
ASSETS.mkdir(exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


parent = read(PARENT)
summary = read(RUN / "summary.json")
verification = read(RUN / "verification.json")
status = read(RUN / "status.json")
proposal = read(RUN / "proposal.json")
provenance = read(RUN / "provenance.json")

assert status["status"] == "completed"
assert verification["status"] == "verified" and verification["primary_target_met"]
assert summary["trajectory"][0]["step"] == 148000
assert summary["trajectory"][-1]["step"] == 200000
for key in ("fidelity", "measurement_loss", "tp_violation"):
    assert math.isclose(parent["trajectory"][-1][key], summary["trajectory"][0][key], rel_tol=1e-12, abs_tol=1e-14)

recorded = read(RUN / "checksums.json")
manifest_checks = {}
for name, expected in sorted(recorded.items()):
    actual = sha(RUN / name)
    manifest_checks[name] = {"recorded": expected, "actual": actual, "matches": expected == actual}
assert [name for name, row in manifest_checks.items() if not row["matches"]] == ["slurm-1939620.out"]
assert recorded["slurm-1939620.out"] == hashlib.sha256(b"").hexdigest()

collection = {
    "status": "verified_and_backed_up",
    "job_id": "1939620",
    "scheduler_state": "COMPLETED",
    "scheduler_exit_code": "0:0",
    "scheduler_elapsed": "00:12:17",
    "scheduler_start": "2026-09-24T10:38:40+02:00",
    "scheduler_end": "2026-09-24T10:50:57+02:00",
    "scheduler_node": "ruche-gpu15",
    "elite_snapshot": str(RUN),
    "all_elite_files_matched_remote_sha256_at_collection": True,
    "internal_manifest": manifest_checks,
    "internal_manifest_note": "All scientific artifacts match. Slurm stdout was hashed while empty inside the wrapper, then Slurm appended the final COMPLETE line after wrapper finalization; the copied final stdout matches Ruche.",
    "report_inputs": {
        "parent_summary": {"path": str(PARENT), "sha256": sha(PARENT)},
        "continuation_summary": {"path": str(RUN / "summary.json"), "sha256": sha(RUN / "summary.json")},
        "verification": {"path": str(RUN / "verification.json"), "sha256": sha(RUN / "verification.json")},
        "status": {"path": str(RUN / "status.json"), "sha256": sha(RUN / "status.json")},
        "proposal": {"path": str(RUN / "proposal.json"), "sha256": sha(RUN / "proposal.json")},
        "provenance": {"path": str(RUN / "provenance.json"), "sha256": sha(RUN / "provenance.json")},
    },
}
(ROOT / "collection_verification.json").write_text(json.dumps(collection, indent=2) + "\n")

BLUE = "#176B9A"
ORANGE = "#BF651D"
INK = "#253343"
MUTED = "#617183"
GRID = "#DCE4EA"


def ticks_linear(lo, hi, count=5):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, count - 1)
    power = 10 ** math.floor(math.log10(raw))
    fraction = raw / power
    nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 2.5 if fraction <= 2.5 else 5 if fraction <= 5 else 10
    step = nice * power
    start = math.floor(lo / step) * step
    values = []
    value = start
    while value <= hi + step * 0.25:
        if value >= lo - step * 0.25:
            values.append(value)
        value += step
    return values


def fmt(value, metric):
    if metric == "fidelity":
        return "{:.0f}".format(value * 100)
    if metric == "measurement_loss":
        return "{:.4f}".format(value)
    if value >= 10:
        return "{:.0f}".format(value)
    if value >= 1:
        return "{:.1f}".format(value)
    return "{:.2g}".format(value)


def chart_svg(path, x_min, x_max, zoom=False):
    width, height = 1120, 340
    panel_w, gap = 340, 30
    left, top, bottom = 62, 46, 55
    plot_h = height - top - bottom
    metrics = [
        ("fidelity", "Fidelity (%)"),
        ("measurement_loss", "Sampled loss"),
        ("tp_violation", "TP residual (log scale)"),
    ]
    old = [row for row in parent["trajectory"] if x_min <= row["step"] <= x_max]
    new = [row for row in summary["trajectory"] if x_min <= row["step"] <= x_max]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="340" viewBox="0 0 1120 340">',
        '<rect width="1120" height="340" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;fill:%s}.small{font-size:11px}.label{font-size:12px}.title{font-size:14px;font-weight:bold}</style>' % INK,
    ]
    for index, (metric, title) in enumerate(metrics):
        x0 = index * (panel_w + gap) + left
        values = [row[metric] for row in old + new]
        if metric == "fidelity":
            y_lo = 0.965 if zoom else 0.0
            y_hi = 1.003
            y_ticks = ticks_linear(y_lo, y_hi, 5)
            transform = lambda y: y
        elif metric == "tp_violation":
            positive = [v for v in values if v > 0]
            y_lo = 10 ** math.floor(math.log10(min(positive)))
            y_hi = 10 ** math.ceil(math.log10(max(positive)))
            y_ticks = [10 ** p for p in range(int(math.log10(y_lo)), int(math.log10(y_hi)) + 1)]
            transform = lambda y: math.log10(y)
        else:
            y_lo = min(values) * 0.995
            y_hi = max(values) * 1.005
            y_ticks = ticks_linear(y_lo, y_hi, 5)
            transform = lambda y: y
        ty_lo, ty_hi = transform(y_lo), transform(y_hi)
        sx = lambda x: x0 + panel_w * (x - x_min) / (x_max - x_min)
        sy = lambda y: top + plot_h * (ty_hi - transform(y)) / (ty_hi - ty_lo)
        parts.append('<text class="title" x="%g" y="20" text-anchor="middle">%s</text>' % (x0 + panel_w / 2, title))
        for xt in range(int(math.ceil(x_min / 50000.0) * 50000), int(x_max) + 1, 50000):
            px = sx(xt)
            parts.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="1"/>' % (px, top, px, top + plot_h, GRID))
            parts.append('<text class="small" x="%g" y="%g" text-anchor="middle">%gk</text>' % (px, top + plot_h + 18, xt / 1000))
        for yt in y_ticks:
            if yt < y_lo or yt > y_hi:
                continue
            py = sy(yt)
            parts.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="1"/>' % (x0, py, x0 + panel_w, py, GRID))
            parts.append('<text class="small" x="%g" y="%g" text-anchor="end">%s</text>' % (x0 - 7, py + 4, fmt(yt, metric)))
        if metric == "fidelity" and y_lo <= 0.99 <= y_hi:
            py = sy(0.99)
            parts.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="1.4" stroke-dasharray="2,5"/>' % (x0, py, x0 + panel_w, py, MUTED))
            parts.append('<text class="small" x="%g" y="%g">99%%</text>' % (x0 + panel_w - 28, py - 5))
        parts.append('<rect x="%g" y="%g" width="%g" height="%g" fill="none" stroke="#B7C3CE"/>' % (x0, top, panel_w, plot_h))
        for rows, color in ((old, BLUE), (new, ORANGE)):
            points = " ".join("{:.2f},{:.2f}".format(sx(row["step"]), sy(row[metric])) for row in rows)
            if points:
                parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>' % (points, color))
        parts.append('<text class="label" x="%g" y="%g" text-anchor="middle">Completed iterations</text>' % (x0 + panel_w / 2, height - 8))
    parts.extend([
        '<line x1="360" y1="329" x2="390" y2="329" stroke="%s" stroke-width="3"/><text class="small" x="397" y="333">Original winning run</text>' % BLUE,
        '<line x1="565" y1="329" x2="595" y2="329" stroke="%s" stroke-width="3"/><text class="small" x="602" y="333">Exact continuation</text>' % ORANGE,
        '</svg>',
    ])
    path.write_text("\n".join(parts))


chart_svg(ASSETS / "trajectory_full.svg", 0, 200000, zoom=False)
chart_svg(ASSETS / "trajectory_late.svg", 100000, 200000, zoom=True)

p0 = parent["trajectory"][-1]
p1 = summary["trajectory"][-1]
meta = summary["metadata"]
first_crossing = summary["first_saved_crossing_0p99"]
rows_final = summary["final_checkpoints_191k_200k"]


def e(value):
    return html.escape(str(value))


def table(headers, rows):
    head = "".join("<th>%s</th>" % e(value) for value in headers)
    body = "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % value for value in row) for row in rows)
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (head, body)


checkpoint_table = table(
    ["Saved step", "Fidelity", "Sampled loss", "TP residual"],
    [["{:,}".format(row["step"]), "{:.5f}%".format(100 * row["fidelity"]), "{:.9g}".format(row["measurement_loss"]), "{:.7g}".format(row["tp_violation"])] for row in rows_final],
)

document = """<!doctype html>
<html><head><meta charset="utf-8"><style>
@page { size: A4 landscape; margin: 13mm 14mm 12mm 14mm; }
body { font-family: 'DejaVu Sans', sans-serif; color: #253343; font-size: 9.5pt; line-height: 1.34; }
.page { page-break-after: always; position: relative; min-height: 165mm; }
.page:last-child { page-break-after: auto; }
h1 { color: #253343; font-size: 21pt; margin: 0 0 2mm 0; border-top: 1px solid #DCE4EA; padding-top: 4mm; }
h2 { color: #176B9A; font-size: 12pt; margin: 4mm 0 2mm 0; }
.kicker { color: #617183; font-size: 8.5pt; margin-bottom: 5mm; }
.callout { color: #176B9A; font-size: 13pt; font-weight: bold; margin: 3mm 0 4mm 0; }
.small { color: #617183; font-size: 8pt; }
.twocol { display: table; width: 100%%; table-layout: fixed; }
.col { display: table-cell; width: 50%%; vertical-align: top; padding-right: 5mm; }
table { border-collapse: collapse; width: 100%%; margin: 2mm 0 4mm 0; font-size: 8.3pt; }
th { background: #253343; color: white; text-align: left; padding: 1.7mm; }
td { padding: 1.5mm 1.7mm; border-bottom: 0.3mm solid #DCE4EA; }
tbody tr:nth-child(even) { background: #F5F7F9; }
img.chart { width: 100%%; height: auto; margin: 1mm 0 1mm 0; }
.footer { position: absolute; bottom: 0; left: 0; right: 0; border-top: 1px solid #DCE4EA; color: #617183; font-size: 7.5pt; padding-top: 2mm; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; }
</style></head><body>

<section class="page">
<h1>Eight-qubit QPT: exact continuation to 200,000 steps</h1>
<div class="kicker">Usual noise σ=0.05 · batch 65,536 · requested exponents 3/4, 1/4, 1/2 · one fixed dataset and seed</div>
<div class="callout">The requested-exponent trajectory reaches 99%% at the saved 174,000-step checkpoint and finishes at 99.13929%%.</div>
%s
<p><b>Answer to the experiment question.</b> More iterations alone were sufficient for this trajectory to cross 99%%. Every saved checkpoint from 191,000 through 200,000 also remained above 99%%, so the endpoint is not a lone saved-threshold excursion.</p>
<p>From 148,000 to 200,000 steps, factor fidelity improves by %.4f percentage points, the fixed-sample loss changes from %.9g to %.9g, and the complete-factor TP residual improves from %.7g to %.7g. TP is reported separately; fidelity is not a physical-feasibility certificate.</p>
<h2>Fixed scientific scope</h2>
<p>This is one exact restart, not a new sweep. It preserves the parent factor, momentum estimate, PCG64 sampling state, global schedule index, dataset, source snapshot and all optimizer settings. The segment adds 52,000 updates and 3,407,872,000 optimizer-row draws, ending at 13,107,200,000 total draws.</p>
<div class="footer">Structured stochastic-FRAMES / QPT · job 1939620 · 24 September 2026 · page 1</div>
</section>

<section class="page">
<h1>Combined saved trajectory</h1>
<div class="kicker">Solid blue: original winning run, steps 0–148,000 · solid orange: exact continuation, steps 148,000–200,000 · no smoothing or fitted curve</div>
<img class="chart" src="report_assets/trajectory_full.svg">
<p class="small">Fidelity is |c†u|²/(||c||²||u||²), computed from complete factors and invariant to nonzero phase/scale. Sampled measurement loss uses the fixed 512-row metric sample with seed 12345. TP is ||T(uu†)−I||<sub>F</sub> from the complete factor and is plotted logarithmically. The dotted line is the 99%% reference.</p>
<div class="footer">Saved evidence at 1,000-step cadence · page 2</div>
</section>

<section class="page">
<h1>Late trajectory and threshold persistence</h1>
<div class="kicker">Zoomed to 100,000–200,000 steps; the scientific traces remain unsmoothed</div>
<img class="chart" src="report_assets/trajectory_late.svg">
<div class="twocol"><div class="col"><h2>Final ten saved checkpoints</h2>%s</div>
<div class="col"><h2>Observed threshold behavior</h2>
<p>First saved F≥99%%: <b>%s steps</b>.</p>
<p>All saved checkpoints 191,000–200,000 ≥99%%: <b>yes</b>.</p>
<p>The prior descriptive projection suggested a crossing near 170–175k; the observed saved crossing is 174k. This agreement does not turn the earlier fit into a confidence interval or validate extrapolation to other seeds.</p>
<p>The TP residual continues improving, from %.7g at restart to %.7g at the endpoint.</p></div></div>
<div class="footer">Threshold assessment uses saved 1,000-step checkpoints · page 3</div>
</section>

<section class="page">
<h1>Schedules, budgets, and runtime definitions</h1>
%s
<h2>Measured continuation runtime</h2>
%s
<p><b>Definitions.</b> Runner time begins after setup and includes compilation, host preparation, transfers, metrics and rolling restart checkpointing. Optimizer time measures GPU update execution. Host preparation accumulates background preparation time and overlaps GPU work under prefetch, so these phase timers must not be added as disjoint totals. Slurm elapsed time is the whole allocation.</p>
<p class="small">Virtual dataset M=24⁸=110,075,314,176 rows, sampled uniformly with replacement and never materialized. The noisy-truth expectation σ²/2=0.00125 is not a measured convergence floor. No fidelity early stopping was used.</p>
<div class="footer">Exact continuation configuration · page 4</div>
</section>

<section class="page">
<h1>Independent verification and provenance</h1>
%s
<h2>Interpretation and limitations</h2>
<p>The independent verifier recomputed fidelity and TP from the complete saved factor using NumPy-only contractions. Saved fidelity %.16g agrees with independently computed %.16g (matrix form %.16g); saved TP %.16g agrees with independently computed %.16g. Truth TP residual is %.3g.</p>
<p>This is one Haar-unitary channel, one fixed noisy dataset, one initialization and one optimizer sampling seed. It answers the bounded continuation question for this trajectory only; it does not establish a success probability, uncertainty interval, universal scale optimum, or physical-feasibility certificate.</p>
<p><b>Backup integrity.</b> All 11 copied files matched Ruche byte-for-byte by SHA-256 at collection. The wrapper’s internal manifest matches every scientific artifact. Its Slurm stdout entry records the empty-file hash because the wrapper computed checksums before Slurm appended the final COMPLETE line; the finalized stdout was separately matched between Ruche and Elite and the preserved manifest was not rewritten.</p>
<p class="small">Frozen scientific source: <code>%s</code><br>Final archive SHA-256: <code>%s</code><br>Resumed factor SHA-256: <code>%s</code></p>
<div class="footer">Verified result and one-seed limitations · page 5</div>
</section>
</body></html>""" % (
    table(["Checkpoint", "Fidelity", "Sampled loss", "TP residual"], [
        ["Parent, 148,000", "{:.5f}%".format(100 * p0["fidelity"]), "{:.9g}".format(p0["measurement_loss"]), "{:.7g}".format(p0["tp_violation"])],
        ["Continuation, 200,000", "{:.5f}%".format(100 * p1["fidelity"]), "{:.9g}".format(p1["measurement_loss"]), "{:.7g}".format(p1["tp_violation"])],
    ]),
    100 * (p1["fidelity"] - p0["fidelity"]), p0["measurement_loss"], p1["measurement_loss"], p0["tp_violation"], p1["tp_violation"],
    checkpoint_table, "{:,}".format(first_crossing), p0["tp_violation"], p1["tp_violation"],
    table(["Quantity", "Exact value"], [
        ["Step size γₖ", "min(1, 0.5/(k+10)<sup>3/4</sup>)"],
        ["Smoothing βₖ", "10<sup>7</sup>/(k+1)<sup>1/4</sup>; no cap"],
        ["Momentum ρₖ", "min(1, 0.6/(k+4)<sup>1/2</sup>)"],
        ["Segment / total steps", "148,000→200,000 / 200,000"],
        ["Batch / rank / τ / precision", "65,536 / 1 / 20 / 64-bit"],
        ["Backend / prefetch / metric", "product-state / yes / sampled 512 rows every 1,000"],
        ["Seeds", "channel, noise, initialization, sampling: 0; metric: 12345"],
    ]),
    table(["Timer", "Measured value", "Scope"], [
        ["Slurm elapsed", "12 min 17 s", "Whole allocation; COMPLETED, exit 0:0"],
        ["Runner", "{:.3f} s ({:.2f} min)".format(meta["wall_seconds"], meta["wall_seconds"] / 60), "Continuation runner after setup"],
        ["Optimizer", "{:.3f} s ({:.2f} min)".format(meta["optimizer_seconds"], meta["optimizer_seconds"] / 60), "52,000 GPU updates"],
        ["Host preparation", "{:.3f} s".format(meta["host_preparation_seconds"]), "Overlapped background time"],
        ["Metric evaluation", "{:.3f} s".format(meta["metric_seconds"]), "Saved-checkpoint metrics"],
    ]),
    table(["Check", "Result"], [
        ["Scheduler", "COMPLETED on ruche-gpu15; exit 0:0"],
        ["Restart continuity", summary["continuity_verification"]],
        ["Exact schedules", summary["schedule_verification"]],
        ["Final step", "200,000; exactly 52,000 additional updates"],
        ["Independent full-factor result", "verified; primary ≥99% target met"],
        ["Final-ten persistence", "all saved checkpoints 191,000–200,000 ≥99%"],
    ]),
    verification["saved_fidelity"], verification["fidelity_numpy"], verification["fidelity_matrix_numpy"],
    verification["saved_tp_violation"], verification["tp_violation_numpy"], verification["truth_tp_violation"],
    provenance["source"], verification["result_sha256"], summary["resumed_factor_sha256"],
)

HTML.write_text(document)
print(str(HTML))
