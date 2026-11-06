"""Generate a concise PDF handoff from the archived Ruche execution record."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
record = json.loads((ROOT / "run_record.json").read_text())
pilot = json.loads((ROOT / "calibration-pilot-2014189/pair.json").read_text())
fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
fig.text(.075, .94, "Quiroga–FRAMES: running on Ruche", fontsize=19,
         weight="bold", color="#164e78")
fig.text(.075, .908, record["checked_at"] + " | Execution handoff", fontsize=10,
         color="#56616d")
y = .86

def paragraph(value, bold=False):
    global y
    for line in textwrap.wrap(value, width=94, break_long_words=False):
        fig.text(.075, y, line, fontsize=10,
                 weight="bold" if bold else "normal", color="#18212b")
        y -= .0175
    y -= .012

paragraph(record["headline"], True)
paragraph("Source: exact clean commit 0586c15 is staged in an isolated Ruche checkout. "
          "The reviewed setup and source-bundle checksums match. Previous source trees "
          "and experiment results are preserved; no GitHub publication was needed.")
paragraph("Completed: Slurm smoke 2014184 (25 s) and calibration-budget pilot 2014189 "
          "(22 s), both exit 0. Both pilot methods completed all 256,000 logical row "
          "accesses: 2,000 Quiroga updates and 8,000 FRAMES updates. Artifact hashes, "
          "JSON/NPZ consistency, source provenance, row budgets and endpoints passed checks.")
paragraph("Pilot physical metrics — one unselected anchor configuration", True)
for method, label in [("printed-adafgd", "Quiroga"), ("frames", "FRAMES")]:
    m = pilot["methods"][method]
    paragraph(f"{label}: fidelity {m['process_fidelity']:.4f}; "
              f"full-design RMSE {m['rmse']:.5f}; normalized TP violation "
              f"{m['normalized_tp_violation']:.5f}. Recovery gate: not passed.")
paragraph("These are execution/stability checks, not a best-versus-best result. "
          "The gate requires fidelity >= 0.99, RMSE <= 1e-6 and normalized TP <= 0.001. "
          "The printed adaptive rule retains a calibrated step constant; the paper "
          "does not specify every constant needed for exact reproduction.")
paragraph(record["calibration_description"], True)
paragraph("Calibration uses eight channel instances, one shared initialization per "
          "channel and three FRAMES sampling streams. A 44-pair-per-channel inventory "
          "covers the independent grids: 352 paired executions, with repeated anchors "
          "deduplicated for scientific analysis. There are 30 Quiroga candidates and "
          "five FRAMES candidates; this is not equal hyperparameter-search effort.")
paragraph("Quiroga tunes a fixed TP penalty and step constant. FRAMES tunes its own "
          "initial TP penalty with exponent 0.25, increasing throughout optimization. "
          "Both share the physical data, initialization, row budget and recovery gate. "
          "Method-native penalized objectives are not compared across methods.")
paragraph(record["next_step"])
fig.text(.075, .066, "Evidence: archived Slurm logs, pair.json/pair.npz, run_record.json and calibration plan.",
         fontsize=8, color="#56616d")
fig.text(.075, .050, "No selected validation or held-out comparison is claimed in this report.",
         fontsize=8, color="#56616d")
assert y > .095, y
target = ROOT / "RUCHE_RUN_SUMMARY.pdf"
fig.savefig(target, format="pdf", metadata={"Title": "Quiroga–FRAMES Ruche execution handoff"})
plt.close(fig)
print(target)
