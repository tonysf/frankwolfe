"""Render the final execution handoff without implying a completed comparison."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
r = json.loads((root / "run_record.json").read_text())
r.update(json.loads((root / "completion_update.json").read_text()))
fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
fig.text(.075, .94, "Quiroga–FRAMES: Ruche run handoff", fontsize=19,
         weight="bold", color="#164e78")
fig.text(.075, .905, r["checked_at"], fontsize=10, color="#56616d")
y = .86

def paragraph(value, bold=False):
    global y
    for line in textwrap.wrap(value, width=80 if bold else 91,
                              break_long_words=False):
        fig.text(.075, y, line, fontsize=10,
                 weight="bold" if bold else "normal", color="#18212b")
        y -= .0165
    y -= .01

paragraph(r["headline"], True)
paragraph("Completed checks: Ruche smoke 2014184 and the 2,000-row-epoch pilot "
          "2014189 finished successfully. Both methods completed the pilot's "
          "256,000 nominal row accesses. JSON/NPZ consistency, checksums, source "
          "provenance and physical metrics recomputed from the saved factors passed.")
paragraph("The untuned pilot did not pass the recovery gate. Quiroga: fidelity "
          "0.1346, RMSE 0.06330, normalized TP 0.43866. FRAMES: fidelity 0.7927, "
          "RMSE 0.01420, normalized TP 0.00683. These anchors do not establish a "
          "best-versus-best result.")
paragraph("What interrupted the first grid", True)
paragraph("Array 2014199 exposed a recording bug when an aggressive Quiroga "
          "configuration overflowed: final-metric evaluation raised before the "
          "failed pair could be archived. Six channel tasks stopped; the remaining "
          "two were cancelled. All outputs and error evidence are preserved.")
paragraph(r["fix_description"])
paragraph(r["tests_description"])
paragraph(r["restart_description"], True)
paragraph("The grid is unchanged: 30 Quiroga configurations tune a fixed TP penalty "
          "and adaptive-step scale; five FRAMES configurations tune their own "
          "initial TP penalty, increasing with exponent 0.25. Shared sensing, "
          "initialization, row budget and physical gate remain fixed.")
paragraph("Eight channels, one initialization per channel and three FRAMES sampling "
          "streams give 352 paired executions. Repeated method-specific results "
          "must be deduplicated: 240 unique Quiroga outcomes and 120 FRAMES outcomes. "
          "The unequal grid sizes do not imply equal hyperparameter-search effort.")
paragraph("Ruche's cpu_short QoS permits one running job per user. Pending channel "
          "tasks therefore wait their turn; they are not evidence of an indefinite "
          "resource request.")
paragraph(r["next_step"])
assert y > .10, y
fig.text(.075, .065, "Evidence and exact hashes: run_record.json, launch plans, archived Slurm logs and pair artifacts.",
         fontsize=8, color="#56616d")
fig.text(.075, .05, "No selected validation or held-out result is claimed. Earlier experiments remain preserved.",
         fontsize=8, color="#56616d")
target = root / "RUCHE_RUN_SUMMARY.pdf"
fig.savefig(target, format="pdf", metadata={"Title": "Quiroga–FRAMES Ruche run handoff"})
plt.close(fig)
print(target)
