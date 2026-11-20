"""One-page interpretation of the existing audited diagnostic; no new runs."""
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
audit_path = root / "audit_complete.json"
raw = audit_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "73ae52860439868eb12fc59be9dc9f4c15d810e9463e0dc17b0811645a32fdd3"
audit = json.loads(raw)
assert audit["audit_passed"] and audit["audited_trials"] == 8
cases = [c for c in audit["cases"] if c["method"] == "printed-adafgd"]
assert len(cases) == 4 and all(c["status"] == "complete" and not c["gate"]["pass"] for c in cases)
output = root / "QPT_QUIROGA_CONCLUSION_20261008.pdf"
assert not output.exists(), "Do not overwrite a report"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
fig.text(.075, .956, "RESEARCH CONCLUSION  /  8 OCTOBER 2026", color="#155b85", fontsize=10, weight="bold")
fig.text(.075, .916, "What can we conclude about Quiroga?", fontsize=18, weight="bold", color="#172632")
y = .867

def paragraph(text, *, bold=False, size=10.1, color="#172632"):
    global y
    lines = textwrap.wrap(text, width=94, break_long_words=False, break_on_hyphens=False)
    fig.text(.075, y, "\n".join(lines), va="top", fontsize=size, linespacing=1.4,
             weight="bold" if bold else "normal", color=color)
    y -= len(lines) * size * 1.4 / (11.69 * 72) + .018

paragraph("Our reconstruction of the printed adaptive update achieves high fidelity, but has not met our joint accuracy/TP gate within the tested budget. This is not a general failure verdict on Quiroga's method.", bold=True, color="#155b85")
paragraph("Evidence: two noiseless, rank-one, two-qubit calibration channels; two fixed TP penalties per channel; 100,000 printed updates each. All four runs completed normally, but 0/4 passed. These are small-scale diagnostics, not an n=8 result.")

rows = []
for key, label, threshold, fmt in (
    ("process_fidelity", "Process fidelity", "≥ 99%", lambda x: f"{100*x:.3f}%"),
    ("rmse", "Data-fit RMSE", "≤ 1e-6", lambda x: f"{x:.3e}"),
    ("normalized_tp_violation", "Normalized TP error", "≤ 1e-3", lambda x: f"{x:.3e}"),
):
    values = [c["endpoint_physical"][key] for c in cases]
    rows.append([label, fmt(min(values)) + " – " + fmt(max(values)), threshold,
                 "4/4 pass" if key == "process_fidelity" else "0/4 pass"])
table_h = .112
ax = fig.add_axes([.075, y-table_h, .85, table_h])
ax.axis("off")
table = ax.table(cellText=rows, colLabels=["Metric", "Observed range", "Our gate", "Outcome"],
                 colWidths=[.25, .38, .18, .19], cellLoc="center", bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9.2)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_facecolor("#155b85" if r == 0 else "#eef3f6")
    cell.set_text_props(color="white" if r == 0 else "#172632", weight="bold" if r == 0 else "normal")
y -= table_h + .025

paragraph("What this means. It is not just a TP problem: RMSE is also about 1,000–1,750 times above our threshold. High fidelity alone therefore gives a more favorable impression than the stricter joint gate.")
paragraph("What we have narrowed down. Switching to a normalized Gell-Mann initializer changed Quiroga's endpoints but did not rescue any run. Armijo factored GD passes on both channels at the same fixed-penalty setting used by two of the printed-update runs. Thus the data and fixed-penalty formulation can support recovery here; optimization/step-size behavior is the leading explanation, not a uniquely established cause. Armijo remains a separate control, not Quiroga's published algorithm.")
paragraph("What we cannot claim. The paper gives the adaptive step only up to proportionality, and not all experimental normalization/initialization choices are pinned down [1]. We have not established an exact reproduction, impossibility of convergence, or failure at higher qubit counts. Nor is a final FRAMES victory established: FRAMES passes only 1/2 cases, narrowly missing the other.")
paragraph("Practical conclusion. Retain these as audited finite-budget results of the declared reconstruction. Before using Quiroga as a definitive best-for-best baseline, resolve or explicitly calibrate the step multiplier and normalization conventions, tune its fixed penalty independently of FRAMES, and validate frozen choices on unseen channels. No new experiment was launched for this note.", color="#155b85")
assert y > .10, f"Body would overlap source footer: {y}"
fig.text(.075, .081, "[1] Quiroga & Kyrillidis, arXiv:2312.01311v1, Sections III-B and V.", fontsize=8.2, color="#155b85",
         url="https://arxiv.org/html/2312.01311v1")
fig.text(.075, .063, "Evidence: audit_complete.json; source 966ef74; Ruche array 2038514; all eight archives audited.", fontsize=7.8, color="#536571")
fig.text(.075, .044, "Detailed paired results: QPT_GELL_MANN_DIAGNOSTIC_20261007.pdf", fontsize=7.8, color="#536571")
fig.savefig(output, metadata={"Title": "Conclusion on the reconstructed Quiroga baseline", "Author": "QPT experiment audit"})
plt.close(fig)
print(output)
