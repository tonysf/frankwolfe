"""Evidence-checked, presentation-format PDF. Reads saved results only."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import subprocess
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path("/home/tony/frankwolfe")
OLD = Path("/home/tony/qpt_gell_mann_init_20261007_1750")
N4 = Path("/home/tony/qpt_quiroga_n4_d62fcf1_20261002_01")
N8 = Path("/home/tony/qpt_n8_continuation_20260923/n8_continue_200k_20260923_01")
OUT = ROOT / "QPT_COAUTHOR_BRIEFING_20261008.pdf"
EVIDENCE = ROOT / "coauthor_briefing_evidence.json"
INK, MUTED, TEAL, ORANGE = "#163044", "#536979", "#087e8b", "#b35d2e"
BG, LIGHT = "#f4f7f9", "#e8f3f3"
PAPER = "https://arxiv.org/html/2312.01311v1#S3.SS2"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

spec = importlib.util.spec_from_file_location("checked_report", ROOT / "build_tuning_report.py")
checked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checked)
inputs = checked.checked_inputs(SimpleNamespace(confirmation=ROOT / "audit_confirmation.json",
                                               revalidation=ROOT / "audit_revalidation.json"))
screen, selected, chosen, literal, stages, followups, jobs = inputs
frozen = [case for _, _, cases in stages for case in cases]
assert len(frozen) == 16 and all(c["gate"]["pass"] for c in frozen)
old_path = OLD / "audit_complete.json"
assert sha(old_path) == "73ae52860439868eb12fc59be9dc9f4c15d810e9463e0dc17b0811645a32fdd3"
old = read(old_path)
assert old["audit_passed"] and old["audited_trials"] == 8
for case in old["cases"]:
    p = Path(case["json_path"])
    assert sha(p) == case["json_sha256"] and sha(p.with_suffix(".npz")) == case["npz_sha256"]
    with np.load(p.with_suffix(".npz"), allow_pickle=False) as z:
        peer = next(c for c in chosen if c["channel"] == case["channel"])
        with np.load(Path(peer["json_path"]).with_suffix(".npz"), allow_pickle=False) as zz:
            for k in ("initial_factor", "truth_factor", "active_observations", "active_rows"):
                assert np.array_equal(z[k], zz[k]), (case["trial_id"], k)

frames = [c for c in old["cases"] if c["method"] == "frames"]
armijo = [c for c in old["cases"] if c["method"] == "armijo-fgd"]
printed_old = [c for c in old["cases"] if c["method"] == "printed-adafgd"]
assert sum(c["gate"]["pass"] for c in frames) == 1
assert all(c["gate"]["pass"] for c in armijo)
assert len(printed_old) == 4 and not any(c["gate"]["pass"] for c in printed_old)
legacy = read(N8 / "verification.json")
assert legacy["status"] == "verified" and legacy["n_qubits"] == 8 and legacy["step"] == 200000
assert legacy["noise_std"] == .05 and sha(N8 / "final.npz") == legacy["result_sha256"]
assert np.isclose(legacy["fidelity_numpy"], .9913929063754066)
legacy_tp = legacy["tp_violation_numpy"] / np.sqrt(2**8)

# Direct Kraus/Born oracle for the historical n=4 endpoints (not optimizer code).
eye = np.eye(16, dtype=complex)
probes = list(eye)
for phase in (1, 1j):
    probes.extend((eye[k] + phase * eye[l]) / np.sqrt(2) for k in range(16) for l in range(k + 1, 16))
probes = np.array(probes)
effects = [np.outer(eye[0], eye[0].conj()) / 37]
for imaginary in (False, True):
    for k in range(1, 16):
        cross = np.outer(eye[0], eye[k].conj())
        effects.append((eye + (1j * (cross - cross.conj().T) if imaginary else cross + cross.conj().T)) / 37)
effects.append(eye - sum(effects))
effects = np.array(effects)

# Recheck older n=4 native archives, preserving their distinct protocol.
n4_paths = [N4 / "long_armijo/armijo_beta10000_extended20000.json"]
n4_paths += [N4 / f"heldout_armijo/seed{s}_armijo_beta10000_extended20000.json" for s in (1, 2, 3)]
n4_paths += [N4 / f"heldout_compare/seed{s}_{label}.json" for s in (1, 2, 3)
             for label in ("ada_beta100_eta3e-3_extended2000", "frames_beta1000_extended2000epochs")]
n4_cases = []
for p in n4_paths:
    m = read(p)
    assert m["design"]["n_qubits"] == 4 and m["active_row_count"] == 8192
    assert m["configuration"]["observation_mode"] == "noiseless"
    for name, expected in m["source_sha256"].items():
        raw = subprocess.check_output(["git", "-C", str(REPO), "show",
                                       "d62fcf192f27d8fb207ce61a1c466fc5059a7e1f:paper/experiments/" + name])
        assert hashlib.sha256(raw).hexdigest() == expected
    method = m["configuration"]["methods"][0]
    ep = m["methods"][method]
    prefix = method.replace("-", "_")
    with np.load(p.with_suffix(".npz"), allow_pickle=False) as z:
        assert json.loads(str(z["metadata_json"].item())) == m
        factor, truth = z[prefix + "_final_factor"], z["truth_factor"]
        K = factor[:, 0].reshape(16, 16).T
        tp = float(np.linalg.norm(K.conj().T @ K - np.eye(16)) / 4)
        fid = float(abs(np.vdot(truth, factor)) ** 2 / (np.vdot(truth, truth).real * np.vdot(factor, factor).real))
        assert np.isclose(tp, ep["normalized_tp_violation"], rtol=1e-8, atol=1e-13)
        assert np.isclose(fid, ep["process_fidelity"], rtol=1e-12, atol=1e-13)
        output_states = probes @ K.T
        born = np.einsum("pi,jik,pk->pj", output_states.conj(), effects, output_states).real.reshape(-1)
        assert np.array_equal(z["active_rows"], np.arange(8192))
        rmse = float(np.sqrt(np.mean((born - z["active_observations"])**2)))
        assert np.isclose(rmse, ep["rmse"], rtol=1e-10, atol=1e-13)
        assert float(z[prefix + "_rmse"][-1]) == ep["rmse"]
        assert int(z[prefix + "_checkpoint_steps"][-1]) == ep["completed_steps"]
        assert int(z[prefix + "_cumulative_data_row_accesses"][-1]) == ep["cumulative_data_row_accesses"]
    passed = ep["status"] in ("complete", "converged") and ep["rmse"] <= 1e-6 and tp <= .001 and fid >= .99
    n4_cases.append(dict(path=str(p), json_sha256=sha(p), npz_sha256=sha(p.with_suffix(".npz")),
                         method=method, seed=m["configuration"]["channel_seed"], gate_pass=passed,
                         endpoint=ep, checked_rmse=rmse, checked_tp=tp, checked_fidelity=fid))
n4_armijo = [c for c in n4_cases if c["method"] == "armijo-fgd"]
assert len(n4_armijo) == 4 and all(c["gate_pass"] for c in n4_armijo)
assert not any(c["gate_pass"] for c in n4_cases if c["method"] != "armijo-fgd")

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 14})

class Slide:
    def __init__(self, number, title, subtitle, source=""):
        self.fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
        self.texts = []
        self.text(.055, .951, "LOW-RANK QUANTUM PROCESS TOMOGRAPHY", 9.5, TEAL, weight="bold")
        self.text(.055, .891, title, 23, weight="bold")
        self.text(.055, .814, subtitle, 11.7, MUTED)
        self.fig.add_artist(Rectangle((.055, .787), .89, .003, transform=self.fig.transFigure, facecolor=TEAL, edgecolor="none"))
        self.text(.055, .047, source, 8.2, MUTED)
        self.text(.055, .022, "Coauthor briefing · evidence through 8 October 2026 · no new optimization for this report", 8, MUTED)
        self.text(.945, .022, str(number), 8, MUTED, ha="right")
    def text(self, x, y, s, size=13, color=INK, **kw):
        t = self.fig.text(x, y, s, va="top", fontsize=size, color=color, **kw)
        self.texts.append(t)
        return t
    def para(self, x, y, s, width=97, size=13, color=INK):
        lines = textwrap.wrap(s, width, break_long_words=False, break_on_hyphens=False)
        self.text(x, y, "\n".join(lines), size, color, linespacing=1.35)
        return y - len(lines) * size * 1.35 / 540 - .023
    def heading(self, x, y, s):
        self.text(x, y, s, 15, weight="bold")
        return y - .05
    def table(self, box, headers, rows, widths, size=12):
        ax = self.fig.add_axes(box)
        ax.axis("off")
        table = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc="left", bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(size)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_linewidth(1.5)
            cell.set_facecolor(INK if r == 0 else (BG if r % 2 else LIGHT))
            cell.set_text_props(color="white" if r == 0 else INK, weight="bold" if r == 0 else "normal")
            cell.PAD = .05
        return table
    def check(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for t in self.texts:
            b = t.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            assert b.x0 >= .012 and b.x1 <= .989 and b.y0 >= .005 and b.y1 <= .99, t.get_text()

pages = []
s = Slide(1, "A strong fixed-penalty baseline now works",
          "The successful adaptive line search is our modification—not a reproduction of published Quiroga adaFGD.",
          "Evidence: latest 64-run audited campaign [E1]; prior paired FRAMES diagnostic [E2].")
pages.append(s)
s.para(.055, .746, "The corrected sensing problem is recoverable. The remaining scientific tasks are to resolve the published step rule and complete a fair, independently tuned FRAMES comparison.", width=111, size=16)
cards = [("16 / 16", "Modified FGD passes", "One frozen setting; 8 calibration\nand 8 historical revalidation channels.", TEAL),
         ("0 / 44", "Printed-rule screen passes", "14 finite completions; 30 numerical\nfailures. These are 10k-budget trials.", ORANGE),
         ("1 / 2", "Prior FRAMES passes", "Both satisfy fidelity and TP; one\nmisses the RMSE gate by 11.1%.", INK)]
for i, (value, label, desc, color) in enumerate(cards):
    x = .055 + i * .305
    s.fig.add_artist(Rectangle((x, .332), .28, .245, transform=s.fig.transFigure, facecolor=BG, edgecolor="none"))
    s.text(x + .018, .554, value, 32, color, weight="bold")
    s.text(x + .018, .458, label, 14, weight="bold")
    s.text(x + .018, .404, desc, 11.3, MUTED, linespacing=1.4)
s.para(.055, .282, "Headline scope: noiseless, rank-one, two-qubit recovery—not noisy or higher-rank tomography, not unseen-test validation, and not a demonstrated FRAMES advantage.", width=114, size=14)
s.para(.055, .164, "Positive higher-qubit context: earlier exploratory Armijo FGD also recovers four tested four-qubit channels. Its protocol differs from the new frozen-setting experiment.", width=121, size=12.5)

s = Slide(2, "What is controlled—and what is actually being compared",
          "Same physical data and initial factor within each two-qubit channel; penalty policies remain method-specific.",
          "Setup and conventions: qpt_quiroga_sensing.md; native archives [E1–E2]; published objective and step: [P1].")
pages.append(s)
s.para(.055, .747, "Two qubits (d=4), rank-one Haar-unitary targets, 16 global input probes × 8 POVM outcomes = 128 Born probabilities. Full noiseless data. Normalized complex-uniform Gell-Mann coefficients initialize the physical factor; POVM constants a=b=0.1 are declared choices.", width=133, size=12.3)
s.text(.055, .611, r"Fixed-penalty FGD:  $J(U)=\frac{1}{2}\|\mathcal{A}(UU^\dagger)-y\|_2^2+\lambda\|\mathrm{Tr}_{out}(UU^\dagger)-I\|_F^2$", 17)
s.table([.055, .276, .89, .275], ["Method label", "Step-selection mechanism", "TP policy"], [
    ["Printed Quiroga adaFGD", "Residual / prediction norm ratio × fixed multiplier", "Fixed λ; tuned independently"],
    ["Armijo FGD control", "Halve trial step until sufficient decrease", "Fixed λ; separate control"],
    ["Exact-ray FGD variant", "Minimize quartic on positive factor ray; no backtracking", "Fixed λ; separate modification"],
    ["FRAMES", "Stochastic FRAMES update and declared schedules", "Fixed initial weight; increasing penalty"],
], [.245, .485, .27], 11.3)
s.para(.055, .239, "Joint recovery gate: fidelity ≥ 0.99 AND full-design RMSE ≤ 1e−6 AND normalized TP violation ≤ 1e−3, with a nonfailed terminal status. Normalized TP = ||Tr_out(UU†) − I||F / √d. The paper does not specify this numerical joint gate.", width=132, size=12)
s.text(.055, .101, "The paper gives an adaptive ratio, not this exact line search; unpublished constants prevent an exact implementation-level reproduction.", 10.6, MUTED)
s.text(.055, .072, "[P1] Quiroga & Kyrillidis, arXiv:2312.01311, §III-B and §V", 9.5, TEAL, url=PAPER)

s = Slide(3, "Latest result: one frozen setting passes all 16 channels",
          "Exact factor-ray FGD, fixed α=5×10⁻⁵ and λ=128α=0.0064; selected before confirmation and revalidation.",
          "[E1] audit_screen.json; audit_confirmation.json; audit_revalidation.json; selection_screen.json.")
pages.append(s)
rows = []
for label, span, cases in stages:
    p = [c["endpoint_physical"] for c in cases]
    rows.append([label + " / " + span, f"{len(cases)}/{len(cases)}",
                 f"{min(v['rmse'] for v in p):.2e} – {max(v['rmse'] for v in p):.2e}",
                 f"{max(v['normalized_tp_violation'] for v in p):.2e}",
                 f"{min(c['completed_steps'] for c in cases):,}–{max(c['completed_steps'] for c in cases):,}"])
s.table([.055, .490, .89, .247], ["Stage / channel seeds", "Gate", "RMSE range", "Max. TP", "Updates"], rows,
        [.34, .07, .26, .13, .20], 12)
s.para(.055, .444, "All 16 runs terminate by the full-gradient stopping test. Worst RMSE: 1.61×10⁻¹⁰; worst normalized TP: 7.80×10⁻¹⁰. Fidelity is approximately one (floating-point roundoff), in 794–1,882 accepted updates.", width=122, size=14)
s.para(.055, .297, "Selection used the worst-channel normalized gate deficit—not runtime. All three line-search penalties passed the two-channel screen; α=5×10⁻⁴ actually used fewer charged rows than the selected α=5×10⁻⁵. No claim of globally optimal hyperparameters.", width=127, size=12.5)
s.para(.055, .160, "Channels 100–107 were used in earlier campaigns: this is historical revalidation, not unseen validation. One initialization per channel. Final held-out channels 1000–1019 remain untouched.", width=131, size=12.2, color=ORANGE)

s = Slide(4, "Step selection changes the observed convergence behavior",
          "Paired channels 0/1: identical data, initialization, rank, objective and fixed λ=0.0064.",
          "[E1] Saved trajectories; solid = channel 0, dashed = channel 1. No optimizer rerun for these figures.")
pages.append(s)
for left, metric, ylabel, gate, ylim in [(.085, "rmse", "Full-design RMSE", 1e-6, (3e-11, .2)),
                                       (.555, "normalized_tp_violation", "Normalized TP violation", 1e-3, (1e-10, 2))]:
    ax = s.fig.add_axes([left, .374, .375, .34])
    for cases, prefix, label, color in ((literal, "printed_adafgd", "Printed ratio (c=2)", ORANGE),
                                      (chosen, "exact_linesearch_fgd", "Exact-ray variant", TEAL)):
        for c in sorted(cases, key=lambda c: c["channel"]):
            with np.load(Path(c["json_path"]).with_suffix(".npz"), allow_pickle=False) as z:
                x = z[prefix + "_cumulative_data_row_accesses"] / 1e6
                v = z[prefix + "_" + metric]
            ax.semilogy(x, v, color=color, ls="-" if c["channel"] == 0 else "--", lw=2,
                        label=label if c["channel"] == 0 else None)
    ax.axhline(gate, color=MUTED, ls=":", label="Gate")
    ax.set(xlabel="Charged row accesses (millions)", ylabel=ylabel, xlim=(0, 1.32), ylim=ylim)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, loc="upper right")
s.para(.055, .286, "Printed-rule screen: 0/44 pass at 10,000 updates; best jointly stable new setting has RMSE 0.00578–0.00895. Larger tested multipliers often diverge. The shrinking-step tail supports a step-selection bottleneck on these cases; it is not a proof of nonconvergence.", width=142, size=11.6)
s.para(.055, .169, "Important context: four earlier printed-rule runs at 100,000 updates reached fidelity 0.9953–0.9987, but still failed both RMSE and TP gates. Row charges include line-search candidates but exclude spectral, TP and checkpoint costs; these curves are not a total-runtime comparison.", width=142, size=11.6)

s = Slide(5, "FRAMES is close to the gate, but the comparison is unfinished",
          "Paired two-qubit channels 0/1 have byte-identical data and initial factors across the archives below.",
          "[E2] Eight-run Gell-Mann diagnostic (7 October); [E1] selected exact-ray runs (8 October). Unequal budgets and step costs.")
pages.append(s)
table_rows = []
sets = [("Printed ratio¹", [c for c in printed_old if c["trial"]["alpha"] == 5e-4]),
        ("FRAMES²", frames), ("Armijo FGD¹", armijo), ("Exact-ray FGD³", chosen)]
for name, cases in sets:
    for c in sorted(cases, key=lambda c: c["channel"]):
        p = c["endpoint_physical"]
        table_rows.append([name, str(c["channel"]), f"{c['completed_steps']:,}",
                           f"{p['rmse']:.3e}", f"{p['normalized_tp_violation']:.3e}",
                           "PASS" if c["gate"]["pass"] else "FAIL"])
s.table([.055, .312, .89, .42], ["Method", "Channel", "Updates", "RMSE", "Normalized TP", "Joint gate"],
        table_rows, [.23, .09, .14, .20, .21, .13], 11.7)
s.para(.055, .278, "¹ Fixed α=5×10⁻⁴ (λ=0.064). ² FRAMES α₀=5×10⁻⁵, α(t)=α₀(t+1)⁰·²⁵, batch 32, radius 2. ³ Fixed α=5×10⁻⁵ (λ=0.0064). FGD runs are unconstrained in factor space; FRAMES uses its norm-ball domain.", width=143, size=10.9)
s.para(.055, .173, "FRAMES satisfies TP and fidelity on both channels; channel 0 misses RMSE by 11.1%. Printed ratio and FRAMES each charge 12.8 million nominal rows here, excluding spectral/TP costs. Both line-search controls reach much lower residuals with fewer charged rows; a final best-for-best wall-time ranking is not established.", width=143, size=11.1)

s = Slide(6, "Higher qubits: preserve the evidence, qualify the claims",
          "Earlier four-qubit recovery is real; the new frozen exact-ray configuration has only been tested at two qubits.",
          "[E3] Independently checked n=4 historical endpoints; [E4] matrix-free code/tests; [E5] verified legacy n=8 snapshot.")
pages.append(s)
s.heading(.055, .743, "Earlier exploratory n=4 results · 8,192 rows · different initialization / tuning")
s.table([.055, .474, .89, .20], ["Method / channel seeds", "Gate", "RMSE range", "Budget / TP policy"], [
    ["Armijo FGD / 0–3", "4/4", "1.52e−7 – 8.66e−7", "20k updates; fixed λ=0.4096"],
    ["Printed ratio / 1–3", "0/3", "4.56e−3 – 4.77e−3", "2k updates; fixed λ=40.96"],
    ["Early FRAMES / 1–3", "0/3", "2.94e−3 – 3.26e−3", "512k updates; CONSTANT penalty"],
], [.29, .075, .255, .38], 11.3)
s.para(.055, .439, "This establishes recoverability at n=4, not a fair speed ranking: Armijo used ~459–460 million charged rows versus 16.384 million for each other run. Early constant-penalty FRAMES is not the intended increasing-penalty comparator. These old channels are no longer held out.", width=140, size=11.6)
s.heading(.055, .310, "Legacy n=8 remains useful—but is a separate benchmark")
s.para(.055, .256, f"A verified local-Pauli continuation reached {100*legacy['fidelity_numpy']:.2f}% factor fidelity after 200k updates, with Gaussian noise σ=0.05 and normalized TP {legacy_tp:.4f}. This is not a current strict-gate certificate or a result on the corrected global-probe design.", width=140, size=11.6)
s.para(.055, .137, "Also correct the old memory claim: 64 GiB at n=8 is the OPTIONAL dense adaptive-numerator matrix. Current code has a matrix-free spectral eigensolve and exact factorized TP. Full passes remain costly; large-n superiority is still unestablished.", width=140, size=11.2, color=ORANGE)

s = Slide(7, "What we can present—and what we should do next",
          "Suggested conclusion: a working fixed-penalty FGD baseline, with exact-paper reproduction and final comparison still open.",
          "No new experiments were submitted for this briefing. All earlier results and failures are retained.")
pages.append(s)
y = s.heading(.055, .743, "Supported conclusions")
y = s.para(.055, y, "1. The corrected rank-one inverse problem supports accurate recovery: modified exact-ray FGD at n=2, and earlier Armijo FGD at n=4.\n2. Step selection materially affects the observed outcome; failure of our printed-rule reconstruction does not refute Quiroga’s paper.", width=133, size=12.4)
y = s.heading(.055, y, "Next experimental decisions")
y = s.para(.055, y, "Freeze explicit method labels. Retain the printed ratio as a reproduction track and line-search FGD as a separate strong control. Tune FRAMES’s initial penalty and continuation independently of FGD’s fixed penalty; compare time-to-the-same-gate, memory and complete operator work. Confirm at n=3/4 before scaling further, then evaluate a genuinely untouched test set.", width=135, size=12.1)
y = s.heading(.055, y, "Checks and evidence")
y = s.para(.055, y, "Latest batch: 64 archive pairs, 16 complete task inventories, three pinned plans; independent endpoint, source, initialization and work-accounting checks. Recorded suite: 597 passed, 55 skipped; 16 additional harness tests. Latest source 4ce1108 is merged locally, not pushed. The new frozen setting has no noisy-data, higher-rank or higher-qubit validation yet.", width=138, size=11.2)
s.text(.055, .181, "[E1] Latest audits: screen / confirmation / historical revalidation; Ruche 2039775 / 2039794 / 2039814.", 9.5, MUTED)
s.text(.055, .150, "[E2] Gell-Mann diagnostic audit, source 966ef74; Ruche 2038514. [E3] n=4 source d62fcf1, 2–3 October.", 9.5, MUTED)
s.text(.055, .119, "[E4] qpt_quiroga_sensing.md and norm tests. [E5] n8_continue_200k_20260923_01/verification.json. Hashes attached.", 9.5, MUTED)
s.text(.055, .085, "[P1] Quiroga & Kyrillidis (2023), §III-B and §V", 10, TEAL, url=PAPER)

assert not OUT.exists() and not EVIDENCE.exists(), "Never overwrite an existing report"
for p in pages:
    p.check()
with OUT.open("xb") as stream:
    with PdfPages(stream, metadata={"Title": "Low-rank QPT: coauthor results briefing, 8 October 2026",
                                  "Author": "QPT experimental audit",
                                  "Subject": "Published-rule reconstruction versus explicit line-search variants; FRAMES and scaling boundaries"}) as pdf:
        for p in pages:
            pdf.savefig(p.fig)
evidence = dict(schema_version=1, kind="qpt_coauthor_briefing_evidence", pdf=str(OUT), pdf_sha256=sha(OUT),
                builder_sha256=sha(__file__), source_commit=checked.COMMIT,
                inputs={str(p): sha(p) for p in [ROOT / "audit_screen.json", ROOT / "audit_confirmation.json",
                    ROOT / "audit_revalidation.json", ROOT / "selection_screen.json", old_path,
                    REPO / "docs/qpt_quiroga_sensing.md", REPO / "tests/test_qpt_quiroga_frames.py",
                    N8 / "verification.json", N8 / "final.npz"]},
                n4_native_archive_checks=n4_cases,
                qualification="n=4 is separate exploratory historical evidence, not the latest frozen-setting validation; no new optimization.")
with EVIDENCE.open("x") as f:
    json.dump(evidence, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
for p in pages:
    plt.close(p.fig)
print(json.dumps({"pdf": str(OUT), "pages": len(pages), "sha256": sha(OUT), "evidence": str(EVIDENCE)}))
