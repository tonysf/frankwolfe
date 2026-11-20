"""Make a separate legacy-results PDF from preserved evidence; no optimization."""
from pathlib import Path
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/tony/frankwolfe')
AUDIT = Path('/home/tony/qpt_higher_qubit_audit_20261004_01')
N8 = Path('/home/tony/qpt_n8_continuation_20260923/n8_continue_200k_20260923_01')
MOM = AUDIT / 'archives/ruche/n8_momentum_sgd_20260924_01'
N10 = REPO / 'results/qpt/coauthor_summary_20260915/n10'
OUT = ROOT / 'QPT_LEGACY_RESULTS_ADDENDUM_20261008.pdf'
MANIFEST = ROOT / 'legacy_addendum_evidence.json'
assert not OUT.exists() and not MANIFEST.exists(), 'Preserve existing reports.'
inputs = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    inputs[str(p)] = sha(p)
    return json.loads(p.read_text())

def verified(p, archive=None):
    v = read(p)
    assert v['status'] == 'verified'
    assert abs(v['fidelity_numpy'] - v['saved_fidelity']) < 1e-11
    assert abs(v['tp_violation_numpy'] - v['saved_tp_violation']) < 1e-9
    if archive is not None:
        inputs[str(archive)] = sha(archive)
        assert inputs[str(archive)] == v['result_sha256']
    return v

n8 = verified(N8 / 'verification.json', N8 / 'final.npz')
momentum = read(MOM / 'summary.json')
assert momentum['status'] == 'complete' and momentum['paired_minibatches_verified']
replicates = []
for s in (1, 2, 3):
    pair = []
    for method in ('frames', 'momentum_sgd'):
        folder = MOM / f'results/{method}_seed{s}'
        v = verified(folder / 'verification.json', folder / 'final.npz')
        row = next(r for r in momentum['runs'] if r['method'] == method and r['sampling_seed'] == s)
        assert row['result_sha256'] == v['result_sha256']
        assert abs(row['final_fidelity'] - v['fidelity_numpy']) < 1e-11
        assert v['data_sha256'] == n8['data_sha256'] and v['initialization_seed'] == 0
        assert v['step'] == 200000 and v['noise_std'] == .05
        pair.append(row['batch_symbols_sha256'])
        if method == 'frames':
            replicates.append(v)
    assert pair[0] == pair[1]
n10 = verified(N10 / 'reduced_noise_verification.json')
n10_outcomes = read(N10 / 'outcomes.json')
fixed10 = next(c for c in n10_outcomes['completed_runs'] if c['condition'] == 'fixed')
audit = read(AUDIT / 'artifact_manifest.json')
inputs[str(REPO / 'docs/qpt_quiroga_sensing.md')] = sha(REPO / 'docs/qpt_quiroga_sensing.md')
assert n8['step'] == 200000 and n10['step'] == 57000
assert n10['noise_std'] == .05 / 32

INK, MUTED, TEAL, LIGHT = '#163044', '#526878', '#087e8b', '#edf5f6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})

class Slide:
    def __init__(self, number, title, subtitle):
        self.fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
        self.text(.055, .95, 'LOW-RANK QPT · LEGACY LOCAL-PAULI RESULTS', 10, TEAL, weight='bold')
        self.text(.055, .885, title, 24, weight='bold')
        self.text(.055, .813, subtitle, 12, MUTED)
        self.fig.add_artist(Rectangle((.055, .785), .89, .003, transform=self.fig.transFigure, color=TEAL))
        self.text(.055, .027, 'Coauthor briefing addendum · 8 October 2026 · preserved results; no new optimization', 8.5, MUTED)
        self.text(.945, .027, str(number), 9, MUTED, ha='right')
    def text(self, x, y, text, size=14, color=INK, **kw):
        return self.fig.text(x, y, text, va='top', fontsize=size, color=color, **kw)
    def para(self, y, text, width=111, size=13, color=INK):
        lines = textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False)
        self.text(.055, y, '\n'.join(lines), size, color, linespacing=1.4)
        return y - len(lines) * size * 1.4 / 540 - .022
    def table(self, box, headers, rows, widths, size=12):
        ax = self.fig.add_axes(box)
        ax.axis('off')
        t = ax.table(cellText=rows, colLabels=headers, colWidths=widths, cellLoc='left', bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(size)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
            cell.set_facecolor(INK if r == 0 else (LIGHT if r % 2 else '#f6f8fa'))
            cell.set_text_props(color='white' if r == 0 else INK, weight='bold' if r == 0 else 'normal')
            cell.PAD = .06
    def save(self, pdf):
        pdf.savefig(self.fig)
        plt.close(self.fig)

with PdfPages(OUT, metadata={'Title': 'QPT: preserved local-Pauli results', 'Author': 'QPT project',
                             'Subject': 'Verified legacy benchmark results and comparison boundaries'}) as pdf:
    s = Slide(1, 'The earlier large-scale results still count',
              'They support a separate, physically meaningful sensing benchmark—not a Quiroga reproduction.')
    rows = [['8 / continuation', '0.05', '200,000', f"{100*n8['fidelity_numpy']:.4f}%", f"{n8['tp_violation_numpy']/16:.5f}"]]
    rows += [[f"8 / sampling seed {v['sampling_seed']}", '0.05', '200,000',
              f"{100*v['fidelity_numpy']:.4f}%", f"{v['tp_violation_numpy']/16:.5f}"] for v in replicates]
    rows += [['10 / reduced noise', '0.0015625', '57,000', f"{100*n10['fidelity_numpy']:.4f}%", f"{n10['tp_violation_numpy']/32:.5f}"],
             ['10 / original noise', '0.05', '100,000', f"{100*fixed10['fidelity']:.6f}%", f"{fixed10['tp']/32:.5f}"]]
    s.table([.055, .375, .89, .36], ['Qubits / run', 'Gaussian noise σ', 'Updates', 'Factor fidelity', 'Normalized TP'],
            rows, [.29, .18, .15, .19, .19], 12)
    y = s.para(.341, 'Eight qubits: about 99% factor fidelity is repeatable across the tested sampling streams. These are the SAME channel, observations and initialization—not four independent channel recoveries.')
    y = s.para(y, 'Ten qubits: the high-fidelity endpoint used 32× lower noise. Recovery at σ = 0.05 was not established. The later cancelled continuation reached only 0.018935% at its last inspected checkpoint (not a terminal result).')
    s.para(y, 'TP is the residual norm divided by √(2ⁿ). Every endpoint here is above the newer 0.001 TP threshold; high normalized factor overlap does not certify an exactly trace-preserving channel.', size=12, color=MUTED)
    s.save(pdf)

    s = Slide(2, 'What changed was mainly the sensing experiment',
              'Both benchmark families use rank-one Haar-unitary targets; neither is a higher-rank channel study.')
    s.table([.055, .385, .89, .35], ['Feature', 'Earlier local-Pauli benchmark', 'Current Quiroga-style benchmark'], [
        ['Input states', 'Products of |0⟩, |1⟩, |+⟩, |+i⟩', 'Basis states + global pairwise superpositions'],
        ['Readout', 'Local X/Y/Z settings and outcomes', 'One generalized 2d-outcome POVM'],
        ['Full virtual row count', '24ⁿ (≈110 billion at n = 8)', '2 × 8ⁿ (≈33.6 million at n = 8)'],
        ['Highlighted data', 'Additive Gaussian observation noise', 'Latest calibration is noiseless'],
        ['Interpretation', 'High-fidelity noisy reconstruction', 'Strict fidelity / RMSE / TP gate'],
    ], [.22, .36, .42], 11.7)
    y = s.para(.35, 'The old inputs and projective measurements are physical, not arbitrary random sensing matrices. The Gaussian corruption is a synthetic observation model, not finite-shot counts. It does not turn the target into a noisy or mixed quantum channel.')
    y = s.para(y, 'There is also a convention difference: for the same generated unitary V, the legacy probabilities correspond to conj(V), while the new backend uses V. Both are valid unitary channels with the same Haar distribution, but same-seed runs are not identical physical instances.', size=12.5)
    s.para(y, 'Consequently, the old fidelity curves and GPU engineering remain useful. Their numerical tuning, convergence and comparison claims cannot simply be transferred to the new probe/POVM geometry.', size=12.5, color=TEAL)
    s.save(pdf)

    s = Slide(3, 'How to present the older work to coauthors',
              'Keep two result tracks: large-scale local-Pauli recovery and matched Quiroga-style comparisons.')
    s.fig.add_artist(Rectangle((.055, .546), .89, .194, transform=s.fig.transFigure, facecolor=LIGHT, edgecolor='none'))
    s.text(.075, .72, 'Suggested headline', 14, TEAL, weight='bold')
    s.text(.075, .674, '“FRAMES achieved approximately 99% factor fidelity on noisy eight-qubit\nlocal-Pauli QPT, and at ten qubits with reduced noise. Trace preservation\nremained approximate; these are separate from our Quiroga-style tests.”', 15, linespacing=1.35)
    y = s.para(.513, 'Baseline evidence: the paired eight-qubit projected momentum-SGD runs remained near 8.73 × 10⁻⁶ fidelity, versus 0.9892–0.9922 for FRAMES, at 200,000 updates. Minibatches were matched; this is evidence about the tested baseline settings, not a general superiority theorem or a published Quiroga reproduction.', size=12.5)
    y = s.para(y, 'Engineering evidence survives: on-demand fixed observations, structured operators, bounded batch workspace and restartable GPU runs enabled these large instances without storing the full observation table. Virtual row counts are not counts of experimentally acquired measurements.', size=12.5)
    y = s.para(y, 'These noisy runs were not designed around today’s noiseless joint gate. Present them as high-fidelity reconstructions with approximate TP, not exact-recovery passes; retain their original objectives.', size=12.5)
    s.text(.055, .172, 'Evidence / verification scope', 11, TEAL, weight='bold')
    s.text(.055, .14, '8q: archived NumPy full-factor verification + locally rechecked result hashes (continuation and all six paired runs).\n10q: preserved full-factor verification record and completed-run table; native 10q factor not reloaded for this addendum.\nDesign: docs/qpt_quiroga_sensing.md. Preservation: higher-qubit audit of 4 October.\nExact file paths, input checksums and numerical records: legacy_addendum_evidence.json beside this PDF.', 9.5, MUTED, linespacing=1.4)
    s.save(pdf)

evidence = {'format': 'qpt_legacy_addendum_evidence_v1', 'pdf': str(OUT), 'pdf_sha256': sha(OUT),
            'builder_sha256': sha(Path(__file__)), 'inputs_sha256': inputs,
            'n8_continuation': n8, 'n8_sampling_replicates': replicates,
            'n8_baseline_summary': momentum, 'n10_reduced_noise': n10, 'n10_outcomes': n10_outcomes,
            'qualification': 'Historical results, not new runs; 10q native factor not reloaded.'}
MANIFEST.write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps({'pdf': str(OUT), 'pages': 3, 'sha256': sha(OUT), 'evidence': str(MANIFEST)}, indent=2))
