"""One-page author-contact checklist; no contact is made."""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).with_name('QUIROGA_REPRODUCTION_QUESTIONS_20261008.pdf')
assert not OUT.exists(), 'Do not overwrite an existing deliverable.'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
INK, MUTED, TEAL = '#163044', '#526878', '#087e8b'

def text(y, body, size=10.5, color=INK, **kw):
    return fig.text(.075, y, body, va='top', fontsize=size, color=color, **kw)

def para(y, body, size=10.5, width=93, color=INK):
    lines = textwrap.wrap(body, width, break_long_words=False, break_on_hyphens=False)
    text(y, '\n'.join(lines), size, color, linespacing=1.28)
    return y - len(lines)*size*1.28/(11.69*72) - .012

def section(y, heading, body):
    text(y, heading, 11.2, TEAL, weight='bold')
    return para(y-.023, body)

text(.953, 'QPT · REPRODUCIBILITY CHECKLIST', 9.5, TEAL, weight='bold')
text(.921, 'What to ask Quiroga and Kyrillidis', 20, weight='bold')
y = para(.878, 'Priority: obtain one working two-qubit adaFGD example, preferably the full 128-row case. Below, “missing values” means not specified in the inspected paper; other items are implementation confirmations, not claims that the paper omits the mathematics.', 10.5)
fig.add_artist(Rectangle((.075, y+.003), .85, .002, transform=fig.transFigure, facecolor=TEAL, edgecolor='none'))
y -= .015
y = section(y, '1. Exact adaptive update — highest priority',
    'Could you share the step-size and update functions actually used? In particular: the proportionality constant, spectral versus Frobenius norm, any squared quantities, inclusion of the TP gradient in the step calculation, and any clipping, denominator safeguard or line search. We should not assume an unreported safeguard exists.')
text(y, r'$\eta_t=c\,\frac{\|\mathcal{A}^{\dagger}(\mathcal{A}(U_tU_t^{\dagger})-f)\|_2}{\|\mathcal{A}(U_tU_t^{\dagger})\|_2}$', 13)
y -= .056
y = section(y, '2. TP penalty and objective scaling — missing λ; confirm conventions',
    'What λ was used for each experiment, and was it fixed during each run? Please confirm summed versus averaged residual loss, scaling of the sensing operator/data, and factors of two in the implemented gradient. A numerical λ is not transferable without these conventions.')
y = section(y, '3. Measurement construction — missing a and b',
    'What values of the POVM constants a,b were used at each dimension? Could you share the probe/POVM generators or the actual sensing matrices for one example, including the final complement outcome? For the 96-row case, which rows were retained and how were they selected?')
y = section(y, '4. Initialization and operator basis — incomplete specification',
    'What uniform interval was sampled for real/imaginary entries? Was the random object U₀ or χ₀, and how was the rank-one factor formed? Was it trace-normalized or projected? Please provide the exact Gell-Mann basis scaling/order, process-matrix normalization, and one saved U₀.')
y = section(y, '5. Evaluation and stopping — confirm the implemented definitions',
    'Could you share the fidelity routine, including normalization and whether the reference is the ideal or noise-affected channel? What TP residual was actually achieved? Please include stopping tolerances, maximum updates and any final normalization/projection. Our strict joint gate is not a reported criterion of their paper.')
y = section(y, '6. Reproduction bundle — more useful than constants alone',
    'Best request: the figure-generating code/configuration plus one target, sensing operator, fixed observation vector, initial factor and RNG seeds. A short trace of step sizes, loss, TP residual and fidelity for the first 5–10 updates would localize any mismatch. For noisy figures, include clipping/renormalization of observations and what was varied across repetitions.')
y = para(y-.004, 'Already stated: rank-one target construction, the probe/POVM families, a summed least-squares objective, and 128/96-row two-qubit settings. Ask for their realized implementation—not a new method or an assumed correction to the printed rule.', 9.5, width=104, color=MUTED)
assert y > .087, f'Content runs into footer: {y}'
text(.072, 'Source: Quiroga & Kyrillidis, arXiv:2312.01311v1, §III-B and §V (click to open).', 8.5, TEAL,
     url='https://arxiv.org/html/2312.01311v1')
text(.050, 'Prepared 8 October 2026 · based on the paper and our implementation audit · no email sent', 8.5, MUTED)
fig.savefig(OUT, metadata={'Title': 'Quiroga QPT reproduction: author questions', 'Author': 'QPT project'})
plt.close(fig)
print(OUT)
