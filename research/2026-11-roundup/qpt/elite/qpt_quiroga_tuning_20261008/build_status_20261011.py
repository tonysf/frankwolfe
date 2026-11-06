"""Read-only status findings rendered as a new PDF; no optimization."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'QPT_STATUS_20261011.pdf'
assert not OUT.exists()
plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42})
fig=plt.figure(figsize=(8.27,11.69),facecolor='white')
INK,TEAL,MUTED='#163044','#087e8b','#526878'
def text(y,s,size=11,color=INK,**kw):
    fig.text(.075,y,s,va='top',fontsize=size,color=color,**kw)
def para(y,s,size=11,width=90):
    lines=textwrap.wrap(s,width,break_long_words=False,break_on_hyphens=False)
    text(y,'\n'.join(lines),size,linespacing=1.3)
    return y-len(lines)*size*1.3/(11.69*72)-.017
def section(y,title,body):
    text(y,title,12,TEAL,weight='bold')
    return para(y-.028,body)
text(.95,'QPT · CURRENT STATUS',10,TEAL,weight='bold')
text(.915,'No new QPT results since 8 October',21,weight='bold')
text(.865,'Checked 11 October 2026, approximately 19:13 Europe/Paris',10,MUTED)
y=section(.82,'Operational status',
    'Live Ruche checks found no running or pending QPT jobs. Scheduler accounting since 8 October shows only the already archived QPT tuning batches from that day. Other running/queued jobs belong to the GW project. No new QPT submission, cancellation, optimization or source edit was made for this status check.')
y=section(y,'Best verified current result',
    'The selected fixed-penalty FGD configuration with our modified exact factor-ray line search passes all three gates on 16 distinct two-qubit channels: fidelity ≥0.99, full-design RMSE ≤10⁻⁶, and normalized TP residual ≤0.001. Worst RMSE is 1.61×10⁻¹⁰; worst normalized TP is 7.80×10⁻¹⁰. These are noiseless rank-one tests, including historical revalidation—not an unseen-test claim.')
y=section(y,'What remains unresolved',
    'The printed Quiroga adaptive ratio passed 0/44 new tuning trials: 14 completed outside the gate and 30 retained numerical failures. That does not disprove the published results. Prior paired FRAMES diagnostics pass 1/2 channels; the other misses only the RMSE threshold. Independently tuned, best-for-best FRAMES versus published adaFGD is unfinished.')
y=section(y,'Older work and available deliverables',
    'The local-Pauli results remain a separate benchmark: eight-qubit fidelity about 98.92–99.22% across sampling streams on one target, and ten-qubit 99.00% with 32× lower observation noise. TP remains approximate. The coauthor briefing, legacy addendum, loss plots and author-question checklist are complete and unchanged.')
y=section(y,'Next scientific step',
    'Request one runnable two-qubit example from Quiroga/Kyrillidis, especially the actual adaptive step, TP weight, POVM constants and initialization conventions. In parallel, independently calibrate the FRAMES initial penalty and increasing-penalty schedule. Freeze settings before final held-out evaluation, then extend the matched comparison to more qubits. No author email has been sent by this assistant.')
y=para(y,'Code: local checkout is clean at 4ce1108 and six commits ahead of its cached origin branch. The completion record says merged locally, not pushed; the live Git remote was not checked today.',10,97)
assert y>.11,y
text(.094,'Evidence checked',10,TEAL,weight='bold')
text(.073,'Live Ruche squeue / sacct; completion_manifest.json; three tuning audits; prior paired audit;\nlocal Git status and bounded QPT artifact inventory. Detailed reports remain beside this PDF.',8.8,MUTED,linespacing=1.3)
text(.035,'Status report only · existing artifacts preserved · no new experimental results',8.5,MUTED)
fig.savefig(OUT,metadata={'Title':'QPT status — 11 October 2026','Author':'QPT project'})
print(OUT)
