"""Coauthor report from verified scalar traces. Reuses ReportLab report styling."""
import json
import math
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
OUT=REPO/'output/pdf/qpt_exponent_comparison_20260922.pdf'
DATA=json.loads((ROOT/'comparison_data.json').read_text())
RUNS=DATA['runs']; BY_NAME={r['case']['name']:r for r in RUNS}
FONTS=Path('/Users/tsf/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
for name,filename in [('DejaVu','DejaVuSans.ttf'),('DejaVu-Bold','DejaVuSans-Bold.ttf'),('DejaVu-Oblique','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/filename)))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu-Oblique',boldItalic='DejaVu-Bold')
INK=colors.HexColor('#253343'); BLUE=colors.HexColor('#176B9A'); MUTED=colors.HexColor('#617183')
LINE=colors.HexColor('#DCE4EA'); PALE=colors.HexColor('#EBF4F9')
PW,PH=landscape(A4); MARGIN=38; WIDTH=PW-2*MARGIN
ST={
 'body':ParagraphStyle('body',fontName='DejaVu',fontSize=9.4,leading=13.4,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='DejaVu',fontSize=8.1,leading=11.4,textColor=MUTED,spaceAfter=7),
 'title':ParagraphStyle('title',fontName='DejaVu-Bold',fontSize=21,leading=26,textColor=INK,spaceAfter=9),
 'section':ParagraphStyle('section',fontName='DejaVu-Bold',fontSize=11.6,leading=15,textColor=BLUE,spaceBefore=10,spaceAfter=6),
 'table':ParagraphStyle('table',fontName='DejaVu',fontSize=8.0,leading=10.4,textColor=INK),
 'head':ParagraphStyle('head',fontName='DejaVu-Bold',fontSize=8.0,leading=10.4,textColor=colors.white),
 'callout':ParagraphStyle('callout',fontName='DejaVu-Bold',fontSize=11.5,leading=16,textColor=BLUE,spaceAfter=8),
}


def p(text,kind='body'):return Paragraph(str(text),ST[kind])
def add(text,kind='body'):story.append(p(text,kind))
def table(headers,rows,widths,pad=5):
    t=Table([[p(x,'head') for x in headers]]+[[p(x,'table') for x in row] for row in rows],
            colWidths=widths,hAlign='LEFT',repeatRows=1)
    commands=[('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
      ('TOPPADDING',(0,0),(-1,-1),pad),('BOTTOMPADDING',(0,0),(-1,-1),pad)]
    for row in range(1,len(rows)+1):
        if row%2==0:commands.append(('BACKGROUND',(0,row),(-1,row),colors.HexColor('#F5F7F9')))
        commands.append(('LINEBELOW',(0,row),(-1,row),.4,LINE))
    t.setStyle(TableStyle(commands));return t


def chrome(canvas,doc):
    canvas.setStrokeColor(LINE);canvas.line(MARGIN,PH-31,PW-MARGIN,PH-31)
    canvas.setFont('DejaVu',7);canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN,PH-22,'STOCHASTIC-FRAMES / QUANTUM PROCESS TOMOGRAPHY')
    canvas.drawRightString(PW-MARGIN,PH-22,'Exponent comparison | 22-23 September 2026')
    canvas.line(MARGIN,30,PW-MARGIN,30)
    canvas.drawString(MARGIN,19,'12 matched-budget comparisons | one dataset / seed per condition | all final results verified')
    canvas.drawRightString(PW-MARGIN,19,str(doc.page))


def label(c):
    if c['n_qubits']<8:return f"n={c['n_qubits']}, β scale {c['smoothing_scale']:,}"
    if c['n_qubits']==8:return f"n=8, B={c['batch_size']:,}"
    return 'n=10, reduced noise*' if c['noise_std']<.05 else 'n=10, usual noise'


def fidelity(x):return f'{x*100:.6f}' if x<.001 else f'{x*100:.4f}'
def compact(x):return f'{x:.4g}'
def newpage(title):story.append(PageBreak());add(title,'title')


doc=BaseDocTemplate(str(OUT),pagesize=(PW,PH),leftMargin=MARGIN,rightMargin=MARGIN,
    topMargin=44,bottomMargin=42,title='Structured stochastic-FRAMES QPT: old versus requested exponents',
    author='QPT experiment comparison',pageCompression=1)
doc.addPageTemplates([PageTemplate(id='landscape',pagesize=(PW,PH),
    frames=[Frame(MARGIN,42,WIDTH,PH-86,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=chrome)])
story=[]

# Page 1: all endpoints, including every small-size sweep setting.
add('Requested exponents: mixed recovery, larger TP residuals','title')
add('Step exponent 1 → 3/4; momentum exponent 0.6 → 1/2; smoothing stays 1/4. Scales and offsets were retained.','small')
add('At 8 qubits and σ = 0.05, batch 65,536 falls from 99.0027% to 90.7749% fidelity at 148,000 steps.','callout')
rows=[]
for r in RUNS:
    c=r['case']; old=r['old'][-1]; new=r['new'][-1]
    rows.append([label(c),f"{c['steps']:,}",fidelity(old['fidelity']),fidelity(new['fidelity']),
        f"{100*(new['fidelity']-old['fidelity']):+.3f}",compact(old['measurement_loss']),compact(new['measurement_loss']),
        compact(old['tp_violation']),compact(new['tp_violation'])])
story.append(table(['Condition','Steps','Old F (%)','New F (%)','Δ F (pp)','Old loss','New loss','Old TP','New TP'],
    rows,[152,60,80,80,60,86,86,80,WIDTH-684],pad=4.8))
story.append(Spacer(1,8))
add('<b>What changed.</b> Fidelity improves in five of twelve comparisons, including n=5 at smoothing scale 1,000 (72.686% → 93.062%). The new n=4, scale-100 run retains 99.669%. Every new endpoint has a larger TP residual. Similar sampled losses can accompany very different recovery.','body')
add('*Reduced-noise control: σ = 0.0015625 (32 times smaller); all other rows use σ = 0.05. F is normalized factor fidelity; TP is the unnormalized Frobenius residual. Δ F uses percentage points. These endpoints describe one dataset and seed, with no scale retuning.','small')

# Page 2: exact formulas, caps, scales and terminal step values.
newpage('Schedules and parameter choices')
add('Update index k starts at 0. Iteration t is after t updates; the last applied update is k=t-1. Rank is 1 throughout.','small')
story.append(table(['Quantity','Old schedule','Requested schedule','Interpretation'],[
 ['Step γ<sub>k</sub>','min(1, a/(k+b)<super>1</super>)','min(1, a/(k+b)<super>3/4</super>)','Weight on the Frank-Wolfe atom'],
 ['Smoothing β<sub>k</sub>','s/(k+1)<super>1/4</super>','s/(k+1)<super>1/4</super>','TP penalty denominator; no cap'],
 ['Momentum ρ<sub>k</sub>','min(1, 2/(k+4)<super>0.6</super>)','min(1, 2/(k+4)<super>1/2</super>)','Weight on the new batch gradient'],
],[95,190,190,WIDTH-475]))
story.append(Spacer(1,8))
add('d<sub>k</sub> = (1-ρ<sub>k</sub>)d<sub>k-1</sub> + ρ<sub>k</sub>ĝ<sub>k</sub>; the first gradient is assigned directly. The exact TP gradient is added after this averaging, divided by β<sub>k</sub>. The factor update is u<sub>k+1</sub> = (1-γ<sub>k</sub>)u<sub>k</sub> + γ<sub>k</sub>s<sub>k</sub>. The retained momentum weight is 1-ρ<sub>k</sub>.')
story.append(table(['Conditions','a / b','Smoothing scale s','Radius τ','Batch B','Fixed steps T'],[
 ['n=4, four scales','2 / 2','10; 100; 1,000; 10,000','10','32','10,000 each'],
 ['n=5, four scales','2 / 2','10; 100; 1,000; 10,000','10','32','10,000 each'],
 ['n=8, usual noise','10 / 10','10,000,000','20','8,192','200,000'],
 ['n=8, usual noise','10 / 10','10,000,000','20','65,536','148,000'],
 ['n=10, usual noise','10 / 10','1,000,000,000','40','65,536','100,000'],
 ['n=10, reduced noise','10 / 10','1,000,000,000','40','65,536','57,000'],
],[155,80,195,70,100,WIDTH-600],pad=4))
story.append(Spacer(1,8))
add('<b>The new late step sizes are much larger.</b> For n=8 at k=147,999: γ changes from 6.75635×10<super>-5</super> to 1.32521×10<super>-3</super> (19.61×); ρ changes from 0.00158077 to 0.00519870 (3.29×). β stays 509,840.781. Thus the comparison changes effective update magnitudes as well as their decay rates.','body')
add('The new step cap is active for k=0,...,11 in n=8/10 and k=0 in n=4/5. The old step cap equals 1 only at k=0. The new ρ equals 1 at k=0; the first-gradient special case makes this inconsequential. Larger smoothing scale weakens TP enforcement throughout the schedule.','small')

# Page 3: definitions and dataset accounting.
newpage('Fidelity, the objective, and the data budget')
add('<b>Fidelity is a separate recovery diagnostic; it is not f.</b> With χ=uu† and synthetic truth χ⋆=cc†, F(u,c)=|c†u|<super>2</super>/(||c||<super>2</super>||u||<super>2</super>). This is the squared overlap of normalized complete factors, equivalently fidelity between their trace-normalized rank-one process matrices. Phase and nonzero scale do not affect it. High F alone does not certify trace preservation or normalization.')
add('<b>The f+g formulation.</b> f(U)=(1/(2M)) Σ<sub>s</sub>(A<sub>s</sub>(UU†)-y<sub>s</sub>)<super>2</super> is measurement least-squares. g(Z) is the indicator of {I}: zero at Z=I and infinity otherwise, with Z=T(UU†). The implemented smoothed objective is f(U)+||T(UU†)-I||<sub>F</sub><super>2</super>/(2β<sub>k</sub>), over ||U||<sub>op</sub>≤τ.')
add('Fidelity and TP use complete factors. Plotted loss and smoothed gap use the same fixed, independent sample of 512 rows (metric seed 12345). The noisy-truth expected loss is σ<super>2</super>/2, not a measured floor or convergence certificate. Smoothed objectives/gaps at different smoothing scales are different penalized objectives.','small')
story.append(table(['Qubits / condition','Full row table M=24<super>n</super>','Batch B','B / M (%)','B × T row draws','Draws / M (%)'],[
 ['4 (each scale)','331,776','32',f'{32/24**4*100:.7g}','320,000',f'{320000/24**4*100:.7g}'],
 ['5 (each scale)','7,962,624','32',f'{32/24**5*100:.7g}','320,000',f'{320000/24**5*100:.7g}'],
 ['8, smaller batch',f'{24**8:,}','8,192',f'{8192/24**8*100:.7g}',f'{8192*200000:,}',f'{8192*200000/24**8*100:.7g}'],
 ['8, larger batch',f'{24**8:,}','65,536',f'{65536/24**8*100:.7g}',f'{65536*148000:,}',f'{65536*148000/24**8*100:.7g}'],
 ['10, usual noise',f'{24**10:,}','65,536',f'{65536/24**10*100:.7g}',f'{65536*100000:,}',f'{65536*100000/24**10*100:.7g}'],
 ['10, reduced noise',f'{24**10:,}','65,536',f'{65536/24**10*100:.7g}',f'{65536*57000:,}',f'{65536*57000/24**10*100:.7g}'],
],[140,175,75,113,147,WIDTH-650],pad=5))
story.append(Spacer(1,8))
add('Rows are sampled uniformly with replacement. Draw counts include repeats and exclude metric/smoke-test work; they are <b>not unique-row coverage</b>. The n=4/5 compact datasets store all noisy observations. The n=8/10 table is virtual: sampled targets are generated from the truth and fixed row-dependent noise. Revisiting a row gives the same target. CPU preparation is prefetched for n=8/10.')
add('Every condition uses Haar-unitary rank-one truth; channel, noise, initialization and sampling seeds are 0. All new runs start from the original U0 and retain their own old dataset hashes. The on-demand noise recipe differs from stored-data generation, and fixed absolute σ does not imply fixed SNR across n. One dataset/seed per condition does not establish robustness or universal optimal settings.','small')

# Pages 4-9: every trajectory, three metrics for every one of the 12 pairs.
PLOTS=[('Four qubits: smoothing scales 10 and 100','n4_strong',
 'At scale 10, new fidelity improves substantially but remains far below 99%. At scale 100, both schedules exceed 99%; the new TP residual is about 9.4 times larger.'),
 ('Four qubits: smoothing scales 1,000 and 10,000','n4_weak',
 'The old schedules exceed 99% at both scales. The requested schedules finish near 96%, with larger TP residuals. All curves show actual saved checkpoints, without fitting or smoothing.'),
 ('Five qubits: smoothing scales 10 and 100','n5_strong',
 'Fidelity axes are zoomed to show these low-recovery cases; neither approaches 99%. The requested schedules modestly improve final fidelity but worsen final TP.'),
 ('Five qubits: smoothing scales 1,000 and 10,000','n5_weak',
 'Scale 1,000 benefits most: final fidelity rises from 72.686% to 93.062%. Scale 10,000 falls from 90.680% to 89.276%. The new TP residual is larger in both.'),
 ('Eight qubits: usual noise, two batch sizes','n8',
 'The old B=65,536 trajectory reaches 99.0027%; the new one reaches 90.7749% at the same 148,000 steps. For B=8,192 at 200,000 steps, fidelity falls from 90.8853% to 75.5377%. Both new TP residuals increase.'),
 ('Ten qubits: usual noise and reduced-noise control','n10',
 'Top: σ=0.05 gives essentially no recovery under either schedule; the fidelity axis is logarithmic. Bottom: the distinct reduced-noise control, σ=0.0015625, falls from 99.0045% to 92.1621% at 57,000 steps.')]
for title,filename,caption in PLOTS:
    newpage(title)
    story.append(Image(str(ROOT/'comparison_figures'/f'{filename}.png'),width=WIDTH,height=WIDTH*7.35/13))
    add(caption,'small')
    add('Fidelity and TP are complete-factor diagnostics. Loss uses 512 fixed sampled rows. Logarithmic loss/TP axes where indicated; dotted fidelity line marks 99% when in range.','small')

# Page 10: exactly matched n8 sampled-row budgets.
newpage('Eight-qubit comparison at equal row-draw budgets')
story.append(Image(str(ROOT/'comparison_figures/n8_draw_budget.png'),width=WIDTH,height=WIDTH*5.35/13))
rows=[]
for schedule in ('old','new'):
    for name,t in [('n8_b8192_fixed',200000),('n8_b65536_fixed',25000)]:
        item=BY_NAME[name]; c=item['case']; point=next(p for p in item[schedule] if p['step']==t)
        rows.append([schedule.capitalize(),f"{c['batch_size']:,}",f'{t:,}',fidelity(point['fidelity'])+'%',
                     compact(point['measurement_loss']),compact(point['tp_violation'])])
story.append(table(['Schedule','Batch','Steps','Fidelity','Sampled loss','TP residual'],rows,[110,110,120,140,140,WIDTH-620],pad=4))
story.append(Spacer(1,6))
add('All four rows use exactly 1,638,400,000 optimizer row draws (1.48844% of the virtual table size). Under the requested schedules, the two fidelities are nearly equal at this budget. Schedules depend on iteration count, so equal draw budgets do not isolate batch variance or equalize γ, β and ρ.','small')

# Page 11: best saved points, timings, validation and provenance.
newpage('Best saved checkpoints and reproducibility')
rows=[]
for r in RUNS:
    c=r['case']; v=r['validation']; best=v['best_new_checkpoint']
    hit=next((p['step'] for p in r['new'] if p['fidelity']>=.99),None)
    rows.append([label(c),fidelity(best['fidelity'])+'%',f"{best['step']:,}",f'{hit:,}' if hit is not None else 'Not reached',
                 f"{v['optimizer_seconds']/60:.3f}",f"{v['runner_seconds']/60:.3f}"])
story.append(table(['New run','Best saved F','Best step','First saved ≥99%','Optimizer (min)','Runner (min)'],
    rows,[168,120,92,138,123,WIDTH-641],pad=4))
story.append(Spacer(1,8))
add('All new runs completed their fixed budgets even after a target crossing. Best values are maxima over saved checkpoints (every 100 updates for n=4/5, every 1,000 for n=8/10). Optimizer timing excludes setup and metrics; runner timing includes preparation, metrics and checkpointing. Neither is whole-job elapsed time.','small')
add('<b>Validation and execution.</b> All five Slurm job records completed with exit 0. Every case passed independent complete-factor NumPy checks in coefficient and matrix bases, plus dataset/U0/source hash and schedule checks. Final and restart archives are backed up locally; all 132 checksum-listed files match. n=4/5 used V100S 32GB on gpu09; n=8/10 used A100-SXM4 40GB on gpu15, float64/complex128, with GPU autotuning disabled.','small')
add('<b>Comparison details.</b> Old n=8 traces join exact continuations at 30k (small batch) and 100k (large batch); U0 matches the fresh new run. The old large batch used chunk 10 then 1, while the new run uses previously validated chunk 1 throughout. Old 99% runs stopped at 148k and 57k; new runs retain those budgets without early stopping. Saved diagnostic β can use the next update at an old early-stopped endpoint versus the last update at a fixed endpoint. This one-index difference affects smoothed diagnostics, not F, loss or TP.','small')
add('<b>Evidence.</b> Campaign: results/qpt/exponents_20260922_01. Comparison tables, all scalar traces and figure scripts are retained there, with command provenance and archive checksums per case. Frozen source: fidelity99_prefetch_validation_20260912_01/source; base revision ba85bff73c30. Jobs: 1925109 and 1925110_0..3. No algorithms/defaults, Git revisions or existing reports were changed for this comparison. The prior coauthor PDF is preserved.','small')

OUT.parent.mkdir(parents=True,exist_ok=True)
doc.build(story)
print(OUT)
