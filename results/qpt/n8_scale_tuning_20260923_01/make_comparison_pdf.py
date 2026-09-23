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
OUT=REPO/'output/pdf/qpt_n8_scale_tuning_20260923.pdf'
DATA=json.loads((ROOT/'comparison_data.json').read_text())
RUNS=DATA['runs']; PLAN=DATA['plan']; PREFLIGHT=DATA['preflight']
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
    canvas.drawRightString(PW-MARGIN,PH-22,'Eight-qubit scale comparison | 23 September 2026')
    canvas.line(MARGIN,30,PW-MARGIN,30)
    canvas.drawString(MARGIN,19,'Four fresh runs | one dataset / seed | requested exponents retained | all final results verified')
    canvas.drawRightString(PW-MARGIN,19,str(doc.page))


def newpage(title):story.append(PageBreak());add(title,'title')
def compact(x):return f'{x:.7g}'
def scales(run):return f"a={run['case']['step_scale']:g}, r={run['case']['rho_scale']:g}"
def pic(name,ratio):story.append(Image(str(ROOT/'comparison_figures'/f'{name}.png'),width=WIDTH,height=WIDTH*ratio))

doc=BaseDocTemplate(str(OUT),pagesize=(PW,PH),leftMargin=MARGIN,rightMargin=MARGIN,
    topMargin=44,bottomMargin=42,title='Structured stochastic-FRAMES QPT: eight-qubit scales and preparation',
    author='QPT experiment comparison',pageCompression=1)
doc.addPageTemplates([PageTemplate(id='landscape',pagesize=(PW,PH),
    frames=[Frame(MARGIN,42,WIDTH,PH-86,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=chrome)])
story=[]
best=max(RUNS,key=lambda r:r['new'][-1]['fidelity'])
old=RUNS[0]['old'][-1]; requested=RUNS[0]['requested'][-1]
hit=any(p['fidelity']>=.99 for r in RUNS for p in r['new'])

# Page 1.
add('Eight qubits: retuning the scales','title')
add('Usual noise σ=0.05 | batch 65,536 | 148,000 iterations | step/smoothing/momentum exponents 3/4, 1/4, 1/2','small')
add(f"Best new fidelity: {100*best['new'][-1]['fidelity']:.4f}% at {scales(best)}. " + ('A new saved checkpoint reaches 99%.' if hit else 'No new saved checkpoint reaches 99%.'),'callout')
rows=[]
for label,a,r,point in [('Old successful exponents','10','2',old),('Requested-exponent baseline','10','2',requested)]+[(f"Scale case {j+1}",f"{x['case']['step_scale']:g}",f"{x['case']['rho_scale']:g}",x['new'][-1]) for j,x in enumerate(RUNS)]:
    rows.append([label,a,r,f"{100*point['fidelity']:.4f}",compact(point['measurement_loss']),compact(point['tp_violation'])])
story.append(table(['Run','Step scale a','Momentum scale r','Final F (%)','Sampled loss','TP residual'],rows,[198,92,117,112,124,WIDTH-643],pad=7))
story.append(Spacer(1,12))
add(f"<b>Recovery improves after reducing the scales.</b> The best new endpoint is {100*(best['new'][-1]['fidelity']-requested['fidelity']):.3f} percentage points above the requested-exponent baseline. Its TP residual is {best['new'][-1]['tp_violation']:.4f}, compared with {requested['tp_violation']:.4f} for that baseline. The old exponents reached 99.0027% at the same endpoint.")
add('<b>What this tests.</b> Only the step and momentum scales vary among the four new cases. All use the same fixed dataset, original initial factor, sampled-row sequence, smoothing schedule and iteration budget. Each starts fresh; none stops early. No new smoothing-scale sweep or seed study is included.')
add('<b>Preparation is faster.</b> Exact host preparation falls from 24.28 to 5.92 ms per batch (4.10×). Three alternating pairs of 1,000-step trials give a median complete-runner speedup of 1.786×. Full factors, momentum and saved diagnostics match bitwise in those checks.','body')
add('All numbers describe one Haar-unitary channel and one seed. The best observed scales are not established universal optima. Fidelity is a normalized factor overlap, distinct from measurement loss and from trace preservation.','small')

# Page 2.
newpage('Exact schedules and effective update sizes')
add('k starts at 0; t completed iterations means the last applied update used k=t-1. All runs have rank 1 and operator-norm radius τ=20.','small')
story.append(table(['Quantity','Old successful exponents','Requested exponents, all five runs','Role'],[
 ['Step γ<sub>k</sub>','min(1, 10/(k+10))','min(1, a/(k+10)<super>3/4</super>)','Weight on the Frank-Wolfe atom'],
 ['Smoothing β<sub>k</sub>','10<super>7</super>/(k+1)<super>1/4</super>','10<super>7</super>/(k+1)<super>1/4</super>','TP penalty denominator; no cap'],
 ['Momentum ρ<sub>k</sub>','min(1, 2/(k+4)<super>0.6</super>)','min(1, r/(k+4)<super>1/2</super>)','Weight on the new batch gradient'],
],[95,205,250,WIDTH-550],pad=6))
story.append(Spacer(1,9))
add('d<sub>k</sub>=(1-ρ<sub>k</sub>)d<sub>k-1</sub>+ρ<sub>k</sub>ĝ<sub>k</sub>; the first gradient is assigned directly. Thus smaller r means more averaging of past gradients. The exact TP gradient is added after this averaging, divided by β<sub>k</sub>. The update is u<sub>k+1</sub>=(1-γ<sub>k</sub>)u<sub>k</sub>+γ<sub>k</sub>s<sub>k</sub>.')
rows=[]
for label,a,r,pe,qe in [('Old',10,2,1,.6),('Requested baseline',10,2,.75,.5)]+[(scales(x),x['case']['step_scale'],x['case']['rho_scale'],.75,.5) for x in RUNS]:
    rows.append([label,f'{min(1,a/10**pe):.6g}',f'{min(1,a/148009**pe):.6g}',f'{min(1,r/4**qe):.6g}',f'{min(1,r/148003**qe):.6g}'])
story.append(table(['Run','γ at k=0','γ at k=147,999','ρ at k=0*','ρ at k=147,999'],rows,[210,137,147,123,WIDTH-617],pad=5))
story.append(Spacer(1,9))
add('The requested baseline keeps a=10, making its late γ about 19.61× the old γ. Reducing a to 0.5 makes the terminal γ about 0.981× the old value; a=1 gives 1.961×. These ratios only match the endpoint: the full time courses still differ. β starts at 10,000,000 and ends at 509,840.781 for the new fixed-budget runs.','body')
add('*The first-gradient assignment bypasses averaging at k=0. The requested baseline step cap is active for k=0,...,11; it is inactive in all four retuned cases. The old step equals 1 only at k=0. Offsets remain 10/1/4; no late penalty strengthening is introduced.','small')

# Page 3.
newpage('Fidelity, the objective, and the virtual dataset')
add('<b>Fidelity is not f in f+g.</b> For the complete rank-one factors c (truth) and u (iterate), F=|c†u|<super>2</super>/(||c||<super>2</super>||u||<super>2</super>). It is the squared overlap of normalized factors, equivalently fidelity of their trace-normalized rank-one process matrices. Global phase and nonzero scale do not change it. A high value does not certify trace preservation or normalization.')
add('<b>Measurement fitting and TP.</b> f(U)=(1/(2M))Σ<sub>s</sub>(A<sub>s</sub>(UU†)-y<sub>s</sub>)<super>2</super>. The function g is the indicator of the TP constraint T(UU†)=I. The implemented smoothed objective is f(U)+||T(UU†)-I||<sub>F</sub><super>2</super>/(2β<sub>k</sub>), subject to ||U||<sub>op</sub>≤20. TP in the tables is the unnormalized Frobenius residual, before squaring.')
story.append(table(['Budget / setting','Value','Meaning'],[
 ['Qubits / factor length','8 / 65,536 complex entries','Rank-one process factor; Hilbert-space dimension 256'],
 ['Total virtual rows M=24<super>8</super>',f'{24**8:,}','Full observation table is not materialized'],
 ['Optimizer batch B','65,536',f'{65536/24**8*100:.8f}% of M per update'],
 ['Budget T / total B×T','148,000 / 9,699,328,000',f'{65536*148000/24**8*100:.6f}% of M as draws, not unique coverage'],
 ['Metric rows / frequency','512 / every 1,000 iterations','Fixed independent sample; metric seed 12345'],
 ['Noise / other seeds','σ=0.05 / all 0','Channel, noise, initialization and optimizer sampling'],
],[185,215,WIDTH-400],pad=6))
story.append(Spacer(1,9))
add('Rows are sampled uniformly with replacement. Targets are generated on demand from the truth plus fixed row-dependent Gaussian noise; revisiting a row gives the same observation. The noiseless target is evaluated on the GPU; row sampling, integer decoding and fixed noise preparation use the CPU. Prefetch overlaps preparation with GPU work.')
add('Fidelity and TP use complete factors. Loss uses the fixed 512-row sample, so similar sampled losses need not imply similar recovery. The noisy-truth expected loss is σ²/2=0.00125; this is not a measured noise floor or a convergence certificate. Draw counts exclude metric and smoke-test work.','small')

# Pages 4 and 5.
for title,figure,caption in [
 ('Step scale 0.5: compare momentum scales','step_half','At the same step scale, r=0.6 gives markedly better fidelity than r=2. The lower new-gradient weight also gives a smaller terminal TP residual in this pair.'),
 ('Step scale 1: compare momentum scales','step_one','The two runs differ only in momentum scale. Compare fidelity and TP directly; the sampled measurement-loss curves occupy a much narrower numerical range.')]:
    newpage(title);pic(figure,7.35/13)
    add(caption,'small')
    add('All trajectories are solid, unsmoothed saved scalar evidence. The dotted fidelity line marks 99%. TP uses a logarithmic axis; loss is linear with scientific notation. Blue and gray repeat the same two reference runs in every panel.','small')

# Page 6.
newpage('Late recovery and equal sampled-row budgets')
pic('late_comparison',5.6/13)
rows=[]
for r in RUNS:
    values={p['step']:p for p in r['new']}
    rows.append([scales(r)]+[f"{100*values[t]['fidelity']:.4f}%" for t in (100000,120000,140000,148000)])
story.append(table(['Scales','100,000 steps','120,000 steps','140,000 steps','148,000 steps'],rows,[210,139,139,139,WIDTH-627],pad=4))
story.append(Spacer(1,6))
add('Every plotted run uses B=65,536, so equal iteration counts also mean equal optimizer row-draw budgets. The final 148,000-step budget is 9.699328 billion draws. Late progress is descriptive evidence; it does not establish that an extension will cross 99% or where a run will settle.','small')

# Page 7.
newpage('Faster preparation, with exact numerical equivalence')
pic('preparation_speedup',3.0/13)
add('The old path decoded sampled integer rows into symbols, then re-encoded those symbols to generate noise. The optimized path retains the original rows for the same NumPy noise function, uses smaller integer temporaries and contiguous decoder writes, and removes redundant host work. PCG64 sampling, row order, symbols and fixed-noise values are preserved.','body')
rows=[]
for label,md in [('Requested baseline (original prep)',RUNS[0]['requested_metadata'])]+[(scales(x)+' (optimized prep)',x['metadata']) for x in RUNS]:
    rows.append([label]+[f"{md[key]/60:.2f}" for key in ('host_preparation_seconds','optimizer_seconds','wall_seconds')])
story.append(table(['148,000-step run','Host prep (min)','Optimizer (min)','Complete runner (min)'],rows,[310,145,145,WIDTH-600],pad=4))
story.append(Spacer(1,7))
add('<b>Timing definitions.</b> Host preparation is accumulated wall time in the background preparation task; it overlaps GPU execution. Optimizer time excludes setup and metrics. Runner time starts after the separately recorded setup phase; it includes compilation, preparation, transfers, metrics and restart checkpointing. It excludes process startup, initial setup, final archive serialization and separate verification/smoke work. These totals must not be added; whole-job elapsed time is reported separately.','small')
add('The paired preflight isolates preparation changes: median runner 28.3168→15.8563 s, 44.0% less time. Full production runs also change scales and ran at different times, so their entire speed difference cannot be assigned to the optimization. Integer GPU decoding was profiled separately (0.0603 ms resident rows; 0.2000 ms including transfer), but is not used in production. GPU noise generation was not implemented.','small')

# Page 8.
newpage('Validation, provenance, and limits')
manifest=json.loads((ROOT/'manifest.json').read_text())
rows=[]
for r in RUNS:
    v=r['validation'];bestpoint=v['best_new_checkpoint'];job=manifest['last_scheduler_check'][v['job_id']]
    first=next((p['step'] for p in r['new'] if p['fidelity']>=.99),None)
    rows.append([scales(r),v['job_id'],job['elapsed'],f"{100*bestpoint['fidelity']:.4f}%",f"{bestpoint['step']:,}",f'{first:,}' if first is not None else 'Not reached'])
story.append(table(['Scales','Slurm task','Job elapsed','Best saved F','Best step','First saved ≥99%'],rows,[130,127,107,116,107,WIDTH-587],pad=6))
story.append(Spacer(1,10))
add('<b>All four results are valid completed runs.</b> Each scheduler record is COMPLETED with exit 0. Each case passes requested-schedule/cap and full-budget checks, source/dataset/U0/sampling hashes, and independent allocated NumPy fidelity and TP calculations from complete saved factors. All 48 case checksum-listed files match locally, including final and restart archives. No algorithms or defaults were changed.')
add('<b>Preparation validation.</b> Preflight 1931810 passed decoder checks for n=1,...,13, exact batch symbols/noise/RNG and partition-invariance checks, three alternating original/optimized 1,000-step comparisons of full factors, momentum and saved diagnostics, and an original-to-optimized restart check. Its 27 checksum-listed files are backed up. The initial preflight 1931798 failed because pytest was unavailable; its evidence is preserved. No pytest suite was run; the standalone checks are the validation.','body')
add('<b>Hardware and source.</b> One NVIDIA A100-SXM4-40GB per job on ruche-gpu15, two CPUs, 12G RAM, array concurrency one; no multi-GPU execution. JAX/jaxlib 0.11.1, NumPy 2.5.3, float64/complex128, XLA GPU autotuning disabled. Original frozen source: fidelity99_prefetch_validation_20260912_01/source, base revision ba85bff73c30. Optimized preparation is isolated under this campaign’s source_fast; original source and main-checkout defaults are preserved.','small')
newpage('Reference runs and reproducibility records')
add('<b>Reference-run details.</b> The old successful trace joins an exact continuation at 100,000 steps (chunk 10, then 1); new runs use the validated chunk-1 path throughout and start at U0. The old run stopped at 148,000 after crossing 99%; new runs have this fixed budget without early stopping. The old endpoint diagnostic β uses the next update, versus the last applied update in new fixed-budget runs; this tiny one-index difference affects smoothed diagnostics, not F, loss or TP.','small')
add('<b>Evidence location.</b> results/qpt/n8_scale_tuning_20260923_01 contains plan.json, manifest.json, comparison_data.json, plotting/report builders, preflight evidence, and per-case scalar traces, command provenance, checksums and archives. Dataset SHA-256: '+PLAN['cases'][0]['data_sha256']+'; U0 SHA-256: '+PLAN['cases'][0]['initial_factor_sha256']+'.','small')
story.append(table(['Reproducibility record','Value'],[
 ['Campaign','n8_scale_tuning_20260923_01'],
 ['Original source snapshot','results/qpt/fidelity99_prefetch_validation_20260912_01/source'],
 ['Selected source snapshot','results/qpt/n8_scale_tuning_20260923_01/source_fast'],
 ['Dataset (reused)','results/qpt/on_demand_profile_20260912_04/n8.npz'],
 ['Prior requested-exponent run','results/qpt/exponents_20260922_01/n8_b65536_fixed'],
 ['Base revision',PLAN['base_commit']],
 ['Environment','anaconda3/2023.09-0/none-none; qpt; XLA_FLAGS=--xla_gpu_autotune_level=0'],
 ['Source hashes','Full per-file SHA-256 records: source_deployment.json and per-case provenance.json'],
 ],[175,WIDTH-175],pad=6))
story.append(Spacer(1,10))
add('One channel and one noise/initialization/sampling seed. The four tested scale pairs do not establish an optimum or convergence guarantee. Previous PDFs and every old/new result remain preserved.','small')

OUT.parent.mkdir(parents=True,exist_ok=True)
doc.build(story)
print(OUT)

