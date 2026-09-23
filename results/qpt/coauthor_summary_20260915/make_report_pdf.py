"""Create the coauthor PDF from existing reports and plotted scalar traces."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, NextPageTemplate, Image,
)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = REPO / "output/pdf/qpt_coauthor_report_20260915.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
FONTS = Path('/Users/tsf/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
for name, filename in [('DejaVu','DejaVuSans.ttf'),('DejaVu-Bold','DejaVuSans-Bold.ttf'),('DejaVu-Oblique','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/filename)))
pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold', italic='DejaVu-Oblique', boldItalic='DejaVu-Bold')
INK=colors.HexColor('#253343')
BLUE=colors.HexColor('#176B9A')
MUTED=colors.HexColor('#617183')
PALE=colors.HexColor('#EBF4F9')
LINE=colors.HexColor('#DCE4EA')
PW,PH=A4
LW,LH=landscape(A4)
MARGIN=40
WIDTH=PW-2*MARGIN
ST={
    'body':ParagraphStyle('body',fontName='DejaVu',fontSize=9.1,leading=13.2,textColor=INK,spaceAfter=8),
    'small':ParagraphStyle('small',fontName='DejaVu',fontSize=8.0,leading=11.4,textColor=MUTED,spaceAfter=7),
    'title':ParagraphStyle('title',fontName='DejaVu-Bold',fontSize=24,leading=29,textColor=INK,spaceAfter=11),
    'section':ParagraphStyle('section',fontName='DejaVu-Bold',fontSize=12.5,leading=16,textColor=BLUE,spaceBefore=13,spaceAfter=8),
    'table':ParagraphStyle('table',fontName='DejaVu',fontSize=7.5,leading=10.5,textColor=INK),
    'tablehead':ParagraphStyle('tablehead',fontName='DejaVu-Bold',fontSize=7.5,leading=10.4,textColor=colors.white),
    'callout':ParagraphStyle('callout',fontName='DejaVu-Bold',fontSize=12.7,leading=18,textColor=BLUE,spaceAfter=8),
}


def para(text,kind='body'):
    return Paragraph(text,ST[kind])


def table(headers, rows, widths, highlight=None):
    content=[[para(str(x),'tablehead') for x in headers]]
    content += [[para(str(x),'table') for x in row] for row in rows]
    obj=Table(content,colWidths=widths,hAlign='LEFT',repeatRows=1)
    style=[('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
           ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
           ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
           ('LINEBELOW',(0,0),(-1,0),.5,INK)]
    for r in range(1,len(content)):
        if r%2==0: style.append(('BACKGROUND',(0,r),(-1,r),colors.HexColor('#F5F7F9')))
        style.append(('LINEBELOW',(0,r),(-1,r),.4,LINE))
    if highlight is not None:
        style.append(('BACKGROUND',(0,highlight+1),(-1,highlight+1),PALE))
    obj.setStyle(TableStyle(style))
    return obj


def chrome(canvas,doc):
    w,h=canvas._pagesize
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN,h-34,w-MARGIN,h-34)
    canvas.setFont('DejaVu',7)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN,h-24,'STOCHASTIC-FRAMES  /  QUANTUM PROCESS TOMOGRAPHY')
    canvas.drawRightString(w-MARGIN,h-24,'Coauthor report | 15 September 2026')
    canvas.line(MARGIN,32,w-MARGIN,32)
    canvas.drawString(MARGIN,20,'Existing experiments through 13 September 2026; no new runs for this report.')
    canvas.drawRightString(w-MARGIN,20,str(doc.page))


doc=BaseDocTemplate(str(OUT),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
                    topMargin=47,bottomMargin=44,title='Structured stochastic-FRAMES QPT - coauthor report',
                    author='Research experiment summary',pageCompression=1)
doc.addPageTemplates([
    PageTemplate(id='portrait',pagesize=A4,frames=[Frame(MARGIN,43,WIDTH,PH-91,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=chrome),
    PageTemplate(id='landscape',pagesize=landscape(A4),frames=[Frame(MARGIN,43,LW-2*MARGIN,LH-89,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=chrome),
])
story=[]
def add(text,kind='body'): story.append(para(text,kind))
def section(text): add(text,'section')
def nextpage(template): story.extend([NextPageTemplate(template),PageBreak()])

# 1. Main finding, selected recovery outcomes, and engineering changes.
add('Structured stochastic-FRAMES<br/>quantum process tomography','title')
add('Results, convergence plots and parameter choices','small')
add('8 qubits: 99.00265% fidelity at the original noise level','callout')
add('The rank-one Haar-unitary instance reached <b>99.00265% normalized factor fidelity at 148,000 iterations</b>, with additive Gaussian observation noise standard deviation <b>σ = 0.05</b>. Independent NumPy calculations in coefficient and matrix bases agree with the saved JAX result.')
add('Trace preservation remains approximate: the final residual is <b>0.19901</b> in Frobenius norm, or <b>1.244% of ||I||<sub>F</sub></b>. This establishes recovery for one dataset and seed; replication across seeds has not been done.')
section('Selected completed runs')
add('These configurations were tuned separately. The table is not a controlled scaling comparison across qubit counts. TP residuals are unnormalized Frobenius norms.','small')
story.append(table(['Qubits','Noise σ','Batch','Iterations','Fidelity','TP residual'],[
    ['4','0.05','32','10,000','99.6561%','0.00218'],
    ['5','0.05','32','10,000','90.6799%','0.07742'],
    ['8','0.05','8,192','200,000','90.8853%','0.48346'],
    ['<b>8</b>','<b>0.05</b>','<b>65,536</b>','<b>148,000</b>','<b>99.00265%</b>','<b>0.19901</b>'],
    ['10','0.05','65,536','100,000','0.004015%','10.40702'],
    ['10','0.0015625','65,536','57,000','99.00455%','0.51053'],
],[44,77,70,83,92,WIDTH-366],highlight=3))
section('What changed computationally')
add('<b>On-demand observations.</b> The synthetic data representation stores the truth, local operators and a fixed row-noise recipe. Only sampled observations are generated. Revisited rows receive the same noise, not a fresh noise draw. The recipe defines a different realization from earlier sequential stored-data generation; existing datasets are preserved.')
add('<b>Smaller batch workspace.</b> An equivalent product-state measurement backend reduces workspace from O(B × 4<super>n</super>) to O(B × 2<super>n</super>), while preserving the measurements and gradients. This avoids the roughly 461 TiB full ten-qubit observation table and makes ten-qubit computation feasible on an A100 40GB.')
add('<b>Prefetch and exact continuation.</b> CPU batch preparation overlaps GPU work. Native restarts retain the factor, momentum, sampling RNG and absolute schedule position. The stochastic-FRAMES update and TP/measurement formulation are unchanged.')
section('Limits of the ten-qubit results')
add('The 99.00455% ten-qubit result used noise 32 times smaller than σ = 0.05. At the original noise, the last inspected baseline-continuation checkpoint was only 0.018935% at 148,000 steps; it preceded cancellation and is not a terminal result. Batch and penalty pilots did not establish useful recovery within the tested budget. That campaign remains stopped.')

# 2. Full-sized convergence plot.
nextpage('landscape')
plotw=LW-2*MARGIN
story.append(Image(str(ROOT/'figures/n8_fidelity_iterations.png'),width=plotw,height=plotw*6.4/12.6))
add('<b>Reading the plot.</b> Batch 8,192 improved from 57.1945% at 30,000 steps to 90.8853% at 200,000. The fresh batch-65,536 trajectory reached 98.48565% at 100,000; its exact continuation first met the saved-checkpoint stopping criterion at 148,000. Checkpoints are every 1,000 steps. The small-batch trace available locally starts at 30,000 steps; its earlier interval is not reconstructed.','small')
add('<b>Same numerical settings:</b> rank 1, τ = 20, float64/complex128, product-state backend; channel, noise, initialization and sampling seeds all 0. The schedule clock is not reset by continuation. Each larger-batch iteration uses eight times as many row draws.','small')

# 3. Sample budget and TP plot, plus precise n8 accounting.
nextpage('landscape')
story.append(Image(str(ROOT/'figures/n8_sampling_and_tp.png'),width=plotw,height=plotw*6.4/12.6))
story.append(table(['Batch B','Iterations T','B / full table (%)','Total optimizer draws B × T','Draws / full table (%)'],[
    ['8,192','200,000','0.00000744218','1,638,400,000','1.48844'],
    ['65,536','148,000','0.0000595374','9,699,328,000','8.81154'],
],[90,100,160,240,plotw-590],highlight=1))
story.append(Spacer(1,7))
add('At equal draw budget the larger-batch configuration also has higher fidelity, but the iteration-dependent schedules have different values at those checkpoints. This comparison does not isolate a pure batch-variance effect. Draw counts exclude diagnostics and separate profiling runs.','small')

# 4. Exact schedules, all configuration choices, and update meaning.
nextpage('portrait')
add('Parameter choices','title')
add('Let k = 0, 1, ... index updates. A checkpoint at iteration t is after t updates; its last applied update uses k = t - 1. All selected runs have rank 1.','small')
story.append(table(['Parameter','Both plotted 8-qubit runs','Meaning'],[
    ['Step size γ<sub>k</sub>','10 / (k + 10)','Weight on the FW atom'],
    ['Smoothing β<sub>k</sub>','10<super>7</super> / (k + 1)<super>0.25</super>','TP penalty denominator'],
    ['Momentum weight ρ<sub>k</sub>','2 / (k + 4)<super>0.6</super>','Weight on the new gradient'],
],[112,205,WIDTH-317]))
story.append(Spacer(1,9))
add('The measurement-gradient estimate is <b>d<sub>k</sub> = (1 - ρ<sub>k</sub>)d<sub>k-1</sub> + ρ<sub>k</sub>ĝ<sub>k</sub></b>, where ĝ<sub>k</sub> comes from the current batch. The retained momentum weight is <b>1 - ρ<sub>k</sub></b>. At k = 0 the code assigns d<sub>0</sub> = ĝ<sub>0</sub> directly. The exact TP gradient is added after this averaging, divided by β<sub>k</sub>.')
add('With linear-minimization atom s<sub>k</sub>, the update is <b>u<sub>k+1</sub> = (1 - γ<sub>k</sub>)u<sub>k</sub> + γ<sub>k</sub>s<sub>k</sub></b>. Step size and ρ are capped at 1, but the caps do not alter these chosen formulas. <b>There is no smoothing cap.</b>')
add('The TP penalty is <b>||T(uu†) - I||<sub>F</sub><super>2</super> / (2β<sub>k</sub>)</b>. Raising the smoothing scale weakens TP enforcement throughout the schedule. Its decay strengthens the penalty over time. Smoothed objectives and gaps across different smoothing scales refer to different penalized objectives.')
section('Selected settings across sizes')
add('For every row: β<sub>k</sub> = β<sub>0</sub>/(k+1)<super>0.25</super> and ρ<sub>k</sub> = 2/(k+4)<super>0.6</super>. Batch sizes and iteration totals are on page 1.','small')
story.append(table(['Qubits','Noise σ','Step size γ<sub>k</sub>','Smoothing scale β<sub>0</sub>','Radius τ'],[
    ['4','0.05','2/(k+2)','100','10'],
    ['5','0.05','2/(k+2)','10,000','10'],
    ['8 (both)','0.05','10/(k+10)','10,000,000','20'],
    ['10','0.05','10/(k+10)','1,000,000,000','40'],
    ['10','0.0015625','10/(k+10)','1,000,000,000','40'],
],[65,90,125,145,WIDTH-425],highlight=2))
story.append(Spacer(1,8))
add('The four/five-qubit smoothing sweep tested scales 10, 100, 1,000 and 10,000, keeping its other settings fixed. Scale 100 was best among tested n = 4 settings on fidelity, sampled loss and TP. At n = 5, scale 10,000 had best recovery, with worse TP than scale 1,000.','small')
section('Actual schedule values for the 8-qubit runs')
story.append(table(['Applied update k','Step γ<sub>k</sub>','Smoothing β<sub>k</sub>','New-gradient ρ<sub>k</sub>'],[
    ['0','1','10,000,000','0.870551*'],
    ['99,999','0.0000999910','562,341.325','0.00199996'],
    ['147,999 (99% result)','0.0000675635','509,840.781','0.00158077'],
    ['199,999 (small batch)','0.0000499978','472,870.805','0.00131950'],
],[150,110,130,WIDTH-390],highlight=2))
story.append(Spacer(1,7))
add('*The first gradient is assigned directly. These are last-applied values; a saved checkpoint β may instead describe the next update. All 48,000 saved values of each schedule in the successful continuation were checked against these formulas.','small')

# 5. Full data sizes, timings, scientific interpretation and provenance.
nextpage('portrait')
ST['body']=ParagraphStyle('body_compact',parent=ST['body'],fontSize=8.8,leading=12.2,spaceAfter=7)
ST['small']=ParagraphStyle('small_compact',parent=ST['small'],fontSize=7.8,leading=10.7,spaceAfter=6)
add('Data budget and reproducibility','title')
add('The full measurement table has M = 24<super>n</super> possible scalar rows. The compact n = 4/5 datasets store observations. The n = 8/10 synthetic datasets represent the table virtually, storing no observation array. Targets are generated on demand from the truth and a fixed row-noise recipe.')
story.append(table(['Qubits','Full table M','Batch B','B / M (%)'],[
    ['4','331,776','32','0.00964506'],
    ['5','7,962,624','32','0.000401878'],
    ['8','110,075,314,176','8,192','0.00000744218'],
    ['8','110,075,314,176','65,536','0.0000595374'],
    ['10','63,403,380,965,376','65,536','0.000000103364'],
],[48,190,95,WIDTH-333],highlight=3))
story.append(Spacer(1,9))
add('Sampling is uniform with replacement; repeated rows retain the same noisy target. B × T / M counts draws relative to table size, <b>not distinct-row coverage</b>. The n = 4/5 runs each used 320,000 draws (96.4506% / 4.01878% of M). The n = 10 original-noise run used 6,553,600,000 draws (0.0103364% of M); the reduced-noise run used 3,735,552,000 (0.00589172%).')
section('Eight-qubit execution and timing')
add('The successful run used one A100 40GB at ruche-gpu15. Its two winning segments totaled <b>21.66 minutes of optimizer time</b>, <b>78.32 minutes of runner time</b> including CPU preparation and checkpointing, and <b>80 minutes 4 seconds of full batch-job elapsed time</b> including setup and the chunk comparison. These measure different scopes.')
add('CPU prefetch was enabled. Preparation chunk size was 10 for the small-batch continuation and the first 100,000 large-batch steps, then 1 for the successful continuation. Two 100-step runs per chunk choice produced bitwise-identical factors, momentum, sampled metrics and sample hashes. Chunk 1 reduced measured optimizer-plus-blocking-transfer time by 29.34%; this is not a whole-run speedup claim. CPU preparation still limited throughput.')
add('The validated implementation passed <b>484 tests plus 13 GPU checks</b>. GPU autotuning was disabled using <font size="8">XLA_FLAGS=--xla_gpu_autotune_level=0</font>. The n = 8 continuation changed no algorithm, objective or CLI default. All datasets and results are preserved; implementation changes remain uncommitted and unpushed.')
section('Metrics and interpretation')
add('Fidelity is <b>|c†u|<super>2</super> / (||c||<super>2</super> ||u||<super>2</super>)</b>, evaluated from complete factors. TP uses the complete factor too. Measurement loss and gap use a fixed independent sample of 512 rows, metric seed 12345, metric batch size 512. Truth supplies synthetic targets, diagnostics and stopping, not a privileged initialization or update direction.')
add('Channel, noise, initialization and sampling seeds are all 0. The successful run shares the original initial factor with the small-batch lineage and resumes complete state at 100,000 steps. The small-batch run resumes at 30,000. A small sampled loss does not certify convergence, and 99% factor fidelity does not imply exact trace preservation. Fixed absolute noise gives different signal-to-noise ratios across sizes.')
section('Evidence and scope')
add('Plots join saved scalar checkpoints with no fitting or smoothing. Source JSON traces, independent complete-factor verifications, command provenance, source/archive checksums and plotting code accompany the Markdown report in results/qpt/coauthor_summary_20260915. Successful comparison jobs: 1889115_0/1; target continuation: 1889210_0. Initial attempts on gpu12 failed during CUDA initialization before optimization.','small')
add('Earlier n = 4/5 values come from the recorded experiment handoff, with original report paths retained; those remote reports were not re-read for this document. No new experiments or remote access were used to prepare it. Robustness across seeds, optimal hyperparameters and a maximum feasible qubit count remain unestablished.','small')

doc.build(story)
print(OUT)
