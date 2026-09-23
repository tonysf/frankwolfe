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
CAMPAIGN=ROOT.parent
REPO=CAMPAIGN.parents[2]
OUT=REPO/'output/pdf/qpt_n8_followups_20260923.pdf'
DATA=json.loads((CAMPAIGN/'comparison_data.json').read_text())
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
    canvas.drawRightString(PW-MARGIN,PH-22,'Eight-qubit follow-up analysis | 23 September 2026')
    canvas.line(MARGIN,30,PW-MARGIN,30)
    canvas.drawString(MARGIN,19,'Saved scalar evidence | one dataset / seed | proposal only: no job submitted')
    canvas.drawRightString(PW-MARGIN,19,str(doc.page))


ANALYSIS=json.loads((ROOT/'late_analysis.json').read_text())
PROPOSAL=json.loads((ROOT/'proposed_experiment.json').read_text())
def newpage(title):story.append(PageBreak());add(title,'title')
doc=BaseDocTemplate(str(OUT),pagesize=(PW,PH),leftMargin=MARGIN,rightMargin=MARGIN,
    topMargin=44,bottomMargin=42,title='Eight-qubit QPT: late convergence and proposed continuation',author='QPT follow-up analysis',pageCompression=1)
doc.addPageTemplates([PageTemplate(id='landscape',pagesize=(PW,PH),frames=[Frame(MARGIN,42,WIDTH,PH-86,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=chrome)])
story=[]
add('The best run is still improving','title')
add('Recommendation: continue the best run unchanged from 148,000 to 200,000 total iterations. No job has been submitted.','callout')
story.append(Image(str(CAMPAIGN/'comparison_figures/late_comparison.png'),width=WIDTH*.94,height=WIDTH*.94*5.6/13))
best=ANALYSIS['observed'][0]
story.append(table(['Best case: a=0.5, r=0.6','100k-120k','120k-138k','138k-148k'],[
 ['Fidelity gain (percentage points)',*[f"{w['gain_pp']:.4f}" for w in best['windows']]],
 ['Gain per 10,000 updates (pp)',*[f"{w['gain_pp_per_10000']:.4f}" for w in best['windows']]],
],[300,155,155,WIDTH-610],pad=4))
story.append(Spacer(1,7))
add('Fidelity increased at every saved checkpoint after 100k, reaching 98.7689%; TP fell 17.2% over 100k-148k. Gains are slowing, but this trajectory is not flat. The scale effects interact: a=0.5 wins at r=0.6, while a=1 wins at r=2. No single scale change explains all four curves.','small')

newpage('One targeted test, with the same exponents')
story.append(table(['Choice','Proposed setting / interpretation'],[
 ['Scientific question','Can more iterations alone close the remaining 0.2311 percentage-point gap to 99%?'],
 ['Starting state','Exact restart of a=0.5, r=0.6 at step 148,000; retain factor, momentum, RNG state and absolute schedule index'],
 ['Total / extra budget','200,000 total / 52,000 extra updates; 3,407,872,000 additional optimizer row draws'],
 ['Step / smoothing / momentum','γ=min(1, 0.5/(k+10)<super>3/4</super>); β=10<super>7</super>/(k+1)<super>1/4</super>; ρ=min(1, 0.6/(k+4)<super>1/2</super>)'],
 ['Other settings','8 qubits; σ=0.05; B=65,536; rank 1; τ=20; same data, fixed noise and validated optimized source'],
 ['Resources / cost','One A100, 2 CPUs, 12G RAM, 30-minute cap; approximately 11.4 min runner / 6.9 min optimizer from measured scaling'],
 ['Outcome / stopping','Run the full fixed budget. Verify endpoint F≥99%; also report first crossing and whether all ten final checkpoints meet 99%. No automatic extension.'],
],[150,WIDTH-150],pad=5))
story.append(Spacer(1,9))
add('Why this budget is plausible, without treating extrapolation as a result','section')
fits=[x for x in ANALYSIS['scenarios'] if x['start']==100000]
labels={'linear_fidelity':'Linear fidelity','exponential_infidelity':'Exponential infidelity decay','power_infidelity':'Power-law infidelity decay'}
story.append(table(['Descriptive fit to steps 100k-148k','Projected 99% crossing','Projected F at 200k'],[
 [labels[x['model']],f"{x['estimated_99_step']/1000:.1f}k",f"{100*x['predicted_fidelity']:.3f}%"] for x in fits],[310,230,WIDTH-540],pad=4))
story.append(Spacer(1,6))
add('Earlier linear extrapolations overpredicted the held-out 148k fidelity by 0.57-1.71 pp. Exponential-error fits overpredicted by 0.15-0.30 pp; power fits had errors -0.047 and +0.014 pp in two retrospective checks. Recent-window power fits suggest roughly 170k-175k for a crossing. These are model-dependent scenarios, not confidence intervals; a fidelity plateau below 99% remains possible.','small')
add('Changing β or other constants now would combine an iteration-budget test with retuning. This continuation keeps the schedule intact. Timing excludes queueing and separate setup/verification; CPU preparation overlaps GPU work. Report TP and sampled loss separately from the fidelity target.','small')

newpage('Draft update for coauthors')
add('Subject: 8-qubit QPT update with the paper’s exponents','section')
for paragraph in (ROOT/'coauthor_summary.md').read_text().split('\n\n')[1:]:
    add(paragraph.strip())
add('Records and scope','section')
add('The full verified comparison remains in qpt_n8_scale_tuning_20260923.pdf. This addendum uses its saved scalar traces; it introduces no new QPT results. Analysis details and model definitions are in followups_20260923/late_analysis.json; exact continuation parameters, command arguments, source/data/restart hashes and validation criteria are in proposed_experiment.json. The editable message draft is coauthor_summary.md.','small')
add('The proposal is unsubmitted and the message is unsent. Automatic campaign follow-up remains paused. Any later run must use a new guarded output directory, preserve the parent archive, and validate the resumed state and complete final factors within the allocation.','small')
OUT.parent.mkdir(parents=True,exist_ok=True)
doc.build(story)
print(OUT)

