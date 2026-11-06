import datetime
import hashlib
import json
from pathlib import Path
import textwrap
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT / 'validation_audit.json').read_text())
q, f = (audit['methods'][m] for m in ('printed-adafgd', 'frames'))
assert audit['status'] == 'all_24_archives_verified_against_remote_inventory_hashes'
assert q['unique_outcomes'] == 8 and f['unique_outcomes'] == 24
assert q['gate_passes'] == f['gate_passes'] == 0
checked = datetime.datetime.fromisoformat(audit['checked_utc']).astimezone(ZoneInfo('Europe/Paris'))
INK, MUTED, BLUE, AMBER = '#172b3a', '#4e6070', '#18546b', '#825017'
fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
fig.text(.085, .953, 'QPT status: runs finished; recovery gate not met', fontsize=16, weight='bold', color=INK, va='top')
fig.text(.085, .920, 'Checked '+checked.strftime('%-d %B %Y, %H:%M %Z'), fontsize=9, color=MUTED, va='top')
def para(text, y, size=10, color=INK, width=101):
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    fig.text(.085, y, '\n'.join(lines), fontsize=size, color=color, va='top', linespacing=1.3)
    return y - len(lines)*size/(11.69*72)*1.3 - .013
def head(text, y):
    fig.text(.085, y, text, fontsize=11.7, weight='bold', color=BLUE, va='top')
    return y-.029
y=head('What is complete', .882)
y=para('The 352-pair calibration is complete and audited. The separate 24-pair validation array '
       '2014377 finished on 5 October at 22:08 CEST. All 24 JSON/NPZ archives, checksums, frozen '
       'settings and source provenance now verify locally. No QPT jobs are currently queued or running.',y)
y=para('This is the corrected n = 2, full-128-row, noiseless rank-one comparison: 8 new channels, '
       'one initialization per channel and 3 FRAMES sampling streams. Validation used 10,000 row epochs '
       '(1.28 million nominal row accesses per method when completed), five times the calibration budget. '
       'These are not n = 4 or high-qubit results.',y,size=9.5,color=MUTED,width=106)
y=head('Final validation outcomes',y)
rows=[['Finished normally',f"{q['complete']} / 8",f"{f['complete']} / 24"],
      ['Numerical failures',f"{q['failed']} / 8",f"{f['failed']} / 24"],
      ['Fidelity ≥ 0.99',f"{q['component_pass_counts']['process_fidelity']} / 8 pass",f"{f['component_pass_counts']['process_fidelity']} / 24 pass"],
      ['Normalized TP error ≤ 0.001',f"{q['component_pass_counts']['normalized_tp_violation']} / 8 pass",f"{f['component_pass_counts']['normalized_tp_violation']} / 24 pass"],
      ['Full-design RMSE ≤ 1e-6',f"{q['component_pass_counts']['rmse']} / 8 pass",f"{f['component_pass_counts']['rmse']} / 24 pass"],
      ['Full recovery gate (all three)','0 / 8 pass','0 / 24 pass']]
height=.214
ax=fig.add_axes([.085,y-height,.83,height]);ax.axis('off')
t=ax.table(cellText=rows,colLabels=['Check','Quiroga (printed adaFGD)','FRAMES'],colWidths=[.46,.29,.25],bbox=[0,0,1,1],cellLoc='left',colLoc='left')
t.auto_set_font_size(False);t.set_fontsize(9.1)
for (r,c),cell in t.get_celld().items():
    cell.set_edgecolor('#d3dee5');cell.set_linewidth(.45);cell.PAD=.055
    cell.set_text_props(color=INK)
    if r==0:cell.set_facecolor('#edf4f7');cell.set_text_props(weight='bold',color=BLUE)
y-=height+.017
y=para('Quiroga repetitions were verified identical and counted once per channel; 24 FRAMES runs '
       'are nested in 8 independent channels. The 9 failed paired archives represent 3 Quiroga channel '
       'failures, not 9 independent failures. All failures are retained; no result is missing.',y,size=9,color=MUTED,width=111)
y=head('What this means',y)
y=para('FRAMES is numerically stable in these runs and has very high fidelity, but measurement fit remains '
       'the common blocker: even its best RMSE is 7.25e-6, above 1e-6. Its median across channel means '
       'is 2.88e-5; five runs also fail TP. High fidelity alone is not successful recovery.',y)
y=para('Quiroga has two separate problems here: numerical divergence on channels 100, 102 and 106, '
       'and poor reconstruction on the five channels that finish. Those five pass TP, but fidelity '
       'ranges only from 0.0216 to 0.2207. This is not merely a TP-gate issue.',y)
y=head('Where the project stands / next decision',y)
y=para('Execution and result preservation are working; the benchmark is not yet ready for a successful-recovery '
       'claim. The frozen choices were independently tuned: Quiroga fixed alpha = 0.5, step multiplier 0.01; '
       'FRAMES initial alpha = 5e-5, increasing with exponent 0.25. Both penalties landed at grid boundaries, '
       'and neither selection passed the calibration gate.',y,size=9.5,width=106)
y=para('Recommended next step: diagnose Quiroga stability and reconstruction, then refine tuning on calibration '
       'data before extending this comparison to larger qubit counts. Keep this validation result fixed; do not '
       'rerank on it. Any revised method/configuration needs fresh confirmation. No new jobs or parameter '
       'changes were made for this status check; the final held-out split remains untouched.',y,size=9.5,color=AMBER,width=106)
assert y>.083,y
fig.text(.085,.061,'Source c7a8c9789b78 • selection fa9b5d5287f26029 • independent aggregation cross-check agrees',fontsize=7.5,color=MUTED)
fig.text(.085,.046,'Evidence: '+str(ROOT/'validation_audit.json'),fontsize=7.0,color=MUTED)
out=ROOT/'QPT_STATUS_20261006.pdf'
assert not out.exists()
with PdfPages(out,metadata={'Title':'QPT experiment status — 6 October 2026'}) as pdf:pdf.savefig(fig)
plt.close(fig)
print(json.dumps({'pdf':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
