"""Plot verified scalar evidence only; no optimization or remote access."""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/qpt-coauthor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatterSciNotation, NullFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'comparison_figures'
OUT.mkdir(exist_ok=True)
PLAN = json.loads((ROOT / 'plan.json').read_text())
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
assert MANIFEST['completed_count'] == 12
OLD, NEW, INK, MUTED = '#176B9A', '#BF651D', '#253343', '#617183'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#B7C3CE',
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'grid.color': '#E0E6EB', 'grid.linewidth': .7, 'pdf.fonttype': 42})


def read(path):
    return json.loads(path.read_text())


def join(segments):
    rows = {}
    for segment in segments:
        for row in segment['trajectory']:
            if row['step'] in rows:
                for key in ('fidelity', 'measurement_loss', 'tp_violation'):
                    assert np.isclose(row[key], rows[row['step']][key], rtol=1e-10, atol=1e-12)
            rows[row['step']] = row
    return [rows[t] for t in sorted(rows)]


DATA = []
for case in PLAN['cases']:
    directory = ROOT / case['name']
    new = read(directory / 'summary.json')
    baseline = read(directory / 'baseline_traces.json')
    validation = read(directory / 'local_validation.json')
    assert validation['status'] == 'verified_and_backed_up'
    old = join(baseline['segments'])
    assert old[0]['step'] == new['trajectory'][0]['step'] == 0
    assert old[-1]['step'] == new['trajectory'][-1]['step'] == case['steps']
    assert old[-1] == validation['baseline'] and new['trajectory'][-1] == validation['new']
    for trace in (old, new['trajectory']):
        assert all(np.isfinite(value) for row in trace for value in row.values())
    DATA.append(dict(case=case, old=old, new=new['trajectory'], validation=validation,
                     metadata=new['metadata'], old_segments=[{k:s[k] for k in ('path','archive_sha256','metadata')} for s in baseline['segments']]))

BY_NAME = {item['case']['name']: item for item in DATA}
legend = [Line2D([0],[0],color=OLD,lw=2.3,label='Old: step 1, smoothing 1/4, momentum 0.6'),
          Line2D([0],[0],color=NEW,lw=2.3,ls='-',label='New: step 3/4, smoothing 1/4, momentum 1/2')]


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=190)
    fig.savefig(OUT / (name + '.pdf'))
    plt.close(fig)


def grid(names, name):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.35))
    fig.subplots_adjust(left=.075,right=.976,bottom=.09,top=.86,hspace=.65,wspace=.31)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.53,.998),ncol=2,frameon=False,fontsize=10.4)
    for row, run_name in enumerate(names):
        item=BY_NAME[run_name]; c=item['case']
        if c['n_qubits'] < 8:
            label=f"{c['n_qubits']} qubits | smoothing scale {c['smoothing_scale']:,} | batch {c['batch_size']:,} | σ = {c['noise_std']}"
        else:
            noise='REDUCED-NOISE CONTROL' if c['noise_std'] < .05 else 'usual noise'
            label=f"{c['n_qubits']} qubits | batch {c['batch_size']:,} | σ = {c['noise_std']} ({noise})"
        fig.text(.075, .902 if row == 0 else .458, label,fontsize=12,weight='bold')
        for col,key in enumerate(('fidelity','measurement_loss','tp_violation')):
            ax=axes[row,col]
            for schedule,color,style in [('old',OLD,'-'),('new',NEW,'-')]:
                trace=item[schedule]
                x=np.array([p['step'] for p in trace])/1000
                y=np.array([p[key] for p in trace])*(100 if key=='fidelity' else 1)
                ax.plot(x,y,color=color,lw=2.0,ls=style)
                ax.plot(x[-1],y[-1],'o',color=color,ms=4)
            ax.set_xlabel('Completed iterations (thousands)',fontsize=10)
            ax.set_xlim(0,c['steps']/1000*1.025)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(True,which='major'); ax.set_axisbelow(True)
            if key=='fidelity':
                ax.set_ylabel('Factor fidelity (%)')
                if run_name=='n10_b65536_fixed':
                    ax.set_yscale('log')
                    ax.set_title('Fidelity - logarithmic scale',fontsize=11)
                else:
                    largest=max(p[key] for sch in ('old','new') for p in item[sch])*100
                    ax.set_ylim(0,103 if largest>90 else largest*1.15)
                    ax.set_title('Fidelity',fontsize=11)
                    if largest>90:
                        ax.axhline(99,color=MUTED,lw=.85,ls=':')
            else:
                ax.set_yscale('log')
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_title('Sampled loss (log scale)' if key=='measurement_loss' else 'TP violation (log scale)',fontsize=11)
                ax.set_ylabel('Least-squares loss' if key=='measurement_loss' else 'Frobenius residual')
                if key=='measurement_loss':
                    ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=9))
                    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10,labelOnlyBase=False,minor_thresholds=(float('inf'),float('inf'))))
                loss_values=[p['measurement_loss'] for sch in ('old','new') for p in item[sch]]
                if key=='measurement_loss' and c['n_qubits']>=8 and max(loss_values)/min(loss_values)<4:
                    # Narrow loss ranges are easier to read as scaled linear axes.
                    ax.set_yscale('linear')
                    ax.set_title('Sampled measurement loss',fontsize=11)
                    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
                    ax.yaxis.set_major_locator(MaxNLocator(4))
    save(fig,name)


GROUPS=[(['n4_beta10','n4_beta100'],'n4_strong'),
        (['n4_beta1000','n4_beta10000'],'n4_weak'),
        (['n5_beta10','n5_beta100'],'n5_strong'),
        (['n5_beta1000','n5_beta10000'],'n5_weak'),
        (['n8_b8192_fixed','n8_b65536_fixed'],'n8'),
        (['n10_b65536_fixed','n10_b65536_relative'],'n10')]
for names,name in GROUPS:
    grid(names,name)

# Compare both n8 batch sizes on exactly matched numbers of sampled row draws.
fig,axes=plt.subplots(1,2,figsize=(13,5.35))
fig.subplots_adjust(left=.07,right=.977,bottom=.16,top=.77,wspace=.28)
for ax,schedule in zip(axes,('old','new')):
    for run_name,color in [('n8_b8192_fixed',OLD),('n8_b65536_fixed',NEW)]:
        item=BY_NAME[run_name]; c=item['case']; trace=item[schedule]
        ax.plot([p['step']*c['batch_size']/1e9 for p in trace],[100*p['fidelity'] for p in trace],
                color=color,lw=2.2,label=f"Batch {c['batch_size']:,}")
    budget=200000*8192/1e9
    ax.axvline(budget,ls=':',color=MUTED,lw=1)
    for run_name,step,color in [('n8_b8192_fixed',200000,OLD),('n8_b65536_fixed',25000,NEW)]:
        point=next(p for p in BY_NAME[run_name][schedule] if p['step']==step)
        ax.scatter([budget],[point['fidelity']*100],color=color,zorder=5,s=35)
    ax.axhline(99,color=MUTED,ls=':',lw=.85)
    ax.set(title='Old exponents' if schedule=='old' else 'Requested exponents',
           xlabel='Cumulative optimizer row draws (billions)',ylabel='Factor fidelity (%)',
           xlim=(0,10),ylim=(0,103))
    ax.grid(True); ax.set_axisbelow(True); ax.legend(loc='lower right',frameon=False)
fig.text(.07,.915,'8 qubits: recovery versus sampled-row budget',fontsize=17,weight='bold')
fig.text(.07,.846,'σ = 0.05 | dotted vertical line: 1.6384 billion draws | repeated rows count again',fontsize=11,color=MUTED)
save(fig,'n8_draw_budget')

csv_rows=[]
for item in DATA:
    c=item['case']; v=item['validation']
    row={'case':c['name'],'n_qubits':c['n_qubits'],'noise_std':c['noise_std'],'batch_size':c['batch_size'],
         'steps':c['steps'],'smoothing_scale':c['smoothing_scale']}
    for label in ('old','new'):
        last=item[label][-1]
        for key in ('fidelity','measurement_loss','tp_violation'):
            row[label+'_'+key]=last[key]
    csv_rows.append(row)
with (ROOT/'comparison_endpoints.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
report={'campaign':PLAN['campaign'],'scope':'12 verified matched-budget comparisons; one dataset and seed per condition',
        'exponents_old':{'step':1,'smoothing':.25,'rho':.6},'exponents_new':PLAN['exponents'],
        'runs':DATA,'input_sha256':{str((ROOT/item['case']['name']/f).relative_to(ROOT)):
        hashlib.sha256((ROOT/item['case']['name']/f).read_bytes()).hexdigest()
        for item in DATA for f in ('summary.json','baseline_traces.json','local_validation.json')}}
(ROOT/'comparison_data.json').write_text(json.dumps(report,indent=2)+'\n')
print('Created seven scientific figures, endpoint CSV and full comparison scalar data.')
