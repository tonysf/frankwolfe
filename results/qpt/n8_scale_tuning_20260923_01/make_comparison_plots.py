"""Plot collected scalar evidence; requires all four cases to be verified."""
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
from matplotlib.ticker import MaxNLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'comparison_figures'
def read(path): return json.loads(path.read_text())
PLAN = read(ROOT/'plan.json')
MANIFEST = read(ROOT/'manifest.json')
assert MANIFEST['completed_count'] == 4
OUT.mkdir(exist_ok=True)
OLD, REQUESTED, NEW = '#176B9A', '#78838D', '#BF651D'
INK, MUTED = '#253343', '#617183'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B7C3CE',
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'grid.color':'#E0E6EB','grid.linewidth':.7,'pdf.fonttype':42})

def join(segments):
    points = {}
    for segment in segments:
        for point in segment['trajectory']:
            if point['step'] in points:
                for key in ('fidelity','measurement_loss','tp_violation'):
                    assert np.isclose(point[key],points[point['step']][key],rtol=1e-10,atol=1e-12)
            points[point['step']] = point
    return [points[t] for t in sorted(points)]

RUNS=[]
for case in PLAN['cases']:
    root=ROOT/case['name']; summary=read(root/'summary.json'); validation=read(root/'local_validation.json')
    assert validation['status']=='verified_and_backed_up'
    old=join(read(root/'baseline_traces.json')['segments'])
    requested=read(root/'requested_baseline_trace.json')
    assert old[-1]==validation['baseline']
    assert summary['trajectory'][-1]==validation['new']
    assert requested['trajectory'][-1]==validation['requested_exponent_baseline']
    assert all(np.isfinite(v) for trace in (old,requested['trajectory'],summary['trajectory']) for row in trace for v in row.values())
    RUNS.append(dict(case=case,new=summary['trajectory'],old=old,requested=requested['trajectory'],
        metadata=summary['metadata'],requested_metadata=requested['metadata'],validation=validation))

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=190)
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

legend=[Line2D([0],[0],color=OLD,lw=2.2,label='Old successful exponents'),
        Line2D([0],[0],color=REQUESTED,lw=2.2,label='Requested exponents: a=10, r=2'),
        Line2D([0],[0],color=NEW,lw=2.2,label='Requested exponents: retuned scales')]
for indices,name in [([0,1],'step_half'),([2,3],'step_one')]:
    fig,axes=plt.subplots(2,3,figsize=(13,7.35))
    fig.subplots_adjust(left=.078,right=.976,bottom=.09,top=.86,hspace=.65,wspace=.33)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.53,.997),ncol=3,frameon=False,fontsize=9.2)
    for row,index in enumerate(indices):
        run=RUNS[index];case=run['case']
        fig.text(.078,.94 if row==0 else .458,
            f"Step scale a={case['step_scale']:g} | momentum scale r={case['rho_scale']:g} | β scale 10⁷ | batch 65,536",
            fontsize=12,weight='bold')
        for col,key in enumerate(('fidelity','measurement_loss','tp_violation')):
            ax=axes[row,col]
            for trace,color in [(run['old'],OLD),(run['requested'],REQUESTED),(run['new'],NEW)]:
                x=[p['step']/1000 for p in trace];y=[p[key]*(100 if key=='fidelity' else 1) for p in trace]
                ax.plot(x,y,color=color,lw=1.9,ls='-')
                ax.plot(x[-1],y[-1],'o',color=color,ms=3.5)
            ax.set_xlabel('Completed iterations (thousands)',fontsize=10)
            ax.set_xlim(0,151);ax.xaxis.set_major_locator(MaxNLocator(5));ax.grid(True);ax.set_axisbelow(True)
            if key=='fidelity':
                ax.set(title='Fidelity',ylabel='Factor fidelity (%)',ylim=(0,103));ax.axhline(99,color=MUTED,lw=.85,ls=':')
            elif key=='measurement_loss':
                ax.set(title='Sampled measurement loss',ylabel='Least-squares loss')
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
                ax.yaxis.set_major_locator(MaxNLocator(4))
            else:
                ax.set(title='TP violation (log scale)',ylabel='Frobenius residual',yscale='log')
                ax.yaxis.set_minor_formatter(NullFormatter())
    save(fig,name)

colors=[NEW,'#9C496E','#38836C','#8A6CB3']
fig,axes=plt.subplots(1,2,figsize=(13,5.6))
fig.subplots_adjust(left=.074,right=.978,bottom=.14,top=.77,wspace=.26)
traces=[('Old successful',RUNS[0]['old'],OLD),('Requested baseline',RUNS[0]['requested'],REQUESTED)]
traces += [(f"a={r['case']['step_scale']:g}, r={r['case']['rho_scale']:g}",r['new'],c) for r,c in zip(RUNS,colors)]
for label,trace,color in traces:
    for ax,key in zip(axes,('fidelity','tp_violation')):
        ax.plot([p['step']/1000 for p in trace],[p[key]*(100 if key=='fidelity' else 1) for p in trace],color=color,lw=2,label=label)
for ax in axes:
    ax.set_xlim(100,148);ax.set_xlabel('Completed iterations (thousands)');ax.grid(True);ax.set_axisbelow(True)
axes[0].set(title='Late fidelity (zoomed vertical axis)',ylabel='Factor fidelity (%)',ylim=(88,100))
axes[0].axhline(99,color=MUTED,lw=.85,ls=':')
axes[1].set(title='Late TP residual (linear scale)',ylabel='Frobenius residual')
late=[p['tp_violation'] for _,trace,_ in traces for p in trace if p['step']>=100000]
axes[1].set_ylim(max(0,min(late)*.85),max(late)*1.07)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.98),ncol=3,frameon=False,fontsize=10)
save(fig,'late_comparison')

selection=read(ROOT/'preflight_retry1/selection.json')
fig,axes=plt.subplots(1,2,figsize=(13,3.0))
fig.subplots_adjust(left=.07,right=.97,bottom=.15,top=.86,wspace=.28)
host=selection['host_timings']
for ax,title,values,unit in [
    (axes[0],'Host preparation per batch',[host['legacy_preparation']['median_seconds']*1000,host['fast_preparation']['median_seconds']*1000],'Milliseconds'),
    (axes[1],'Complete 1,000-step runner',[selection['runner_medians_seconds']['original'],selection['runner_medians_seconds']['fast']],'Seconds')]:
    ax.bar(['Original','Optimized'],values,color=[OLD,NEW],width=.5)
    ax.set(title=title,ylabel=unit,ylim=(0,max(values)*1.25));ax.grid(True,axis='y');ax.set_axisbelow(True)
    for j,value in enumerate(values):ax.text(j,value+max(values)*.025,f'{value:.2f}',ha='center',fontsize=12)
save(fig,'preparation_speedup')

hashes={str((ROOT/run['case']['name']/name).relative_to(ROOT)):hashlib.sha256((ROOT/run['case']['name']/name).read_bytes()).hexdigest()
    for run in RUNS for name in ('summary.json','baseline_traces.json','requested_baseline_trace.json','local_validation.json')}
(ROOT/'comparison_data.json').write_text(json.dumps(dict(runs=RUNS,plan=PLAN,preflight=selection,input_sha256=hashes),indent=2)+'\n')
print('Created four figures and verified comparison_data.json.')
