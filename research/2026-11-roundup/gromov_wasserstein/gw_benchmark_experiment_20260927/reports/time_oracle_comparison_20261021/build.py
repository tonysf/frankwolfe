"""Measured active time and accepted-update pair contributions for GW runs."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
DATASETS=[('bmmc6224','BMMC · 6,224 cells'),('snare_full','SNARE-seq · 10,000 cells')]
EPS=[1e-3,3e-4,1e-4,3e-5]
EC=['#b88a00','#009e73','#d55e00','#222222']
ET=['10⁻³','3×10⁻⁴','10⁻⁴','3×10⁻⁵']
FC='#7b3294'
CC='#cc79a7'
PC='#0072b2'


def load(path):return json.loads(path.read_text())
def losses(r):return np.array([p['rounded_loss'] for p in r['trace']])
def iterations(r):return np.array([p['iteration'] for p in r['trace']])
def times(r):return np.array([p['elapsed_seconds'] for p in r['trace']])
def pair_work(r):
    return iterations(r)*(r['config']['pairs'] if r['method']=='frames' else r['dimensions']['n']*r['dimensions']['m'])


def verify(rows):
    for dataset,_ in DATASETS:
        group=[r for r in rows if r['run_id'].startswith(dataset+'/')]
        identity=group[0]
        for r in group:
            assert r['array_files_sha256']==identity['array_files_sha256']
            assert r['manifest_sha256']==identity['manifest_sha256']
            assert r['config']['dtype']=='float64'
            assert iterations(r)[0]==0 and (np.diff(iterations(r))>0).all()
            assert np.isfinite(losses(r)).all()
            assert max(p['rounded_residuals']['both_l1'] for p in r['trace'])<1e-12
            if r['status']=='completed':
                assert iterations(r)[-1]==r['completed_updates']
                assert losses(r)[-1]==r['final_loss']
            if 'elapsed_seconds' in r['trace'][0]:
                assert (times(r)>0).all() and (np.diff(times(r))>0).all()
                assert times(r)[-1]<=r['elapsed_seconds']
            if r['method']=='frames':
                assert r['counters']['stochastic_gradient_calls']==r['completed_updates']
                assert r['counters']['pair_samples']==r['completed_updates']*r['config']['pairs']
            else:assert r['counters']['full_gradient_calls']==r['completed_updates']


def figure(title,subtitle):
    fig,axes=plt.subplots(1,2,figsize=(14,7.7))
    fig.subplots_adjust(left=.075,right=.98,top=.68,bottom=.235,wspace=.22)
    fig.suptitle(title,x=.075,y=.97,ha='left',fontsize=18,fontweight='bold')
    fig.text(.075,.925,subtitle,fontsize=10,color='#444444')
    for ax,(_,name) in zip(axes,DATASETS):
        ax.set_title(name,fontsize=13)
        ax.set_ylabel('Feasible unregularized GW loss')
        ax.grid(True,color='#e5e7eb',linewidth=.7)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:.4f}'))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    return fig,axes


def legend(fig,axes):
    found={}
    for ax in axes:
        handles,labels=ax.get_legend_handles_labels()
        for h,l in zip(handles,labels):
            if l not in found:found[l]=h
    fig.legend(found.values(),found.keys(),loc='upper left',bbox_to_anchor=(.07,.89),ncol=2,
               frameon=False,fontsize=9.4,handlelength=2.5,columnspacing=2)


def notes(fig,texts):
    for y,text in zip((.16,.126,.092,.058,.025),texts):fig.text(.075,y,text,fontsize=9,color='#444444')


def frames(ax,rs,axis):
    if axis=='time':
        lo=max(times(r)[0] for r in rs);hi=min(times(r)[-1] for r in rs)
        grid=np.unique(np.concatenate(([lo,hi],*[times(r)[(times(r)>=lo)&(times(r)<=hi)] for r in rs])))
        values=np.array([np.interp(grid,times(r),losses(r)) for r in rs])
        for r in rs:
            ax.plot(times(r),losses(r),color=FC,alpha=.25,linewidth=.8)
            ax.scatter(times(r)[-1],losses(r)[-1],color=FC,s=20,marker='x',zorder=5)
    else:
        grid=pair_work(rs[0])
        assert all(np.array_equal(pair_work(r),grid) for r in rs)
        values=np.array([losses(r) for r in rs])
        # Zero work precedes the first gradient. Do not draw a fictitious
        # transition across budgets smaller than a full gradient evaluation.
        grid=grid[1:];values=values[:,1:]
    ax.fill_between(grid,values.min(axis=0),values.max(axis=0),color=FC,alpha=.16)
    ax.plot(grid,np.median(values,axis=0),color=FC,linewidth=2.1,
            label='FRAMES, 1,024 pairs, A100: median and range of 3 seeds')


def render_time(replay,sweep,cpu,pdf):
    fig,axes=figure('GW loss versus measured wall-clock time',
       'FP64 · One A100 per GPU run; CPU run explicitly marked · Curves use recorded timestamps · Logarithmic time axis')
    for ax,(ds,_) in zip(axes,DATASETS):
        rr=[r for r in replay if r['run_id'].startswith(ds+'/')]
        fr=[r for r in rr if r['method']=='frames'];pot=next(r for r in rr if r['method']=='pot-full')
        frames(ax,fr,'time')
        ax.plot(times(pot),losses(pot),color=PC,linewidth=1.8,
                label='Full-gradient POT, ε=10⁻³, product start: 2,000 updates')
        if ds=='bmmc6224':
            ax.plot(times(cpu),losses(cpu),color=CC,linestyle='-.',linewidth=1.8,
                    label='FRAMES, 4,096 pairs, CPU: seed 811')
        completed=[r for r in sweep if r['run_id'].startswith(ds+'/') and r['status']=='completed']
        for e,color,et in zip(EPS[:3],EC[:3],ET[:3]):
            group=[r for r in completed if r['config']['epsilon']==e]
            best=min(group,key=lambda r:r['final_loss'])
            ax.scatter([r['elapsed_seconds'] for r in group],[r['final_loss'] for r in group],
                       color=color,marker='D',s=22,alpha=.5,zorder=5)
            ax.scatter(best['elapsed_seconds'],best['final_loss'],color=color,marker='D',s=54,
                       edgecolors='white',linewidth=.5,zorder=6,label=f'POT sweep, ε={et}: completed-run endpoints')
        best=min(completed,key=lambda r:r['final_loss'])
        ax.annotate(f'Best POT: {best["elapsed_seconds"]/3600:.2f} h',
                    xy=(best['elapsed_seconds'],best['final_loss']),xytext=(-7,14),
                    textcoords='offset points',ha='right',fontsize=9,color='#b34c00')
        ax.set_xscale('log')
        ax.set_xlim(.5,110000)
        ax.set_xticks([1,10,60,600,3600,36000,86400],['1 s','10 s','1 min','10 min','1 h','10 h','24 h'])
        ax.set_xlabel('Measured active wall-clock time (log scale)')
    legend(fig,axes)
    notes(fig,[
      'Lines: timed replay (POT 2,000 updates; FRAMES 20,000). Diamonds: all completed starts in the entropy sweep; larger = lowest loss at that ε.',
      'The sweep saved total runtime but no loss timestamps: diamonds are final results only, not reconstructed trajectories. Unfinished sweep runs omitted here.',
      'Time includes data loading, optimization, loss diagnostics and checkpoint work; excludes queueing, prefactor tuning and runtime initialization.',
      'FRAMES time median/range interpolate only within shared observed time support. CPU and A100 timings are different hardware comparisons.',
      'POT sweep starts vary; FRAMES starts from the product coupling. The 2,000-update POT replay has no early stopping; the sweep does.',
    ])
    fig.savefig(OUT/'loss_vs_time.png',dpi=180,facecolor='white');pdf.savefig(fig);plt.close(fig)


def render_oracle(replay,sweep,cpu,pdf):
    fig,axes=figure('GW loss versus training oracle work',
        'Pair contributions at accepted updates · FRAMES: actual pair draws · Full-gradient POT: equivalent complete weighted sums')
    for ax,(ds,_) in zip(axes,DATASETS):
        rr=[r for r in replay if r['run_id'].startswith(ds+'/')]
        fr=[r for r in rr if r['method']=='frames'];baseline=next(r for r in rr if r['method']=='pot-full')
        frames(ax,fr,'oracle')
        ax.plot(pair_work(baseline)[1:],losses(baseline)[1:],color=PC,linewidth=1.5,alpha=.8,
                label='Full-gradient POT, ε=10⁻³, product start: 2,000 updates')
        if ds=='bmmc6224':
            ax.plot(pair_work(cpu)[1:],losses(cpu)[1:],color=CC,linewidth=1.8,linestyle='-.',
                    label='FRAMES, 4,096 pairs, CPU: seed 811')
        runs=[r for r in sweep if r['run_id'].startswith(ds+'/')]
        for e,color,et in zip(EPS,EC,ET):
            group=[r for r in runs if r['config']['epsilon']==e]
            selected=min(group,key=lambda r:losses(r)[-1])
            for r in group:ax.plot(pair_work(r)[1:],losses(r)[1:],color=color,linewidth=.7,alpha=.20)
            partial=selected['status']!='completed'
            label=f'POT sweep, ε={et}: '+('unfinished product start' if partial else 'best completed start')
            ax.plot(pair_work(selected)[1:],losses(selected)[1:],color=color,linewidth=1.9,
                    linestyle='--' if partial else '-',label=label)
            ax.scatter(pair_work(selected)[-1],losses(selected)[-1],color=color,
                       s=42 if partial else 25,marker='X' if partial else 'o',zorder=5)
        ax.set_xscale('log')
        ax.set_xlim(800,3e11)
        ax.set_xticks([1e3,1e5,1e7,1e9,1e11],['10³','10⁵','10⁷','10⁹','10¹¹'])
        ax.set_xlabel('Training pair contributions (log scale)')
    legend(fig,axes)
    notes(fig,[
       'After k accepted updates: FRAMES counts qk sampled pairs; full-gradient POT counts nmk weighted-pair equivalents (n × m coupling).',
       'POT evaluates the full sum using matrix products; nm equivalents are not nm literal stochastic-oracle calls. This axis is not FLOPs or runtime.',
       'Loss diagnostics and interrupted/retried updates are excluded. Sinkhorn iterations reuse the gradient and add no gradient calls.',
       'Lines connect diagnostics; zero-work initialization is omitted. Faint POT lines: other starts. Unfinished endpoints: BMMC k=75; SNARE k=50.',
       'Counting one batched gradient evaluation as one call instead gives k calls for either method, reproducing the iteration axis.',
    ])
    fig.savefig(OUT/'loss_vs_oracle_work.png',dpi=180,facecolor='white');pdf.savefig(fig);plt.close(fig)


def main():
    replay_path=OUT/'input/timing_replay.json'
    sweep_path=ROOT/'reports/entropy_sweep_progress_20261021/input/ruche_traces.json'
    replay=load(replay_path)['runs'];sweep=load(sweep_path)['runs']
    cpu_path=ROOT/'experiments/bmmc_cpu_batch4096_20261018/runs/frames_q4096_seed811/metadata.json'
    cpu=load(cpu_path)
    cpu.update(run_id='bmmc6224/frames_q4096_cpu_seed811',source=str(cpu_path),
               source_sha256=hashlib.sha256(cpu_path.read_bytes()).hexdigest())
    all_rows=replay+sweep+[cpu]
    verify(all_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42})
    with PdfPages(OUT/'time_and_oracle_plots.pdf') as pdf:
        render_time(replay,sweep,cpu,pdf);render_oracle(replay,sweep,cpu,pdf)
    fields=['run_id','method','status','record_type','iteration','elapsed_seconds','accepted_pair_contributions','feasible_gw_loss']
    with (OUT/'curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in all_rows:
            for p,w in zip(r['trace'],pair_work(r)):
                writer.writerow(dict(run_id=r['run_id'],method=r['method'],status=r['status'],record_type='diagnostic',
                   iteration=p['iteration'],elapsed_seconds=p.get('elapsed_seconds',''),accepted_pair_contributions=int(w),feasible_gw_loss=p['rounded_loss']))
            if r in sweep and r['status']=='completed':
                writer.writerow(dict(run_id=r['run_id'],method=r['method'],status=r['status'],record_type='timed_final_endpoint',
                   iteration=r['completed_updates'],elapsed_seconds=r['elapsed_seconds'],accepted_pair_contributions=int(pair_work(r)[-1]),feasible_gw_loss=r['final_loss']))
    summary={'metric':'rounded_loss: feasible unregularized GW loss',
      'oracle_axis':'Accepted-update pair contributions: q*k for FRAMES, n*m*k equivalents for full-gradient POT; not literal full-gradient call count or FLOPs.',
      'oracle_exclusions':['diagnostics','interrupted/retried updates'],
      'time_scope':'Recorded active runtime, including loading, diagnostics and checkpoint work; excludes queueing, tuning and runtime initialization.',
      'sweep_time_limitation':'Only completed total runtimes; no timestamped loss trajectory. Unfinished sweep runs excluded from time plot.',
      'runs':[dict(run_id=r['run_id'],source=r['source'],source_sha256=r['source_sha256'],config=r['config'],
                completed_updates=r['completed_updates'],last_diagnostic_iteration=int(iterations(r)[-1]),last_loss=float(losses(r)[-1]),
                active_seconds=r['elapsed_seconds'],counters=r['counters'],last_pair_contributions=int(pair_work(r)[-1])) for r in all_rows]}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'runs':len(all_rows),'pdf':str(OUT/'time_and_oracle_plots.pdf')},indent=2))


if __name__=='__main__':main()
