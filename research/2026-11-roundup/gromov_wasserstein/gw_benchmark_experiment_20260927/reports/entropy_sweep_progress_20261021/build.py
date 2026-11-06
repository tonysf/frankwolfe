"""Plot observed feasible GW losses; partial trajectories end at actual diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
EPS = [1e-3, 3e-4, 1e-4, 3e-5]
COLORS = ['#b88a00', '#009e73', '#d55e00', '#0072b2']
EPS_TEXT = ['10⁻³', '3×10⁻⁴', '10⁻⁴', '3×10⁻⁵']


def load_frames(dataset, prefix):
    paths = [ROOT / f'reports/wallclock_comparison_20261018/input/{dataset}/confirmation/{prefix}_seed{s}/metadata.json'
             for s in (811, 812, 813)]
    if dataset == 'bmmc6224':
        paths.append(ROOT / 'experiments/bmmc_cpu_batch4096_20261018/runs/frames_q4096_seed811/metadata.json')
    rows = []
    for p in paths:
        raw = p.read_bytes()
        r = json.loads(raw)
        r.update(source=str(p), source_sha256=hashlib.sha256(raw).hexdigest(),
                 run_id=f'{dataset}/frames_q{r["config"]["pairs"]}_{r["config"]["device"]}_seed{r["seed"]}')
        rows.append(r)
    return rows


def xy(row):
    return ([p['iteration'] for p in row['trace']], [p['rounded_loss'] for p in row['trace']])


def verify(rows):
    identity = rows[0]
    for r in rows:
        x, y = xy(r)
        assert x[0] == 0 and all(a < b for a, b in zip(x, x[1:])), r['run_id']
        assert np.isfinite(y).all()
        assert r['config']['dtype'] == 'float64'
        assert r['array_files_sha256'] == identity['array_files_sha256']
        assert r['manifest_sha256'] == identity['manifest_sha256']
        assert x[-1] <= r['completed_updates']
        assert max(p['rounded_residuals']['both_l1'] for p in r['trace']) < 1e-12
        if r['status'] == 'completed':
            assert x[-1] == r['completed_updates'] and y[-1] == r['final_loss']


def setup_axis(ax):
    ax.grid(True, color='#e5e7eb', linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel('Optimizer iterations')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.4f}'))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def start_label(r):
    c = r['config']
    return 'product start' if c['initialization'] == 'product' else f'permutation mixture, seed {c["seed"]}'


def compare(dataset, name, pot, frames, stamp, pdf):
    selected = [min([r for r in pot if r['config']['epsilon'] == e],
                    key=lambda r: r['trace'][-1]['rounded_loss']) for e in EPS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.1), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.subplots_adjust(left=.085, right=.98, top=.69, bottom=.20, wspace=.27)
    fig.suptitle(f'{name}: FRAMES and the POT entropy sweep', x=.085, y=.97,
                 ha='left', fontsize=18, fontweight='bold')
    fig.text(.085, .921, f'FP64 · Feasible unregularized GW loss · Lower is better · Saved results as of {stamp}', fontsize=10, color='#444444')
    x, _ = xy(frames[0])
    ys = np.array([xy(r)[1] for r in frames[:3]])
    assert all(xy(r)[0] == x for r in frames[:3])
    for ax in axes:
        setup_axis(ax)
        ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color='#7b3294', alpha=.14)
        ax.plot(x, np.median(ys, axis=0), color='#7b3294', linewidth=2.1,
                label='FRAMES, 1,024 pairs: median of 3 seeds (band: range)')
        if len(frames) == 4:
            ax.plot(*xy(frames[3]), color='#cc79a7', linewidth=2.0, linestyle='-.',
                    label='FRAMES, 4,096 pairs: CPU, seed 811')
        for r, color, eps in zip(selected, COLORS, EPS_TEXT):
            partial = r['status'] != 'completed'
            px, py = xy(r)
            label = f'POT, ε={eps}: {start_label(r)}' + (' (unfinished)' if partial else '')
            ax.plot(px, py, color=color, linewidth=1.8, linestyle='--' if partial else '-',
                    marker='o', markersize=2.5, label=label)
            ax.scatter(px[-1], py[-1], marker='X' if partial else 'o', color=color, s=48 if partial else 22, zorder=5)
    axes[0].set_title('Full iteration horizon')
    axes[0].set_xlim(0, 20000)
    axes[0].set_ylabel('Feasible unregularized GW loss')
    early_max = 250 if dataset == 'bmmc6224' else 650
    axes[1].set_title(f'First {early_max} iterations')
    axes[1].set_xlim(0, early_max)
    values = [p['rounded_loss'] for r in pot + frames for p in r['trace']]
    ymin, ymax = min(values), max(values)
    padding = (ymax-ymin)*.06
    for ax in axes:
        ax.set_ylim(ymin-padding, ymax+padding)
    best = min([r for r in pot if r['status'] == 'completed'], key=lambda r: r['final_loss'])
    bx, by = xy(best)
    axes[0].hlines(by[-1], bx[-1], 20000, color='#d55e00', linewidth=1.1, linestyle=(0,(2,3)), alpha=.75)
    axes[0].annotate(f'Best completed POT: {by[-1]:.6f}', xy=(6500, by[-1]), xytext=(6500, by[-1]+padding*.35),
                     fontsize=10, color='#b34c00')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.08,.885), ncol=2,
               fontsize=9.5, frameon=False, columnspacing=1.8, handlelength=2.5)
    notes = [
        'POT curves select the lowest final loss among completed starts at each ε; at 3×10⁻⁵, only the ongoing product start is available.',
        'POT dots are measured losses; × marks an unfinished trajectory. The horizontal dotted line is a final-value reference only.',
        'FRAMES uses product initialization; POT initialization varies. Iterations have different computational costs.',
    ]
    if dataset == 'bmmc6224':
        notes.append('The 4,096-pair CPU run is one replicate; CPU/CUDA random streams differ, so batch size is not isolated.')
    for y, text in zip((.13,.099,.068,.037), notes):
        fig.text(.085,y,text,fontsize=8.7,color='#444444')
    path=OUT/f'{dataset}_comparison.png'
    fig.savefig(path,dpi=180,facecolor='white')
    pdf.savefig(fig)
    plt.close(fig)
    return selected


def sweep(dataset, name, rows, frames, stamp, pdf):
    fig, axes = plt.subplots(2,2,figsize=(12,8.8))
    fig.subplots_adjust(left=.09,right=.98,top=.80,bottom=.14,wspace=.25,hspace=.35)
    fig.suptitle(f'{name}: every available POT trajectory', x=.09,y=.975,ha='left',fontsize=18,fontweight='bold')
    fig.text(.09,.931,f'Full-gradient entropy-regularized GW updates · Sinkhorn tolerance 10⁻⁷ · {stamp}',fontsize=10,color='#444444')
    starts=[('product',810),('permutation-mixture',901),('permutation-mixture',902),('permutation-mixture',903)]
    colors=['#333333','#0072b2','#009e73','#d55e00']
    lines=['-','--','-.',':']
    best_frames=min(r['final_loss'] for r in frames)
    all_vals=[p['rounded_loss'] for r in rows for p in r['trace']]
    lo,hi=min(min(all_vals),best_frames),max(all_vals)
    pad=(hi-lo)*.06
    for ax,e,etext in zip(axes.flat,EPS,EPS_TEXT):
        setup_axis(ax)
        subset=[r for r in rows if r['config']['epsilon']==e]
        for (init,seed),color,style in zip(starts,colors,lines):
            match=[r for r in subset if r['config']['initialization']==init and r['seed']==seed]
            if not match:continue
            r=match[0];x,y=xy(r);partial=r['status']!='completed'
            ax.plot(x,y,color=color,linestyle=style,linewidth=1.7,marker='o',markersize=2.3,
                    label='Product start' if init=='product' else f'Permutation mixture, seed {seed}')
            ax.scatter(x[-1],y[-1],color=color,s=50 if partial else 24,marker='X' if partial else 'o',zorder=5)
        ax.axhline(best_frames,color='#7b3294',linewidth=1.1,linestyle=(0,(4,3)),label='Best FRAMES at 20,000 iterations (reference)')
        ax.set_title(f'ε = {etext}'+ (' — ongoing product start' if e==3e-5 else ' — four completed starts'),fontsize=11)
        ax.set_ylim(lo-pad,hi+pad)
        max_x=max(xy(r)[0][-1] for r in subset)
        ax.set_xlim(0,max_x*1.06)
    for ax in axes[:,0]:ax.set_ylabel('Feasible unregularized GW loss')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.085,.905),ncol=2,frameon=False,fontsize=10)
    fig.text(.09,.07,'Dots are recorded diagnostics; × marks the last diagnostic of an unfinished run. All panels use the same loss scale.',fontsize=9,color='#444444')
    fig.text(.09,.04,'Other starts at ε=3×10⁻⁵ and all ε=10⁻⁶ runs have no trajectories yet. Dashed purple lines are references, not POT runs.',fontsize=9,color='#444444')
    fig.savefig(OUT/f'{dataset}_all_pot_runs.png',dpi=180,facecolor='white')
    pdf.savefig(fig)
    plt.close(fig)


def main():
    export_path=OUT/'input/ruche_traces.json'
    export=json.loads(export_path.read_text())
    stamp=datetime.fromisoformat(export['fetched_utc']).astimezone(ZoneInfo('Europe/Paris')).strftime('%d %b, %H:%M Paris')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42})
    all_rows=[];summary={'fetched_utc':export['fetched_utc'],'metric':'rounded_loss: feasible unregularized GW loss',
                        'input_export_sha256':hashlib.sha256(export_path.read_bytes()).hexdigest(),'datasets':{}}
    groups=[]
    with PdfPages(OUT/'runs_so_far.pdf') as pdf:
        for ds,name,prefix in [('bmmc6224','BMMC · 6,224 cells','c08'),('snare_full','SNARE-seq · 10,000 cells','c06')]:
            pot=[r for r in export['runs'] if r['run_id'].startswith(ds+'/')]
            frames=load_frames(ds,prefix)
            verify(pot+frames)
            assert len(pot)==13 and sum(r['status']=='completed' for r in pot)==12
            selected=compare(ds,name,pot,frames,stamp,pdf)
            all_rows.extend(pot+frames)
            groups.append((ds,name,pot,frames))
            summary['datasets'][ds]={'completed_pot_runs':12,'partial_pot_runs':1,
              'selected_pot_runs':[r['run_id'] for r in selected],
              'runs':[dict(run_id=r['run_id'],status=r['status'],completed_updates=r['completed_updates'],
                           last_diagnostic_iteration=xy(r)[0][-1],last_feasible_loss=xy(r)[1][-1],
                           config=r['config'],source=r['source'],source_sha256=r['source_sha256']) for r in pot+frames]}
        for ds,name,pot,frames in groups:sweep(ds,name,pot,frames,stamp,pdf)
    with (OUT/'curves.csv').open('w',newline='') as f:
        fields=['run_id','method','status','epsilon','pairs','device','seed','initialization','iteration','feasible_gw_loss']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in all_rows:
            c=r['config']
            for p in r['trace']:
                writer.writerow(dict(run_id=r['run_id'],method=r['method'],status=r['status'],
                  epsilon=c.get('epsilon') if r['method']=='pot-full' else '',pairs=c.get('pairs',''),
                  device=c['device'],seed=r['seed'],initialization=c.get('initialization','product'),
                  iteration=p['iteration'],feasible_gw_loss=p['rounded_loss']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'runs':len(all_rows),'pdf':str(OUT/'runs_so_far.pdf'),'fetched':stamp},indent=2))


if __name__=='__main__':main()
