"""Plot completed FraMES long runs from checksum-bound Ruche artifacts."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATASETS = {'bmmc6224': 'BMMC · 6,224 cells', 'snare_full': 'SNARE · 10,000 cells',
            'shareseq_full': 'SHARE · 31,890 cells'}
COLORS = {'product': '#1264a3', 'randomized': '#cc641b'}

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    collection=json.loads((OUT/'source/collection.json').read_text())
    for name, record in collection.items():
        assert digest(OUT/'source'/name)==record['sha256']
    reports={k:json.loads((OUT/'source'/f'{k}_report.json').read_text()) for k in ('original','retry')}
    cfg_path=ROOT/'extensions/long_horizon_20261007/campaign.json'
    assert all(r['config_sha256']==digest(cfg_path) for r in reports.values())
    config=json.loads(cfg_path.read_text())
    runs=[]
    for part,dataset in [('original','bmmc6224'),('retry','snare_full')]:
        chosen=[dict(r, campaign_part=part) for r in reports[part]['runs']
                if r['dataset']==dataset and r['method']=='smoothing-fw']
        assert len(chosen)==6 and all(r['status']=='completed' and r['completed_updates']==20000 for r in chosen)
        runs.extend(chosen)
    status=json.loads((OUT/'source/share_status.json').read_text())
    assert status['status']=='completed' and status['returncode']==0
    assert status['metadata_sha256']==digest(OUT/'source/share_metadata.json')
    spec=importlib.util.spec_from_file_location('long_report', ROOT/'extensions/long_horizon_20261007/report_campaign.py')
    report_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(report_module)
    shareconfig_path=ROOT/'extensions/long_horizon_share_20261007/campaign.json'
    shareconfig=json.loads(shareconfig_path.read_text())
    sharemetadata=json.loads((OUT/'source/share_metadata.json').read_text())
    share=report_module.extract_result(sharemetadata,shareconfig,'smoothing-fw','product',810)
    assert share['last_rounded_loss']==status['last_rounded_loss']
    runs.append(dict(share,dataset='shareseq_full',method='smoothing-fw',initialization='product',seed=810,
                     candidate_id='fw06',status='completed',campaign_part='share'))
    checkpoints=[0]+config['final_checkpoint_iterations']
    observations=[]; summaries=[]; endpoints=[]
    for dataset in DATASETS:
        for start in ('product','randomized'):
            population=sorted([r for r in runs if r['dataset']==dataset and r['initialization']==start],key=lambda r:r['seed'])
            if not population:
                assert (dataset,start)==('shareseq_full','randomized');continue
            expected=[810] if dataset=='shareseq_full' else [810,811,812]
            assert [r['seed'] for r in population]==expected
            for r in population:
                points={p['iteration']:p for p in r['trajectory']}
                assert sorted(points)==checkpoints
                initial=points[0]['rounded_loss']
                for iteration in checkpoints:
                    p=points[iteration];loss=p['rounded_loss'];assert np.isfinite(loss)
                    observations.append(dict(dataset=dataset,initialization=start,seed=r['seed'],candidate_id=r['candidate_id'],
                        iteration=iteration,loss=loss,loss_reduction_percent=100*(initial-loss)/initial,
                        precision=p['precision'],campaign_part=r['campaign_part']))
            for iteration in checkpoints:
                rows=[p for p in observations if (p['dataset'],p['initialization'],p['iteration'])==(dataset,start,iteration)]
                row=dict(dataset=dataset,initialization=start,iteration=iteration,n=len(rows))
                for field in ('loss','loss_reduction_percent'):
                    q25,median,q75=np.quantile([p[field] for p in rows],[.25,.5,.75],method='linear')
                    row.update({field+'_median':float(median),field+'_q25':float(q25),field+'_q75':float(q75)})
                summaries.append(row)
            series=[p for p in summaries if p['dataset']==dataset and p['initialization']==start]
            at={p['iteration']:p for p in series}
            endpoints.append(dict(dataset=dataset,initialization=start,n=len(population),initial_loss=at[0]['loss_median'],
                loss_at_2000=at[2000]['loss_median'],loss_at_20000=at[20000]['loss_median'],
                median_reduction_percent_at_20000=at[20000]['loss_reduction_percent_median'],
                lowest_observed_median_iteration=min(series,key=lambda p:p['loss_median'])['iteration']))
    assert len(runs)==13 and len(observations)==169 and len(summaries)==65
    save_csv('observations.csv',observations);save_csv('summary.csv',summaries);save_csv('endpoints.csv',endpoints)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(15,8.4),sharex=True)
    fig.subplots_adjust(left=.078,right=.982,bottom=.17,top=.815,wspace=.30,hspace=.32)
    fig.suptitle('FraMES: loss over 20,000 updates',fontsize=20,fontweight='bold',y=.975)
    fig.text(.5,.93,'BMMC and SNARE: three-seed medians and interquartile bands  |  SHARE: one completed run',ha='center',fontsize=11,color='#44505d')
    handles=[Line2D([],[],color=COLORS[start],marker='o' if start=='product' else 's',linestyle='-' if start=='product' else '--',lw=2,label=start.capitalize()+' initialization') for start in COLORS]
    handles.append(Line2D([],[],color='#828b97',linestyle=':',lw=1.4,label='Earlier 2,000-update budget'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.905),ncol=3,frameon=False)
    for col,(dataset,title) in enumerate(DATASETS.items()):
        for row,field in enumerate(('loss','loss_reduction_percent')):
            ax=axes[row,col]
            for start,color in COLORS.items():
                values=[p for p in summaries if p['dataset']==dataset and p['initialization']==start]
                if not values:continue
                x=np.array([p['iteration'] for p in values]);y=np.array([p[field+'_median'] for p in values])
                if values[0]['n']==3:
                    ax.fill_between(x,[p[field+'_q25'] for p in values],[p[field+'_q75'] for p in values],color=color,alpha=.15,lw=0)
                ax.plot(x,y,color=color,marker='o' if start=='product' else 's',markersize=4.3,
                    linestyle='-' if start=='product' else '--',lw=1.9,zorder=3)
            ax.axvline(2000,color='#828b97',lw=1.2,linestyle=':',zorder=0)
            ax.grid(alpha=.17);ax.set_xlim(-300,20400);ax.set_xticks([0,5000,10000,15000,20000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:'0' if x==0 else f'{x/1000:g}k'))
            if row==0:
                ax.set_title(title+'\n'+('n = 1 · product start · preliminary' if dataset=='shareseq_full' else 'n = 3 per initialization'),loc='left',pad=10)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
            else:
                ax.axhline(0,color='#828b97',lw=.7,alpha=.5)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_xlabel('Completed outer updates')
        axes[0,0].set_ylabel('Feasible GW loss\n(lower is better)')
        axes[1,0].set_ylabel('Loss reduction from initialization (%)\n(higher is better)')
    fig.text(.078,.104,'Markers show saved current-iterate losses. Lines connect observations; shaded bands are IQRs, not confidence intervals.',fontsize=9,color='#44505d')
    fig.text(.078,.079,'Percentage reduction is computed per seed before aggregation; it is not an optimality gap. Axis scales differ by dataset.',fontsize=9,color='#44505d')
    fig.text(.078,.054,'Loss: unregularized squared GW objective of the rounded feasible plan. Intermediate diagnostics: FP32; final endpoints: FP64.',fontsize=9,color='#44505d')
    fig.text(.078,.029,'Snapshot: 8 October 2026, 10:30 Paris. SHARE randomized initialization and remaining seeds are still pending.',fontsize=9,color='#44505d')
    fig.savefig(OUT/'frames_20000_iterations.png',dpi=175,facecolor='white')
    fig.savefig(OUT/'frames_20000_iterations.pdf',facecolor='white');plt.close(fig)
    provenance=dict(source_files=collection,config_sha256=digest(cfg_path),share_config_sha256=digest(shareconfig_path),
        completed_runs=len(runs),observations=len(observations),summaries=len(summaries),checkpoints=checkpoints,
        aggregation='All three seeds for each BMMC/SNARE initialization; single SHARE product seed810. No incomplete medians or missing-seed substitution.',
        output_statistic='Current rounded feasible-plan loss; exact final FP64 last-rounded loss at20000. This plot is not the theorem sampled output statistic.',
        plotter_sha256=digest(__file__),endpoints=endpoints,
        files={name:digest(OUT/name) for name in ('observations.csv','summary.csv','endpoints.csv','frames_20000_iterations.png','frames_20000_iterations.pdf')})
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'endpoints':endpoints},indent=2))

if __name__=='__main__':main()
