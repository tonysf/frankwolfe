"""Present the recorded stochastic comparison with common oracle and momentum."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from method_names import method_name
from summarize_campaign import stat_text, write_flat_csv
ROOT=Path(__file__).resolve().parent
METHODS=('smoothing-fw','stochastic-kl-momentum')
COLORS={'smoothing-fw':'#0072B2','stochastic-kl-momentum':'#AA3377'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def family(r):return 'BMMC n=1000' if r['family']=='bmmc' else 'Clean synthetic n=512' if not r['noise'] else 'Noisy synthetic n=512'
def errorbar(ax,point,xkey,ykey,method,label):
    x=point['metrics'][xkey];y=point['metrics'][ykey]
    if x['median'] is None or y['median'] is None:return
    ax.errorbar(x['median'],y['median'],xerr=[[x['median']-x['q25']],[x['q75']-x['median']]] if x['count']>1 else None,yerr=[[y['median']-y['q25']],[y['q75']-y['median']]] if y['count']>1 else None,fmt='o',color=COLORS[method],capsize=2,markerfacecolor=COLORS[method] if point['coverage_complete'] else 'none',label=method_name(method) if label else None)
def main():
    source=ROOT/'reports/stage_c_final/report.json';report=json.loads(source.read_text())
    assert report['execution_complete'] and report['provenance_valid']
    assert sha(source)=='39730329a43e14f0ce3af063f8263bb5b6b45e6acc45696ccbff164813d9086c'
    runs=[x for x in report['runs'] if x['method'] in METHODS]
    assert Counter(x['method'] for x in runs)==Counter({x:110 for x in METHODS})
    assert all(x['eligible_final_output'] for x in runs)
    curves=[x for x in report['across_problem_aggregates'] if x['method'] in METHODS]
    endpoints=[x for x in report['final_endpoint_aggregates'] if x['method'] in METHODS]
    assert len(curves)==len(endpoints)==12
    out=ROOT/'reports/stochastic_primary';out.mkdir(exist_ok=True)
    files=[]
    for phase in ('final_product','final_randomized'):
        selected=[x for x in curves if x.get('comparison_phase',x['phase'])==phase]
        assert len(selected)==6
        for kind,metrics in [('quality',[('time_to_output','rounded_loss'),('cumulative_sketch_probes','rounded_loss')]),('feasibility',[('actual_solver_seconds','col_l1'),('actual_solver_seconds','col_l2')])]:
            fig,axes=plt.subplots(2,3,figsize=(13,7.4))
            for col,title in enumerate(('Clean synthetic n=512','Noisy synthetic n=512','BMMC n=1000')):
                for method in METHODS:
                    curve=next(x for x in selected if family(x)==title and x['method']==method)
                    for row,(xkey,ykey) in enumerate(metrics):
                        for i,point in enumerate(curve['points']):errorbar(axes[row,col],point,xkey,ykey,method,i==0)
                for row,(xkey,ykey) in enumerate(metrics):
                    ax=axes[row,col];ax.set_title(title);ax.set_xscale('log');ax.grid(alpha=.2)
                    ax.set_xlabel('Actual solver + output repair time (s)' if xkey=='time_to_output' else 'Rademacher probes drawn' if xkey=='cumulative_sketch_probes' else 'Actual solver time (s)')
                    if col==0:ax.set_ylabel('Full GW loss at rounded output' if ykey=='rounded_loss' else 'Raw column '+('L1' if ykey=='col_l1' else 'L2')+' residual')
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=1,fontsize=9)
            start='Product start; 300-second solver budget' if phase=='final_product' else 'Randomized feasible start; 30-second solver budget'
            fig.suptitle('Same stochastic gradient oracle and momentum — '+start+'\nq=128; rho(k)=0.3/(k+1)^(1/2); paired instance and seeds',fontsize=11)
            note='Observed checkpoints only; no interpolation to a common probe budget. Open markers: incomplete population.' if kind=='quality' else 'Synthetic: across-problem IQR after within-instance medians. BMMC: repeat IQR on one fixed dataset.'
            fig.text(.5,.105,note,ha='center',fontsize=9);fig.tight_layout(rect=(0,.15,1,.91))
            for ext in ('png','pdf'):
                name=phase+'_'+kind+'.'+ext;fig.savefig(out/name,dpi=160);files.append(name)
            plt.close(fig)
    result=dict(generated_utc=datetime.now(timezone.utc).isoformat(),source_report=str(source.relative_to(ROOT)),source_report_sha256=sha(source),scope='Same stochastic gradient oracle and momentum, at prescribed solver-time budgets; not a fixed-oracle-budget experiment.',method_ids=list(METHODS),run_count=len(runs),matched_pairs=110,curves=curves,endpoints=endpoints,figures=files,source_sha256=sha(__file__))
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');write_flat_csv(out/'runs.csv',runs)
    lines=['# Primary stochastic comparison','', 'Compare **Smoothed row FW + momentum (sketch gradient)** with **Transport KL-proximal gradient + momentum (sketch gradient)**. Both use the same q=128 Rademacher sketch and momentum recurrence. There are 110 corresponding instance/seed/start cells, with 220 valid final outputs. This comparison was included before held-out execution; the presentation selects methods by oracle/momentum compatibility, not by which one wins.','', '[Matching audit](../STOCHASTIC_MATCH_AUDIT.md) · [method definitions](../METHOD_NAMES.md) · [complete original results](../stage_c_final_named/report.md)','', 'Matching covers instance, initialization, stochastic seed, gradient estimator/batch, iteration-indexed momentum schedule and initialization convention, GPU model/application libraries/CUDA runtime/precision, output rounding and prescribed solver-time budget. Twenty-five pairs differ in Ruche OS/driver revisions and 64 ran on different hosts. The complete machine environment is not identical in every pair. The same iteration-indexed schedule does not imply equal momentum coefficients at a given wall time. Constraint handling and the associated step/smoothing parameters differ by algorithm.','', '**Scope limits:** equal solver time is not equal oracle work: KL inner iterations reduce the number of gradient calls performed within a time budget. The work-axis panels display the actual recorded probe counts; sparse time checkpoints do not establish equal-probe-budget performance and are not interpolated to pretend otherwise. Momentum-KL inherits the plain-KL step settings without an independent tuning grid; this is not a comparison of independently optimal hyperparameters.','', 'Final rounded losses below are float64 evaluations of the saved outputs. Entries give median [Q1, Q3]. Product cases have 300-second budgets; randomized cases have 30-second budgets. Actual output times and current-checkpoint losses are in the plots and source JSON. Synthetic summaries weight each of five problems equally after within-problem medians; BMMC is one fixed dataset.','', '| Start | Data | Method | Final rounded GW loss |','|---|---|---|---:|']
    for x in sorted(endpoints,key=lambda x:(x['initialization']!='product',family(x),METHODS.index(x['method']))):lines.append('| '+('Product' if x['initialization']=='product' else 'Randomized')+' | '+family(x)+' | '+method_name(x['method'])+' | '+stat_text(x['metrics']['rounded_loss_float64'])+' |')
    lines += ['', 'At the recorded endpoints, momentum-KL gives lower median loss for both synthetic families under each start condition; smoothed row FW gives lower median loss on fixed BMMC under each start condition. This mixed, configuration-specific result is not a uniform winner, significance claim or oracle-complexity result.','', '| Method | Valid outputs | Runs with inner stopping failures | Known inner stopping failures | Warnings |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        selected=[x for x in runs if x['method']==method]
        lines.append('| '+method_name(method)+' | 110/110 | '+str(sum(bool(x['counters'].get('inner_stopping_failures',0)) for x in selected))+' | '+str(sum(x['counters'].get('inner_stopping_failures',0) or 0 for x in selected))+' | '+str(sum(x.get('warning_count',0) for x in selected))+' |')
    lines += ['', 'A valid rounded output is not a certificate that every inner KL solve met its stopping tolerance. Exact-gradient methods are kept in the separate original diagnostic/context archive and are absent from these figures. Plain sketch-gradient KL, the independently tuned row-projection variant and native Sampled GW have additional estimator/momentum/tuning differences and are secondary comparisons.','', 'Figures show checkpoint medians with horizontal and vertical IQRs. Synthetic IQRs are across problem-level medians; BMMC IQRs are across repeats on its one fixed dataset. Open markers denote incomplete checkpoint populations.','', 'Figures:','']+['- ['+name+']('+name+')' for name in files]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'source_report_sha256':sha(source),'runs':len(runs),'matched_pairs':110,'figure_files':len(files)}))
if __name__=='__main__':main()
