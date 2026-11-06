"""Constructed layout fixture only: these values are NOT EXPERIMENT DATA."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import summarize_momentum_replays as report
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

out=Path(__file__).resolve().parent
plan=report.read_json(ROOT/'configs/momentum_diagnostic_replays/finals.json')
runs=[]
for trial in plan['runs']:
    if trial['original_phase']!='final_randomized': continue
    spec=trial['original_run_spec']
    run=dict(run_id=trial['run_id'],original_phase=trial['original_phase'],method=trial['method'],instance=trial['instance'],family=spec['family'],noise=spec['noise'],prescribed_checkpoint_seconds=spec['checkpoints'],verified=spec['seed']!=104,checkpoints=[])
    if run['verified']:
        for target in spec['checkpoints']:
            if target==30 and spec['seed']==103: continue
            factor={'smoothing-fw':1.,'smoothing-fw-exact':.6,'smoothing-row-projection':1.3}[trial['method']]
            run['checkpoints'].append(dict(target_seconds=target,original_solver_seconds=target+.01,momentum_error_frobenius=factor/(target+1),stale_current_output_error_frobenius=1.2*factor/(target+1)))
    runs.append(run)
# Make BMMC exact complete to show a filled marker beside incomplete open markers.
for run in runs:
    if run['family']=='bmmc' and run['method']=='smoothing-fw-exact':
        run['verified']=True
        run['checkpoints']=[dict(target_seconds=t,original_solver_seconds=t+.01,momentum_error_frobenius=.6/(t+1),stale_current_output_error_frobenius=.72/(t+1)) for t in run['prescribed_checkpoint_seconds']]
original_savefig=Figure.savefig

def marked_savefig(fig,*args,**kwargs):
    if not getattr(fig,'fixture_watermarked',False):
        fig.text(.5,.5,'CONSTRUCTED PARSER FIXTURE — NOT EXPERIMENT DATA',ha='center',va='center',rotation=18,fontsize=22,alpha=.24,color='darkred')
        fig.fixture_watermarked=True
    return original_savefig(fig,*args,**kwargs)

Figure.savefig=marked_savefig
figures,curves=report.make_plots(runs,out)
(out/'fixture_populations.json').write_text(json.dumps(dict(scope=__doc__,figures=figures,curve_summaries=curves),indent=2)+'\n')
