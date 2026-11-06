"""Audit completed fixed-horizon results and plot feasible GW loss by iteration.

POT controls optimize f+epsilon*h; FraMES optimizes the unregularized constrained
GW problem. The common diagnostic below is f of the rounded feasible coupling.
The report consumes exact FP64 diagnostics recorded during allocated GPU work;
it does not claim an independent reevaluation of large saved plans.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign

LABELS = {'pot-full': 'Full-gradient entropic GW, POT step',
          'pot-pair': 'Pair-sampled entropic GW, POT step',
          'frames': 'FraMES, retuned prefactors'}


def audit_trace(meta):
    trace = meta['trace']
    iterations = [r['iteration'] for r in trace]
    campaign.require(iterations == sorted(set(iterations)) and iterations[0] == 0
                     and iterations[-1] == meta['target_updates'], 'Incomplete diagnostic trajectory')
    for row in trace:
        for key in ('raw_loss','rounded_loss','raw_entropy','rounded_entropy',
                    'raw_regularized_loss','rounded_regularized_loss'):
            campaign.require(math.isfinite(row[key]), 'Nonfinite objective diagnostic')
        campaign.require(0 <= row['rounded_residuals']['both_l1'] <= 512*sys.float_info.epsilon,
                         'Rounded plan fails FP64 feasibility audit')
        campaign.require(abs(row['rounded_regularized_loss']-row['rounded_loss']
                             -meta['config']['epsilon']*row['rounded_entropy']) <= 1e-12,
                         'Objective and entropy disagree')
    campaign.require(meta['final_loss'] == trace[-1]['rounded_loss'], 'Final metric differs from trace')
    count = meta['completed_updates']
    counters = meta['counters']
    if meta['method'] == 'pot-full':
        campaign.require(counters['full_gradient_calls'] == count and counters['pair_samples'] == 0,
                         'Full-gradient oracle count differs')
    else:
        campaign.require(counters['stochastic_gradient_calls'] == count
                         and counters['pair_samples'] == count*meta['config']['pairs'],
                         'Pair oracle count differs')
    if meta['method'].startswith('pot-'):
        campaign.require(len(meta['inner_trace']) == count, 'Missing inner certificates')
        for k, row in enumerate(meta['inner_trace']):
            campaign.require(row['iteration'] == k and row['converged']
                             and row['proximal_optimality_certified'] and row['certificate_valid'],
                             'Unaccepted inner solve')
            scale = max(1., abs(row['primal_objective']), abs(row['dual_objective']))
            gap, stable, guard = (row[key] for key in ('primal_dual_gap', 'stable_primal_dual_gap', 'gap_roundoff_tolerance'))
            campaign.require(all(math.isfinite(v) for v in (gap,stable,guard,scale)), 'Nonfinite inner certificate')
            campaign.require(0 <= row['marginal_l1'] <= 1e-7 and guard >= 0
                             and min(gap,stable) >= -guard and abs(gap-stable) <= guard
                             and (max(gap,stable)+guard)/scale <= 1e-7,
                             'Inner feasibility or guarded gap failed')
    return trace


def aggregate(traces):
    common = set(r['iteration'] for r in traces[0])
    by_iteration = []
    for trace in traces:
        lookup = {r['iteration']: r for r in trace}
        by_iteration.append(lookup)
        common.intersection_update(lookup)
    return [dict(iteration=k, median=statistics.median(values), minimum=min(values), maximum=max(values), n=len(values))
            for k in sorted(common)
            for values in [[lookup[k]['rounded_loss'] for lookup in by_iteration]]]


def collect(root, config, source):
    datasets = {}
    for dataset, instance in config['instances'].items():
        base = Path(root)/'results'/dataset
        selected = campaign.read(base/'selection.json') if (base/'selection.json').exists() else None
        if selected is not None:
            expected = campaign.selection(config, base, source)
            campaign.require(selected['candidate_id'] == expected['candidate_id']
                             and selected['prefactors'] == expected['prefactors']
                             and selected['ranking'] == expected['ranking'], 'Frozen selection differs from complete development results')
        rows, missing, groups, failures = [], [], {}, []
        if (base/'failed.json').exists():
            failures.append(dict(path=str(base/'failed.json'), receipt=campaign.read(base/'failed.json')))
        for task in campaign.tasks(config, selected):
            path = base/task['task_id']/'metadata.json'
            if not path.exists() or campaign.read(path).get('status') != 'completed':
                failure = path.parent/'failure.json'
                if failure.exists():
                    failures.append(dict(path=str(failure), receipt=campaign.read(failure)))
                missing.append(task['task_id']); continue
            meta = campaign.read(path)
            loss = campaign.validate_result(meta, task, instance, config, source)
            trace = audit_trace(meta)
            phase = task['task_id'].split('/')[0]
            record = dict(task_id=task['task_id'], phase=phase, method=task['method'], seed=task['seed'],
                          final_loss=loss, initial_loss=trace[0]['rounded_loss'], iterations=meta['completed_updates'],
                          loss_reduction_percent=100*(1-loss/trace[0]['rounded_loss']) if trace[0]['rounded_loss'] else None,
                          metadata_sha256=campaign.sha256(path), metadata_path=str(path),
                          prefactors=task.get('prefactors'), last_diagnostics=trace[-1])
            rows.append(record)
            if phase != 'development':
                groups.setdefault(task['method'], []).append(trace)
        curves = {method: aggregate(traces) for method,traces in groups.items()}
        expected_counts = {'pot-full':1,'pot-pair':3,'frames':3}
        coverage = {method:dict(observed=len(groups.get(method,[])), expected=count)
                    for method,count in expected_counts.items()}
        complete = selected is not None and not missing and len(rows) == 55
        datasets[dataset] = dict(status='completed' if complete else 'partial', n=instance['n'],
                                runs=rows, missing_tasks=missing, selection=selected, curves=curves,
                                failures=failures, confirmation_coverage=coverage)
    return datasets


def plots(datasets, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name,data in datasets.items():
        if not data['curves']: continue
        fig, axes = plt.subplots(1,2,figsize=(11,4.5))
        for method,curve in data['curves'].items():
            x=[r['iteration'] for r in curve]; y=[r['median'] for r in curve]
            low=[r['minimum'] for r in curve]; high=[r['maximum'] for r in curve]
            observed=data['confirmation_coverage'][method]['observed']
            label=LABELS[method]+f' (n={observed})'
            for ax in axes:
                line=ax.plot(x,y,label=label,linewidth=1.8)[0]
                if observed > 1: ax.fill_between(x,low,high,color=line.get_color(),alpha=.16)
        axes[0].set_xlim(0,20000); axes[1].set_xlim(0,2000)
        for ax in axes:
            ax.set_xlabel('Outer updates');ax.set_ylabel('Feasible full squared GW loss f(P)');ax.grid(alpha=.2)
        axes[0].set_title('Full horizon'); axes[1].set_title('First 2,000 updates')
        handles, labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.08),fontsize=8)
        fig.suptitle(f'{name}: {data["n"]:,} cells | FP64 | '+('COMPLETE' if data['status']=='completed' else 'PARTIAL'))
        fig.text(.5,.015,'POT controls: f + 0.001 h, fixed step 1000. FraMES: unregularized f, fixed theoretical exponents.\n'
                 'Shown: current rounded losses; median and observed seed range. Development trials excluded from curves.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.25,1,.94))
        paths=[]
        for extension in ('png','pdf'):
            path=directory/f'{name}_loss_iterations.{extension}';fig.savefig(path,dpi=170);paths.append(str(path))
        plt.close(fig);artifacts[name]=paths
    return artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args(); config=campaign.load_config(args.config)
    source=HERE.parents[1]
    data=collect(args.root,config,source)
    complete=all(v['status']=='completed' for v in data.values())
    if not args.partial: campaign.require(complete,'Incomplete campaign; use --partial for clearly labelled available results')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Unique snapshots prevent simultaneous per-dataset reports overwriting files.
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot=args.output.parent/('snapshot_'+stamp)
    artifacts=plots(data,snapshot)
    result=dict(status='completed' if complete else 'partial', generated_utc=campaign.now(),
                config_sha256=campaign.sha256(args.config), datasets=data, artifacts=artifacts,
                gate_receipts={str(p.relative_to(args.root)):campaign.read(p) for p in sorted((args.root/'checks').glob('**/gate.json'))},
                evaluation_scope='Exact FP64 actual-marginal objective and rounding diagnostics recorded in allocated GPU runs; small independent POT and distributed parity tests validate the implementation. Large plans are not independently reevaluated by this report.',
                scientific_scope='POT-step controls optimize entropy-regularized GW; FraMES optimizes unregularized constrained GW. Their plotted common diagnostic is feasible GW loss, without entropy. Empirical POT-step stochastic behavior has no claimed theorem guarantee.',
                uncertainty='Median and observed seed range; not a confidence interval.')
    campaign.write(snapshot/'report.json',result)
    # CPU report jobs use a file lock when promoting the latest snapshot.
    import fcntl
    with (args.output.parent/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        campaign.write(args.output,result)
    with (snapshot/'endpoints.csv').open('w',newline='') as f:
        fields=['dataset','task_id','phase','method','seed','iterations','initial_loss','final_loss','loss_reduction_percent']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for dataset,item in data.items():
            for row in item['runs']:w.writerow(dict(dataset=dataset,**{k:row.get(k) for k in fields if k!='dataset'}))
    print(json.dumps(dict(status=result['status'],output=str(args.output),snapshot=str(snapshot))))

if __name__=='__main__': main()
