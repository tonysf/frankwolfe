"""Metadata-only audit/report of the fixed-horizon MSBPG comparator campaign with frozen prior baselines.

Missing or failed runs remain visible. Final summaries require every configured
seed; endpoint losses and MSBPG's sampled displacement are separate. Plots
show observations only, without interpolation, extrapolation or incumbent loss.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METHODS = ('published-msbpg',)
LABELS = {'smoothing-fw': 'FraMES (frozen completed run)',
          'published-smd': 'Fresh-gradient entropic SMD (frozen reference)',
          'published-msbpg': 'MSBPG (Ding-Li-Toh, theorem parameter choices)'}

# Unique module name avoids importing another extension's campaign module.
import importlib.util
_spec = importlib.util.spec_from_file_location('_msbpg_campaign_contract', Path(__file__).with_name('campaign.py'))
_contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_contract)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, minimum=None):
    require(isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value), 'Nonfinite/invalid '+name)
    require(minimum is None or value >= minimum, 'Out-of-range '+name)
    return float(value)


def integer(value, name, minimum=0):
    require(isinstance(value,int) and not isinstance(value,bool) and value>=minimum, 'Invalid '+name)
    return value


def same(actual, expected):
    if isinstance(expected,float):
        return isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=0.)
    return type(actual) is type(expected) and actual == expected


def quantiles(values):
    """Linear-interpolation quartiles, not a best-seed statistic."""
    ordered = sorted(finite(value,'summary value') for value in values)
    require(bool(ordered),'Empty summary')
    def q(fraction):
        x=(len(ordered)-1)*fraction; lo=int(x); hi=min(lo+1,len(ordered)-1)
        return ordered[lo]+(x-lo)*(ordered[hi]-ordered[lo])
    return dict(n=len(ordered),median=statistics.median(ordered),q25=q(.25),q75=q(.75))


def validate_result(metadata, config, dataset, phase, candidate, start, seed, selection=None):
    """Independently validate receipts without loading a dense plan or torch."""
    require(metadata.get('status')=='completed','Runner did not complete')
    method = next((m for m in METHODS if candidate in config['methods'][m]),None)
    require(method is not None,'Unknown candidate')
    expected_iterations=config[phase+'_iterations']
    actual=metadata['config']
    require(metadata.get('method_key')==method and actual.get('method')==method,'Method mismatch')
    expected=dict(iterations=expected_iterations,seed=seed,output_seed=seed+20000,
                  init_seed=seed+10000,initialization=start,**config['runner'])
    # Storage and checkpoint settings may be recorded elsewhere by the controller;
    # every configured scalar runner setting still binds the numerical receipt.
    expected.update({k:v for k,v in candidate.items() if k!='id'})
    for key,value in expected.items():
        require(key in actual and same(actual[key],value),'Runner config mismatch: '+key)
    n=config['instances'][dataset]['n']
    require(metadata.get('dimensions')==dict(source=n,target=n),'Dataset dimensions mismatch')
    require(metadata['environment']['world_size']==config['instances'][dataset]['world_size'],'World size mismatch')
    manifest_id=config['instances'][dataset].get('manifest_id')
    if manifest_id is not None:
        require(metadata['instance_manifest'].get('id')==manifest_id,'Manifest identifier mismatch')
    require(metadata['instance_manifest'].get('array_files_sha256')==metadata.get('array_files_sha256'),'Input hashes disagree with instance manifest')
    require(metadata['array_files_sha256']==config['instances'][dataset]['array_files_sha256'], 'Inputs differ from frozen baseline dataset')
    require(metadata['instance_manifest_sha256']==config['instances'][dataset]['instance_manifest_sha256'], 'Manifest differs from frozen baseline dataset')
    if selection is not None:
        for field in ('source_file_sha256','instance_manifest_sha256','array_files_sha256'):
            require(metadata.get(field)==selection.get(field),'Selection/input binding mismatch: '+field)
    counters=metadata['counters']
    require(counters['completed_updates']==counters['stochastic_gradient_calls']==expected_iterations,'Incomplete updates/oracle count')
    require(counters['sketch_probes']==expected_iterations*actual['batch'],'Probe count mismatch')
    require(counters['inner_stopping_failures']==0,'Inner solve failure')
    require(counters['initialization_oracle_calls']==1,'Initialization oracle count mismatch')
    output=metadata['output_selection']
    selected=integer(output['index'],'selected index')
    require(selected<expected_iterations and output['horizon']==expected_iterations and output['last_iterate_index']==expected_iterations,'Output indexing mismatch')
    require(output.get('independent_rng') is True,'Output RNG independence missing')
    probability=(selected+1)**-.5/sum((k+1)**-.5 for k in range(expected_iterations))
    require(math.isclose(finite(output['probability'],'selection probability',0),probability,rel_tol=1e-12),'Output probability mismatch')
    if method=='published-msbpg':
        inner=metadata['inner_trace']
        require(len(inner)==expected_iterations,'Inner trace incomplete')
        for k,info in enumerate(inner):
            require(info['iteration']==k and info['converged'] is True and info['proximal_optimality_certified'] is True and info['certificate_valid'] is True,'Rejected inner step')
            require(finite(info['marginal_l1'],'inner marginal residual',0)<=actual['inner_tol'],'Inner marginal tolerance exceeded')
            require(finite(info['normalized_gap_upper_bound'],'guarded inner gap')<=actual['inner_gap_tol'],'Guarded inner gap tolerance exceeded')
            require(finite(info['normalized_primal_dual_gap'],'inner gap')<=actual['inner_gap_tol'],'Inner gap tolerance exceeded')
            require(finite(info['primal_witness_marginal_l1'],'witness residual',0)<=1e-12,'Infeasible primal witness')
        require(sum(x['inner_iterations'] for x in inner)==counters['inner_iterations_total'],'Inner iteration counter mismatch')
        require(sum(x['certificate_checks'] for x in inner)==counters['certificate_checks'],'Certificate counter mismatch')
    else:
        require(counters['inner_iterations_total']==counters['certificate_checks']==0,'FW has unexpected proximal work')
    validation=metadata['final_float64_validation']
    require(validation['dtype']=='float64' and validation['exact_objective'] is True and validation['optimizer_state_released'] is True,'Missing final FP64 validation')
    plans=validation['plans']
    require(set(plans)=={'selected_raw','selected_rounded','last_raw','last_rounded'},'Missing saved output validation')
    for name,plan in plans.items():
        finite(plan['loss'],name+' loss')
        finite(plan['marginal_residuals']['min_entry'],name+' minimum entry',0)
        residual=finite(plan['marginal_residuals']['both_l1'],name+' marginal residual',0)
        if name.endswith('_rounded'):
            require(residual<=config['final_marginal_l1_tolerance'],'Final rounded output exceeds configured feasibility tolerance')
    world=metadata['environment']['world_size']
    require(len(metadata['ranks'])==world and {r['rank'] for r in metadata['ranks']}==set(range(world)),'Incomplete rank receipts')
    for rank in metadata['ranks']:
        rows=n//world+int(rank['rank']<n%world)
        require(set(rank['saved_plans'])==set(plans),'Incomplete rank output snapshots')
        for saved in rank['saved_plans'].values():
            require(saved['shape']==[rows,n] and len(saved['array_sha256'])==64,'Invalid shard shape/hash')
    solver=finite(metadata['timing']['solver_seconds'],'solver time',0)
    points=[]
    previous=-1
    for row in metadata['trace']:
        iteration=integer(row['iteration'],'trace iteration')
        require(previous<=iteration<=expected_iterations,'Nonmonotone/out-of-range trace')
        previous=iteration
        elapsed=finite(row['actual_solver_seconds'],'trace solver time',0)
        feasible=finite(row['time_to_this_feasible_output'],'feasible output time',0)
        repair=finite(row['repair_seconds_current'],'repair time',0)
        require(elapsed<=solver+1e-8 and math.isclose(feasible,elapsed+repair,rel_tol=1e-9,abs_tol=1e-8),'Inconsistent diagnostic clock')
        loss=finite(row['rounded_loss'],'trace rounded loss')
        if row['kind'] in ('initial','current_diagnostic','last'):
            points.append(dict(kind=row['kind'],iteration=iteration,time_to_this_feasible_output=feasible,
                               rounded_loss=plans['last_rounded']['loss'] if row['kind']=='last' else loss))
    lasts=[row for row in points if row['kind']=='last' and row['iteration']==expected_iterations]
    require(len(lasts)==1,'Missing final current-iterate diagnostic')
    statistic = _contract.validate_theory_metadata(metadata, actual)
    return dict(sampled_normalized_displacement_squared=statistic,
                theorem_constants=metadata['theorem_constants'],
                selected_update_statistic=metadata['selected_update_statistic'],
                last_rounded_loss=plans['last_rounded']['loss'],
                selected_raw_loss=plans['selected_raw']['loss'] if method=='published-msbpg' else None,
                selected_index=selected,selected_probability=probability,
                solver_seconds=solver,last_feasible_output_seconds=lasts[0]['time_to_this_feasible_output'],
                completed_updates=expected_iterations,stochastic_gradient_calls=counters['stochastic_gradient_calls'],
                sketch_probes=counters['sketch_probes'],inner_iterations=counters['inner_iterations_total'],
                certificate_checks=counters['certificate_checks'],trajectory=points)


def selection_receipt(root, dataset, config, config_hash):
    path=root/dataset/'selection.json'
    if not path.exists():
        return None,dict(status='missing',file=str(path))
    try:
        selection=read(path)
        require(selection.get('status')=='completed','Selection did not complete')
        require(selection.get('config_sha256')==config_hash,'Selection configuration hash mismatch')
        require(selection.get('baseline_sha256')==config['baseline']['sha256'],'Selection baseline hash mismatch')
        require(set(selection['selected'])==set(METHODS),'Selection must include only MSBPG')
        for method in METHODS:
            require(selection['selected'][method] in {x['id'] for x in config['methods'][method]},'Unknown selected candidate')
        return selection,dict(status='completed',file=str(path),sha256=sha256(path),selected=selection['selected'])
    except Exception as exc:
        return None,dict(status='invalid',file=str(path),error=str(exc))


def compare_final_summaries(summaries):
    """Compare like endpoint quantities; keep the theorem statistic separate."""
    rows = []
    for dataset, start in sorted({(x['dataset'], x['initialization']) for x in summaries}):
        population = {x['method']: x for x in summaries if x['dataset'] == dataset and x['initialization'] == start}
        row = dict(dataset=dataset, initialization=start, status='incomplete')
        if all(method in population and population[method]['status'] == 'completed'
               for method in ('smoothing-fw', 'published-msbpg', 'published-smd')):
            stats = {method: population[method]['statistics'] for method in population}
            fw, msbpg = stats['smoothing-fw'], stats['published-msbpg']
            row.update(status='completed',
                       frames_median_loss=fw['last_rounded_loss']['median'],
                       msbpg_median_loss=msbpg['last_rounded_loss']['median'],
                       reference_smd_median_loss=stats['published-smd']['last_rounded_loss']['median'],
                       msbpg_minus_frames_median_loss=msbpg['last_rounded_loss']['median']-fw['last_rounded_loss']['median'],
                       ratio_msbpg_to_frames_solver_time=msbpg['solver_seconds']['median']/fw['solver_seconds']['median'],
                       ratio_msbpg_to_frames_feasible_output_time=msbpg['last_feasible_output_seconds']['median']/fw['last_feasible_output_seconds']['median'],
                       msbpg_sampled_normalized_displacement_squared=msbpg['sampled_normalized_displacement_squared'],
                       timing_scope='Ratio at the same update count, not time to equal accuracy; frozen baseline measured in earlier jobs.')
        rows.append(row)
    return rows


def audit(config_path, root):
    root=Path(root); config_path=Path(config_path); config=read(config_path)
    config_hash=sha256(config_path)
    baseline = _contract.load_baseline(config_path, config)
    records=[]; selections={}
    for dataset in config['instances']:
        selection,selections[dataset]=selection_receipt(root,dataset,config,config_hash)
        for phase,seeds in (('tuning',config['development_seeds']),('final',config['final_seeds'])):
            for method in METHODS:
                candidates=config['methods'][method]
                if phase=='final':
                    chosen=selection['selected'][method] if selection else None
                    candidates=[next(x for x in candidates if x['id']==chosen)] if chosen else [None]
                for candidate in candidates:
                    for start in config['initializations']:
                        for seed in seeds:
                            row=dict(dataset=dataset,phase=phase,method=method,candidate_id=candidate['id'] if candidate else None,
                                     initialization=start,seed=seed,status='missing',metadata_file=None)
                            if candidate is None:
                                row['reason']='Candidate selection unavailable or invalid'
                                records.append(row); continue
                            output=root/dataset/phase/candidate['id']/start/f'seed{seed}'/'output'
                            metadata_file=output/'metadata.json'; row['metadata_file']=str(metadata_file)
                            controller_status=output.parent/'status.json'
                            if controller_status.exists():
                                try:
                                    controller_receipt=read(controller_status)
                                    if controller_receipt.get('status')=='failed':
                                        row.update(status='failed',failure_files=[str(controller_status)],
                                                   errors=[controller_receipt.get('error','Controller rejected run')])
                                        records.append(row); continue
                                except Exception as exc:
                                    row.update(status='invalid',reason='Unreadable controller status: '+str(exc))
                                    records.append(row); continue
                            if metadata_file.exists():
                                try:
                                    request=read(output.parent/'request.json')
                                    receipt=read(output.parent/'status.json')
                                    require(request.get('config_sha256')==config_hash and request.get('baseline_sha256')==config['baseline']['sha256'] and request.get('instance')==dataset
                                            and request.get('candidate_id')==candidate['id'] and request.get('stage')==phase,
                                            'Run request/config/path mismatch')
                                    require(receipt.get('status')=='completed' and receipt.get('returncode')==0,'Controller run receipt did not complete')
                                    require(receipt.get('metadata_sha256')==sha256(metadata_file),'Metadata differs from controller receipt')
                                    result=read(metadata_file)
                                    require(result.get('config')==request.get('expected_runner_config'),'Runner differs from frozen request')
                                    for field in ('source_file_sha256','instance_manifest_sha256','array_files_sha256'):
                                        require(result.get(field)==request.get(field),'Run provenance mismatch: '+field)
                                    row.update(validate_result(result,config,dataset,phase,candidate,start,seed,
                                                               selection=selection if phase=='final' else None))
                                    row.update(status='completed',metadata_sha256=sha256(metadata_file))
                                except Exception as exc:
                                    row.update(status='invalid',reason=str(exc))
                            else:
                                failures=sorted(output.glob('failure*.json'))+sorted(output.glob('rank*/inner_failure.json'))
                                status_path=output.parent/'status.json'
                                if status_path.exists():
                                    try:
                                        if read(status_path).get('status')=='failed': failures.append(status_path)
                                    except Exception as exc:
                                        row.update(status='invalid',reason='Unreadable controller status: '+str(exc))
                                if failures:
                                    row.update(status='failed',failure_files=[str(x) for x in failures])
                                    row['errors']=[]
                                    for failure in failures:
                                        try: row['errors'].append(read(failure).get('error','Failure receipt'))
                                        except Exception as exc: row['errors'].append('Unreadable failure: '+str(exc))
                                elif (output/'progress.json').exists():
                                    try: row.update(status='started_without_result',progress=read(output/'progress.json'))
                                    except Exception as exc: row.update(status='invalid',reason='Unreadable progress: '+str(exc))
                            records.append(row)
    summaries=[]
    for dataset in config['instances']:
        for start in config['initializations']:
            for method in METHODS:
                expected=[r for r in records if r['dataset']==dataset and r['phase']=='final' and r['initialization']==start and r['method']==method]
                valid=[r for r in expected if r['status']=='completed']
                complete=len(valid)==len(config['final_seeds'])
                stats=None
                if complete:
                    fields=['last_rounded_loss','solver_seconds','last_feasible_output_seconds','stochastic_gradient_calls','sketch_probes','inner_iterations','certificate_checks']
                    if method=='published-msbpg': fields.extend(['selected_raw_loss', 'sampled_normalized_displacement_squared'])
                    stats={field:quantiles([r[field] for r in valid]) for field in fields}
                summaries.append(dict(dataset=dataset,initialization=start,method=method,
                                      status='completed' if complete else 'incomplete',expected_seeds=config['final_seeds'],
                                      completed_seeds=[r['seed'] for r in valid],statistics=stats))
    counts={status:sum(r['status']==status for r in records) for status in ('completed','missing','failed','invalid','started_without_result')}
    complete=all(r['status']=='completed' for r in records) and all(s['status']=='completed' for s in selections.values())
    combined_runs = records + [dict(row, frozen_reference=True) for row in baseline['runs']]
    combined_summaries = summaries + [dict(row, frozen_reference=True) for row in baseline['final_summaries']]
    comparisons = compare_final_summaries(combined_summaries)
    return dict(status='completed' if complete else 'incomplete',
                baseline_provenance=baseline['origin'],baseline_artifact_sha256=config['baseline']['sha256'],
                reused_reference_runs=len(baseline['runs']),
                comparison_scope='Only MSBPG ran in this campaign; FraMES and plain SMD are frozen completed results with matching data/oracle/seed settings. Timing measurements come from separate runs.',
                final_comparisons=comparisons,config_file=str(config_path),config_sha256=config_hash,
                task_counts=counts,selections=selections,runs=combined_runs,final_summaries=combined_summaries,
                objective='Unregularized squared GW loss; no entropy term or factor of one half.',
                interpretation='Endpoint feasible GW loss and independently selected normalized MSBPG displacement are distinct. The latter uses the paper step alpha, not KL eta; it is not a KKT or exact-proximal certificate. Finite inner tolerance is not covered by the exact-proximal theorem.',
                summary_policy='Median and linear-interpolation IQR across all configured final seeds; no complete summary for missing/failed populations.',
                trajectory_policy='Observed current rounded outputs only, without interpolation, extrapolation, time filling, or best-checkpoint selection. The final endpoint uses its FP64 validated loss.')


def figures(report, root):
    """Optional figures contain actual observations, not time-aligned imputations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return dict(status='unavailable',reason='matplotlib is not installed',files=[])
    valid=[r for r in report['runs'] if r['phase']=='final' and r['status']=='completed']
    files=[]
    for dataset,start in sorted({(r['dataset'],r['initialization']) for r in valid}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
        for method,color,marker in (('smoothing-fw','#1565c0','o'),('published-smd','#d84315','x'),('published-msbpg','#388e3c','+')):
            runs=[r for r in valid if r['dataset']==dataset and r['initialization']==start and r['method']==method]
            for i,run in enumerate(runs):
                points=run['trajectory']
                for axis,key in zip(axes,('iteration','time_to_this_feasible_output')):
                    axis.scatter([p[key] for p in points],[p['rounded_loss'] for p in points],color=color,marker=marker,alpha=.6,s=17,
                                 label=f"{LABELS[method]} ({len(runs)} seeds)" if i==0 else None)
        axes[0].set_xlabel('Completed outer updates'); axes[1].set_xlabel('Actual solver + current rounding time (s)')
        for axis in axes:
            axis.set_ylabel('Current rounded GW loss'); axis.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{dataset} · {start} initialization · observed points')
        folder=Path(root)/'figures'; folder.mkdir(exist_ok=True)
        for suffix in ('svg','png'):
            path=folder/f'{dataset}_{start}.{suffix}'
            fig.savefig(path,dpi=150); files.append(str(path))
        plt.close(fig)
    return dict(status='generated',files=files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    report=audit(args.config,args.out)
    args.out.mkdir(parents=True,exist_ok=True)
    report['figures']=dict(status='disabled',files=[]) if args.no_plots else figures(report,args.out)
    (args.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    fields=['dataset','phase','method','candidate_id','initialization','seed','status','last_rounded_loss','selected_raw_loss',
            'sampled_normalized_displacement_squared','frozen_reference','selected_index','selected_probability','completed_updates','stochastic_gradient_calls','sketch_probes',
            'inner_iterations','certificate_checks','solver_seconds','last_feasible_output_seconds','metadata_file','reason']
    with (args.out/'report.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(report['runs'])
    print(json.dumps(dict(status=report['status'],task_counts=report['task_counts'],report=str(args.out/'report.json'))))


if __name__=='__main__':
    main()
