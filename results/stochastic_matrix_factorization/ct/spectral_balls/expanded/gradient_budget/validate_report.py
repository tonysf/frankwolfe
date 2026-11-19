"""Bounded synthetic continuation/report fixture; no optimizer runs or live writes."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gradient_report_fixture',ROOT/'update_report.py')
U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)


def main():
    stage=Path(tempfile.mkdtemp(prefix='report_fixture_',dir=ROOT))
    report=U.module('fixture_gradient_base',U.FEASIBILITY/'report.py')
    baseline=U.module('fixture_gradient_baseline',U.BASE/'update_report.py')
    prior=U.read(U.BASE/'report_update.json')
    previous=U.read(prior['corrected_summary']);joint=U.read(prior['corrected_joint_summary'])
    original=copy.deepcopy(previous);old_joint=copy.deepcopy(joint)
    live={p:U.digest(p) for p in [U.PAPER/report.BODY,U.PAPER/(report.STEM+'.pdf'),U.PAPER.parent/report.APPENDIX,
        *(folder/'report_update.json' for folder in [U.FEASIBILITY,U.SPRING,U.BASE,U.TRAJECTORIES])]}
    analysis,runs,slices,sizes,legacy,_=U.load_retained_runs(report,baseline)
    assert len(analysis.VARIANTS)==5 and len(analysis.plot_variants)==6
    extended={}
    for case in previous['cases']:
        for method in U.METHODS:
            old=case['arms'][method];folder=Path(old['source_directory'])
            meta=U.read(folder/'metadata.json');trace=U.read(folder/'trace.json')
            origin=copy.deepcopy(trace)
            n=case['n_train'];step=2*n if method=='palm' else 256;curv=12*n if method=='palm' else 768
            initial_gradient=U.gradient_views(trace[-1]);initial_metrics=trace[-1]
            while U.gradient_views(trace[-1])<U.CAP:
                row=copy.deepcopy(trace[-1]);row['iteration']+=1
                for direction in ('forward','backward'):
                    row[direction+'_views']+=step+curv
                    row['curvature_'+direction+'_views']+=curv
                    row['gradient_'+direction+'_views']=U.gradient_views(row,direction)
                row['gradient_calls']+=2 if method=='palm' else 4
                row['curvature_operator_calls']+=12
                row['oracle_calls']=row['gradient_calls']+row['curvature_operator_calls']
                fraction=min(1.,(U.gradient_views(row)-initial_gradient)/(U.CAP-initial_gradient))
                for field in ('raw_loss','feasible_loss','validation_loss','reference_relative_error'):
                    row[field]=initial_metrics[field]*(1-.4*fraction)
                trace.append(row)
            # Deliberately spectacular overshoot must never enter table/targets.
            if U.gradient_views(trace[-1])>U.CAP:
                for field in ('raw_loss','feasible_loss','validation_loss','reference_relative_error'):
                    trace[-1][field]=1e-10
            new_folder=stage/'synthetic_runs'/f"{case['slice_id']}_{n}_{case['seed']}_{method}"
            new_folder.mkdir(parents=True)
            cfg=dict(meta['config'],run_id=new_folder.name,gradient_view_budget=U.CAP)
            meta.update(config=cfg,continuation={'fixture_only':True,'source_directory':str(folder)})
            U.write(new_folder/'metadata.json',meta)
            U.write(new_folder/'trace.json',trace)
            extended[(*U.key(case),method)]=dict(config=cfg,meta=meta,trace=trace,source_directory=str(new_folder))
            assert trace[:len(origin)]==origin
    anchors={'previous_summary':{'path':prior['corrected_summary'],'sha256':prior['corrected_summary_sha256']},'fixture_only':True}
    updated=U.replace_summary(previous,extended,analysis,anchors)
    assert previous==original and joint==old_joint
    for before,after in zip(previous['cases'],updated['cases']):
        for method in before['arms']:
            if method not in U.METHODS:
                assert after['arms'][method]==before['arms'][method]
                continue
            arm=after['arms'][method];endpoint=arm['probes']['view_cap']
            assert endpoint['gradient_forward_views']<=U.CAP and endpoint['forward_views']>U.CAP
            assert endpoint['raw_loss']>1e-10
            assert arm['inclusive_budget_probes']==before['arms'][method]['probes']
            assert endpoint['forward_views']==endpoint['gradient_forward_views']+endpoint['curvature_forward_views']
    for n in ('360','1080'):
        assert updated['cohorts'][n]['view_cap']['sframes_vs']['prox_sarah']==previous['cohorts'][n]['view_cap']['sframes_vs']['prox_sarah']
        assert abs(updated['gradient_budget_extension']['attribution'][n]['palm']['total_to_gradient_ratio']['median_across_groups']-7)<1e-12
    # A target reached only in a discarded overshooting row remains censored.
    spring=next(run for k,run in extended.items() if k[3]=='spring_sarah')
    target={'relative_repaired_loss':1e-6,'max_relative_d_distance':.01}
    assert not U.target_hit(spring,target,analysis)['reached']
    source=U.source_text((U.PAPER/report.BODY).read_text(),updated,joint,report,legacy)
    assert 'equal gradient-view budgets' in source and 'curvature work is excluded from this cap' in source
    assert '13.4' in source and '21.2' in source
    assert source.count(r'\begin{figure}')==2 and source.count(r'\begin{table}')==1
    (stage/report.BODY).write_text(source)
    for name in report.ASSETS:
        if name!=report.BODY:shutil.copyfile(U.PAPER/name,stage/name)
    appendix=U.appendix_text((U.PAPER.parent/report.APPENDIX).read_text(),updated)
    (stage/report.APPENDIX).write_text(appendix)
    # The original upstream marked table remains stable under its publisher.
    chain=stage/'appendix_chain';(chain/'revised').mkdir(parents=True)
    (chain/report.APPENDIX).write_text(appendix)
    report.supplement_results(chain/'revised',U.read(U.FEASIBILITY/'selection.json'),U.read(U.FEASIBILITY/'summary.json'))
    assert (chain/report.APPENDIX).read_text()==appendix
    count,observations=U.render_plots(analysis,runs,extended,slices,sizes,analysis.group_map(previous['design']),stage)
    assert count==448
    for n in sizes:
        stats=U.read(stage/f'statistics_{n}.json')
        assert not stats['individual_trajectories_drawn'] and len(stats['methods'])==6
        for arm in stats['methods'].values():
            assert set(arm['axes'])=={'iteration','gradient_calls','gradient_forward_views'}
            assert arm['axes']['gradient_forward_views']['grid'][-1]<=U.CAP
    pages=baseline.compile_pdf(stage,report)
    U.write(stage/'gradient_comparison_summary.json',updated)
    U.chain_audits(stage,report,{'fixture_only':True,'publication_audit':str(stage/'report_update.json')},pages)
    for name,folder in [('feasibility_report_update.json',U.FEASIBILITY),('spring_report_update.json',U.SPRING),
                        ('baseline_report_update.json',U.BASE),('trajectory_report_update.json',U.TRAJECTORIES)]:
        old=U.read(folder/'report_update.json');new=U.read(stage/name)
        for key in old:
            if key not in ('pages','staged_sha256','published_sha256'):assert old[key]==new[key]
        assert all(U.digest(stage/name)==sha for name,sha in new['published_sha256'].items())
    # Exercise transaction rollback, interrupted mixed states and foreign edits
    # using three temporary files; production destinations are never touched.
    tx=stage/'transaction';payload=tx/'report_stage_fixture';backup=tx/'report_previous'
    payload.mkdir(parents=True);backup.mkdir()
    dest={name:tx/name for name in ('a','b','c')}
    for name,path in dest.items():
        path.write_text('old '+name);(backup/name).write_text('old '+name);(payload/name).write_text('new '+name)
    audit={'stage':str(payload),'backup':str(backup),'original_sha256':{n:U.digest(p) for n,p in dest.items()},
           'published_sha256':{n:U.digest(payload/n) for n in dest}}
    pending=tx/'report_update.pending.json';U.write(pending,audit)
    dest['a'].write_text('foreign')
    try:U.finish(tx,report,audit,dest,pending)
    except ValueError:pass
    else:raise AssertionError('Foreign edit accepted')
    dest['a'].write_text('old a')
    actual_copy=report.atomic_copy
    def fail_once(source,destination):
        if source==payload/'b':raise OSError('synthetic publication failure')
        actual_copy(source,destination)
    report.atomic_copy=fail_once
    try:U.finish(tx,report,audit,dest,pending)
    except OSError:pass
    else:raise AssertionError('Rollback fixture did not interrupt')
    report.atomic_copy=actual_copy
    assert all(U.digest(p)==audit['original_sha256'][n] for n,p in dest.items())
    actual_copy(payload/'a',dest['a'])
    U.finish(tx,report,audit,dest,pending)
    assert all(U.digest(p)==audit['published_sha256'][n] for n,p in dest.items())
    assert all(U.digest(p)==sha for p,sha in live.items())
    result=dict(status='passed',fixture_only=True,no_optimizer_runs=True,no_live_publication=True,
        stage=str(stage),pages=pages,figures=2,tables=1,physical_traces=count,native_observations=observations,
        updater_sha256=U.digest(ROOT/'update_report.py'),validator_sha256=U.digest(Path(__file__)),
        plotting_sha256=U.digest(ROOT/'plot_trajectories.py'),checks=['real retained-cohort loader with five summary/six plot arms',
        'gradient cap excludes overshoot','actual total/curvature counters retained','old inclusive probes preserved',
        'unchanged SFRAMES and ProxSARAH comparisons','gradient first-hit censoring','two mean/SD figures and one table compiled',
        'predecessor audit chain and historical appendix replay','foreign edit rejected','rollback and mixed-state recovery'])
    U.write(ROOT/'report_fixture_validation.json',result)
    print(result)


if __name__=='__main__':main()
