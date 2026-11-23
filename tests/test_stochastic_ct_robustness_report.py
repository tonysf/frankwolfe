"""Scientific summaries, deduplicated labels and compact CT paper rendering."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from paper.experiments.stochastic_ct import robustness_report as report


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def selection_fixture():
    chosen={family:{mode:dict(parameters=dict(batch=batch,beta_scale=4.,gamma0=.1,rho0=1.))
                    for mode in ('joint','loss')}
            for family,batch in [('sframes_b64',64),('sframes_full','full'),('frames','full')]}
    chosen['prox_sarah']={'loss':dict(parameters=dict(curvature_scale=.25))}
    chosen['joint_lbfgsb']={'loss':dict(parameters=dict(memory=20))}
    return dict(global_selection=chosen)


def save_run(root,cfg,*,scale=1.,status='completed'):
    method=cfg['method'];n=cfg['n_train'];seed=cfg['seed']
    initial=.04*scale
    nonnegative=method in ('prox_sarah','spring_sarah','joint_lbfgsb')
    variation=1+.001*(seed%10)
    endpoint=.006*scale*variation
    trace=[]
    for k,work,fraction,kind in [(0,0,1.,'initial'),(1,n,.7,'accepted_update'),
                                (2,2*n,.15,'accepted_update'),(2,69120,.15,'budget_stop')]:
        raw=initial if k==0 else (initial*.7 if k==1 else endpoint)
        distance=0 if k==0 or nonnegative else .01*scale
        curvature=6*work if method=='spring_sarah' else 0
        trace.append(dict(iteration=k,observation_type=kind,raw_loss=raw,
            feasible_loss=raw if nonnegative or k==0 else raw+distance*.001,
            validation_loss=raw*1.03,reference_relative_error=.7 if k==0 else .3*scale,
            d_distance=distance,gradient_forward_views=work,gradient_backward_views=work,
            forward_views=work+curvature,backward_views=work+curvature,
            gradient_calls=work//n,oracle_calls=(work+curvature)//n,
            curvature_forward_views=curvature,curvature_backward_views=curvature))
    meta=dict(config=cfg,initial_factor_frobenius_norm=2.,source_sha256={'fixture.py':'fixture'})
    folder=root/'runs'/cfg['run_id']
    write(folder/'metadata.json',meta);write(folder/'trace.json',trace)
    write(folder/'result.json',dict(status=status,metadata=meta,final=trace[-1],
        trace_sha256=hashlib.sha256((folder/'trace.json').read_bytes()).hexdigest(),
        error=None if status=='completed' else 'fixture numerical failure'))
    return trace


def make_main_fixture(root):
    """Four view counts, fourteen slices, seven labels and five stochastic seeds."""
    groups=('Mix1','Mix2','Mix3','Almond','Walnut','LavaStone')
    slices=[f'slice{i:05}' for i in range(101,115)]
    cohort=dict(evaluation_slices=slices,evaluation_slice_groups={sid:groups[i%6] for i,sid in enumerate(slices)},
        confirmation_slices=[],confirmation_slice_groups={},development_slices=['slice00001','slice00002'])
    write(root/'cohort.json',cohort)
    physical=[('sframes',64,['sframes_joint','sframes_loss']),
              ('sframes','full',['sframes_full']),('frames','full',['frames']),
              ('prox_sarah',None,['prox_sarah']),('joint_lbfgsb',None,['joint_lbfgsb']),
              ('spring_sarah',None,['spring_sarah'])]
    configs=[]
    for si,sid in enumerate(slices):
        for n in (120,360,1080,2160):
            for mi,(method,batch,labels) in enumerate(physical):
                deterministic=method in ('frames','joint_lbfgsb') or batch=='full'
                for seed in ((201,) if deterministic else range(202,207)):
                    params={'batch':batch,'beta_scale':4.,'gamma0':.1} if batch is not None else (
                        {'curvature_scale':.25} if method=='prox_sarah' else
                        {'memory':20} if method=='joint_lbfgsb' else {'batch':64,'refresh_period':16})
                    cfg=dict(run_id=f'{sid}_{n}_{labels[0]}_{seed}',phase='evaluation',study='main',
                        variant=labels[0],studies=['main'],variants=labels,
                        memberships=[dict(study='main',variant=label) for label in labels],
                        slice_id=sid,n_train=n,rank=64,grid=512,seed=seed,method=method,
                        radius_scale=2,initialization='standard',parameters=params,view_budget=69120)
                    configs.append(cfg)
                    save_run(root,cfg,scale=1+.015*si+.03*mi+.00001*n)
    write(root/'plans/main.json',dict(stage='evaluation',runs=configs))
    write(root/'selection.json',selection_fixture())
    return cohort,configs


@pytest.fixture(scope='module')
def main_fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('robustness_report')
    cohort,configs=make_main_fixture(root)
    return root,cohort,configs,report.summarize(root,'fixture')


def test_main_fixture_expands_aliases_without_inventing_physical_runs(main_fixture):
    root,cohort,configs,summary=main_fixture
    assert len(configs)==1008
    assert summary['planned_physical_runs']==summary['completed_physical_runs']==1008
    assert not summary['pending_runs'] and not summary['failed_runs']
    cells=report.main_cells(summary)
    assert len(cells)==28
    for n in (120,360,1080,2160):
        joint,loss=cells[n,'sframes_joint'],cells[n,'sframes_loss']
        assert joint['run_ids']==loss['run_ids']
        assert joint['physical_runs']==loss['physical_runs']==70
        assert joint['metrics']==loss['metrics']
        for variant in ('frames','sframes_full','joint_lbfgsb'):
            assert cells[n,variant]['physical_runs']==14
        for variant in ('prox_sarah','spring_sarah'):
            assert cells[n,variant]['physical_runs']==70
    row=report.read_run(root,configs[0])['trace'][-1]
    assert row['comparison_views']==69120
    assert row['relative_d_distance']==row['d_distance']/2


def test_tuning_settings_are_separate_cells(tmp_path):
    write(tmp_path/'cohort.json',dict(evaluation_slice_groups={},confirmation_slice_groups={},development_slices=['slice00001']))
    configs=[]
    for index,beta in enumerate((1.,4.)):
        cfg=dict(run_id=f'tune{index}',phase='development',study='coarse',variant='sframes_b64',
            slice_id='slice00001',n_train=1080,rank=64,seed=110,method='sframes',radius_scale=2,
            initialization='standard',parameters=dict(batch=64,beta_scale=beta,gamma0=.1),view_budget=69120)
        configs.append(cfg);save_run(tmp_path,cfg,scale=1+index)
    write(tmp_path/'plans/coarse.json',dict(runs=configs))
    summary=report.summarize(tmp_path)
    cells=summary['studies']['coarse']
    assert len(cells)==2 and {cell['parameters']['beta_scale'] for cell in cells}=={1.,4.}
    assert all(cell['physical_runs']==1 for cell in cells)
    assert len({cell['metrics']['feasible_loss']['median_across_groups'] for cell in cells})==2


def test_paired_changes_are_formed_per_slice_then_per_group():
    groups={'s1':'A','s2':'A','s3':'A','s4':'B'}
    left={'s1':2.,'s2':4.,'s3':9.,'s4':1.}
    right={'s1':1.,'s2':8.,'s3':3.,'s4':2.}
    summary=report.paired_groups(left,right,groups,bootstrap_seed=7)
    assert summary['by_group']=={'A':100.,'B':-50.}
    assert summary['median_percent_difference']==25.
    assert summary['groups']==2 and summary['groups_improved']==1
    assert summary['bootstrap_95_percentile']==[-50.,100.]
    assert summary['resampling_unit']=='acquisition group'
    # Duplicating each slice in A should not increase the group's bootstrap weight.
    copies={sid+'_copy':group for sid,group in groups.items() if group=='A'}
    duplicate=report.paired_groups(left|{sid+'_copy':left[sid] for sid in ('s1','s2','s3')},
        right|{sid+'_copy':right[sid] for sid in ('s1','s2','s3')},groups|copies,bootstrap_seed=7)
    assert duplicate==summary
    with pytest.raises(ValueError,match='positive common'):
        report.paired_groups(left,{**right,'s4':0},groups)


def test_work_axis_uses_last_observed_iterate_without_interpolation():
    runs={}
    for sid,x,y in [('a',[0,2,4],[1.,.5,.25]),('b',[0,1,4],[1.,.75,.5])]:
        runs[(sid,1,360,'method')]=dict(trace=[dict(iteration=k,comparison_views=work,raw_loss=value)
            for k,(work,value) in enumerate(zip(x,y))])
    grid,mean,sd=report.plotting.summarize(runs,['a','b'],{'a':'A','b':'B'},360,'method','raw_loss','comparison_views')
    assert grid.tolist()==[0,1,2,4]
    assert np.allclose(mean,[1.,.875,.625,.375])
    assert np.allclose(sd,[0.,.125,.125,.125])


def test_document_preserves_estimator_and_has_no_obsolete_domain_claim(main_fixture):
    root,_,_,summary=main_fixture
    source=(Path(__file__).parent/'fixtures/stochastic_ct_pre_compact.tex').read_text()
    body=report.paper_body(source,summary,selection_fixture())
    estimator=re.search(r'\\paragraph\{Stochastic gradient estimator\.\}.*?(?=\\paragraph\{Evaluation\.\})',source,re.S).group()
    assert estimator in body
    assert body.count(r'\begin{figure}')==2 and body.count(r'\begin{table}')==1
    assert 'These three additional baselines' not in body
    assert 'NNLS' not in body and 'multiplicative' not in body.lower()
    assert 'Joint L-BFGS-B minimizes the same loss' in body
    assert 'not paired percentage changes' in body
    assert 'exact current gradients without averaging' in body
    assert r'N\in\{120,360,1080,2160\}' in body


def test_main_cells_rejects_duplicate_parameter_settings(main_fixture):
    broken=deepcopy(main_fixture[3])
    duplicate=deepcopy(broken['studies']['main'][0])
    duplicate['parameters']['gamma0']=.2
    broken['studies']['main'].append(duplicate)
    with pytest.raises(ValueError,match='duplicate'):
        report.main_cells(broken)


def test_development_failure_receipt_is_an_outcome_not_pending(tmp_path):
    cfg=dict(run_id='failure',phase='development',study='coarse',variant='sframes_b64',
        method='sframes',slice_id='slice00001',n_train=1080,rank=64,seed=110,
        parameters=dict(batch=64,beta_scale=1.,gamma0=.1),view_budget=8640)
    write(tmp_path/'cohort.json',dict(evaluation_slice_groups={},development_slices=['slice00001']))
    write(tmp_path/'plans/coarse.json',dict(runs=[cfg]))
    write(tmp_path/'runs/failure/candidate_failure.json',dict(status='invalid_candidate',config=cfg,error='nonfinite arithmetic'))
    summary=report.summarize(tmp_path)
    assert summary['pending_runs']==[]
    assert len(summary['failed_runs'])==1 and summary['failed_runs'][0]['run_id']=='failure'


def test_unknown_result_status_is_not_silently_treated_as_a_tuning_failure(tmp_path):
    cfg=dict(run_id='failed',phase='development',study='coarse',variant='sframes_b64',method='sframes',
        slice_id='slice00001',n_train=1080,rank=64,seed=110,parameters={},view_budget=69120)
    save_run(tmp_path,cfg,status='operational_failure')
    with pytest.raises(ValueError,match='status|operational'):
        report.read_run(tmp_path,cfg)
