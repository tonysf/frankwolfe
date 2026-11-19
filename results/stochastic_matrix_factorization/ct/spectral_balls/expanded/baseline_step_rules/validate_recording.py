"""Small recording, failure, selection and resume fixtures; no real CT jobs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np

import campaign
import run as engine


class TinyProblem:
    def __init__(self,data_file,grid,n_train):
        self.train_ids=np.arange(n_train);self.validation_ids=np.arange(n_train,n_train+2)
        self.reference=np.array([[.8,.2,.4],[.3,.7,.1],[.1,.4,.6]])
    def forward(self,image,ids):return (1+np.asarray(ids)[:,None]/100)*image.reshape(1,-1)
    def backward(self,values,ids):return np.sum((1+np.asarray(ids)[:,None]/100)*values,axis=0).reshape(3,3)
    def grad(self,U,V,ids,block=None):
        residual=self.forward(U@V.T-self.reference,ids);g=self.backward(residual,ids)/residual.size
        if block==0:return g@V
        if block==1:return g.T@U
        return g@V,g.T@U
    def image_loss(self,image,ids):return float(.5*np.mean(self.forward(image-self.reference,ids)**2))
    def close(self):pass


def fixtures():
    source_root=engine.ROOT;sources={name:engine.sha256(source_root/name) for name in engine.SOURCES}
    original_root=campaign.ROOT;original_run_plan=campaign.run_plan;original_update=campaign.update
    engine_reports=[]
    with tempfile.TemporaryDirectory(prefix='baseline-recording-') as directory:
        root=Path(directory)
        for name in engine.SOURCES:(root/name).symlink_to(source_root/name)
        (root/'data').mkdir();(root/'calibration').mkdir()
        data=root/'data/tiny.npz';np.savez(data,fixture=np.ones(1))
        U=np.array([[.6,.1],[.3,.5],[.2,.4]]);V=np.array([[.5,.2],[.2,.6],[.4,.3]])
        for seed in (110,201):
            np.savez(root/f'calibration/tiny_seed{seed}.npz',U=U,V=V,caps=np.array([2.,2.]),curvature=np.array([.2,.2]),
                     data_sha256=engine.sha256(data),problem_sha256=sources['ct_problem.py'])
        engine.write_json(root/'specification.json',dict(test='tiny recording fixture'))
        engine.write_json(root/'implementation_manifest.json',dict(current_source_sha256=sources,
            specification_sha256=engine.sha256(root/'specification.json'),authorized_changes=dict(test='recording fixture')))
        engine.ROOT=root;engine.CTProblem=TinyProblem;real_abort=engine.check_abort
        for method,cap in (('palm',250),('prox_sarah',33)):
            cfg=dict(run_id=method,phase='evaluation',method=method,variant=method,record_every_iteration=True,
                     slice_id='tiny',n_train=8,grid=3,rank=2,batch=8 if method=='palm' else 4,
                     calibration_seed=201,seed=202,view_budget=cap)
            if method=='prox_sarah':cfg.update(scale_multiplier=1.,scale_exponent=0)
            calls=0
            def interrupt_after_saved_step():
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('injected interruption')
            engine.check_abort=interrupt_after_saved_step
            try:engine.run(cfg,runs_dir=root/'resumed')
            except RuntimeError as exc:assert str(exc)=='injected interruption'
            else:raise AssertionError('Expected interruption')
            assert len(engine.read_json(root/'resumed'/method/'trace.json'))==2
            engine.check_abort=real_abort
            engine.run(cfg,runs_dir=root/'resumed');engine.run(cfg,runs_dir=root/'uninterrupted')
            trace=engine.read_json(root/'resumed'/method/'trace.json')
            assert trace==engine.read_json(root/'uninterrupted'/method/'trace.json')
            params=engine.read_json(root/'resumed'/method/'metadata.json')['optimizer']
            for row in trace:
                expected=campaign.expected_counts(method,8,row['iteration'],params)
                assert all(row[k]==v for k,v in expected.items())
                if row['iteration']:
                    step=row['last_step'];assert step['component_gradient_calls']==step['forward_views']-step['curvature_forward_views']
            assert trace[-2]['forward_views']<cap<=trace[-1]['forward_views']
            hashes={name:engine.sha256(root/'resumed'/method/name) for name in ('metadata.json','trace.json','result.json')}
            engine.run(cfg,runs_dir=root/'resumed')
            assert hashes=={name:engine.sha256(root/'resumed'/method/name) for name in hashes}
            if method=='prox_sarah':
                sparse=dict(cfg,run_id='sparse',phase='development',record_every_iteration=False,calibration_seed=110)
                engine.run(sparse,runs_dir=root/'sparse')
                st=engine.read_json(root/'sparse/sparse/trace.json')
                assert st==[trace[0],trace[-2],trace[-1]]
                maker=engine.prox_paper.make_solver
                def failing_factory(problem,U,V,caps,*,seed,config):
                    solver=maker(problem,U,V,caps,seed=seed,config=config)
                    def fail():raise FloatingPointError('injected scientific nonfinite')
                    solver.step=fail;return solver
                engine.prox_paper.make_solver=failing_factory
                failure=dict(sparse,run_id='scientific_failure')
                engine.run(failure,runs_dir=root/'failure')
                fr=engine.read_json(root/'failure/scientific_failure/result.json')
                assert fr['status']=='failed_scientific' and fr['final']['iteration']==0
                engine.prox_paper.make_solver=maker
            engine_reports.append(dict(method=method,resume_equals_uninterrupted=True,dense_observations=len(trace),
                                       final_views=trace[-1]['forward_views'],idempotent=True))
        campaign.ROOT=root/'selection_fixture';campaign.ROOT.mkdir()
        campaign.write(campaign.ROOT/'specification.json',campaign.specification())
        campaign.write(campaign.ROOT/'protocol.json',dict(test='selection fixture'))
        events=[];campaign.update=lambda *args,**kwargs:events.append((args,kwargs))
        def synthetic(plan,protocol):
            return [dict(run_id=c['run_id'],slice_id=c['slice_id'],n_train=c['n_train'],seed=c['seed'],
                scale_exponent=c['scale_exponent'],status='completed',relative_loss=.2+.01*(c['scale_exponent']+6)**2)
                for c in plan['runs']]
        campaign.run_plan=synthetic
        selected=campaign.develop({})
        assert selected['selected']['scale_exponent']==-6 and len(selected['history'])==3
        assert campaign.develop({})==selected
        cases=synthetic(campaign.dev_plan([-1,0],0),{})
        for case in cases:case['relative_loss']=.3
        _,winner=campaign.rank_candidates(cases,[-1,0]);assert winner['scale_exponent']==0
        cases[-1]['relative_loss']=2.
        rows,winner=campaign.rank_candidates(cases,[-1,0]);assert winner['scale_exponent']==-1
        assert next(r for r in rows if r['scale_exponent']==0)['failed_endpoint_count']==1
        cohort=campaign.read(campaign.OLD/'cohort.json')
        references=[c for c in campaign.read(campaign.OLD/'evaluation_plan.json')['runs'] if c['variant'] in ('palm','prox_sarah')]
        campaign.write(campaign.ROOT/'evaluation_template.json',dict(slices=cohort['evaluation_slices'],
            groups=cohort['evaluation_slice_groups'],original_references=references))
        plan=campaign.evaluation_plan(selected)
        assert len(plan['runs'])==len({c['run_id'] for c in plan['runs']})==168
        assert all(sum(c['method']==method for c in plan['runs'])==84 for method in ('palm','prox_sarah'))
        assert all(c['seed'] in (202,203,204) for c in plan['runs'])
        campaign.ROOT=root/'guard_fixture';campaign.ROOT.mkdir()
        campaign.write(campaign.ROOT/'specification.json',campaign.specification())
        campaign.write(campaign.ROOT/'protocol.json',dict(test='guard fixture'))
        def none_eligible(plan,protocol):
            records=synthetic(plan,protocol)
            for record in records:record['relative_loss']=2.
            return records
        campaign.run_plan=none_eligible
        assert campaign.develop({}) is None
        review=campaign.read(campaign.ROOT/'selection_review.json')
        assert max(review['tested_exponents'])==5 and review['status']=='needs_review'
        assert not (campaign.ROOT/'selection.json').exists()
    campaign.ROOT=original_root;campaign.run_plan=original_run_plan;campaign.update=original_update
    return dict(status='passed',source_sha256=sources,test_source_sha256=engine.sha256(Path(__file__)),
        recording=engine_reports,sparse_last_within_budget_exact=True,scientific_failure_retained=True,
        operational_interruption_raises=True,edge_extension_and_resume=True,ties_favor_larger_scale=True,
        single_bad_endpoint_not_masked=True,upper_guard_review=True,evaluation_physical_runs=168,
        no_real_ct_operator_or_production_job=True)


if __name__=='__main__':
    result=fixtures();campaign.write(Path(__file__).with_name('engine_validation.json'),result)
    print(json.dumps(result,indent=2))
