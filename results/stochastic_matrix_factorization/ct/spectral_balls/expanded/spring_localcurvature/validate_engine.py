"""Bounded recording/resume test on a tiny linear problem; never touches real runs."""
import json
from pathlib import Path
import tempfile

import numpy as np

import run as engine


class TinyProblem:
    def __init__(self,data_file,grid,n_train):
        self.train_ids=np.arange(n_train)
        self.validation_ids=np.arange(n_train,n_train+2)
        self.reference=np.array([[.8,.2,.4],[.3,.7,.1],[.1,.4,.6]])

    def forward(self,image,ids):
        return (1+np.asarray(ids)[:,None]/100)*image.reshape(1,-1)

    def backward(self,values,ids):
        return np.sum((1+np.asarray(ids)[:,None]/100)*values,axis=0).reshape(3,3)

    def grad(self,U,V,ids,block=None):
        residual=self.forward(U@V.T-self.reference,ids)
        image_grad=self.backward(residual,ids)/residual.size
        if block==0:return image_grad@V
        if block==1:return image_grad.T@U
        return image_grad@V,image_grad.T@U

    def image_loss(self,image,ids):
        residual=self.forward(image-self.reference,ids)
        return float(.5*np.mean(residual**2))

    def close(self):
        pass


def validate():
    source_root=engine.ROOT
    sources={name:engine.sha256(source_root/name) for name in engine.SOURCES}
    with tempfile.TemporaryDirectory(prefix='spring-engine-test-') as directory:
        root=Path(directory)
        for name in engine.SOURCES:(root/name).symlink_to(source_root/name)
        (root/'data').mkdir();(root/'calibration').mkdir()
        data=root/'data'/'tiny.npz';np.savez(data,fixture=np.ones(1))
        U=np.array([[.6,.1],[.3,.5],[.2,.4]])
        V=np.array([[.5,.2],[.2,.6],[.4,.3]])
        np.savez(root/'calibration'/'tiny_seed201.npz',U=U,V=V,caps=np.array([2.,2.]),
                 curvature=np.array([.2,.2]),data_sha256=engine.sha256(data),
                 problem_sha256=sources['ct_problem.py'])
        engine.write_json(root/'specification.json',dict(test='tiny recording fixture'))
        engine.write_json(root/'implementation_manifest.json',dict(current_source_sha256=sources,
            specification_sha256=engine.sha256(root/'specification.json'),
            authorized_changes=dict(fixture='Tiny linear recording and interruption test')))
        cfg=dict(run_id='tiny',phase='evaluation',method='spring_sarah',variant='spring_sarah',
                 record_every_iteration=True,slice_id='tiny',n_train=8,grid=3,rank=2,batch=4,
                 calibration_seed=201,seed=202,view_budget=210,
                 optimizer=dict(batch=4,refresh_period=16,power_iterations=5,step_coefficient=.5))
        engine.ROOT=root;engine.CTProblem=TinyProblem
        real_abort=engine.check_abort
        checks=0
        def interrupt_after_saved_update():
            nonlocal checks
            checks+=1
            if checks==3:raise RuntimeError('injected interruption after first saved update')
        engine.check_abort=interrupt_after_saved_update
        try:
            engine.run(cfg,runs_dir=root/'resumed')
        except RuntimeError as exc:
            assert str(exc).startswith('injected interruption')
        else:
            raise AssertionError('Expected bounded interruption did not happen')
        partial=engine.read_json(root/'resumed'/'tiny'/'trace.json')
        assert [r['iteration'] for r in partial]==[0,1]
        engine.check_abort=real_abort
        engine.run(cfg,runs_dir=root/'resumed')
        engine.run(cfg,runs_dir=root/'uninterrupted')
        resumed=engine.read_json(root/'resumed'/'tiny'/'trace.json')
        uninterrupted=engine.read_json(root/'uninterrupted'/'tiny'/'trace.json')
        assert resumed==uninterrupted
        assert [r['iteration'] for r in resumed]==list(range(len(resumed)))
        assert resumed[-2]['forward_views']<cfg['view_budget']<=resumed[-1]['forward_views']
        for row in resumed:
            assert row['forward_views']==row['backward_views']
            assert row['oracle_calls']==row['gradient_calls']+row['curvature_operator_calls']
            assert row['curvature_operator_calls']==12*row['iteration']
            assert row['curvature_forward_views']==48*row['iteration']
            if row['iteration']:
                assert row['last_step']['curvature_sample_views_u']==4
                assert row['last_step']['curvature_sample_views_v']==4
                step=row['last_step']
                assert step['component_gradient_calls']==step['forward_views']-step['curvature_forward_views']
        result=root/'resumed'/'tiny'/'result.json'
        before=engine.sha256(result)
        engine.run(cfg,runs_dir=root/'resumed')
        assert engine.sha256(result)==before
        return dict(status='passed',source_sha256=sources,
            test_source_sha256=engine.sha256(Path(__file__)),fixture='3x3 image, rank2,8 views,4-view curvature',
            observations=len(resumed),final_forward_views=resumed[-1]['forward_views'],
            interrupted_resume_equals_uninterrupted=True,all_native_steps_recorded=True,
            inclusive_work_counters=True,first_crossing_retained=True,completed_rerun_idempotent=True,
            production_runs_created=False)


if __name__=='__main__':
    result=validate()
    engine.write_json(Path(__file__).with_name('engine_validation.json'),result)
    print(json.dumps(result,indent=2))
