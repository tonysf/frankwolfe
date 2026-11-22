"""Scientific separation, selection, deduplication and artifact integrity."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.experiments.stochastic_ct import robustness_campaign as c


def fixture_inputs():
    names=('Mix1','Mix2','Mix3','Almond','Walnut','LavaStone')
    groups = {f'slice{100+i:05}': names[i%6] for i in range(14)}
    fresh = {f'slice{300+i:05}': names[i] for i in range(6)}
    cohort = dict(development_slices=['slice00001','slice00002'], evaluation_slices=list(groups),
        evaluation_slice_groups=groups, confirmation_slices=list(fresh), confirmation_slice_groups=fresh,
        panel_slices=list(groups)[:6])
    inputs = {}
    for sid in cohort['development_slices']+list(groups)+list(fresh):
        for rank in (32,64,128):
            for initial in ('standard','perturbed','random'):
                inputs[c.input_key(sid,rank,initial)] = dict(data_file='/data/'+sid+'.npz',data_sha256='a'*64,
                    calibration_file=f'/calibration/{sid}_{rank}_{initial}.npz',calibration_sha256='b'*64,
                    calibration_seed=110 if sid in cohort['development_slices'] else 201)
    return cohort,inputs


def selection_fixture():
    chosen = {family:{mode:dict(parameters=c.smoothing_parameters(family,4.,.1)) for mode in ('joint','loss')}
              for family in c.FAMILIES}
    chosen['prox_sarah']={'loss':dict(parameters=dict(curvature_scale=.25))}
    chosen['joint_lbfgsb']={'loss':dict(parameters=dict(c.BASELINE_FIXED,memory=20))}
    return dict(global_selection=chosen,per_view_selection={str(n):deepcopy(chosen) for n in (360,1080)})


def outcomes(plan, callback=None):
    result=[]
    for cfg in plan['runs']:
        values=dict(relative_loss=.2,relative_distance=.005)
        if callback is not None:
            values.update(callback(cfg))
        result.append(dict(status='completed',run_id=cfg['run_id'],n_train=cfg['n_train'],
            slice_id=cfg['slice_id'],seed=cfg['seed'],**values))
    return result


def test_screen_and_baseline_development_cover_only_declared_cases():
    cohort,inputs=fixture_inputs()
    plan=c.coarse_plan(cohort,inputs)
    assert plan['expected_runs']==300
    assert {x['slice_id'] for x in plan['runs']}==set(cohort['development_slices'])
    assert {x['n_train'] for x in plan['runs']}=={1080}
    assert {x['view_budget'] for x in plan['runs']}=={8640}
    assert {x['seed'] for x in plan['runs']}=={110}
    for family in c.FAMILIES:
        chosen=[x for x in plan['runs'] if x['variant']==family]
        assert len(chosen)==50
        assert {(x['parameters']['beta_scale'],x['parameters']['gamma0']) for x in chosen}=={
            (b,g) for b in c.BETA_GRID for g in c.GAMMA_GRID}
    baselines=c.baseline_plan(cohort,inputs)
    assert baselines['expected_runs']==116
    assert sum(x['method']=='joint_lbfgsb' for x in baselines['runs'])==20
    assert {x['parameters']['memory'] for x in baselines['runs'] if x['method']=='joint_lbfgsb'}=={5,10,20,40,80}


def test_shortlist_contains_distinct_loss_winner_and_deterministic_confirmation_once():
    cohort,inputs=fixture_inputs()
    plan=c.coarse_plan(cohort,inputs)
    def metrics(cfg):
        beta,gamma=cfg['parameters']['beta_scale'],cfg['parameters']['gamma0']
        if (beta,gamma)==(64.,.4):
            return dict(relative_loss=.01,relative_distance=.5)
        return dict(relative_loss=.2+gamma/10,relative_distance=.005+beta/100000)
    shortlist=c.choose_shortlist(outcomes(plan,metrics),plan)
    assert all(len(v)==4 for v in shortlist['choices'].values())
    assert all(any(p['beta_scale']==64 and p['gamma0']==.4 for p in v) for v in shortlist['choices'].values())
    confirmed=c.confirmation_plan(cohort,inputs,shortlist)
    assert confirmed['expected_runs']==224
    for family in c.FAMILIES:
        chosen=[x for x in confirmed['runs'] if x['variant']==family]
        assert {x['n_train'] for x in chosen}=={360,1080}
        assert len(chosen)==(16 if family in ('frames','sframes_full') else 48)


def test_joint_selection_uses_raw_feasibility_and_declared_fallback():
    rows=[dict(eligible=True,parameters=dict(beta_scale=b,gamma0=.1),loss=loss,
               distance=distance,joint_attainment=hit,target_ratio=ratio)
          for b,loss,distance,hit,ratio in [(1,.1,.02,0,2),(4,.2,.005,1,.67),(16,.15,.009,.5,.9)]]
    assert c.ordered(rows)[0]['parameters']['beta_scale']==4
    assert c.ordered(rows,'loss')[0]['parameters']['beta_scale']==1
    for row in rows:
        row['joint_attainment']=0
    assert c.ordered(rows)[0]['parameters']['beta_scale']==4
    rows[1]['eligible']=False
    assert c.ordered(rows)[0]['parameters']['beta_scale']==16
    with pytest.raises(ValueError,match='No eligible'):
        c.ordered([dict(eligible=False)])


def test_failed_candidate_never_becomes_a_partial_case_selection():
    cohort,inputs=fixture_inputs()
    plan=c.coarse_plan(cohort,inputs)
    cases=outcomes(plan)
    bad=cases[0]
    bad.update(status='invalid_candidate')
    scores=c.score_candidates(cases,plan)
    row=next(x for x in scores if bad['run_id'] in x['run_ids'])
    assert row['eligible'] is False and 'loss' not in row
    with pytest.raises(ValueError,match='every planned'):
        c.score_candidates(cases[:-1],plan)
    with pytest.raises(ValueError,match='Duplicate'):
        c.score_candidates(cases+[cases[0]],plan)
    wrong=deepcopy(plan)
    wrong['runs'][0]['phase']='evaluation'
    with pytest.raises(ValueError,match='Evaluation data'):
        c.score_candidates(cases,wrong)


def test_global_selection_balances_view_counts_and_reports_boundaries():
    cohort,inputs=fixture_inputs()
    shortlist=dict(choices={f:[c.smoothing_parameters(f,1.,.05),c.smoothing_parameters(f,4.,.1)] for f in c.FAMILIES})
    plan=c.confirmation_plan(cohort,inputs,shortlist)
    def metrics(cfg):
        low=cfg['parameters']['beta_scale']==1
        return dict(relative_loss=(.10 if low else .20) if cfg['n_train']==360 else (.28 if low else .12),
                    relative_distance=.001)
    baselines=c.baseline_plan(cohort,inputs)
    selection=c.select_parameters(outcomes(plan,metrics),plan,outcomes(baselines),baselines)
    assert selection['evaluation_data_used'] is False
    for family in c.FAMILIES:
        assert selection['global_selection'][family]['joint']['parameters']['beta_scale']==4
        assert selection['per_view_selection']['360'][family]['joint']['parameters']['beta_scale']==1
    assert c.boundary_flags(dict(beta_scale=.25,gamma0=.1,memory=80))==['beta_scale','memory']


def test_main_and_ablation_work_is_deduplicated_with_explicit_memberships():
    cohort,inputs=fixture_inputs()
    plan=c.evaluation_plan(cohort,inputs,selection_fixture())
    assert plan['expected_runs']==len({x['run_id'] for x in plan['runs']})
    assert plan['study_run_counts']['main']==14*4*(5+1+1+5+1+5)
    assert plan['study_run_counts']['fresh_confirmation']==6*2*(5+1+1+5+1+5)
    assert plan['study_run_counts']['stationarity']==8
    assert {cohort['evaluation_slice_groups'][sid] for sid in plan['stationarity_slices']}=={'Mix1','LavaStone'}
    assert plan['study_run_counts']['stress']==120
    main=[x for x in plan['runs'] if 'main' in x['studies']]
    assert {x['n_train'] for x in main}==set(c.VIEWS)
    for cfg in main:
        if cfg['method']=='sframes' and cfg['parameters']['batch']==64:
            assert {'sframes_joint','sframes_loss'}<=set(cfg['variants'])
    assert all(x['phase']=='evaluation' for x in plan['runs'])
    assert set(plan['studies']['main']).isdisjoint(plan['studies']['fresh_confirmation'])
    assert set(plan['studies']['main']) & set(plan['studies']['stress'])
    assert set(plan['studies']['main']) & set(plan['studies']['batch'])
    # Whole-run identity includes the actual starting file/domain, not an alias.
    initial=[x for x in plan['runs'] if 'initialization' in x['studies']]
    assert {x['initialization'] for x in initial}=={'standard','perturbed','random'}
    assert {x['radius_scale'] for x in initial}=={2}
    assert any(x['method']=='joint_lbfgsb' for x in initial)
    rank=[x for x in plan['runs'] if 'rank' in x['studies']]
    assert any(x['method']=='joint_lbfgsb' and x['rank']==128 for x in rank)
    radius=[x for x in plan['runs'] if 'radius' in x['studies']]
    assert {x['radius_scale'] for x in radius}=={1,2,4}
    assert all(x['method']!='joint_lbfgsb' for x in radius)


def test_condition_retuning_is_limited_and_excludes_identical_defaults():
    cohort,inputs=fixture_inputs()
    selection=selection_fixture()
    plan=c.evaluation_plan(cohort,inputs,selection)
    assert 'condition_retuning' not in plan['studies']
    selection['per_view_selection']['360']['frames']['joint']['parameters']['gamma0']=.2
    plan=c.evaluation_plan(cohort,inputs,selection)
    assert plan['study_run_counts']['condition_retuning']==6
    cases=[x for x in plan['runs'] if 'condition_retuning' in x['studies']]
    assert {x['n_train'] for x in cases}=={360}
    assert {x['method'] for x in cases}=={'frames'}
    assert {x['slice_id'] for x in cases}==set(cohort['panel_slices'])


def test_evaluation_validation_rejects_overshoot_and_preserves_candidate_failure(tmp_path):
    cohort,inputs=fixture_inputs()
    cfg=c.make_config(inputs,phase='development',study='test',variant='frames',sid='slice00001',n=360,
        method='frames',parameters=c.smoothing_parameters('frames',4,.1),cap=720)
    folder=tmp_path/'runs'/cfg['run_id'];folder.mkdir(parents=True)
    protocol=dict(numerical_sources={'engine.py':'a'})
    c.write(tmp_path/'protocol.json',protocol)
    meta=dict(config=cfg,cpu=0,source_sha256=protocol['numerical_sources'],initial_factor_frobenius_norm=2.,
        numpy='2.5.4',scipy='1.18.1',protocol_sha256=c.digest(tmp_path/'protocol.json'))
    trace=[dict(iteration=k,observation_type='initial' if k==0 else 'accepted_update',
        forward_views=360*k,backward_views=360*k,raw_loss=.1,feasible_loss=.1,d_distance=.01) for k in range(3)]
    c.write(folder/'metadata.json',meta);c.write(folder/'trace.json',trace)
    def result(status='completed'):
        return dict(status=status,metadata=meta,final=trace[-1],trace_sha256=c.digest(folder/'trace.json'),error='nonfinite test')
    c.write(folder/'result.json',result())
    case=c.validate_result(cfg,0,protocol,tmp_path)
    assert case['relative_loss']==1 and case['relative_distance']==.005
    trace[-1]['forward_views']=721;c.write(folder/'trace.json',trace);c.write(folder/'result.json',result())
    with pytest.raises(ValueError,match='cap exceeded'):
        c.validate_result(cfg,0,protocol,tmp_path)
    c.write(folder/'result.json',result('numerical_failure'))
    assert c.validate_result(cfg,0,protocol,tmp_path)['status']=='invalid_candidate'
    cfg['phase']='evaluation';meta['config']=cfg;c.write(folder/'metadata.json',meta);c.write(folder/'result.json',result('numerical_failure'))
    with pytest.raises(ValueError,match='evaluation requires review'):
        c.validate_result(cfg,0,protocol,tmp_path)


def test_frozen_records_reject_mutation(tmp_path):
    p=tmp_path/'record.json'
    c.frozen(p,dict(value=1));c.frozen(p,dict(value=1))
    with pytest.raises(ValueError,match='Frozen record'):
        c.frozen(p,dict(value=2))


def test_historical_reuse_preserves_old_sources_and_requires_its_verifier(tmp_path,monkeypatch):
    cohort,inputs=fixture_inputs()
    cfg=c.make_config(inputs,phase='development',study='test',variant='joint_lbfgsb',sid='slice00001',n=360,
        method='joint_lbfgsb',parameters=dict(c.BASELINE_FIXED,memory=20),cap=720)
    folder=tmp_path/'runs'/cfg['run_id'];folder.mkdir(parents=True)
    protocol=dict(numerical_sources={'new_engine.py':'new'},frozen_files={'historical_reuse_catalog.json':'catalog'})
    c.write(tmp_path/'protocol.json',protocol)
    old_sources={'joint_lbfgsb_solver.py':'old_executed_source'}
    meta=dict(config=cfg,cpu=0,source_sha256=old_sources,producer='historical_reuse',
        initial_factor_frobenius_norm=2.,numpy='2.5.4',scipy='1.18.1',
        protocol_sha256=c.digest(tmp_path/'protocol.json'))
    trace=[dict(iteration=k,observation_type='initial' if k==0 else 'accepted_update',
        forward_views=360*k,backward_views=360*k,raw_loss=.1,feasible_loss=.1,d_distance=0.) for k in range(3)]
    c.write(folder/'metadata.json',meta);c.write(folder/'trace.json',trace)
    def result():
        return dict(status='completed',metadata=meta,final=trace[-1],trace_sha256=c.digest(folder/'trace.json'))
    c.write(folder/'result.json',result())
    calls=[]
    helper=SimpleNamespace(verify_materialized=lambda *args,**kwargs:calls.append(args) or True)
    monkeypatch.setattr(c,'reuse_module',lambda root,protocol:helper)
    case=c.validate_result(cfg,0,protocol,tmp_path)
    assert case['producer']=='historical_reuse' and len(calls)==1
    assert c.read(folder/'metadata.json')['source_sha256']==old_sources
    # A verified historical record may honestly lack an unrecorded dependency
    # version. It must not be filled in as if the old execution had recorded it.
    meta.pop('scipy');meta['unknown_dependency_versions']=['scipy']
    c.write(folder/'metadata.json',meta);c.write(folder/'result.json',result())
    assert c.validate_result(cfg,0,protocol,tmp_path)['producer']=='historical_reuse'
    helper.verify_materialized=lambda *args,**kwargs:False
    with pytest.raises(ValueError,match='reuse evidence'):
        c.validate_result(cfg,0,protocol,tmp_path)
    meta.pop('producer');c.write(folder/'metadata.json',meta);c.write(folder/'result.json',result())
    with pytest.raises(ValueError,match='numerical sources'):
        c.validate_result(cfg,0,protocol,tmp_path)


def test_worker_materializes_declared_reuse_before_running_the_engine(tmp_path,monkeypatch):
    import sys
    protocol=dict(frozen_files={'historical_reuse_catalog.json':'frozen'})
    c.write(tmp_path/'protocol.json',protocol)
    monkeypatch.setattr(c.os,'sched_setaffinity',lambda pid,cpus:None)
    executed=[]
    monkeypatch.setitem(sys.modules,'robustness_run',SimpleNamespace(run=lambda *a,**k:executed.append(a) or {'status':'completed'}))
    helper=SimpleNamespace(try_materialize=lambda *a,**k:True)
    monkeypatch.setattr(c,'reuse_module',lambda root,protocol:helper)
    cfg={'run_id':'equivalent'}
    assert c.worker(cfg,0,tmp_path)['producer']=='historical_reuse'
    assert executed==[]
    helper.try_materialize=lambda *a,**k:False
    assert c.worker(cfg,0,tmp_path)=={'status':'completed'}
    assert len(executed)==1
    # A catalog that was not included in the protocol cannot enable reuse.
    c.write(tmp_path/'protocol.json',dict(frozen_files={}))
    helper.try_materialize=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unfrozen reuse attempted'))
    c.worker(cfg,0,tmp_path)
    assert len(executed)==2
