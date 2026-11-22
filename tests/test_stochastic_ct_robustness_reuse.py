"""Historical imports preserve dense evidence and reject changed provenance."""

from copy import deepcopy
import json

import pytest

from paper.experiments.stochastic_ct import robustness_reuse as r


def fixture(tmp_path, method='sframes', overshoot=False):
    root, origin = tmp_path/'new',tmp_path/'old'
    root.mkdir()
    origin.mkdir()
    data,calibration = tmp_path/'data.npz',tmp_path/'calibration.npz'
    data.write_bytes(b'fixed data')
    calibration.write_bytes(b'fixed initialization and caps')
    source = tmp_path/'old_solver.py'
    source.write_text('fixed numerical source\n')
    cfg=dict(run_id='new_case',phase='evaluation',slice_id='slice00601',method=method,
        n_train=360,grid=512,rank=64,seed=202,calibration_seed=201,view_budget=128,
        record_every_iteration=True,record_stationarity=False,initialization='standard',radius_scale=2.,
        data_file=str(data),data_sha256=r.digest(data),calibration_file=str(calibration),
        calibration_sha256=r.digest(calibration))
    cfg['parameters'] = dict(batch=64,beta_scale=4.,gamma0=.1,rho0=1.)
    oldcfg = deepcopy(cfg)
    oldcfg.update(beta_scale=4.,rate=.1,scale_multiplier=.25)
    optimizer = dict(batch=64,gamma0=.1,rho0=1.,mu0=3.)
    if method == 'prox_sarah':
        cfg['parameters'] = dict(curvature_scale=.25)
        optimizer = dict(paper_variant='ProxSARAH-v3')
        cfg['view_budget'] = oldcfg['view_budget'] = 127 if overshoot else 128
    metadata=dict(config=oldcfg,optimizer=optimizer,initial_factor_frobenius_norm=2.,
        source_sha256={'old_solver.py':r.digest(source)},numpy='2.5.4',astra='2.5.0',
        data_sha256=cfg['data_sha256'],calibration_sha256=cfg['calibration_sha256'],caps=[3.,4.])
    trace=[dict(iteration=i,forward_views=64*i,backward_views=64*i,gradient_calls=i,
                raw_loss=1/(i+1),feasible_loss=1/(i+1),validation_loss=.7,
                d_distance=.01*i) for i in range(3)]
    r.write(origin/'metadata.json',metadata)
    r.write(origin/'trace.json',trace)
    r.write(origin/'result.json',dict(status='completed',metadata=metadata,final=trace[-1]))
    entry=dict(signature=r._signature(oldcfg,metadata),stage='fixture',origin_directory=str(origin),
        original_sources={'old_solver.py':dict(path=str(source),sha256=r.digest(source))},
        artifacts={name:r.digest(origin/name) for name in ('metadata.json','result.json','trace.json')},
        excluded_overshoot_rows=[2] if overshoot else [])
    r.write(root/r.CATALOG,dict(adapter_sha256=r.digest(r.__file__),entries=[entry],target_native_sources={}))
    protocol=dict(frozen_files={r.CATALOG:r.digest(root/r.CATALOG)})
    r.write(root/'protocol.json',protocol)
    return root,origin,cfg,protocol


def test_dense_import_retains_original_evidence_and_unknown_counters(tmp_path):
    root,origin,cfg,protocol=fixture(tmp_path)
    before={p.name:p.read_bytes() for p in origin.iterdir()}
    assert r.try_materialize(cfg,2,root,protocol)
    assert r.verify_materialized(cfg,2,root,protocol)
    folder=root/'runs'/cfg['run_id']
    meta,trace=r.read(folder/'metadata.json'),r.read(folder/'trace.json')
    assert meta['producer']=='historical_reuse'
    assert meta['historical_reuse']['numerical_engine_executed'] is False
    assert meta['historical_reuse']['unknown_dependency_versions']==['scipy']
    assert 'scipy' not in meta
    assert meta['source_sha256']==r.read(origin/'metadata.json')['source_sha256']
    assert [row['observation_type'] for row in trace]==['initial','accepted_update','accepted_update']
    assert trace[-1]['gradient_forward_views']==128
    assert 'diagnostic_forward_views' not in trace[-1]
    assert 'factor_u_sha256' not in trace[-1]
    assert all((origin/name).read_bytes()==content for name,content in before.items())
    assert all((folder/'historical_origin'/name).read_bytes()==content for name,content in before.items())
    assert not (folder/'checkpoint.pkl').exists()


@pytest.mark.parametrize('change',[
    dict(seed=203),dict(rank=32),dict(radius_scale=1.),dict(n_train=1080),
    dict(initialization='perturbed'),dict(view_budget=256),dict(record_stationarity=True),
    dict(parameters=dict(batch=64,beta_scale=16.,gamma0=.1,rho0=1.)),
    dict(data_sha256='0'*64),dict(calibration_seed=999),dict(phase='development')])
def test_scientific_mismatch_is_not_reused(tmp_path,change):
    root,_,cfg,protocol=fixture(tmp_path)
    assert r.try_materialize(cfg|change,2,root,protocol) is False
    assert not (root/'runs').exists()


def test_original_evidence_mutation_rejected(tmp_path):
    root,origin,cfg,protocol=fixture(tmp_path)
    assert r.try_materialize(cfg,2,root,protocol)
    (origin/'trace.json').write_text('[]')
    with pytest.raises(ValueError,match='Original historical evidence changed'):
        r.verify_materialized(cfg,2,root,protocol)


def test_input_source_catalog_and_materialized_mutations_rejected(tmp_path):
    root,_,cfg,protocol=fixture(tmp_path)
    assert r.try_materialize(cfg,2,root,protocol)
    folder=root/'runs'/cfg['run_id']
    meta=r.read(folder/'metadata.json')
    meta['source_sha256']={'pretend_new_runner':'bad'}
    r.write(folder/'metadata.json',meta)
    with pytest.raises(ValueError,match='Imported metadata changed'):
        r.verify_materialized(cfg,2,root,protocol)
    (tmp_path/'data.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Reuse input bytes changed'):
        r.verify_materialized(cfg,2,root,protocol)


def test_one_overshoot_is_trimmed_without_fabricating_an_iterate(tmp_path):
    root,origin,cfg,protocol=fixture(tmp_path,method='prox_sarah',overshoot=True)
    assert r.try_materialize(cfg,0,root,protocol)
    trace=r.read(root/'runs'/cfg['run_id']/'trace.json')
    assert [x['iteration'] for x in trace]==[0,1]
    assert trace[-1]['gradient_forward_views']==64
    assert len(r.read(origin/'trace.json'))==3
    assert r.verify_materialized(cfg,0,root,protocol)


def test_sparse_trace_and_unaudited_overshoot_rejected(tmp_path):
    _,origin,_,_=fixture(tmp_path)
    meta,trace=r.read(origin/'metadata.json'),r.read(origin/'trace.json')
    with pytest.raises(ValueError,match='not dense'):
        r._normalize([trace[0],trace[-1]],meta,128)
    with pytest.raises(ValueError,match='one-update overshoot'):
        r._normalize(trace,meta,127)
    with pytest.raises(ValueError,match='stopped early'):
        r._normalize(trace,meta,256)


def test_stale_catalog_or_source_bytes_are_rejected(tmp_path):
    root,_,cfg,protocol=fixture(tmp_path)
    catalog=r.read(root/r.CATALOG)
    catalog['entries'][0]['original_sources']['old_solver.py']['sha256']='0'*64
    r.write(root/r.CATALOG,catalog)
    with pytest.raises(ValueError,match='catalog changed'):
        r.try_materialize(cfg,0,root,protocol)
    protocol['frozen_files'][r.CATALOG]=r.digest(root/r.CATALOG)
    with pytest.raises(ValueError,match='executed numerical source changed'):
        r.try_materialize(cfg,0,root,protocol)


def test_scipy_and_astra_versions_are_not_silently_filled(tmp_path):
    root,origin,cfg,protocol=fixture(tmp_path)
    assert r.try_materialize(cfg,0,root,protocol)
    meta=r.read(root/'runs'/cfg['run_id']/'metadata.json')
    assert {k:meta.get(k) for k in ('numpy','scipy','astra')} == {
        k:r.read(origin/'metadata.json').get(k) for k in ('numpy','scipy','astra')}
