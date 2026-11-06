"""Supplement provenance/coverage tests with in-memory manifests and no data access."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import prepare_momentum_ablation as m
import prepare_stage_c_campaign as g


@pytest.fixture
def design(monkeypatch):
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('A supplemental generator must not open a dataset.'))
    campaign=json.loads(g.CAMPAIGN_PATH.read_text())
    supplement=json.loads(m.CONFIG_PATH.read_text())
    tuning_path=g.ROOT/'configs/stage_c_manifests/tuning.json'
    tuning=json.loads(tuning_path.read_text())
    catalog=g.candidate_catalog(campaign)
    selection=dict(schema_version=1,campaign_config_sha256=g.sha256_file(g.CAMPAIGN_PATH),
        tuning_manifest_sha256=g.sha256_file(tuning_path),selection_source='tuning phase only; fixture, not actual results',
        selections={method:dict(status='selected',candidate_id='c00',score=.5) for method in g.TUNABLE})
    selection['selections']['stochastic-kl']['candidate_id']='c07'
    selection['selections']['smoothing-fw']['candidate_id']='c01'
    selection['candidate_scores']=[]
    for method,record in selection['selections'].items():
        count=sum(r['method']==method and r['candidate_id']==record['candidate_id'] for r in tuning['runs'])
        selection['candidate_scores'].append(dict(method=method,candidate_id=record['candidate_id'],eligible=True,
            valid_runs=count,prescribed_runs=count,disqualifying_run_ids=[],score=record['score']))
    hashes=dict(campaign_hash=selection['campaign_config_sha256'],tuning_hash=selection['tuning_manifest_sha256'])
    choices=m.resolve_selection(campaign,tuning,selection,supplement,**hashes)
    class FakeHashes:
        def metadata(self,description):
            return dict(instance_sha256='a'*64,expected_array_sha256={k:'b'*64 for k in ('A','B','a','b')},product_loss_float64=.75)
    panels=g.make_finals(campaign,choices,FakeHashes())
    bindings=dict(campaign_config_sha256=hashes['campaign_hash'],tuning_manifest_sha256=hashes['tuning_hash'],
        selection_sha256='c'*64,selection_path='/fixture/selection.json',tuning_manifest_path=str(tuning_path),
        supplemental_config_path=str(m.CONFIG_PATH),supplemental_config_sha256=g.sha256_file(m.CONFIG_PATH),
        base_manifest_path={p:'/fixture/'+p+'.json' for p in ('final_product','final_randomized')},
        base_manifest_sha256={p:'d'*64 for p in ('final_product','final_randomized')})
    docs={p:dict(schema_version=1,phase=p,campaign_config_sha256=hashes['campaign_hash'],
        tuning_manifest_sha256=hashes['tuning_hash'],selection_sha256=bindings['selection_sha256'],
        run_count=len(panels[p]),runs=panels[p]) for p in ('final_product','final_randomized')}
    return campaign,supplement,tuning,selection,choices,bindings,docs,hashes


def test_supplement_preserves_all_paired_cells_and_changes_only_momentum(design):
    campaign,supplement,_,_,choices,bindings,docs,_=design
    prior=copy.deepcopy(docs)
    generated=m.make_supplements(campaign,choices,docs,supplement,bindings)
    assert docs==prior
    assert set(generated)=={'momentum_product','momentum_randomized'}
    total=0
    for phase,doc in generated.items():
        assert doc['run_count']==55 and len({r['run_id'] for r in doc['runs']})==55
        assert doc['comparison_phase']==supplement['phases'][phase]['comparison_phase']
        assert doc['campaign_config_sha256']==doc['parent_campaign_config_sha256']==bindings['campaign_config_sha256']
        parents={r['run_id']:r for r in docs[doc['comparison_phase']]['runs']}
        for run in doc['runs']:
            parent=parents[run['parent_run_id']]
            for key in ('instance','instance_sha256','expected_array_sha256','product_loss_float64','n','m','family','problem_seed','noise',
                        'seed','init_seed','initialization','budget','checkpoints','deterministic'):
                assert run[key]==parent[key]
            assert run['method']==run['implementation']=='stochastic-kl-momentum'
            assert run['candidate_id']=='kl_c07_rho_c01'
            assert run['source_candidate_id']=='c07' and run['momentum_source_candidate_id']=='c01'
            assert run['parameters']==parent['parameters']|{'rho0':.3,'momentum_beta':.5}
            expected=parent['argv'].copy();expected[expected.index('--method')+1]='stochastic-kl-momentum';expected+=['--rho0','0.3']
            assert run['argv']==expected
        total+=doc['raw_solver_budget_seconds']
    assert total==18150
    randomized=generated['momentum_randomized']['runs']
    for instance in {r['instance'] for r in randomized}:
        assert {(r['seed'],r['init_seed']) for r in randomized if r['instance']==instance}=={(s,900000+s) for s in range(100,105)}


@pytest.mark.parametrize('key',['campaign_config_sha256','tuning_manifest_sha256','selection_sha256'])
def test_stale_base_manifest_is_rejected(design,key):
    campaign,supplement,_,_,choices,bindings,docs,_=design
    docs['final_product'][key]='0'*64
    with pytest.raises(ValueError,match='stale'):
        m.make_supplements(campaign,choices,docs,supplement,bindings)


@pytest.mark.parametrize('change', ['missing','seed','eta','hash'])
def test_missing_or_mismatched_base_cells_rejected(design,change):
    campaign,supplement,_,_,choices,bindings,docs,_=design
    run=next(r for r in docs['final_randomized']['runs'] if r['method']=='stochastic-kl')
    if change=='missing':docs['final_randomized']['runs'].remove(run)
    elif change=='seed':run['init_seed']=999
    elif change=='eta':run['parameters']['eta']=999
    else:run['instance_sha256']='missing'
    with pytest.raises(ValueError):m.make_supplements(campaign,choices,docs,supplement,bindings)


@pytest.mark.parametrize('change', ['unselected','ineligible','missing_trial','stale_campaign','stale_tuning','not_authoritative'])
def test_final_selection_gate_has_no_fallback(design,change):
    campaign,supplement,tuning,selection,_,_,_,hashes=design
    if change=='unselected':selection['selections']['smoothing-fw']['status']='no_valid_setting'
    elif change=='ineligible':selection['candidate_scores'][0]['eligible']=False
    elif change=='missing_trial':selection['candidate_scores'][0]['valid_runs']-=1
    elif change=='stale_campaign':selection['campaign_config_sha256']='0'*64
    elif change=='stale_tuning':selection['tuning_manifest_sha256']='0'*64
    else:selection['selection_source']='held-out selection'
    with pytest.raises(ValueError):m.resolve_selection(campaign,tuning,selection,supplement,**hashes)


def test_supplement_argv_matches_real_runner_interface(design):
    import run_benchmark
    campaign,supplement,_,_,choices,bindings,docs,_=design
    generated=m.make_supplements(campaign,choices,docs,supplement,bindings)
    for manifest in generated.values():
        for run in manifest['runs']:
            args=run_benchmark.parser().parse_args(run['argv'][1:]+['--instance','fixture.npz','--out','fixture-output'])
            assert args.method=='stochastic-kl-momentum' and args.oracle=='sketch' and args.batch==128
            assert args.rho0==choices['smoothing-fw']['parameters']['rho0']
            assert args.eta==choices['stochastic-kl']['parameters']['eta']
            assert args.inner_tol==choices['stochastic-kl']['parameters']['inner_tol']
            assert args.budget==run['budget'] and args.initialization==run['initialization']
