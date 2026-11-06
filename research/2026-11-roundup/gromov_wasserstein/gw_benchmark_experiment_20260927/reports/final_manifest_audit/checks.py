"""Read-only campaign structure audit; all instance metadata is an in-memory stub."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import prepare_stage_c_campaign as g

campaign=json.loads(g.CAMPAIGN_PATH.read_text())
theory=json.loads((ROOT/'configs/theory_config.json').read_text())
catalog=g.candidate_catalog(campaign)
assert {k:theory[k] for k in ('alpha','beta','delta')} == {k:campaign['schedules'][k] for k in ('alpha','beta','delta')}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config_hash=sha(g.CAMPAIGN_PATH)
tuning_hash=sha(ROOT/'configs/stage_c_manifests/tuning.json')

class NeverReadData:
    def __init__(self):self.names=set()
    def metadata(self,description):
        self.names.add(description['instance'])
        return {}

def cli(run):
    a=run['argv'];assert len(a)%2==1
    keys=a[1::2];assert len(keys)==len(set(keys))
    return dict(zip(keys,a[2::2]))

expected_counts={'final_product':253,'final_randomized':330,'small_pg':30}
expected_methods={
 'final_product':{'smoothing-fw':55,'smoothing-fw-exact':11,'full-kl':11,'stochastic-kl':55,'native-sampled-gw':55,'fw':11,'smoothing-row-projection':55},
 'final_randomized':{m:55 for m in ('smoothing-fw','smoothing-fw-exact','full-kl','stochastic-kl','fw','smoothing-row-projection')},
 'small_pg':{'pg':10,'full-kl':10,'fw':10}}
primary_names={f'metric512{"_noisy" if noise else ""}_s{s}.npz' for noise in (False,True) for s in range(2000,2005)}|{'bmmc1000.npz'}
small_names={f'metric128{"_noisy" if noise else ""}_s{s}.npz' for noise in (False,True) for s in range(2000,2005)}
seed_pairs={(s,900000+s) for s in range(100,105)}
seen_candidates=defaultdict(set)
passes=[]
with patch.object(g.np,'load',side_effect=AssertionError('Audit must never read a dataset')):
 for trial in range(12):
    selection={'schema_version':1,'campaign_config_sha256':config_hash,'tuning_manifest_sha256':tuning_hash,
               'selections':{m:{'status':'selected','candidate_id':f'c{(trial+i)%12:02d}','score':1.} for i,m in enumerate(g.TUNABLE)}}
    choices=g.selected_catalog(selection,catalog,campaign_sha256=config_hash,tuning_manifest_sha256=tuning_hash)
    source=NeverReadData()
    panels=g.make_finals(campaign,choices,source)
    assert set(source.names)==primary_names|small_names
    assert {phase:len(runs) for phase,runs in panels.items()}==expected_counts
    assert len({r['run_id'] for runs in panels.values() for r in runs})==613
    for phase,runs in panels.items():
        assert Counter(r['method'] for r in runs)==expected_methods[phase]
        budget=300 if phase=='final_product' else 30
        checkpoints=[1,3,10,30,100,300] if phase=='final_product' else [1,3,10,30]
        assert all(r['budget']==budget and r['checkpoints']==checkpoints for r in runs)
        assert sum(r['budget'] for r in runs)==campaign['raw_solver_budget_seconds'][phase]
        descriptions=small_names if phase=='small_pg' else primary_names
        assert {r['instance'] for r in runs}==descriptions
        cells=defaultdict(list)
        for r in runs:
            method=r['method'];origin='smoothing-fw' if method=='smoothing-fw-exact' else method
            assert r['candidate_id']==choices[origin]['candidate_id']
            assert r['candidate_parameters']==choices[origin]['parameters']
            assert r['source_method_for_selection']==origin
            assert r['deterministic']==(method not in g.STOCHASTIC)
            assert r['seed']==r['stochastic_seed'] and r['init_seed']==r['initialization_seed']
            assert r['problem_seed'] in ({None} if r['family']=='bmmc' else set(range(2000,2005)))
            params=r['candidate_parameters'];a=cli(r)
            assert a['--device']=='cuda' and a['--dtype']=='float32' and a['--threads']=='1'
            assert float(a['--budget'])==budget and [float(x) for x in a['--checkpoints'].split(',')]==checkpoints
            assert int(a['--seed'])==r['seed'] and int(a['--init-seed'])==r['init_seed']
            assert ('--initialization' in a)==(phase=='final_randomized')
            if method in g.TUNABLE:seen_candidates[method].add(r['candidate_id'])
            if method=='native-sampled-gw':
                assert a['--batch']==str(params['batch']) and float(a['--epsilon'])==params['epsilon']
                assert phase=='final_product' and r['argv'][0]=='run_native_sampled.py'
            else:
                assert int(a['--inner-max'])==1000 and r['argv'][0]=='run_benchmark.py'
                assert a['--method']==('smoothing-fw' if method=='smoothing-fw-exact' else method)
            if method.startswith('smoothing-'):
                assert float(a['--mu-scale'])==params['mu_scale']
                assert float(a['--gamma0'])==params['gamma0'] and float(a['--rho0'])==params['rho0']
                assert '--mu0' not in a
            if method in ('smoothing-fw','smoothing-fw-exact','smoothing-row-projection','stochastic-kl'):
                assert a['--oracle']==('exact' if method=='smoothing-fw-exact' else 'sketch') and a['--batch']=='128'
            if method=='smoothing-row-projection':
                assert float(a['--eta'])==params['eta_coefficient']*params['mu_scale']/(r['n']*r['m'])
                assert float(a['--eta-power'])==.25
            elif method=='pg':
                assert float(a['--eta'])==params['eta_coefficient']/r['n']**2
                assert float(a['--inner-tol'])==params['inner_tol'] and int(a['--pg-max-n'])==128
            elif method in ('full-kl','stochastic-kl'):
                assert float(a['--eta'])==params['eta'] and float(a['--inner-tol'])==params['inner_tol']
                assert float(a['--eta-power'])==0
            cells[(method,r['instance'])].append(r)
        for (method,name),group in cells.items():
            if phase=='final_randomized':
                assert len(group)==5 and {(r['seed'],r['init_seed']) for r in group}==seed_pairs
                assert all(r['initialization']=='randomized' for r in group)
            else:
                expected_seeds=set(range(100,105)) if phase=='final_product' and method in g.STOCHASTIC else {100}
                assert len(group)==len(expected_seeds) and {r['seed'] for r in group}==expected_seeds
                assert all(r['initialization']=='product' and r['init_seed']==100 for r in group)
    passes.append(trial)
 assert all(values=={f'c{i:02d}' for i in range(12)} for values in seen_candidates.values())
 tuning=g.make_tuning(campaign,catalog,NeverReadData())
 assert all(r['problem_seed'] in ({None} if r['family']=='bmmc' else {1000,1001,1002}) for r in tuning)
 assert all(r['seed'] in {0,1,2} for r in tuning)
 assert not ({r['instance'] for r in tuning if r['family']=='synthetic'} & (primary_names|small_names))

result={'scope':'Structural audit only:12 fabricated in-memory selections, no dataset np.load calls, no optimizer runs, no held-out manifest files generated.',
        'campaign_sha256':config_hash,'tuning_manifest_sha256':tuning_hash,'generator_sha256':sha(ROOT/'prepare_stage_c_campaign.py'),
        'audit_script_sha256':sha(Path(__file__)), 'candidate_scenarios_passed':len(passes),
        'final_specifications_checked':len(passes)*613,'phase_counts':expected_counts,'method_counts':expected_methods,
        'candidate_ids_checked_per_method':{m:sorted(v) for m,v in seen_candidates.items()},
        'all_budget_seed_initialization_parameter_propagation_assertions_passed':True,
        'synthetic_development_and_final_instance_names_disjoint':True,
        'dataset_reads':0,'remaining_condition':'Generate actual final manifests only after the real provenance-verified all-terminal tuning selection is frozen.'}
Path(__file__).with_name('result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('candidate_ids_checked_per_method','method_counts')},indent=2))
