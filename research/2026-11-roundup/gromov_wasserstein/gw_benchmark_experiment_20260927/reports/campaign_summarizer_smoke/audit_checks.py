"""Adversarial parser checks using copied n16 CPU evidence; not scientific data."""
from pathlib import Path
import copy,csv,json,shutil,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import summarize_campaign as sc
BASE=ROOT/'reports/campaign_summarizer_smoke'
m=json.loads((BASE/'tuning.json').read_text());spec=m['runs'][0];mh=sc.digest(BASE/'tuning.json')
checks={}

def trial(name,change,wanted,chosen_spec=None,source=None):
    current=copy.deepcopy(chosen_spec or spec)
    with tempfile.TemporaryDirectory(prefix='campaign_parser_audit_') as td:
        root=Path(td);dest=root/current['run_id'];dest.parent.mkdir(parents=True)
        shutil.copytree(source or BASE/'input'/current['run_id'],dest)
        metadata=json.loads((dest/'metadata.json').read_text())
        change(dest,metadata,current)
        (dest/'metadata.json').write_text(json.dumps(metadata))
        result=sc.inspect_run(current,[root],'tuning',mh,'parser-smoke')
        assert wanted(result),(name,result['issues'],result['status'])
        if name in {'precision_cli_mismatch_rejected','wrong_receipt_spec_rejected','infrastructure_receipt_requires_review'}:
            assert not result['provenance_valid'],name
        checks[name]=True

def reject(text):return lambda r:not r['eligible_final_output'] and any(text in msg for msg in r['issues'])
trial('unchanged_saved_cpu_output_passes',lambda *a:None,lambda r:r['eligible_final_output'] and bool(r['final_npz_sha256']))
def corrupt_plan(p,m,s):
    with np.load(p/'final.npz') as z:a={k:z[k].copy() for k in z.files}
    a['P_rounded'][0,0]+=.01
    np.savez(p/'final.npz',**a)
trial('stale_metadata_nonfeasible_saved_plan_rejected',corrupt_plan,reject('differs from its float64 validation'))
trial('non_float64_validation_rejected',lambda p,m,s:m['final_float64_validation'].update(dtype='float32'),reject('No float64'))
trial('unknown_completed_status_rejected',lambda p,m,s:m.update(status='completed_unreviewed'),reject('Unrecognized'))
def wrong_receipt(p,m,s):
    rs=copy.deepcopy(s);rs['seed']=9000
    (p/'campaign_receipt.json').write_text(json.dumps(dict(run_id=s['run_id'],manifest_sha256=mh,run_spec=rs,status=m['status'])))
trial('wrong_receipt_spec_rejected',wrong_receipt,reject('run specification'))
def infrastructure_receipt(p,m,s):
    (p/'campaign_receipt.json').write_text(json.dumps(dict(run_id=s['run_id'],manifest_sha256=mh,run_spec=s,status='infrastructure_failure',infrastructure_error='Synthetic parser receipt failure.')))
trial('infrastructure_receipt_requires_review',infrastructure_receipt,lambda r:r['execution_terminal'] and not r['provenance_valid'] and not r['eligible_final_output'])
def wrong_cli(p,m,s):s['argv']=['run_benchmark.py','--dtype','float32']
trial('precision_cli_mismatch_rejected',wrong_cli,reject('CLI parameter dtype'))
def mutate_trace(field,fn):
    def change(p,m,s):
        with (p/'trace.csv').open() as f:rows=list(csv.DictReader(f));fields=list(rows[0])
        fn(rows)
        with (p/'trace.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return change
trial('backdated_output_time_rejected',mutate_trace('time',lambda rows:rows[1].update(time_to_this_feasible_output='0')),reject('precedes'))
trial('same_completed_iteration_reuse_rejected',mutate_trace('iteration',lambda rows:rows[2].update(iteration=rows[1]['iteration'])),reject('reused'))
trial('solver_clock_beyond_final_rejected',mutate_trace('clock',lambda rows:rows[-1].update(actual_solver_seconds='100')),reject('impossible'))
ns=copy.deepcopy(spec);src=ROOT/'reports/practical_summarizer_smoke/native_early_input/toy16/native-sampled-gw';nm=json.loads((src/'metadata.json').read_text())
ns.update(method='native-sampled-gw',implementation='native-sampled-gw',parameters={},budget=nm['config']['budget'],checkpoints=[float(nm['config']['budget'])],seed=nm['config']['seed'],init_seed=nm['config']['init_seed'],instance_sha256=nm['instance_sha256'],expected_array_sha256=nm['array_sha256'])
trial('real_native_max_iter_stop_eligible_without_budget',lambda *a:None,lambda r:r['eligible_final_output'] and r['native_max_iter_exhausted'] and not r['budget_met'],ns,src)
trial('native_subbudget_without_stop_reason_rejected',lambda p,m,s:m['instrumentation'].update(native_early_stop=False,max_iter_exhausted=False),reject('stopping reason'),ns,src)
record=sc.inspect_run(spec,[BASE/'input'],'tuning',mh,'parser-smoke')
candidates,_=sc.candidate_audit([record],None)
selection={'selections':{spec['method']:{'status':'selected','candidate_id':spec['candidate_id'],'score':None}}}
assert sc.candidate_audit([record],selection)[1][0]['selection_audit']=='MISMATCH';checks['null_authoritative_score_rejected']=True
selection['selections'][spec['method']]['score']=np.nextafter(candidates[0]['score'],np.inf)
assert sc.candidate_audit([record],selection)[1][0]['selection_audit']=='MISMATCH';checks['near_but_unequal_authoritative_score_rejected']=True
same=[copy.deepcopy(record),copy.deepcopy(record)]
for i,r in enumerate(same):r['initialization_plan_sha256']=str(i);r['run_id']+=str(i)
assert sc.initialization_audit(same) and all(not r['eligible_final_output'] for r in same);checks['cross_method_initialization_hash_mismatch_invalidates_group']=True
validation=json.loads((BASE/'validation.json').read_text());validation.update(checks);validation['script_sha256']=sc.digest(ROOT/'summarize_campaign.py');validation['scope']='Adversarial parser checks on preserved CPU outputs; no new benchmark results.'
(BASE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(checks,indent=2))
# The following are parser fixtures: genuine archived pilot output plus synthetic
# launcher control records. They are never reported as campaign observations.
pilot=ROOT/'ruche_results/1961459/metric512_s1001/native-sampled-gw'
pm=json.loads((pilot/'metadata.json').read_text())
ps=copy.deepcopy(ns);ps.update(run_id='fixture/native',instance='metric512_s1001.npz',n=512,m=512,budget=30,checkpoints=[1,3,10,30],seed=0,init_seed=2026,instance_sha256=pm['instance_sha256'],expected_array_sha256=pm['array_sha256'],argv=['run_native_sampled.py','--device','cuda','--dtype','float32','--threads','1','--max-iter','1000000'])
with tempfile.TemporaryDirectory(prefix='source_ledger_parser_fixture_') as td:
    root=Path(td);dest=root/ps['run_id'];dest.parent.mkdir(parents=True);shutil.copytree(pilot,dest)
    receipt=dict(run_id=ps['run_id'],run_spec=ps,manifest_sha256=mh,status=pm['status'])
    (dest/'campaign_receipt.json').write_text(json.dumps(receipt))
    control=root/'_campaign';control.mkdir();lp=control/'shard_00.json'
    ledger=dict(source_commit=pm['source_commit'],manifest_sha256=mh,records=[receipt]);lp.write_text(json.dumps(ledger))
    def read():return sc.inspect_run(ps,[root],'tuning',mh,'campaign',sc.source_ledgers([root]))
    output=read();assert output['eligible_final_output'],output['issues'];checks['real_a100_pilot_passes_full_parser_with_explicit_synthetic_control_fixture']=True
    ledger['source_commit']='wrong';lp.write_text(json.dumps(ledger));assert not read()['provenance_valid'];checks['wrong_ledger_source_commit_rejected']=True
    ledger['source_commit']=pm['source_commit'];lp.write_text(json.dumps(ledger))
    meta=copy.deepcopy(pm);meta['environment']['cuda_matmul_allow_tf32']=True;(dest/'metadata.json').write_text(json.dumps(meta));assert reject('precision/determinism')(read());checks['tf32_policy_mismatch_rejected']=True
    meta=copy.deepcopy(pm);meta['environment']['gpu']['name']='different GPU';(dest/'metadata.json').write_text(json.dumps(meta));assert reject('A100')(read());checks['wrong_gpu_model_rejected']=True
    (dest/'metadata.json').write_text(json.dumps(pm));(dest/'source_snapshot'/'gw_core.py').write_text('# changed fixture\n');assert reject('source hash mismatch')(read());checks['modified_source_snapshot_rejected']=True
validation.update(checks);validation['script_sha256']=sc.digest(ROOT/'summarize_campaign.py');validation['provenance_fixture_note']='Source ledger cases use genuine earlier A100 pilot outputs and synthetic control receipts solely as parser fixtures, never as campaign results.'
(BASE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k not in validation or 'ledger' in k or 'a100' in k or 'policy' in k or 'gpu' in k or 'snapshot' in k},indent=2))
import subprocess
with tempfile.TemporaryDirectory(prefix='completion_semantics_parser_fixture_') as td:
    root=Path(td);chosen=copy.deepcopy(spec);dest=root/'input'/chosen['run_id'];dest.parent.mkdir(parents=True);shutil.copytree(BASE/'input'/chosen['run_id'],dest)
    md=json.loads((dest/'metadata.json').read_text());md.update(status='failed',error='Deliberate parser fixture failure; no new solver run.');(dest/'metadata.json').write_text(json.dumps(md))
    manifest=copy.deepcopy(m);manifest['runs']=[chosen];mp=root/'manifest.json';mp.write_text(json.dumps(manifest))
    cmd=[sys.executable,str(ROOT/'summarize_campaign.py'),'--manifest',str(mp),'--root',str(root/'input'),'--out',str(root/'report'),'--scope','parser-smoke','--require-complete']
    result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stderr
    report=json.loads((root/'report'/'report.json').read_text());assert report['execution_complete'] and not report['all_outputs_valid'] and report['status']=='execution_complete_with_failed_or_invalid_outputs';checks['completed_failed_trials_are_execution_complete']=True
    result=subprocess.run(cmd+['--write-selection',str(root/'forbidden.json')],capture_output=True,text=True);assert result.returncode!=0 and not (root/'forbidden.json').exists();checks['parser_scope_cannot_freeze_authoritative_selection']=True
    result=subprocess.run([sys.executable,str(ROOT/'summarize_campaign.py'),'--manifest',str(ROOT/'configs/stage_c_manifests/tuning.json'),'--root',str(root/'absent'),'--out',str(root/'missing'),'--write-selection',str(root/'provisional.json')],capture_output=True,text=True)
    assert result.returncode!=0 and not (root/'provisional.json').exists() and 'no provisional choice' in result.stderr;checks['pending_tuning_cannot_freeze_selection']=True
validation.update(checks);validation['script_sha256']=sc.digest(ROOT/'summarize_campaign.py');(BASE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print('Completion and selection-readiness checks passed.')
# Report-only final/supplement lineage and statistics tests use in-memory records.
sel={'schema_version':1,'campaign_config_sha256':'base_config','tuning_manifest_sha256':'tuning_hash','selections':{
    'stochastic-kl':{'status':'selected','candidate_id':'c02','score':0.4},
    'smoothing-fw':{'status':'selected','candidate_id':'c07','score':0.6}}}
fm={'phase':'final_product','sha256':'base_manifest','campaign_config_sha256':'base_config','tuning_manifest_sha256':'tuning_hash','selection_sha256':'selection_hash'}
fr=copy.deepcopy(record);fr.update(phase='final_product',method='stochastic-kl',candidate_id='c02',source_method_for_selection='stochastic-kl',run_id='fixture_base',source_manifest_sha256='base_manifest')
assert sc.selection_provenance([fm],[fr],None,None,'campaign');checks['final_phase_requires_frozen_selection']=True
assert not sc.selection_provenance([fm],[fr],sel,'selection_hash','campaign');checks['valid_final_only_selection_lineage_passes']=True
bad_fm={**fm,'tuning_manifest_sha256':'wrong'}
assert sc.selection_provenance([bad_fm],[fr],sel,'selection_hash','campaign');checks['final_only_wrong_tuning_hash_rejected']=True
bad_fr={**fr,'candidate_id':'c03'}
assert sc.selection_provenance([fm],[bad_fr],sel,'selection_hash','campaign');checks['final_candidate_must_equal_selected_id']=True
sm={**fm,'phase':'momentum_product','sha256':'supplement_manifest','comparison_phase':'final_product','supplemental_config_sha256':'supplement_config','parent_campaign_config_sha256':'base_config','base_manifest_sha256':'base_manifest'}
sr={**fr,'phase':'momentum_product','method':'stochastic-kl-momentum','candidate_id':'kl_c02_rho_c07','source_candidate_id':'c02','momentum_source_method_for_selection':'smoothing-fw','momentum_source_candidate_id':'c07','comparison_phase':'final_product','run_id':'fixture_supplement','source_manifest_sha256':'supplement_manifest','parent_run_id':'fixture_base','parent_manifest_sha256':'base_manifest'}
assert not sc.selection_provenance([fm,sm],[fr,sr],sel,'selection_hash','campaign');checks['supplement_matches_both_frozen_parent_candidates']=True
assert sc.selection_provenance([fm,sm],[fr,{**sr,'momentum_source_candidate_id':'c08'}],sel,'selection_hash','campaign');checks['supplement_wrong_momentum_parent_rejected']=True
assert sc.selection_provenance([fm,{**sm,'base_manifest_sha256':'wrong'}],[fr,sr],sel,'selection_hash','campaign');checks['supplement_wrong_baseline_manifest_rejected']=True
assert sc.selection_provenance([fm,sm],[fr,{**sr,'comparison_phase':'final_randomized'}],sel,'selection_hash','campaign');checks['supplement_per_run_comparison_phase_bound']=True
assert sc.selection_provenance([fm,sm],[fr,{**sr,'parent_run_id':'missing_pair'}],sel,'selection_hash','campaign');checks['supplement_requires_exact_paired_parent_run']=True
assert sc.selection_provenance([fm,sm],[fr,{**sr,'seed':9999}],sel,'selection_hash','campaign');checks['supplement_pair_must_share_stochastic_seed']=True
assert sc.selection_provenance([sm],[sr],sel,'selection_hash','campaign');checks['supplement_comparison_requires_supplied_base_manifest']=True
assert sc.selection_provenance([fm,sm],[fr,{**sr,'parameters':{}}],sel,'selection_hash','campaign');checks['supplement_pair_must_inherit_all_kl_parameters']=True

actual=copy.deepcopy(record);actual.update(phase='final_product',comparison_phase='final_product',method='stochastic-kl',candidate_id='c00')
for observation in actual['observations'].values():observation['rounded_loss']=123.0
fixed,endpoints=sc.endpoint_aggregates([actual]);assert endpoints[0]['metrics']['rounded_loss_float64']['median']==record['rounded_loss_float64'];checks['endpoint_summary_uses_actual_float64_not_trace_loss']=True
failed=copy.deepcopy(actual);failed.update(run_id='failed_endpoint_fixture',seed=999,status='failed',eligible_final_output=False)
fixed,endpoints=sc.endpoint_aggregates([actual,failed]);assert endpoints[0]['counts']['failed_or_invalid_units']==1 and endpoints[0]['counts']['valid_final_units']==1 and endpoints[0]['metrics']['rounded_loss_float64']['count']==1 and endpoints[0]['resource_units']==2;checks['endpoint_failures_counted_without_inventing_quality']=True
siblings=[]
for i in range(3):
    x=copy.deepcopy(actual);x.update(instance='problemA',run_id='A'+str(i),seed=i,rounded_loss_float64=0.0);siblings.append(x)
x=copy.deepcopy(actual);x.update(instance='problemB',run_id='B',seed=0,rounded_loss_float64=10.0);siblings.append(x)
fixed,endpoints=sc.endpoint_aggregates(siblings);assert endpoints[0]['metrics']['rounded_loss_float64']['median']==5.0;checks['final_endpoints_equal_weight_independent_instances']=True
early_spec={**ns,'run_id':'toy16/native-sampled-gw'}
early=sc.inspect_run(early_spec,[src.parents[1]],'final_product',mh,'parser-smoke');assert early['eligible_final_output'],early['issues']
fixed,endpoints=sc.endpoint_aggregates([early]);assert endpoints[0]['metrics']['actual_solver_seconds']['median']<early['budget'] and endpoints[0]['counts']['missing_checkpoint_observations']>0;checks['native_early_endpoint_retains_actual_time_and_missing_windows']=True
supp=copy.deepcopy(actual);supp.update(phase='momentum_product',comparison_phase='final_product',method='stochastic-kl-momentum',candidate_id='kl_c00_rho_c00')
fixed,broad,duplicates=sc.aggregate([actual,supp]);assert {r['phase'] for r in broad}=={'final_product','momentum_product'} and {r['comparison_phase'] for r in broad}=={'final_product'};checks['supplement_keeps_provenance_phase_but_shares_comparison_panel']=True
with tempfile.TemporaryDirectory(prefix='stage_d_evidence_test_') as td:
    evidence=sc.evidence_inventory(Path(td),{},[])
    assert evidence['decision']=='not_automatically_opened' and evidence['sources']['true_gap_vs_iteration']['available'] and evidence['sources']['true_gap_vs_oracle_work']['available'];checks['stage_d_links_true_gap_evidence_without_automatic_gate']=True
validation.update(checks);validation['script_sha256']=sc.digest(ROOT/'summarize_campaign.py');(BASE/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print('Final endpoint and supplemental report checks passed; focused checks:',len(checks))
