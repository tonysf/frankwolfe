"""Small parser-only adversarial fixtures; no optimizers, arrays, or GPUs."""
import copy
import csv
import json
from pathlib import Path

import pytest
import summarize_momentum_replays as report

H='a'*64
REV='b'*40


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    calls=[]
    def expected_source(revision):
        calls.append(revision);assert revision==REV;return H
    monkeypatch.setattr(report,'utility_hash',expected_source)
    root=tmp_path/'download'; manifest=tmp_path/'pilot.json';rid='pilot/toy/smoothing-fw'
    config=dict(method='smoothing-fw',oracle='sketch',batch=8,gap_max_n=0,dtype='float64',initialization='product',budget=1.,checkpoints='1')
    sources={name:H for name in ['run_benchmark.py','smoothing_fw.py','gw_core.py','theory_config.json']}
    trial=dict(run_id=rid,original_phase='pilot',method='smoothing-fw',instance='toy.npz',instance_sha256=H,original_run_dir='/remote/original/'+rid,original_source_commit='c'*40,source_snapshot_sha256=sources,artifact_binding='frozen_at_preparation',original_artifact_sha256={name:H for name in ['metadata.json','trace.csv','final.npz']},original_run_spec=None,original_config=config)
    plan=dict(schema_version=1,purpose='missing_momentum_diagnostics_only',phase='pilot',out_root='/remote/replays/pilot',run_count=1,runs=[trial])
    write(manifest,plan)
    identity=dict(manifest_sha256=report.digest(manifest),source_commit=REV,task_count=1,out_root=plan['out_root'])
    write(root/'_replay/binding.json',identity)
    binding=dict(metadata_sha256=H,trace_sha256=H,final_sha256=H,instance_sha256=H,source_file_sha256={k:v for k,v in sources.items() if k.endswith('.py')},theory_config_sha256=H)
    counters={key:0 for key in report.COUNTERS};counters.update(oracle_evaluations=2,stochastic_gradient_calls=2,sketch_probes=16,initialization_oracle_calls=1,row_lmo_calls=2,attempted_updates=2,completed_updates=2)
    required={'raw_loss','rounded_loss','repair_l1','mu','smoothing_point_iteration','true_smoothed_gap'}|set(report.RESIDUALS)|{'rounded_'+k for k in report.RESIDUALS}|{'cumulative_'+k for k in report.COUNTERS}
    rows=[];proof_rows=[]
    for index,K in enumerate((0,2)):
        fields=required|({'last_update_gamma','last_update_rho','last_update_mu'} if K else set())
        row=dict(original_trace_row=index,kind='initial' if K==0 else 'checkpoint',output_iteration=K,momentum_point_iteration=max(0,K-1),stored_momentum_iteration=max(0,K-1),tracking_error_point_iteration=K,momentum_error_matches_current_output=str(K==0),replay_diagnostic_oracle_draws=0,original_solver_oracle_evaluations=max(1,K),original_actual_solver_seconds=.01 if K==0 else 1.01,original_time_to_this_feasible_output=.02 if K==0 else 1.02,original_target_solver_seconds='' if K==0 else 1.,momentum_error_frobenius=.2,stored_momentum_current_output_tracking_error_frobenius=.3)
        row.update({'recomputed_'+key:0. for key in fields});row['recomputed_cumulative_sketch_probes']=max(K,1)*8
        rows.append(row);proof_rows.append(dict(original_trace_row=index,iteration=K,compared_fields=sorted(fields),exact_match=True))
    path=root/rid;path.mkdir(parents=True);write_csv(path/'diagnostics.csv',rows)
    endpoint=dict(expected_array_sha256=H,replay_array_sha256=H,expected_shape=[8,8],replay_shape=[8,8],expected_dtype='float64',replay_dtype='float64',bitwise_match=True,max_abs_difference=0.,difference_all_finite=True)
    proof=dict(schema_version=1,status='verified',verified=True,bindings=binding,utility_sha256=H,original_source_commit=trial['original_source_commit'],original_status='completed',original_config=config,run_dir=trial['original_run_dir'],original_counters=counters,replay_counters=counters,completed_updates=2,error=None,failure_kind=None,checks={**dict.fromkeys(report.PROOF_CHECKS,True),'checkpoint_scalars':proof_rows,'initialization_hash':'unavailable_in_legacy_original; initial_metrics_and_terminal_arrays_required'},checkpoint_count=2,candidate_checkpoint_count=2,initialization_plan_sha256=H,initial_rng_state_sha256=H,terminal_rng_state_sha256=H,P_raw_array_sha256=H,P_rounded_array_sha256=H,endpoint_comparisons={'P_raw':endpoint,'P_rounded':endpoint},replay_diagnostic_counts=dict(loss_calls=6,full_gradient_calls=3,assignment_lmo_calls=0,rounding_calls=2,marginal_residual_calls=4,oracle_draws=0),original_timing={'solver_seconds':1.01},timing=dict(controller_elapsed_seconds=3.,replay_worker_total_seconds=2.,replay_initialization_seconds=.1,replay_update_seconds=.5,replay_diagnostics_seconds=.5,replay_other_seconds=.9),missed_original_checkpoints=[])
    def seal():
        proof['diagnostics_sha256']=report.digest(path/'diagnostics.csv') if (path/'diagnostics.csv').exists() else None;write(path/'verification.json',proof)
        receipt=dict(run_spec=trial,manifest_sha256=report.digest(manifest),controller_source_commit=REV,original_bindings=binding,original_campaign_receipt_sha256=None,status=proof['status'],returncode=0 if proof['verified'] else 1,output_sha256=dict(verification_sha256=report.digest(path/'verification.json'),diagnostics_sha256=proof['diagnostics_sha256'] if proof['verified'] else None))
        write(path/'replay_receipt.json',receipt)
        ledger={**identity,'task_index':0,'assigned_count':1,'records':[receipt],'status':'completed'}
        write(root/'_replay/shard_00.json',ledger)
    seal()
    return dict(root=root,manifest=manifest,path=path,trial=trial,plan=plan,proof=proof,rows=rows,seal=seal,calls=calls)


def test_complete_proof_preserves_original_clock_and_uses_receipt_revision(fixture):
    f=fixture;summary,runs,_=report.inspect_phase(f['manifest'],f['root'])
    assert not summary['errors'];r=runs[0]
    assert r['verified'] and r['terminal'] and r['provenance_valid']
    assert r['controller_elapsed_seconds']==3. # not 3+worker2+subtotals
    assert r['checkpoints'][-1]['original_solver_seconds']==1.01
    assert r['checkpoints'][-1]['momentum_point_iteration']==1
    assert f['calls'] and set(f['calls'])=={REV}


@pytest.mark.parametrize('tamper',['proof_flag','point_index','utility_hash'])
def test_tampered_proof_rejected_even_with_resealed_artifact_hashes(fixture,tamper):
    f=fixture
    if tamper=='proof_flag':f['proof']['checks']['P_raw_bitwise']=False
    elif tamper=='utility_hash':f['proof']['utility_sha256']='d'*64
    else:
        f['rows'][-1]['momentum_point_iteration']=2;write_csv(f['path']/'diagnostics.csv',f['rows'])
    f['seal']()
    _,runs,_=report.inspect_phase(f['manifest'],f['root'])
    assert runs[0]['status']=='invalid' and not runs[0]['verified'] and not runs[0]['checkpoints']


def test_changed_diagnostics_hash_rejected(fixture):
    f=fixture
    with (f['path']/'diagnostics.csv').open('a') as stream:stream.write('\n')
    _,runs,_=report.inspect_phase(f['manifest'],f['root'])
    assert runs[0]['status']=='invalid' and 'Diagnostics hash' in runs[0]['issues'][0]


def test_missing_prescribed_receipt_remains_visible(fixture):
    f=fixture;(f['path']/'replay_receipt.json').unlink()
    _,runs,_=report.inspect_phase(f['manifest'],f['root'])
    assert len(runs)==1 and runs[0]['status']=='missing' and not runs[0]['terminal']


def test_duplicate_scientific_record_rejected(fixture):
    f=fixture;plan=copy.deepcopy(f['plan']);plan['runs'].append(copy.deepcopy(plan['runs'][0]));plan['run_count']=2
    write(f['manifest'],plan)
    with pytest.raises(ValueError,match='Duplicate'):report.inspect_phase(f['manifest'],f['root'])


def test_failed_original_remains_terminal_unverified_with_cost(fixture):
    f=fixture;f['proof'].update(status='failed',verified=False,original_status='failed',failure_kind='unsupported_original_failure',error='Failed original; no reconstructed endpoint.')
    (f['path']/'diagnostics.csv').unlink();f['seal']()
    summary,runs,_=report.inspect_phase(f['manifest'],f['root'])
    r=runs[0];assert not summary['errors']
    assert r['terminal'] and r['provenance_valid'] and not r['verified'] and r['status']=='failed'
    assert r['failure_kind']=='unsupported_original_failure' and r['controller_elapsed_seconds']==3.
    assert r['checkpoints']==[]


def test_prescribed_plot_denominators_include_missing_replays_and_windows(tmp_path):
    # Read frozen scalar manifests only: deterministic exact product has five
    # independent synthetic instances and one BMMC instance, not five fake starts.
    plan=report.read_json(report.ROOT/'configs/momentum_diagnostic_replays/finals.json')
    trials=[r for r in plan['runs'] if r['original_phase']=='final_product' and r['method']=='smoothing-fw-exact']
    runs=[report.inspect_trial(t,tmp_path/t['run_id'],'unused',None,None) for t in trials]
    empty=report.curve_summaries(runs)
    assert all(p['observed_run_count']==0 and p['x'] is None and not p['coverage_complete'] for p in empty)
    for p in empty:
        expected=1 if p['family']=='bmmc' else 5
        assert p['prescribed_run_count']==p['prescribed_instance_count']==expected
    clean=[r for r in runs if r['family']=='synthetic' and r['noise']==0]
    bmmc=[r for r in runs if r['family']=='bmmc']
    for index,r in enumerate(clean[:4]+bmmc):
        r['verified']=True
        targets=[1.,300.] if index!=3 else [1.]
        r['checkpoints']=[dict(target_seconds=t,original_solver_seconds=t+.01,momentum_error_frobenius=.2,stale_current_output_error_frobenius=.3) for t in targets]
    points={(p['family'],p['target_seconds']):p for p in report.curve_summaries(runs)}
    assert points['synthetic_clean',1.]['observed_run_count']==4
    assert points['synthetic_clean',300.]['observed_run_count']==3
    assert points['synthetic_clean',300.]['prescribed_run_count']==5
    assert points['synthetic_clean',300.]['observed_instance_count']==3
    assert not points['synthetic_clean',300.]['coverage_complete']
    assert points['bmmc',300.]['observed_run_count']==1
    assert points['bmmc',300.]['prescribed_run_count']==1
    assert points['bmmc',300.]['coverage_complete']
    assert points['bmmc',300.]['primary']['count']==1
