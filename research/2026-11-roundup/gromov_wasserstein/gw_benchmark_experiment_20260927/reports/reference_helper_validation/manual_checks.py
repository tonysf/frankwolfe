"""Manual parser checks; run: python reports/reference_helper_validation/manual_checks.py.

Uses preserved CPU report JSON only; temporary mutations are not benchmark data.
"""
import copy,hashlib,json,math,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import summarize_feasible_references as s
source=ROOT/'reports/campaign_summarizer_smoke/report/report.json'
report=json.loads(source.read_text());checks={}
base=s.references(report)
assert len(base)==1 and base[0]['known_f_star']==0.0 and base[0]['expected_runs']==4
checks['real_preserved_cpu_report_passes_and_tuning_excluded']=True

def reject(name,mutate):
 r=copy.deepcopy(report);mutate(r)
 try:s.references(r)
 except (ValueError,KeyError):checks[name]=True
 else:raise AssertionError(name)

reject('incomplete_report_rejected',lambda r:r.update(execution_complete=False))
reject('invalid_provenance_rejected',lambda r:r.update(provenance_valid=False))
reject('contradictory_selection_errors_rejected',lambda r:r.update(selection_provenance_errors=['wrong binding']))
reject('missing_final_cell_rejected',lambda r:r['runs'].pop())
reject('all_invalid_instance_rejected',lambda r:[x.update(eligible_final_output=False,status='failed') for x in r['runs'] if x['phase']!='tuning'])
reject('nonfinite_eligible_loss_rejected',lambda r:next(x for x in r['runs'] if x['phase']!='tuning').update(rounded_loss_float64=float('nan')))
reject('mixed_instance_hashes_rejected',lambda r:next(x for x in r['runs'] if x['phase']!='tuning').update(expected_instance_sha256='wrong'))
reject('false_campaign_completeness_rejected',lambda r:r.update(scope='campaign'))
clean=copy.deepcopy(report)
first=next(x for x in clean['runs'] if x['phase']!='tuning');first.update(rounded_loss_float64=-1e-15)
r=s.references(clean)[0]
assert r['known_f_star']==0.0 and r['reference_value']==0.0 and r['best_observed_rounded_loss_float64']==-1e-15
checks['noiseless_optimum_zero_and_negative_observation_retained_separately']=True
for family,noise in [('synthetic',0.1),('bmmc',None)]:
 noisy=copy.deepcopy(clean)
 for x in noisy['runs']:
  if x['phase']!='tuning':x.update(family=family,noise=noise)
 r=s.references(noisy)[0]
 assert r['known_f_star'] is None and r['descriptive_f_ref']==-1e-15 and r['reference_value']==-1e-15
 checks[family+'_descriptive_reference_keeps_signed_observation']=True
mixed=copy.deepcopy(report)
next(x for x in mixed['runs'] if x['phase']!='tuning').update(budget=300)
r=s.references(mixed)[0]
assert r['mixed_budgets'] and r['included_budgets_seconds']==[.03,300]
checks['mixed_budgets_explicit']=True
# Native callback repair can precede later epilogue work for the same output.
native=copy.deepcopy(report)
native_run=min((x for x in native['runs'] if x['phase']!='tuning'),key=lambda x:x['rounded_loss_float64'])
native_run['method']='native-sampled-gw'
native_run['actual_solver_seconds']=native_run['final_time_to_output']+0.1
observed=s.references(native)[0]['best_observed_source']
assert observed['final_time_to_output']==native_run['final_time_to_output'] and observed['actual_solver_seconds']==native_run['actual_solver_seconds']
assert observed['final_time_to_output']<observed['actual_solver_seconds']
checks['native_same_output_availability_before_epilogue_preserved']=True
reject('negative_solver_clock_rejected',lambda r:next(x for x in r['runs'] if x['phase']!='tuning').update(actual_solver_seconds=-1))
reject('negative_output_availability_rejected',lambda r:next(x for x in r['runs'] if x['phase']!='tuning').update(final_time_to_output=-1))

with tempfile.TemporaryDirectory(prefix='feasible_reference_outputs_') as td:
 out=Path(td)/'valid';document=s.write_outputs(source,out)
 assert document['input_report_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
 assert all((out/name).exists() for name in ['feasible_references.json','feasible_references.csv','feasible_references.md'])
 checks['three_outputs_and_exact_input_hash']=True
 try:s.write_outputs(source,out)
 except FileExistsError:checks['existing_reference_archive_preserved']=True
 else:raise AssertionError('overwrite allowed')
 bad=Path(td)/'bad.json';bad.write_text(json.dumps({**report,'execution_complete':False}))
 try:s.write_outputs(bad,Path(td)/'rejected')
 except ValueError:assert not (Path(td)/'rejected').exists();checks['invalid_report_writes_nothing']=True
 else:raise AssertionError('invalid report accepted')
result={'scope':'Parser and adversarial checks only; no benchmark/data reads.','checks':checks,'count':len(checks),'script_sha256':hashlib.sha256((ROOT/'summarize_feasible_references.py').read_bytes()).hexdigest()}
Path(__file__).with_name('validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
