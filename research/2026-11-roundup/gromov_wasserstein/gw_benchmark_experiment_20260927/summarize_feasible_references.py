"""Protocol 14 references from a completed final report; no solver/data access.

This only summarizes already validated actual rounded outputs. The minimum across
included final runs is a descriptive reference, never optimizer selection or a
new optimality claim. Different solver budgets remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

FINAL_PHASES={'final_product','final_randomized','small_pg','momentum_product','momentum_randomized'}
COMPLETED={'complete_provided_manifests','complete_provided_manifests_with_numerical_issues',
           'execution_complete_with_failed_or_invalid_outputs'}
ELIGIBLE={'completed','completed_with_warnings','completed_with_inner_failures'}
RESIDUALS=('row_l1','col_l1','col_l2','both_l1','mass_error','min_entry')


def finite(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(name+' must be a finite recorded number.')
    return value


def references(report):
    if (report.get('schema_version')!=1 or report.get('execution_complete') is not True
        or report.get('provenance_valid') is not True or report.get('status') not in COMPLETED
        or report.get('selection_provenance_errors') or report.get('initialization_hash_audit_errors')):
        raise ValueError('Require a completed, provenance-valid campaign report; partial/invalid reports cannot define final references.')
    if report.get('scope')=='campaign':
        coverage=report.get('stage_d_evidence',{}).get('required_stage_c_phase_coverage',{})
        if (set(coverage)!=FINAL_PHASES or any(c.get('prescribed_count') is None
            or c.get('provided_count')!=c['prescribed_count'] or c.get('terminal_count')!=c['prescribed_count'] for c in coverage.values())):
            raise ValueError('All prescribed final and supplemental phases must finish before defining campaign references.')
    runs=report['runs']
    if any(r['phase'] not in FINAL_PHASES | {'tuning'} for r in runs):
        raise ValueError('Unrecognized phase; only prescribed final phases may contribute.')
    finals=[r for r in runs if r['phase'] in FINAL_PHASES]
    if not finals or len({r['run_id'] for r in finals})!=len(finals):
        raise ValueError('Final runs must be present with unique run IDs.')
    expected={m['sha256']:m['expected_runs'] for m in report['manifests'] if m['phase'] in FINAL_PHASES}
    if dict(Counter(r['source_manifest_sha256'] for r in finals))!=expected:
        raise ValueError('Final run records do not cover every prescribed final-manifest cell.')
    grouped=defaultdict(list)
    for run in finals:
        if run.get('execution_terminal') is not True or run.get('provenance_valid') is not True:
            raise ValueError('Nonterminal or unverified final run: '+run['run_id'])
        grouped[run['instance']].append(run)
    result=[]
    for instance,members in sorted(grouped.items()):
        first=members[0]
        hashes={r.get('expected_instance_sha256') or r.get('instance_sha256') for r in members}
        if None in hashes or len(hashes)!=1:
            raise ValueError(instance+': inconsistent or absent mathematical-instance hash.')
        if any(any(r.get(k)!=first.get(k) for k in ('family','n','noise','problem_seed')) for r in members):
            raise ValueError(instance+': inconsistent mathematical-instance description.')
        budgets=sorted({finite(r['budget'],'budget') for r in members})
        if not budgets or budgets[0]<=0:
            raise ValueError(instance+': solver budgets must be positive.')
        valid=[r for r in members if r.get('eligible_final_output') is True]
        if not valid:
            raise ValueError(instance+': no eligible final output; no observed reference can be invented.')
        for r in valid:
            if r['status'] not in ELIGIBLE:
                raise ValueError('Eligible-output flag disagrees with run status: '+r['run_id'])
            finite(r['rounded_loss_float64'],'rounded final float64 loss')
            clock=finite(r['actual_solver_seconds'],'actual solver time')
            output=finite(r['final_time_to_output'],'actual feasible output time')
            residual=r['final_rounded_residuals']
            if any(finite(residual[key],'rounded '+key)<0 for key in RESIDUALS):
                raise ValueError('Invalid recorded rounded residuals: '+r['run_id'])
            eps={'float32':2**-23,'float64':2**-52}.get(r.get('solver_dtype'))
            # The reporter validates output availability at its observation clock.
            # Native epilogue work can extend total solver time after that output.
            if (clock<0 or output<0 or eps is None
                or not 0<=finite(residual['both_l1'],'rounded marginal residual')<=100*eps
                or finite(residual['min_entry'],'rounded minimum entry')<0):
                raise ValueError('Invalid recorded final feasibility/time evidence: '+r['run_id'])
        best=min(valid,key=lambda r:(r['rounded_loss_float64'],r['run_id']))
        clean=first['family']=='synthetic' and first['noise']==0
        best_loss=best['rounded_loss_float64']  # Retain signed cancellation values exactly.
        source={k:best.get(k) for k in ('run_id','phase','method','candidate_id','budget','actual_solver_seconds',
                                       'final_time_to_output','final_rounded_residuals','final_npz_sha256')}
        source['final_rounded_residuals']={key:best['final_rounded_residuals'][key] for key in RESIDUALS}
        source['rounded_loss_float64']=best_loss
        result.append({'instance':instance,'instance_sha256':next(iter(hashes)),
            **{k:first.get(k) for k in ('family','n','noise','problem_seed')},
            'known_f_star':0.0 if clean else None,'descriptive_f_ref':None if clean else best_loss,
            'reference_value':0.0 if clean else best_loss,
            'reference_kind':'known_noiseless_optimum' if clean else 'descriptive_observed_feasible_reference_not_certified_optimum',
            'best_observed_rounded_loss_float64':best_loss,'best_observed_source':source,
            'expected_runs':len(members),'eligible_final_runs':len(valid),'ineligible_final_runs':len(members)-len(valid),
            'included_budgets_seconds':budgets,'mixed_budgets':len(budgets)>1,
            'phase_counts':dict(sorted(Counter(r['phase'] for r in members).items())),
            'native_early_or_max_iter_runs':sum(r.get('native_early_stop',False) or r.get('native_max_iter_exhausted',False) for r in members),
            'valid_final_runs_meeting_budget':sum(r.get('budget_met',False) for r in valid)})
    return result


def write_outputs(source_path,out):
    payload=source_path.read_bytes()
    report=json.loads(payload)
    rows=references(report)  # Validate fully before creating any output files.
    names=['feasible_references.json','feasible_references.csv','feasible_references.md']
    if any((out/name).exists() for name in names):
        raise FileExistsError('Reference outputs already exist; preserve them and choose a new output directory.')
    sha=hashlib.sha256(payload).hexdigest()
    document={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_report':str(source_path.resolve()),'input_report_sha256':sha,'source_scope':report.get('scope'),
        'source_status':report['status'],'source_manifest_hashes':[m['sha256'] for m in report['manifests'] if m['phase'] in FINAL_PHASES],
        'scope':'Descriptive final-output references only; no recomputation, repair, checkpoint search, parameter tuning or optimizer selection.',
        'timing_policy':'Recorded output availability may precede final total solver time when a native epilogue returns the same previously repaired plan; both clocks are retained.',
        'mixed_budget_policy':'The minimum spans the explicitly listed included budgets and phases; it is not a matched-budget algorithm comparison.',
        'update_policy':'A lower independently evaluated feasible output requires a newly archived reference report; negative differences are never clipped.',
        'tie_policy':'Exact loss ties use lexicographic run_id only to identify a reproducible source.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'command':sys.argv,'instances':rows}
    csv_rows=[]
    lines=['# Feasible references','',document['scope'],'','Source scope: '+str(report.get('scope'))+'. Input report SHA256: `'+sha+'`.','',
           'Clean synthetic instances retain known f*=0 exactly. Their best observed values are recorded separately, including negative cancellation. Other references are observed feasible values, not certified optima. Mixed budgets remain explicit and do not define a fair optimizer ranking.','',
           '| Instance | Reference kind | Reference value | Best observed float64 loss | Eligible / expected | Budgets (s) | Source run | Source phase / method / budget (s) | Source solver / output time (s) | Rounded both-L1 |',
           '|---|---|---:|---:|---:|---|---|---|---:|---:|']
    for r in rows:
        source=r['best_observed_source'];res=source['final_rounded_residuals']
        flat={k:r[k] for k in ('instance','instance_sha256','family','n','noise','problem_seed','known_f_star','descriptive_f_ref','reference_value','reference_kind','best_observed_rounded_loss_float64','expected_runs','eligible_final_runs','ineligible_final_runs','mixed_budgets')}
        flat.update(included_budgets_seconds=json.dumps(r['included_budgets_seconds']),phase_counts=json.dumps(r['phase_counts']),input_report_sha256=sha)
        flat.update({'source_'+k:v for k,v in source.items() if k!='final_rounded_residuals'})
        flat.update({'source_rounded_'+k:v for k,v in res.items()});csv_rows.append(flat)
        number=lambda x:format(x,'.17g')
        lines.append(f"| {r['instance']} | {r['reference_kind']} | {number(r['reference_value'])} | {number(r['best_observed_rounded_loss_float64'])} | {r['eligible_final_runs']}/{r['expected_runs']} | {', '.join(map(number,r['included_budgets_seconds']))} | {source['run_id']} | {source['phase']} / {source['method']} / {number(source['budget'])} | {number(source['actual_solver_seconds'])} / {number(source['final_time_to_output'])} | {number(res['both_l1'])} |")
    lines += ['',document['mixed_budget_policy'],document['timing_policy'],document['update_policy'],'']
    out.mkdir(parents=True,exist_ok=True)
    (out/names[0]).write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    with (out/names[1]).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    (out/names[2]).write_text('\n'.join(lines))
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=write_outputs(args.report,args.out)
    print(json.dumps({'instances':len(result['instances']),'input_report_sha256':result['input_report_sha256'],'output':str(args.out.resolve())}))


if __name__=='__main__':
    main()
