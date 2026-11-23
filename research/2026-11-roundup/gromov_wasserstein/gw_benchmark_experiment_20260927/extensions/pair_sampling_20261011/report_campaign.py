"""Report completed fixed-horizon comparisons and preserve every missing/failure state."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import statistics

from campaign import (METHODS,arguments,directory,expected_config,finite,identity,load_config,
                      read,require,sha256,validate_gate,validate_metadata,write)


LABELS={'smoothing-fw':'FraMES with gradient momentum',
        'published-smd':'Stochastic entropic mirror descent with fresh gradients',
        'published-msbpg':'MSBPG with the published coupled step and momentum schedules'}


def audit(config_path,source,prepared,out):
    c=load_config(config_path);records=[];gates=[]
    for dataset,instance in c['instances'].items():
        provenance=identity(c,source,prepared,dataset)
        for method in METHODS:
            gate_path=Path(out).parent/'preflight'/dataset/method/'gate.json'
            gates.append(dict(dataset=dataset,method=method,file=str(gate_path),
                              result=read(gate_path) if gate_path.exists() else dict(status='missing')))
            for seed in c['seeds']:
                target=directory(out,dataset,method,seed);record=dict(dataset=dataset,method=method,method_name=LABELS[method],seed=seed,
                    initialization='product',oracle='pair-sampling',pairs_per_call=c['pairs_per_call'],status='missing')
                try:
                    status_path=target/'status.json'
                    if not status_path.exists():
                        progress=target/'output/progress.json'
                        if progress.exists():record.update(status='started_without_result',progress=read(progress))
                        else:
                            failures=list((target/'output').glob('failure*.json'))+list((target/'output').glob('rank*/inner_failure.json'))
                            if failures:record.update(status='failed',failures=[read(p) for p in failures])
                        records.append(record);continue
                    status=read(status_path)
                    if status['status']!='completed':record.update(status='failed',failure=status);records.append(record);continue
                    request_path=target/'request.json';request=read(request_path)
                    require(status['returncode']==0 and status['request_sha256']==sha256(request_path),'Invalid controller completion receipt.')
                    for key,value in provenance.items():require(request[key]==value,'Request provenance differs: '+key)
                    require(request['gate']==validate_gate(c,source,prepared,gate_path,dataset,method),'Production gate receipt differs.')
                    values=arguments(c,prepared,target/'output',dataset,method,seed,'production')
                    expected=expected_config(source,values)
                    require(request['expected_runner_config']==expected,'Frozen request differs.')
                    path=target/'output/metadata.json';require(status['metadata_sha256']==sha256(path),'Numerical receipt changed.')
                    metadata=read(path);loss=validate_metadata(metadata,expected,provenance,instance,c['final_marginal_l1_tolerance'])
                    last=[r for r in metadata['trace'] if r['kind']=='last'];require(len(last)==1,'Missing last-iterate diagnostic.')
                    record.update(status='completed',last_rounded_loss=loss,solver_seconds=metadata['timing']['solver_seconds'],
                                  last_feasible_output_seconds=last[0]['time_to_this_feasible_output'],
                                  total_elapsed_seconds=metadata['timing']['total_elapsed_seconds'],
                                  completed_updates=expected['iterations'],stochastic_gradient_calls=metadata['counters']['stochastic_gradient_calls'],
                                  pair_samples=metadata['counters']['pair_samples'],inner_iterations=metadata['counters']['inner_iterations_total'],
                                  selected_raw_loss=(metadata['final_float64_validation']['plans']['selected_raw']['loss'] if method!='smoothing-fw' else None),
                                  selected_index=metadata['output_selection']['index'],metadata_file=str(path),metadata_sha256=sha256(path),
                                  observed_current_trajectory=[{key:r[key] for key in ('iteration','rounded_loss','time_to_this_feasible_output')}
                                      for r in metadata['trace'] if r['kind'] in ('initial','current_diagnostic','last')])
                except Exception as exc:record.update(status='invalid',error=repr(exc))
                records.append(record)
    summaries=[]
    for dataset in c['instances']:
        for method in METHODS:
            valid=[r for r in records if r['dataset']==dataset and r['method']==method and r['status']=='completed']
            complete={r['seed'] for r in valid}==set(c['seeds'])
            summaries.append(dict(dataset=dataset,method=method,method_name=LABELS[method],status='completed' if complete else 'incomplete',
                seeds=[r['seed'] for r in valid],medians={key:statistics.median(r[key] for r in valid)
                    for key in ('last_rounded_loss','solver_seconds','last_feasible_output_seconds','total_elapsed_seconds')}
                    if complete else None))
    counts={name:sum(r['status']==name for r in records) for name in ('completed','missing','started_without_result','failed','invalid')}
    return dict(status='completed' if counts['completed']==36 else 'incomplete',config_sha256=sha256(config_path),
                counts=counts,runs=records,summaries=summaries,gates=gates,
                scope='Fixed-prefactor comparison: three stochastic methods use the same pair-sampling oracle and pair budget. Real-data prefactors were previously selected using Rademacher sketches; no retuning or loss-based stopping. This is not a ranking of independently retuned best configurations.',
                interpretation='Independent sampled baseline output and last-rounded GW loss are separate. Neither loss establishes global optimality or stationarity. Fixed 1e-7 proximal accuracy has no claimed inexact-outer theorem.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','source','prepared','out','report-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.report_dir.mkdir(parents=True,exist_ok=True)
    report=audit(a.config,a.source,a.prepared,a.out);write(a.report_dir/'report.json',report)
    fields=['dataset','method','method_name','seed','status','completed_updates','stochastic_gradient_calls','pair_samples','last_rounded_loss',
            'selected_raw_loss','selected_index','solver_seconds','last_feasible_output_seconds','total_elapsed_seconds','inner_iterations','error']
    with (a.report_dir/'report.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['runs'])
    print(report['status'],report['counts'],flush=True)


if __name__=='__main__':main()
