"""Bounded bookkeeping fixtures; no new optimizer or CT operator evaluations."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import campaign


def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid fixture was not rejected')


def case_fixture(method,n,case,meta,prefix,folder,protocol):
    origin=Path(case['source_directory']);run_id=f'fixture_{method}_{n}'
    cfg={k:meta['config'][k] for k in ('phase','method','slice_id','n_train','seed','grid','rank','calibration_seed')}
    cfg.update(run_id=run_id,record_every_iteration=True,gradient_view_budget=campaign.CAP,
        origin_directory=str(origin),origin_run_id=meta['config']['run_id'],origin_config=meta['config'],
        **{f'origin_{key}_sha256':campaign.digest(origin/filename) for key,filename in
           (('checkpoint','checkpoint.pkl'),('metadata','metadata.json'),('trace','trace.json'),('result','result.json'))})
    trace=deepcopy(prefix)
    while campaign.gradient_views(trace[-1])<campaign.CAP:
        row=deepcopy(trace[-1]);step=row['last_step'];row['iteration']+=1
        for name in campaign.COUNTERS:row[name]+=step[name]
        row['gradient_forward_views']=campaign.gradient_views(row)
        row['gradient_backward_views']=campaign.gradient_views(row,'backward')
        trace.append(row)
    continuation=dict(source_directory=str(origin.resolve()),run_id=cfg['origin_run_id'],iteration=prefix[-1]['iteration'],
        gradient_forward_views=campaign.gradient_views(prefix[-1]),gradient_backward_views=campaign.gradient_views(prefix[-1],'backward'),
        source_metadata_sha256=cfg['origin_metadata_sha256'],source_trace_sha256=cfg['origin_trace_sha256'],
        source_checkpoint_sha256=cfg['origin_checkpoint_sha256'],source_result_sha256=cfg['origin_result_sha256'],
        source_code_sha256=meta['source_sha256'],prefix_observations=len(prefix),optimizer_and_rng_state_restored=True,
        source_trace_prefix_preserved=True,includes_source_final_update_past_inclusive_view_cap=True)
    new=deepcopy(meta);new.update(config=cfg,cpu=6,source_sha256=protocol['numerical_sources'],
        implementation_manifest_sha256=protocol['implementation_manifest_sha256'],specification_sha256=protocol['specification_sha256'],
        authorized_implementation_changes=campaign.read(folder/'implementation_manifest.json')['authorized_changes'],
        continuation=continuation,imported_checkpoint=str(origin/'checkpoint.pkl'),recovery_origin='continued_from_completed_checkpoint',
        budget=dict(metric='gradient_views',per_direction=campaign.CAP,definition='total_views - curvature_views',cumulative_from_initialization=True))
    target=folder/'runs'/run_id;target.mkdir(parents=True)
    def save():
        campaign.write(target/'metadata.json',new)
        campaign.write(target/'trace.json',trace)
        campaign.write(target/'result.json',dict(new,status='completed',iteration=trace[-1]['iteration'],final=trace[-1]))
    save();plan={'groups':{cfg['slice_id']:case['group']}}
    result=campaign.validate_result(cfg,6,plan,protocol)
    assert result['prefix_preserved'] and result['endpoint']['iteration']>=prefix[-1]['iteration']
    assert result['inclusive_endpoint']==[r for r in prefix if r['forward_views']<=campaign.CAP][-1]
    assert result['gradient_forward_views']<=campaign.CAP<=campaign.gradient_views(result['final_observation'])
    if method=='palm':assert result['endpoint']['iteration']==(96 if n==360 else 32)
    saved=trace[0]['raw_loss'];trace[0]['raw_loss']+=.01;save()
    rejected(lambda:campaign.validate_result(cfg,6,plan,protocol));trace[0]['raw_loss']=saved
    saved=trace[-1]['gradient_forward_views'];trace[-1]['gradient_forward_views']+=1;save()
    rejected(lambda:campaign.validate_result(cfg,6,plan,protocol));trace[-1]['gradient_forward_views']=saved
    trace[-1].pop('gradient_backward_views');save()
    rejected(lambda:campaign.validate_result(cfg,6,plan,protocol));trace[-1]['gradient_backward_views']=campaign.gradient_views(trace[-1],'backward')
    new['continuation']['source_checkpoint_sha256']='tampered';save()
    rejected(lambda:campaign.validate_result(cfg,6,plan,protocol));new['continuation']['source_checkpoint_sha256']=cfg['origin_checkpoint_sha256']
    save();assert campaign.validate_result(cfg,6,plan,protocol)==result
    return dict(method=method,n_train=n,endpoint_iteration=result['endpoint']['iteration'],
        endpoint_gradient_views=result['gradient_forward_views'],final_gradient_overshoot=result['final_gradient_overshoot'],
        original_inclusive_endpoint_unchanged=True,source_prefix_exact=True,
        prefix_tampering_rejected=True,counter_tampering_rejected=True,missing_alias_rejected=True,
        continuation_provenance_tampering_rejected=True)


def main():
    root=campaign.ROOT;records=campaign.source_records()
    report=campaign.read(campaign.BASE/'report_update.json')
    groups=campaign.read(report['corrected_summary'])['design']['evaluation_slice_groups']
    checked=[]
    try:
        with tempfile.TemporaryDirectory(prefix='gradient_campaign_fixture_') as temporary:
            campaign.ROOT=folder=Path(temporary)
            campaign.write(folder/'specification.json',campaign.specification())
            campaign.write(folder/'implementation_manifest.json',dict(authorized_changes={'run.py':'fixture continuation engine'}))
            protocol=dict(numerical_sources={'run.py':'fixture'},implementation_manifest_sha256=campaign.digest(folder/'implementation_manifest.json'),
                specification_sha256=campaign.digest(folder/'specification.json'))
            plan=campaign.make_plan(records,groups)
            assert len(plan['runs'])==168 and len(plan['assignments'])==168
            assert all(sum(cfg['method']==method for cfg in plan['runs'])==84 for method in ('palm','spring_sarah'))
            assert set(plan['assignments'].values())==set(campaign.CPUS)
            assert campaign.make_plan(records,groups)==plan
            for method in ('palm','spring_sarah'):
                for n in (360,1080):
                    m,case,meta,prefix=next(record for record in records if record[0]==method and record[1]['n_train']==n)
                    case=dict(case,group=groups[case['slice_id']])
                    checked.append(case_fixture(method,n,case,meta,prefix,folder,protocol))
    finally:campaign.ROOT=root
    result=dict(status='passed',campaign_sha256=campaign.digest(root/'campaign.py'),
        validator_sha256=campaign.digest(Path(__file__)),no_real_ct_operator_or_optimizer_evaluation=True,
        physical_extensions=168,unchanged_origin_files_verified=True,deterministic_balanced_assignment=True,cases=checked)
    campaign.write(root/'campaign_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
