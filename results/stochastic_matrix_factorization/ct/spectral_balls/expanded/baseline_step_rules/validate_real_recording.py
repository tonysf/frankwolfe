"""One real update per factory through the recorder; no production cases."""
import json
from pathlib import Path
import tempfile

import campaign
from run import run


def validate():
    root=campaign.ROOT;protocol=campaign.verify_prepared()
    fixture=campaign.read(root/'method_smoke.json');records=[]
    for method in ('palm','prox_sarah'):
        previous=next(c for c in fixture['cases'] if c['method']==method and c['n_train']==360)
        cfg=campaign.config('evaluation','slice03611',360,203,method,None if method=='palm' else 0)
        cfg['run_id']='recording_smoke_'+cfg['run_id']
        cfg['view_budget']=5040 if method=='palm' else 360
        folder=root/'smoke_runs'/cfg['run_id']
        run(cfg,cpu=6,runs_dir=root/'smoke_runs')
        meta,result,trace=(campaign.read(folder/name) for name in ('metadata.json','result.json','trace.json'))
        assert len(trace)==2 and [r['iteration'] for r in trace]==[0,1]
        assert meta['optimizer']==previous['optimizer']
        assert trace[0]['raw_loss']==previous['trace'][0]['loss']
        assert trace[1]['raw_loss']==previous['trace'][1]['loss']
        assert trace[1]['last_step']==previous['trace'][1]['step']
        if method=='palm':old_id='evaluation_slice03611_360_palm_seed202_a2p3'
        else:old_id='evaluation_slice03611_360_prox_sarah_seed203_a2p3_m96_sminibatch'
        plan=dict(matched_run_ids={cfg['run_id']:old_id},
                  matched_source_directories={cfg['run_id']:str(campaign.OLD/'runs'/old_id)},
                  groups={'slice03611':'Mix2'})
        with tempfile.TemporaryDirectory(prefix='baseline-record-audit-') as temp:
            local=Path(temp);(local/'runs').symlink_to(root/'smoke_runs',target_is_directory=True)
            (local/'implementation_manifest.json').symlink_to(root/'implementation_manifest.json')
            campaign.ROOT=local
            try:checked=campaign.validate_result(cfg,6,plan,protocol)
            finally:campaign.ROOT=root
        assert checked['status']=='completed' and checked['endpoint']['forward_views']==cfg['view_budget']
        hashes={name:campaign.digest(folder/name) for name in ('metadata.json','trace.json','result.json')}
        run(cfg,cpu=6,runs_dir=root/'smoke_runs')
        assert hashes=={name:campaign.digest(folder/name) for name in hashes}
        records.append(dict(method=method,source_directory=str(folder),run_id=cfg['run_id'],
            source_file_hashes=hashes,matched_prior_direct_solver_smoke=True,
            campaign_result_validator_passed=True,historical_initialization_exact=True,
            completed_rerun_idempotent=True,first_step=trace[1]['last_step'],
            initial_raw_loss=trace[0]['raw_loss'],raw_loss=trace[1]['raw_loss'],
            relative_loss=trace[1]['raw_loss']/trace[0]['raw_loss']))
    assert not (root/'runs').exists() and not (root/'selection.json').exists()
    return dict(status='passed',source_sha256=protocol['numerical_sources'],
        test_source_sha256=campaign.digest(Path(__file__)),
        protocol_sha256=campaign.digest(root/'protocol.json'),
        prior_method_smoke_sha256=campaign.digest(root/'method_smoke.json'),cases=records,
        production_runs_or_selection_created=False)


if __name__=='__main__':
    result=validate();campaign.write(campaign.ROOT/'real_engine_validation.json',result)
    print(json.dumps(result,indent=2))
