"""One real CT iteration through the prepared recorder, outside production runs."""
from copy import deepcopy
import json
import math
from pathlib import Path

import campaign
from run import run


def validate():
    protocol=campaign.verify_prepared()
    plan=campaign.read(campaign.ROOT/'evaluation_plan.json')
    original=next(c for c in plan['runs'] if
                  c['slice_id']=='slice03611' and c['n_train']==360 and c['seed']==203)
    cfg=deepcopy(original)
    cfg['run_id']='real_engine_smoke_'+original['run_id']
    cfg['view_budget']=2*cfg['n_train']+768
    folder=campaign.ROOT/'smoke_runs'/cfg['run_id']
    run(cfg,cpu=6,runs_dir=campaign.ROOT/'smoke_runs')
    meta,result,trace=(campaign.read(folder/name) for name in ('metadata.json','result.json','trace.json'))
    historical=campaign.OLD/'runs'/plan['matched_run_ids'][original['run_id']]
    old_meta=campaign.read(historical/'metadata.json')
    old_initial=campaign.read(historical/'trace.json')[0]
    assert len(trace)==2 and [r['iteration'] for r in trace]==[0,1]
    assert all(trace[0][k]==v for k,v in old_initial.items())
    assert all(meta[k]==old_meta[k] for k in campaign.PROBLEM_FIELDS)
    assert meta['optimizer']==campaign.PARAMS
    assert meta['source_sha256']==protocol['numerical_sources']
    assert meta['recovery_origin']=='fresh' and meta['continuation'] is None and meta['imported_checkpoint'] is None
    assert result['status']=='completed' and result['final']==trace[-1]
    assert all(result.get(k)==v for k,v in meta.items())
    end=trace[-1]
    expected=dict(forward_views=1488,backward_views=1488,gradient_calls=2,oracle_calls=14,
                  curvature_operator_calls=12,curvature_forward_views=768,curvature_backward_views=768,
                  projection_calls=2)
    assert all(end[k]==v for k,v in expected.items())
    assert end['last_step']['refresh_u'] and end['last_step']['refresh_v']
    assert end['last_step']['component_gradient_calls']==720
    for block in ('u','v'):
        assert end['last_step'][f'curvature_sample_views_{block}']==64
        assert math.isclose(end['last_step'][f'eta_{block}']*end['last_step'][f'curvature_{block}'],.5,rel_tol=1e-13)
    hashes={name:campaign.digest(folder/name) for name in ('metadata.json','trace.json','result.json')}
    run(cfg,cpu=6,runs_dir=campaign.ROOT/'smoke_runs')
    assert hashes=={name:campaign.digest(folder/name) for name in hashes}
    assert not (campaign.ROOT/'runs').exists()
    return dict(status='passed',source_sha256=protocol['numerical_sources'],
        test_source_sha256=campaign.digest(Path(__file__)),
        protocol_sha256=campaign.digest(campaign.ROOT/'protocol.json'),source_directory=str(folder),
        matched_historical_run=str(historical),source_file_hashes=hashes,
        historical_initial_diagnostics_exact=True,historical_problem_metadata_exact=True,
        first_step_counters=expected,eta_times_current_curvature=.5,
        initial_raw_loss=trace[0]['raw_loss'],final_raw_loss=end['raw_loss'],
        relative_raw_loss=end['raw_loss']/trace[0]['raw_loss'],
        completed_rerun_idempotent=True,production_runs_created=False)


if __name__=='__main__':
    result=validate()
    campaign.write(campaign.ROOT/'real_engine_validation.json',result)
    print(json.dumps(result,indent=2))
