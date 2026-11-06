"""Declare a calibration-only diagnostic; no scientific runs are performed."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
COMMIT='c7a8c9789b7814ede1c95e53eef13bfe1dd58c63'
step={'scale':2.0,'offset':2.0,'exponent':1.0,'cap':1.0}
fast={'scale':4.0,'offset':4.0,'exponent':1.0,'cap':1.0}
configs=[
 {'id':'q-original-anchor','method':'printed-adafgd','alpha':.5,'eta_scale':.01,'tau':2.0},
 {'id':'q-a5e-4-e3e-1','method':'printed-adafgd','alpha':5e-4,'eta_scale':.3,'tau':2.0},
 {'id':'armijo-a5e-4','method':'armijo-fgd','alpha':5e-4,'eta_scale':.01,'tau':2.0},
 {'id':'f-original-anchor','method':'frames','alpha':5e-5,'eta_scale':.01,'tau':2.0},
 {'id':'f-fast-average','method':'frames','alpha':5e-5,'eta_scale':.01,'tau':2.0,'step_schedule':fast},
 {'id':'f-tau2p02','method':'frames','alpha':5e-5,'eta_scale':.01,'tau':2.02},
 {'id':'f-tau2p1','method':'frames','alpha':5e-5,'eta_scale':.01,'tau':2.1},
 {'id':'f-lower-alpha','method':'frames','alpha':5e-6,'eta_scale':.01,'tau':2.0},
 {'id':'f-lower-alpha-fast','method':'frames','alpha':5e-6,'eta_scale':.01,'tau':2.0,'step_schedule':fast},
 {'id':'armijo-original-alpha','method':'armijo-fgd','alpha':.5,'eta_scale':.01,'tau':2.0},
 {'id':'armijo-a5e-5','method':'armijo-fgd','alpha':5e-5,'eta_scale':.01,'tau':2.0}]
for aname,alpha in [('5e-6',5e-6),('5e-5',5e-5),('5e-4',5e-4),('5e-3',5e-3)]:
    for ename,eta in [('3e-2',.03),('1e-1',.1),('3e-1',.3),('1',1.0),('3',3.0)]:
        name=f'q-a{aname}-e{ename}'
        if name not in {c['id'] for c in configs}:
            configs.append({'id':name,'method':'printed-adafgd','alpha':alpha,'eta_scale':eta,'tau':2.0})
assert len(configs)==30
trials=[dict(config,id=f"c{channel}-{config['id']}",channel=channel)
        for channel in [0,1] for config in configs]
plan={
 'schema_version':1,'kind':'qpt_calibration_resolution_diagnostic',
 'campaign_id':'qpt_resolution_20261006_1748','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source':{'commit':COMMIT,'repo':'/gpfs/workdir/silvetian/frankwolfe-quiroga-'+COMMIT},
 'shared':{'n_qubits':2,'rank':1,'active_rows':128,'row_epochs':10000,'checkpoint_epochs':100,
           'batch_size':32,'frames_penalty_exponent':.25,
           'rho':{'scale':2.0,'offset':4.0,'exponent':.6,'cap':1.0},'step':step,
           'observation_mode':'noiseless','initialization':'trace_normalized_choi_uniform',
           'gate':{'fidelity_min':.99,'rmse_max':1e-6,'normalized_tp_max':.001}},
 'channels':[0,1],'trials':trials,
 'resources':{'minimum_seconds_before_new_trial':120,'account':'fwllm','partition':'cpu_short',
              'cpus_per_task':2,'memory':'1G','walltime_seconds':1200,'array':'0-1%1'},
 'role':'exploratory calibration diagnostic; not primary comparison selection or unseen validation',
 'seed_rule':{'initialization':'200000+10*channel','sampling':'300000+100*channel'},
 'questions':[
    'Does a larger printed multiplier recover low-penalty candidates which were at the old upper step-size grid boundary?',
    'Can Armijo FGD recover the same data and initial factor under fixed penalties? It is a diagnostic control, not Quiroga reproduction.',
    'Does FRAMES improve when averaging is faster, the norm ball has modest slack, or its initial penalty is reduced?'],
 'budget_qualification':'Q10k updates and F40k updates at batch32 each charge1.28M nominal rows when complete. Armijo is limited to10k iterations and separately reports all gradient/trial-objective row charges, not an equal-budget primary competitor.',
 'analysis_policy':'No selection or reranking from validation channels100..107; no heldout data. Preserve all failures. Two calibration channels and one sampling stream are a diagnostic pilot, not a best-for-best final result. Any expanded calibration requires its own frozen selection and fresh confirmation.',
 'preservation':'Earlier calibration, frozen manifest and validation outputs remain unchanged. Optimizer source remains pinned; there are no update-equation fixes in this pilot.',
 'expected_counts':{'trials':60,'trials_per_channel':30,'printed':42,'frames':12,'armijo':6}}
out=ROOT/'diagnostic_plan.json'
with out.open('x') as stream:json.dump(plan,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'trials':len(trials)}))
