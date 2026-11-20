"""Freeze a small independent FRAMES grid before any grid outcome is observed."""
from run_local import ROOT,COMMIT,source,write,now,digest
base=dict(alpha0=5e-5,penalty_exponent=.25,step_scale=4.,step_offset=4.,step_exponent=1.,
          rho_scale=2.,rho_offset=4.,rho_exponent=.6,tau=2.,batch_size=32)
configs=[]
for ai,alpha in enumerate((5e-6,5e-5,5e-4)):
    for pi,power in enumerate((.125,.25)):
        configs.append(dict(base,id=f'penalty-a{ai}-p{pi}',alpha0=alpha,penalty_exponent=power))
configs += [dict(base,id='step2',step_scale=2.,step_offset=2.),
            dict(base,id='step8',step_scale=8.,step_offset=8.),
            dict(base,id='rho-standard',rho_scale=4.,rho_offset=8.,rho_exponent=2/3)]
trials=[{**c,'config_id':c['id'],'id':f"{c['id']}-c{channel}-s0",'channel':channel,
          'steps':100000,'metrics_every':4000,'sampling_replicate':0} for c in configs for channel in (0,1)]
plan=dict(schema_version=1,scope='local_n2_full_noiseless_frames_calibration',phase='screen',
    backend='jax_dense_cpu64_scan',source_commit=COMMIT,source_sha256=source(),created_utc=now(),
    harness_sha256={k:digest(ROOT/k) for k in ('run_local.py','jax_backend.py','make_screen_plan.py')},
    gate={'fidelity_min':.99,'rmse_max':1e-6,'normalized_tp_max':.001},trials=trials,
    incumbent_config_id='penalty-a1-p1',
    screen_selection_rule=['ascending_failed_trials','descending_gate_passes','ascending_worst_endpoint_deficit','ascending_mean_log_deficit','config_id'],
    long_stage={'steps':400000,'metrics_every':4000,'channels':[0,1],
                'promotion':'top two distinct configurations plus incumbent, deduplicated; fresh starts, no fake resume',
                'selection':'Require both endpoint joint gates. Minimize worst persistent-recorded gate step, then worst endpoint deficit, then config_id.'},
    confirmation={'channels':list(range(2,8)),'steps':400000,'sampling_replicates':[0,1],
                  'freeze_before_any_confirmation':True,'role':'Previously exposed channels, not unseen final test.'},
    final_heldout={'channels':list(range(1000,1020)),'use':'FORBIDDEN in this calibration'},
    resources={'max_workers':2,'threads_per_worker':1,'cpu_affinity':[0,1],'nice':10,
               'maximum_wall_seconds_per_batch':3600},
    notes=['No Ruche use. No Quiroga retuning. FRAMES penalties independent of FGD.',
           'Fixed batch32/tau2. All p>0; alpha0 fixed at initialization then grows with absolute update index.',
           'Grid is exploratory calibration, not a final best-for-best timing comparison; scan includes compilation and lacks interior wall times.',
           'Record all numerical failures. No winner if long stage has no all-channel pass; open a separately declared calibration round.'])
write(ROOT/'plan_screen.json',plan)
print(ROOT/'plan_screen.json')
