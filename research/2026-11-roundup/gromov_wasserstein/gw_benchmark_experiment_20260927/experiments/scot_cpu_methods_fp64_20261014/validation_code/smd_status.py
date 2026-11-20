from pathlib import Path
import importlib.util,json,os,datetime as dt,platform
import numpy as np
import ot
root=Path('/home/tony/gw_benchmark_experiment_20260927')
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load(root/'extensions/scot_cpu_methods_fp64_20261014/run_comparison.py','controller')
f=load(root/'extensions/scot_reproduction_20261014/reference.py','reference')
out=c.DEFAULT_OUT;config,binding=c.configuration(out)
arrays={x:np.load(out/'reference/prepared/scot1047'/f'{x}.npy') for x in ['A','B','a','b']}
rows=[]
for method,candidate,seed in c.tune_cells(config):
 if method!='published-smd':continue
 audited=c.audited(out,config,binding,'tune',method,candidate,seed)
 path=Path(audited['directory']);metadata=c.read(path/'metadata.json')
 saved=metadata['ranks'][0]['saved_plans']['last_rounded'];plan=np.load(saved['file'])
 result=f.evaluate(arrays['A'],arrays['B'],plan,arrays['a'],arrays['b'],ot)
 discrepancy=abs(result['loss']-audited['last_rounded_loss_float64'])
 assert plan.dtype==np.float64 and result['independent_loss_agrees']
 assert result['residuals']['both_l1']<=100*np.finfo(np.float64).eps
 assert discrepancy<=1e-11
 rows.append(dict(name=c.name('tune',candidate,seed),candidate=candidate,seed=seed,status='passed',
  metadata_sha256=audited['metadata_sha256'],receipt_sha256=c.sha256(out/f'{c.name("tune",candidate,seed)}_receipt.json'),
  last_rounded_plan_sha256=c.sha256(saved['file']),dtype=str(plan.dtype),shape=list(plan.shape),
  completed_updates=metadata['counters']['completed_updates'],oracle_pairs=metadata['counters']['pair_samples'],
  inner_stopping_failures=metadata['counters']['inner_stopping_failures'],inner_acceptance_maxima=metadata['inner_acceptance_maxima'],
  controller_audit_passed=True,runner_loss=audited['last_rounded_loss_float64'],numpy_runner_absolute_difference=discrepancy,
  independent_numpy_pot=result))
record=dict(schema_version=1,status='passed',checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
 scope='Status audit of four completed SMD tuning runs only. Original controller validates source, input, configuration, oracle budget, certificates and all saved-plan hashes; independent NumPy and POT evaluate each actual saved FP64 last-rounded plan. No optimization or re-repair performed.',
 validator_path=__file__,validator_sha256=c.sha256(__file__),controller_sha256=c.sha256(c.__file__),
 config_sha256=c.sha256(out/'local_config.json'),source_file_sha256=config['source_file_sha256'],input_binding=binding,
 environment=dict(hostname=platform.node(),numpy=np.__version__,pot=ot.__version__,thread_environment={k:os.environ[k] for k in c.THREAD_ENV}),
 acceptance=dict(saved_rounded_combined_marginal_l1=100*np.finfo(np.float64).eps,independent_objective_absolute_difference=1e-11),runs=rows)
c.create_json(out/'status_smd_validation_20261014.json',record)
old=out/'launch_attempts/attempt1/local_config.json';oldcfg=c.read(old)
diffs={k:dict(archived=oldcfg.get(k),current=config.get(k)) for k in set(oldcfg)|set(config) if oldcfg.get(k)!=config.get(k)}
assert set(diffs)=={'created_at'}
incident=dict(schema_version=1,status='active_runs_preserved_recovery_pending',recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
 summary='The first detached controller survived. A sandbox process-visibility check incorrectly reported it and its workers absent; archiving live output paths and launching a second controller caused duplicate SMD work and an MSBPG receipt collision.',
 correction=dict(record=str(out/'launch_attempts/attempt1/ABORTED.json'),record_sha256=c.sha256(out/'launch_attempts/attempt1/ABORTED.json'),
  incorrect_claim='All recorded PIDs were confirmed absent; detached child processes did not survive termination of the exec session.',
  corrected_fact='An escalated host process-table check on 2026-10-14 showed original controller PID 2644975 alive, PPID 1, with four active MSBPG child workers. Sandboxed process-table visibility was insufficient to establish absence.'),
 original_controller=dict(pid=2644975,started_local='2026-10-14 14:33:57 Europe/Paris',current_children=[2652272,2652315,2652333,2652363]),
 duplicate_controller=dict(pid=2645356,service='scot-fp64-methods-20261014.service',exit_code=1,
  failure='FileExistsError on tune_msbpg00_seed810_receipt.json',failed_utc='2026-10-14T13:48:12.022955+00:00'),
 archived_config_sha256=c.sha256(old),current_config_sha256=c.sha256(out/'local_config.json'),config_differences=diffs,
 scientific_configuration_identical=True,
 smd=dict(status='four_current_final_runs_validated',validation_receipt=str(out/'status_smd_validation_20261014.json'),validation_receipt_sha256=c.sha256(out/'status_smd_validation_20261014.json'),
  evidence='Archived first-attempt worker logs and current second-attempt worker logs each report 2000 completed updates. Current completion receipts identify second-attempt worker PIDs and match current metadata and all saved-plan hashes.',
  caveat='The two controllers temporarily used overlapping SMD output paths; final current artifacts passed full controller and independent NumPy/POT endpoint audits. Timing is not comparable and is excluded from the scientific comparison.'),
 msbpg=dict(status='four_original_controller_workers_active_at_host_check',duplicate_runs_observed=False,
  precompletion_provenance_checked=True,
  evidence='All four active MSBPG receipts match the current configuration hash, full executed command, source identity, input binding and thread policy. Each worker is a child of the original controller. The second controller stopped before starting any MSBPG worker.'),
 recovery=dict(required='Wait for the original controller and four existing MSBPG workers to settle using host-visible PID identity checks, then independently audit all eight tuning results before selecting, confirming, finalizing and rendering the report.',
  reason='Original SMD futures used archived configuration SHA 23b907... and failed after local_config was replaced. The original executor waits for already-queued MSBPG tasks before surfacing that pending failure, so its campaign cannot automatically progress to selection.',
  preserve='Keep frozen numerical/controller sources and configuration unchanged, preserve all prior failed-stage and archived evidence, and record recovery under a new wrapper/service.'),
 evidence_files={str(p.relative_to(out)):c.sha256(p) for p in [out/'stage_campaign.json',out/'stage_tune_both_0_end.json',out/'systemd_campaign.log',out/'service_launch.json']})
c.create_json(out/'launch_incident_20261014.json',incident)
print(json.dumps(dict(status='passed',runs=[dict(name=x['name'],loss=x['independent_numpy_pot']['loss'],feasibility=x['independent_numpy_pot']['residuals']['both_l1']) for x in rows],validation_file=str(out/'status_smd_validation_20261014.json'),incident_file=str(out/'launch_incident_20261014.json'))))
