"""Controller contract tests without Slurm submissions or data processing."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wallclock_campaign_tests_module', HERE / 'campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


@pytest.fixture
def config():
    return campaign.load_config(HERE / 'campaign.json')


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    ext = root / campaign.EXT
    ext.mkdir(parents=True)
    (ext / 'campaign.json').write_bytes((HERE / 'campaign.json').read_bytes())
    old = root / 'extensions/large_pot_frames_20261016'
    old.mkdir()
    (old / 'campaign.json').write_bytes((HERE.parent / 'large_pot_frames_20261016/campaign.json').read_bytes())
    (ext / 'runner.py').write_text('# numerical source fixture\n')
    (root / 'SHA256SUMS').write_text('fixture manifest\n')
    return root


def test_exact_eight_preselected_runs(config):
    tasks = campaign.tasks(config)
    assert len(tasks) == 8
    assert [task['dataset'] for task in tasks] == ['bmmc6224'] * 4 + ['snare_full'] * 4
    assert [task['method'] for task in tasks] == ['pot-full','frames','frames','frames'] * 2
    assert [task['seed'] for task in tasks] == [810,811,812,813] * 2
    assert [task['iterations'] for task in tasks] == [2000,20000,20000,20000] * 2
    assert tasks[1]['prefactors'] == dict(gamma0=.3,rho0=1.,mu_scale=10.)
    assert tasks[5]['prefactors'] == dict(gamma0=.3,rho0=.1,mu_scale=10.)


@pytest.mark.parametrize('change', ['seed','grid','data','parallel','retention','tolerance','diagnostics'])
def test_config_rejects_scientific_or_storage_changes(source,change):
    path=source / campaign.EXT / 'campaign.json'
    c=campaign.read(path)
    if change=='seed': c['frames']['confirmation_seeds'][0]=810
    if change=='grid': c['frames']['selected']['bmmc6224']['prefactors']['mu_scale']=100.
    if change=='data': c['instances']['snare_full']['array_files_sha256']['A.npy']='0'*64
    if change=='parallel': c['scheduler']['maximum_concurrent_gpu_jobs']=2
    if change=='retention': c['storage']['completed_checkpoint_retention']=True
    if change=='tolerance': c['inner_tol']=1e-6
    if change=='diagnostics': c['runner']['diagnostic_every']=100
    campaign.write(path,c)
    with pytest.raises(ValueError): campaign.load_config(path)


def test_runner_is_fixed_one_gpu_and_prunes_final_states(config,tmp_path):
    for task in campaign.tasks(config):
        cmd=campaign.runner_command('/source',tmp_path,config,task,1234,tmp_path/'checkpoints')
        assert '--retain-checkpoint' not in cmd
        assert 'torch.distributed.run' not in cmd
        assert cmd[1]=='/source/extensions/wallclock_replay_20261018/runner.py'
        options=dict(zip(cmd[2::2],cmd[3::2]))
        assert options['--checkpoint-dir']==str(tmp_path/'checkpoints'/task['dataset']/task['task_id'])
        assert options['--out']==str(tmp_path/'results'/task['dataset']/task['task_id'])
        assert options['--diagnostic-every']=='500'
        assert options['--pairs']=='1024' and options['--dtype']=='float64'


def test_check_storage_rejects_external_checkpoint_root(config,tmp_path):
    root=tmp_path/'campaign'; root.mkdir()
    external=tmp_path/'external'; external.mkdir()
    receipt=dict(quota_verified=True,quota_available_bytes=config["storage"]["required_quota_available_bytes"],checkpoint_root=str(external),verified_utc=campaign.now())
    campaign.write(root/'storage_receipt.json',receipt)
    with pytest.raises(ValueError,match='isolated'): campaign.check_storage(root,config,fresh=True)
    checkpoint=root/'checkpoints'; checkpoint.mkdir()
    receipt['checkpoint_root']=str(checkpoint)
    campaign.write(root/'storage_receipt.json',receipt)
    assert campaign.check_storage(root,config,fresh=True)==checkpoint


def test_reference_comparison_requires_every_measured_diagnostic():
    task=dict(dataset='bmmc6224',task_id='preflight/frames',reference_task_id='confirmation/c08_seed811')
    refs={'datasets':{'bmmc6224':{'runs':{'confirmation/c08_seed811':{'trace':[{'iteration':0,'rounded_loss':.2},{'iteration':1,'rounded_loss':.19}]}}}}}
    meta={'trace':[{'iteration':0,'rounded_loss':.2},{'iteration':1,'rounded_loss':.19}]}
    assert all(p['absolute_error']==0 for p in campaign.compare_reference(meta,task,refs,1e-11))
    meta['trace'][1]['rounded_loss']+=1e-6
    with pytest.raises(ValueError,match='trajectory'): campaign.compare_reference(meta,task,refs,1e-11)
    meta['trace'][1]={'iteration':2,'rounded_loss':.19}
    with pytest.raises(ValueError,match='lacks'): campaign.compare_reference(meta,task,refs,1e-11)


def metadata(config,task,source):
    instance=config['instances'][task['dataset']]
    return dict(status='completed',completed_updates=task['iterations'],target_updates=task['iterations'],
        config=campaign.expected_config(config,task),array_files_sha256=instance['array_files_sha256'],
        manifest_sha256=instance['instance_manifest_sha256'],world_size=1,dimensions=dict(n=instance['n'],m=instance['n']),
        identity=dict(device_model='NVIDIA A100-SXM4-40GB'),retained_checkpoint=False,
        source_file_sha256={str(campaign.EXT/'runner.py'):campaign.sha256(source/campaign.EXT/'runner.py')},
        schedule=dict(inner_tolerance=1e-7,gap_tolerance=1e-7,eta=1000.,gamma_exponent=.75,rho_exponent=.5,mu_exponent=.25),
        trace=[dict(iteration=0,elapsed_seconds=1.,rounded_loss=.2),dict(iteration=task['iterations'],elapsed_seconds=10.,rounded_loss=.19)],
        elapsed_seconds=11.,final_loss=.19)


def test_result_checks_elapsed_times(config,source):
    task=campaign.tasks(config)[0]
    meta=metadata(config,task,source)
    assert campaign.validate_result(meta,task,config,source)==.19
    for field,value in [('elapsed_seconds',float('nan')),('elapsed_seconds',.5),('iteration',0)]:
        bad=copy.deepcopy(meta);bad['trace'][-1][field]=value
        with pytest.raises(ValueError):campaign.validate_result(bad,task,config,source)
    bad=copy.deepcopy(meta);bad['elapsed_seconds']=9.
    with pytest.raises(ValueError,match='Final time'):campaign.validate_result(bad,task,config,source)
    bad=copy.deepcopy(meta);bad['identity']['device_model']='V100'
    with pytest.raises(ValueError,match='A100'):campaign.validate_result(bad,task,config,source)


def test_completed_confirmation_cleanup_is_idempotent_and_confined(config,tmp_path):
    task=campaign.tasks(config)[1];dataset=task['dataset']
    base=tmp_path/'checkpoints';folder=base/dataset/task['task_id'];generation=folder/'generation-000000020000-test';generation.mkdir(parents=True)
    shard=generation/'rank-0000.pt';shard.write_bytes(b'fixture')
    record=dict(iteration=20000,identity={'fixture':True},generation=generation.name,shards=[{'file':shard.name,'bytes':7,'sha256':campaign.sha256(shard)}])
    campaign.write(generation/'manifest.json',record);campaign.write(folder/'COMMITTED.json',record)
    meta=dict(status='completed',completed_updates=20000,target_updates=20000,retained_checkpoint=False,checkpoint=record,identity=record['identity'],world_size=1)
    # A completed replay confirmation is pruned; SHARE confirmations are unaffected.
    campaign.reconcile_completed_checkpoint(base,dataset,task,meta)
    assert list(folder.iterdir())==[]
    campaign.reconcile_completed_checkpoint(base,dataset,task,meta)
    outside=tmp_path/'outside';outside.mkdir();(outside/'keep').write_text('keep')
    folder.rmdir();folder.symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):campaign.reconcile_completed_checkpoint(base,dataset,task,meta)
    assert (outside/'keep').read_text()=='keep'


def test_checkpoint_cursor_rejects_missing_state(config,tmp_path,source):
    checkpoints=tmp_path/'checkpoints';checkpoints.mkdir()
    cursor=campaign.checkpoint_cursor(source,tmp_path,config,checkpoints)
    assert cursor['task_index']==0 and not cursor['has_checkpoint']
    task=campaign.tasks(config)[0]
    campaign.write(tmp_path/'results'/task['dataset']/task['task_id']/'progress.json',dict(status='running'))
    with pytest.raises(ValueError,match='lacks committed'):campaign.checkpoint_cursor(source,tmp_path,config,checkpoints)


def test_sbatch_is_idempotent_and_refuses_parallel_stream(config,tmp_path,source,monkeypatch):
    root=tmp_path/'run';root.mkdir()
    called=[]
    def fake_run(command,**kwargs):
        called.append(command)
        return SimpleNamespace(returncode=0,stdout='1234\n',stderr='')
    monkeypatch.setattr(campaign.subprocess,'run',fake_run)
    assert campaign.submit_job(source,root,'production',request_key='fixed')=='1234'
    assert campaign.submit_job(source,root,'production',request_key='fixed')=='1234'
    assert len(called)==1 and '--gres=gpu:1' in called[0]
    with pytest.raises(ValueError,match='serial'):campaign.submit_job(source,root,'production',world=2)
    with pytest.raises(ValueError,match='serial'):campaign.submit_job(source,root,'production',dataset='bmmc6224')
    request=root/'submission_requests'/(campaign.hashlib.sha256(b'fixed').hexdigest()+'.json')
    prior=campaign.read(request);prior['status']='submitting';campaign.write(request,prior)
    with pytest.raises(ValueError,match='Ambiguous'):campaign.submit_job(source,root,'production',request_key='fixed')


def test_initial_dependencies_form_one_gpu_chain(config,tmp_path,source,monkeypatch):
    calls=[]
    monkeypatch.setattr(campaign,'check_source',lambda source:None)
    monkeypatch.setattr(campaign,'check_storage',lambda *a,**k:tmp_path/'checkpoints')
    (source/campaign.EXT/'references.json').write_text('{}')
    def submit(source,root,phase,**kwargs):
        job=str(100+len(calls));calls.append((job,phase,kwargs));return job
    monkeypatch.setattr(campaign,'submit_job',submit)
    receipt=campaign.submit_initial(source,tmp_path/'run',source/campaign.EXT/'campaign.json')
    assert receipt['jobs']==['100','101','102','103','104']
    assert [phase for _,phase,_ in calls]==['cpu-gate','gpu-gate','production','dispatch','report']
    assert calls[1][2]['dependency']=='afterok:100'
    assert calls[2][2]['dependency']=='afterok:101'
    assert calls[3][2]['dependency']=='afterany:102'
    assert calls[4][2]['dependency']=='afterany:103'


def test_dispatch_resumes_one_stream_idempotently(config,tmp_path,source,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','501')
    monkeypatch.setattr(campaign,'slurm_terminal_record',lambda job:dict(state='COMPLETED',exit_code='0:0',job_id=job))
    monkeypatch.setattr(campaign,'check_storage',lambda *a,**k:tmp_path/'checkpoints')
    monkeypatch.setattr(campaign,'checkpoint_cursor',lambda *a:dict(task_index=2,iteration=5000,has_checkpoint=True))
    calls=[]
    def submit(source,root,phase,**kwargs):
        calls.append((phase,kwargs));return str(600+len(calls))
    monkeypatch.setattr(campaign,'submit_job',submit)
    path=source/campaign.EXT/'campaign.json'
    result=campaign.dispatch(source,tmp_path,path,'500')
    assert result['status']=='continued'
    assert calls[0]==('production',dict(dependency='afterok:501',request_key='500:continuation'))
    assert calls[1]==('dispatch',dict(dependency='afterany:601',cpu=True,parent_job='601',request_key='601:dispatch'))
    assert campaign.dispatch(source,tmp_path,path,'500')==result
    assert len(calls)==2


def test_dispatch_freezes_uncheckpointed_timeout(config,tmp_path,source,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','501')
    monkeypatch.setattr(campaign,'slurm_terminal_record',lambda job:dict(state='TIMEOUT',exit_code='0:0',job_id=job))
    monkeypatch.setattr(campaign,'check_storage',lambda *a,**k:tmp_path/'checkpoints')
    monkeypatch.setattr(campaign,'checkpoint_cursor',lambda *a:dict(task_index=0,iteration=0,has_checkpoint=False))
    calls=[]
    monkeypatch.setattr(campaign,'submit_job',lambda *a,**k:calls.append((a,k)) or '600')
    with pytest.raises(ValueError,match='no verified state'):
        campaign.dispatch(source,tmp_path,source/campaign.EXT/'campaign.json','500')
    assert campaign.read(tmp_path/'stream/failed.json')['status']=='failed'
    assert len(calls)==1 and calls[0][0][2]=='report'
