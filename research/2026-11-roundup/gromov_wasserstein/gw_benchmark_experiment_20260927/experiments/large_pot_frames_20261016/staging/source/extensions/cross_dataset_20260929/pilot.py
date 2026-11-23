"""Bounded allocated feasibility/resource pilot before independent tuning."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from campaign import (digest,write,config_digest,pilot_specs,input_hashes,
                      validate_metadata,validate_saved_output,audit_pair,pilot_guard)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Pilot requires Slurm.')
    config=json.loads(args.config.read_text());root=Path(config['pilot_output'])
    root.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',runtime_commit=config['runtime_commit'],config_sha256=config_digest(config),
        controller_sha256=digest(Path(__file__).with_name('campaign.py')),pilot_runner_sha256=digest(__file__),
        job_id=os.environ['SLURM_JOB_ID'],records=[],start_unix=time.time(),input_sha256=input_hashes(config))
    write(root/'pilot.json',state)
    try:
        runs=pilot_specs(config)
        manifest=dict(schema_version=1,runs=runs,config_sha256=config_digest(config))
        write(root/'manifest.json',manifest)
        runtime=Path(config['runtime_directory'])
        command=[sys.executable,str(runtime/'run_campaign_shard.py'),'--manifest',str(root/'manifest.json'),
            '--source-commit',config['runtime_commit'],'--data-root',config['data_root'],
            '--out-root',str(root),'--task-index','0','--task-count','1']
        with (root/'launcher.log').open('x') as log:
            process=subprocess.run(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,check=False)
        if process.returncode:raise RuntimeError('Pilot launcher failed; inspect launcher.log and shard receipts.')
        shard=json.loads((root/'_campaign/shard_00.json').read_text())
        if shard['source_commit']!=config['runtime_commit'] or shard['manifest_sha256']!=digest(root/'manifest.json'):
            raise ValueError('Pilot shard source/manifest mismatch.')
        if shard['source_sha256']!=config['launcher_sha256'] or len(shard['records'])!=len(runs):
            raise ValueError('Pilot shard launcher/count mismatch.')
        audited=[]
        for run,receipt in zip(runs,shard['records']):
            folder=root/run['run_id'];metadata=json.loads((folder/'metadata.json').read_text())
            if receipt['run_spec']!=run or receipt['returncode']!=0 or receipt['infrastructure_error'] is not None:
                raise ValueError('Pilot run receipt failure.')
            if not validate_metadata(metadata,run,config):raise ValueError('Pilot scientific output failed validation.')
            saved=validate_saved_output(folder/'final.npz',run['n'],metadata)
            gpu=metadata['environment']['gpu']
            if 'A100' not in gpu['name']:raise ValueError('Pilot requires A100 hardware.')
            memory=metadata['memory']
            if memory['cuda_peak_reserved_bytes']>=.8*gpu['total_memory_bytes']:
                raise RuntimeError('Pilot GPU memory gate failed; full campaign remains blocked.')
            if memory['cpu_process_peak_rss_kib']*1024>=.8*32*1024**3:
                raise RuntimeError('Pilot host memory gate failed; full campaign remains blocked.')
            with (folder/'trace.csv').open(newline='') as stream:
                initial=[row for row in csv.DictReader(stream) if row['kind']=='initial']
            if len(initial)!=1:raise ValueError('Missing or duplicate initial observation.')
            state['records'].append(dict(run_spec=run,metadata_path=str((folder/'metadata.json').relative_to(root)),
                metadata_sha256=digest(folder/'metadata.json'),final_path=str((folder/'final.npz').relative_to(root)),
                final_sha256=saved['sha256'],trace_path=str((folder/'trace.csv').relative_to(root)),
                trace_sha256=digest(folder/'trace.csv'),initial_rounded_loss=float(initial[0]['rounded_loss']),
                returncode=receipt['returncode'],memory=memory,resource_gate_passed=True))
            audited.append(dict(**run,eligible=True,array_sha256=metadata['array_sha256'],
                initialization_plan_sha256=metadata['initialization_plan_sha256'],environment=metadata['environment']))
            write(root/'pilot.json',state)
        for a,b in zip(audited[::2],audited[1::2]):audit_pair(a,b)
        state['common_loss_targets']={}
        for instance in config['instances']:
            initial=next(x['initial_rounded_loss'] for x in state['records'] if x['run_spec']['instance_id']==instance['id'])
            fractions=config['loss_target_fractions']
            state['common_loss_targets'][instance['id']]=dict(baseline_loss=initial,fractions=fractions,losses=[initial*f for f in fractions])
        state.update(status='completed',end_unix=time.time(),resource_gate_passed=True)
        write(root/'pilot.json',state)
        pilot_guard(config)
        print(json.dumps(dict(status='completed',runs=len(runs),instances=[i['id'] for i in config['instances']]),indent=2))
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',end_unix=time.time())
        write(root/'pilot.json',state)
        raise
if __name__=='__main__':main()
