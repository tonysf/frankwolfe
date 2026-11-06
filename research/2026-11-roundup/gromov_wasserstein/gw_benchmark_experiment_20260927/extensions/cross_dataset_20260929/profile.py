"""Launch selected-parameter descriptive component profiles in an allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from campaign import frozen_selection,config_digest,digest,write,input_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--task',type=int,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Profiling requires Slurm.')
    config=json.loads(args.config.read_text());selection=frozen_selection(config,args.out)
    input_hashes(config)
    instance=config['instances'][args.task];runtime=Path(config['runtime_directory'])
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=runtime,text=True).strip()
    if source!=config['runtime_commit'] or dirty:raise ValueError('Profile runtime is not the frozen clean revision.')
    dest=args.out/'profiles'/instance['id'];dest.mkdir(parents=True,exist_ok=False)
    receipt=dict(config_sha256=config_digest(config),selection_sha256=digest(args.out/'selection.json'),
        controller_sha256=digest(__file__),instance=instance,job_id=os.environ['SLURM_JOB_ID'],records=[],status='running')
    write(dest/'receipt.json',receipt)
    try:
        for method in config['methods'] if args.task%2==0 else list(reversed(config['methods'])):
            parameters=selection['selected'][instance['id']][method]['parameters'];path=dest/(method+'.json')
            command=[sys.executable,str(runtime/'profile_components.py'),'--instance',str(Path(config['data_root'])/instance['filename']),
                '--instance-sha256',instance['sha256'],'--method',method,'--parameters',json.dumps(parameters),
                '--seed','400','--out',str(path)]
            with (dest/(method+'.log')).open('x') as log:
                process=subprocess.run(command,cwd=runtime,stdout=log,stderr=subprocess.STDOUT,timeout=1500,check=False)
            if process.returncode:raise RuntimeError('Component profile failed: '+method)
            data=json.loads(path.read_text())
            if data['source_commit']!=source or data['source_dirty'] or json.loads(data['config']['parameters'])!=parameters:
                raise ValueError('Profile source/parameter identity mismatch.')
            for name,sha in data['source_file_sha256'].items():
                if digest(runtime/name)!=sha:raise ValueError('Profile source hash mismatch.')
            receipt['records'].append(dict(method=method,path=str(path),sha256=digest(path),parameters=parameters))
            write(dest/'receipt.json',receipt)
        receipt['status']='completed';write(dest/'receipt.json',receipt)
    except Exception as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}');write(dest/'receipt.json',receipt);raise
if __name__=='__main__':main()
