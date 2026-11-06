"""Scheduled production-shape GPU proof that diagnostic hooks preserve updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

from trajectory_parity import compare_trajectories


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','prepared','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    values={};records={}
    result=dict(status='running',checks={})
    try:
        for name,q,script,enabled in (
            ('q2048_legacy',2048,'batch_size_20261008/run_batch.py',None),
            ('q2048_enabled',2048,'probe8192_20261009/run_diagnostic.py','enabled'),
            ('q8192_disabled',8192,'probe8192_20261009/run_diagnostic.py','disabled'),
            ('q8192_enabled',8192,'probe8192_20261009/run_diagnostic.py','enabled')):
            out=args.out/name
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=1',
                str(args.source/'extensions'/script),'--instance',str(args.prepared/'metric512_s1000'),
                '--out',str(out),'--device','cuda','--method','smoothing-fw','--dtype','float32',
                '--iterations','20','--seed','710','--init-seed','10710','--output-seed','20710',
                '--initialization','product','--batch',str(q),'--gamma0','1','--rho0','.3','--mu-scale','10',
                '--save-plans','none','--checkpoint-iterations','1,2,5,10,20','--progress-every','20',
                '--exact-block-rows','256','--certificate-block-rows','256']
            if enabled is not None:command+=['--direction-diagnostics',enabled]
            with (args.out/(name+'.log')).open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            path=out/'metadata.json';metadata=json.loads(path.read_text())
            if metadata['dimensions']!={'source':512,'target':512}:
                raise ValueError('GPU parity must use the full frozen 512-point synthetic instance.')
            if metadata['environment']['world_size']!=1 or not all('A100' in (r.get('gpu') or '') for r in metadata['ranks']):
                raise ValueError('GPU parity must run on exactly one A100.')
            if metadata['config']['batch']!=q or metadata['counters']['sketch_probes']!=q*20:
                raise ValueError('GPU parity probe budget differs.')
            if enabled == 'enabled' and [r['iteration'] for r in metadata.get('direction_trace',[])] != [0,1,2,5,10]:
                raise ValueError('Enabled GPU parity run did not observe the requested preupdate directions.')
            if enabled == 'disabled' and 'direction_trace' in metadata:
                raise ValueError('Disabled GPU parity control unexpectedly contains direction diagnostics.')
            values[name]=metadata
            records[name]=dict(command=command,metadata_file=str(path.resolve()),metadata_sha256=sha256(path))
        result['checks']['q2048_legacy']=compare_trajectories(values['q2048_legacy'],values['q2048_enabled'])
        result['checks']['q8192_enabled_disabled']=compare_trajectories(values['q8192_disabled'],values['q8192_enabled'])
        metadata=values['q8192_enabled']
        result.update(status='passed',source_file_sha256=metadata['source_file_sha256'],
            reference_source_file_sha256=values['q2048_legacy']['source_file_sha256'],
            array_files_sha256=metadata['array_files_sha256'],instance_manifest_sha256=metadata['instance_manifest_sha256'],
            helper_source_sha256={name:sha256(args.source/'extensions/probe8192_20261009'/name)
                                  for name in ('verify_gpu.py','trajectory_parity.py')},runs=records)
        write_json(args.out/'parity.json',result)
        print('Passed q2048 prior-runner and q8192 hook/no-hook GPU trajectory parity.',flush=True)
    except Exception as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),runs=records)
        write_json(args.out/'parity.json',result)
        raise


if __name__=='__main__':main()
