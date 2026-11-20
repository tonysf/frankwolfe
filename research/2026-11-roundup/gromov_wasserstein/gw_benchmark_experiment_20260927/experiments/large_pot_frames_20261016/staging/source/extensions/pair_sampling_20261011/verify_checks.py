"""Run bounded correctness checks in a scheduled CPU or GPU allocation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

from campaign import METHODS,execute,require,source_hashes,read,write,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('cpu','gpu'),required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--world-size',type=int,choices=(1,4),default=1)
    args=p.parse_args();require(bool(os.environ.get('SLURM_JOB_ID')),'Checks require a scheduled allocation.')
    args.out.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='running',mode=args.mode,world_size=0 if args.mode=='cpu' else args.world_size,
                 slurm_job_id=os.environ['SLURM_JOB_ID'])
    try:
        extension=args.source/'extensions/pair_sampling_20261011'
        if args.mode=='cpu':
            junit=args.out/'pytest.xml'
            command=[sys.executable,'-m','pytest','-q',str(extension),'--junitxml',str(junit)]
            require(execute(command,args.out/'checks.log',3300)==0,'Scheduled CPU tests failed.')
            tree=ET.parse(junit).getroot();suites=[tree] if tree.tag=='testsuite' else list(tree.iter('testsuite'))
            tests={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
            require(tests['tests']-tests['skipped']>0 and tests['failures']==tests['errors']==0,'No passing CPU test evidence.')
            receipt.update(tests=tests,junit_sha256=sha256(junit))
        else:
            path=args.out/'distributed.json'
            command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={args.world_size}',
                     str(extension/'validate_distributed.py'),'--device','cuda','--out',str(path)]
            require(execute(command,args.out/'checks.log',3300)==0,'Scheduled distributed GPU checks failed.')
            proof=read(path)
            require(proof['status']=='passed' and proof['world_size']==args.world_size,'Missing actual-topology correctness proof.')
            require(proof['source_file_sha256']==source_hashes(args.source),'GPU proof used another numerical source.')
            require(proof['validation_source_sha256']==sha256(extension/'validate_distributed.py'),'GPU proof helper changed.')
            require(proof['device'].startswith('cuda') and proof['backend']=='nccl','GPU proof did not use CUDA/NCCL.')
            require({row['method'] for row in proof['update_checks']}==set(METHODS),'GPU proof omits a method.')
            require(all(row['independent_dense_update_agreement'] is True for row in proof['update_checks']),'Missing independent update agreement.')
            require(len(proof['oracle_checks'])==3,'Missing oracle edge cases.')
            receipt.update(distributed_file=str(path),distributed_sha256=sha256(path),
                           oracle_checks=proof['oracle_checks'],update_checks=proof['update_checks'])
        receipt.update(status='passed',source_file_sha256=source_hashes(args.source),command=command)
        write(args.out/'checks.json',receipt)
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc),traceback=traceback.format_exc());write(args.out/'checks.json',receipt);raise


if __name__=='__main__':main()
