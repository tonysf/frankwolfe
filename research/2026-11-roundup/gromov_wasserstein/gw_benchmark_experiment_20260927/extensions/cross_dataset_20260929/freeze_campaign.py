"""Bind prepared input identity and runtime release before any pilot or tuning."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--runtime-commit',required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    here=Path(__file__).resolve().parent;out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    data=args.data_root.resolve();prep=json.loads((data/'preparation.json').read_text())
    if prep['status']!='completed':raise ValueError('Dataset preparation incomplete.')
    if prep['binding']['preparation_script_sha256']!=sha(here/'prepare_snare.py') or prep['binding']['source_manifest_sha256']!=sha(here/'data_sources.json'):
        raise ValueError('Preparation code or raw source identity differs from this release.')
    config=json.loads((here/'campaign_template.json').read_text());instances=[]
    expected_bmmc='99989d1cc37a90a35c1f2db48734951ba6184437ddadff711272465c8cacbebe'
    if sha(data/'bmmc1000.npz')!=expected_bmmc:raise ValueError('Canonical BMMC changed.')
    instances.append(dict(id='bmmc1000',n=1000,filename='bmmc1000.npz',sha256=expected_bmmc))
    for iid in ('snare3000','snare_full'):
        item=prep['instances'][iid]
        record={k:item[k] for k in ('n','filename','sha256','correspondence_file','correspondence_sha256')}
        record['id']=iid
        for key,hashkey in (('filename','sha256'),('correspondence_file','correspondence_sha256')):
            if sha(data/record[key])!=record[hashkey]:raise ValueError('Prepared input hash mismatch: '+iid)
        instances.append(record)
    config.update(instances=instances,data_root=str(data),config_path=str(out/'campaign.json'),
        pilot_output=str(out/'pilot'),runtime_directory=str(args.runtime.resolve()),runtime_commit=args.runtime_commit,
        runtime_file_sha256={name:sha(args.runtime/name) for name in ('run_benchmark.py','smoothing_fw.py','gw_core.py','measurements.py')},
        launcher_sha256=sha(args.runtime/'run_campaign_shard.py'),profile_sha256=sha(args.runtime/'profile_components.py'),
        preparation_manifest_sha256=sha(data/'preparation.json'),controller_sha256=sha(here/'campaign.py'),
        pilot_runner_sha256=sha(here/'pilot.py'),reporter_sha256=sha(here/'report.py'))
    (out/'campaign.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(config=str(out/'campaign.json'),config_sha256=sha(out/'campaign.json'),instances=[dict(id=i['id'],n=i['n']) for i in instances])))
if __name__=='__main__':main()
