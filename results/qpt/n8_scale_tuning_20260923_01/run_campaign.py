"""Run the authorized n8 scale comparison inside a Slurm allocation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('This script must run inside a Slurm allocation.')
import numpy as np

ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text())
CHOICE=json.loads((ROOT/'preflight_retry1'/'selection.json').read_text())
assert CHOICE['status']=='validated'
SOURCE=Path(CHOICE['source'])
EXPECTED_SOURCE_HASHES=CHOICE['source_sha256']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def summarize(path):
    fields={'step':'checkpoint_steps','fidelity':'process_fidelity_proxy',
            'measurement_loss':'measurement_loss','tp_violation':'tp_violation',
            'smoothed_gap':'smoothed_gap','smoothed_objective':'smoothed_objective',
            'smoothing_parameter':'checkpoint_smoothing_parameters'}
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(str(z['metadata_json'].item()))
        arrays={key:z[name] for key,name in fields.items()}
        length=len(arrays['step'])
        assert all(len(a)==length and np.all(np.isfinite(a)) for a in arrays.values())
        assert np.all(np.diff(arrays['step'])>0)
        rows=[{key:int(a[i]) if key=='step' else float(a[i]) for key,a in arrays.items()} for i in range(length)]
    return dict(metadata=m,trajectory=rows)


def check_output(path,case,steps):
    result=summarize(path)
    m=result['metadata']
    assert m['n_qubits']==case['n_qubits'] and m['noise_std']==case['noise_std']
    assert m['initial_factor_sha256']==case['initial_factor_sha256']
    assert m['n_steps']==steps and m['start_step']==0 and m['resume_from'] is None
    assert m['batch_size']==case['batch_size'] and m['tau']==case['tau']
    assert m['device_platform']=='gpu' and m['fidelity_target'] is None
    assert result['trajectory'][0]['step']==0 and result['trajectory'][-1]['step']==steps
    k=np.arange(steps,dtype=float)
    expected={'step_sizes':np.minimum(1,case['step_scale']/(k+case['step_offset'])**.75),
              'smoothing_parameters':case['smoothing_scale']/(k+1)**.25,
              'momentum_weights':np.minimum(1,case['rho_scale']/(k+case['rho_offset'])**.5)}
    with np.load(path,allow_pickle=False) as z:
        assert np.all(np.isfinite(z['final_factor'])) and np.all(np.isfinite(z['gradient_estimate']))
        for field,values in expected.items():
            assert z[field].shape==values.shape and np.allclose(z[field],values,rtol=1e-13,atol=0),field
    result['schedule_verification']='All applied scalar values match requested exponents, scales, offsets and caps.'
    return result


def run(case):
    out=ROOT/case['name']
    out.mkdir()  # A repeated invocation cannot overwrite or repeat this case.
    status=dict(name=case['name'],status='validating',started_at_utc=now(),
                job_id=os.environ['SLURM_JOB_ID'],array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                hostname=socket.gethostname())
    write(out/'status.json',status)
    try:
        hashes={name:sha(SOURCE/name) for name in EXPECTED_SOURCE_HASHES}
        assert hashes==EXPECTED_SOURCE_HASHES,'Validated frozen source changed'
        assert sha(case['data'])==case['data_sha256'],'Dataset changed'
        with np.load(case['data'],allow_pickle=False) as z:
            data_meta=json.loads(str(z['metadata_json'].item()))
            assert int(z['n_qubits'])==case['n_qubits']
            assert data_meta['noise_std']==case['noise_std']
            assert data_meta['channel_seed']==data_meta['noise_seed']==0
        common=[sys.executable,'-m','paper.experiments.quantum_process_tomography_structured_jax',
            '--data',case['data'],'--device','gpu','--precision','64',
            '--measurement-backend',case['measurement_backend'],'--rank','1','--tau',str(case['tau']),
            '--batch-size',str(case['batch_size']),'--chunk-steps',str(case['chunk_steps']),
            '--metrics-every',str(case['metrics_every']),'--metric-mode','sampled','--metric-samples','512',
            '--metric-batch-size',str(case['metric_batch_size']),'--metric-seed','12345',
            '--initialization-seed','0','--sampling-seed','0',
            '--rho-scale',str(case['rho_scale']),'--rho-offset',str(case['rho_offset']),'--rho-exponent','0.5',
            '--smoothing-scale',str(case['smoothing_scale']),'--smoothing-offset','1','--smoothing-exponent','0.25',
            '--step-scale',str(case['step_scale']),'--step-offset',str(case['step_offset']),'--step-exponent','0.75']
        if case['prefetch']: common.append('--prefetch')
        command=common+['--steps',str(case['steps']),'--restart-path',str(out/'latest_restart.npz'),
                        '--save',str(out/'final.npz')]
        provenance=dict(case=case,command=command,source_sha256=hashes,data_sha256=case['data_sha256'],
                        data_metadata=data_meta,base_commit=PLAN['base_commit'],
                        environment={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_ARRAY_JOB_ID','SLURM_ARRAY_TASK_ID','CUDA_VISIBLE_DEVICES','XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                        hostname=socket.gethostname(),started_at_utc=now(),driver_sha256=sha(__file__))
        write(out/'provenance.json',provenance)
        references=case.get('baseline_archives',[])
        baseline_configuration=None
        write(out/'requested_baseline_trace.json',dict(path=case['requested_exponent_baseline_archive'],archive_sha256=sha(case['requested_exponent_baseline_archive']),**summarize(case['requested_exponent_baseline_archive'])))
        if case.get('baseline_report'):
            report=json.loads(Path(case['baseline_report']).read_text())
            row=next(r for r in report['runs'] if r['backend']=='structured' and r['purpose']=='timing' and r['status']=='success')
            assert row['compact_input_sha256']==case['data_sha256']
            references=[row['structured_result_path']]
            baseline_configuration=report['configuration']
        write(out/'baseline_traces.json',dict(configuration=baseline_configuration,
            segments=[dict(path=p,archive_sha256=sha(p),**summarize(p)) for p in references]))
        status['status']='smoke_running';write(out/'status.json',status)
        print('SMOKE',case['name'],flush=True)
        with (out/'smoke.log').open('x') as log:
            subprocess.run(common+['--steps','100','--save',str(out/'smoke.npz')],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(out/'smoke_summary.json',check_output(out/'smoke.npz',case,100))
        status['status']='running';status['production_started_at_utc']=now();write(out/'status.json',status)
        print('PRODUCTION',case['name'],flush=True)
        with (out/'optimizer.log').open('x') as log:
            subprocess.run(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=check_output(out/'final.npz',case,case['steps'])
        write(out/'summary.json',summary)
        with (out/'verification.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'verify_saved_factors.py'),'--data',case['data'],
                            '--result',str(out/'final.npz'),'--save',str(out/'verification.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        assert sha(case['data'])==case['data_sha256']
        write(out/'checksums.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('status.json','checksums.json')})
        status.update(status='completed',completed_at_utc=now(),final=summary['trajectory'][-1],
                      optimizer_seconds=summary['metadata']['optimizer_seconds'],runner_seconds=summary['metadata']['wall_seconds'])
        write(out/'status.json',status)
        print('COMPLETE',case['name'],json.dumps(status['final']),flush=True)
    except Exception:
        status.update(status='failed',failed_at_utc=now(),error=traceback.format_exc())
        write(out/'status.json',status)
        raise


parser=argparse.ArgumentParser()
parser.add_argument('--group',choices=['small','large'],required=True)
args=parser.parse_args()
selected=[c for c in PLAN['cases'] if c['group']==args.group]
if args.group=='large':
    index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    selected=[c for c in selected if c['array_task']==index]
    assert len(selected)==1
for case in selected: run(case)
