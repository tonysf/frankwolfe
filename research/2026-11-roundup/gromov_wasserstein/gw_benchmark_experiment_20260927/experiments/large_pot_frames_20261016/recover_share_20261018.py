"""One reviewed retry after CUDA initialization failed before any SHARE update."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/large_pot_frames_20261016_6a472d4')
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE / 'extensions/large_pot_frames_20261016'))
import campaign as c

dataset = ROOT / 'results/shareseq_full'
out = dataset / 'baseline/pot-full_seed810'
config = SOURCE / c.EXT / 'campaign.json'
recovery = ROOT / 'recovery/20261018_cuda_initialization'
c.check_source(SOURCE)
c.require_gate(ROOT / 'checks/shareseq_full/gate.json', c.identity(SOURCE, config))
subprocess.run([sys.executable, str(ROOT / 'record_storage.py')], check=True, stdout=subprocess.DEVNULL)
assert c.slurm_terminal_record('2074313')['state'] == 'FAILED'
assert c.slurm_terminal_record('2074314')['state'] == 'FAILED'
with (dataset / 'controller.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (dataset / 'dispatch_state.json').exists()
    assert not (dataset / 'completed.json').exists()
    for name in ('metadata.json', 'progress.json', 'failure.json'):
        assert not (out / name).exists(), name
    assert not list((ROOT / 'checkpoints/shareseq_full').glob('**/COMMITTED.json'))
    log = (out / 'job_2074313.log').read_text()
    assert 'torch.cuda.set_device(local_rank)' in log
    assert 'CUDA-capable device(s) is/are busy or unavailable' in log
    assert 'ruche-gpu15.cluster' in log
    assert c.read(dataset / 'failed.json')['job_id'] == '2074313'
    recovery.mkdir(parents=True, exist_ok=False)
    review = dict(status='reviewed', time=c.now(), prior_production='2074313',
        prior_dispatch='2074314', classification='CUDA device initialization failure before any numerical work',
        numerical_source_unchanged=True, completed_updates=0,
        failed_log_sha256=c.sha256(out / 'job_2074313.log'),
        exclusion_scope='Avoid gpu15 on this retry only; automatic later allocations retain the original exclusions.')
    c.write(recovery / 'review.json', review)
    entries = [json.loads(line) for line in (ROOT / 'submission.jsonl').read_text().splitlines()]
    old = next(x for x in entries if x.get('stdout', '').strip() == '2074313')
    # The validated gate receipt is checked above; its historical Slurm ID
    # need not remain in the scheduler's live dependency cache.
    command = [arg for arg in old['command'] if not arg.startswith('--dependency=')]
    command[1:1] = ['--hold', '--exclude=ruche-gpu13,ruche-gpu15,ruche-gpu16,ruche-gpu19']
    key = 'shareseq_full:20261018-cuda-init:production'
    request = ROOT / 'submission_requests' / (hashlib.sha256(key.encode()).hexdigest() + '.json')
    assert not request.exists()
    record = dict(time=c.now(), phase='production', dataset='shareseq_full', world_size=4,
                  command=command, request_key=key)
    c.write(request, dict(record, status='submitting'))
    result = subprocess.run(command, text=True, capture_output=True)
    record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    with (ROOT / 'submission.jsonl').open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    result.check_returncode()
    job = result.stdout.strip().split(';')[0]
    assert job.isdigit()
    c.write(request, dict(record, status='submitted', job_id=job))
    dispatch = c.submit_job(SOURCE, ROOT, 'dispatch', 'shareseq_full',
        dependency='afterany:' + job, cpu=True, parent_job=job,
        request_key=f'shareseq_full:{job}:dispatch')
    (dataset / 'failed.json').rename(recovery / 'previous_failed.json')
    receipt = dict(review, status='submitted_held', production_job=job, dispatcher_job=dispatch,
                   recovery_directory=str(recovery), command=command)
    c.write(recovery / 'receipt.json', receipt)
subprocess.run(['scontrol', 'release', job], check=True)
receipt.update(status='released', released_utc=c.now())
c.write(recovery / 'receipt.json', receipt)
print(json.dumps(receipt, indent=2))
