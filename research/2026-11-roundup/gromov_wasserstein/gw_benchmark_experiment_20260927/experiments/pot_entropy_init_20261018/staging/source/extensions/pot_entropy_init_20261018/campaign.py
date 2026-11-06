"""Dependency-driven full-gradient entropic GW sensitivity experiment on Ruche.

Each dataset has one serial GPU stream. A small CPU dispatcher continues only
verified committed states; an inner numerical failure rejects that task without
relaxing its tolerance, while infrastructure failures freeze the stream.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXT = Path('extensions/pot_entropy_init_20261018')
DATASETS = ('bmmc6224', 'snare_full')
_spec = importlib.util.spec_from_file_location('entropy_campaign_common', HERE.parent / 'large_pot_frames_20261016/campaign.py')
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
require, read, write, sha256 = _common.require, _common.read, _common.write, _common.sha256
now, check_source, identity = _common.now, _common.check_source, _common.identity
slurm_terminal_record = _common.slurm_terminal_record


def load_config(path):
    c = read(path)
    require(c['schema_version'] == 1 and set(c['instances']) == set(DATASETS), 'Unexpected schema/datasets')
    require(c['dtype'] == 'float64' and c['method'] == 'pot-full', 'FP64 full-gradient required')
    require(c['epsilons'] == [.001, .0003, .0001, .00003], 'Entropy grid changed')
    require(c['initializations'] == [dict(kind='product', seed=810)] +
            [dict(kind='permutation-mixture', seed=s) for s in (901, 902, 903)], 'Initialization design changed')
    require(c['init_mixture_weight'] == .9 and c['iterations'] == 2000, 'Initialization weight/horizon changed')
    require(c['inner_tol'] == c['inner_gap_tol'] == 1e-7 and c['inner_max'] == 100000, 'Inner solve specification changed')
    require((c['outer_tolerance'], c['outer_min_iterations'], c['outer_stable_updates']) ==
            (1e-6, 50, 10), 'Outer stopping specification changed')
    require(0 < c['runner']['max_seconds'] <= 22*3600, 'Insufficient Slurm margin')
    for instance in c['instances'].values():
        require(instance['world_size'] == 1 and Path(instance['path']).is_absolute(), 'Wrong dataset topology/path')
        require(set(instance['array_files_sha256']) == {'A.npy', 'B.npy', 'a.npy', 'b.npy'}, 'Missing array identity')
    return c


def tasks(c):
    return [dict(task_id=f'epsilon_{epsilon:.0e}/{init["kind"]}_seed{init["seed"]}',
                 method='pot-full', epsilon=epsilon, initialization=init['kind'],
                 seed=init['seed'], iterations=c['iterations'])
            for epsilon in c['epsilons'] for init in c['initializations']]


def task_config(c, task):
    return dict(method=task['method'], seed=task['seed'], iterations=task['iterations'],
                epsilon=task['epsilon'], initialization=task['initialization'],
                init_mixture_weight=c['init_mixture_weight'], dtype='float64', device='cuda',
                inner_max=c['inner_max'], diagnostic_every=c['runner']['diagnostic_every'],
                exact_block_rows=128, outer_tolerance=c['outer_tolerance'],
                outer_min_iterations=c['outer_min_iterations'],
                outer_stable_updates=c['outer_stable_updates'])


def check_storage(root, c, fresh=False):
    receipt = read(Path(root) / 'storage_receipt.json')
    require(receipt.get('quota_verified') is True, 'Storage receipt must establish quota')
    require(receipt['quota_available_bytes'] >= c['storage']['required_quota_available_bytes'],
            'Verified quota headroom below 70 GiB (60 GiB existing SHARE plus 10 GiB new campaign)')
    checkpoint_root = Path(receipt['checkpoint_root'])
    require(checkpoint_root.is_absolute() and checkpoint_root.is_dir(), 'Checkpoint directory missing')
    if fresh:
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(receipt['verified_utc'])).total_seconds()
        require(0 <= age <= c['storage']['receipt_max_age_seconds'], 'Quota receipt stale')
    return checkpoint_root


def validate_identity(record, task, instance, c, source):
    require(record['config'] == task_config(c, task), 'Immutable run configuration differs')
    require(record['array_files_sha256'] == instance['array_files_sha256'], 'Dataset arrays differ')
    require(record['manifest_sha256'] == instance['instance_manifest_sha256'], 'Dataset manifest differs')
    require(record['world_size'] == 1 and record['n'] == record['m'] == instance['n'], 'Dataset dimensions/topology differ')
    require(bool(record['source_file_sha256']), 'Numerical source identity missing')
    for name, expected in record['source_file_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid source path')
        require(sha256(Path(source)/name) == expected, 'Numerical source differs: '+name)


def validate_result(meta, task, instance, c, source):
    require(meta['status'] == 'completed', 'Run incomplete')
    validate_identity(meta['identity'], task, instance, c, source)
    require(meta['config'] == meta['identity']['config'], 'Result configuration disagrees with identity')
    require(meta['target_updates'] == task['iterations'], 'Target horizon differs')
    done = meta['completed_updates']
    require(0 < done <= task['iterations'], 'Completed update count invalid')
    require(math.isfinite(meta['final_loss']), 'Final loss nonfinite')
    if done < task['iterations']:
        require(meta['stopping_reason'] == 'outer_update_l1_tolerance', 'Unexplained early stopping')
        require(done >= c['outer_min_iterations'] and
                meta['stable_updates'] >= c['outer_stable_updates'] and
                meta['last_outer_update_l1'] <= c['outer_tolerance'], 'Outer stopping condition not satisfied')
    schedule = meta['schedule']
    require(schedule['inner_tolerance'] == schedule['gap_tolerance'] == c['inner_tol'], 'Inner tolerances differ')
    require(schedule['eta'] == 1/task['epsilon'], 'POT step differs')
    require(meta['trace'][-1]['iteration'] == done, 'Final diagnostic missing')
    return meta['final_loss']


def validate_numerical_failure(record, task, instance, c, source):
    require(record['status'] == 'failed' and record['failure_kind'] == 'inner_certificate_not_met',
            'Only explicit inner-certificate failure can reject one task')
    validate_identity(record['identity'], task, instance, c, source)
    require(record.get('certificate') is not None, 'Numerical failure lacks certificate evidence')
    require(0 <= record['completed_updates'] < task['iterations'], 'Failure update count invalid')


def task_finished(out, task, instance, c, source):
    out = Path(out)
    if (out/'unsuccessful.json').exists():
        unsuccessful = read(out/'unsuccessful.json')
        require(unsuccessful['failure_sha256'] == sha256(out/'failure.json'), 'Failure evidence changed')
        validate_numerical_failure(read(out/'failure.json'), task, instance, c, source)
        return True
    if (out/'metadata.json').exists() and read(out/'metadata.json').get('status') == 'completed':
        validate_result(read(out/'metadata.json'), task, instance, c, source)
        return True
    return False


def runner_command(source, root, c, dataset, task, seconds, checkpoint_root):
    options = dict(task_config(c, task), instance=c['instances'][dataset]['path'],
                   out=str(Path(root)/'results'/dataset/task['task_id']),
                   checkpoint_dir=str(Path(checkpoint_root)/dataset/task['task_id']),
                   max_seconds=seconds, checkpoint_every=c['runner']['checkpoint_every'],
                   checkpoint_seconds=c['runner']['checkpoint_seconds'])
    command = [sys.executable, str(Path(source)/EXT/'runner.py')]
    for key, value in options.items():
        command += ['--'+key.replace('_', '-'), str(value)]
    return command


def submit_job(source, root, phase, dataset='', dependency=None, cpu=False,
               seconds=None, parent_job='', request_key=None):
    source, root = Path(source), Path(root)
    name = f'gwentropy_{phase}_{dataset or "all"}'
    command = ['sbatch', '--parsable', '--job-name='+name,
               '--output='+str(root/'scheduler'/(name+'-%j.out'))]
    if dependency:
        command += ['--dependency='+dependency, '--kill-on-invalid-dep=yes']
    if cpu:
        command += [str(source/EXT/'cpu.sbatch'), str(source), str(root), phase, dataset, parent_job]
    else:
        if seconds:
            command += ['--time='+str(math.ceil(seconds/60))]
        command += [str(source/EXT/'gpu.sbatch'), str(source), str(root), phase, dataset, '1']
    receipt = dict(time=now(), phase=phase, dataset=dataset, command=command, request_key=request_key)
    request_path = None
    if request_key:
        request_path = root/'submission_requests'/(hashlib.sha256(request_key.encode()).hexdigest()+'.json')
        if request_path.exists():
            prior = read(request_path)
            require(prior['command'] == command, 'Idempotent command changed')
            if prior['status'] == 'submitted':
                return prior['job_id']
            require(prior['status'] == 'failed', 'Ambiguous sbatch request; refusing possible duplicate')
        write(request_path, dict(receipt, status='submitting'))
    result = subprocess.run(command, text=True, capture_output=True)
    receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    with (root/'submission.jsonl').open('a') as stream:
        import json
        stream.write(json.dumps(receipt, sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    if result.returncode:
        if request_path:
            write(request_path, dict(receipt, status='failed'))
        raise ValueError('sbatch failed; receipt retained: '+result.stderr)
    job = result.stdout.strip().split(';')[0]
    require(job.isdigit(), 'Unrecognized sbatch identifier')
    if request_path:
        write(request_path, dict(receipt, status='submitted', job_id=job))
    return job


def submit_initial(source, root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    c = load_config(config)
    check_source(source)
    check_storage(root, c, fresh=True)
    (root/'submission_lock').mkdir()
    (root/'scheduler').mkdir(exist_ok=True)
    write(root/'campaign_identity.json', dict(**identity(source, config), submitted_utc=now(), source=str(source)))
    cpu = submit_job(source, root, 'cpu-gate', cpu=True, request_key='initial:cpu-gate')
    jobs = [cpu]
    datasets = {}
    for dataset in DATASETS:
        gate = submit_job(source, root, 'dataset-gate', dataset, dependency='afterok:'+cpu,
                          seconds=4*3600, request_key=f'initial:{dataset}:gate')
        run = submit_job(source, root, 'production', dataset, dependency='afterok:'+gate,
                         request_key=f'initial:{dataset}:production')
        dispatcher = submit_job(source, root, 'dispatch', dataset, dependency='afterany:'+run,
                                cpu=True, parent_job=run, request_key=f'{dataset}:{run}:dispatch')
        datasets[dataset] = dict(gate_job_id=gate, production_job_id=run, dispatcher_job_id=dispatcher)
        jobs += [gate, run, dispatcher]
    write(root/'initial_submission.json', dict(status='queued', jobs=jobs, datasets=datasets, time=now()))


def require_gate(path, expected):
    gate = read(path)
    require(gate['status'] == 'passed', 'Prior gate not passed')
    for key, value in expected.items():
        require(gate[key] == value, 'Prior gate source/config differs')


def cpu_gate(source, root, config):
    out = Path(root)/'checks/cpu'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'tests.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', str(EXT)], cwd=source,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(Path(source)/EXT/'validation.py'), '--device', 'cpu', '--out', str(out)],
                   cwd=source, check=True)
    record = read(out/'checks.json')
    require(record['status'] == 'passed' and record['world_size'] == 1 and record['checks'], 'CPU gate did not pass')
    write(out/'gate.json', dict(status='passed', **identity(source, config), time=now()))


def dataset_gate(source, root, config, dataset):
    c = load_config(config)
    expected = identity(source, config)
    require_gate(Path(root)/'checks/cpu/gate.json', expected)
    check_storage(root, c)
    out = Path(root)/'checks'/dataset
    subprocess.run([sys.executable, str(Path(source)/EXT/'validation.py'), '--device', 'cuda',
                    '--out', str(out), '--instance', c['instances'][dataset]['path']], cwd=source, check=True)
    record = read(out/'checks.json')
    require(record['status'] == 'passed' and record['world_size'] == 1 and record['checks'], 'Dataset gate did not pass')
    write(out/'gate.json', dict(status='passed', **expected, time=now(), checks_sha256=sha256(out/'checks.json')))


def production(source, root, config, dataset):
    c, root = load_config(config), Path(root)
    expected = identity(source, config)
    require_gate(root/'checks'/dataset/'gate.json', expected)
    checkpoint_root = check_storage(root, c)
    dataset_root = root/'results'/dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (dataset_root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (dataset_root/'failed.json').exists(), 'Dataset frozen; manual review required')
        if (dataset_root/'completed.json').exists():
            return
        job = os.environ.get('SLURM_JOB_ID')
        require(job and job.isdigit(), 'Production requires Slurm allocation')
        chunk_path = dataset_root/f'chunk_{job}.json'
        write(chunk_path, dict(status='running', job_id=job, **expected, started_utc=now()))
        started, complete = time.monotonic(), False
        try:
            for task in tasks(c):
                out = dataset_root/task['task_id']
                if task_finished(out, task, c['instances'][dataset], c, source):
                    continue
                remaining = c['runner']['max_seconds'] - (time.monotonic()-started)
                if remaining < 600:
                    break
                out.mkdir(parents=True, exist_ok=True)
                write(chunk_path, dict(status='running', job_id=job, task_id=task['task_id'], **expected, updated_utc=now()))
                with (out/f'job_{job}.log').open('a') as stream:
                    result = subprocess.run(runner_command(source, root, c, dataset, task, remaining, checkpoint_root),
                                            cwd=source, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    require((out/'failure.json').exists(), 'Runner failed without classified numerical evidence')
                    validate_numerical_failure(read(out/'failure.json'), task, c['instances'][dataset], c, source)
                    write(out/'unsuccessful.json', dict(status='unsuccessful', failure_sha256=sha256(out/'failure.json'),
                          task=task, job_id=job, time=now(), note='Inner certificate not met; no tolerance relaxation or accepted result.'))
                    continue
                progress = read(out/'progress.json')
                require(progress['status'] in ('completed', 'checkpointed'), 'Runner neither completed nor checkpointed')
                if progress['status'] == 'checkpointed':
                    break
                validate_result(read(out/'metadata.json'), task, c['instances'][dataset], c, source)
            else:
                complete = True
            if complete:
                unsuccessful = sum((dataset_root/t['task_id']/'unsuccessful.json').exists() for t in tasks(c))
                write(dataset_root/'completed.json', dict(status='completed', **expected, time=now(),
                      runs=len(tasks(c)), successful_runs=len(tasks(c))-unsuccessful, unsuccessful_runs=unsuccessful,
                      last_job_id=job))
            write(chunk_path, dict(status='completed' if complete else 'checkpointed', job_id=job, **expected, time=now()))
        except BaseException as exc:
            write(chunk_path, dict(status='failed', error=repr(exc), traceback=traceback.format_exc(), time=now(), job_id=job))
            raise


def checkpoint_cursor(source, root, c, dataset, checkpoint_root):
    instance = c['instances'][dataset]
    dataset_root = Path(root)/'results'/dataset
    for index, task in enumerate(tasks(c)):
        out = dataset_root/task['task_id']
        if task_finished(out, task, instance, c, source):
            continue
        require(not (out/'failure.json').exists(), 'Unclassified failure requires review')
        folder = Path(checkpoint_root)/dataset/task['task_id']
        marker = folder/'COMMITTED.json'
        if not marker.exists():
            require(not (out/'progress.json').exists(), 'Started task lacks committed checkpoint')
            return dict(task_index=index, task_id=task['task_id'], iteration=0, has_checkpoint=False)
        record = read(marker)
        validate_identity(record['identity'], task, instance, c, source)
        require(record['world_size'] == 1 and 0 <= record['iteration'] <= task['iterations'], 'Checkpoint topology/update count differs')
        generation = record['generation']
        require(Path(generation).name == generation and generation.startswith('generation-'), 'Invalid checkpoint generation')
        require(read(folder/generation/'manifest.json') == record, 'Checkpoint manifests differ')
        require(len(record['shards']) == 1, 'Missing checkpoint shard')
        shard = record['shards'][0]
        require(shard['file'] == 'rank-0000.pt', 'Invalid checkpoint shard name')
        path = folder/generation/shard['file']
        require(path.stat().st_size == shard['bytes'] and sha256(path) == shard['sha256'], 'Checkpoint shard integrity failed')
        return dict(task_index=index, task_id=task['task_id'], iteration=record['iteration'], has_checkpoint=True,
                    committed_manifest_sha256=sha256(marker))
    return dict(task_index=len(tasks(c)), task_id=None, iteration=0, has_checkpoint=False)


def summarize(source, root, c):
    rows = []
    for dataset in DATASETS:
        for task in tasks(c):
            out = Path(root)/'results'/dataset/task['task_id']
            row = dict(dataset=dataset, **task, status='pending_or_running')
            if (out/'unsuccessful.json').exists():
                task_finished(out, task, c['instances'][dataset], c, source)
                row.update(status='unsuccessful', failure=read(out/'failure.json'))
            elif (out/'metadata.json').exists() and read(out/'metadata.json').get('status') == 'completed':
                meta = read(out/'metadata.json')
                validate_result(meta, task, c['instances'][dataset], c, source)
                row.update(status='completed', final_loss=meta['final_loss'], completed_updates=meta['completed_updates'],
                           stopping_reason=meta['stopping_reason'], metadata_path=str(out/'metadata.json'))
            rows.append(row)
    write(Path(root)/'summary.json', dict(generated_utc=now(), runs=rows,
          status='completed' if all(x['status'] in ('completed','unsuccessful') for x in rows) else 'partial'))


def dispatch(source, root, config, dataset, parent_job):
    root, c = Path(root), load_config(config)
    dataset_root = root/'results'/dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID')
    require(job and job.isdigit(), 'Dispatcher requires Slurm CPU allocation')
    decision_path = dataset_root/f'dispatch_from_{parent_job}.json'
    with (dataset_root/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if decision_path.exists():
            return read(decision_path)
        prior = read(dataset_root/'dispatch_state.json') if (dataset_root/'dispatch_state.json').exists() else None
        if prior:
            require(prior['next_job_id'] == str(parent_job), 'Stale dispatcher cannot fork stream')
        scheduler = slurm_terminal_record(parent_job)
        try:
            require(not (dataset_root/'failed.json').exists(), 'Dataset frozen after failure')
            allowed = scheduler['state'] in ('TIMEOUT','NODE_FAIL') or (scheduler['state'] == 'COMPLETED' and scheduler['exit_code'] == '0:0')
            require(allowed, 'Parent scheduler state requires review: '+scheduler['state']+' '+scheduler['exit_code'])
            cursor = checkpoint_cursor(source, root, c, dataset, check_storage(root, c))
            if (dataset_root/'completed.json').exists():
                require(cursor['task_index'] == len(tasks(c)), 'Completion receipt lacks all reviewed tasks')
                decision = dict(status='completed', scheduler=scheduler, cursor=cursor, time=now())
            else:
                previous = prior['cursor'] if prior else dict(task_index=0, iteration=0)
                position = (cursor['task_index'], cursor['iteration'])
                previous_position = (previous['task_index'], previous['iteration'])
                require(position >= previous_position, 'Committed progress regressed')
                failures = 0 if position > previous_position else (prior['no_progress_count'] if prior else 0)+1
                require(failures <= c['scheduler']['maximum_no_progress_retries'], 'Repeated allocations made no committed progress')
                if scheduler['state'] in ('TIMEOUT','NODE_FAIL'):
                    require(cursor['has_checkpoint'] or cursor['task_index'] > previous['task_index'], 'Interrupted task has no verified checkpoint/completed prefix')
                next_job = submit_job(source, root, 'production', dataset, dependency='afterok:'+job,
                                      request_key=f'{dataset}:{parent_job}:continuation')
                next_dispatch = submit_job(source, root, 'dispatch', dataset, dependency='afterany:'+next_job, cpu=True,
                                           parent_job=next_job, request_key=f'{dataset}:{next_job}:dispatch')
                decision = dict(status='continued', scheduler=scheduler, cursor=cursor, no_progress_count=failures,
                                next_job_id=next_job, next_dispatch_job_id=next_dispatch, time=now())
                write(dataset_root/'dispatch_state.json', decision)
            write(decision_path, decision)
        except BaseException as exc:
            failure = dict(status='failed', error=repr(exc), scheduler=scheduler, traceback=traceback.format_exc(), time=now(), job_id=parent_job)
            write(dataset_root/'failed.json', failure)
            write(decision_path, failure)
            raise
    with (root/'summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summarize(source, root, c)
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('submit','cpu-gate','dataset-gate','production','dispatch'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--world', type=int, choices=(1,), default=1)
    p.add_argument('--parent-job')
    args = p.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    config = source/EXT/'campaign.json'
    check_source(source)
    if args.phase == 'submit':
        submit_initial(source, root, config)
    elif args.phase == 'cpu-gate':
        cpu_gate(source, root, config)
    else:
        require(args.dataset, 'Dataset required')
        if args.phase == 'dataset-gate':
            dataset_gate(source, root, config, args.dataset)
        elif args.phase == 'production':
            production(source, root, config, args.dataset)
        else:
            require(args.parent_job, 'Parent job required')
            dispatch(source, root, config, args.dataset, args.parent_job)


if __name__ == '__main__':
    main()
