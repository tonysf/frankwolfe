"""One serial, immutable Ruche stream replaying eight already selected runs with clocks.

No tuning, altered horizon or new initialization. CPU and real-data GPU gates
precede production; an afterany CPU dispatcher resumes committed work without
polling. Completed replay tensors are pruned after retaining measured traces.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXT = Path('extensions/wallclock_replay_20261018')
DATASETS = ('bmmc6224', 'snare_full')

def require(value, message):
    if not value:
        raise ValueError(message)

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(source, config):
    return {'config_sha256': sha256(config), 'source_manifest_sha256': sha256(Path(source) / 'SHA256SUMS')}

def check_source(source):
    subprocess.run(['sha256sum', '--status', '-c', 'SHA256SUMS'], cwd=source, check=True)

def require_gate(path, expected):
    gate = read(path)
    require(gate['status'] == 'passed', 'Prior gate not passed')
    for key, value in expected.items():
        require(gate[key] == value, 'Prior gate source/config differs')

def slurm_terminal_record(job):
    require(str(job).isdigit(), 'Invalid parent job identifier')
    command = ['sacct', '--noheader', '--parsable2', '--jobs=' + str(job), '--format=JobIDRaw,State,ExitCode']
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 3 and fields[0] == str(job):
            return {'job_id': str(job), 'state': fields[1].split()[0].rstrip('+'), 'exit_code': fields[2]}
    # Accounting can lag a finished dependency. The scheduler retains the live job record.
    result = subprocess.run(['scontrol', 'show', 'job', str(job), '--oneliner'], text=True, capture_output=True, check=True)
    fields = dict(word.split('=', 1) for word in result.stdout.split() if '=' in word)
    require(fields.get('JobId') == str(job), 'Scheduler did not identify parent job')
    return {'job_id': str(job), 'state': fields['JobState'], 'exit_code': fields['ExitCode']}

def submit_job(source, root, phase, dataset='', world=1, dependency=None, cpu=False, seconds=None, parent_job='', request_key=None):
    source, root = Path(source), Path(root)
    c = load_config(source / EXT / 'campaign.json')
    require(world == 1 and not dataset, 'Replay must remain one serial single-GPU stream')
    name = f'gwtime_{phase}_{dataset or world}'
    command = ['sbatch', '--parsable', '--job-name=' + name,
               '--output=' + str(root / 'scheduler' / (name + '-%j.out'))]
    if dependency:
        command += ['--dependency=' + dependency, '--kill-on-invalid-dep=yes']
    if cpu:
        command += [str(source / EXT / 'cpu.sbatch'), str(source), str(root), phase, dataset, parent_job]
    else:
        command += [f'--gres=gpu:{world}', f'--cpus-per-task={2 * world}', f'--mem={32 * world}G']
        if seconds:
            command += ['--time=' + str(math.ceil(seconds / 60))]
        command += [str(source / EXT / 'gpu.sbatch'), str(source), str(root), phase, dataset, str(world)]
    receipt = {'time': now(), 'phase': phase, 'dataset': dataset, 'world_size': world, 'command': command, 'request_key': request_key}
    request_path = None
    if request_key:
        request_path = root / 'submission_requests' / (hashlib.sha256(request_key.encode()).hexdigest() + '.json')
        if request_path.exists():
            prior = read(request_path)
            require(prior['command'] == command, 'Idempotent request command changed')
            if prior['status'] == 'submitted':
                return prior['job_id']
            require(prior['status'] == 'failed', 'Ambiguous prior sbatch request; refusing possible duplicate')
        write(request_path, dict(receipt, status='submitting'))
    result = subprocess.run(command, text=True, capture_output=True)
    receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    with (root / 'submission.jsonl').open('a') as f:
        f.write(json.dumps(receipt, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    if result.returncode != 0:
        if request_path:
            write(request_path, dict(receipt, status='failed'))
        raise ValueError('sbatch failed; partial receipt retained: ' + result.stderr)
    job = result.stdout.strip().split(';')[0]
    require(job.isdigit(), 'Unrecognized sbatch job identifier')
    if request_path:
        write(request_path, dict(receipt, status='submitted', job_id=job))
    return job

def reconcile_completed_checkpoint(checkpoint_root, dataset, task, meta):
    """Finish interrupted completed replay pruning within its exact checkpoint directory.

    Called only after completed-result validation. The completed metadata
    contains the durable checkpoint manifest, so cleanup remains idempotent
    even if an earlier cleanup already removed COMMITTED.json or some shards.
    Unexpected paths or a different on-disk manifest stop the stream.
    """
    require(meta['status'] == 'completed' and meta['completed_updates'] == meta['target_updates'] == task['iterations'],
            'Checkpoint cleanup requires a complete prescribed horizon')
    require(dataset in DATASETS, 'Checkpoint cleanup dataset is not in this campaign')
    relative = Path(dataset) / task['task_id']
    require(not relative.is_absolute() and '..' not in relative.parts, 'Checkpoint cleanup path escapes its root')
    base = Path(checkpoint_root).resolve()
    folder = base
    for part in relative.parts:
        folder = folder / part
        require(not folder.is_symlink(), 'Checkpoint cleanup cannot follow symlinks')
    if not folder.exists():
        return
    require(folder.is_dir() and folder.resolve().is_relative_to(base), 'Checkpoint cleanup directory is not confined')
    if not list(folder.iterdir()):
        return
    require(meta.get('retained_checkpoint') is False, 'Checkpoint cleanup requires explicit nonretention')
    record = meta['checkpoint']
    require(record['iteration'] == task['iterations'] and record['identity'] == meta['identity'],
            'Completed checkpoint metadata horizon/identity differs')
    generation_name = record['generation']
    require(Path(generation_name).name == generation_name and generation_name.startswith('generation-'),
            'Invalid completed checkpoint generation')
    require(len(record['shards']) == meta['world_size'], 'Completed checkpoint manifest lacks rank shards')
    expected_shards = {f'rank-{rank:04d}.pt' for rank in range(meta['world_size'])}
    require({shard['file'] for shard in record['shards']} == expected_shards, 'Invalid completed checkpoint shard names')
    require({path.name for path in folder.iterdir()} <= {'COMMITTED.json', generation_name},
            'Unexpected completed checkpoint directory contents')
    marker, generation = folder / 'COMMITTED.json', folder / generation_name
    require(not marker.is_symlink() and not generation.is_symlink(), 'Checkpoint cleanup cannot follow symlinks')
    if marker.exists():
        require(read(marker) == record, 'Completed checkpoint commit differs from result metadata')
    if generation.exists():
        require(generation.is_dir(), 'Completed checkpoint generation is not a directory')
        manifest = generation / 'manifest.json'
        require(not manifest.is_symlink(), 'Checkpoint cleanup cannot follow symlinks')
        if manifest.exists():
            require(read(manifest) == record, 'Completed checkpoint generation differs from result metadata')
        entries = list(generation.iterdir())
        require({path.name for path in entries} <= expected_shards | {'manifest.json'},
                'Unexpected files in completed checkpoint generation')
        require(all(path.is_file() and not path.is_symlink() for path in entries),
                'Completed checkpoint generation contains unsafe entries')
        # All paths and available manifests were checked before the first unlink.
        for path in entries:
            path.unlink()
        generation.rmdir()
    if marker.exists():
        marker.unlink()
    descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def load_config(path):
    c = read(path)
    require(c['schema_version'] == 1 and c['experiment'] == 'wallclock_replay_20261018', 'Wrong replay schema')
    require(set(c['instances']) == set(DATASETS), 'Exactly BMMC and SNARE are replayed')
    require(c['dtype'] == 'float64' and c['initialization'] == 'product', 'FP64 product initialization required')
    require(c['pairs'] == 1024 and c['epsilon'] == .001 and c['eta'] == 1000., 'Oracle or POT step changed')
    require(c['inner_tol'] == c['inner_gap_tol'] == 1e-7 and c['inner_max'] == 100000, 'Inner solver changed')
    require(c['baseline'] == {'full_seed': 810, 'iterations': 2000}, 'Full-gradient control changed')
    f = c['frames']
    require(f['iterations'] == 20000 and f['confirmation_seeds'] == [811, 812, 813], 'FRAMES horizons/seeds changed')
    require(f['exponents'] == {'gamma': .75, 'rho': .5, 'mu': .25} and f['early_stopping'] is False, 'FRAMES schedules changed')
    require(f['selected'] == {
        'bmmc6224': {'candidate_id': 'c08', 'prefactors': {'gamma0': .3, 'rho0': 1., 'mu_scale': 10.}},
        'snare_full': {'candidate_id': 'c06', 'prefactors': {'gamma0': .3, 'rho0': .1, 'mu_scale': 10.}}}, 'Selected prefactors changed')
    require(c['runner'] == {'checkpoint_every': 5000, 'checkpoint_seconds': 1800,
        'diagnostic_every': 500, 'max_seconds': 18000}, 'Replay runtime settings changed')
    require(c['preflight'] == {'iterations': 2, 'max_seconds': 2700, 'reference_absolute_tolerance': 1e-11}, 'Replay gate changed')
    require(c['scheduler']['maximum_concurrent_gpu_jobs'] == 1, 'Only one serial replay stream is permitted')
    require(c['scheduler']['cpus_per_gpu'] == 2 and c['scheduler']['memory_gib_per_gpu'] == 32, 'Replay resource request changed')
    require(c['scheduler']['time'] == '06:00:00', 'Six-hour replay allocation required')
    require(c['scheduler']['account'] == 'fwllm' and c['scheduler']['gpu_partition'] == 'gpua100', 'Wrong allocation')
    require(c['storage']['completed_checkpoint_retention'] is False, 'Replay final tensors must be pruned')
    require(c['storage']['required_quota_available_bytes'] >= 65 * 1024**3, 'Global checkpoint reserve missing')
    original = read(Path(path).parent.parent / 'large_pot_frames_20261016/campaign.json')
    for name in DATASETS:
        require(c['instances'][name] == original['instances'][name], 'Original data identity/topology changed')
        require(c['instances'][name]['world_size'] == 1, 'Replays require exactly one A100')
    return c


def tasks(c):
    result = []
    for dataset in DATASETS:
        result.append(dict(dataset=dataset, task_id='baseline/pot-full_seed810', method='pot-full', seed=810, iterations=2000))
        chosen = c['frames']['selected'][dataset]
        for seed in c['frames']['confirmation_seeds']:
            result.append(dict(dataset=dataset, task_id=f'confirmation/{chosen["candidate_id"]}_seed{seed}',
                method='frames', seed=seed, iterations=20000, **chosen))
    return result


def check_storage(root, c, fresh=False):
    receipt = read(Path(root) / 'storage_receipt.json')
    require(receipt.get('quota_verified') is True, 'Receipt must establish quota headroom')
    require(receipt['quota_available_bytes'] >= c['storage']['required_quota_available_bytes'], 'Insufficient global quota headroom')
    base = Path(root).resolve()
    checkpoint_root = Path(receipt['checkpoint_root'])
    require(checkpoint_root.is_absolute() and checkpoint_root.is_dir(), 'Checkpoint root missing')
    require(checkpoint_root.resolve() == base / 'checkpoints' and not checkpoint_root.is_symlink(), 'Replay checkpoints must use their isolated root')
    if fresh:
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(receipt['verified_utc'])).total_seconds()
        require(0 <= age <= c['storage']['receipt_max_age_seconds'], 'Quota receipt is stale')
    return checkpoint_root


def expected_config(c, task):
    value = dict(method=task['method'], seed=task['seed'], iterations=task['iterations'], epsilon=c['epsilon'],
        pairs=c['pairs'], gamma0=1., rho0=1., mu_scale=10., dtype='float64', device='cuda',
        inner_max=c['inner_max'], diagnostic_every=c['runner']['diagnostic_every'], exact_block_rows=128)
    value.update(task.get('prefactors', {}))
    return value


def validate_result(meta, task, c, source):
    instance = c['instances'][task['dataset']]
    require(meta['status'] == 'completed' and meta['completed_updates'] == meta['target_updates'] == task['iterations'], 'Prescribed horizon incomplete')
    require(meta['config'] == expected_config(c, task), 'Result configuration changed')
    require(meta['array_files_sha256'] == instance['array_files_sha256'] and meta['manifest_sha256'] == instance['instance_manifest_sha256'], 'Result data identity changed')
    require(meta['world_size'] == 1 and meta['dimensions'] == dict(n=instance['n'], m=instance['n']), 'Result dimensions/topology differ')
    require('A100' in meta['identity']['device_model'], 'Result did not run on A100')
    require(meta.get('retained_checkpoint') is False, 'Replay final checkpoint unexpectedly retained')
    require(meta['source_file_sha256'], 'Numerical source identity missing')
    for name, digest in meta['source_file_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe numerical source path')
        require(sha256(Path(source) / name) == digest, 'Numerical source changed')
    require(meta['schedule']['inner_tolerance'] == meta['schedule']['gap_tolerance'] == c['inner_tol'], 'Inner certificate tolerance changed')
    require(meta['schedule']['eta'] == c['eta'], 'POT step changed')
    require({k:meta['schedule'][k+'_exponent'] for k in ('gamma','rho','mu')} == c['frames']['exponents'], 'Theoretical schedule changed')
    trace = meta['trace']
    require(trace and trace[0]['iteration'] == 0 and trace[-1]['iteration'] == task['iterations'], 'Trace endpoints incomplete')
    previous_time, previous_iteration = -1., -1
    for point in trace:
        elapsed = point['elapsed_seconds']
        require(math.isfinite(elapsed) and elapsed >= previous_time, 'Missing or nonmonotone observed timing')
        require(point['iteration'] > previous_iteration and math.isfinite(point['rounded_loss']), 'Invalid loss trajectory')
        previous_time, previous_iteration = elapsed, point['iteration']
    require(meta['elapsed_seconds'] >= previous_time and math.isfinite(meta['elapsed_seconds']), 'Final time precedes diagnostic')
    require(meta['final_loss'] == trace[-1]['rounded_loss'], 'Final loss differs from final diagnostic')
    return meta['final_loss']


def runner_command(source, root, c, task, seconds, checkpoint_root):
    dataset = task['dataset']
    command = [sys.executable, str(Path(source) / EXT / 'runner.py')]
    options = dict(instance=c['instances'][dataset]['path'], out=str(Path(root) / 'results' / dataset / task['task_id']),
        checkpoint_dir=str(Path(checkpoint_root) / dataset / task['task_id']), method=task['method'], seed=task['seed'],
        iterations=task['iterations'], max_seconds=seconds, checkpoint_every=c['runner']['checkpoint_every'],
        checkpoint_seconds=c['runner']['checkpoint_seconds'], diagnostic_every=c['runner']['diagnostic_every'],
        epsilon=c['epsilon'], pairs=c['pairs'], dtype='float64', device='cuda', inner_max=c['inner_max'])
    options.update(task.get('prefactors', {}))
    for key, value in options.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    return command


def compare_reference(meta, task, references, tolerance):
    reference_id = task.get('reference_task_id', task['task_id'])
    reference = references['datasets'][task['dataset']]['runs'][reference_id]
    by_iteration = {point['iteration']: point for point in reference['trace']}
    comparisons = []
    for point in meta['trace']:
        require(point['iteration'] in by_iteration, 'Original trace lacks replay diagnostic')
        expected = by_iteration[point['iteration']]['rounded_loss']
        error = abs(point['rounded_loss'] - expected)
        require(error <= tolerance, f'Replay changed numerical trajectory at update {point["iteration"]}: error {error}')
        comparisons.append(dict(iteration=point['iteration'], absolute_error=error))
    return comparisons


def cpu_gate(source, root, config):
    load_config(config)
    out = Path(root) / 'checks/cpu'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'tests.log').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', str(EXT)], cwd=source,
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(out / 'gate.json', dict(status='passed', **identity(source, config), time=now()))


def gpu_gate(source, root, config):
    c = load_config(config)
    require_gate(Path(root) / 'checks/cpu/gate.json', identity(source, config))
    checkpoint_root = check_storage(root, c)
    references = read(Path(source) / EXT / 'references.json')
    records = []
    started = time.monotonic()
    for original in tasks(c):
        if original['seed'] not in (810, 811):
            continue
        task = dict(original, iterations=2, reference_task_id=original['task_id'], task_id='preflight/' + original['method'])
        remaining = c['preflight']['max_seconds'] - (time.monotonic() - started)
        require(remaining > 600, 'Actual-size GPU gate exceeded budget')
        subprocess.run(runner_command(source, root, c, task, remaining, checkpoint_root), cwd=source, check=True)
        meta = read(Path(root) / 'results' / task['dataset'] / task['task_id'] / 'metadata.json')
        validate_result(meta, task, c, source)
        require(meta['max_memory_reserved_fraction'] <= c['scheduler']['maximum_gpu_reserved_fraction'], 'Insufficient GPU memory margin')
        comparisons = compare_reference(meta, task, references, c['preflight']['reference_absolute_tolerance'])
        reconcile_completed_checkpoint(checkpoint_root, task['dataset'], task, meta)
        records.append(dict(dataset=task['dataset'], method=task['method'], seed=task['seed'], reference_comparison=comparisons,
            maximum_reference_loss_error=max(point['absolute_error'] for point in comparisons),
            max_memory_reserved_fraction=meta['max_memory_reserved_fraction'], elapsed_seconds=meta['elapsed_seconds']))
    require(len(records) == 4, 'Actual-size gate must cover both methods and both datasets')
    write(Path(root) / 'checks/gpu/gate.json', dict(status='passed', **identity(source, config), time=now(), runs=records))


def production(source, root, config):
    root, c = Path(root), load_config(config)
    expected = identity(source, config)
    require_gate(root / 'checks/gpu/gate.json', expected)
    checkpoint_root = check_storage(root, c)
    state_root = root / 'stream'
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (state_root / 'failed.json').exists(), 'Replay stream frozen after failure')
        if (state_root / 'completed.json').exists():
            return
        job = os.environ.get('SLURM_JOB_ID')
        require(job and job.isdigit(), 'Production requires Slurm allocation')
        chunk = state_root / f'chunk_{job}.json'
        write(chunk, dict(status='running', job_id=job, **expected, started_utc=now()))
        started = time.monotonic()
        complete = False
        try:
            references = read(Path(source) / EXT / 'references.json')
            for task in tasks(c):
                out = root / 'results' / task['dataset'] / task['task_id']
                if (out / 'metadata.json').exists() and read(out / 'metadata.json').get('status') == 'completed':
                    meta = read(out / 'metadata.json')
                    validate_result(meta, task, c, source)
                    compare_reference(meta, task, references, c['preflight']['reference_absolute_tolerance'])
                    reconcile_completed_checkpoint(checkpoint_root, task['dataset'], task, meta)
                    continue
                remaining = c['runner']['max_seconds'] - (time.monotonic() - started)
                if remaining < 600:
                    break
                out.mkdir(parents=True, exist_ok=True)
                write(chunk, dict(status='running', task_id=task['task_id'], dataset=task['dataset'], job_id=job, **expected, time=now()))
                with (out / f'job_{job}.log').open('a') as stream:
                    subprocess.run(runner_command(source, root, c, task, remaining, checkpoint_root), cwd=source,
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
                progress = read(out / 'progress.json')
                require(progress['status'] in ('completed', 'checkpointed'), 'Runner neither completed nor checkpointed')
                if progress['status'] == 'checkpointed':
                    break
                meta = read(out / 'metadata.json')
                validate_result(meta, task, c, source)
                compare_reference(meta, task, references, c['preflight']['reference_absolute_tolerance'])
                reconcile_completed_checkpoint(checkpoint_root, task['dataset'], task, meta)
            else:
                complete = True
            if complete:
                write(state_root / 'completed.json', dict(status='completed', **expected, runs=8, last_job_id=job, time=now()))
            write(chunk, dict(status='completed' if complete else 'checkpointed', job_id=job, **expected, time=now()))
        except BaseException as exc:
            write(chunk, dict(status='failed', job_id=job, error=repr(exc), traceback=traceback.format_exc(), time=now()))
            raise


def checkpoint_cursor(source, root, c, checkpoint_root):
    for index, task in enumerate(tasks(c)):
        out = Path(root) / 'results' / task['dataset'] / task['task_id']
        if (out / 'metadata.json').exists() and read(out / 'metadata.json').get('status') == 'completed':
            meta = read(out / 'metadata.json')
            validate_result(meta, task, c, source)
            reconcile_completed_checkpoint(checkpoint_root, task['dataset'], task, meta)
            continue
        require(not (out / 'failure.json').exists(), 'Explicit runner failure requires review')
        folder = Path(checkpoint_root) / task['dataset'] / task['task_id']
        marker = folder / 'COMMITTED.json'
        cursor = dict(task_index=index, dataset=task['dataset'], task_id=task['task_id'], iteration=0, has_checkpoint=False)
        if not marker.exists():
            require(not (out / 'progress.json').exists(), 'Started task lacks committed checkpoint')
            return cursor
        checkpoint = read(marker)
        instance = c['instances'][task['dataset']]
        ident = checkpoint['identity']
        require(checkpoint['world_size'] == ident['world_size'] == 1, 'Checkpoint topology changed')
        require(ident['n'] == ident['m'] == instance['n'], 'Checkpoint dimensions changed')
        require(ident['array_files_sha256'] == instance['array_files_sha256'] and ident['manifest_sha256'] == instance['instance_manifest_sha256'], 'Checkpoint dataset changed')
        require(ident['config'] == expected_config(c, task), 'Checkpoint run settings changed')
        require(ident['source_file_sha256'], 'Checkpoint numerical source identity missing')
        for name, digest in ident['source_file_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid checkpoint source path')
            require(sha256(Path(source) / name) == digest, 'Checkpoint numerical source changed')
        generation = checkpoint['generation']
        require(Path(generation).name == generation and generation.startswith('generation-'), 'Unsafe checkpoint generation')
        require(read(folder / generation / 'manifest.json') == checkpoint, 'Checkpoint commit/generation differ')
        require(0 <= checkpoint['iteration'] <= task['iterations'], 'Checkpoint cursor outside horizon')
        require(len(checkpoint['shards']) == 1, 'Checkpoint requires exactly one shard')
        shard = checkpoint['shards'][0]
        require(shard['file'] == 'rank-0000.pt', 'Wrong checkpoint shard name')
        path = folder / generation / shard['file']
        require(path.is_file() and path.stat().st_size == shard['bytes'] and sha256(path) == shard['sha256'], 'Checkpoint shard integrity failed')
        return dict(cursor, iteration=checkpoint['iteration'], has_checkpoint=True, committed_manifest_sha256=sha256(marker))
    return dict(task_index=8, dataset=None, task_id=None, iteration=0, has_checkpoint=False)


def dispatch(source, root, config, parent_job):
    root, c = Path(root), load_config(config)
    state_root = root / 'stream'
    state_root.mkdir(parents=True, exist_ok=True)
    dispatcher_job = os.environ.get('SLURM_JOB_ID')
    require(dispatcher_job and dispatcher_job.isdigit(), 'Dispatcher requires CPU Slurm allocation')
    path = state_root / f'dispatch_from_{parent_job}.json'
    with (state_root / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return read(path)
        prior = read(state_root / 'dispatch_state.json') if (state_root / 'dispatch_state.json').exists() else None
        if prior:
            require(prior['next_job_id'] == str(parent_job), 'Stale dispatcher cannot fork replay stream')
        scheduler = slurm_terminal_record(parent_job)
        try:
            require(not (state_root / 'failed.json').exists(), 'Replay stream already frozen')
            allowed = scheduler['state'] in ('TIMEOUT','NODE_FAIL') or (scheduler['state'] == 'COMPLETED' and scheduler['exit_code'] == '0:0')
            require(allowed, 'Parent scheduler state requires review: ' + scheduler['state'] + ' ' + scheduler['exit_code'])
            cursor = checkpoint_cursor(source, root, c, check_storage(root,c))
            if (state_root / 'completed.json').exists():
                receipt = read(state_root / 'completed.json')
                require(receipt['runs'] == cursor['task_index'] == 8, 'Completion requires all eight runs')
                for key,value in identity(source, config).items():
                    require(receipt[key] == value, 'Completion source/config changed')
                report = submit_job(source, root, 'report', cpu=True, request_key=f'{parent_job}:final-report')
                decision = dict(status='completed', scheduler=scheduler, cursor=cursor, report_job_id=report,time=now())
            else:
                previous = prior['cursor'] if prior else dict(task_index=0,iteration=0)
                position, old_position = (cursor['task_index'],cursor['iteration']), (previous['task_index'],previous['iteration'])
                require(position >= old_position, 'Committed progress regressed')
                failures = 0 if position > old_position else (prior['no_progress_count'] if prior else 0)+1
                require(failures <= c['scheduler']['maximum_no_progress_retries'], 'Repeated allocations made no progress')
                if scheduler['state'] in ('TIMEOUT','NODE_FAIL'):
                    require(cursor['has_checkpoint'] or cursor['task_index'] > previous['task_index'], 'Interrupted run has no verified state')
                next_job = submit_job(source,root,'production',dependency='afterok:'+dispatcher_job,request_key=f'{parent_job}:continuation')
                next_dispatch = submit_job(source,root,'dispatch',dependency='afterany:'+next_job,cpu=True,parent_job=next_job,request_key=f'{next_job}:dispatch')
                decision = dict(status='continued',scheduler=scheduler,cursor=cursor,no_progress_count=failures,
                    next_job_id=next_job,next_dispatch_job_id=next_dispatch,time=now())
                write(state_root / 'dispatch_state.json',decision)
            write(path,decision)
            return decision
        except BaseException as exc:
            failure = dict(status='failed',error=repr(exc),scheduler=scheduler,traceback=traceback.format_exc(),time=now())
            write(state_root / 'failed.json',failure)
            write(path,failure)
            submit_job(source,root,'report',cpu=True,request_key=f'{parent_job}:failure-report')
            raise


def collect_report(source, root, config):
    root,c = Path(root),load_config(config)
    records,missing = [],[]
    references = read(Path(source) / EXT / 'references.json')
    for task in tasks(c):
        path = root / 'results' / task['dataset'] / task['task_id'] / 'metadata.json'
        if not path.exists() or read(path).get('status') != 'completed':
            missing.append(dict(dataset=task['dataset'],task_id=task['task_id']))
            continue
        meta = read(path)
        validate_result(meta,task,c,source)
        comparisons = compare_reference(meta,task,references,c['preflight']['reference_absolute_tolerance'])
        records.append(dict(**task,final_loss=meta['final_loss'],elapsed_seconds=meta['elapsed_seconds'],
            trace=meta['trace'],counters=meta['counters'],timing=meta.get('timing'),metadata_path=str(path),metadata_sha256=sha256(path),
            maximum_reference_loss_error=max(point['absolute_error'] for point in comparisons)))
    failure = read(root / 'stream/failed.json') if (root / 'stream/failed.json').exists() else None
    report = dict(status='failed' if failure else ('completed' if len(records)==8 else 'partial'),
        **identity(source,config),generated_utc=now(),completed_runs=len(records),total_runs=8,runs=records,missing_tasks=missing,failure=failure)
    write(root / 'reports/report.json',report)
    if report['status'] == 'completed':
        subprocess.run([sys.executable, str(Path(source) / EXT / 'report.py'), '--root', str(root),
            '--output', str(root / 'reports')], cwd=source, check=True)
    return report


def submit_initial(source, root, config):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    c = load_config(config)
    check_source(source)
    check_storage(root,c,fresh=True)
    require((Path(source) / EXT / 'references.json').is_file(), 'Frozen original traces are required')
    (root / 'submission_lock').mkdir()
    (root / 'scheduler').mkdir(exist_ok=True)
    write(root / 'campaign_identity.json',dict(**identity(source,config),source=str(source),submitted_utc=now()))
    cpu = submit_job(source,root,'cpu-gate',cpu=True,request_key='initial:cpu-gate')
    gate = submit_job(source,root,'gpu-gate',dependency='afterok:'+cpu,seconds=3600,request_key='initial:gpu-gate')
    production_job = submit_job(source,root,'production',dependency='afterok:'+gate,request_key='initial:production')
    dispatcher = submit_job(source,root,'dispatch',dependency='afterany:'+production_job,cpu=True,parent_job=production_job,request_key=f'{production_job}:dispatch')
    report = submit_job(source,root,'report',dependency='afterany:'+dispatcher,cpu=True,request_key='initial:report')
    receipt = dict(status='queued',jobs=[cpu,gate,production_job,dispatcher,report],cpu_gate=cpu,gpu_gate=gate,
        production_job=production_job,dispatcher_job=dispatcher,initial_report=report,time=now())
    write(root / 'initial_submission.json',receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['submit','cpu-gate','gpu-gate','production','dispatch','report'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--world',type=int,choices=[1],default=1)
    p.add_argument('--parent-job')
    args = p.parse_args()
    source,root=args.source.resolve(),args.root.resolve()
    config=source / EXT / 'campaign.json'
    check_source(source)
    if args.phase=='submit':
        submit_initial(source,root,config)
    elif args.phase=='cpu-gate':
        cpu_gate(source,root,config)
    elif args.phase=='gpu-gate':
        gpu_gate(source,root,config)
    elif args.phase=='production':
        production(source,root,config)
    elif args.phase=='dispatch':
        require(args.parent_job,'Parent job required')
        dispatch(source,root,config,args.parent_job)
    else:
        (root / 'reports').mkdir(exist_ok=True)
        with (root / 'reports/report_process.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            collect_report(source,root,config)


if __name__ == '__main__':
    main()
