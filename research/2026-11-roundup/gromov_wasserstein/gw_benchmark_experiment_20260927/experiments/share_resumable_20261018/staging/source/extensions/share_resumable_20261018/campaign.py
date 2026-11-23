"""Event-driven Ruche campaign: validated gates, fixed horizons and resumable jobs.

One serial SHARE stream bounds checkpoint storage. Final confirmation states
retain P, momentum and RNG for future extensions beyond the original horizon. Slurm dependencies and
one continuation submission per checkpointed allocation replace active polling.
Only the final feasible loss of all complete development runs selects FraMES.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXT = Path('extensions/share_resumable_20261018')
DATASETS = ('shareseq_full',)


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


def load_config(path):
    c = read(path)
    require(c['schema_version'] == 1 and set(c['instances']) == set(DATASETS), 'Unexpected campaign schema/datasets')
    require(c['dtype'] == 'float64' and c['initialization'] == 'product', 'FP64 product initialization required')
    require(c['pairs'] == 1024 and c['epsilon'] == .001 and c['eta'] == 1000, 'POT step/oracle changed')
    require(c['inner_tol'] == c['inner_gap_tol'] == 1e-7, 'Numerical tolerances changed')
    require(c['baseline'] == {'iterations': 2000, 'full_seed': 810, 'pair_seeds': [811, 812, 813]}, 'Baseline design changed')
    f = c['frames']
    require(f['iterations'] == 20000 and f['development_seeds'] == [810, 814] and f['confirmation_seeds'] == [811, 812, 813], 'FraMES horizons/seeds changed')
    require(f['gamma0'] == [.3, 1.] and f['mu_scale'] == [.1, 1., 10., 100.] and f['rho0'] == [.1, .3, 1.], 'FraMES grid changed')
    require(f['exponents'] == {'gamma': .75, 'rho': .5, 'mu': .25} and not f['early_stopping'], 'Theoretical exponents or stopping changed')
    require(0 < c['runner']['max_seconds'] <= 22 * 3600, 'Insufficient Slurm walltime margin')
    for name, v in c['instances'].items():
        require(v['world_size'] == (4 if name == 'shareseq_full' else 1), 'Wrong GPU topology')
        require(set(v['array_files_sha256']) == {'A.npy', 'B.npy', 'a.npy', 'b.npy'}, 'Missing array identity')
        require(Path(v['path']).is_absolute(), 'Dataset paths must be absolute')
        require(v['old_prefactors'] in [x['prefactors'] for x in candidates(c)], 'Old control absent from grid')
    require(c['storage']['completed_checkpoint_retention'] == 'frames_confirmation_only', 'Final confirmation checkpoint retention required')
    require(c['storage']['required_quota_available_bytes'] >= 68 * 1024 ** 3, 'Checkpoint and other-stream storage reserve missing')
    require(c['storage']['retained_confirmation_state_count'] == 3, 'All final confirmation states must survive')
    return c


def candidates(c):
    f = c['frames']
    return [dict(candidate_id=f'c{i:02d}', prefactors=dict(gamma0=g, mu_scale=m, rho0=r))
            for i, (g, m, r) in enumerate(itertools.product(f['gamma0'], f['mu_scale'], f['rho0']))]


def tasks(c, selected=None):
    b, f = c['baseline'], c['frames']
    result = [dict(task_id=f'baseline/pot-full_seed{b["full_seed"]}', method='pot-full', seed=b['full_seed'], iterations=b['iterations'])]
    result += [dict(task_id=f'baseline/pot-pair_seed{s}', method='pot-pair', seed=s, iterations=b['iterations']) for s in b['pair_seeds']]
    result += [dict(task_id=f'development/{x["candidate_id"]}_seed{s}', method='frames', seed=s,
                    iterations=f['iterations'], **x) for x in candidates(c) for s in f['development_seeds']]
    if selected is not None:
        result += [dict(task_id=f'confirmation/{selected["candidate_id"]}_seed{s}', method='frames', seed=s,
                        iterations=f['iterations'], candidate_id=selected['candidate_id'], prefactors=selected['prefactors'])
                   for s in f['confirmation_seeds']]
    return result


def identity(source, config):
    return {'config_sha256': sha256(config), 'source_manifest_sha256': sha256(Path(source) / 'SHA256SUMS')}


def check_source(source):
    subprocess.run(['sha256sum', '--status', '-c', 'SHA256SUMS'], cwd=source, check=True)


def check_storage(root, c, fresh=False):
    receipt = read(Path(root) / 'storage_receipt.json')
    require(receipt['quota_available_bytes'] >= c['storage']['required_quota_available_bytes'], 'Verified quota headroom below 68 GiB (64 GiB SHARE plus 4 GiB other campaigns)')
    require(receipt.get('quota_verified') is True, 'Storage receipt must establish quota, not filesystem free space')
    checkpoint_root = Path(receipt['checkpoint_root'])
    require(checkpoint_root.is_absolute() and checkpoint_root.is_dir(), 'Checkpoint directory missing')
    if fresh:
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(receipt['verified_utc'])).total_seconds()
        require(0 <= age <= c['storage']['receipt_max_age_seconds'], 'Quota receipt is stale')
    return checkpoint_root


def validate_result(meta, task, instance, c, source=None):
    require(meta['status'] == 'completed', 'Run is not complete')
    require(meta['completed_updates'] == meta['target_updates'] == task['iterations'], 'Fixed horizon incomplete')
    require(math.isfinite(meta['final_loss']), 'Nonfinite final feasible GW loss')
    require(meta['array_files_sha256'] == instance['array_files_sha256'], 'Result dataset differs')
    require(meta['manifest_sha256'] == instance['instance_manifest_sha256'], 'Result manifest differs')
    require(meta['dimensions'] == dict(n=instance['n'], m=instance['n']) and meta['world_size'] == instance['world_size'], 'Result dimensions/topology differ')
    if source is not None:
        require(bool(meta['source_file_sha256']), 'Missing numerical source identity')
        for name, expected in meta['source_file_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid numerical source path')
            require(sha256(Path(source) / name) == expected, 'Result numerical source differs: ' + name)
    cfg = meta['config']
    for key in ('method', 'seed', 'iterations'):
        require(cfg[key] == task[key], 'Run configuration differs: ' + key)
    require(cfg['dtype'] == c['dtype'] and cfg['pairs'] == c['pairs'] and cfg['epsilon'] == c['epsilon'], 'Numerical configuration differs')
    require(cfg['inner_max'] == c['inner_max'] and cfg['diagnostic_every'] == c['runner']['diagnostic_every'], 'Inner budget/diagnostic schedule changed')
    schedule = meta['schedule']
    require(schedule['inner_tolerance'] == schedule['gap_tolerance'] == c['inner_tol'], 'Result tolerances differ')
    require(schedule['eta'] == c['eta'], 'Result POT step differs')
    require({k: schedule[k + '_exponent'] for k in ('gamma', 'rho', 'mu')} == c['frames']['exponents'], 'Result theoretical exponents differ')
    for key, value in task.get('prefactors', {}).items():
        require(cfg[key] == value, 'FraMES prefactor differs: ' + key)
    if task['method'] == 'frames' and task['task_id'].startswith('confirmation/'):
        require(meta.get('retained_checkpoint') is True, 'Final FRAMES confirmation checkpoint was not retained')
        require(meta['checkpoint']['iteration'] == task['iterations'], 'Retained final state is at the wrong horizon')
    return float(meta['final_loss'])


def validate_retained_state(checkpoint_root, dataset, task, meta):
    """Require the durable final state before reporting a confirmation complete.

    The runner hashes each rank shard when committing it; a future extension
    rechecks every hash before loading. Here lightweight manifest/size checks
    ensure retention without rereading 15 GiB after every controller restart.
    """
    if task['method'] != 'frames' or not task['task_id'].startswith('confirmation/'):
        return
    folder = Path(checkpoint_root) / dataset / task['task_id']
    marker = folder / 'COMMITTED.json'
    require(marker.is_file(), 'Final confirmation checkpoint is missing')
    record = read(marker)
    require(record == meta['checkpoint'], 'Retained checkpoint differs from completed result')
    require(record['iteration'] == task['iterations'] and record['identity'] == meta['identity'], 'Retained state horizon/identity differs')
    generation = record['generation']
    require(Path(generation).name == generation and generation.startswith('generation-'), 'Invalid retained generation')
    require(read(folder / generation / 'manifest.json') == record, 'Retained generation manifest differs')
    require(len(record['shards']) == meta['world_size'], 'Retained state is missing rank shards')
    for rank, shard in enumerate(record['shards']):
        require(shard['file'] == f'rank-{rank:04d}.pt', 'Retained checkpoint rank name differs')
        path = folder / generation / shard['file']
        require(path.is_file() and path.stat().st_size == shard['bytes'], 'Retained checkpoint shard is missing or truncated')


def selection(c, dataset_root, source=None):
    rows = []
    instance = c['instances'][Path(dataset_root).name]
    for candidate in candidates(c):
        values = []
        for task in tasks(c):
            if task.get('candidate_id') == candidate['candidate_id']:
                values.append(validate_result(read(Path(dataset_root) / task['task_id'] / 'metadata.json'), task, instance, c, source))
        require(len(values) == len(c['frames']['development_seeds']), 'Missing development seeds')
        rows.append(dict(**candidate, final_losses=values, median_final_feasible_loss=statistics.median(values)))
    winner = min(rows, key=lambda x: (x['median_final_feasible_loss'], x['candidate_id']))
    return {'status': 'selected', 'criterion': c['frames']['selection'], 'candidate_id': winner['candidate_id'],
            'prefactors': winner['prefactors'], 'development_seed_count': 2, 'full_horizon': 20000,
            'ranking': sorted(rows, key=lambda x: (x['median_final_feasible_loss'], x['candidate_id']))}


def runner_command(source, root, c, dataset, task, seconds, checkpoint_root):
    source, root = Path(source), Path(root)
    world = c['instances'][dataset]['world_size']
    command = [sys.executable]
    if world > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={world}']
    command += [str(source / EXT / 'runner.py')]
    options = dict(instance=c['instances'][dataset]['path'], out=str(root / 'results' / dataset / task['task_id']),
                   checkpoint_dir=str(Path(checkpoint_root) / dataset / task['task_id']),
                   method=task['method'], seed=task['seed'], iterations=task['iterations'],
                   max_seconds=seconds, checkpoint_every=c['runner']['checkpoint_every'],
                   checkpoint_seconds=c['runner']['checkpoint_seconds'],
                   diagnostic_every=c['runner']['diagnostic_every'], epsilon=c['epsilon'], pairs=c['pairs'],
                   dtype=c['dtype'], device='cuda', inner_max=c['inner_max'])
    options.update(task.get('prefactors', {}))
    for key, value in options.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    if task['method'] == 'frames' and task['task_id'].startswith('confirmation/'):
        command.append('--retain-checkpoint')
    return command


def submit_job(source, root, phase, dataset='', world=1, dependency=None, cpu=False, seconds=None, parent_job='', request_key=None):
    source, root = Path(source), Path(root)
    c = load_config(source / EXT / 'campaign.json')
    name = f'gwshare_resume_{phase}_{dataset or world}'
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


def submit_initial(source, root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    c = load_config(config)
    check_source(source)
    check_storage(root, c, fresh=True)
    (root / 'submission_lock').mkdir()  # Exclusive: partial submissions cannot be duplicated silently.
    (root / 'scheduler').mkdir(exist_ok=True)
    write(root / 'campaign_identity.json', dict(**identity(source, config), submitted_utc=now(), source=str(source)))
    cpu = submit_job(source, root, 'cpu-gate', cpu=True)
    tiny = {world: submit_job(source, root, 'tiny-gate', world=world, dependency='afterok:' + cpu, seconds=3600) for world in (4,)}
    jobs = [cpu, *tiny.values()]
    for dataset in DATASETS:
        world = c['instances'][dataset]['world_size']
        gate = submit_job(source, root, 'dataset-gate', dataset, world, 'afterok:' + tiny[world], seconds=4 * 3600)
        run = submit_job(source, root, 'production', dataset, world, 'afterok:' + gate)
        dispatcher = submit_job(source, root, 'dispatch', dataset, dependency='afterany:' + run, cpu=True,
                                parent_job=run, request_key=f'{dataset}:{run}:dispatch')
        jobs += [gate, run, dispatcher]
    report = submit_job(source, root, 'report', dependency='afterany:' + ':'.join(jobs), cpu=True)
    write(root / 'initial_submission.json', {'status': 'queued', 'jobs': jobs, 'initial_report': report, 'time': now()})


def cpu_gate(source, root, config):
    out = Path(root) / 'checks/cpu'
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-m', 'pytest', '-q', str(EXT), 'extensions/large_pot_frames_20261016/test_checkpoint.py']
    with (out / 'tests.log').open('w') as f:
        subprocess.run(command, cwd=source, stdout=f, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(Path(source) / EXT / 'validation.py'), '--device', 'cpu', '--out', str(out)], check=True)
    record = read(out / 'checks.json')
    require(record['status'] == 'passed' and record['world_size'] == 1, 'CPU checks did not pass')
    write(out / 'gate.json', dict(status='passed', **identity(source, config), time=now()))


def tiny_gate(source, root, config, world):
    out = Path(root) / f'checks/gpu{world}'
    require_gate(Path(root) / 'checks/cpu/gate.json', identity(source, config))
    command = [sys.executable]
    if world > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={world}']
    command += [str(Path(source) / EXT / 'validation.py'), '--device', 'cuda', '--out', str(out)]
    subprocess.run(command, cwd=source, check=True)
    record = read(out / 'checks.json')
    require(record['status'] == 'passed' and record['world_size'] == world and record.get('checks'), 'GPU checks did not pass')
    resume_checks = tiny_resume_checks(source, out, world)
    write(out / 'gate.json', dict(status='passed', **identity(source, config), time=now(), resume_checks=resume_checks))


def tiny_resume_checks(source, out, world):
    """Exercise distributed RNG/state persistence before allocating full data."""
    import numpy as np
    out = Path(out)
    instance = out / 'resume_input'
    instance.mkdir()
    rng = np.random.default_rng(780)
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(6, 3))
    values = dict(A=(np.linalg.norm(x[:, None]-x[None, :], axis=-1)*.03).astype('float32'),
                  B=(np.linalg.norm(y[:, None]-y[None, :], axis=-1)*.03).astype('float32'),
                  a=np.ones(8, dtype='float64')/8, b=np.ones(6, dtype='float64')/6)
    for name, value in values.items():
        np.save(instance / (name + '.npy'), value)
    write(instance / 'manifest.json', dict(n=8, m=6, array_files_sha256={name + '.npy': sha256(instance / (name + '.npy')) for name in values}))
    prefix = [sys.executable]
    if world > 1:
        prefix += ['-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={world}']
    records = []
    for method in ('pot-full', 'pot-pair', 'frames'):
        command = prefix + [str(Path(source) / EXT / 'runner.py'), '--instance', str(instance), '--method', method,
                            '--seed', '810', '--iterations', '4', '--pairs', '1024', '--device', 'cuda',
                            '--dtype', 'float64', '--checkpoint-every', '2', '--diagnostic-every', '2']
        full, resumed = out / ('resume_' + method + '_full'), out / ('resume_' + method + '_split')
        subprocess.run(command + ['--out', str(full)], check=True)
        subprocess.run(command + ['--out', str(resumed), '--chunk-updates', '2'], check=True)
        require(read(resumed / 'progress.json')['status'] == 'checkpointed', 'Tiny resume gate did not checkpoint')
        subprocess.run(command + ['--out', str(resumed)], check=True)
        a, b = read(full / 'metadata.json'), read(resumed / 'metadata.json')
        require(a['status'] == b['status'] == 'completed', 'Tiny resume gate incomplete')
        require(a['trace'] == b['trace'] and a['counters'] == b['counters'], 'GPU resume altered trajectory or work counters')
        records.append(dict(method=method, exact_trace_and_counter_match=True, world_size=world))
    return records


def require_gate(path, expected):
    gate = read(path)
    require(gate['status'] == 'passed', 'Prior gate not passed')
    for key, value in expected.items():
        require(gate[key] == value, 'Prior gate source/config differs')


def dataset_gate(source, root, config, dataset):
    c = load_config(config)
    expected = identity(source, config)
    world = c['instances'][dataset]['world_size']
    require_gate(Path(root) / f'checks/gpu{world}/gate.json', expected)
    checkpoint_root = check_storage(root, c)
    records = []
    started = time.monotonic()
    for method in ('pot-full', 'pot-pair', 'frames'):
        task = dict(task_id='preflight/' + method, method=method, seed=810,
                    iterations=c['preflight']['iterations'])
        if method == 'frames':
            task['prefactors'] = c['instances'][dataset]['old_prefactors']
        remaining = c['preflight']['max_seconds'] - (time.monotonic() - started)
        require(remaining > 600, 'Actual-size gate exceeded its time budget')
        command = runner_command(source, root, c, dataset, task, remaining, checkpoint_root)
        subprocess.run(command + ['--chunk-updates', '1'], cwd=source, check=True)
        progress = read(Path(root) / 'results' / dataset / task['task_id'] / 'progress.json')
        require(progress['status'] == 'checkpointed' and progress['completed_updates'] == 1, 'Actual-size checkpoint failed')
        remaining = c['preflight']['max_seconds'] - (time.monotonic() - started)
        require(remaining > 300, 'Actual-size resume has insufficient time budget')
        subprocess.run(runner_command(source, root, c, dataset, task, remaining, checkpoint_root), cwd=source, check=True)
        meta = read(Path(root) / 'results' / dataset / task['task_id'] / 'metadata.json')
        validate_result(meta, task, c['instances'][dataset], c, source)
        require(meta['max_memory_reserved_fraction'] <= c['scheduler']['maximum_gpu_reserved_fraction'], 'Insufficient A100 memory margin')
        records.append({'method': method, 'final_loss': meta['final_loss'], 'max_memory_reserved_fraction': meta['max_memory_reserved_fraction']})
    write(Path(root) / 'checks' / dataset / 'gate.json', dict(status='passed', **expected, time=now(), runs=records))


def production(source, root, config, dataset):
    c = load_config(config)
    root = Path(root)
    expected = identity(source, config)
    require_gate(root / 'checks' / dataset / 'gate.json', expected)
    checkpoint_root = check_storage(root, c)
    dataset_root = root / 'results' / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (dataset_root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (dataset_root / 'failed.json').exists(), 'Dataset stream previously failed; manual review required')
        if (dataset_root / 'completed.json').exists():
            return
        job = os.environ.get('SLURM_JOB_ID')
        require(job and job.isdigit(), 'Production requires a Slurm allocation')
        chunk_path = dataset_root / f'chunk_{job}.json'
        write(chunk_path, dict(status='running', job_id=job, **expected, started_utc=now()))
        started = time.monotonic()
        selected = read(dataset_root / 'selection.json') if (dataset_root / 'selection.json').exists() else None
        if selected is not None:
            for key, value in expected.items():
                require(selected[key] == value, 'Frozen selection source/config differs')
            recalculated = selection(c, dataset_root, source)
            require(all(selected[key] == value for key, value in recalculated.items()), 'Frozen selection differs from complete development results')
        complete = False
        try:
            while True:
                for task in tasks(c, selected):
                    out = dataset_root / task['task_id']
                    if (out / 'metadata.json').exists() and read(out / 'metadata.json').get('status') == 'completed':
                        meta = read(out / 'metadata.json')
                        validate_result(meta, task, c['instances'][dataset], c, source)
                        validate_retained_state(checkpoint_root, dataset, task, meta)
                        continue
                    remaining = c['runner']['max_seconds'] - (time.monotonic() - started)
                    if remaining < 600:
                        break
                    out.mkdir(parents=True, exist_ok=True)
                    write(chunk_path, dict(status='running', job_id=job, task_id=task['task_id'], **expected, updated_utc=now()))
                    command = runner_command(source, root, c, dataset, task, remaining, checkpoint_root)
                    with (out / f'job_{job}.log').open('a') as f:
                        subprocess.run(command, cwd=source, stdout=f, stderr=subprocess.STDOUT, check=True)
                    progress = read(out / 'progress.json')
                    require(progress['status'] in ('completed', 'checkpointed'), 'Runner did not finish or checkpoint')
                    if progress['status'] == 'checkpointed':
                        break
                    meta = read(out / 'metadata.json')
                    validate_result(meta, task, c['instances'][dataset], c, source)
                    validate_retained_state(checkpoint_root, dataset, task, meta)
                else:
                    if selected is None:
                        selected = selection(c, dataset_root, source)
                        write(dataset_root / 'selection.json', dict(selected, **expected, selected_utc=now()))
                        continue
                    complete = True
                break
            if complete:
                write(dataset_root / 'completed.json', dict(status='completed', **expected, time=now(), runs=55, last_job_id=job))
            write(chunk_path, dict(status='completed' if complete else 'checkpointed', job_id=job, **expected, time=now()))
        except BaseException as exc:
            write(chunk_path, dict(status='failed', error=repr(exc), traceback=traceback.format_exc(), time=now(), job_id=job))
            raise


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


def checkpoint_cursor(source, root, c, dataset, checkpoint_root):
    """Verify complete prefix and the next committed checkpoint without loading tensors."""
    dataset_root = Path(root) / 'results' / dataset
    selected = read(dataset_root / 'selection.json') if (dataset_root / 'selection.json').exists() else None
    if selected is not None:
        recalculated = selection(c, dataset_root, source)
        require(all(selected[key] == value for key, value in recalculated.items()), 'Selection no longer matches complete development data')
    for index, task in enumerate(tasks(c, selected)):
        out = dataset_root / task['task_id']
        if (out / 'metadata.json').exists() and read(out / 'metadata.json').get('status') == 'completed':
            meta = read(out / 'metadata.json')
            validate_result(meta, task, c['instances'][dataset], c, source)
            validate_retained_state(checkpoint_root, dataset, task, meta)
            continue
        require(not (out / 'failure.json').exists(), 'Explicit runner numerical/resource failure requires review')
        folder = Path(checkpoint_root) / dataset / task['task_id']
        marker = folder / 'COMMITTED.json'
        if not marker.exists():
            require(not (out / 'progress.json').exists(), 'Started task lacks a committed checkpoint')
            return {'task_index': index, 'task_id': task['task_id'], 'iteration': 0, 'has_checkpoint': False}
        checkpoint = read(marker)
        instance = c['instances'][dataset]
        identity_record = checkpoint['identity']
        require(checkpoint['world_size'] == instance['world_size'], 'Checkpoint topology differs')
        require(identity_record['world_size'] == instance['world_size'] and identity_record['n'] == identity_record['m'] == instance['n'], 'Checkpoint dimensions/topology differ')
        require(identity_record['array_files_sha256'] == instance['array_files_sha256'] and identity_record['manifest_sha256'] == instance['instance_manifest_sha256'], 'Checkpoint data identity differs')
        expected_config = dict(method=task['method'], seed=task['seed'],
            epsilon=c['epsilon'], pairs=c['pairs'], gamma0=1., rho0=1., mu_scale=10., dtype='float64', device='cuda',
            inner_max=c['inner_max'], diagnostic_every=c['runner']['diagnostic_every'], exact_block_rows=128)
        expected_config.update(task.get('prefactors', {}))
        require(identity_record['config'] == expected_config, 'Checkpoint immutable configuration differs')
        require(bool(identity_record['source_file_sha256']), 'Checkpoint source identity missing')
        for name, digest in identity_record['source_file_sha256'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Invalid checkpoint source path')
            require(sha256(Path(source) / name) == digest, 'Checkpoint source identity differs')
        generation = checkpoint['generation']
        require(Path(generation).name == generation and generation.startswith('generation-'), 'Invalid checkpoint generation')
        require(read(folder / generation / 'manifest.json') == checkpoint, 'Checkpoint commit/generation manifests differ')
        require(0 <= checkpoint['iteration'] <= task['iterations'], 'Checkpoint iteration outside prescribed horizon')
        require(len(checkpoint['shards']) == instance['world_size'], 'Missing checkpoint shards')
        for rank, shard in enumerate(checkpoint['shards']):
            require(shard['file'] == f'rank-{rank:04d}.pt', 'Checkpoint shard name differs')
            path = folder / generation / shard['file']
            require(path.stat().st_size == shard['bytes'] and sha256(path) == shard['sha256'], 'Checkpoint shard integrity failed')
        return {'task_index': index, 'task_id': task['task_id'], 'iteration': checkpoint['iteration'],
                'has_checkpoint': True, 'committed_manifest_sha256': sha256(marker)}
    return {'task_index': len(tasks(c, selected)), 'task_id': None, 'iteration': 0, 'has_checkpoint': False}


def dispatch(source, root, config, dataset, parent_job):
    """A small afterany CPU job resumes valid states after completion or interruption."""
    root, c = Path(root), load_config(config)
    dataset_root = root / 'results' / dataset
    dispatcher_job = os.environ.get('SLURM_JOB_ID')
    require(dispatcher_job and dispatcher_job.isdigit(), 'Dispatcher requires a Slurm CPU allocation')
    dataset_root.mkdir(parents=True, exist_ok=True)
    decision_path = dataset_root / f'dispatch_from_{parent_job}.json'
    with (dataset_root / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if decision_path.exists():
            return read(decision_path)
        prior = read(dataset_root / 'dispatch_state.json') if (dataset_root / 'dispatch_state.json').exists() else None
        if prior:
            require(prior['next_job_id'] == str(parent_job), 'Stale/out-of-order dispatcher cannot fork a stream')
        scheduler = slurm_terminal_record(parent_job)
        try:
            require(not (dataset_root / 'failed.json').exists(), 'Dataset already frozen after failure')
            allowed = scheduler['state'] in ('TIMEOUT', 'NODE_FAIL') or (scheduler['state'] == 'COMPLETED' and scheduler['exit_code'] == '0:0')
            require(allowed, 'Parent scheduler state requires review: ' + scheduler['state'] + ' ' + scheduler['exit_code'])
            checkpoint_root = check_storage(root, c)
            cursor = checkpoint_cursor(source, root, c, dataset, checkpoint_root)
            if (dataset_root / 'completed.json').exists():
                require(cursor['task_index'] == 55, 'Completion receipt lacks all 55 valid full-horizon runs')
                report = submit_job(source, root, 'report', cpu=True, request_key=f'{dataset}:{parent_job}:final-report')
                decision = dict(status='completed', scheduler=scheduler, cursor=cursor, report_job_id=report, time=now())
            else:
                previous_cursor = prior['cursor'] if prior else {'task_index': 0, 'iteration': 0}
                position = (cursor['task_index'], cursor['iteration'])
                previous_position = (previous_cursor['task_index'], previous_cursor['iteration'])
                require(position >= previous_position, 'Committed progress regressed')
                failures = 0 if position > previous_position else (prior['no_progress_count'] if prior else 0) + 1
                require(failures <= c['scheduler']['maximum_no_progress_retries'], 'Repeated allocations made no committed progress')
                if scheduler['state'] in ('TIMEOUT', 'NODE_FAIL'):
                    require(cursor['has_checkpoint'] or cursor['task_index'] > previous_cursor['task_index'], 'Interrupted task has no verified checkpoint/completed prefix')
                next_job = submit_job(source, root, 'production', dataset, c['instances'][dataset]['world_size'],
                    dependency='afterok:' + dispatcher_job,
                    request_key=f'{dataset}:{parent_job}:continuation')
                next_dispatch = submit_job(source, root, 'dispatch', dataset, dependency='afterany:' + next_job, cpu=True,
                    parent_job=next_job, request_key=f'{dataset}:{next_job}:dispatch')
                decision = dict(status='continued', scheduler=scheduler, cursor=cursor, no_progress_count=failures,
                                next_job_id=next_job, next_dispatch_job_id=next_dispatch, time=now())
                write(dataset_root / 'dispatch_state.json', decision)
            write(decision_path, decision)
            return decision
        except BaseException as exc:
            failure = dict(status='failed', error=repr(exc), scheduler=scheduler, traceback=traceback.format_exc(), time=now(), job_id=parent_job)
            write(dataset_root / 'failed.json', failure)
            write(decision_path, failure)
            submit_job(source, root, 'report', cpu=True, request_key=f'{dataset}:{parent_job}:failure-report')
            raise


def collect_report(source, root, config):
    """Small metadata-only summary; final confirmation tensors stay in checkpoints."""
    root, c = Path(root), load_config(config)
    dataset = 'shareseq_full'
    dataset_root = root / 'results' / dataset
    selected = read(dataset_root / 'selection.json') if (dataset_root / 'selection.json').exists() else None
    records, missing = [], []
    for task in tasks(c, selected):
        out = dataset_root / task['task_id']
        path = out / 'metadata.json'
        if not path.exists() or read(path).get('status') != 'completed':
            missing.append(task['task_id'])
            continue
        meta = read(path)
        validate_result(meta, task, c['instances'][dataset], c, source)
        if task['task_id'].startswith('confirmation/'):
            checkpoint_root = Path(read(root / 'storage_receipt.json')['checkpoint_root'])
            validate_retained_state(checkpoint_root, dataset, task, meta)
        records.append(dict(task_id=task['task_id'], method=task['method'], seed=task['seed'],
            iterations=task['iterations'], prefactors=task.get('prefactors'), final_loss=meta['final_loss'],
            initial_loss=meta['trace'][0]['rounded_loss'], counters=meta.get('counters'),
            metadata_path=str(path), metadata_sha256=sha256(path),
            retained_for_extension=task['method'] == 'frames' and task['task_id'].startswith('confirmation/')))
    failure = read(dataset_root / 'failed.json') if (dataset_root / 'failed.json').exists() else None
    report = dict(status='failed' if failure else ('completed' if len(records) == 55 else 'partial'),
        **identity(source, config), generated_utc=now(), dataset=dataset, n=31890,
        completed_runs=len(records), total_runs=55, selected=selected, runs=records,
        missing_tasks=missing, failure=failure,
        scientific_scope='Unchanged full/pair POT controls and pair-sampled FRAMES tuning; retain final FRAMES states for future anytime continuation.')
    write(root / 'reports/report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['submit', 'cpu-gate', 'tiny-gate', 'dataset-gate', 'production', 'dispatch', 'report'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--world', type=int, choices=[1, 4], default=1)
    p.add_argument('--parent-job')
    args = p.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    config = source / EXT / 'campaign.json'
    check_source(source)
    if args.phase == 'submit':
        submit_initial(source, root, config)
    elif args.phase == 'cpu-gate':
        cpu_gate(source, root, config)
    elif args.phase == 'tiny-gate':
        tiny_gate(source, root, config, args.world)
    elif args.phase == 'dataset-gate':
        require(args.dataset, 'Dataset required')
        dataset_gate(source, root, config, args.dataset)
    elif args.phase == 'production':
        require(args.dataset, 'Dataset required')
        production(source, root, config, args.dataset)
    elif args.phase == 'dispatch':
        require(args.dataset and args.parent_job, 'Dataset and parent job required')
        dispatch(source, root, config, args.dataset, args.parent_job)
    else:
        # Serialize metadata snapshots so an older read cannot replace a newer report.
        (root / 'reports').mkdir(exist_ok=True)
        with (root / 'reports/report_process.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            collect_report(source, root, config)


if __name__ == '__main__':
    main()
