"""Schedule, select and audit the frozen, independently tuned dataset study."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
POLICY = ('Independently tune each method and instance with 12 candidates, three development seeds '
          'and both starts at 60s; select once before fresh final seeds 300-304. Shared q128/rho0=.3. '
          'This supersedes the frozen runner momentum_config.parameter_policy describing Stage C inheritance.')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def config_digest(config):
    return digest(Path(config.get('config_path', HERE/'campaign.json')))


def validate_config(config):
    instances = config['instances']
    if not instances or len({i['id'] for i in instances}) != len(instances):
        raise ValueError('Frozen instances must be nonempty and have unique IDs.')
    filenames = []
    for instance in instances:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', instance['id']):
            raise ValueError('Unsafe instance ID.')
        if not isinstance(instance['n'], int) or instance['n'] < 2:
            raise ValueError('Instance shape must be a positive square dimension.')
        for field in ('filename', 'correspondence_file'):
            if field not in instance:
                continue
            value = instance[field]
            if not isinstance(value, str) or Path(value).name != value or value in ('.', '..'):
                raise ValueError('Frozen data must use safe filenames.')
            key = 'sha256' if field == 'filename' else 'correspondence_sha256'
            if not re.fullmatch('[a-f0-9]{64}', instance.get(key, '')):
                raise ValueError('Frozen data requires SHA256 bindings.')
        if ('correspondence_file' in instance) != ('correspondence_sha256' in instance):
            raise ValueError('Correspondence filename and hash must be paired.')
        filenames.append(instance['filename'])
    if len(set(filenames)) != len(filenames):
        raise ValueError('Each instance requires a distinct frozen filename.')
    if set(config['methods']) != {'smoothing-fw', 'stochastic-kl-momentum'} or len(config['methods']) != 2:
        raise ValueError('This campaign requires the prescribed stochastic pair.')
    prescribed_ids = {'c'+str(i).zfill(2) for i in range(12)}
    if any(set(config['candidates'][method]) != prescribed_ids for method in config['methods']):
        raise ValueError('Each method requires exactly the twelve prescribed candidates.')
    if config['initializations'] != ['product', 'randomized']:
        raise ValueError('Both prescribed starts are required.')
    if config['development_seeds'] != [10, 11, 12] or config['final_seeds'] != list(range(300, 305)):
        raise ValueError('Development and final seed populations must remain frozen.')
    if config['budgets'] != {'tuning': 60, 'final': 300}:
        raise ValueError('The prescribed solver-time budgets must remain frozen.')


def task_count(config, phase):
    if phase not in ('tuning', 'final'):
        raise ValueError('Task phase must be tuning or final.')
    return len(config['instances']) * (12 if phase == 'tuning' else 10)


def build_run(config, phase, instance, method, cid, parameters, start, seed, budget, checkpoints):
    init_seed = 100 if start == 'product' else 900000 + seed
    run_id = '/'.join((phase, instance['id'], method, cid, start, 's'+str(seed)))
    argv = [config.get('runner_filename', 'run_benchmark.py'), '--method', method, '--device', 'cuda', '--dtype', 'float32',
        '--threads', '1', '--initialization', start, '--seed', str(seed), '--init-seed', str(init_seed),
        '--budget', str(budget), '--checkpoints', ','.join(map(str, checkpoints)),
        '--oracle', 'sketch', '--batch', '128', '--rho0', '0.3', '--inner-max', '1000',
        '--gap-max-n', '512', '--oracle-selection-status', 'independently_tuned_cross_dataset']
    for key, value in parameters.items():
        argv += ['--'+key.replace('_', '-'), str(value)]
    run = dict(run_id=run_id, phase=phase, instance_id=instance['id'], n=instance['n'], method=method,
        candidate_id=cid, initialization=start, seed=seed, init_seed=init_seed, budget=budget,
        instance=instance['filename'], instance_sha256=instance['sha256'], parameters=parameters,
        argv=argv, parameter_selection_policy=POLICY)
    if 'correspondence_file' in instance:
        run.update(correspondence_file=instance['correspondence_file'],
                   correspondence_sha256=instance['correspondence_sha256'])
        argv += ['--correspondence-file', str(Path(config['data_root'])/instance['correspondence_file']),
                 '--correspondence-sha256', instance['correspondence_sha256']]
    return run


def pilot_specs(config):
    validate_config(config)
    parameters = {'smoothing-fw': {'gamma0': 1.0, 'mu_scale': 10.0},
                  'stochastic-kl-momentum': {'eta': .1, 'eta_power': 0.0, 'inner_tol': 1e-4}}
    return [build_run(config, 'pilot', instance, method, 'pilot', parameters[method],
                      'product', 250, 30, [10, 30])
            for instance in config['instances'] for method in config['methods']]


def task_specs(config, phase, task, selection=None):
    validate_config(config)
    if not 0 <= task < task_count(config, phase):
        raise ValueError('Invalid array task.')
    methods, instances = config['methods'], config['instances']
    if phase == 'tuning':
        instance = instances[task // 12]
        candidate = 'c' + str(task % 12).zfill(2)
        cells = list(itertools.product(config['initializations'], config['development_seeds']))
    else:
        cells_all = list(itertools.product(instances, config['initializations'], config['final_seeds']))
        instance, start, seed = cells_all[task]
        cells = [(start, seed)]
    runs = []
    for cell, (start, seed) in enumerate(cells):
        order = methods if (task + cell) % 2 == 0 else list(reversed(methods))
        for method in order:
            cid = candidate if phase == 'tuning' else selection['selected'][instance['id']][method]['candidate_id']
            parameters = config['candidates'][method][cid]
            budget = config['budgets'][phase]
            checkpoints = [budget] if phase == 'tuning' else [10, 30, 100, budget]
            runs.append(build_run(config, phase, instance, method, cid, parameters, start, seed, budget, checkpoints))
    return runs


def choose_candidates(records, config):
    validate_config(config)
    instances = {i['id']: i for i in config['instances']}
    expected = {(iid, m, cid, start, seed) for iid in instances for m in config['methods']
        for cid in config['candidates'][m] for start in config['initializations'] for seed in config['development_seeds']}
    keys = [(r['instance_id'], r['method'], r['candidate_id'], r['initialization'], r['seed']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Selection requires every prescribed development cell exactly once.')
    selected, scores = {}, []
    for iid, instance in instances.items():
        selected[iid] = {}
        for method in config['methods']:
            eligible = []
            for cid in config['candidates'][method]:
                group = [r for r in records if (r['instance_id'], r['method'], r['candidate_id']) == (iid, method, cid)]
                if any(r['n'] != instance['n'] for r in group):
                    raise ValueError('Development record shape differs from frozen instance.')
                ok = all(r['eligible'] and r['loss'] is not None and math.isfinite(r['loss']) for r in group)
                score = statistics.median(r['loss'] for r in group) if ok else None
                scores.append(dict(instance_id=iid, n=instance['n'], method=method, candidate_id=cid,
                                   eligible=ok, score=score, valid_count=sum(r['eligible'] for r in group)))
                if ok:
                    eligible.append((score, cid))
            if not eligible:
                raise ValueError('No eligible candidate for '+iid+'/'+method)
            score, cid = min(eligible)
            selected[iid][method] = dict(candidate_id=cid, score=score,
                                         parameters=config['candidates'][method][cid])
    return dict(selected=selected, scores=scores, rule=config['selection_rule'], policy=POLICY)


def validate_metadata(metadata, run, config):
    if metadata.get('source_commit') != config['runtime_commit'] or metadata.get('source_dirty'):
        raise ValueError('Runner source provenance mismatch.')
    if metadata.get('source_file_sha256') != config['runtime_file_sha256']:
        raise ValueError('Runner file hashes differ from frozen runtime.')
    if metadata.get('instance_sha256') != run['instance_sha256']:
        raise ValueError('Runner input hash mismatch.')
    expected = dict(method=run['method'], initialization=run['initialization'], seed=run['seed'],
        init_seed=run['init_seed'], budget=run['budget'], batch=128, rho0=.3, oracle='sketch',
        dtype='float32', device='cuda', threads=1, inner_max=1000,
        checkpoints=run['argv'][run['argv'].index('--checkpoints')+1], **run['parameters'])
    if 'correspondence_file' in run:
        expected.update(correspondence_file=str(Path(config['data_root'])/run['correspondence_file']),
                        correspondence_sha256=run['correspondence_sha256'])
        if metadata.get('correspondence_sha256') != run['correspondence_sha256']:
            raise ValueError('Correspondence input hash mismatch.')
    for key, value in {'alpha': .75, 'beta': .5, 'delta': .25}.items():
        if metadata.get('theory_config', {}).get(key) != value:
            raise ValueError('Theory schedule mismatch: '+key)
    for key, value in expected.items():
        if metadata['config'].get(key) != value:
            raise ValueError('Runner parameter mismatch: '+key)
    if metadata['status'] not in ('completed', 'completed_with_inner_failures', 'failed'):
        raise ValueError('Run is not terminal.')
    if metadata['status'] == 'failed':
        return False
    validation = metadata.get('final_float64_validation')
    return bool(validation and math.isfinite(validation['raw_loss']) and math.isfinite(validation['rounded_loss'])
        and validation['rounded']['both_l1'] <= 1.2e-5 and metadata['counters']['completed_updates'] > 0
        and metadata['timing']['solver_seconds'] >= run['budget'])


def safe_pilot_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in ('.', '..') for part in path.parts):
        raise ValueError('Unsafe pilot artifact path.')
    resolved = (root/path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Pilot artifact escapes output root.')
    return resolved


def audit_pair(left, right):
    for key in ('instance_id', 'n', 'initialization', 'seed', 'init_seed', 'instance_sha256', 'array_sha256'):
        if left[key] != right[key]:
            raise ValueError('Paired run inputs differ: '+key)
    for key in ('correspondence_file', 'correspondence_sha256'):
        if left.get(key) != right.get(key):
            raise ValueError('Paired correspondence inputs differ: '+key)
    if left['eligible'] and right['eligible']:
        if not left.get('initialization_plan_sha256') or left['initialization_plan_sha256'] != right.get('initialization_plan_sha256'):
            raise ValueError('Paired initial plans differ or lack a hash.')
    for key in ('hostname', 'platform', 'packages', 'cuda_runtime', 'gpu', 'nvidia_smi'):
        if left['environment'][key] != right['environment'][key]:
            raise ValueError('Paired environments differ: '+key)


def input_hashes(config):
    validate_config(config)
    hashes = {}
    for instance in config['instances']:
        for key, hash_key in (('filename', 'sha256'), ('correspondence_file', 'correspondence_sha256')):
            if key not in instance:
                continue
            filename = instance[key]
            hashes[filename] = digest(Path(config['data_root'])/filename)
            if hashes[filename] != instance[hash_key]:
                raise ValueError('Frozen input changed after preparation: '+filename)
    return hashes


def validate_loss_targets(config):
    fractions = [.99, .97, .95, .90]
    if config.get('loss_target_fractions') != fractions:
        raise ValueError('The common-loss target fraction policy must remain frozen.')
    root = Path(config['pilot_output'])
    state = json.loads((root/'pilot.json').read_text())
    if (state['config_sha256'] != config_digest(config) or state['controller_sha256'] != digest(__file__)
            or state['status'] != 'completed'):
        raise ValueError('Loss targets lack a completed, bound pilot.')
    if [record['run_spec'] for record in state['records']] != pilot_specs(config):
        raise ValueError('Loss targets lack the complete prescribed pilot pairs.')
    initial = {}
    for record in state['records']:
        trace_path = safe_pilot_path(root, record['trace_path'])
        if digest(trace_path) != record['trace_sha256']:
            raise ValueError('Pilot trace binding mismatch.')
        with trace_path.open(newline='') as stream:
            rows = [row for row in csv.DictReader(stream) if row.get('kind') == 'initial']
        if len(rows) != 1:
            raise ValueError('Pilot must record exactly one initial loss.')
        loss = float(rows[0]['rounded_loss'])
        if not math.isfinite(loss) or loss < 0 or loss != record['initial_rounded_loss']:
            raise ValueError('Pilot initial loss disagrees with the recorded trace.')
        run = record['run_spec']
        initial[(run['instance_id'], run['method'])] = loss
    expected = {}
    for instance in config['instances']:
        iid = instance['id']
        baseline = initial[(iid, config['methods'][0])]
        if abs(baseline-initial[(iid, config['methods'][1])]) > 1e-6:
            raise ValueError('Pilot paired initial losses differ.')
        expected[iid] = dict(baseline_loss=baseline, fractions=fractions,
                             losses=[baseline*fraction for fraction in fractions])
    if state.get('common_loss_targets') != expected:
        raise ValueError('Common loss targets differ from the frozen initial-loss rule.')
    return expected


def pilot_guard(config, validate_plans=True):
    pilot_root = Path(config['pilot_output'])
    state = json.loads((pilot_root/'pilot.json').read_text())
    if state['status'] != 'completed' or state.get('resource_gate_passed') is not True:
        raise ValueError('Pilot did not complete every prescribed instance; campaign remains gated.')
    if (state['runtime_commit'] != config['runtime_commit'] or state['config_sha256'] != config_digest(config)
            or state['controller_sha256'] != digest(__file__)):
        raise ValueError('Pilot source/configuration binding mismatch.')
    if state.get('pilot_runner_sha256') != config['pilot_runner_sha256']:
        raise ValueError('Pilot launcher source binding mismatch.')
    if 'pilot_job_id' in config and str(state['job_id']) != str(config['pilot_job_id']):
        raise ValueError('Pilot scheduler job identity mismatch.')
    prescribed = pilot_specs(config)
    if [r['run_spec'] for r in state['records']] != prescribed:
        raise ValueError('Pilot lacks the complete prescribed instance pairs.')
    hashes = input_hashes(config)
    validate_loss_targets(config)
    results = []
    for record, run in zip(state['records'], prescribed):
        metadata_path = safe_pilot_path(pilot_root, record['metadata_path'])
        final_path = safe_pilot_path(pilot_root, record['final_path'])
        if digest(metadata_path) != record['metadata_sha256']:
            raise ValueError('Pilot metadata binding mismatch.')
        metadata = json.loads(metadata_path.read_text())
        if record['returncode'] != 0 or not validate_metadata(metadata, run, config):
            raise ValueError('Pilot pair failed validation.')
        if record.get('resource_gate_passed') is not True:
            raise ValueError('Pilot record lacks the passed resource gate.')
        gpu, memory = metadata['environment']['gpu'], metadata['memory']
        total, reserved, rss = gpu['total_memory_bytes'], memory['cuda_peak_reserved_bytes'], memory['cpu_process_peak_rss_kib']
        if ('A100' not in gpu['name'] or not all(math.isfinite(value) for value in (total, reserved, rss))
                or total <= 0 or not 0 <= reserved < .8*total or not 0 <= rss*1024 < .8*32*1024**3):
            raise ValueError('Pilot recorded hardware or memory fails the resource gate.')
        if record.get('memory') != memory:
            raise ValueError('Pilot resource summary disagrees with bound metadata.')
        if validate_plans:
            saved = validate_saved_output(final_path, run['n'], metadata)
            if saved['sha256'] != record['final_sha256']:
                raise ValueError('Pilot saved output binding mismatch.')
        results.append(dict(**run, eligible=True, array_sha256=metadata['array_sha256'],
                            initialization_plan_sha256=metadata.get('initialization_plan_sha256'),
                            environment=metadata['environment']))
    for left, right in zip(results[::2], results[1::2]):
        audit_pair(left, right)
    return hashes


def eligibility_guard(config, root):
    eligibility = json.loads((root/'eligibility.json').read_text())
    if (eligibility['status'] != 'passed' or eligibility['config_sha256'] != config_digest(config)
            or eligibility['controller_sha256'] != digest(__file__)
            or eligibility['pilot_sha256'] != digest(Path(config['pilot_output'])/'pilot.json')):
        raise ValueError('Validated pilot eligibility binding mismatch.')
    hashes = pilot_guard(config, validate_plans=False)
    if hashes != eligibility['input_sha256']:
        raise ValueError('Input inventory changed after the eligibility gate.')
    return eligibility


def validate_saved_output(path, n, metadata):
    import numpy as np
    audit = dict(sha256=digest(path), arrays={})
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'P_raw', 'P_rounded'}:
            raise ValueError('Saved output lacks the prescribed raw/rounded plans.')
        for name in ('P_raw', 'P_rounded'):
            P = arrays[name]
            if P.shape != (n, n) or P.dtype != np.float32 or not np.isfinite(P).all() or (P < 0).any():
                raise ValueError('Saved plan has wrong shape/dtype or invalid entries.')
            row_l1 = float(np.abs(P.sum(axis=1, dtype=np.float64) - 1/n).sum())
            col_l1 = float(np.abs(P.sum(axis=0, dtype=np.float64) - 1/n).sum())
            reference = metadata['final_float64_validation']['rounded' if name == 'P_rounded' else 'raw']
            if not all(math.isfinite(reference[key]) for key in ('row_l1','col_l1')):
                raise ValueError('Runner residual metadata is nonfinite.')
            if abs(row_l1-reference['row_l1']) > 1e-9 or abs(col_l1-reference['col_l1']) > 1e-9:
                raise ValueError('Saved marginal residuals disagree with runner validation.')
            if name == 'P_rounded' and row_l1+col_l1 > 1.2e-5:
                raise ValueError('Saved rounded output is infeasible.')
            audit['arrays'][name] = dict(shape=list(P.shape), dtype=str(P.dtype),
                                        row_l1=row_l1, col_l1=col_l1)
    return audit


def frozen_selection(config, root):
    selection = json.loads((root/'selection.json').read_text())
    if (selection['config_sha256'] != config_digest(config) or
            selection['controller_sha256'] != digest(__file__)):
        raise ValueError('Selection configuration/controller identity mismatch.')
    recomputed = choose_candidates(selection['records'], config)
    if selection['selected'] != recomputed['selected'] or selection['scores'] != recomputed['scores']:
        raise ValueError('Frozen selection disagrees with its development records.')
    return selection


def load_task(config, root, phase, task):
    folder = root/phase/('task_'+str(task).zfill(3))
    manifest = json.loads((folder/'manifest.json').read_text())
    receipt = json.loads((folder/'controller.json').read_text())
    if receipt['controller_sha256'] != digest(__file__):
        raise ValueError('Controller source changed.')
    if receipt['config_sha256'] != config_digest(config) or manifest['config_sha256'] != receipt['config_sha256']:
        raise ValueError('Campaign configuration changed.')
    if receipt['manifest_sha256'] != digest(folder/'manifest.json') or receipt['returncode'] != 0:
        raise ValueError('Task execution/provenance failed.')
    selection = frozen_selection(config, root) if phase == 'final' else None
    expected = task_specs(config, phase, task, selection)
    if manifest['runs'] != expected:
        raise ValueError('Manifest differs from prospective task specification.')
    if selection and receipt['selection_sha256'] != digest(root/'selection.json'):
        raise ValueError('Final task used a different selection file.')
    shard = json.loads((folder/'_campaign/shard_00.json').read_text())
    if shard['manifest_sha256'] != receipt['manifest_sha256'] or shard['source_commit'] != config['runtime_commit']:
        raise ValueError('Shard manifest/source identity mismatch.')
    if config.get('launcher_sha256') and shard['source_sha256'] != config['launcher_sha256']:
        raise ValueError('Shard launcher source changed.')
    if shard['input_sha256'] != {run['instance']: run['instance_sha256'] for run in expected}:
        raise ValueError('Shard input hashes differ from the frozen task.')
    by_id = {r['run_id']: r for r in shard['records']}
    if len(by_id) != len(shard['records']) or set(by_id) != {r['run_id'] for r in expected}:
        raise ValueError('Shard receipts have missing or duplicate run IDs.')
    if shard['status'] not in ('completed','completed_with_failed_runs') or len(shard['records']) != len(manifest['runs']):
        raise ValueError('Task has missing/infrastructure-failed attempts.')
    results = []
    for run in manifest['runs']:
        path = folder/run['run_id']/'metadata.json'
        metadata = json.loads(path.read_text())
        run_receipt = json.loads((path.parent/'campaign_receipt.json').read_text())
        if run_receipt != by_id[run['run_id']] or run_receipt['run_spec'] != run or run_receipt['manifest_sha256'] != receipt['manifest_sha256']:
            raise ValueError('Run receipt is not bound to the expected task.')
        if run_receipt['infrastructure_error'] is not None or run_receipt['status'] != metadata['status'] or run_receipt['returncode'] != (1 if metadata['status'] == 'failed' else 0):
            raise ValueError('Run receipt status disagrees with terminal output.')
        eligible = validate_metadata(metadata, run, config)
        if eligible and not (path.parent/'final.npz').is_file():
            raise ValueError('Saved final output missing.')
        saved = validate_saved_output(path.parent/'final.npz', run['n'], metadata) if eligible else None
        v = metadata.get('final_float64_validation')
        results.append(dict(**run, eligible=eligible, loss=v['rounded_loss'] if eligible else None,
            metadata=str(path), metadata_sha256=digest(path), saved_output_audit=saved, status=metadata['status'],
            instrumented_process_seconds=run_receipt['instrumented_process_seconds'],
            counters=metadata['counters'], timing=metadata['timing'], memory=metadata['memory'],
            final_float64_validation=v, correspondence=metadata.get('correspondence'),
            environment=metadata['environment'],
            initialization_plan_sha256=metadata.get('initialization_plan_sha256'),
            array_sha256=metadata['array_sha256']))
    # Consecutive methods share the instance, start, seed and actual compute environment.
    for left, right in zip(results[::2], results[1::2]):
        audit_pair(left, right)
    return results


def run_task(config, root, phase, task):
    eligibility_guard(config, root)
    selection = frozen_selection(config, root) if phase == 'final' else None
    runs = task_specs(config, phase, task, selection)
    folder = root/phase/('task_'+str(task).zfill(3))
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, runs=runs, config_sha256=config_digest(config),
                    parameter_selection_policy=POLICY)
    write(folder/'manifest.json', manifest)
    receipt = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], task=task, phase=phase,
        config_sha256=manifest['config_sha256'], manifest_sha256=digest(folder/'manifest.json'),
        controller_sha256=digest(__file__), selection_sha256=digest(root/'selection.json') if selection else None,
        start_unix=time.time(), returncode=None, parameter_selection_policy=POLICY)
    write(folder/'controller.json', receipt)
    runtime = Path(config['runtime_directory'])
    command = [sys.executable, str(runtime/'run_campaign_shard.py'), '--manifest', str(folder/'manifest.json'),
        '--source-commit', config['runtime_commit'], '--data-root', config['data_root'],
        '--out-root', str(folder), '--task-index', '0', '--task-count', '1']
    with (folder/'launcher.log').open('x') as log:
        code = subprocess.run(command, cwd=runtime, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    receipt.update(returncode=code, end_unix=time.time(), status='completed' if code == 0 else 'failed')
    write(folder/'controller.json', receipt)
    if code:
        raise RuntimeError('Campaign shard failed; inspect '+str(folder/'launcher.log'))
    results = load_task(config, root, phase, task)
    write(folder/'validation.json', dict(runs=len(results), valid=sum(r['eligible'] for r in results),
                                       records=results, paired_provenance_valid=True))


def select(config, root):
    records = [r for task in range(task_count(config, 'tuning')) for r in load_task(config, root, 'tuning', task)]
    attempt = dict(records=records, status='selecting', config_sha256=config_digest(config))
    write(root/'selection_attempt.json', attempt)
    try:
        result = choose_candidates(records, config)
    except ValueError as exc:
        attempt.update(status='failed', error=str(exc))
        write(root/'selection_attempt.json', attempt)
        raise
    result.update(config_sha256=config_digest(config), controller_sha256=digest(__file__),
                  records=records, job_id=os.environ['SLURM_JOB_ID'], created_unix=time.time())
    if (root/'selection.json').exists():
        raise FileExistsError('Frozen selection already exists.')
    write(root/'selection.json', result)
    attempt.update(status='completed', selection_sha256=digest(root/'selection.json'))
    write(root/'selection_attempt.json', attempt)
    print(json.dumps(dict(selected=result['selected'], tuning_count=len(records))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('gate','tuning','select','final'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--config', type=Path, default=HERE/'campaign.json')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run study stages in scheduler allocations.')
    config = json.loads(args.config.read_text())
    if Path(config.get('config_path', HERE/'campaign.json')).resolve() != args.config.resolve():
        raise ValueError('Frozen config_path must identify the supplied configuration.')
    validate_config(config)
    if args.phase == 'gate':
        hashes = pilot_guard(config)
        write(args.out/'eligibility.json', dict(status='passed', config_sha256=config_digest(config),
              controller_sha256=digest(__file__), input_sha256=hashes,
              pilot_sha256=digest(Path(config['pilot_output'])/'pilot.json'),
              job_id=os.environ['SLURM_JOB_ID']))
    elif args.phase == 'select':
        select(config, args.out)
    else:
        limit = task_count(config, args.phase)
        if not 0 <= args.task < limit:
            raise ValueError('Invalid array task.')
        run_task(config, args.out, args.phase, args.task)


if __name__ == '__main__':
    main()
