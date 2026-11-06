"""Schedule, select and audit the independently tuned BMMC scaling study."""
import argparse
import csv
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

HERE = Path(__file__).resolve().parent
POLICY = ('Independently tune each method and size with 12 candidates, three development seeds '
          'and both starts at60s; select once before fresh final seeds200-204. Sharedq128/rho0=.3. '
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


def task_specs(config, phase, task, selection=None):
    methods, sizes = config['methods'], config['sizes']
    if phase == 'tuning':
        n = sizes[task // 12]
        candidate = 'c' + str(task % 12).zfill(2)
        cells = list(itertools.product(config['initializations'], config['development_seeds']))
    else:
        cells_all = list(itertools.product(sizes, config['initializations'], config['final_seeds']))
        n, start, seed = cells_all[task]
        cells = [(start, seed)]
    runs = []
    for cell, (start, seed) in enumerate(cells):
        order = methods if (task + cell) % 2 == 0 else list(reversed(methods))
        for method in order:
            cid = candidate if phase == 'tuning' else selection['selected'][str(n)][method]['candidate_id']
            parameters = config['candidates'][method][cid]
            budget = config['budgets'][phase]
            checkpoints = [budget] if phase == 'tuning' else [30, 100, budget]
            init_seed = 100 if start == 'product' else 900000 + seed
            run_id = '/'.join((phase, 'n'+str(n), method, cid, start, 's'+str(seed)))
            argv = ['run_benchmark.py', '--method', method, '--device', 'cuda', '--dtype', 'float32',
                '--threads', '1', '--initialization', start, '--seed', str(seed), '--init-seed', str(init_seed),
                '--budget', str(budget), '--checkpoints', ','.join(map(str, checkpoints)),
                '--oracle', 'sketch', '--batch', '128', '--rho0', '0.3', '--inner-max', '1000',
                '--gap-max-n', '512', '--oracle-selection-status', 'independently_tuned_bmmc_scaling']
            for key, value in parameters.items():
                argv += ['--'+key.replace('_', '-'), str(value)]
            runs.append(dict(run_id=run_id, phase=phase, n=n, method=method, candidate_id=cid,
                initialization=start, seed=seed, init_seed=init_seed, budget=budget,
                instance='bmmc'+str(n)+'.npz', parameters=parameters, argv=argv,
                parameter_selection_policy=POLICY))
    return runs


def choose_candidates(records, config):
    expected = {(n, m, cid, start, seed) for n in config['sizes'] for m in config['methods']
        for cid in config['candidates'][m] for start in config['initializations'] for seed in config['development_seeds']}
    keys = [(r['n'], r['method'], r['candidate_id'], r['initialization'], r['seed']) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Selection requires every prescribed development cell exactly once.')
    selected, scores = {}, []
    for n in config['sizes']:
        selected[str(n)] = {}
        for method in config['methods']:
            eligible = []
            for cid in config['candidates'][method]:
                group = [r for r in records if (r['n'], r['method'], r['candidate_id']) == (n, method, cid)]
                ok = all(r['eligible'] and r['loss'] is not None and math.isfinite(r['loss']) for r in group)
                score = statistics.median(r['loss'] for r in group) if ok else None
                scores.append(dict(n=n, method=method, candidate_id=cid, eligible=ok, score=score,
                                   valid_count=sum(r['eligible'] for r in group)))
                if ok:
                    eligible.append((score, cid))
            if not eligible:
                raise ValueError('No eligible candidate for '+str(n)+'/'+method)
            score, cid = min(eligible)
            selected[str(n)][method] = dict(candidate_id=cid, score=score,
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
        dtype='float32', device='cuda', threads=1, inner_max=1000, **run['parameters'])
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


def pilot_guard(config):
    pilot_root = Path(config['pilot_output'])
    state = json.loads((pilot_root/'pilot.json').read_text())
    if state['status'] != 'completed' or state['job_id'] != str(config['pilot_job_id']):
        raise ValueError('Pilot did not complete both sizes; campaign remains gated.')
    if state['runtime_commit'] != config['runtime_commit'] or state['extension_sha256']['pilot.json'] != config['pilot_config_sha256']:
        raise ValueError('Pilot source/configuration binding mismatch.')
    reconstruction = json.loads((pilot_root/'data/canonical_reconstruction.json').read_text())
    if not reconstruction['all_six_arrays_bitwise_equal']:
        raise ValueError('Canonical preparation reconstruction failed.')
    entries = {r['path']: r for r in json.loads((pilot_root/'SHA256SUMS.json').read_text())['files']}
    hashes = {}
    for n in config['sizes']:
        group = [r for r in state['records'] if r['n'] == n]
        if len(group) != 2 or {r['method'] for r in group} != set(config['methods']):
            raise ValueError('Pilot lacks one complete pair at '+str(n))
        for r in group:
            metadata = r['metadata']; v = metadata.get('final_float64_validation')
            if r['returncode'] != 0 or metadata['status'] not in ('completed', 'completed_with_inner_failures') or not v or v['rounded']['both_l1'] > 1.2e-5:
                raise ValueError('Pilot pair failed validation.')
            relative = 'n'+str(n)+'/'+r['method']+'/metadata.json'
            if digest(pilot_root/relative) != entries[relative]['sha256'] or digest(pilot_root/relative) != r['metadata_sha256']:
                raise ValueError('Pilot metadata binding mismatch.')
        name = 'bmmc'+str(n)+'.npz'
        hashes[name] = digest(pilot_root/'data'/name)
        if hashes[name] != entries['data/'+name]['sha256']:
            raise ValueError('Frozen larger instance changed after pilot.')
        if any(r['metadata']['instance_sha256'] != hashes[name] for r in group):
            raise ValueError('Pilot instance binding mismatch.')
    return hashes


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
    if (selection['config_sha256'] != digest(HERE/'campaign.json') or
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
    if receipt['config_sha256'] != digest(HERE/'campaign.json') or manifest['config_sha256'] != receipt['config_sha256']:
        raise ValueError('Campaign configuration changed.')
    if receipt['manifest_sha256'] != digest(folder/'manifest.json') or receipt['returncode'] != 0:
        raise ValueError('Task execution/provenance failed.')
    selection = frozen_selection(config, root) if phase == 'final' else None
    expected = task_specs(config, phase, task, selection)
    inventory = {r['path']: r for r in json.loads((Path(config['pilot_output'])/'SHA256SUMS.json').read_text())['files']}
    for run in expected:
        run['instance_sha256'] = inventory['data/'+run['instance']]['sha256']
    if manifest['runs'] != expected:
        raise ValueError('Manifest differs from prospective task specification.')
    if selection and receipt['selection_sha256'] != digest(root/'selection.json'):
        raise ValueError('Final task used a different selection file.')
    shard = json.loads((folder/'_campaign/shard_00.json').read_text())
    if shard['manifest_sha256'] != receipt['manifest_sha256'] or shard['source_commit'] != config['runtime_commit']:
        raise ValueError('Shard manifest/source identity mismatch.')
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
            final_float64_validation=v, environment=metadata['environment'],
            initialization_plan_sha256=metadata.get('initialization_plan_sha256'),
            array_sha256=metadata['array_sha256']))
    # Each consecutive pair shares size/start/seed and its actual compute environment.
    for left, right in zip(results[::2], results[1::2]):
        for key in ('n','initialization','seed','instance_sha256','array_sha256'):
            if left[key] != right[key]:
                raise ValueError('Paired run inputs differ: '+key)
        if left['eligible'] and right['eligible']:
            if left['initialization_plan_sha256'] != right['initialization_plan_sha256']:
                raise ValueError('Paired initial plans differ.')
        for key in ('hostname','platform','packages','cuda_runtime','gpu','nvidia_smi'):
            if left['environment'][key] != right['environment'][key]:
                raise ValueError('Paired environments differ: '+key)
    return results


def run_task(config, root, phase, task):
    hashes = pilot_guard(config)
    selection = frozen_selection(config, root) if phase == 'final' else None
    runs = task_specs(config, phase, task, selection)
    for run in runs:
        run['instance_sha256'] = hashes[run['instance']]
    folder = root/phase/('task_'+str(task).zfill(3))
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, runs=runs, config_sha256=digest(HERE/'campaign.json'),
                    parameter_selection_policy=POLICY)
    write(folder/'manifest.json', manifest)
    receipt = dict(status='running', job_id=os.environ['SLURM_JOB_ID'], task=task, phase=phase,
        config_sha256=manifest['config_sha256'], manifest_sha256=digest(folder/'manifest.json'),
        controller_sha256=digest(__file__), selection_sha256=digest(root/'selection.json') if selection else None,
        start_unix=time.time(), returncode=None, parameter_selection_policy=POLICY)
    write(folder/'controller.json', receipt)
    runtime = Path(config['runtime_directory'])
    command = [sys.executable, str(runtime/'run_campaign_shard.py'), '--manifest', str(folder/'manifest.json'),
        '--source-commit', config['runtime_commit'], '--data-root', config['pilot_output']+'/data',
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
    records = [r for task in range(24) for r in load_task(config, root, 'tuning', task)]
    attempt = dict(records=records, status='selecting', config_sha256=digest(HERE/'campaign.json'))
    write(root/'selection_attempt.json', attempt)
    try:
        result = choose_candidates(records, config)
    except ValueError as exc:
        attempt.update(status='failed', error=str(exc))
        write(root/'selection_attempt.json', attempt)
        raise
    result.update(config_sha256=digest(HERE/'campaign.json'), controller_sha256=digest(__file__),
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
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run study stages in scheduler allocations.')
    config = json.loads((HERE/'campaign.json').read_text())
    if args.phase == 'gate':
        hashes = pilot_guard(config)
        write(args.out/'eligibility.json', dict(status='passed', config_sha256=digest(HERE/'campaign.json'),
              controller_sha256=digest(__file__), input_sha256=hashes, pilot_job_id=config['pilot_job_id'],
              job_id=os.environ['SLURM_JOB_ID']))
    elif args.phase == 'select':
        select(config, args.out)
    else:
        limit = 24 if args.phase == 'tuning' else 20
        if not 0 <= args.task < limit:
            raise ValueError('Invalid array task.')
        run_task(config, args.out, args.phase, args.task)


if __name__ == '__main__':
    main()
