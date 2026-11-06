"""Frozen fixed-horizon campaign controller; GPU work runs only through Slurm.

Every task/run directory is created exclusively. A failed or partial task is
preserved and cannot silently turn into a resumed or retuned experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

METHODS = ('smoothing-fw', 'published-smd')
ARRAYS = ('A.npy', 'B.npy', 'a.npy', 'b.npy')
PLANS = ('selected_raw', 'selected_rounded', 'last_raw', 'last_rounded')
SOURCE_FILES = (
    'published_stochastic_20261002/run_published.py',
    'published_stochastic_20261002/smd.py',
    'published_stochastic_20261002/certified_prox.py',
    'published_stochastic_20261002/inner_solver.json',
    'shareseq_25000_20260930/run_sharded.py',
    'shareseq_25000_20260930/sharded_gw.py',
)
RUNNER_KEYS = {'dtype', 'batch', 'inner_tol', 'inner_gap_tol', 'inner_max',
               'check_every', 'certificate_block_rows', 'exact_block_rows', 'progress_every'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, nonnegative=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'{label} must be a finite number.')
    require(not nonnegative or value >= 0, f'{label} must be nonnegative.')
    return float(value)


def validate_config(config):
    require(config['schema_version'] == 1, 'Unsupported campaign schema.')
    require(set(config['methods']) == set(METHODS), 'Exactly the two frozen stochastic methods are required.')
    development, final = config['development_seeds'], config['final_seeds']
    require(len(development) == len(final) == 3, 'Exactly three development and final seeds are required.')
    require(len(set(development)) == len(set(final)) == 3 and not set(development) & set(final),
            'Development and final seeds must be unique and disjoint.')
    require(all(isinstance(s, int) and not isinstance(s, bool) and 0 <= s < 2**63 - 20000
                for s in development + final), 'Invalid seed.')
    require(config['initializations'] == ['product', 'randomized'], 'Both prescribed starts are required.')
    require(set(config['runner']) == RUNNER_KEYS, 'Runner arguments must be specified explicitly and completely.')
    runner = config['runner']
    require(runner['batch'] == 128 and runner['dtype'] == 'float32', 'Production uses float32 plans and q128.')
    require(runner['inner_tol'] == runner['inner_gap_tol'] == 1e-7, 'Both selected inner tolerances must be 1e-7.')
    for key in RUNNER_KEYS - {'dtype', 'inner_tol', 'inner_gap_tol'}:
        require(isinstance(runner[key], int) and runner[key] > 0, f'Invalid runner parameter {key}.')
    for stage in ('tuning', 'final'):
        horizon = config[f'{stage}_iterations']
        require(isinstance(horizon, int) and horizon > 0, 'Fixed horizons must be positive integers.')
        checkpoints = config[f'{stage}_checkpoint_iterations']
        require(checkpoints == sorted(set(checkpoints))
                and all(isinstance(k, int) and 0 < k <= horizon for k in checkpoints), 'Invalid fixed checkpoints.')
    require(config['final_iterations'] >= config['tuning_iterations'], 'Final horizon must not be shorter than tuning.')
    for value in config['checkpoint_seconds']:
        require(finite(value, 'Time checkpoint') > 0, 'Time checkpoints must be positive.')
    require(finite(config['run_timeout_seconds'], 'Run timeout') > 0, 'Run timeout must be positive.')
    require(finite(config['final_marginal_l1_tolerance'], 'Final feasibility tolerance') > 0,
            'Final feasibility tolerance must be positive.')
    for name, instance in config['instances'].items():
        require(name and '/' not in name and name not in ('.', '..'), 'Invalid instance identifier.')
        require(isinstance(instance['n'], int) and instance['n'] > 0, 'Invalid instance size.')
        require(instance['world_size'] in (1, 4), 'Supported production allocations are one or four GPUs.')
    ids = []
    for method, candidates in config['methods'].items():
        require(len(candidates) == 12, 'Each method must have exactly twelve predeclared candidates.')
        expected = {'id', 'gamma0', 'rho0', 'mu_scale'} if method == 'smoothing-fw' else {'id', 'prefactor_fraction'}
        for candidate in candidates:
            require(set(candidate) == expected, f'Unexpected {method} candidate fields.')
            identifier = candidate['id']
            require(identifier and '/' not in identifier and identifier not in ('.', '..'), 'Invalid candidate ID.')
            ids.append(identifier)
            for key, value in candidate.items():
                if key != 'id':
                    require(finite(value, key) > 0 and (key == 'mu_scale' or value <= 1), 'Invalid prefactor.')
    require(len(set(ids)) == len(ids), 'Candidate identifiers must be globally unique.')
    return config


def identity(config_path, source, prepared, instance):
    source, data = Path(source), Path(prepared) / instance
    manifest = read_json(data / 'manifest.json')
    hashes = manifest.get('array_files_sha256', {})
    require(set(hashes) == set(ARRAYS), 'Prepared manifest must identify all four canonical arrays.')
    require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for v in hashes.values()), 'Malformed input SHA256.')
    return dict(config_sha256=sha256(config_path),
                source_file_sha256={name: sha256(source / 'extensions' / name) for name in SOURCE_FILES},
                controller_source_sha256=sha256(source / 'extensions/published_stochastic_20261002/campaign.py'),
                instance_manifest_sha256=sha256(data / 'manifest.json'), array_files_sha256=hashes)


def run_directory(out, instance, stage, candidate, initialization, seed):
    return Path(out) / instance / stage / candidate['id'] / initialization / f'seed{seed}'


def runner_config(config, prepared, out, instance, stage, method, candidate, initialization, seed):
    values = dict(instance=str((Path(prepared) / instance).resolve()), out=str(Path(out).resolve()),
                  method=method, device='cuda', iterations=config[f'{stage}_iterations'],
                  seed=seed, output_seed=seed + 20000, init_seed=seed + 10000,
                  initialization=initialization, prefactor_fraction=1., rho0=.3, gamma0=.1, mu_scale=.1,
                  save_plans=('all' if stage == 'final' and (instance != 'shareseq_full' or seed == config['final_seeds'][0]) else 'none'),
                  checkpoint_iterations=','.join(map(str, config[f'{stage}_checkpoint_iterations'])),
                  checkpoints=','.join(map(str, config['checkpoint_seconds'])))
    values.update(config['runner'])
    values.update({key: value for key, value in candidate.items() if key != 'id'})
    return values


def runner_command(source, world_size, arguments):
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc-per-node={world_size}', str(Path(source) / 'extensions/published_stochastic_20261002/run_published.py')]
    for key, value in arguments.items():
        command += ['--' + key.replace('_', '-'), str(value)]
    return command


def validate_metadata(metadata, expected, provenance, instance_config, feasibility_tolerance):
    require(metadata.get('status') == 'completed', 'Runner did not complete.')
    require(metadata.get('config') == expected, 'Runner configuration differs from the frozen request.')
    require(metadata.get('method_key') == expected['method'], 'Method identity differs.')
    for key in ('source_file_sha256', 'instance_manifest_sha256', 'array_files_sha256'):
        require(metadata.get(key) == provenance[key], f'Provenance mismatch: {key}.')
    n, world, horizon = instance_config['n'], instance_config['world_size'], expected['iterations']
    require(metadata.get('dimensions') == {'source': n, 'target': n}, 'Unexpected problem dimensions.')
    require(metadata['environment']['world_size'] == world, 'Unexpected distributed world size.')
    ranks = metadata['ranks']
    require(len(ranks) == world and sorted(rank['rank'] for rank in ranks) == list(range(world)), 'Incomplete rank records.')
    counters = metadata['counters']
    require(counters['completed_updates'] == counters['stochastic_gradient_calls'] == horizon,
            'Fixed horizon or one-gradient-per-update contract violated.')
    require(counters['sketch_probes'] == 128 * horizon and counters['inner_stopping_failures'] == 0,
            'Oracle budget or inner solve failure detected.')
    require(counters['initialization_oracle_calls'] == int(expected['method'] == 'smoothing-fw'),
            'Unexpected initialization oracle count.')
    validation = metadata['final_float64_validation']
    require(validation['dtype'] == 'float64' and validation['exact_objective'] is True,
            'Independent final FP64 objective validation missing.')
    require(set(validation['plans']) == set(PLANS), 'Missing final output validation.')
    for name, record in validation['plans'].items():
        finite(record['loss'], f'{name} loss')
        residual = finite(record['marginal_residuals']['both_l1'], f'{name} residual', nonnegative=True)
        if name.endswith('_rounded'):
            require(residual <= feasibility_tolerance, f'{name} is not feasible to the frozen tolerance.')
    for rank in ranks:
        require(set(rank['saved_plans']) == set(PLANS), 'Incomplete output hashes.')
        # The historical row partition distributes the remainder to the earliest ranks.
        rows = n // world + int(rank['rank'] < n % world)
        for plan in rank['saved_plans'].values():
            require(plan['shape'] == [rows, n] and plan['dtype'] == expected['dtype'], 'Unexpected output shard shape/dtype.')
            require(isinstance(plan['array_sha256'], str) and len(plan['array_sha256']) == 64,
                    'Missing output array hash.')
            require(plan['persisted'] is (expected['save_plans'] == 'all'), 'Incorrect plan storage policy.')
            require(plan['array_bytes'] == rows * n * 4, 'Incorrect output array byte count.')
            if plan['persisted']:
                require(Path(plan['file']).is_file() and Path(plan['file']).stat().st_size == plan['bytes']
                        and len(plan['file_sha256']) == 64, 'Persisted plan file is missing or truncated.')
            else:
                require(plan['file'] is None and plan['file_sha256'] is None and plan['bytes'] == 0,
                        'Metadata-only output unexpectedly names a persisted plan.')
    selection = metadata['output_selection']
    require(selection['horizon'] == horizon and 0 <= selection['index'] < horizon
            and selection['last_iterate_index'] == horizon and selection['independent_rng'] is True
            and selection['output_seed'] == expected['output_seed'], 'Invalid independently sampled preupdate output.')
    for key in ('preparation_seconds', 'solver_seconds', 'repair_seconds', 'exact_diagnostic_seconds',
                'serialization_seconds', 'final_float64_validation_seconds', 'total_elapsed_seconds'):
        finite(metadata['timing'][key], f'Timing {key}', nonnegative=True)
    trace = metadata['inner_trace']
    if expected['method'] == 'published-smd':
        require(len(trace) == horizon, 'Missing per-update proximal certificates.')
        for certificate in trace:
            require(certificate['converged'] is True and certificate['proximal_optimality_certified'] is True,
                    'Uncertified proximal update.')
            for key, tolerance in (('marginal_l1', expected['inner_tol']),
                                   ('normalized_gap_upper_bound', expected['inner_gap_tol'])):
                value = finite(certificate[key], key, nonnegative=True)
                require(value <= tolerance, f'Inner certificate exceeds {key} tolerance.')
            require(finite(certificate['normalized_primal_dual_gap'], 'Raw normalized gap') <= expected['inner_gap_tol'],
                    'Raw normalized gap exceeds tolerance.')
        require(counters['inner_iterations_total'] == sum(c['inner_iterations'] for c in trace)
                and counters['certificate_checks'] == sum(c['certificate_checks'] for c in trace),
                'Inner work counters differ from certificates.')
    else:
        require(not trace and counters['inner_iterations_total'] == counters['certificate_checks'] == 0,
                'FW must not perform proximal work.')
    return validation['plans']['last_rounded']['loss']


def execute(command, log_path, timeout):
    with Path(log_path).open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # torchrun owns rank subprocesses; terminate the whole new session
            # before recording failure, rather than leaving GPU workers alive.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            # The launcher can exit before a stubborn rank does. Always send
            # the final signal to the group, even when the launcher has exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def execute_run(config, provenance, source, prepared, out, instance, stage, method,
                candidate, initialization, seed, *, executor=execute):
    directory = run_directory(out, instance, stage, candidate, initialization, seed)
    directory.mkdir(parents=True, exist_ok=False)
    expected = runner_config(config, prepared, directory / 'output', instance, stage,
                             method, candidate, initialization, seed)
    command = runner_command(source, config['instances'][instance]['world_size'], expected)
    request = dict(**provenance, stage=stage, instance=instance, candidate_id=candidate['id'],
                   expected_runner_config=expected, command=command, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write_json(directory / 'request.json', request)
    start = time.monotonic()
    try:
        code = executor(command, directory / 'runner.log', config['run_timeout_seconds'])
        require(code == 0, f'Runner subprocess failed with exit status {code}.')
        path = directory / 'output/metadata.json'
        score = validate_metadata(read_json(path), expected, provenance,
                                  config['instances'][instance], config['final_marginal_l1_tolerance'])
        status = dict(status='completed', returncode=code, elapsed_seconds=time.monotonic() - start,
                      metadata_file=str(path), metadata_sha256=sha256(path), last_rounded_loss=score)
        write_json(directory / 'status.json', status)
        return status
    except Exception as exc:
        write_json(directory / 'status.json', dict(status='failed', returncode=locals().get('code'),
                   elapsed_seconds=time.monotonic() - start, error=repr(exc), traceback=traceback.format_exc()))
        raise


def choose(scores):
    """Select only complete six-cell candidates, minimizing median feasible loss."""
    valid = [row for row in scores if row['valid']]
    require(bool(valid), 'No candidate has all six valid development cells; no fallback is allowed.')
    return min(valid, key=lambda row: (row['median_last_rounded_loss'], row['candidate_id']))['candidate_id']


def collect_selection(config, provenance, source, prepared, out, instance):
    scores = {}
    for method in METHODS:
        rows = []
        for candidate in config['methods'][method]:
            cells, errors = [], []
            for initialization in config['initializations']:
                for seed in config['development_seeds']:
                    directory = run_directory(out, instance, 'tuning', candidate, initialization, seed)
                    try:
                        request, status = read_json(directory / 'request.json'), read_json(directory / 'status.json')
                        for key, value in provenance.items():
                            require(request.get(key) == value, f'Request provenance mismatch: {key}.')
                        require(status['status'] == 'completed' and status['returncode'] == 0,
                                'Development subprocess did not complete.')
                        path = directory / 'output/metadata.json'
                        require(status['metadata_sha256'] == sha256(path), 'Development metadata changed after validation.')
                        expected = runner_config(config, prepared, directory / 'output', instance, 'tuning',
                                                 method, candidate, initialization, seed)
                        require(request['expected_runner_config'] == expected, 'Frozen development request changed.')
                        loss = validate_metadata(read_json(path), expected, provenance,
                                                 config['instances'][instance], config['final_marginal_l1_tolerance'])
                        cells.append(dict(initialization=initialization, seed=seed, loss=loss,
                                          metadata_file=str(path), metadata_sha256=sha256(path)))
                    except (ValueError, KeyError, OSError, TypeError) as exc:
                        errors.append(dict(initialization=initialization, seed=seed, error=repr(exc)))
            valid = len(cells) == 6 and not errors
            rows.append(dict(candidate_id=candidate['id'], valid=valid,
                             median_last_rounded_loss=statistics.median(cell['loss'] for cell in cells) if valid else None,
                             cells=cells, errors=errors))
        scores[method] = rows
    return dict(status='completed', **provenance, instance=instance, scores=scores,
                selected={method: choose(rows) for method, rows in scores.items()},
                selection_rule='Minimum median final FP64 last-rounded GW loss over all six development cells; candidate ID breaks ties.',
                development_seeds=config['development_seeds'], final_seeds=config['final_seeds'],
                tuning_horizon=config['tuning_iterations'], final_horizon=config['final_iterations'],
                scientific_scope='Empirical feasible-loss prefactor selection; independently sampled SMD outputs are retained separately.')


def load_selection(config, provenance, out, instance):
    selected = read_json(Path(out) / instance / 'selection.json')
    require(selected['status'] == 'completed', 'Selection did not complete.')
    for key, value in provenance.items():
        require(selected.get(key) == value, f'Selection provenance mismatch: {key}.')
    require(selected['development_seeds'] == config['development_seeds']
            and selected['final_seeds'] == config['final_seeds'], 'Selection seed split differs.')
    for method in METHODS:
        require(selected['selected'][method] == choose(selected['scores'][method]), 'Selection does not minimize its recorded scores.')
        require(selected['selected'][method] in {c['id'] for c in config['methods'][method]}, 'Unknown selected candidate.')
        winner = next(row for row in selected['scores'][method] if row['candidate_id'] == selected['selected'][method])
        require({(c['initialization'], c['seed']) for c in winner['cells']}
                == {(start, seed) for start in config['initializations'] for seed in config['development_seeds']},
                'Winner does not contain exactly the development cells.')
        require(len(winner['cells']) == 6 and winner['median_last_rounded_loss'] == statistics.median(c['loss'] for c in winner['cells']),
                'Winner score does not match development evidence.')
        require(all(sha256(c['metadata_file']) == c['metadata_sha256'] for c in winner['cells']),
                'Selected development evidence changed.')
    return selected


def task_runs(config, instance, stage, task_index, selection=None):
    if stage == 'tuning':
        require(0 <= task_index < 12, 'Tuning task index must be in [0,11].')
        cells = [(start, seed) for start in config['initializations'] for seed in config['development_seeds']]
        candidates = {method: config['methods'][method][task_index] for method in METHODS}
    elif stage == 'final':
        cells = [(start, seed) for start in config['initializations'] for seed in config['final_seeds']]
        require(0 <= task_index < len(cells), 'Final task index must be in [0,5].')
        cells = [cells[task_index]]
        candidates = {method: next(c for c in config['methods'][method] if c['id'] == selection['selected'][method]) for method in METHODS}
    else:
        raise ValueError('Only tuning and final stages execute optimization.')
    result = []
    for cell_index, (initialization, seed) in enumerate(cells):
        order = METHODS if (task_index + cell_index) % 2 == 0 else METHODS[::-1]
        result.extend((method, candidates[method], initialization, seed) for method in order)
    return result


def run_task(args, *, executor=execute):
    config = validate_config(read_json(args.config))
    require(args.instance in config['instances'], 'Unknown instance.')
    provenance = identity(args.config, args.source, args.prepared, args.instance)
    directory = Path(args.out) / args.instance / 'tasks' / args.stage / f'task_{args.task_index:03d}'
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'request.json', dict(**provenance, args=vars(args), slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    try:
        if args.stage == 'select':
            for index in range(12):
                status = read_json(Path(args.out) / args.instance / 'tasks/tuning' / f'task_{index:03d}/status.json')
                require(status['status'] == 'completed' and status['completed_runs'] == 12, 'A tuning task is incomplete or failed.')
            selection = collect_selection(config, provenance, args.source, args.prepared, args.out, args.instance)
            path = Path(args.out) / args.instance / 'selection.json'
            require(not path.exists(), 'Selection already exists; refusing to overwrite it.')
            write_json(path, selection)
            write_json(directory / 'status.json', dict(status='completed', selected=selection['selected'], selection_sha256=sha256(path)))
            return selection
        require(bool(os.environ.get('SLURM_JOB_ID')) or executor is not execute,
                'GPU campaign tasks require an allocated Slurm job.')
        selected = load_selection(config, provenance, args.out, args.instance) if args.stage == 'final' else None
        runs = task_runs(config, args.instance, args.stage, args.task_index, selected)
        completed = []
        for method, candidate, initialization, seed in runs:
            write_json(directory / 'progress.json', dict(status='running', completed_runs=len(completed),
                       total_runs=len(runs), current=dict(method=method, candidate_id=candidate['id'], initialization=initialization, seed=seed)))
            completed.append(execute_run(config, provenance, args.source, args.prepared, args.out, args.instance,
                                         args.stage, method, candidate, initialization, seed, executor=executor))
        status = dict(status='completed', completed_runs=len(completed), runs=completed, **provenance)
        if selected:
            status['selection_sha256'] = sha256(Path(args.out) / args.instance / 'selection.json')
        write_json(directory / 'status.json', status)
        write_json(directory / 'progress.json', dict(status='completed', completed_runs=len(completed), total_runs=len(runs)))
        return status
    except Exception as exc:
        write_json(directory / 'status.json', dict(status='failed', error=repr(exc), traceback=traceback.format_exc(), **provenance))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source', 'prepared', 'out', 'instance'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--stage', choices=('tuning', 'select', 'final'), required=True)
    p.add_argument('--task-index', type=int, default=0)
    return p


if __name__ == '__main__':
    run_task(parser().parse_args())
