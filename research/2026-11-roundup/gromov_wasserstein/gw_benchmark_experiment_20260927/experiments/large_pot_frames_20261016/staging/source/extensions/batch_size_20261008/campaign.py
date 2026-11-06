"""Frozen FraMES probe-count diagnostic; no numerical optimizer implementation."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_batch_published_controller',
    ROOT.parent / 'published_stochastic_20261002/campaign.py')
published = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(published)
ARRAYS, PLANS = published.ARRAYS, published.PLANS
SOURCE_FILES = (*published.SOURCE_FILES, 'batch_size_20261008/run_batch.py')
CONTROLLER_FILES = ('batch_size_20261008/campaign.py', 'published_stochastic_20261002/campaign.py')
RUNNER_KEYS = published.RUNNER_KEYS - {'batch'}
sha256, read_json, write_json = published.sha256, published.read_json, published.write_json
require, finite, execute = published.require, published.finite, published.execute


def _hash(value):
    return (isinstance(value, str) and len(value) == 64 and value != '0'*64
            and all(character in '0123456789abcdef' for character in value))


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(config):
    require(config['schema_version'] == 1, 'Unsupported campaign schema.')
    require(config['method'] == 'smoothing-fw', 'Only FraMES belongs in this ablation.')
    require(config['batches'] == [128, 1024, 2048], 'Frozen probe counts changed.')
    require(config['development_seeds'] == [710, 711, 712], 'Frozen development seeds changed.')
    require(config['initializations'] == ['product'], 'Product initialization must remain frozen.')
    require(config['plan_storage'] == 'none', 'Do not persist dense outputs.')
    require(config['checkpoint_seconds'] == [], 'Use fixed iteration diagnostics.')
    require(set(config['runner']) == RUNNER_KEYS, 'All fixed runner arguments must be explicit.')
    require(config['runner']['dtype'] == 'float32', 'Float32 production optimizer required.')
    require(config['runner']['inner_tol'] == config['runner']['inner_gap_tol'] == 1e-7,
            'Keep inherited inner tolerances fixed, although FraMES performs no inner solve.')
    for name in RUNNER_KEYS - {'dtype', 'inner_tol', 'inner_gap_tol'}:
        require(_integer(config['runner'][name]) and config['runner'][name] > 0, 'Invalid runner flag '+name)
    require(config['final_marginal_l1_tolerance'] == 100*2**-23, 'Frozen feasibility tolerance changed.')
    require(bool(config['instances']), 'No active datasets.')
    for name, instance in config['instances'].items():
        require(name and name not in ('.', '..') and '/' not in name and '\\' not in name, 'Invalid dataset ID.')
        require(_integer(instance['n']) and instance['n'] > 0 and instance['world_size'] == 1,
                'This ablation uses one GPU per square problem.')
        require(_hash(instance['instance_manifest_sha256']), 'Missing frozen input manifest SHA256.')
        require(set(instance['array_files_sha256']) == set(ARRAYS)
                and all(_hash(v) for v in instance['array_files_sha256'].values()), 'Missing frozen array hashes.')
        T = instance['final_iterations']
        require(_integer(T) and T > 0, 'Invalid fixed horizon.')
        checkpoints = instance['final_checkpoint_iterations']
        require(checkpoints == sorted(set(checkpoints)) and T in checkpoints
                and all(_integer(k) and 0 < k <= T for k in checkpoints), 'Invalid diagnostic grid.')
        require(set(instance['prefactors']) == {'gamma0', 'rho0', 'mu_scale'}, 'Unexpected prefactor fields.')
        for key, value in instance['prefactors'].items():
            require(finite(value, key) > 0 and (key == 'mu_scale' or value <= 1), 'Invalid prefactor.')
        require(0 < finite(instance['run_timeout_seconds'], 'Timeout') <= 23*3600,
                'Individual run must fit within twenty-three hours.')
    p = config['preflight']
    require(_integer(p['iterations']) and p['iterations'] > 0 and p['batch'] == max(config['batches'])
            and p['seed'] == config['development_seeds'][0], 'Invalid preflight workload.')
    require(0 < finite(p['run_timeout_seconds'], 'Preflight timeout') <= 3600, 'Invalid preflight timeout.')
    require(finite(p['runtime_safety_factor'], 'Runtime safety factor') >= 2,
            'Require at least twice extrapolated execution time.')
    require(0 < finite(p['maximum_gpu_reserved_fraction'], 'GPU memory bound') <= .9,
            'Require at least ten percent GPU memory headroom.')
    return config


def load_config(path):
    config = validate_config(read_json(path))
    config['_config_path'] = str(Path(path).resolve())
    return config


def cell_for_task(config, task_index):
    cells = [(q, seed) for q in config['batches'] for seed in config['development_seeds']]
    require(_integer(task_index) and 0 <= task_index < len(cells), 'Task index must be in [0,8].')
    return cells[task_index]


def run_directory(out, dataset, q, seed):
    return Path(out) / dataset / f'q{q}' / 'product' / f'seed{seed}'


def identity(config_path, source, prepared, dataset):
    config = load_config(config_path)
    require(dataset in config['instances'], 'Unknown dataset.')
    source = Path(source).resolve()
    path = Path(prepared) / dataset / 'manifest.json'
    manifest, instance = read_json(path), config['instances'][dataset]
    require(sha256(path) == instance['instance_manifest_sha256'], 'Prepared manifest differs from frozen input.')
    require(manifest.get('array_files_sha256') == instance['array_files_sha256'], 'Prepared array hashes differ.')
    return dict(config_sha256=sha256(config_path),
                source_file_sha256={name: sha256(source / 'extensions' / name) for name in SOURCE_FILES},
                controller_source_sha256={name: sha256(source / 'extensions' / name) for name in CONTROLLER_FILES},
                instance_manifest_sha256=sha256(path), array_files_sha256=instance['array_files_sha256'])


def _current_provenance(config, source, prepared, dataset):
    require(load_config(config['_config_path']) == config, 'Configuration changed after loading.')
    return identity(config['_config_path'], source, prepared, dataset)


def runner_config(config, prepared, output, dataset, q, seed, *, preflight=False):
    require(dataset in config['instances'], 'Unknown dataset.')
    require(q in config['batches'] and seed in config['development_seeds'], 'Unknown diagnostic cell.')
    instance = config['instances'][dataset]
    T = config['preflight']['iterations'] if preflight else instance['final_iterations']
    checkpoints = [T] if preflight else instance['final_checkpoint_iterations']
    result = dict(instance=str((Path(prepared) / dataset).resolve()), out=str(Path(output).resolve()),
                  method='smoothing-fw', device='cuda', iterations=T, seed=seed, output_seed=seed+20000,
                  init_seed=seed+10000, initialization='product', prefactor_fraction=1., batch=q,
                  save_plans='none', checkpoint_iterations=','.join(map(str, checkpoints)), checkpoints='')
    result.update(config['runner'])
    result.update(instance['prefactors'])
    return result


def runner_command(source, world_size, arguments):
    command = published.runner_command(Path(source).resolve(), world_size, arguments)
    command[5] = str(Path(source).resolve() / 'extensions/batch_size_20261008/run_batch.py')
    return command


# Independent generalized metadata audit; legacy q128 auditor stays unchanged.
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
    require(counters['sketch_probes'] == expected['batch'] * horizon and counters['inner_stopping_failures'] == 0,
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


def validate_diagnostics(metadata, expected):
    require(metadata['schedule']['gamma_exponent'] == .75
            and metadata['schedule']['rho_exponent'] == .5
            and metadata['schedule']['mu_exponent'] == .25, 'Theoretical schedule exponents differ.')
    points = metadata['trace']
    required = {0, expected['iterations'], *[int(k) for k in expected['checkpoint_iterations'].split(',') if k]}
    require(required <= {p['iteration'] for p in points}, 'Missing requested iteration diagnostics.')
    for point in points:
        require(_integer(point['iteration']) and 0 <= point['iteration'] <= expected['iterations'],
                'Invalid diagnostic iteration.')
        for key in ('raw_loss', 'rounded_loss'):
            finite(point[key], key)
        for key in ('actual_solver_seconds', 'repair_l1', 'repair_seconds_current'):
            finite(point[key], key, nonnegative=True)
        for key in ('raw_residuals', 'rounded_residuals'):
            finite(point[key]['both_l1'], key, nonnegative=True)
        require(point['rounded_residuals']['both_l1'] <= 100*2**-23, 'Infeasible diagnostic rounded output.')
    for rank in metadata['ranks']:
        for phase in ('preparation', 'optimization_and_diagnostics', 'float64_validation'):
            memory = rank['memory'][phase]
            require(finite(memory['cuda_total_memory_bytes'], 'GPU memory') > 0, 'Missing GPU memory size.')
            for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
                finite(memory[key], key, nonnegative=True)
    return metadata


def _audit_metadata(path, expected, provenance, config, dataset):
    metadata = read_json(path)
    score = validate_metadata(metadata, expected, provenance, config['instances'][dataset],
                              config['final_marginal_l1_tolerance'])
    validate_diagnostics(metadata, expected)
    return score, metadata


def validate_gate(config, source, prepared, gate_path):
    gate = read_json(gate_path)
    require(gate['status'] == 'passed', 'Production-shape GPU preflight did not pass.')
    require(gate['config_sha256'] == sha256(config['_config_path']), 'Preflight configuration differs.')
    require(set(gate['instances']) == set(config['instances']), 'Preflight did not cover both datasets.')
    for dataset, record in gate['instances'].items():
        require(record['provenance'] == _current_provenance(config, source, prepared, dataset),
                'Preflight source/input provenance differs.')
        require(record['resource_checks']['passed'] is True, 'Preflight resource gate failed.')
    return sha256(gate_path)


def _execute_run(config, source, prepared, out, dataset, q, seed, *, preflight=False,
                 executor=execute, gate_sha256=None):
    provenance = _current_provenance(config, source, prepared, dataset)
    directory = run_directory(out, dataset, q, seed)
    directory.mkdir(parents=True, exist_ok=False)
    expected = runner_config(config, prepared, directory / 'output', dataset, q, seed, preflight=preflight)
    command = runner_command(source, config['instances'][dataset]['world_size'], expected)
    request = dict(**provenance, stage='preflight' if preflight else 'diagnostic', instance=dataset,
                   method='smoothing-fw', initialization='product', seed=seed, batch=q,
                   gate_sha256=gate_sha256, expected_runner_config=expected, command=command,
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))
    path = directory / 'request.json'
    write_json(path, request)
    begun = time.monotonic()
    try:
        timeout = config['preflight']['run_timeout_seconds'] if preflight else config['instances'][dataset]['run_timeout_seconds']
        code = executor(command, directory / 'runner.log', timeout)
        require(code == 0, f'Runner subprocess failed with exit status {code}.')
        require(_current_provenance(config, source, prepared, dataset) == provenance,
                'Frozen source/config/input changed during execution.')
        metadata_path = directory / 'output/metadata.json'
        score, metadata = _audit_metadata(metadata_path, expected, provenance, config, dataset)
        status = dict(status='completed', returncode=code, elapsed_seconds=time.monotonic()-begun,
                      request_sha256=sha256(path), metadata_file=str(metadata_path.resolve()),
                      metadata_sha256=sha256(metadata_path), last_rounded_loss=score)
        write_json(directory / 'status.json', status)
        return dict(request=request, status=status, metadata=metadata, last_rounded_loss=score)
    except Exception as exc:
        write_json(directory / 'status.json', dict(status='failed', returncode=locals().get('code'),
                   elapsed_seconds=time.monotonic()-begun, request_sha256=sha256(path),
                   error=repr(exc), traceback=traceback.format_exc()))
        raise


def validate_run(config, source, prepared, out, dataset, q, seed):
    provenance = _current_provenance(config, source, prepared, dataset)
    directory = run_directory(out, dataset, q, seed)
    request_path, status_path = directory / 'request.json', directory / 'status.json'
    request, status = read_json(request_path), read_json(status_path)
    require(status['status'] == 'completed' and status['returncode'] == 0, 'Run did not complete successfully.')
    require(status['request_sha256'] == sha256(request_path), 'Frozen request changed after execution.')
    for key, value in provenance.items():
        require(request.get(key) == value, 'Request provenance mismatch: '+key)
    for key, value in dict(stage='diagnostic', instance=dataset, method='smoothing-fw',
                           initialization='product', seed=seed, batch=q).items():
        require(request.get(key) == value, 'Request identity mismatch: '+key)
    require(_hash(request.get('gate_sha256')), 'Missing passed preflight identity.')
    expected = runner_config(config, prepared, directory / 'output', dataset, q, seed)
    require(request['expected_runner_config'] == expected, 'Frozen runner request differs.')
    require(request['command'] == runner_command(source, config['instances'][dataset]['world_size'], expected),
            'Frozen runner command differs.')
    path = directory / 'output/metadata.json'
    require(status['metadata_sha256'] == sha256(path), 'Metadata changed after validation.')
    score, metadata = _audit_metadata(path, expected, provenance, config, dataset)
    require(status['last_rounded_loss'] == score, 'Stored loss differs from validated metadata.')
    return dict(request=request, status=status, metadata=metadata, last_rounded_loss=score)


def run_task(args, *, executor=execute):
    config = load_config(args.config)
    q, seed = cell_for_task(config, args.task_index)
    gate_hash = validate_gate(config, args.source, args.prepared, args.gate)
    return _execute_run(config, args.source, args.prepared, args.out, args.instance, q, seed,
                        executor=executor, gate_sha256=gate_hash)['status']


def resource_checks(config, dataset, metadata):
    """Conservative runtime extrapolation; early objective value never gates jobs."""
    p, instance = config['preflight'], config['instances'][dataset]
    timing = metadata['timing']
    solver = finite(timing['solver_seconds'], 'Preflight solver time', nonnegative=True)
    require(solver > 0, 'Preflight timing must be positive.')
    # Fixed diagnostics are few, but budget all observed non-solver work once per
    # requested final diagnostic, covering FP64 output validation and launch/I/O.
    non_solver = max(0., timing['total_elapsed_seconds'] - solver)
    estimated = (solver / p['iterations'] * instance['final_iterations']
                 + non_solver * (len(instance['final_checkpoint_iterations'])+3))
    reserved = max(phase['cuda_peak_reserved_bytes'] / phase['cuda_total_memory_bytes']
                   for rank in metadata['ranks'] for phase in rank['memory'].values())
    require(reserved <= p['maximum_gpu_reserved_fraction'], 'Production-shape GPU memory headroom insufficient.')
    require(estimated * p['runtime_safety_factor'] <= instance['run_timeout_seconds'],
            'Production runtime estimate exceeds declared timeout with safety margin.')
    return dict(passed=True, maximum_gpu_reserved_fraction=reserved,
                extrapolated_runtime_seconds=estimated,
                runtime_with_safety_factor_seconds=estimated*p['runtime_safety_factor'],
                allowed_runtime_seconds=instance['run_timeout_seconds'],
                scope='Resource/finite-output gate only; no loss-improvement condition.')


def preflight(args, *, executor=execute):
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gate = dict(status='running', config_sha256=sha256(args.config), instances={})
    try:
        for dataset in config['instances']:
            run = _execute_run(config, args.source, args.prepared, out / 'runs', dataset,
                config['preflight']['batch'], config['preflight']['seed'], preflight=True, executor=executor)
            checks = resource_checks(config, dataset, run['metadata'])
            gate['instances'][dataset] = dict(provenance=_current_provenance(config, args.source, args.prepared, dataset),
                metadata_sha256=run['status']['metadata_sha256'], resource_checks=checks)
        gate['status'] = 'passed'
        write_json(out / 'gate.json', gate)
        return gate
    except Exception as exc:
        gate.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        write_json(out / 'gate.json', gate)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('run', 'preflight'):
        child = sub.add_parser(action)
        for name in ('config', 'source', 'prepared', 'out'):
            child.add_argument('--'+name, required=True)
        if action == 'run':
            child.add_argument('--instance', required=True)
            child.add_argument('--task-index', type=int, required=True)
            child.add_argument('--gate', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    result = preflight(args) if args.action == 'preflight' else run_task(args)
    print(result['status'], flush=True)
