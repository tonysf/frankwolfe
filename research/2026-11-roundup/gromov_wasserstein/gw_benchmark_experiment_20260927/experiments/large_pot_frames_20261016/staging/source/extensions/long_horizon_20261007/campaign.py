"""Longer fixed-horizon reruns with previously selected, immutable parameters.

The two existing controllers supply command construction and their independent
numerical audits. This module changes only the experiment horizon and storage
policy; it does not implement or modify any optimizer.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import time
import traceback


def _load_legacy(name, directory):
    path = Path(__file__).resolve().parents[1] / directory / 'campaign.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


published = _load_legacy('_long_horizon_published_controller', 'published_stochastic_20261002')
msbpg = _load_legacy('_long_horizon_msbpg_controller', 'msbpg_20261005')
METHODS = ('smoothing-fw', 'published-smd', 'published-msbpg')
ARRAYS, PLANS, RUNNER_KEYS = published.ARRAYS, published.PLANS, published.RUNNER_KEYS
SOURCE_FILES = tuple(dict.fromkeys((*published.SOURCE_FILES, *msbpg.SOURCE_FILES)))
CONTROLLER_FILES = ('long_horizon_20261007/campaign.py',
                    'published_stochastic_20261002/campaign.py', 'msbpg_20261005/campaign.py')
sha256, read_json, write_json = published.sha256, published.read_json, published.write_json
require, finite, execute = published.require, published.finite, published.execute


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _identifier(value):
    return isinstance(value, str) and value not in ('', '.', '..') and '/' not in value and '\\' not in value


def legacy(method):
    require(method in METHODS, 'Unknown method.')
    return msbpg if method == 'published-msbpg' else published


def validate_config(config):
    require(config['schema_version'] == 1, 'Unsupported campaign schema.')
    require(set(config['methods']) == set(METHODS), 'All three frozen stochastic methods are required.')
    require(config['initializations'] == ['product', 'randomized'], 'Both original initializations are required.')
    require(config['final_seeds'] == [810, 811, 812], 'Original final seeds must remain frozen.')
    require(config['plan_storage'] == 'none', 'This campaign retains diagnostics and hashes without dense plans.')
    horizon = config['final_iterations']
    require(isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0,
            'Invalid final horizon.')
    checkpoints = config['final_checkpoint_iterations']
    require(checkpoints == sorted(set(checkpoints)) and horizon in checkpoints
            and all(isinstance(k, int) and not isinstance(k, bool) and 0 < k <= horizon for k in checkpoints),
            'Invalid fixed checkpoints or missing final checkpoint.')
    require(config['checkpoint_seconds'] == [], 'Only fixed iteration checkpoints are requested.')
    require(set(config['runner']) == RUNNER_KEYS, 'Runner flags must be explicit and complete.')
    runner = config['runner']
    require(runner['batch'] == 128 and runner['dtype'] == 'float32', 'Original float32/q128 configuration required.')
    require(runner['inner_tol'] == runner['inner_gap_tol'] == 1e-7, 'Both inner tolerances must remain 1e-7.')
    for key in RUNNER_KEYS - {'dtype', 'inner_tol', 'inner_gap_tol'}:
        require(isinstance(runner[key], int) and not isinstance(runner[key], bool) and runner[key] > 0,
                'Invalid runner parameter: '+key)
    require(set(config['run_timeout_seconds_by_method']) == set(METHODS), 'Each method needs an explicit timeout.')
    for timeout in config['run_timeout_seconds_by_method'].values():
        require(0 < finite(timeout, 'Run timeout') <= 23*3600, 'Run timeout must be positive and at most twenty-three hours.')
    require(finite(config['final_marginal_l1_tolerance'], 'Final feasibility tolerance') == 100*2**-23,
            'Final feasibility tolerance must remain frozen.')
    require(bool(config['instances']), 'At least one dataset must be active.')
    for name, instance in config['instances'].items():
        require(_identifier(name), 'Invalid dataset identifier.')
        require(isinstance(instance['n'], int) and instance['n'] > 0, 'Invalid dataset dimension.')
        require(instance['world_size'] in (1, 4), 'Only one- or four-GPU allocations are supported.')
        require(_hash(instance['instance_manifest_sha256']), 'Missing frozen manifest SHA256.')
        require(set(instance['array_files_sha256']) == set(ARRAYS)
                and all(_hash(value) for value in instance['array_files_sha256'].values()), 'Invalid frozen array hashes.')
    identifiers = []
    fields = {'smoothing-fw': {'id', 'gamma0', 'rho0', 'mu_scale'},
              'published-smd': {'id', 'prefactor_fraction'},
              'published-msbpg': {'id', 'coupling_fraction', 'momentum_prefactor'}}
    for method, candidates in config['methods'].items():
        require(len(candidates) == 12, 'Keep the original twelve-candidate catalogs.')
        for candidate in candidates:
            require(set(candidate) == fields[method], 'Unexpected candidate parameters for '+method)
            require(_identifier(candidate['id']), 'Invalid candidate identifier.')
            identifiers.append(candidate['id'])
            for key, value in candidate.items():
                if key != 'id':
                    require(finite(value, key) > 0 and (key == 'mu_scale' or value <= 1), 'Invalid candidate prefactor.')
    require(len(set(identifiers)) == len(identifiers), 'Candidate identifiers must be globally unique.')
    require(set(config['selected_candidates']) == set(config['instances']), 'Selected datasets differ from active datasets.')
    for dataset, selections in config['selected_candidates'].items():
        require(set(selections) == set(METHODS), 'Each dataset requires three frozen selections.')
        for method, candidate in selections.items():
            require(candidate in config['methods'][method], 'Selected candidate differs from its original catalog.')
    specification = config['selections_provenance']
    require(_identifier(specification['file']) and _hash(specification['sha256']), 'Invalid frozen selector artifact.')
    return config


def load_config(path):
    """Validate the configuration and bind every active selection to its evidence."""
    path = Path(path).resolve()
    config = validate_config(read_json(path))
    specification = config['selections_provenance']
    artifact_path = path.parent / specification['file']
    require(sha256(artifact_path) == specification['sha256'], 'Frozen selection artifact hash mismatch.')
    artifact = read_json(artifact_path)
    require(artifact['schema_version'] == 1, 'Unsupported frozen selection schema.')
    for dataset, instance in config['instances'].items():
        evidence = artifact['selections'][dataset]
        require(set(evidence) == set(METHODS), 'Frozen artifact omits a method.')
        for method, record in evidence.items():
            require(record['candidate'] == config['selected_candidates'][dataset][method],
                    'Selected candidate differs from frozen selection evidence.')
            require(_hash(record['selection_file_sha256']), 'Original selection hash missing.')
            require(record['array_files_sha256'] == instance['array_files_sha256'], 'Frozen selection input arrays differ.')
            require(record['instance_manifest_sha256'] == instance['instance_manifest_sha256'],
                    'Frozen selection input manifest differs.')
    config['_config_path'] = str(path)
    return config


def cell_for_task(config, task_index):
    cells = [(start, seed) for start in config['initializations'] for seed in config['final_seeds']]
    require(isinstance(task_index, int) and not isinstance(task_index, bool) and 0 <= task_index < len(cells),
            'Task index must be in [0,5].')
    return cells[task_index]


def run_directory(out, dataset, method, start, seed):
    return Path(out) / dataset / method / start / f'seed{seed}'


def identity(config_path, source, prepared, dataset, method):
    config = load_config(config_path)
    require(dataset in config['instances'], 'Unknown active dataset.')
    module = legacy(method)
    source = Path(source).resolve()
    manifest_path = Path(prepared) / dataset / 'manifest.json'
    manifest = read_json(manifest_path)
    instance = config['instances'][dataset]
    require(sha256(manifest_path) == instance['instance_manifest_sha256'], 'Prepared manifest differs from frozen input.')
    require(manifest.get('array_files_sha256') == instance['array_files_sha256'], 'Prepared arrays differ from frozen input.')
    # Each unchanged numerical runner independently hashes all four array files
    # while loading them. The controller/reporter need only read the small bound
    # manifest, avoiding repeated multi-GB array scans during report generation.
    return dict(config_sha256=sha256(config_path),
                frozen_selections_sha256=config['selections_provenance']['sha256'],
                source_file_sha256={name: sha256(source / 'extensions' / name) for name in module.SOURCE_FILES},
                controller_source_sha256={name: sha256(source / 'extensions' / name) for name in CONTROLLER_FILES},
                instance_manifest_sha256=sha256(manifest_path), array_files_sha256=instance['array_files_sha256'])


def runner_config(config, prepared, output, dataset, method, start, seed):
    require(dataset in config['instances'] and method in METHODS, 'Unknown dataset or method.')
    require((start, seed) in [cell_for_task(config, index) for index in range(6)], 'Unknown final seed/start cell.')
    candidate = config['selected_candidates'][dataset][method]
    arguments = legacy(method).runner_config(config, prepared, output, dataset, 'final', method, candidate, start, seed)
    arguments['save_plans'] = 'none'
    return arguments


def runner_command(source, world_size, arguments):
    return legacy(arguments['method']).runner_command(Path(source).resolve(), world_size, arguments)


def _current_provenance(config, source, prepared, dataset, method):
    current = load_config(config['_config_path'])
    require(current == config, 'Configuration changed after it was loaded.')
    return identity(config['_config_path'], source, prepared, dataset, method)


def validate_run(config, source, prepared, out, dataset, method, start, seed):
    provenance = _current_provenance(config, source, prepared, dataset, method)
    directory = run_directory(out, dataset, method, start, seed)
    request_path, status_path = directory / 'request.json', directory / 'status.json'
    request, status = read_json(request_path), read_json(status_path)
    require(status['status'] == 'completed' and status['returncode'] == 0, 'Run did not complete successfully.')
    require(status['request_sha256'] == sha256(request_path), 'Frozen request changed after validation.')
    for key, value in provenance.items():
        require(request.get(key) == value, 'Request provenance mismatch: '+key)
    for key, value in dict(stage='final', instance=dataset, method=method, initialization=start, seed=seed,
                           candidate_id=config['selected_candidates'][dataset][method]['id']).items():
        require(request.get(key) == value, 'Request identity mismatch: '+key)
    expected = runner_config(config, prepared, directory / 'output', dataset, method, start, seed)
    require(request['expected_runner_config'] == expected, 'Frozen runner request differs.')
    require(request['command'] == runner_command(source, config['instances'][dataset]['world_size'], expected),
            'Frozen runner command differs.')
    path = directory / 'output/metadata.json'
    require(status['metadata_sha256'] == sha256(path), 'Metadata changed after validation.')
    metadata = read_json(path)
    score = legacy(method).validate_metadata(metadata, expected, provenance, config['instances'][dataset],
                                              config['final_marginal_l1_tolerance'])
    require(status['last_rounded_loss'] == score, 'Stored loss differs from validated metadata.')
    return dict(request=request, status=status, metadata=metadata, last_rounded_loss=score)


def run_task(args, *, executor=execute):
    config = load_config(args.config)
    start, seed = cell_for_task(config, args.task_index)
    method, dataset = args.method, args.instance
    provenance = _current_provenance(config, args.source, args.prepared, dataset, method)
    directory = run_directory(args.out, dataset, method, start, seed)
    directory.mkdir(parents=True, exist_ok=False)
    expected = runner_config(config, args.prepared, directory / 'output', dataset, method, start, seed)
    command = runner_command(args.source, config['instances'][dataset]['world_size'], expected)
    request = dict(**provenance, stage='final', instance=dataset, method=method, initialization=start, seed=seed,
                   candidate_id=config['selected_candidates'][dataset][method]['id'],
                   expected_runner_config=expected, command=command,
                   slurm_job_id=os.environ.get('SLURM_JOB_ID'), slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))
    request_path = directory / 'request.json'
    write_json(request_path, request)
    begun = time.monotonic()
    try:
        code = executor(command, directory / 'runner.log', config['run_timeout_seconds_by_method'][method])
        require(code == 0, f'Runner subprocess failed with exit status {code}.')
        # Recheck the small bound sources/configs after execution, before accepting
        # the result. Metadata also carries hashes measured by the numerical runner.
        require(_current_provenance(config, args.source, args.prepared, dataset, method) == provenance,
                'Frozen provenance changed during execution.')
        path = directory / 'output/metadata.json'
        score = legacy(method).validate_metadata(read_json(path), expected, provenance, config['instances'][dataset],
                                                  config['final_marginal_l1_tolerance'])
        status = dict(status='completed', returncode=code, elapsed_seconds=time.monotonic()-begun,
                      request_sha256=sha256(request_path), metadata_file=str(path.resolve()),
                      metadata_sha256=sha256(path), last_rounded_loss=score)
        write_json(directory / 'status.json', status)
        return status
    except Exception as exc:
        write_json(directory / 'status.json', dict(status='failed', returncode=locals().get('code'),
                   elapsed_seconds=time.monotonic()-begun, request_sha256=sha256(request_path),
                   error=repr(exc), traceback=traceback.format_exc()))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source', 'prepared', 'out', 'instance'):
        result.add_argument('--'+name, required=True)
    result.add_argument('--method', choices=METHODS, required=True)
    result.add_argument('--task-index', type=int, required=True)
    return result


if __name__ == '__main__':
    run_task(parser().parse_args())
