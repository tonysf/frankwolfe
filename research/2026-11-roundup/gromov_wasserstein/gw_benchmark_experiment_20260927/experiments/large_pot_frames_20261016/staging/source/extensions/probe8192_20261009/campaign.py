"""Frozen q8192 FraMES continuation with dense, read-only diagnostics.

The isolated prior controller supplies unchanged receipt, process, provenance,
and metadata audits. Only this module's private imported instance is configured
for the new runner, fixed cells, diagnostic contract, and stronger launch gate.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_q8192_base_controller', ROOT.parent/'batch_size_20261008/campaign.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
_base_validate_config = base.validate_config
_base_validate_diagnostics = base.validate_diagnostics
SOURCE_FILES = (*base.published.SOURCE_FILES, 'probe8192_20261009/run_diagnostic.py',
                'probe8192_20261009/direction_diagnostics.py')
CONTROLLER_FILES = ('probe8192_20261009/campaign.py', 'probe8192_20261009/verify_gpu.py',
                    'probe8192_20261009/trajectory_parity.py',
                    'batch_size_20261008/campaign.py', 'batch_size_20261008/run_batch.py',
                    'published_stochastic_20261002/campaign.py')
ARRAYS, PLANS = base.ARRAYS, base.PLANS
sha256, read_json, write_json, require, finite, execute = (
    base.sha256, base.read_json, base.write_json, base.require, base.finite, base.execute)
DENSE_CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 200, 500, *range(1000, 200001, 1000)]
DATASET = 'metric512_s1000'


def validate_config(config):
    compatible = copy.deepcopy(config)
    compatible['batches'] = [128, 1024, 2048]
    compatible['preflight']['batch'] = 2048
    _base_validate_config(compatible)
    require(config['batches'] == [8192] and set(config['instances']) == {DATASET},
            'Only q8192 on the frozen clean synthetic instance is authorized.')
    instance = config['instances'][DATASET]
    require(instance['n'] == 512 and instance['final_iterations'] == 200000,
            'Frozen 512-point, 200000-update experiment changed.')
    require(instance['prefactors'] == dict(gamma0=1., rho0=.3, mu_scale=10.),
            'Keep original synthetic prefactors fixed.')
    require(instance['final_checkpoint_iterations'] == DENSE_CHECKPOINTS,
            'Dense checkpoint grid changed.')
    require(instance['run_timeout_seconds'] == 7200, 'Production timeout must remain two hours.')
    require(config['preflight']['batch'] == 8192 and config['preflight']['iterations'] == 2000,
            'Production-shape q8192 preflight must use 2000 updates.')
    require(config['preflight']['runtime_safety_factor'] >= 2., 'Insufficient runtime safety margin.')
    require(config['direction_diagnostics'] == 'enabled', 'Production direction diagnostics must be enabled.')
    receipt = config['local_validation']
    require(receipt['file'] == 'local_validation.json', 'Unexpected local validation receipt.')
    if 'baseline' in config:
        require(config['baseline']['file'] == 'baseline_report.json' and base._hash(config['baseline']['sha256']),
                'Invalid frozen prior-report identity.')
    return config


def runner_config(config, prepared, output, dataset, q, seed, *, preflight=False):
    require(dataset == DATASET and q == 8192 and seed in config['development_seeds'], 'Unknown diagnostic cell.')
    instance = config['instances'][dataset]
    T = config['preflight']['iterations'] if preflight else instance['final_iterations']
    checkpoints = [k for k in instance['final_checkpoint_iterations'] if k <= T]
    values = dict(instance=str((Path(prepared)/dataset).resolve()), out=str(Path(output).resolve()),
                  method='smoothing-fw', device='cuda', iterations=T, seed=seed,
                  output_seed=seed+20000, init_seed=seed+10000, initialization='product',
                  prefactor_fraction=1., batch=q, save_plans='none',
                  checkpoint_iterations=','.join(map(str, checkpoints)), checkpoints='',
                  direction_diagnostics='enabled')
    values.update(config['runner'])
    values.update(instance['prefactors'])
    return values


def runner_command(source, world_size, arguments):
    command = base.published.runner_command(Path(source).resolve(), world_size, arguments)
    command[5] = str(Path(source).resolve()/'extensions/probe8192_20261009/run_diagnostic.py')
    return command


def validate_diagnostics(metadata, expected):
    _base_validate_diagnostics(metadata, expected)
    required = {0, *map(int, expected['checkpoint_iterations'].split(','))} - {expected['iterations']}
    directions = metadata['direction_trace']
    require([row['iteration'] for row in directions] == sorted(required),
            'Missing, duplicated, or reordered preupdate direction diagnostics.')
    previous_solver = previous_wall = -1.
    timing = metadata['timing']
    finite(timing['direction_diagnostic_seconds'], 'Direction diagnostic time', nonnegative=True)
    for row in directions:
        k = row['iteration']
        require(row['plan_index'] == row['momentum_index'] == k
                and row['position'] == 'after momentum update, before FW step', 'Direction/iterate index mismatch.')
        require(row['optimizer_dtype'] == expected['dtype'] and row['full_gradient_dtype'] == row['norm_dtype'] == 'float64',
                'Direction diagnostic precision mismatch.')
        prescriptions = dict(gamma=expected['gamma0']/(k+1)**.75,
            rho_scheduled=expected['rho0']/math.sqrt(k+1),
            rho_effective=1. if k == 0 else expected['rho0']/math.sqrt(k+1),
            mu=expected['mu_scale']/512/(k+1)**.25)
        require(all(math.isclose(finite(row[key], key), value, rel_tol=1e-14)
                    for key, value in prescriptions.items()), 'Direction schedule parameters differ.')
        require(row['momentum_initialization_reuses_first_gradient'] is (k == 0), 'Initialization diagnostic mismatch.')
        for key in ('penalty_value', 'row_centered_momentum_norm', 'row_centered_full_gradient_norm',
                    'row_centered_penalty_norm', 'row_centered_momentum_error_norm', 'raw_structure_l1'):
            finite(row[key], key, nonnegative=True)
        for key in ('row_l1', 'col_l1', 'col_l2', 'both_l1', 'mass_error'):
            finite(row['residuals'][key], key, nonnegative=True)
        require(finite(row['residuals']['min_entry'], 'min entry') >= 0, 'Negative coupling entry.')
        for ratio, numerator, denominator in (
            ('penalty_to_momentum_ratio', 'row_centered_penalty_norm', 'row_centered_momentum_norm'),
            ('row_centered_momentum_relative_error', 'row_centered_momentum_error_norm', 'row_centered_full_gradient_norm')):
            value = row[ratio]
            if row[denominator] == 0:
                require(value is None, 'Undefined norm ratio must be null.')
            else:
                require(math.isclose(finite(value, ratio, nonnegative=True), row[numerator]/row[denominator], rel_tol=1e-12, abs_tol=1e-15),
                        'Inconsistent diagnostic norm ratio.')
        for key in ('lmo_agreement_with_without_penalty',
                    'lmo_agreement_momentum_full_gradient_without_penalty',
                    'lmo_agreement_momentum_full_gradient_with_penalty'):
            require(0 <= finite(row[key], key) <= 1, 'LMO agreement outside [0,1].')
        for key in ('lmo_with_penalty', 'lmo_without_penalty'):
            choices = row[key]
            require(base._integer(choices['selected_column_count']) and 1 <= choices['selected_column_count'] <= 512,
                    'Invalid selected-column count.')
            for field in ('largest_row_fraction', 'largest_source_mass'):
                require(0 <= finite(choices[field], field) <= 1+1e-6, 'Invalid largest-column fraction.')
        solver = finite(row['actual_solver_seconds'], 'Direction solver time', nonnegative=True)
        wall = finite(row['actual_end_to_end_seconds'], 'Direction wall time', nonnegative=True)
        require(previous_solver <= solver <= timing['solver_seconds']+1e-8
                and previous_wall <= wall <= timing['total_elapsed_seconds']+1e-8
                and solver <= wall+1e-8, 'Inconsistent direction timing.')
        previous_solver, previous_wall = solver, wall
    previous_wall = -1.
    for row in metadata['trace']:
        wall = finite(row['actual_end_to_end_seconds'], 'Objective wall time', nonnegative=True)
        require(previous_wall <= wall <= timing['total_elapsed_seconds']+1e-8
                and row['actual_solver_seconds'] <= wall+1e-8, 'Inconsistent objective wall time.')
        previous_wall = wall
    return metadata


def _local_validation(config, source, provenance):
    path = Path(source)/'extensions/probe8192_20261009'/config['local_validation']['file']
    record = read_json(path)
    require(record['status'] == 'passed', 'Local tests did not pass.')
    require(base._integer(record['tests']['passed']) and record['tests']['passed'] > 0
            and record['tests']['failures'] == record['tests']['errors'] == 0, 'Missing successful local test evidence.')
    require(record['source_file_sha256'] == provenance['source_file_sha256'], 'Local tests used different numerical source.')
    return dict(file=str(path.resolve()), sha256=sha256(path))


def _gpu_validation(path, provenance):
    record = read_json(path)
    require(record['status'] == 'passed', 'GPU trajectory parity did not pass.')
    require(record['source_file_sha256'] == provenance['source_file_sha256'], 'GPU tests used different numerical source.')
    for name in ('q8192_enabled_disabled', 'q2048_legacy'):
        require(record['checks'][name]['status'] == 'passed', 'Missing GPU parity check: '+name)
    require(record['array_files_sha256'] == provenance['array_files_sha256']
            and record['instance_manifest_sha256'] == provenance['instance_manifest_sha256'],
            'GPU parity input hashes differ.')
    helpers = {name: provenance['controller_source_sha256']['probe8192_20261009/'+name]
               for name in ('verify_gpu.py', 'trajectory_parity.py')}
    require(record['helper_source_sha256'] == helpers, 'GPU verification helper source differs.')
    reference = {name: provenance['source_file_sha256'][name] for name in base.published.SOURCE_FILES}
    reference['batch_size_20261008/run_batch.py'] = provenance['controller_source_sha256']['batch_size_20261008/run_batch.py']
    require(record['reference_source_file_sha256'] == reference, 'Prior-runner parity source differs.')
    require(set(record['runs']) == {'q2048_legacy', 'q2048_enabled', 'q8192_disabled', 'q8192_enabled'},
            'Incomplete GPU parity runs.')
    for run in record['runs'].values():
        require(sha256(run['metadata_file']) == run['metadata_sha256'], 'GPU parity metadata changed.')
    return dict(file=str(Path(path).resolve()), sha256=sha256(path))


def validate_gate(config, source, prepared, gate_path):
    gate = read_json(gate_path)
    require(gate['status'] == 'passed', 'Production-shape GPU preflight did not pass.')
    require(gate['config_sha256'] == sha256(config['_config_path']), 'Preflight configuration differs.')
    require(set(gate['instances']) == {DATASET}, 'Preflight did not cover the frozen dataset.')
    record = gate['instances'][DATASET]
    provenance = base._current_provenance(config, source, prepared, DATASET)
    require(record['provenance'] == provenance, 'Preflight source/input provenance differs.')
    require(record['resource_checks']['passed'] is True, 'Preflight resource gate failed.')
    require(gate['local_validation'] == _local_validation(config, source, provenance), 'Local test receipt changed.')
    require(gate['gpu_validation'] == _gpu_validation(gate['gpu_validation']['file'], provenance), 'GPU test receipt changed.')
    return sha256(gate_path)


def resource_checks(config, metadata, *, process_elapsed_seconds=None):
    """Forecast all work, including dense full-gradient/feasible-loss diagnostics."""
    p, instance, t = config['preflight'], config['instances'][DATASET], metadata['timing']
    T, pilot = instance['final_iterations'], p['iterations']
    solver = finite(t['solver_seconds'], 'Preflight solver time', nonnegative=True)
    require(solver > 0, 'Preflight solver time must be positive.')
    direction = finite(t['direction_diagnostic_seconds'], 'Direction time', nonnegative=True)
    objective = finite(t['exact_diagnostic_seconds'], 'Objective time', nonnegative=True)
    repair = finite(t['repair_seconds'], 'Rounding time', nonnegative=True)
    logging = finite(t['progress_logging_seconds'], 'Logging time', nonnegative=True)
    measured_variable = direction+objective+repair+logging
    observed_total = finite(t['total_elapsed_seconds'], 'Total time', nonnegative=True)
    if process_elapsed_seconds is not None:
        observed_total = max(observed_total, finite(process_elapsed_seconds, 'Subprocess elapsed time', nonnegative=True))
    fixed = max(0., observed_total-solver-measured_variable)
    count = len(metadata['direction_trace'])
    require(count > 0 and len(metadata['trace']) > 0, 'Empty calibration diagnostics.')
    final_directions = len(instance['final_checkpoint_iterations'])
    final_objectives = len(instance['final_checkpoint_iterations'])+3  # initial, selected, last
    final_logs = T//config['runner']['progress_every']+len(instance['final_checkpoint_iterations'])+2
    pilot_logs = len(metadata['progress_reporting']['completed_updates_recorded'])
    require(pilot_logs > 0, 'Missing calibration progress records.')
    estimated = (solver/pilot*T + direction/count*final_directions
                 + (objective+repair)/len(metadata['trace'])*final_objectives
                 + logging/pilot_logs*final_logs + fixed)
    reserved = max(phase['cuda_peak_reserved_bytes']/phase['cuda_total_memory_bytes']
                   for rank in metadata['ranks'] for phase in rank['memory'].values())
    require(reserved <= p['maximum_gpu_reserved_fraction'], 'Production-shape GPU memory headroom insufficient.')
    require(estimated*p['runtime_safety_factor'] <= instance['run_timeout_seconds'],
            'Production runtime estimate exceeds declared timeout with safety margin.')
    return dict(passed=True, maximum_gpu_reserved_fraction=reserved,
                observed_calibration_process_seconds=observed_total,
                extrapolated_total_runtime_seconds=estimated,
                runtime_with_safety_factor_seconds=estimated*p['runtime_safety_factor'],
                allowed_runtime_seconds=instance['run_timeout_seconds'],
                scope='All measured solver, dense diagnostic, repair, logging and fixed costs; no loss-based gate.')


def preflight(args, *, executor=execute):
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gate = dict(status='running', config_sha256=sha256(args.config), instances={})
    try:
        provenance = base._current_provenance(config, args.source, args.prepared, DATASET)
        gate['local_validation'] = _local_validation(config, args.source, provenance)
        command = [sys.executable, str(Path(args.source)/'extensions/probe8192_20261009/verify_gpu.py'),
                   '--source', str(args.source), '--prepared', str(args.prepared), '--out', str(out/'gpu_validation')]
        code = executor(command, out/'gpu_validation.log', 1200)
        require(code == 0, 'GPU parity helper failed with exit status '+str(code))
        gate['gpu_validation'] = _gpu_validation(out/'gpu_validation/parity.json', provenance)
        run = base._execute_run(config, args.source, args.prepared, out/'runs', DATASET,
            8192, config['preflight']['seed'], preflight=True, executor=executor)
        gate['instances'][DATASET] = dict(provenance=provenance,
            metadata_sha256=run['status']['metadata_sha256'],
            resource_checks=resource_checks(config, run['metadata'], process_elapsed_seconds=run['status']['elapsed_seconds']))
        require(base._current_provenance(config, args.source, args.prepared, DATASET) == provenance,
                'Source/input/config changed during preflight.')
        gate['status'] = 'passed'
        write_json(out/'gate.json', gate)
        return gate
    except Exception as exc:
        gate.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        write_json(out/'gate.json', gate)
        raise


# The imported module is private to this extension: its generic orchestration
# resolves the explicit experiment-specific contracts above at each call.
base.validate_config = validate_config
base.SOURCE_FILES, base.CONTROLLER_FILES = SOURCE_FILES, CONTROLLER_FILES
base.runner_config, base.runner_command = runner_config, runner_command
base.validate_diagnostics, base.validate_gate = validate_diagnostics, validate_gate
load_config, identity, run_directory, cell_for_task = base.load_config, base.identity, base.run_directory, base.cell_for_task
validate_run, run_task = base.validate_run, base.run_task


if __name__ == '__main__':
    args = base.parser().parse_args()
    result = preflight(args) if args.action == 'preflight' else run_task(args)
    print(result['status'], flush=True)
