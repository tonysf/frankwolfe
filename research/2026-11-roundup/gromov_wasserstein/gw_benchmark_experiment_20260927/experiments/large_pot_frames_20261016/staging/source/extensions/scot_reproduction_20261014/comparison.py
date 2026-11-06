"""Reference-gated stochastic comparison on the released SCOT 1,047-cell data.

All numerical updates and their audits are reused from the pair-sampling
campaign. Only prefactors are selected, on one development seed at the same
fixed horizon subsequently used for three independent confirmation seeds.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import traceback

ROOT = Path(__file__).resolve().parent
METHODS = ('smoothing-fw', 'published-smd', 'published-msbpg')
LABELS = {
    'smoothing-fw': 'FraMES with gradient momentum',
    'published-smd': 'Stochastic entropic mirror descent with fresh gradients',
    'published-msbpg': 'MSBPG with theorem-prescribed coupled schedules',
}
RUNNER_KEYS = {'dtype', 'inner_tol', 'inner_gap_tol', 'inner_max', 'check_every',
               'certificate_block_rows', 'exact_block_rows', 'progress_every'}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


pair = module(ROOT.parent / 'pair_sampling_20261011/campaign.py', '_scot_pair_campaign')
read, write, sha256, require, finite = pair.read, pair.write, pair.sha256, pair.require, pair.finite


def load_config(path):
    path = Path(path).resolve()
    c = read(path)
    require(c['schema_version'] == 1, 'Unsupported SCOT comparison schema.')
    require(c['instance'] == dict(id='scot1047', n=1047, world_size=1), 'Wrong reference instance.')
    require(set(c['methods']) == set(METHODS), 'All three stochastic methods are required.')
    require(c['oracle'] == 'pair-sampling' and c['pairs_per_call'] == 1024, 'The common pair budget must be 1024.')
    require(c['initialization'] == 'product' and c['plan_storage'] == 'none', 'Incorrect start or storage policy.')
    require(c['tuning_iterations'] == c['final_iterations'] == 20000, 'Both fixed horizons must be 20000.')
    require(c['tuning_seed'] == 810 and c['final_seeds'] == [811, 812, 813], 'Development and confirmation seeds changed.')
    require(c['selection_metric'] == 'independent_float64_last_rounded_gw_loss', 'Selection must use the common final feasible loss.')
    require(0 < c['run_timeout_seconds'] <= 23 * 3600, 'Run timeout exceeds the Slurm safety margin.')
    require(0 < c['final_marginal_l1_tolerance'] <= 100 * 2**-23, 'Invalid final feasibility tolerance.')
    r = c['runner']
    require(set(r) == RUNNER_KEYS, 'Unexpected or missing numerical runner arguments.')
    require(r['dtype'] == 'float32' and r['inner_tol'] == r['inner_gap_tol'] == 1e-7, 'Production precision or inner tolerance changed.')
    for key in RUNNER_KEYS - {'dtype', 'inner_tol', 'inner_gap_tol'}:
        require(type(r[key]) is int and r[key] > 0, 'Invalid runner argument: ' + key)
    p = c['preflight']
    require(type(p['iterations']) is int and 0 < p['iterations'] < c['tuning_iterations'], 'Invalid preflight horizon.')
    require(type(p['seed']) is int and 0 <= p['seed'] < 2**63 - 20000 and p['seed'] not in [c['tuning_seed']] + c['final_seeds'], 'Preflight must use its own seed.')
    require(0 < p['run_timeout_seconds'] <= c['run_timeout_seconds'] and p['runtime_safety_factor'] >= 2, 'Missing resource safety margin.')
    require(0 < p['maximum_gpu_reserved_fraction'] < 1, 'Invalid memory headroom.')
    for phase in ('tuning', 'final'):
        checkpoints = c[phase + '_checkpoint_iterations']
        require(checkpoints == sorted(set(checkpoints)) and checkpoints[-1] == c[phase + '_iterations']
                and all(type(k) is int and 0 < k <= c[phase + '_iterations'] for k in checkpoints), 'Invalid fixed checkpoints.')
    require(c['tuning_checkpoint_iterations'] == c['final_checkpoint_iterations'], 'Development and confirmation diagnostics must match.')
    identifiers = []
    fields = {
        'smoothing-fw': {'id', 'gamma0', 'rho0', 'mu_scale'},
        'published-smd': {'id', 'prefactor_fraction'},
        'published-msbpg': {'id', 'coupling_fraction', 'momentum_prefactor'},
    }
    for method in METHODS:
        require(len(c['methods'][method]) == 9, 'Each method receives nine declared candidates.')
        values = []
        for candidate in c['methods'][method]:
            require(set(candidate) == fields[method], 'Only the declared prefactors can be tuned.')
            identifier = candidate['id']
            require(isinstance(identifier, str) and identifier and all(ch.isalnum() or ch in '_-' for ch in identifier), 'Invalid candidate ID.')
            identifiers.append(identifier)
            for key, value in candidate.items():
                if key != 'id':
                    require(finite(value, key) > 0 and (key == 'mu_scale' or value <= 1), 'Invalid theory prefactor.')
            if method == 'smoothing-fw':
                require(candidate['rho0'] == .3, 'FraMES momentum prefactor is fixed for this sweep.')
            values.append(tuple((key, candidate[key]) for key in sorted(candidate) if key != 'id'))
        require(len(set(values)) == len(values), 'Duplicate candidate prefactors.')
    require(len(set(identifiers)) == len(identifiers), 'Candidate IDs must be globally unique.')
    c['_config_path'] = str(path)
    return c


def candidates(c):
    return [(method, candidate) for method in METHODS for candidate in c['methods'][method]]


def candidate_at(c, index):
    cells = candidates(c)
    require(type(index) is int and 0 <= index < len(cells), 'Invalid candidate index.')
    return cells[index]


def context(args):
    c = load_config(args.config)
    source = Path(args.source).resolve()
    args.out = str(Path(args.out).resolve())
    reference_path = Path(args.reference).resolve()
    if reference_path.is_dir():
        reference_path /= 'reference.json'
    require(sha256(__file__) == sha256(source / 'extensions/scot_reproduction_20261014/comparison.py'), 'Controller differs from the staged source.')
    reference_module = module(source / 'extensions/scot_reproduction_20261014/reference.py', '_scot_reference_gate')
    reference = reference_module.verify_reference(reference_path)
    require(reference['status'] == 'passed', 'Published reference did not reproduce successfully.')
    prepared = reference['prepared_instance']
    directory = (reference_path.parent / prepared['relative_directory']).resolve()
    manifest_path = directory / 'manifest.json'
    manifest = read(manifest_path)
    require(directory.name == c['instance']['id'] and manifest['n'] == manifest['m'] == 1047, 'Reference data dimensions differ.')
    require(sha256(manifest_path) == prepared['manifest_sha256'], 'Reference manifest changed.')
    require(manifest['array_files_sha256'] == prepared['array_files_sha256'], 'Reference array manifest differs.')
    require({name: sha256(directory / name) for name in pair.ARRAYS} == prepared['array_files_sha256'], 'Reference array files changed.')
    binding = dict(config_sha256=sha256(c['_config_path']), reference_file=str(reference_path),
                   reference_sha256=sha256(reference_path), comparison_source_sha256=sha256(__file__),
                   reference_source_sha256=sha256(source / 'extensions/scot_reproduction_20261014/reference.py'))
    return c, source, directory.parent, prepared, binding


def legacy_config(c, prepared, method, candidate, phase):
    """Supply the existing runner/controller contract without its old dataset policy."""
    horizon_phase = 'tuning' if phase in ('preflight', 'tune') else 'final'
    instance = dict(n=1047, world_size=1, instance_manifest_sha256=prepared['manifest_sha256'],
                    array_files_sha256=prepared['array_files_sha256'], prefactors={method: candidate},
                    preflight_iterations=c['preflight']['iterations'], final_iterations=c[horizon_phase + '_iterations'],
                    final_checkpoint_iterations=c[horizon_phase + '_checkpoint_iterations'])
    return dict(_config_path=c['_config_path'], instances={'scot1047': instance}, pairs_per_call=1024,
                runner=c['runner'], preflight=c['preflight'], run_timeout_seconds=c['run_timeout_seconds'],
                final_marginal_l1_tolerance=c['final_marginal_l1_tolerance'])


def base_directory(out, phase, candidate):
    return Path(out) / phase / candidate['id']


def gate_path(out, candidate):
    return base_directory(out, 'preflight', candidate) / 'gate.json'


def cell_directory(out, phase, method, candidate, seed):
    return pair.directory(base_directory(out, phase, candidate), 'scot1047', method, seed)


def check_cell(c, source, prepared_root, prepared, out, phase, method, candidate, seed, expected_gate):
    lc = legacy_config(c, prepared, method, candidate, phase)
    target = cell_directory(out, phase, method, candidate, seed)
    status = read(target / 'status.json')
    request_path = target / 'request.json'
    request = read(request_path)
    require(status['status'] == 'completed' and status['returncode'] == 0
            and status['request_sha256'] == sha256(request_path), 'Run receipt is incomplete or changed.')
    provenance = pair.identity(lc, source, prepared_root, 'scot1047')
    for key, value in provenance.items():
        require(request[key] == value, 'Run provenance differs: ' + key)
    require(request['gate'] == expected_gate, 'Run launch gate differs.')
    runner_phase = 'preflight' if phase == 'preflight' else 'production'
    expected = pair.expected_config(source, pair.arguments(lc, prepared_root, target / 'output', 'scot1047', method, seed, runner_phase))
    require(request['expected_runner_config'] == expected, 'Requested numerical arguments differ.')
    path = target / 'output/metadata.json'
    require(status['metadata_file'] == str(path.resolve()) and status['metadata_sha256'] == sha256(path), 'Numerical metadata changed.')
    metadata = read(path)
    loss = pair.validate_metadata(metadata, expected, provenance, lc['instances']['scot1047'], c['final_marginal_l1_tolerance'])
    require(status['last_rounded_loss'] == loss, 'Final feasible objective receipt differs.')
    return metadata, status, lc


def verify_gate(c, source, prepared_root, prepared, out, method, candidate, binding):
    path = gate_path(out, candidate)
    gate = read(path)
    require(gate['status'] == 'passed' and gate['binding'] == binding
            and gate['method'] == method and gate['candidate'] == candidate, 'Missing or mismatched candidate preflight.')
    proofs = gate['correctness_checks']
    for name in ('cpu', 'gpu'):
        receipt = proofs[name]
        require(pair.check_receipt(receipt['file'], source, 1 if name == 'gpu' else None) == receipt, 'Correctness receipt changed.')
    launch_gate = dict(binding=binding, stage='preflight', correctness_checks=proofs)
    metadata, status, lc = check_cell(c, source, prepared_root, prepared, out, 'preflight', method, candidate,
                                    c['preflight']['seed'], launch_gate)
    require(gate['calibration'] == status, 'Preflight completion receipt changed.')
    forecast = pair.resource_checks(lc, lc['instances']['scot1047'], metadata, status)
    require(gate['resource_checks'] == forecast, 'Resource forecast changed.')
    return dict(file=str(path.resolve()), sha256=sha256(path))


def preflight(args):
    c, source, prepared_root, prepared, binding = context(args)
    method, candidate = candidate_at(c, args.candidate_index)
    path = gate_path(args.out, candidate)
    path.parent.mkdir(parents=True, exist_ok=False)
    gate = dict(status='running', binding=binding, method=method, candidate=candidate)
    try:
        proofs = dict(cpu=pair.check_receipt(args.cpu_checks, source), gpu=pair.check_receipt(args.gpu_checks, source, 1))
        gate['correctness_checks'] = proofs
        lc = legacy_config(c, prepared, method, candidate, 'preflight')
        metadata, status, _ = pair.run_cell(lc, source, prepared_root, path.parent, 'scot1047', method,
                                           c['preflight']['seed'], 'preflight',
                                           gate_receipt=dict(binding=binding, stage='preflight', correctness_checks=proofs))
        gate.update(status='passed', calibration=status,
                    resource_checks=pair.resource_checks(lc, lc['instances']['scot1047'], metadata, status))
        write(path, gate)
    except Exception as exc:
        gate.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        write(path, gate)
        raise


def tune(args):
    c, source, prepared_root, prepared, binding = context(args)
    method, candidate = candidate_at(c, args.candidate_index)
    receipt = verify_gate(c, source, prepared_root, prepared, args.out, method, candidate, binding)
    lc = legacy_config(c, prepared, method, candidate, 'tune')
    pair.run_cell(lc, source, prepared_root, base_directory(args.out, 'tune', candidate), 'scot1047', method,
                  c['tuning_seed'], 'production', gate_receipt=dict(binding=binding, stage='tune', preflight=receipt))


def tuning_records(c, source, prepared_root, prepared, out, binding):
    records = []
    for method, candidate in candidates(c):
        gate = verify_gate(c, source, prepared_root, prepared, out, method, candidate, binding)
        metadata, status, _ = check_cell(c, source, prepared_root, prepared, out, 'tune', method, candidate,
                                        c['tuning_seed'], dict(binding=binding, stage='tune', preflight=gate))
        target = cell_directory(out, 'tune', method, candidate, c['tuning_seed'])
        records.append(dict(method=method, candidate=candidate, seed=c['tuning_seed'],
                            iterations=c['tuning_iterations'], last_rounded_loss=status['last_rounded_loss'],
                            metadata_file=status['metadata_file'], metadata_sha256=status['metadata_sha256'],
                            status_sha256=sha256(target / 'status.json'), request_sha256=sha256(target / 'request.json'),
                            solver_seconds=metadata['timing']['solver_seconds']))
    return records


def select_candidates(records):
    return {method: min((row for row in records if row['method'] == method),
                        key=lambda row: (row['last_rounded_loss'], row['candidate']['id']))['candidate'] for method in METHODS}


def select(args):
    c, source, prepared_root, prepared, binding = context(args)
    path = Path(args.out) / 'selection.json'
    require(not path.exists(), 'Selection already exists; it cannot be silently replaced.')
    records = tuning_records(c, source, prepared_root, prepared, args.out, binding)
    write(path, dict(status='completed', binding=binding, metric=c['selection_metric'],
                     tie_break='lexicographically smallest candidate ID', tuning_seed=c['tuning_seed'],
                     tuning_iterations=c['tuning_iterations'], candidates=records, selected=select_candidates(records),
                     interpretation='Empirical final feasible loss selects prefactors; independently sampled theorem outputs are retained separately.'))


def verify_selection(c, source, prepared_root, prepared, out, binding):
    path = Path(out) / 'selection.json'
    value = read(path)
    require(value['status'] == 'completed' and value['binding'] == binding
            and value['metric'] == c['selection_metric'] and value['tuning_seed'] == c['tuning_seed']
            and value['tuning_iterations'] == c['tuning_iterations'], 'Selection provenance or criterion differs.')
    expected_cells = {(method, candidate['id']) for method, candidate in candidates(c)}
    require(len(value['candidates']) == len(expected_cells)
            and {(r['method'], r['candidate']['id']) for r in value['candidates']} == expected_cells, 'Selection omitted or duplicated a candidate.')
    # Selection was established by a full metadata audit in its CPU job. Each
    # confirmation cheaply rehashes those immutable receipts instead of parsing
    # all 27 large inner-certificate traces again on an allocated GPU.
    for row in value['candidates']:
        declared = next(v for v in c['methods'][row['method']] if v['id'] == row['candidate']['id'])
        require(row['candidate'] == declared and row['seed'] == c['tuning_seed']
                and row['iterations'] == c['tuning_iterations'], 'Selected candidate evidence differs.')
        target = cell_directory(out, 'tune', row['method'], declared, c['tuning_seed'])
        metadata_path = target / 'output/metadata.json'
        status = read(target / 'status.json')
        require(row['metadata_file'] == str(metadata_path.resolve()) and sha256(metadata_path) == row['metadata_sha256']
                and status['status'] == 'completed' and status['metadata_sha256'] == row['metadata_sha256']
                and status['last_rounded_loss'] == row['last_rounded_loss']
                and sha256(target / 'status.json') == row['status_sha256']
                and sha256(target / 'request.json') == row['request_sha256'] == status['request_sha256'],
                'Tuning evidence changed after selection.')
        finite(row['last_rounded_loss'], 'tuning loss')
    require(value['selected'] == select_candidates(value['candidates']), 'Selected prefactor is not the declared best final feasible loss.')
    return value, dict(file=str(path.resolve()), sha256=sha256(path))


def confirm(args):
    c, source, prepared_root, prepared, binding = context(args)
    cells = [(method, seed) for method in METHODS for seed in c['final_seeds']]
    require(type(args.task_index) is int and 0 <= args.task_index < len(cells), 'Invalid confirmation task index.')
    method, seed = cells[args.task_index]
    selection, selected_receipt = verify_selection(c, source, prepared_root, prepared, args.out, binding)
    candidate = selection['selected'][method]
    gate = verify_gate(c, source, prepared_root, prepared, args.out, method, candidate, binding)
    lc = legacy_config(c, prepared, method, candidate, 'confirm')
    pair.run_cell(lc, source, prepared_root, base_directory(args.out, 'confirm', candidate), 'scot1047', method,
                  seed, 'production', gate_receipt=dict(binding=binding, stage='confirm', preflight=gate, selection=selected_receipt))


def current_trajectory(metadata):
    """Do not count the duplicate last current diagnostic as another observation."""
    points = {}
    for row in metadata['trace']:
        if row['kind'] not in ('initial', 'current_diagnostic', 'last'):
            continue
        k = row['iteration']
        if k in points:
            require(math.isclose(points[k]['rounded_loss'], row['rounded_loss'], rel_tol=0., abs_tol=1e-12), 'Duplicate current losses disagree.')
        points[k] = {key: row[key] for key in ('iteration', 'rounded_loss', 'time_to_this_feasible_output')}
    return [points[k] for k in sorted(points)]


def completed_report(args):
    c, source, prepared_root, prepared, binding = context(args)
    selection, selected_receipt = verify_selection(c, source, prepared_root, prepared, args.out, binding)
    records = []
    for method in METHODS:
        candidate = selection['selected'][method]
        gate = verify_gate(c, source, prepared_root, prepared, args.out, method, candidate, binding)
        for seed in c['final_seeds']:
            record = dict(dataset='scot1047', method=method, method_name=LABELS[method], candidate=candidate,
                          seed=seed, status='missing', initialization='product', oracle='pair-sampling', pairs_per_call=1024)
            target = cell_directory(args.out, 'confirm', method, candidate, seed)
            try:
                if (target / 'status.json').exists() and read(target / 'status.json')['status'] != 'completed':
                    record.update(status='failed', error=read(target / 'status.json').get('error'))
                elif not (target / 'status.json').exists():
                    if target.exists():
                        record['status'] = 'started_without_result'
                else:
                    metadata, status, _ = check_cell(c, source, prepared_root, prepared, args.out, 'confirm', method, candidate,
                                                    seed, dict(binding=binding, stage='confirm', preflight=gate, selection=selected_receipt))
                    final = next(row for row in metadata['trace'] if row['kind'] == 'last')
                    record.update(status='completed', last_rounded_loss=status['last_rounded_loss'],
                                  solver_seconds=metadata['timing']['solver_seconds'],
                                  last_feasible_output_seconds=final['time_to_this_feasible_output'],
                                  total_elapsed_seconds=metadata['timing']['total_elapsed_seconds'],
                                  completed_updates=metadata['counters']['completed_updates'],
                                  stochastic_gradient_calls=metadata['counters']['stochastic_gradient_calls'],
                                  pair_samples=metadata['counters']['pair_samples'],
                                  inner_iterations=metadata['counters']['inner_iterations_total'],
                                  selected_raw_loss=metadata['final_float64_validation']['plans']['selected_raw']['loss'],
                                  selected_index=metadata['output_selection']['index'],
                                  metadata_file=status['metadata_file'], metadata_sha256=status['metadata_sha256'],
                                  observed_current_trajectory=current_trajectory(metadata))
            except Exception as exc:
                record.update(status='invalid', error=repr(exc))
            records.append(record)
    summaries = []
    for method in METHODS:
        complete = [row for row in records if row['method'] == method and row['status'] == 'completed']
        summaries.append(dict(method=method, method_name=LABELS[method], completed_seeds=[row['seed'] for row in complete],
                              medians={key: statistics.median(row[key] for row in complete)
                                       for key in ('last_rounded_loss', 'solver_seconds', 'last_feasible_output_seconds', 'total_elapsed_seconds')}
                              if len(complete) == len(c['final_seeds']) else None))
    counts = {status: sum(row['status'] == status for row in records)
              for status in ('completed', 'missing', 'started_without_result', 'failed', 'invalid')}
    result = dict(status='completed' if counts['completed'] == 9 else 'incomplete', binding=binding,
                  selection=selected_receipt, selected_prefactors=selection['selected'], counts=counts, runs=records, summaries=summaries,
                  scope='Nine prefactors per method, one development seed, then three independent confirmation seeds; all runs use 20000 updates and 1024 iid coupling pairs per stochastic gradient.',
                  objective='Full squared GW discrepancy without a factor one-half or entropy. Stochastic updates and online diagnostics cast the common reference arrays to float32; independent final validation reloads the original float64 arrays and promotes the saved float32 plans.',
                  timing='Checkpoint times include solver work and the current feasibility repair, excluding earlier repairs, diagnostics, logging and serialization. Final total_elapsed_seconds is reported separately.',
                  caveats='Finite 1e-7 proximal accuracy is a numerical approximation, not an inexact outer convergence theorem. Empirical last/current losses and independent random theorem outputs are distinct.')
    write(Path(args.out) / 'report.json', result)
    fields = ['dataset', 'method', 'method_name', 'seed', 'status', 'completed_updates', 'last_rounded_loss',
              'solver_seconds', 'last_feasible_output_seconds', 'total_elapsed_seconds', 'pair_samples', 'inner_iterations', 'error']
    with (Path(args.out) / 'report.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    with (Path(args.out) / 'current_trajectories.csv').open('w', newline='') as stream:
        fields = ['dataset', 'method', 'seed', 'iteration', 'rounded_loss', 'time_to_this_feasible_output']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            for point in row.get('observed_current_trajectory', []):
                writer.writerow(dict(dataset=row['dataset'], method=row['method'], seed=row['seed'], **point))
    print(json.dumps(dict(status=result['status'], counts=counts)), flush=True)


def report(args):
    """An after-any report must also describe a blocked prerequisite chain."""
    try:
        completed_report(args)
    except Exception as exc:
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        evidence = []
        for pattern in ('preflight/*/gate.json', 'tune/*/scot1047/*/seed*/status.json',
                        'confirm/*/scot1047/*/seed*/status.json'):
            for path in sorted(out.glob(pattern)):
                try:
                    value = read(path)
                    evidence.append(dict(file=str(path), sha256=sha256(path), status=value.get('status'), error=value.get('error')))
                except Exception as receipt_error:
                    evidence.append(dict(file=str(path), status='unreadable', error=repr(receipt_error)))
        reference = Path(args.reference).resolve()
        if reference.is_dir():
            reference /= 'reference.json'
        result = dict(status='blocked_dependency', error=repr(exc), traceback=traceback.format_exc(),
                      config_file=str(Path(args.config).resolve()), reference_file=str(reference),
                      reference_sha256=sha256(reference) if reference.is_file() else None,
                      selection_file=str(out / 'selection.json'), stage_receipts=evidence, runs=[], summaries=[],
                      interpretation='No comparison is asserted because a prerequisite or audit failed. Existing numerical results and failure receipts remain untouched.')
        write(out / 'report.json', result)
        print(json.dumps(dict(status=result['status'], error=result['error'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    for name in ('preflight', 'tune', 'select', 'confirm', 'report'):
        command = commands.add_parser(name)
        for field in ('config', 'source', 'reference', 'out'):
            command.add_argument('--' + field, required=True)
        if name in ('preflight', 'tune'):
            command.add_argument('--candidate-index', type=int, required=True)
        if name == 'preflight':
            command.add_argument('--cpu-checks', required=True)
            command.add_argument('--gpu-checks', required=True)
        if name == 'confirm':
            command.add_argument('--task-index', type=int, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Comparison execution and substantial receipt processing require a Slurm allocation.')
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
