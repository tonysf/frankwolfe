"""Parallel local FP64 SMD/MSBPG comparison on the verified SCOT geometry.

A separate specification freezes four prefactors per method before tuning.
Four physical-core leases permit parallel work; elapsed times are descriptive,
not benchmark comparisons. --detach starts one bounded wave without polling.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import functools
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'experiments/scot_cpu_methods_fp64_20261014'
METHODS = ('published-smd', 'published-msbpg')
CHECKPOINTS = [1, 10, 50, 100, 200, 500, 1000, 1500, 2000]
TIME_CHECKPOINTS = ''
SEEDS = [811, 812, 813]
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
PLANS = {'selected_raw', 'selected_rounded', 'last_raw', 'last_rounded'}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prior = load_module(ROOT / 'extensions/scot_cpu_fp64_20261014/compare_frames.py', '_scot_methods_prior')
read, write_json, sha256, require, create_json = prior.read, prior.write_json, prior.sha256, prior.require, prior.create_json


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@functools.lru_cache(maxsize=1)
def runner():
    return load_module(ROOT / 'extensions/pair_sampling_20261011/run_pair.py', '_scot_methods_runner')


def source_identity(source_commit=None):
    if source_commit is None:
        source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(source_commit=source_commit,
                controller_sha256=sha256(__file__), source_file_sha256=runner().source_hashes(),
                helper_source_sha256={name: sha256(ROOT / 'extensions' / name) for name in (
                    'scot_cpu_fp64_20261014/compare_frames.py', 'scot_reproduction_20261014/reference.py',
                    'msbpg_20261005/campaign.py')})


def inputs(out):
    reference, binding = prior.inputs(out)
    frames = read(out / 'frames_result/completed.json')
    config = read(out / 'frames_result/local_config.json')
    require(frames['status'] == 'completed' and frames['config'] == config, 'Frozen FraMES comparison is incomplete.')
    require(config['optimizer_dtype'] == 'float64' and config['fixed_horizon'] == 2000
            and config['pairs_per_call'] == 1024 and len(config['candidates']) == 4
            and config['tuning_seed'] == 810 and config['confirmation_seeds'] == SEEDS, 'FraMES comparison budget/precision differs.')
    for key, value in binding.items():
        require(config[key] == value, 'Frozen FraMES reference/input binding differs: ' + key)
    require({row['seed'] for row in frames['confirmation']} == set(SEEDS)
            and len(frames['confirmation']) == 3, 'Frozen FraMES confirmation seeds differ.')
    binding.update(frames_completed_sha256=sha256(out / 'frames_result/completed.json'),
                   frames_config_sha256=sha256(out / 'frames_result/local_config.json'))
    return reference, binding


def physical_cores():
    """Choose one allowed logical CPU per actual physical core, on this host."""
    selected = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (int((directory / 'physical_package_id').read_text()), int((directory / 'core_id').read_text()))
        selected.setdefault(key, cpu)
    require(len(selected) >= 4, 'Four distinct physical cores are not available in this CPU affinity.')
    return [dict(cpu_id=cpu, package_id=key[0], core_id=key[1]) for key, cpu in list(selected.items())[:4]]


def validate_spec(spec):
    require(spec['schema_version'] == 1 and set(spec['methods']) == set(METHODS), 'Specification must include both published methods.')
    require(spec['workers'] == 4 and 0 < spec['run_timeout_seconds'] <= 23 * 3600, 'Invalid worker count or bounded runtime.')
    identifiers = []
    fields = {'published-smd': {'id', 'prefactor_fraction'},
              'published-msbpg': {'id', 'coupling_fraction', 'momentum_prefactor'}}
    for method in METHODS:
        candidates = spec['methods'][method]
        require(len(candidates) == 4, 'Exactly four prefactors per method must match the FraMES tuning budget.')
        seen = []
        for candidate in candidates:
            require(set(candidate) == fields[method], 'Unexpected prefactor or exponent override.')
            require(isinstance(candidate['id'], str) and candidate['id'] and all(ch.isalnum() or ch in '_-' for ch in candidate['id']), 'Invalid candidate identifier.')
            identifiers.append(candidate['id'])
            for key, value in candidate.items():
                if key != 'id':
                    require(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and 0 < value <= 1, 'Prefactor violates its theorem bound.')
            seen.append(tuple((k, candidate[k]) for k in sorted(candidate) if k != 'id'))
        require(len(set(seen)) == 4, 'Duplicate prefactor candidate.')
    require(len(set(identifiers)) == 8, 'Candidate IDs must be globally unique.')
    return spec


def init(out, specification):
    spec = validate_spec(read(specification))
    reference, binding = inputs(out)
    require(shutil.which('taskset') is not None, 'CPU affinity launcher taskset is unavailable.')
    config = dict(created_at=now(), specification=spec, specification_sha256=sha256(specification),
                  methods=spec['methods'], workers=4, run_timeout_seconds=spec['run_timeout_seconds'],
                  fixed_horizon=2000, tuning_seed=810, confirmation_seeds=SEEDS, pairs_per_call=1024,
                  selected_metric='runner_recomputed_float64_last_rounded_loss_at_fixed_horizon',
                  selection_evaluation='The runner reloads the original float64 arrays and reevaluates the saved last-rounded plan with its separate Torch FP64 validator. The final report checks these endpoint objectives and the selected winner using NumPy/POT; selection itself does not use a separate numerical library.',
                  optimizer_dtype='float64', final_evaluation_dtype='float64', initialization='uniform product',
                  inner_tol=1e-7, inner_gap_tol=1e-7, inner_max=10000, check_every=10,
                  checkpoint_iterations=CHECKPOINTS, checkpoint_seconds=TIME_CHECKPOINTS,
                  threads_per_process=1, thread_environment=THREAD_ENV, physical_cores=physical_cores(),
                  reference_loss=reference['rounded']['loss'], **binding, **source_identity(),
                  theory_scope='Published SMD step and MSBPG coupled step/momentum schedules are unchanged. Only theorem-admissible prefactors are tuned; finite 1e-7 proximal tolerances do not assert an inexact outer theorem.',
                  timing_scope='Up to four one-thread processes overlap. Runtime values describe these executions and are not used for method speed rankings.')
    create_json(out / 'local_config.json', config)
    (out / 'runs').mkdir(exist_ok=True)
    (out / 'core_leases').mkdir(exist_ok=True)
    return dict(config_sha256=sha256(out / 'local_config.json'))


def configuration(out):
    config = read(out / 'local_config.json')
    spec = validate_spec(config['specification'])
    require(config['methods'] == spec['methods'] and config['workers'] == spec['workers']
            and config['run_timeout_seconds'] == spec['run_timeout_seconds'], 'Frozen specification changed.')
    _, binding = inputs(out)
    for key, value in dict(binding, **source_identity(config['source_commit'])).items():
        require(config[key] == value, 'Frozen source/input binding changed: ' + key)
    require(config['optimizer_dtype'] == config['final_evaluation_dtype'] == 'float64'
            and config['fixed_horizon'] == 2000 and config['pairs_per_call'] == 1024
            and config['tuning_seed'] == 810 and config['confirmation_seeds'] == SEEDS
            and config['inner_tol'] == config['inner_gap_tol'] == 1e-7
            and config['inner_max'] == 10000 and config['check_every'] == 10
            and config['checkpoint_iterations'] == CHECKPOINTS and config['checkpoint_seconds'] == TIME_CHECKPOINTS
            and config['threads_per_process'] == 1 and config['thread_environment'] == THREAD_ENV
            and config['initialization'] == 'uniform product' and config['physical_cores'] == physical_cores(), 'Frozen numerical or execution policy changed.')
    return config, binding


def name(phase, candidate, seed):
    return f'{phase}_{candidate["id"]}_seed{seed}'


def directory(out, phase, candidate, seed):
    return out / 'runs' / name(phase, candidate, seed)


def arguments(out, phase, method, candidate, seed):
    values = dict(instance=str(out / 'reference/prepared/scot1047'), out=str(directory(out, phase, candidate, seed)),
                  method=method, device='cpu', dtype='float64', iterations=2000, seed=seed,
                  output_seed=seed + 20000, init_seed=seed + 10000, initialization='product',
                  oracle='pair-sampling', pairs_per_call=1024, save_plans='all', progress_every=1000,
                  checkpoint_iterations=','.join(map(str, CHECKPOINTS)), checkpoints=TIME_CHECKPOINTS,
                  inner_tol=1e-7, inner_gap_tol=1e-7, inner_max=10000, check_every=10)
    values.update({key: value for key, value in candidate.items() if key != 'id'})
    return [word for key, value in values.items() for word in ('--' + key.replace('_', '-'), str(value))]


@contextlib.contextmanager
def core_lease(out, config):
    deadline = time.monotonic() + config['run_timeout_seconds']
    while True:
        for core in config['physical_cores']:
            stream = (out / 'core_leases' / f"cpu{core['cpu_id']}.lock").open('a+')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield core
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
                stream.close()
            return
        require(time.monotonic() < deadline, 'Timed out waiting for one of four physical-core slots.')
        time.sleep(1.)


def audit_numerics(metadata, expected, binding):
    require(metadata['status'] == 'completed' and metadata['method_key'] == expected['method']
            and metadata['config'] == expected, 'Runner did not complete the exact declared method/configuration.')
    require(metadata['source_file_sha256'] == runner().source_hashes(), 'Numerical source hash mismatch.')
    for key in ('array_files_sha256', 'instance_manifest_sha256'):
        require(metadata[key] == binding[key], 'Numerical input hash mismatch: ' + key)
    require(metadata['dimensions'] == dict(source=1047, target=1047), 'Wrong dataset dimensions.')
    env, counters = metadata['environment'], metadata['counters']
    require(env['world_size'] == 1 and env['deterministic_algorithms'] is True
            and len(metadata['ranks']) == 1 and metadata['ranks'][0]['device'] == 'cpu', 'Incorrect CPU execution topology.')
    horizon = expected['iterations']
    require(counters['completed_updates'] == counters['stochastic_gradient_calls'] == horizon
            and counters['pair_samples'] == horizon * 1024 and counters['sketch_probes'] == 0
            and counters['initialization_oracle_calls'] == int(expected['method'] == 'published-msbpg')
            and counters['inner_stopping_failures'] == 0, 'Wrong oracle budget or failed inner solve.')
    oracle = metadata['gradient_oracle']
    require(oracle['name'] == 'pair-sampling' and oracle['pair_samples_per_call'] == 1024
            and oracle['probes_per_call'] == 0 and oracle['target_marginals_substituted'] is False, 'Incorrect stochastic oracle.')
    streams = metadata['random_streams']
    require(streams['pairs']['seed'] == expected['seed'] and streams['pairs']['with_replacement'] is True
            and streams['output']['seed'] == expected['output_seed'] and streams['output']['independent_generator'] is True
            and streams['initialization']['seed'] == expected['init_seed'], 'Random streams differ.')
    trace = metadata['inner_trace']
    require(len(trace) == horizon, 'Missing inner-certificate records.')
    for k, row in enumerate(trace):
        require(row['iteration'] == k and row['converged'] is True and row['proximal_optimality_certified'] is True
                and row['certificate_valid'] is True and 0 < row['inner_iterations'] <= 10000, 'Uncertified or over-budget proximal solve.')
        for field, tolerance in [('marginal_l1', 1e-7), ('normalized_gap_upper_bound', 1e-7), ('primal_witness_marginal_l1', 1e-12)]:
            require(math.isfinite(row[field]) and 0 <= row[field] <= tolerance, 'Inner acceptance failed: ' + field)
        require(math.isfinite(row['normalized_primal_dual_gap']) and row['normalized_primal_dual_gap'] <= 1e-7, 'Raw inner gap failed.')
    require(sum(row['inner_iterations'] for row in trace) == counters['inner_iterations_total']
            and sum(row['certificate_checks'] for row in trace) == counters['certificate_checks'], 'Inner work counters disagree.')
    if expected['method'] == 'published-msbpg':
        theorem = load_module(ROOT / 'extensions/msbpg_20261005/campaign.py', '_scot_cpu_msbpg_scalar_audit')
        theorem.validate_theory_metadata(metadata, expected)
    else:
        bound = metadata['cmax']
        require(math.isfinite(bound) and bound >= 0, 'Invalid SMD objective cost bound.')
        for k, row in enumerate(trace):
            eta = expected['prefactor_fraction'] * (.25 / bound if bound else 1.) / math.sqrt(k + 1)
            require(math.isclose(row['eta'], eta, rel_tol=1e-12)
                    and row['prefactor_fraction'] == expected['prefactor_fraction'], 'SMD published step prescription differs.')
    selection = metadata['output_selection']
    require(selection['horizon'] == horizon and 0 <= selection['index'] < horizon and selection['last_iterate_index'] == horizon
            and selection['independent_rng'] is True and selection['output_seed'] == expected['output_seed'], 'Invalid theorem output selection.')
    probability = (selection['index'] + 1)**-.5 / sum((k + 1)**-.5 for k in range(horizon))
    require(math.isclose(selection['probability'], probability, rel_tol=1e-12), 'Theorem output weight differs.')
    saved = metadata['ranks'][0]['saved_plans']
    require(set(saved) == PLANS, 'Missing saved plans.')
    for row in saved.values():
        require(row['dtype'] == 'float64' and row['shape'] == [1047, 1047] and row['persisted'] is True
                and row['array_bytes'] == 1047 * 1047 * 8 and row['file_sha256'] == sha256(row['file']), 'Saved FP64 output differs.')
    final = metadata['final_float64_validation']
    require(final['dtype'] == 'float64' and final['exact_objective'] is True and set(final['plans']) == PLANS, 'Missing separate FP64 final validation.')
    for key, row in final['plans'].items():
        require(math.isfinite(row['loss']) and row['marginal_residuals']['min_entry'] >= 0, 'Nonfinite or negative output.')
        if key.endswith('_rounded'):
            require(row['marginal_residuals']['both_l1'] <= 100 * 2**-52, 'Rounded FP64 output is infeasible.')
    return final['plans']['last_rounded']['loss']


def validate_run(out, config, binding, phase, method, candidate, seed, receipt):
    path = directory(out, phase, candidate, seed) / 'metadata.json'
    require(receipt['exit_code'] == 0 and receipt['config_sha256'] == sha256(out / 'local_config.json')
            and receipt['input_binding'] == binding and receipt['source_identity'] == source_identity(config['source_commit'])
            and receipt['thread_environment'] == THREAD_ENV and receipt['cpu_core'] in config['physical_cores']
            and receipt['metadata_sha256'] == sha256(path), 'Controller completion provenance differs.')
    expected_args = arguments(out, phase, method, candidate, seed)
    expected_command = ['taskset', '-c', str(receipt['cpu_core']['cpu_id']), sys.executable,
                        str(ROOT / 'extensions/pair_sampling_20261011/run_pair.py')] + expected_args
    require(receipt['command'] == expected_command, 'Executed numerical command differs.')
    metadata = read(path)
    expected = vars(runner().parser().parse_args(expected_args))
    loss = audit_numerics(metadata, expected, binding)
    return dict(method=method, candidate=candidate, seed=seed, directory=str(path.parent),
                last_rounded_loss_float64=loss, metadata_sha256=sha256(path), receipt=receipt)


def execute(out, config, binding, phase, method, candidate, seed):
    label = name(phase, candidate, seed)
    receipt_path = out / f'{label}_receipt.json'
    require(not directory(out, phase, candidate, seed).exists(), 'Refusing to reuse a numerical output.')
    receipt = dict(status='waiting_for_core', name=label, method=method, candidate=candidate, seed=seed,
                   created_utc=now(), config_sha256=sha256(out / 'local_config.json'), input_binding=binding,
                   source_identity=source_identity(config['source_commit']), thread_environment=THREAD_ENV,
                   timing_scope='Parallel local execution; not a benchmark timing comparison.')
    create_json(receipt_path, receipt)
    start = time.monotonic()
    try:
        with core_lease(out, config) as core:
            cmd = ['taskset', '-c', str(core['cpu_id']), sys.executable,
                   str(ROOT / 'extensions/pair_sampling_20261011/run_pair.py')] + arguments(out, phase, method, candidate, seed)
            receipt.update(status='running', cpu_core=core, command=cmd, started_utc=now())
            write_json(receipt_path, receipt)
            with (out / f'{label}.log').open('x') as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, **THREAD_ENV), stdout=log, stderr=subprocess.STDOUT)
                receipt['pid'] = process.pid
                write_json(receipt_path, receipt)
                try:
                    code = process.wait(timeout=config['run_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise
            receipt.update(exit_code=code, finished_utc=now(), process_elapsed_seconds=time.monotonic() - start)
            require(code == 0, f'{label} failed with exit {code}; see preserved log.')
            receipt['metadata_sha256'] = sha256(directory(out, phase, candidate, seed) / 'metadata.json')
            value = validate_run(out, config, binding, phase, method, candidate, seed, receipt)
            receipt['status'] = 'completed'
            write_json(receipt_path, receipt)
            print(json.dumps(dict(name=label, status='completed', loss=value['last_rounded_loss_float64'])), flush=True)
            return value
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), finished_utc=now())
        write_json(receipt_path, receipt)
        raise


def audited(out, config, binding, phase, method, candidate, seed):
    receipt = read(out / f'{name(phase, candidate, seed)}_receipt.json')
    require(receipt['status'] == 'completed', 'Run has not completed successfully.')
    return validate_run(out, config, binding, phase, method, candidate, seed, receipt)


def tune_cells(config):
    return [(method, candidate, 810) for method in METHODS for candidate in config['methods'][method]]


def best(tuned):
    return {method: min((row for row in tuned if row['method'] == method),
                        key=lambda row: row['last_rounded_loss_float64'])['candidate'] for method in METHODS}


def choose(out, config, binding):
    tuned = [audited(out, config, binding, 'tune', *cell) for cell in tune_cells(config)]
    result = dict(status='completed', config_sha256=sha256(out / 'local_config.json'),
                  selection_metric=config['selected_metric'], selected=best(tuned), tuning=tuned)
    create_json(out / 'selection.json', result)
    return result


def selected(out, config, binding):
    selection = read(out / 'selection.json')
    tuned = [audited(out, config, binding, 'tune', *cell) for cell in tune_cells(config)]
    require(selection['status'] == 'completed' and selection['config_sha256'] == sha256(out / 'local_config.json')
            and selection['selection_metric'] == config['selected_metric'] and selection['tuning'] == tuned
            and selection['selected'] == best(tuned), 'Prefactor selection is incomplete or changed.')
    return selection


def wave(args, out, config, binding):
    selection = selected(out, config, binding) if args.phase == 'confirm' else None
    cells = ([(method, selection['selected'][method], seed) for method in METHODS for seed in SEEDS]
             if selection else tune_cells(config))
    if args.method:
        cells = [cell for cell in cells if cell[0] == args.method]
    stop = len(cells) if args.stop is None else args.stop
    require(0 <= args.start < stop <= len(cells), 'Invalid wave interval.')
    chosen = cells[args.start:stop]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(chosen))) as pool:
        runs = list(pool.map(lambda cell: execute(out, config, binding, args.phase, *cell), chosen))
    return dict(runs=runs, selection_sha256=sha256(out / 'selection.json') if selection else None)


def finalize(out, config, binding):
    selection = selected(out, config, binding)
    confirmed = [audited(out, config, binding, 'confirm', method, selection['selected'][method], seed)
                 for method in METHODS for seed in SEEDS]
    value = dict(status='completed', config=config, tuning=selection['tuning'], selected=selection['selected'],
                 confirmation=confirmed, selection_sha256=sha256(out / 'selection.json'))
    create_json(out / 'completed.json', value)
    return dict(completed_file=str(out / 'completed.json'))


def stage_label(args):
    if args.phase in ('tune', 'confirm'):
        return f'{args.phase}_{args.method or "both"}_{args.start}_{args.stop if args.stop is not None else "end"}'
    return args.phase


def durable_stage(out, label, operation):
    path = out / f'stage_{label}.json'
    receipt = dict(status='running', stage=label, pid=os.getpid(), started_utc=now(), controller_sha256=sha256(__file__))
    create_json(path, receipt)
    try:
        result = operation()
        receipt.update(status='completed', result=result, finished_utc=now())
        write_json(path, receipt)
        print(json.dumps(dict(stage=label, status='completed')), flush=True)
        return result
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), finished_utc=now())
        write_json(path, receipt)
        raise


def campaign(out, specification):
    require(specification is not None, '--spec is required for a new end-to-end campaign.')
    durable_stage(out, 'init', lambda: init(out, specification))
    config, binding = configuration(out)
    for phase in ('tune', 'confirm'):
        args = SimpleNamespace(phase=phase, method=None, start=0, stop=None)
        durable_stage(out, stage_label(args), lambda: wave(args, out, config, binding))
        if phase == 'tune':
            durable_stage(out, 'select', lambda: choose(out, config, binding))
    durable_stage(out, 'finalize', lambda: finalize(out, config, binding))

    def render_report():
        script = ROOT / 'extensions/scot_cpu_methods_fp64_20261014/summarize.py'
        script_sha256 = sha256(script)
        command = [sys.executable, str(script), '--root', str(out)]
        with (out / 'report.log').open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
        require(result.returncode == 0, f'Final report failed with exit {result.returncode}; see report.log.')
        return dict(command=command, exit_code=result.returncode, log=str(out / 'report.log'), reporter_sha256=script_sha256)

    durable_stage(out, 'report', render_report)
    return dict(completed_file=str(out / 'completed.json'), report_log=str(out / 'report.log'))


def detach(args, argv):
    require(args.phase in ('tune', 'confirm', 'campaign'), 'Only numerical waves or the end-to-end campaign may be detached.')
    out = args.out.resolve()
    label = stage_label(args)
    path = out / f'detached_{label}.json'
    receipt = dict(status='launching', created_utc=now(), controller_sha256=sha256(__file__))
    create_json(path, receipt)
    command = [sys.executable, str(Path(__file__).resolve())] + [word for word in argv if word != '--detach']
    try:
        with (out / f'detached_{label}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, **THREAD_ENV), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt.update(status='launched', pid=process.pid, command=command, log=str(out / f'detached_{label}.log'))
        write_json(path, receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as exc:
        receipt.update(status='failed_to_launch', error=repr(exc))
        write_json(path, receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('init', 'tune', 'select', 'confirm', 'finalize', 'campaign'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_dir(), 'Prepare the new experiment input links before running.')
    if args.detach:
        detach(args, sys.argv[1:])
        return
    def operation():
        if args.phase == 'campaign':
            return campaign(out, args.spec)
        if args.phase == 'init':
            require(args.spec is not None, '--spec is required to freeze the predeclared candidate grid.')
            return init(out, args.spec)
        else:
            config, binding = configuration(out)
            return (wave(args, out, config, binding) if args.phase in ('tune', 'confirm') else
                    choose(out, config, binding) if args.phase == 'select' else finalize(out, config, binding))
    success = False
    try:
        durable_stage(out, stage_label(args), operation)
        success = True
    finally:
        detached_path = out / f'detached_{stage_label(args)}.json'
        if detached_path.exists():
            receipt = read(detached_path)
            if receipt.get('pid') == os.getpid():
                receipt.update(status='completed' if success else 'failed', exit_code=0 if success else 1,
                               finished_utc=now(), stage_receipt=str(out / f'stage_{stage_label(args)}.json'))
                write_json(detached_path, receipt)


if __name__ == '__main__':
    main()
