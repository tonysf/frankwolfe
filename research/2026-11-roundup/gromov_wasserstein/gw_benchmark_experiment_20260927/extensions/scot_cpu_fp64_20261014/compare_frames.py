"""Staged FP64 replay of the fixed local SCOT FraMES comparison.

The old FP32 experiment stays immutable. Tuning uses two one-thread workers;
fresh-seed confirmations run individually and must be launched sequentially.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'experiments/scot_cpu_fp64_20261014'
SOURCE_COMMIT = '731fdadc553da493baca2e323b62d571cda463e0'
CANDIDATES = [dict(id='fw04', gamma0=.3, rho0=.3, mu_scale=1.),
              dict(id='fw05', gamma0=.3, rho0=.3, mu_scale=10.),
              dict(id='fw07', gamma0=1., rho0=.3, mu_scale=1.),
              dict(id='fw08', gamma0=1., rho0=.3, mu_scale=10.)]
HORIZON = 2000
CHECKPOINTS = [1, 10, 50, 100, 200, 500, 1000, 1500, 2000]
TIME_CHECKPOINTS = '30,60,120,180,240,300,600'
SEEDS = [811, 812, 813]
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


reference = load_module(ROOT / 'extensions/scot_reproduction_20261014/reference.py', '_scot_fp64_reference')
write_json, sha256 = reference.write_json, reference.sha256


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def create_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def runner():
    return load_module(ROOT / 'extensions/pair_sampling_20261011/run_pair.py', '_scot_fp64_pair_runner')


def inputs(out):
    require(out.is_dir(), 'Prepare the new experiment input directory before launching.')
    record = reference.verify_reference(out / 'reference')
    reference_path = out / 'reference/reference.json'
    baseline_path = out / 'baseline_local/replay.json'
    baseline = read(baseline_path)
    require(baseline['status'] == 'passed' and baseline['gate']['passed'], 'The reused local FP64 baseline did not pass.')
    require(baseline['reference_sha256'] == sha256(reference_path), 'Baseline reference hash differs.')
    for key in ('manifest_sha256', 'array_files_sha256'):
        require(baseline['prepared_instance'][key] == record['prepared_instance'][key], 'Baseline geometry differs: ' + key)
    require(baseline['solver']['effective_overrides'] == dict(epsilon=.001, loss_fun='square_loss', log=True, verbose=False), 'Baseline solver settings changed.')
    require(baseline['solver']['backend'] == 'NumPy CPU, float64 input arrays', 'Baseline precision differs.')
    require(all(baseline['environment']['thread_environment'][key] == value for key, value in THREAD_ENV.items()), 'Baseline thread settings differ.')
    for name, key in [('P_raw.npy', 'raw_plan_sha256'), ('P_rounded.npy', 'rounded_plan_sha256'), ('pot_log.json', 'pot_log_sha256')]:
        require(sha256(out / 'baseline_local' / name) == baseline[key], 'Baseline artifact changed: ' + name)
    for name, expected in baseline['source_file_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Invalid archived baseline source path.')
        require(sha256(out / 'baseline_local' / path) == expected, 'Archived baseline source changed.')
    return record, dict(reference_sha256=sha256(reference_path), baseline_sha256=sha256(baseline_path),
                        instance_manifest_sha256=record['prepared_instance']['manifest_sha256'],
                        array_files_sha256=record['prepared_instance']['array_files_sha256'])


def identity():
    actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(actual_commit == SOURCE_COMMIT, 'Repository HEAD differs from the declared source commit.')
    return dict(source_commit=SOURCE_COMMIT, controller_sha256=sha256(__file__),
                source_file_sha256=runner().source_hashes())


def prescribed_config(record, binding):
    return dict(fixed_horizon=HORIZON, candidates=CANDIDATES, tuning_seed=810,
                confirmation_seeds=SEEDS, pairs_per_call=1024,
                selected_metric='independent_float64_last_rounded_loss_at_fixed_horizon',
                threads_per_process=1, tuning_workers=2, confirmation_workers=1,
                optimizer_dtype='float64', final_evaluation_dtype='float64', initialization='uniform product',
                checkpoint_iterations=CHECKPOINTS, checkpoint_seconds=TIME_CHECKPOINTS,
                selection_scope='Same four candidates as the prior local FP32 comparison; only optimization dtype changes to float64.',
                theory_scope='Unchanged decay exponents .75/.5/.25; empirical prefactor tuning. Final loss is not a theorem-output guarantee.',
                timing_scope='Tuning timings excluded from CPU comparisons because two runs overlap. Confirmation stages must run serially, one thread each; a controller lock prevents overlap. Solver+current repair time and end-to-end time remain distinct.',
                reference_loss=record['rounded']['loss'], baseline_dtype='float64',
                **binding, **identity())


def frozen_config(out, *, create=False):
    record, binding = inputs(out)
    expected = prescribed_config(record, binding)
    path = out / 'local_config.json'
    if create:
        create_json(path, dict(created_at=now(), **expected))
    value = read(path)
    require({key: value[key] for key in expected} == expected and set(value) == set(expected) | {'created_at'}, 'Frozen FP64 configuration, sources or input hashes changed.')
    return value


def run_directory(out, phase, candidate, seed):
    return out / 'pilot_fw04_fp64' if phase == 'pilot' else out / 'runs' / f'{phase}_{candidate["id"]}_seed{seed}'


def run_name(phase, candidate, seed):
    return 'pilot_fw04_fp64' if phase == 'pilot' else f'{phase}_{candidate["id"]}_seed{seed}'


def command(out, phase, candidate, seed):
    pilot = phase == 'pilot'
    return [sys.executable, str(ROOT / 'extensions/pair_sampling_20261011/run_pair.py'),
            '--instance', str(out / 'reference/prepared/scot1047'), '--out', str(run_directory(out, phase, candidate, seed)),
            '--method', 'smoothing-fw', '--device', 'cpu', '--dtype', 'float64',
            '--iterations', str(200 if pilot else HORIZON), '--seed', str(seed),
            '--output-seed', str(seed + 20000), '--init-seed', str(seed + 10000),
            '--initialization', 'product', '--oracle', 'pair-sampling', '--pairs-per-call', '1024', '--save-plans', 'all',
            '--gamma0', str(candidate['gamma0']), '--rho0', str(candidate['rho0']), '--mu-scale', str(candidate['mu_scale']),
            '--progress-every', '200' if pilot else '1000',
            '--checkpoint-iterations', ','.join(map(str, [1, 10, 50, 100, 200] if pilot else CHECKPOINTS)),
            '--checkpoints', '' if pilot else TIME_CHECKPOINTS]


def validate(out, phase, candidate, seed, receipt, binding):
    path = run_directory(out, phase, candidate, seed) / 'metadata.json'
    metadata = read(path)
    numerical = runner()
    expected = vars(numerical.parser().parse_args(command(out, phase, candidate, seed)[2:]))
    require(metadata['status'] == 'completed' and metadata['config'] == expected, 'Completed runner arguments differ from the FP64 prescription.')
    require(metadata['source_file_sha256'] == numerical.source_hashes(), 'Numerical sources changed.')
    for key in ('array_files_sha256', 'instance_manifest_sha256'):
        require(metadata[key] == binding[key], 'Run input hash differs: ' + key)
    count = metadata['counters']
    horizon = 200 if phase == 'pilot' else HORIZON
    require(count['completed_updates'] == count['stochastic_gradient_calls'] == horizon
            and count['pair_samples'] == horizon * 1024 and count['sketch_probes'] == 0, 'Incomplete run or incorrect oracle work.')
    require(count['inner_iterations_total'] == count['inner_stopping_failures'] == 0 and not metadata['inner_trace'], 'Unexpected FraMES inner solver work.')
    require(metadata['environment']['world_size'] == 1 and metadata['environment']['deterministic_algorithms'] is True
            and len(metadata['ranks']) == 1 and metadata['ranks'][0]['device'] == 'cpu', 'Wrong execution device or determinism.')
    schedule = metadata['schedule']
    require((schedule['gamma_exponent'], schedule['rho_exponent'], schedule['mu_exponent']) == (.75, .5, .25), 'Theory exponents changed.')
    expected_plans = {'selected_raw', 'selected_rounded', 'last_raw', 'last_rounded'}
    require(set(metadata['ranks'][0]['saved_plans']) == expected_plans, 'Missing saved output plan.')
    for key, plan in metadata['ranks'][0]['saved_plans'].items():
        require(plan['dtype'] == 'float64' and plan['persisted'] is True and plan['shape'] == [1047, 1047], 'Saved optimization output is not the full FP64 plan.')
        require(plan['file_sha256'] == sha256(Path(plan['file'])), 'Saved plan checksum mismatch.')
    validation = metadata['final_float64_validation']
    require(validation['dtype'] == 'float64' and validation['exact_objective'] is True
            and set(validation['plans']) == expected_plans, 'Missing independent final FP64 validation.')
    for name, plan in validation['plans'].items():
        require(math.isfinite(plan['loss']) and plan['marginal_residuals']['min_entry'] >= 0, 'Invalid final plan/loss.')
        if name.endswith('_rounded'):
            require(plan['marginal_residuals']['both_l1'] <= 100 * 2**-52, 'FP64 rounded output is infeasible.')
    require(receipt['exit_code'] == 0 and receipt['command'] == command(out, phase, candidate, seed)
            and receipt['optimizer_dtype'] == 'float64' and receipt['thread_environment'] == THREAD_ENV, 'Controller completion receipt differs.')
    require(receipt['input_binding'] == binding and receipt['source_identity'] == identity(), 'Controller source/input binding changed.')
    require(receipt.get('metadata_sha256', sha256(path)) == sha256(path), 'Numerical metadata changed.')
    loss = validation['plans']['last_rounded']['loss']
    return dict(candidate=candidate, seed=seed, directory=str(path.parent),
                last_rounded_loss_float64=loss, metadata_sha256=sha256(path), receipt=receipt)


def execute(out, phase, candidate, seed, binding):
    name = run_name(phase, candidate, seed)
    output = run_directory(out, phase, candidate, seed)
    receipt_path = out / f'{name}_receipt.json'
    require(not output.exists() and not receipt_path.exists(), 'Refusing to reuse a previous numerical output or receipt.')
    cmd = command(out, phase, candidate, seed)
    receipt = dict(status='running', name=name, command=cmd, exit_code=None, started_utc=now(),
                   concurrent_tuning=phase == 'tune', cpu_threads_per_process=1, optimizer_dtype='float64',
                   thread_environment=THREAD_ENV, input_binding=binding, source_identity=identity(),
                   config_sha256=sha256(out / 'local_config.json') if phase != 'pilot' else None)
    create_json(receipt_path, receipt)
    started = time.perf_counter()
    try:
        with (out / f'{name}.log').open('x') as stream:
            result = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, **THREAD_ENV), stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=3600)
        receipt.update(exit_code=result.returncode, process_elapsed_seconds=time.perf_counter() - started,
                       finished_utc=now())
        require(result.returncode == 0, f'{name} failed with exit {result.returncode}; see log.')
        metadata_path = output / 'metadata.json'
        receipt['metadata_sha256'] = sha256(metadata_path)
        value = validate(out, phase, candidate, seed, receipt, binding)
        receipt['status'] = 'completed'
        write_json(receipt_path, receipt)
        print(json.dumps(dict(name=name, status='completed', loss=value['last_rounded_loss_float64'],
                              solver_seconds=read(metadata_path)['timing']['solver_seconds'])), flush=True)
        return value
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                       process_elapsed_seconds=time.perf_counter() - started, finished_utc=now())
        write_json(receipt_path, receipt)
        raise


def audited_run(out, phase, candidate, seed, binding):
    path = out / f'{run_name(phase, candidate, seed)}_receipt.json'
    receipt = read(path)
    require(receipt['status'] == 'completed', 'Run receipt did not complete.')
    if phase != 'pilot':
        require(receipt['config_sha256'] == sha256(out / 'local_config.json'), 'Frozen configuration binding changed.')
    return validate(out, phase, candidate, seed, receipt, binding)


def select(tuned):
    # Python's stable min retains the original declared candidate-order tie rule.
    return min(tuned, key=lambda row: row['last_rounded_loss_float64'])['candidate']


def selection(out, config, binding):
    value = read(out / 'selection.json')
    tuned = [audited_run(out, 'tune', candidate, 810, binding) for candidate in CANDIDATES]
    require(value['status'] == 'completed' and value['config_sha256'] == sha256(out / 'local_config.json')
            and value['selection_metric'] == config['selected_metric'] and value['tuning'] == tuned
            and value['selected'] == select(tuned), 'Tuning selection changed or is incomplete.')
    return value


def phase_pilot(out):
    _, binding = inputs(out)
    return dict(pilot=execute(out, 'pilot', CANDIDATES[0], 809, binding))


def phase_tune(out):
    config = frozen_config(out, create=True)
    _, binding = inputs(out)
    (out / 'runs').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tuned = list(pool.map(lambda candidate: execute(out, 'tune', candidate, 810, binding), CANDIDATES))
    value = dict(status='completed', selected=select(tuned), tuning=tuned,
                 selection_metric=config['selected_metric'], config_sha256=sha256(out / 'local_config.json'))
    create_json(out / 'selection.json', value)
    return dict(selected=value['selected'], tuning=tuned)


def phase_confirm(out, seed):
    require(seed in SEEDS, 'Confirmation seed must be one of 811, 812 or 813.')
    config = frozen_config(out)
    _, binding = inputs(out)
    chosen = selection(out, config, binding)
    value = execute(out, 'confirm', chosen['selected'], seed, binding)
    create_json(out / f'confirmation_seed{seed}.json', dict(status='completed', run=value,
                config_sha256=sha256(out / 'local_config.json'), selection_sha256=sha256(out / 'selection.json')))
    return dict(confirmation=value)


def phase_finalize(out):
    config = frozen_config(out)
    _, binding = inputs(out)
    chosen = selection(out, config, binding)
    confirmed = [audited_run(out, 'confirm', chosen['selected'], seed, binding) for seed in SEEDS]
    for seed, run in zip(SEEDS, confirmed):
        receipt = read(out / f'confirmation_seed{seed}.json')
        require(receipt['status'] == 'completed' and receipt['run'] == run
                and receipt['config_sha256'] == sha256(out / 'local_config.json')
                and receipt['selection_sha256'] == sha256(out / 'selection.json'), 'Confirmation stage receipt changed.')
    intervals = sorted((row['receipt']['started_utc'], row['receipt']['finished_utc']) for row in confirmed)
    require(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])), 'Confirmation processes overlapped; timings are invalid.')
    value = dict(status='completed', config=config, tuning=chosen['tuning'], selected=chosen['selected'], confirmation=confirmed)
    create_json(out / 'completed.json', value)
    return dict(completed_file=str(out / 'completed.json'), selected=chosen['selected'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('pilot', 'tune', 'confirm', 'finalize'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_dir(), 'Experiment directory and verified input links must already exist.')
    require(args.phase == 'confirm' or args.seed is None, '--seed is only valid for confirmation.')
    stage = args.phase + (f'_seed{args.seed}' if args.phase == 'confirm' else '')
    lock = out / '.controller.lock'
    create_json(lock, dict(pid=os.getpid(), stage=stage, started_utc=now()))
    receipt_path = out / f'stage_{stage}.json'
    started = time.perf_counter()
    receipt = dict(status='running', stage=stage, started_utc=now(), controller_sha256=sha256(__file__))
    try:
        create_json(receipt_path, receipt)
        value = phase_confirm(out, args.seed) if args.phase == 'confirm' else globals()['phase_' + args.phase](out)
        receipt.update(status='completed', result=value, finished_utc=now(), process_elapsed_seconds=time.perf_counter() - started)
        write_json(receipt_path, receipt)
        print(json.dumps(dict(stage=stage, status='completed')), flush=True)
    except Exception as exc:
        # Do not overwrite a receipt from an earlier attempted stage.
        if receipt_path.exists() and read(receipt_path).get('started_utc') == receipt['started_utc']:
            receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                           finished_utc=now(), process_elapsed_seconds=time.perf_counter() - started)
            write_json(receipt_path, receipt)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
