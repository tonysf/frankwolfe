"""Launch the full-gradient and sampled POT-scale controls on physical CPU cores."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'experiments/scot_pot_step_20261015'
REFERENCE = ROOT / 'experiments/scot_cpu_20261014/reference'
CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000]
THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, exclusive=False):
    path = Path(path)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(data)
        temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def physical_cores():
    selected = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity = tuple(int((directory / key).read_text()) for key in ('physical_package_id', 'core_id'))
        selected.setdefault(identity, cpu)
    require(len(selected) >= 4, 'Four physical CPU cores are required.')
    return list(selected.values())[:4]


def source_hashes():
    result = module(HERE / 'run.py', '_entropy_runner_identity').source_hashes()
    for path in (Path(__file__), HERE / 'summarize.py',
                 ROOT / 'extensions/scot_reproduction_20261014/reference.py'):
        result[str(path.relative_to(ROOT))] = sha256(path)
    return result


def prepare(out, iterations):
    require(isinstance(iterations, int) and 1 <= iterations <= 2000, 'Invalid fixed horizon.')
    require(not out.exists(), 'Experiment directory already exists; do not overwrite or relaunch.')
    reference = module(ROOT / 'extensions/scot_reproduction_20261014/reference.py', '_entropy_reference')
    record = reference.verify_reference(REFERENCE)
    instance = (REFERENCE / 'prepared/scot1047').resolve()
    hashes = {name: sha256(instance / name) for name in ('A.npy', 'B.npy', 'a.npy', 'b.npy')}
    cores = physical_cores()
    checkpoints = sorted({k for k in CHECKPOINTS if k <= iterations} | {iterations})
    run_specs = [('full_gradient', 'full', 811), ('pair_seed811', 'pair', 811),
                 ('pair_seed812', 'pair', 812), ('pair_seed813', 'pair', 813)]
    config = dict(schema_version=1, created_at=now(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_file_sha256=source_hashes(), reference=str(REFERENCE),
        reference_record_sha256=sha256(REFERENCE / 'reference.json'),
        reference_repaired_gw_loss=record['rounded']['loss'],
        instance=str(instance), array_files_sha256=hashes,
        instance_manifest_sha256=sha256(instance / 'manifest.json'),
        epsilon=.001, iterations=iterations, pairs_per_call=1024,
        checkpoint_iterations=checkpoints, inner_tol=1e-7, inner_gap_tol=1e-7,
        inner_max=10000, check_every=10, threads_per_process=1,
        inner_solver='FP64 matrix-vector Sinkhorn; original-objective certificate and log-domain fallback',
        thread_environment=THREAD_ENV, physical_cores=cores,
        run_timeout_seconds=24 * 3600, hostname=platform.node(),
        runs=[dict(id=name, oracle=oracle, seed=seed, output_seed=seed + 20000,
                   iterations=iterations, cpu=cpu)
              for (name, oracle, seed), cpu in zip(run_specs, cores)],
        initialization='uniform product a b^T', dtype='float64',
        objective='F_epsilon(P)=f(P)+epsilon*h(P); h=sum(P*(log(P)-1)); f is full squared GW without a half factor',
        update='argmin_Q <G_k,Q>+epsilon*h(Q), Q in U(a,b); full or iid pair gradient',
        proximal_certificate_objective='<G_k,Q>+epsilon*KL(Q||a*b^T); differs from the entropy-regularized linear problem by a feasible constant',
        schedule='eta_k=1/epsilon=1000, constant; no tuning', eta=1000.,
        theory=dict(source='https://pythonot.github.io/_modules/ot/gromov/_bregman.html#entropic_gromov_wasserstein',
                    reference='POT entropic_gromov_wasserstein PGD update at epsilon=0.001',
                    scope='Empirical comparison at the published POT step. This is not the conservative Fatkhullin-He theorem prescription. Independently uniform selected output is diagnostic only.'),
        interpretation='Fixed POT-step full-gradient control and stochastic counterpart, differing only in the GW gradient oracle. No method speed comparison.')
    out.mkdir(parents=True)
    (out / 'runs').mkdir()
    (out / 'logs').mkdir()
    (out / '.gitignore').write_text('runs/\nlogs/\n*.lock\n*.log\nreport/*.png\nreport/*.pdf\nreport_partial/\n')
    write(out / 'config.json', config, exclusive=True)
    return config


def validate(config):
    require(config['source_file_sha256'] == source_hashes(), 'Frozen source changed.')
    instance = Path(config['instance'])
    require(config['array_files_sha256'] == {name: sha256(instance / name) for name in ('A.npy', 'B.npy', 'a.npy', 'b.npy')}, 'Frozen arrays changed.')
    require(config['instance_manifest_sha256'] == sha256(instance / 'manifest.json'), 'Frozen manifest changed.')
    require(config['reference_record_sha256'] == sha256(REFERENCE / 'reference.json'), 'Reference changed.')
    require(config['physical_cores'] == physical_cores(), 'Physical core allocation changed.')
    require(config['epsilon'] == .001 and config['pairs_per_call'] == 1024
            and config['inner_tol'] == config['inner_gap_tol'] == 1e-7
            and config['thread_environment'] == THREAD_ENV, 'Numerical/execution settings changed.')


def arguments(out, config, spec):
    values = dict(instance=config['instance'], out=str(out / 'runs' / spec['id']),
                  oracle=spec['oracle'], seed=spec['seed'], iterations=config['iterations'],
                  epsilon=config['epsilon'], pairs=config['pairs_per_call'],
                  checkpoints=','.join(map(str, config['checkpoint_iterations'])), progress_every=100,
                  inner_max=config['inner_max'])
    return ['taskset', '--cpu-list', str(spec['cpu']), sys.executable, str(HERE / 'run.py')] + [
        word for key, value in values.items() for word in ('--' + key.replace('_', '-'), str(value))]


def execute_run(out, config, spec):
    name = spec['id']
    command = arguments(out, config, spec)
    receipt = dict(status='running', id=name, started_at=now(), command=command, cpu=spec['cpu'])
    receipt_path = out / f'receipt_{name}.json'
    write(receipt_path, receipt, exclusive=True)
    start = time.monotonic()
    try:
        with (out / 'logs' / f'{name}.log').open('x') as log:
            completed = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=config['run_timeout_seconds'], check=False)
        receipt['exit_code'] = completed.returncode
        require(completed.returncode == 0, f'{name} runner exited with {completed.returncode}.')
        metadata = read(out / 'runs' / name / 'metadata.json')
        require(metadata['status'] == 'completed'
                and metadata['counters']['completed_updates'] == config['iterations'], 'Incomplete runner result.')
        receipt.update(status='completed', metadata_sha256=sha256(out / 'runs' / name / 'metadata.json'))
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    receipt.update(finished_at=now(), elapsed_seconds=time.monotonic() - start)
    write(receipt_path, receipt)
    return receipt


def campaign(out):
    config = read(out / 'config.json')
    validate(config)
    # Keep this lock for the entire campaign. A separate immutable receipt also
    # prevents an accidental second execution after this process exits.
    with (out / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((out / 'runs').iterdir()), 'Existing runs require explicit audit, not automatic relaunch.')
        write(out / 'started.json', dict(started_at=now(), pid=os.getpid(),
              pid_namespace=os.readlink('/proc/self/ns/pid'),
              config_sha256=sha256(out / 'config.json')), exclusive=True)
        state = dict(status='running', started_at=now(), config_sha256=sha256(out / 'config.json'), runs=[])
        write(out / 'campaign.json', state)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(execute_run, out, config, spec) for spec in config['runs']]
                for future in concurrent.futures.as_completed(futures):
                    state['runs'].append(future.result())
                    write(out / 'campaign.json', state)
            all_passed = all(row['status'] == 'completed' for row in state['runs'])
            command = [sys.executable, str(HERE / 'summarize.py'), '--out', str(out)]
            if not all_passed:
                command.append('--partial')
            with (out / 'logs' / 'report.log').open('x') as log:
                report = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                                        stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
            state['report_exit_code'] = report.returncode
            require(all_passed and report.returncode == 0, 'A run or final validation failed; inspect preserved receipts/logs.')
            state.update(status='completed', finished_at=now(),
                         report_sha256=sha256(out / 'report/report.json'))
            write(out / 'completed.json', state, exclusive=True)
        except Exception as exc:
            state.update(status='failed', finished_at=now(), error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            write(out / 'campaign.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--iterations', type=int, default=2000)
    args = parser.parse_args()
    out = args.out.resolve()
    result = prepare(out, args.iterations) if args.phase == 'prepare' else campaign(out)
    print(json.dumps(dict(out=str(out), phase=args.phase,
                         status=result.get('status', 'prepared')), allow_nan=False))


if __name__ == '__main__':
    main()
