"""Bounded paired BMMC scaling pilot; no retuning or automatic replication."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Pilot must run under Slurm.')
    here = Path(__file__).resolve().parent
    config = json.loads((here / 'pilot.json').read_text())
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.runtime, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=args.runtime, text=True).strip()
    if actual != config['runtime_commit'] or dirty:
        raise RuntimeError('Runtime is not the specified clean immutable release.')
    host_bytes = int(os.environ['SLURM_MEM_PER_NODE']) * 1024**2
    if host_bytes != config['host_memory_bytes']:
        raise RuntimeError('Actual host-memory allocation differs from frozen pilot configuration.')
    args.out.mkdir(parents=True, exist_ok=False)
    data = args.out / 'data'; data.mkdir()
    state = dict(status='running', started_unix=time.time(), job_id=os.environ['SLURM_JOB_ID'],
                 runtime_commit=actual, config=config,
                 extension_sha256={p.name: digest(p) for p in here.iterdir() if p.is_file()},
                 records=[], gates=[], attempts=[],
                 allocation_deadline_unix=int(os.environ['GW_PILOT_DEADLINE_EPOCH']))
    ledger = args.out / 'pilot.json'
    write(ledger, state)

    def execute(command, log, timeout):
        available = state['allocation_deadline_unix'] - time.time() - 300
        if available < 1:
            raise RuntimeError('Allocation time reserve reached before launching child.')
        timeout = min(timeout, available)
        attempt = dict(command=command, log=str(log), timeout_seconds=timeout,
                       started_unix=time.time(), status='running')
        state['attempts'].append(attempt)
        write(ledger, state)
        started = time.perf_counter()
        code = None
        try:
            with log.open('x') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        cwd=args.runtime, timeout=timeout, check=False)
            code = result.returncode
            attempt.update(status='completed' if code == 0 else 'failed', returncode=code)
        except (subprocess.TimeoutExpired, OSError) as exc:
            attempt.update(status='infrastructure_failure', returncode=None,
                           error=type(exc).__name__ + ': ' + str(exc))
        seconds = time.perf_counter() - started
        attempt.update(ended_unix=time.time(), process_seconds=seconds)
        write(ledger, state)
        return code, seconds

    def prepare(n):
        command = [sys.executable, str(here / 'prepare_instances.py'), '--root', str(args.root),
                   '--runtime', str(args.runtime), '--out', str(data), '--n', str(n)]
        code, seconds = execute(command, args.out / ('prepare_' + str(n) + '.log'), 300)
        state.setdefault('preparation', []).append(dict(n=n, returncode=code, seconds=seconds))
        write(ledger, state)
        if code != 0:
            raise RuntimeError('Data preparation failed: ' + str(n))

    try:
        code, _ = execute([sys.executable, '-u', str(here / 'startup_probe.py'), str(args.runtime / 'capture_environment.py'), '--device', 'cuda',
                 '--out', str(args.out / 'environment.json')], args.out / 'environment.log', config['startup_timeout_seconds'])
        if code != 0:
            raise RuntimeError('Allocated environment validation failed.')
        environment = json.loads((args.out / 'environment.json').read_text())
        gpu_bytes = environment['gpu']['total_memory_bytes']
        prepare(1000)
        previous = None
        for index, n in enumerate(config['sizes']):
            if previous is not None:
                ratio = (n / previous['n']) ** 2
                gpu_estimate = max(r['metadata']['memory']['cuda_peak_reserved_bytes'] for r in previous['records']) * ratio
                cpu_estimate = max(r['metadata']['memory']['cpu_process_peak_rss_kib'] * 1024 for r in previous['records']) * ratio
                seconds_left = state['allocation_deadline_unix'] - time.time()
                passed = (gpu_estimate < .8 * gpu_bytes and
                          cpu_estimate < .8 * host_bytes and seconds_left > 4200)
                state['gates'].append(dict(from_n=previous['n'], to_n=n,
                    projected_gpu_reserved_bytes=gpu_estimate, projected_cpu_peak_bytes=cpu_estimate,
                    seconds_left=seconds_left, passed=passed,
                    rule='valid previous outputs; quadratic memory projection below80% of allocations; >4200s remaining; no loss-ranking gate'))
                write(ledger, state)
                if not passed:
                    state['status'] = 'stopped_at_resource_gate'
                    break
            prepare(n)
            methods = config['methods'] if index % 2 == 0 else list(reversed(config['methods']))
            records = []
            for method in methods:
                output = args.out / ('n' + str(n)) / method
                output.parent.mkdir(exist_ok=True)
                command = [sys.executable, str(args.runtime / 'run_benchmark.py'),
                    '--instance', str(data / ('bmmc' + str(n) + '.npz')), '--out', str(output),
                    '--method', method, '--device', 'cuda', '--dtype', 'float32', '--threads', '1',
                    '--seed', '100', '--init-seed', '100', '--initialization', 'product',
                    '--budget', '300', '--checkpoints', '30,100,300', '--oracle', 'sketch',
                    '--batch', '128', '--rho0', '0.3', '--gap-max-n', '512',
                    '--inner-max', '1000', '--oracle-selection-status', 'user_requested_fixed_settings_scaling_pilot']
                if method == 'smoothing-fw':
                    command += ['--gamma0', '1', '--mu-scale', '10']
                else:
                    command += ['--eta', '10', '--eta-power', '0', '--inner-tol', '0.0001']
                code, seconds = execute(command, output.parent / (method + '.log'), 1800)
                path = output / 'metadata.json'
                metadata = json.loads(path.read_text()) if path.exists() else None
                record = dict(n=n, method=method, command=command, returncode=code,
                              process_seconds=seconds, metadata=metadata,
                              metadata_sha256=digest(path) if path.exists() else None)
                state['records'].append(record); records.append(record)
                write(ledger, state)
                valid = (code == 0 and metadata is not None and
                         metadata['status'] in ('completed', 'completed_with_inner_failures') and
                         metadata['timing']['solver_seconds'] >= 300 and
                         metadata['counters']['completed_updates'] > 0 and
                         metadata['final_float64_validation'] is not None and
                         math.isfinite(metadata['final_float64_validation']['rounded_loss']) and
                         math.isfinite(metadata['final_float64_validation']['raw_loss']) and
                         metadata['final_float64_validation']['rounded']['both_l1'] <= 1.2e-5 and
                         (output / 'final.npz').is_file())
                if not valid:
                    raise RuntimeError('Pilot output invalid; retain evidence and stop: ' + str(output))
            for key in ('initialization_plan_sha256', 'instance_sha256', 'array_sha256'):
                if records[0]['metadata'][key] != records[1]['metadata'][key]:
                    raise RuntimeError('Matched pair differs: ' + key)
            previous = dict(n=n, records=records)
        else:
            state['status'] = 'completed'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        state['ended_unix'] = time.time()
        write(ledger, state)
        inventory = [dict(path=str(p.relative_to(args.out)), bytes=p.stat().st_size, sha256=digest(p))
                     for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json']
        write(args.out / 'SHA256SUMS.json', dict(files=inventory))
        print(json.dumps(dict(status=state['status'], runs=len(state['records']), out=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
