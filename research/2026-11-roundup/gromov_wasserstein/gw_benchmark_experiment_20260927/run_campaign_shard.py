"""Execute a deterministic shard of a frozen experiment manifest.

Runs remain separate processes. Solver failures are data; infrastructure failures
are recorded and abort a shard after three consecutive occurrences. Existing run
folders are never overwritten. A rerun resumes only matching terminal records.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUNNERS = {'run_benchmark.py', 'run_native_sampled.py'}
TERMINAL = {'completed', 'completed_with_warnings', 'completed_with_inner_failures', 'failed'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in path.parts):
        raise ValueError(f'Unsafe manifest path: {value!r}')
    return path


def validate_manifest(manifest, data_root):
    runs = manifest['runs']
    ids = [run['run_id'] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate run IDs.')
    for run in runs:
        run_relative = safe_relative(run['run_id'])
        if run_relative.parts[0] == '_campaign':
            raise ValueError('Run IDs cannot use the reserved control directory.')
        instance = safe_relative(run['instance'])
        if len(instance.parts) != 1:
            raise ValueError('Instance must be a frozen filename, not a path.')
        argv = run['argv']
        if not argv or argv[0] not in RUNNERS or any(not isinstance(x, str) for x in argv):
            raise ValueError('Unrecognized runner or argument types.')
        budget = float(run['budget'])
        if not math.isfinite(budget) or budget <= 0 or '--budget' not in argv:
            raise ValueError('Invalid or missing run budget.')
        if float(argv[argv.index('--budget') + 1]) != budget:
            raise ValueError('Runner budget disagrees with the manifest budget.')
        if '--out' in argv or '--instance' in argv:
            raise ValueError('Output and instance are supplied by the launcher.')
        path = data_root / instance
        if not path.is_file():
            raise FileNotFoundError(path)
    # Hash each distinct instance once rather than once per candidate/repeat.
    actual = {name: digest(data_root / name) for name in sorted({r['instance'] for r in runs})}
    for run in runs:
        expected = run.get('instance_sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f'Missing frozen input SHA256: {run["instance"]}')
        if expected != actual[run['instance']]:
            raise ValueError(f'Frozen input hash mismatch: {run["instance"]}')
    return actual


def validate_source(expected_commit):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if actual != expected_commit:
        raise ValueError(f'Expected source commit {expected_commit}, found {actual}.')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError('Campaign source tree is dirty; use the frozen clean release.')
    return actual


def _run_shard(args):
    if args.task_count < 1 or not 0 <= args.task_index < args.task_count:
        raise ValueError('Invalid shard index/count.')
    manifest_path = Path(args.manifest).resolve()
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    data_root, out_root = Path(args.data_root).resolve(), Path(args.out_root).resolve()
    hashes = validate_manifest(manifest, data_root)
    assigned = manifest['runs'][args.task_index::args.task_count]
    out_root.mkdir(parents=True, exist_ok=True)
    control = out_root / '_campaign'
    control.mkdir(exist_ok=True)
    ledger_path = control / f'shard_{args.task_index:02d}.json'
    if ledger_path.exists():
        prior = json.loads(ledger_path.read_text())
        if prior['manifest_sha256'] != manifest_hash or prior['task_count'] != args.task_count:
            raise ValueError('Refusing to resume a different manifest or sharding.')
    current_commit = validate_source(args.source_commit)
    if ledger_path.exists() and prior['source_commit'] != current_commit:
        raise ValueError('Resume requires the same immutable source revision.')
    ledger = dict(manifest=str(manifest_path), manifest_sha256=manifest_hash,
                  task_index=args.task_index, task_count=args.task_count,
                  source_commit=current_commit,
                  source_sha256=digest(__file__), input_sha256=hashes,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                  assigned_count=len(assigned), records=[], status='running', start_unix=time.time())
    atomic_json(ledger_path, ledger)
    infrastructure_streak = 0
    for number, run in enumerate(assigned):
        validate_source(args.source_commit)
        run_path = out_root / safe_relative(run['run_id'])
        receipt_path = run_path / 'campaign_receipt.json'
        metadata_path = run_path / 'metadata.json'
        log_path = control / 'logs' / (run['run_id'].replace('/', '__') + '.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / run['argv'][0]), *run['argv'][1:],
                   '--instance', str(data_root / run['instance']), '--out', str(run_path)]
        receipt = dict(run_id=run['run_id'], manifest_sha256=manifest_hash,
                       command=command, start_unix=time.time(), run_spec=run)
        if run_path.exists():
            if not receipt_path.is_file() or not metadata_path.is_file():
                raise RuntimeError(f'Existing incomplete run requires explicit review: {run_path}')
            previous = json.loads(receipt_path.read_text())
            metadata = json.loads(metadata_path.read_text())
            if previous['manifest_sha256'] != manifest_hash or metadata.get('status') not in TERMINAL:
                raise RuntimeError(f'Existing run does not match this frozen campaign: {run_path}')
            receipt = previous
            receipt = {**receipt, 'resumed_terminal_record': True}
        else:
            run_path.parent.mkdir(parents=True, exist_ok=True)
            timeout = max(120., float(run['budget']) * 6 + 60)
            started = time.perf_counter()
            try:
                with log_path.open('x') as log:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=timeout, check=False)
                receipt.update(returncode=process.returncode, infrastructure_error=None)
            except (subprocess.TimeoutExpired, OSError) as exc:
                receipt.update(returncode=None, infrastructure_error=f'{type(exc).__name__}: {exc}')
            receipt['instrumented_process_seconds'] = time.perf_counter() - started
            receipt['end_unix'] = time.time()
            try:
                metadata = json.loads(metadata_path.read_text())
                status = metadata.get('status')
                if status not in TERMINAL:
                    raise ValueError(f'Nonterminal runner status {status!r}')
                receipt['status'] = status
                if receipt['returncode'] not in (0, 1) or (receipt['returncode'] == 1 and status != 'failed'):
                    raise ValueError('Runner exit code disagrees with terminal metadata.')
                if receipt['infrastructure_error'] is not None:
                    raise ValueError(receipt['infrastructure_error'])
            except (OSError, ValueError) as exc:
                receipt['status'] = 'infrastructure_failure'
                receipt['infrastructure_error'] = f'{type(exc).__name__}: {exc}'
            run_path.mkdir(parents=True, exist_ok=True)
            atomic_json(receipt_path, receipt)
        infrastructure_streak = infrastructure_streak + 1 if receipt['status'] == 'infrastructure_failure' else 0
        ledger['records'].append(receipt)
        ledger['finished_count'] = len(ledger['records'])
        ledger['updated_unix'] = time.time()
        atomic_json(ledger_path, ledger)
        print(json.dumps(dict(shard=args.task_index, finished=number+1, total=len(assigned),
                              run_id=run['run_id'], status=receipt['status'])), flush=True)
        if infrastructure_streak >= 3:
            ledger['status'] = 'aborted_infrastructure_failures'
            atomic_json(ledger_path, ledger)
            return 2
    ledger['status'] = 'completed_with_failed_runs' if any(r['status'] == 'failed' for r in ledger['records']) else 'completed'
    if any(r['status'] == 'infrastructure_failure' for r in ledger['records']):
        ledger['status'] = 'completed_with_infrastructure_failures'
    ledger['end_unix'] = time.time()
    atomic_json(ledger_path, ledger)
    return 0 if ledger['status'] != 'completed_with_infrastructure_failures' else 2


def run_shard(args):
    try:
        return _run_shard(args)
    except Exception as exc:
        # Record source/resume errors as terminal control failures without
        # changing any completed solver output or another manifest's ledger.
        error = dict(status='aborted_exception', error=f'{type(exc).__name__}: {exc}',
                     task_index=args.task_index, expected_source_commit=args.source_commit,
                     manifest=str(args.manifest), end_unix=time.time())
        control = Path(args.out_root).resolve() / '_campaign'
        control.mkdir(parents=True, exist_ok=True)
        atomic_json(control / f'launcher_error_{args.task_index:02d}_{os.getpid()}.json', error)
        ledger_path = control / f'shard_{args.task_index:02d}.json'
        if ledger_path.is_file():
            ledger = json.loads(ledger_path.read_text())
            if (ledger.get('status') == 'running'
                    and ledger.get('manifest_sha256') == digest(args.manifest)
                    and ledger.get('source_commit') == args.source_commit):
                ledger.update(status='aborted_exception', error=error['error'], end_unix=error['end_unix'])
                atomic_json(ledger_path, ledger)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--source-commit', required=True, help='Full immutable expected source revision; checked before every run.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--task-count', type=int, default=4)
    raise SystemExit(run_shard(parser.parse_args()))


if __name__ == '__main__':
    main()
