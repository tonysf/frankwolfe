"""Run an immutable companion diagnostic shard, without modifying original runs.

Only verified reconstructions publish diagnostics. Bound reconstruction failures
are scientific records and do not trigger resubmission. Missing/nonterminal
originals, provenance mismatches, or incomplete companion outputs abort as
infrastructure failures. Resume revalidates both the originals and companions.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_momentum_replays import (SOURCES, TERMINAL, artifact_hashes, canonical_absolute,
    digest, overlaps, read_json, safe_relative, snapshot_hashes, validate_plan)

ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def bind_once(path, value):
    """Atomically create an immutable shared binding, or require exact equality."""
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if read_json(path) != value:
                raise ValueError('Companion root already belongs to another manifest, sharding or source.')
    finally:
        tmp.unlink()


def validate_source(expected_commit):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip()
    if actual != expected_commit or dirty:
        raise ValueError('Replay controller must use the expected immutable clean source revision.')
    return actual


def assert_config(spec, config):
    """Every CLI field frozen by the scientific manifest must match metadata."""
    argv = spec['argv']
    if argv[0] != 'run_benchmark.py' or (len(argv) - 1) % 2:
        raise ValueError('Unsupported original command schema.')
    for flag, value in zip(argv[1::2], argv[2::2]):
        if not flag.startswith('--'):
            raise ValueError('Malformed original command option.')
        key = flag[2:].replace('-', '_')
        actual = config.get(key)
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            equal = math.isfinite(float(actual)) and float(actual) == float(value)
        else:
            equal = actual == value
        if not equal:
            raise ValueError(f'Original config differs from frozen option {flag}.')
    if config.get('initialization', 'product') != spec.get('initialization', 'product'):
        raise ValueError('Original initialization differs from frozen manifest.')


def verify_original(trial, data_root):
    original = canonical_absolute(trial['original_run_dir'])
    if original.resolve() != original:
        raise ValueError('Original directory must not use symlink aliases.')
    meta = read_json(original / 'metadata.json')
    if meta.get('status') not in TERMINAL:
        raise ValueError('Original trajectory is missing or nonterminal.')
    if (meta.get('source_commit') != trial['original_source_commit']
            or meta.get('source_dirty') is not False):
        raise ValueError('Original source revision or clean-state mismatch.')
    sources = snapshot_hashes(original)
    if sources != trial['source_snapshot_sha256']:
        raise ValueError('Original archived source differs from frozen replay manifest.')
    if {name: sources[name] for name in SOURCES[:3]} != meta['source_file_sha256']:
        raise ValueError('Original metadata does not bind its archived source.')
    instance = canonical_absolute(str(Path(data_root) / trial['instance']))
    if meta['instance_sha256'] != trial['instance_sha256'] or digest(instance) != trial['instance_sha256']:
        raise ValueError('Original canonical input hash mismatch.')
    if meta['config']['out'] != str(original) or meta['config']['instance'] != str(instance):
        raise ValueError('Original recorded input/output path mismatch.')
    implementation = (trial.get('original_run_spec') or {}).get('implementation',
        (trial.get('original_config') or {}).get('method', trial['method']))
    if meta['config']['method'] != implementation:
        raise ValueError('Original method mismatch.')
    if trial['method'] == 'smoothing-fw-exact' and meta['config']['oracle'] != 'exact':
        raise ValueError('Exact-oracle scientific label does not match the original configuration.')
    if trial.get('original_config') is not None and meta['config'] != trial['original_config']:
        raise ValueError('Original config changed since replay preparation.')
    spec = trial.get('original_run_spec')
    receipt_hash = None
    if spec is not None:
        rec_path = original / 'campaign_receipt.json'
        rec = read_json(rec_path)
        if (rec.get('run_spec') != spec or rec.get('run_id') != spec['run_id']
                or rec.get('manifest_sha256') != trial['original_manifest_sha256']
                or rec.get('status') != meta['status'] or rec.get('infrastructure_error') is not None
                or rec.get('returncode') not in (0, 1)
                or (rec.get('returncode') == 1 and meta['status'] != 'failed')):
            raise ValueError('Original terminal campaign receipt mismatch.')
        assert_config(spec, meta['config'])
        receipt_hash = digest(rec_path)
    artifacts = artifact_hashes(original)
    if trial.get('original_artifact_sha256') is not None and artifacts != trial['original_artifact_sha256']:
        raise ValueError('Original artifacts changed since replay preparation.')
    bindings = dict(metadata_sha256=artifacts['metadata.json'], trace_sha256=artifacts['trace.csv'],
                    final_sha256=artifacts['final.npz'], instance_sha256=trial['instance_sha256'],
                    source_file_sha256={name: sources[name] for name in SOURCES[:3]},
                    theory_config_sha256=sources['theory_config.json'])
    return meta, bindings, receipt_hash


PROOF_CHECKS = ('source_and_instance_bindings', 'environment_match', 'solver_counters',
                'diagnostics_preserved_oracle_rng', 'P_raw_bitwise', 'P_rounded_bitwise',
                'original_files_unchanged')


def verify_companion(out, bindings):
    path = Path(out) / 'verification.json'
    verification = read_json(path)
    status = verification.get('status')
    if (verification.get('schema_version') != 1 or status not in {'verified', 'failed'}
            or verification.get('verified') is not (status == 'verified')
            or verification.get('bindings') != bindings):
        raise ValueError('Replay verification is missing, nonterminal or has different original bindings.')
    utility_hash = digest(ROOT / 'replay_momentum_diagnostics.py')
    if (status == 'verified' or verification.get('utility_sha256') is not None) and verification.get('utility_sha256') != utility_hash:
        raise ValueError('Replay utility hash differs from the immutable controller source.')
    diagnostics = Path(out) / 'diagnostics.csv'
    if diagnostics.is_file() != (status == 'verified'):
        raise ValueError('Only verified companion results may publish diagnostics.csv.')
    diagnostic_hash = digest(diagnostics) if diagnostics.is_file() else None
    if status == 'verified':
        if verification.get('diagnostics_sha256') != diagnostic_hash:
            raise ValueError('Published diagnostics hash differs from its verification proof.')
        checks = verification.get('checks', {})
        if any(checks.get(name) is not True for name in PROOF_CHECKS):
            raise ValueError('Verified replay is missing required fidelity checks.')
        count = verification.get('checkpoint_count')
        proof = checks.get('checkpoint_scalars')
        if (type(count) is not int or count <= 0 or not isinstance(proof, list)
                or len(proof) != count):
            raise ValueError('Verified replay has incomplete checkpoint proof coverage.')
        with diagnostics.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != count:
            raise ValueError('Published diagnostic rows do not match checkpoint_count.')
        seen = set()
        for row, item in zip(rows, proof):
            compared = item.get('compared_fields')
            if (item.get('exact_match') is not True
                    or type(item.get('original_trace_row')) is not int
                    or type(item.get('iteration')) is not int
                    or item['original_trace_row'] < 0 or item['iteration'] < 0
                    or not isinstance(compared, list) or not compared
                    or any(not isinstance(name, str) or not name for name in compared)
                    or not {'raw_loss', 'rounded_loss', 'true_smoothed_gap'}.issubset(compared)
                    or item['original_trace_row'] in seen):
                raise ValueError('Checkpoint scalar proof does not establish exact matches.')
            seen.add(item['original_trace_row'])
            if (str(item['original_trace_row']) != row.get('original_trace_row')
                    or str(item['iteration']) != row.get('output_iteration')):
                raise ValueError('Checkpoint scalar proof does not match the published diagnostic row.')
    return status, dict(verification_sha256=digest(path), diagnostics_sha256=diagnostic_hash)


def execute_shard(args):
    if args.task_count < 1 or not 0 <= args.task_index < args.task_count:
        raise ValueError('Invalid shard index/count.')
    manifest = Path(args.manifest).resolve()
    if digest(manifest) != args.manifest_sha256:
        raise ValueError('Replay manifest hash mismatch.')
    plan = validate_plan(read_json(manifest))
    validate_source(args.source_commit)
    out_root = canonical_absolute(plan['out_root'])
    if overlaps(out_root, ROOT):
        raise ValueError('Companion output must be disjoint from the immutable controller source.')
    if out_root.resolve() != out_root:
        raise ValueError('Companion root must not use symlink aliases.')
    control = out_root / '_replay'
    control.mkdir(parents=True, exist_ok=True)
    identity = dict(manifest_sha256=args.manifest_sha256, task_count=args.task_count,
                    source_commit=args.source_commit, out_root=str(out_root))
    bind_once(control / 'binding.json', identity)
    with (control / f'shard_{args.task_index:02d}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assigned = plan['runs'][args.task_index::args.task_count]
        ledger_path = control / f'shard_{args.task_index:02d}.json'
        previous = read_json(ledger_path) if ledger_path.exists() else None
        ledger = dict(**identity, task_index=args.task_index, assigned_count=len(assigned),
                      slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                      slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                      slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                      start_unix=time.time(), records=[], status='running')
        if previous:
            if any(previous.get(key) != value for key, value in identity.items()):
                raise ValueError('Prior shard identity differs.')
            ledger['previous_execution'] = {key: previous.get(key) for key in
                ('slurm_job_id', 'slurm_array_job_id', 'slurm_array_task_id', 'start_unix', 'end_unix', 'status')}
        atomic_json(ledger_path, ledger)
        try:
            for trial in assigned:
                ledger['current_run_id'] = trial['run_id']
                ledger['current_original_run_dir'] = trial['original_run_dir']
                atomic_json(ledger_path, ledger)
                validate_source(args.source_commit)
                meta, bindings, original_receipt_hash = verify_original(trial, plan['data_root'])
                out = out_root / safe_relative(trial['run_id'])
                if out.resolve() != out:
                    raise ValueError('Companion trial path must not use symlink aliases.')
                receipt_path = out / 'replay_receipt.json'
                trial_binding = dict(run_spec=trial, manifest_sha256=args.manifest_sha256,
                                     controller_source_commit=args.source_commit,
                                     original_bindings=bindings,
                                     original_campaign_receipt_sha256=original_receipt_hash)
                if out.exists():
                    receipt = read_json(receipt_path)
                    if any(receipt.get(k) != value for k, value in trial_binding.items()):
                        raise ValueError(f'Existing companion differs or is incomplete: {out}')
                    status, files = verify_companion(out, bindings)
                    if receipt.get('status') != status or receipt.get('output_sha256') != files:
                        raise ValueError('Existing terminal companion evidence changed.')
                    record = {**receipt, 'resumed_terminal_record': True}
                else:
                    log_path = control / 'logs' / (trial['run_id'].replace('/', '__') + '.log')
                    log_path.parent.mkdir(exist_ok=True)
                    command = [sys.executable, '-B', str(ROOT / 'replay_momentum_diagnostics.py'),
                               '--run-dir', trial['original_run_dir'], '--instance',
                               str(Path(plan['data_root']) / trial['instance']), '--out', str(out),
                               '--device', meta['config']['device']]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    prior_elapsed = float(meta.get('timing', {}).get('total_elapsed_seconds_before_final_metadata_write', meta['config']['budget']))
                    timeout = max(180., 6 * max(float(meta['config']['budget']), prior_elapsed) + 120.)
                    if not math.isfinite(timeout):
                        raise ValueError('Invalid original timing for infrastructure timeout.')
                    record = dict(**trial_binding, command=command, start_unix=time.time(),
                                  infrastructure_timeout_seconds=timeout)
                    with log_path.open('x') as log:
                        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                              timeout=timeout, check=False,
                                              env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                    record.update(end_unix=time.time(), returncode=proc.returncode)
                    status, files = verify_companion(out, bindings)
                    if proc.returncode != (0 if status == 'verified' else 1):
                        raise ValueError('Replay process exit code and verification status disagree.')
                    _, after_bindings, after_receipt = verify_original(trial, plan['data_root'])
                    if after_bindings != bindings or after_receipt != original_receipt_hash:
                        raise ValueError('Original artifacts changed during the companion replay.')
                    record.update(status=status, output_sha256=files)
                    atomic_json(receipt_path, record)
                ledger['records'].append(record)
                ledger.update(finished_count=len(ledger['records']), updated_unix=time.time())
                atomic_json(ledger_path, ledger)
                print(json.dumps(dict(task=args.task_index, finished=len(ledger['records']),
                    total=len(assigned), run_id=trial['run_id'], status=record['status'])), flush=True)
            ledger.update(current_run_id=None, current_original_run_dir=None, status=('completed_with_unverified_replays' if any(r['status'] == 'failed' for r in ledger['records']) else 'completed'), end_unix=time.time())
            atomic_json(ledger_path, ledger)
            return 0
        except Exception as exc:
            ledger.update(status='infrastructure_failure', error=f'{type(exc).__name__}: {exc}',
                          end_unix=time.time())
            atomic_json(ledger_path, ledger)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--task-index', required=True, type=int)
    p.add_argument('--task-count', required=True, type=int)
    args = p.parse_args()
    try:
        code = execute_shard(args)
    except Exception as exc:
        print(json.dumps(dict(status='infrastructure_failure', error=f'{type(exc).__name__}: {exc}')), file=sys.stderr, flush=True)
        code = 2
    raise SystemExit(code)


if __name__ == '__main__':
    main()
