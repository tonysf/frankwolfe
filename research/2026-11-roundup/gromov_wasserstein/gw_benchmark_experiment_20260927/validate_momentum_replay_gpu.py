"""Bounded allocated-A100 checks for archived momentum diagnostic reconstruction."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from replay_momentum_diagnostics import input_bindings
from run_momentum_replay_shard import verify_companion

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--pilot-manifest', type=Path, required=True)
    parser.add_argument('--pilot-manifest-sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This GPU validation requires an allocated Slurm job.')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != args.source_commit:
        raise RuntimeError('Preflight controller source mismatch.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Preflight source must be immutable and clean.')
    if digest(args.pilot_manifest) != args.pilot_manifest_sha256:
        raise RuntimeError('Pilot replay manifest changed.')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', source_commit=args.source_commit,
                  job_id=os.environ['SLURM_JOB_ID'], commands=[], checks=[],
                  pilot_manifest_sha256=args.pilot_manifest_sha256)
    started = time.perf_counter()

    def save():
        (args.out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')

    def command(argv):
        before = time.perf_counter()
        record = dict(argv=[str(x) for x in argv], returncode=None)
        report['commands'].append(record)
        save()
        result = subprocess.run(record['argv'], cwd=ROOT, check=False)
        record.update(returncode=result.returncode, elapsed_seconds=time.perf_counter()-before)
        save()
        if result.returncode:
            raise RuntimeError(f'Preflight child failed: {record}')

    def replay(original, label):
        meta = json.loads((original / 'metadata.json').read_text())
        output = args.out / 'reconstructions' / label
        command([sys.executable, ROOT / 'replay_momentum_diagnostics.py',
                 '--run-dir', original, '--instance', meta['instance'],
                 '--out', output, '--device', 'cuda'])
        path = output / 'verification.json'
        verification = json.loads(path.read_text())
        verify_companion(output, input_bindings(original, Path(meta['instance'])))
        if not verification['verified'] or verification['status'] != 'verified':
            raise RuntimeError(f'Unverified reconstruction: {path}')
        report['checks'].append(dict(label=label, original=str(original),
                                    verification=str(path), sha256=digest(path),
                                    checkpoint_count=verification['checkpoint_count']))
        save()

    try:
        command([sys.executable, ROOT / 'capture_environment.py', '--device', 'cuda',
                 '--out', args.out / 'environment.json'])
        # Original af7 toys cover both constraint updates and both initializers.
        for start in ('product', 'randomized'):
            for method in ('smoothing-fw', 'smoothing-row-projection'):
                original = args.experiment_root / 'a100_campaign/preflight_1961539' / start / method
                replay(original, f'af7/{start}/{method}')
        # Actual aaeb campaign-sized histories check old metadata/receipt schemas.
        for instance, method in [('metric512_s1000', 'smoothing-fw'),
                                  ('bmmc1000', 'smoothing-row-projection')]:
            original = (args.experiment_root / 'a100_campaign/tuning_aaeb5fa/tuning' /
                        method / 'c00' / instance / 's0_i0')
            replay(original, f'aaeb/{instance}/{method}')
        # Prospective logging fix: no performance comparison is made from these toys.
        for start in ('product', 'randomized'):
            for label, method, oracle in [('smoothing-fw', 'smoothing-fw', 'sketch'),
                                          ('smoothing-fw-exact', 'smoothing-fw', 'exact'),
                                          ('smoothing-row-projection', 'smoothing-row-projection', 'sketch')]:
                original = args.out / 'prospective_smokes' / start / label
                command([sys.executable, ROOT / 'run_benchmark.py', '--instance',
                         args.experiment_root / 'data/toy16_s1000.npz', '--out', original,
                         '--device', 'cuda', '--dtype', 'float32', '--threads', '1',
                         '--method', method, '--oracle', oracle, '--initialization', start,
                         '--batch', '128', '--rho0', '.3', '--eta', '.001', '--eta-power', '.25',
                         '--seed', '100', '--init-seed', '900100', '--budget', '.05',
                         '--checkpoints', '.02,.05'])
                replay(original, f'prospective/{start}/{label}')
        # These six successful attachments are the production pilot completion;
        # later production arrays must not repeat them as new diagnostic records.
        command([sys.executable, ROOT / 'run_momentum_replay_shard.py', '--manifest',
                 args.pilot_manifest, '--manifest-sha256', args.pilot_manifest_sha256,
                 '--source-commit', args.source_commit, '--task-index', '0', '--task-count', '1'])
        pilot = json.loads(args.pilot_manifest.read_text())
        if len(pilot['runs']) != 6:
            raise RuntimeError('Expected exactly six original pilot trajectories.')
        for trial in pilot['runs']:
            path = Path(pilot['out_root']) / trial['run_id'] / 'verification.json'
            value = json.loads(path.read_text())
            verify_companion(path.parent, input_bindings(Path(trial['original_run_dir']),
                                                       Path(pilot['data_root']) / trial['instance']))
            if value['status'] != 'verified' or not value['verified']:
                raise RuntimeError(f'Unverified pilot: {path}')
            report['checks'].append(dict(label=trial['run_id'], verification=str(path),
                                        sha256=digest(path), checkpoint_count=value['checkpoint_count']))
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        save()
    print(json.dumps(dict(status=report['status'], verified_replays=len(report['checks']),
                          report=str(args.out / 'validation.json'))))


if __name__ == '__main__':
    main()
