"""One-cell allocated dispatcher smoke, using a toy development instance only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    instance = args.experiment_root / 'data/toy16_s1000.npz'
    manifest = out / 'manifest.json'
    spec = dict(run_id='dispatcher-smoke/stochastic-kl-momentum/toy16',
                instance=instance.name, instance_sha256=digest(instance), budget=.1,
                argv=['run_benchmark.py', '--method', 'stochastic-kl-momentum',
                      '--device', 'cuda', '--dtype', 'float32', '--threads', '1',
                      '--batch', '128', '--rho0', '.3', '--eta', '.1', '--eta-power', '0',
                      '--inner-tol', '1e-6', '--inner-max', '1000', '--seed', '100',
                      '--init-seed', '900100', '--initialization', 'randomized',
                      '--budget', '.1', '--checkpoints', '.03,.1'])
    manifest.write_text(json.dumps({'schema_version': 1, 'runs': [spec]}, indent=2)+'\n')
    plan = out / 'plan.json'
    plan.write_text(json.dumps(dict(schema_version=1, tasks=[dict(
        manifest=str(manifest), manifest_sha256=digest(manifest),
        out_root=str(out/'outputs'), task_index=0, task_count=1)]), indent=2)+'\n')
    subprocess.run([sys.executable, str(source/'run_panel_array.py'),
                    '--plan', str(plan), '--plan-sha256', digest(plan), '--task-index', '0',
                    '--source-directory', str(source), '--source-commit', args.source_commit], check=True)
    receipt = json.loads((out/'outputs'/spec['run_id']/'campaign_receipt.json').read_text())
    assert receipt['status'] in ('completed', 'completed_with_inner_failures')
    assert receipt['manifest_sha256'] == digest(manifest)
    ledger = json.loads((out/'outputs/_campaign/shard_00.json').read_text())
    assert ledger['status'] == 'completed' and ledger['finished_count'] == 1
    assert ledger['source_commit'] == args.source_commit
    print('Allocated dispatcher smoke passed.', flush=True)


if __name__ == '__main__':
    main()
