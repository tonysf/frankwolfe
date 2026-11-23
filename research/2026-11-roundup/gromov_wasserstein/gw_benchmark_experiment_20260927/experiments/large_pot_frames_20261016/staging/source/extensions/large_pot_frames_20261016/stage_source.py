"""Stage tracked experiment code without copying datasets or mutating a checkout."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def stage(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0')
    selected = sorted(n for n in names if n and (
        (n.startswith('extensions/') and Path(n).suffix in ('.py', '.json', '.sh', '.sbatch'))
        or (n.startswith('gwbench/') and Path(n).suffix == '.py')
        or n in ('gw_core.py', 'pyproject.toml', 'requirements.txt')))
    required = ['extensions/large_pot_frames_20261016/' + n for n in (
        'campaign.py', 'campaign.json', 'runner.py', 'checkpoint.py', 'prox.py', 'validation.py',
        'report.py', 'gpu.sbatch', 'cpu.sbatch', 'submit_campaign.sh')]
    if not set(required) <= set(selected):
        raise ValueError('Commit new experiment files before staging; missing ' + repr(set(required) - set(selected)))
    dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *selected], cwd=repo).decode().splitlines()
    if dirty:
        raise ValueError('Staged source must match the recorded commit: ' + repr(dirty))
    destination.mkdir(parents=True)
    hashes = {}
    for name in selected:
        src, dst = repo / name, destination / name
        if src.is_symlink():
            raise ValueError('Source symlink is not allowed: ' + name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode().strip()
    (destination / 'SOURCE_COMMIT').write_text(commit + '\n')
    hashes['SOURCE_COMMIT'] = hashlib.sha256((commit + '\n').encode()).hexdigest()
    (destination / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items())))
    (destination / 'stage_receipt.json').write_text(json.dumps({'commit': commit, 'file_count': len(hashes),
        'source_manifest_sha256': hashlib.sha256((destination / 'SHA256SUMS').read_bytes()).hexdigest()}, indent=2) + '\n')
    return hashes


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    stage(args.repo, args.out)
