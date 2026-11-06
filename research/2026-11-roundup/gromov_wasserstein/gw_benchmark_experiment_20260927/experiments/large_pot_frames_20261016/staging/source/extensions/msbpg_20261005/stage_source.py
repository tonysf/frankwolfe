"""Export a reviewed Git commit as a small, checksummed Ruche source package.

Runs locally and performs no SSH, Slurm submission, dataset copy, or cleanup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output-parent', type=Path, required=True)
    args = p.parse_args()
    repo = args.repo.resolve()
    def git(*words):
        return subprocess.check_output(['git', '-C', str(repo), *words])
    commit = git('rev-parse', 'HEAD').decode().strip()
    short = commit[:7]
    prefix = 'source_' + short
    paths = ['gw_core.py']
    for directory in ('extensions/msbpg_20261005', 'extensions/published_stochastic_20261002'):
        paths += git('ls-tree', '-r', '--name-only', commit, '--', directory).decode().splitlines()
    paths += ['extensions/shareseq_25000_20260930/' + name
              for name in ('run_sharded.py', 'sharded_gw.py')]
    paths = sorted(set(paths))
    subprocess.run(['git', '-C', str(repo), 'diff', '--exit-code', commit, '--', *paths], check=True)
    required = {'extensions/msbpg_20261005/' + name for name in
                ('stable_prox.py', 'validate_late_steps.py', 'validate.sbatch',
                 'frozen_baselines.json', 'campaign.json', 'submit_campaign.sh')}
    if not required.issubset(paths):
        raise ValueError('Required MSBPG package files are missing: ' + repr(sorted(required-set(paths))))
    args.output_parent.mkdir(parents=True, exist_ok=False)
    target = args.output_parent / prefix
    target.mkdir()
    hashes = {}
    for name in paths:
        value = git('show', commit + ':' + name)
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        if path.suffix in ('.sh', '.sbatch'):
            path.chmod(0o755)
        hashes[name] = hashlib.sha256(value).hexdigest()
    (target/'SOURCE_COMMIT').write_text(commit + '\n')
    hashes['SOURCE_COMMIT'] = hashlib.sha256((commit+'\n').encode()).hexdigest()
    manifest = ''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items()))
    (target/'SHA256SUMS').write_text(manifest)
    archive = args.output_parent / (prefix + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(target, arcname=prefix)
    receipt = dict(source_commit=commit, source_directory=str(target), file_count=len(hashes),
                   source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
                   archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   archive_bytes=archive.stat().st_size, files=paths,
                   payload_scope='Git-tracked code, configs, tests, theory notes, and small frozen baseline receipts; no datasets or manuscripts')
    (args.output_parent/'staging.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, indent=2))


if __name__ == '__main__':
    main()
