"""Export this isolated 8192-probe experiment from a committed, clean source tree."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    def git(*words):
        return subprocess.check_output(['git', '-C', str(args.repo), *words])
    commit = git('rev-parse', 'HEAD').decode().strip()
    paths = ['gw_core.py', 'extensions/shareseq_25000_20260930/run_sharded.py',
             'extensions/shareseq_25000_20260930/sharded_gw.py']
    for directory in ('probe8192_20261009', 'batch_size_20261008', 'published_stochastic_20261002', 'msbpg_20261005'):
        paths += git('ls-tree', '-r', '--name-only', commit, '--', 'extensions/'+directory).decode().splitlines()
    paths = sorted(set(paths))
    subprocess.run(['git', '-C', str(args.repo), 'diff', '--exit-code', commit, '--', *paths], check=True)
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = 'source_'+commit[:7]
    target = args.out/prefix
    target.mkdir()
    hashes = {}
    for name in paths:
        content = git('show', commit+':'+name)
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if path.suffix in ('.sh', '.sbatch'):
            path.chmod(0o755)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (target/'SOURCE_COMMIT').write_text(commit+'\n')
    hashes['SOURCE_COMMIT'] = hashlib.sha256((commit+'\n').encode()).hexdigest()
    manifest = ''.join(f'{digest}  {name}\n' for name, digest in sorted(hashes.items()))
    (target/'SHA256SUMS').write_text(manifest)
    archive = args.out/(prefix+'.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(target, arcname=prefix)
    receipt = dict(source_commit=commit, source_directory=str(target.resolve()),
                   archive=str(archive.resolve()), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   source_manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(), files=paths)
    (args.out/'staging.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
