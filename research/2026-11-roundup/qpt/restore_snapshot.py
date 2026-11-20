#!/usr/bin/env python3
"""Restore selected QPT archive paths into a fresh directory without running them."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def safe_join(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe manifest path: ' + relative)
    destination = root / rel
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError('Manifest path escapes root: ' + relative)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Refusing a symlink in a manifest path: ' + str(current))
    return destination


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='', help='Logical prefix, e.g. elite/qpt_frames_n4_tuning_20261012')
    parser.add_argument('--destination', type=Path, help='Fresh output directory; no differing file is overwritten')
    parser.add_argument('--repository-root', type=Path, help='frankwolfe checkout supplying byte-identical repository references')
    parser.add_argument('--list', action='store_true', help='List matching logical paths without writing')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'inventory.json').read_text())
    entries = [entry for entry in manifest['files'] if entry['logical_path'].startswith(args.prefix)]
    if args.list:
        print('\n'.join(entry['logical_path'] for entry in entries))
        return
    if args.destination is None:
        parser.error('--destination is required unless --list is used')
    repo = args.repository_root
    if repo is None:
        repo = next((parent for parent in root.parents if (parent / 'frank_wolfe').is_dir()), None)
    if not entries:
        parser.error('No logical paths matched --prefix')
    if args.destination.is_symlink():
        raise ValueError('Destination cannot be a symlink')
    destination = args.destination.resolve()
    if destination.exists() and (destination.is_symlink() or not destination.is_dir()):
        raise ValueError('Destination must be a directory')
    # Validate every source and destination before the first write.
    plan = []
    for entry in entries:
        if 'archive_path' in entry:
            source = safe_join(root, entry['archive_path'])
        else:
            if repo is None:
                raise ValueError('--repository-root is needed for repository references')
            source = safe_join(repo.resolve(), entry['repository_path'])
        target = safe_join(destination, entry['logical_path'])
        if not source.is_file() or sha256(source) != entry['sha256']:
            raise ValueError('Source missing or checksum mismatch: ' + str(source))
        if target.exists() and (not target.is_file() or target.is_symlink() or sha256(target) != entry['sha256']):
            raise ValueError('Refusing to overwrite an existing different path: ' + str(target))
        plan.append((source, target, entry['sha256']))
    for source, target, expected in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # Exclusive creation also protects against a concurrently created target.
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        if sha256(target) != expected:
            raise ValueError('Restored checksum mismatch: ' + str(target))
    print('Restored {} verified files to {}'.format(len(plan), destination))


if __name__ == '__main__':
    main()
