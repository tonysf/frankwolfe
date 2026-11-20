#!/usr/bin/env python3
"""Verify this GW roundup or reconstruct an origin's selected file layout.

Only Python's standard library is used. All records/hashes/paths are validated
before any output is written. Existing output files and symlinks are refused.
"""
from pathlib import Path
import argparse
import hashlib
import json


def no_symlinks(path):
    path = path.absolute()
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise ValueError(f'Symlink refused: {part}')
    return path


def inside(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts:
        raise ValueError(f'Unsafe manifest path: {relative}')
    path = no_symlinks(root / rel)
    path.relative_to(root.absolute())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', choices=('elite', 'ruche'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = no_symlinks(Path(__file__).absolute().parent)
    inventory = inside(root, 'inventory.json')
    manifest = json.loads(inventory.read_text())
    records = [x for x in manifest['files'] if not args.origin or x['origin'] == args.origin]
    if args.out and not args.origin:
        parser.error('--out requires --origin')
    out = no_symlinks(args.out) if args.out else None
    if out and out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('--out must be absent or an empty directory')

    verified = {}
    plans = {}
    # Prevalidate the entire operation before creating the destination.
    for record in records:
        source = inside(root, record['stored_path'])
        if not source.is_file():
            raise ValueError(f'Missing/non-regular source: {source}')
        if record['stored_path'] not in verified:
            payload = source.read_bytes()
            verified[record['stored_path']] = (
                hashlib.sha256(payload).hexdigest(), len(payload)
            )
        expected = (record['sha256'], record['bytes'])
        if verified[record['stored_path']] != expected:
            raise ValueError(f"Hash/size mismatch: {record['stored_path']}")
        # Validate logical paths even in verify-only mode.
        destination = inside(out or root, record['path'])
        if out:
            previous = plans.get(destination)
            if previous and previous[1] != expected:
                raise ValueError(f'Conflicting duplicate destination: {destination}')
            plans.setdefault(destination, (source, expected))
    for destination in plans:
        if any(parent in plans for parent in destination.parents):
            raise ValueError(f'File/directory destination conflict: {destination}')
        if destination.exists():
            raise ValueError(f'Existing destination refused: {destination}')

    if out:
        out.mkdir(parents=True, exist_ok=True)
        for destination, (source, expected) in plans.items():
            no_symlinks(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents an existing file from being overwritten.
            with destination.open('xb') as stream:
                stream.write(source.read_bytes())
    print(f'Verified {len(records)} logical files in {len(verified)} stored files.')
    if out:
        print(f'Reconstructed {len(plans)} selected {args.origin} paths in {out}')


if __name__ == '__main__':
    main()
