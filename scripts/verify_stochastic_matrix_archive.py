#!/usr/bin/env python3
"""Verify frozen matrix-factorization evidence without importing solver code."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVE = REPOSITORY / "results" / "stochastic_matrix_factorization"


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {value!r}")
    return str(path)


def digest_stream(stream):
    digest = hashlib.sha256()
    count = 0
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return count, digest.hexdigest()
        count += len(block)
        digest.update(block)


def check_file(root, record):
    name = relative_path(record["path"])
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular archive file: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive file leaves its root: {path}")
    with path.open("rb") as stream:
        size, digest = digest_stream(stream)
    if (size, digest) != (record["size"], record["sha256"]):
        raise ValueError(f"Archive hash/size mismatch: {path}")
    return size


def verify_study(root):
    manifest = json.loads((root / "archive_manifest.json").read_text())
    seen = set()
    file_count = byte_count = trace_count = 0
    for record in manifest["files"]:
        path = relative_path(record["path"])
        if path in seen:
            raise ValueError(f"Duplicate manifest file: {path}")
        seen.add(path)
        byte_count += check_file(root, record)
        file_count += 1
    for record in manifest.get("compressed_archives", []):
        archive_path = relative_path(record["path"])
        if archive_path not in seen:
            byte_count += check_file(root, record)
            file_count += 1
            seen.add(archive_path)
        else:
            check_file(root, record)
        expected = {}
        for member in record["members"]:
            name = relative_path(member["path"])
            if name in expected:
                raise ValueError(f"Duplicate expected tar member: {name}")
            expected[name] = member
        found = set()
        with tarfile.open(root / archive_path, mode="r|gz") as archive:
            for member in archive:
                name = relative_path(member.name)
                if not member.isfile() or name in found or name not in expected:
                    raise ValueError(f"Unexpected or nonregular tar member: {name}")
                found.add(name)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"Unreadable tar member: {name}")
                with stream:
                    size, digest = digest_stream(stream)
                wanted = expected[name]
                if (size, digest) != (wanted["size"], wanted["sha256"]):
                    raise ValueError(f"Compressed member hash/size mismatch: {name}")
                trace_count += 1
        if found != set(expected):
            raise ValueError("Compressed archive is missing expected members")
    return {"study": root.name, "files": file_count, "bytes": byte_count,
            "compressed_members": trace_count, "status": "verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ARCHIVE,
                        help="Archive root containing ct/ and factorization/.")
    args = parser.parse_args()
    for study in ("factorization", "ct"):
        print(json.dumps(verify_study(args.root / study), sort_keys=True))


if __name__ == "__main__":
    main()
