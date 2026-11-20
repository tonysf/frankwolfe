"""Check this experiment snapshot without importing its numerical code."""

import hashlib
import json
from pathlib import Path
import sys


def verify(root):
    manifest = json.loads((root / "file_manifest.json").read_text())
    expected = {}
    errors = []
    for entry in manifest["files"]:
        relative = entry["path"]
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"unsafe manifest path: {relative}")
            continue
        if relative in expected:
            errors.append(f"duplicate manifest path: {relative}")
        expected[relative] = entry
        if path.is_symlink() or not path.is_file():
            errors.append(f"missing or nonregular file: {relative}")
            continue
        payload = path.read_bytes()
        if len(payload) != entry["bytes"]:
            errors.append(f"size mismatch: {relative}")
        if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            errors.append(f"SHA-256 mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    }
    actual.discard("file_manifest.json")
    for relative in sorted(actual - set(expected)):
        errors.append(f"unexpected file: {relative}")
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"Verified {len(expected)} files ({sum(e['bytes'] for e in expected.values()):,} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(verify(Path(__file__).resolve().parent))
