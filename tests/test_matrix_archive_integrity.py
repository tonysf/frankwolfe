"""The archive checker detects corruption and unsafe compressed evidence."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "matrix_archive_integrity",
    Path(__file__).resolve().parents[1] / "scripts/verify_stochastic_matrix_archive.py",
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def record(path, content):
    return {"path": path, "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def write_manifest(root, files, archives=()):
    (root / "archive_manifest.json").write_text(json.dumps(
        {"files": files, "compressed_archives": list(archives)}))


def test_source_corruption_is_detected(tmp_path):
    source = b"frozen numerical source\n"
    (tmp_path / "run.py").write_bytes(source)
    write_manifest(tmp_path, [record("run.py", source)])
    assert CHECKER.verify_study(tmp_path)["status"] == "verified"
    (tmp_path / "run.py").write_bytes(b"modified numerical source\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        CHECKER.verify_study(tmp_path)


def make_archive(root, member_name, data, expected_name=None):
    path = root / "reported_traces.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    result = record(path.name, path.read_bytes())
    result["members"] = [record(expected_name or member_name, data)]
    write_manifest(root, [], [result])
    return result


def test_compressed_original_bytes_are_verified(tmp_path):
    archive = make_archive(tmp_path, "runs/one/trace.json", b"[{\"iteration\":0}]")
    assert CHECKER.verify_study(tmp_path)["compressed_members"] == 1
    archive["members"][0]["sha256"] = "0" * 64
    write_manifest(tmp_path, [], [archive])
    with pytest.raises(ValueError, match="member hash/size mismatch"):
        CHECKER.verify_study(tmp_path)


def test_tar_path_traversal_is_rejected(tmp_path):
    make_archive(tmp_path, "../outside.json", b"[]", "runs/trace.json")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        CHECKER.verify_study(tmp_path)
    assert not (tmp_path.parent / "outside.json").exists()


def test_missing_compressed_member_is_rejected(tmp_path):
    archive = make_archive(tmp_path, "runs/one.json", b"[]")
    archive["members"].append(record("runs/two.json", b"[]"))
    write_manifest(tmp_path, [], [archive])
    with pytest.raises(ValueError, match="missing expected members"):
        CHECKER.verify_study(tmp_path)
