"""Integrity and portability checks for the captured CT paper section."""

import json
from pathlib import Path
import shutil

import pytest

from paper.experiments.stochastic_ct import bundle


def test_captured_bundle_and_archived_links():
    result = bundle.validate_bundle(require_archive=True)
    assert result["snapshot_date"] == "2026-11-19"
    assert result["figures"] == 2 and result["tables"] == 1
    assert result["archive_links_verified"]
    assert result["captured_pdf_sha256"] == (
        "b0359dd827c29dfedc7062ff8fa6f2efbe65a26815d98eb6fd7f01ff796862f2"
    )


def test_appendix_relocation_preserves_original_prose():
    manifest = json.loads((bundle.BUNDLE / "manifest.json").read_text())
    original = (bundle.BUNDLE / "provenance/CT_REPRODUCIBILITY.source.md").read_text()
    for replacement in manifest["appendix_link_relocations"]:
        assert replacement["from"] in original
        original = original.replace(replacement["from"], replacement["to"])
    assert original == (bundle.BUNDLE / "CT_REPRODUCIBILITY.md").read_text()
    for name, record in manifest["files"].items():
        if name.startswith("report/"):
            assert record["copy_mode"] == "byte-identical"
            assert record["sha256"] == record["original_sha256"]


def test_tampering_is_rejected_and_build_protects_capture(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(bundle.BUNDLE, copied, ignore=shutil.ignore_patterns("build", "__pycache__"))
    with pytest.raises(ValueError, match="cannot be a build destination"):
        bundle.build_report(copied / "report", root=copied)
    body = copied / "report/ct_experiments.tex"
    original = body.read_bytes()
    body.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="differs from its manifest"):
        bundle.validate_bundle(copied)
    body.write_bytes(original)
    assert bundle.validate_bundle(copied)["status"] == "verified"
