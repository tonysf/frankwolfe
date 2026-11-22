"""Lossless evidence packaging and publication guards, using only local Git refs."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

from paper.experiments.stochastic_ct import bundle
from paper.experiments.stochastic_ct import nonnegative_git_export as predecessor
from paper.experiments.stochastic_ct import robustness_export as exporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def initialize_repo(repo):
    repo.mkdir()
    git(repo, "init", "--initial-branch=stochastic-frames")
    git(repo, "config", "user.name", "Local robustness export test")
    git(repo, "config", "user.email", "robustness-test@example.invalid")


def test_trace_parts_are_lossless_deterministic_and_exclude_restart_data(tmp_path):
    root, archive = tmp_path / "campaign", tmp_path / "archive"
    archive.mkdir()
    expected = {}
    for i in range(4):
        for name in ("metadata.json", "trace.json", "result.json", "candidate_failure.json"):
            path = root / "runs" / f"case_{i}" / name
            write(path, dict(case=i, payload="dense observation " * 100, name=name))
            expected[str(path.relative_to(root))] = path.read_bytes()
        for name in ("checkpoint.pkl", "data.npz", "calibration.npz", "progress.json", "private.log"):
            (path.parent / name).write_bytes(b"must not be archived")
    parts = exporter.pack_runs(root, archive, limit=5000)
    assert len(parts) > 1
    recovered = {}
    for part in parts:
        path = archive / part["path"]
        assert sha(path) == part["sha256"]
        assert path.stat().st_size == part["size"]
        assert sum(item["size"] for item in part["members"]) <= 5000
        with tarfile.open(path) as packed:
            assert set(packed.getnames()) == {item["path"] for item in part["members"]}
            for info, item in zip(packed.getmembers(), part["members"]):
                assert info.isfile() and info.uid == info.gid == info.mtime == 0
                content = packed.extractfile(info).read()
                assert hashlib.sha256(content).hexdigest() == item["sha256"]
                assert len(content) == item["size"]
                assert info.name not in recovered
                recovered[info.name] = content
    assert recovered == expected
    assert exporter.pack_runs(root, archive, limit=5000) == parts


@pytest.mark.parametrize("kind", ["symlink", "outside_directory", "oversize"])
def test_trace_archive_rejects_unsafe_or_oversize_members(tmp_path, kind):
    root, archive = tmp_path / "campaign", tmp_path / "archive"
    root.mkdir()
    archive.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside / "trace.json", dict(outside=True))
    case = root / "runs/case"
    case.parent.mkdir()
    if kind == "outside_directory":
        case.symlink_to(outside, target_is_directory=True)
    else:
        case.mkdir()
        if kind == "symlink":
            (case / "trace.json").symlink_to(outside / "trace.json")
        else:
            (case / "trace.json").write_bytes(b"x" * 1001)
    with pytest.raises(ValueError, match="Archive members|archive part limit"):
        exporter.pack_runs(root, archive, limit=1000)


@pytest.fixture
def scientific_export(tmp_path):
    repo = tmp_path / "repo"
    root = tmp_path / "ct/spectral_balls/expanded/reviewer_robustness_20261122"
    root.mkdir(parents=True)
    initialize_repo(repo)
    captured = repo / exporter.BUNDLE
    original = json.loads((bundle.BUNDLE / "manifest.json").read_text())
    for name in [*original["files"], "manifest.json"]:
        destination = captured / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle.BUNDLE / name, destination)
    # Existing archive links must remain valid in a full published checkout.
    for name in ("README.md", "CT_REPRODUCIBILITY.md"):
        for _, relative in bundle.local_markdown_links(captured / name):
            destination = (captured / relative).resolve()
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("Unchanged predecessor evidence.\n")
    git(repo, "add", "-f", ".")
    git(repo, "commit", "-m", "Capture predecessor bundle in local fixture")
    head = git(repo, "rev-parse", "HEAD")
    configs = []
    for phase in ("development", "evaluation"):
        run_id = phase + "_fixture"
        cfg = dict(run_id=run_id, phase=phase, method="sframes")
        configs.append(cfg)
        folder = root / "runs" / run_id
        meta = dict(config=cfg, fixture=True)
        write(folder / "metadata.json", meta)
        write(folder / "trace.json", [dict(iteration=0, fixture=True)])
        write(folder / "result.json", dict(status="completed", metadata=meta))
        (folder / "checkpoint.pkl").write_bytes(b"external restart state")
    write(root / "plans/all.json", dict(runs=configs))
    write(root / "stages/selection_audit.json", dict(selected=True))
    (root / "frozen_sources").mkdir()
    (root / "frozen_sources/example.py").write_text("# Pinned numerical source fixture.\n")
    (root / "raw_data.npz").write_bytes(b"external raw CT arrays")
    write(root / "protocol.json", dict(fixture=True))
    write(root / "selection.json", dict(selected=True))
    write(root / "completion.json", dict(status="completed", evaluation_runs=1,
          protocol_sha256=sha(root / "protocol.json")))
    write(root / "report_summary.json", dict(stage="completed", planned_physical_runs=2,
          completed_physical_runs=2, pending_runs=[], failed_runs=[]))
    stage = root / "report_stage_fixture"
    (stage / "report").mkdir(parents=True)
    runtime = root.parent / "reports/paper_section/revised"
    runtime.mkdir(parents=True)
    for name in exporter.backend.REPORT_FILES:
        shutil.copyfile(captured / "report" / name, stage / "report" / name)
        shutil.copyfile(captured / "report" / name, runtime / name)
    public_appendix = (captured / "CT_REPRODUCIBILITY.md").read_text()
    (stage / "CT_REPRODUCIBILITY.md").write_text(public_appendix + "\n[Expanded protocol](ROBUSTNESS_EVALUATION.md)\n")
    (stage / "ROBUSTNESS_RESULTS.md").write_text("# Complete results\n\nSynthetic local fixture.\n")
    (stage / "ROBUSTNESS_EVALUATION.md").write_text("# Protocol\n\nSynthetic local fixture.\n")
    statistics = {}
    for n in (360, 1080):
        path = stage / "report" / f"statistics_{n}.json"
        write(path, dict(training_views=n, observations=2))
        statistics[path.name] = sha(path)
    audit = dict(status="completed_published", generated_utc="2026-11-22T12:00:00+00:00",
        stage=str(stage), pages=5, figures=2, tables=1, physical_runs=2,
        evaluation_runs=1, development_runs=1, failed_development_runs=0,
        protocol_sha256=sha(root / "protocol.json"), summary_sha256=sha(root / "report_summary.json"),
        published_sha256={name: sha(stage / "report" / name) for name in exporter.backend.REPORT_FILES},
        statistics=statistics,
        supplement_sha256={name: sha(stage / name) for name in
                           ("CT_REPRODUCIBILITY.md", "ROBUSTNESS_RESULTS.md", "ROBUSTNESS_EVALUATION.md")})
    publication = dict(repo=str(repo), expected_head=head, expected_branch="stochastic-frames",
                       runtime_report=str(runtime), runtime_assets={str(runtime / name): sha(runtime / name)
                       for name in exporter.backend.REPORT_FILES})
    write(root / "staged_report.json", audit)
    write(root / "publication_config.json", publication)
    return root, repo, publication, audit, original


def test_scientific_stage_is_scoped_and_preserves_report_provenance(scientific_export):
    root, repo, publication, audit, original = scientific_export
    plan = exporter.build_stage(root, publication, audit)
    stage = Path(plan["stage_files"])
    archive = stage / exporter.ARCHIVE
    captured = stage / exporter.BUNDLE
    manifest = json.loads((captured / "manifest.json").read_text())
    assert manifest["snapshot"]["physical_trajectories"] == 1
    assert manifest["snapshot"]["development_runs"] == 1
    assert manifest["snapshot"]["figures"] == 2 and manifest["snapshot"]["tables"] == 1
    assert manifest["provenance"]["predecessor_snapshot"] == original["snapshot"]
    assert manifest["provenance"]["predecessor_pdf_sha256"] == original["files"]["report/sframes_ct_section.pdf"]["sha256"]
    assert manifest["provenance"]["publication_audit"]["sha256"] == sha(archive / "report_update.json")
    expected_source = (Path(audit["stage"]) / "CT_REPRODUCIBILITY.md").read_text()
    for relocation in reversed(original["appendix_link_relocations"]):
        expected_source = expected_source.replace(relocation["to"], relocation["from"])
    assert (captured / "provenance/CT_REPRODUCIBILITY.source.md").read_text() == expected_source
    for name in ("ROBUSTNESS_RESULTS.md", "ROBUSTNESS_EVALUATION.md"):
        record = manifest["files"][name]
        assert record["sha256"] == record["original_sha256"] == sha(captured / name)
        assert record["source_relative_path"].startswith("spectral_balls/expanded/reports/")
        assert plan["input_files"][str(Path(audit["stage"]) / name)] == sha(captured / name)
    assert bundle.validate_bundle(captured)["status"] == "verified"
    assert all(exporter.backend.allowed_path(name) for name in plan["files"])
    assert not any(name.endswith((".npz", ".pkl", ".log")) or "__pycache__" in name for name in plan["files"])
    archived = json.loads((archive / "archive_manifest.json").read_text())
    for record in archived["files"]:
        assert sha(archive / record["path"]) == record["sha256"]
    assert archived["previous_archives_preserved"] is True
    for name, expected in audit["statistics"].items():
        assert sha(archive / "report" / name) == expected
    for part in archived["compressed_archives"]:
        with tarfile.open(archive / part["path"]) as packed:
            for info in packed:
                assert packed.extractfile(info).read() == (root / info.name).read_bytes()
    for name in audit["published_sha256"]:
        assert plan["input_files"][str(Path(audit["stage"]) / "report" / name)] == audit["published_sha256"][name]
    assert exporter.build_stage(root, publication, audit) == plan
    assert git(repo, "rev-parse", "HEAD") == publication["expected_head"]
    assert git(repo, "status", "--porcelain") == ""


def test_full_export_publishes_only_after_validation_and_updates_runtime(scientific_export, tmp_path):
    root, repo, publication, audit, _ = scientific_export
    runtime = Path(publication["runtime_report"])
    editorial = runtime / "editorial_revision.json"
    write(editorial, dict(description="Historical editorial change", evaluation_audit=dict(old=True),
                         pdf_sha256=sha(runtime / "sframes_ct_section.pdf")))
    previous_editorial = editorial.read_bytes()
    publication["runtime_assets"][str(editorial)] = sha(editorial)
    remote = tmp_path / "publication.git"
    remote.mkdir()
    git(remote, "init", "--bare", "--initial-branch=stochastic-frames")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-u", "origin", "stochastic-frames")
    publication.update(remote="origin", remote_branch="stochastic-frames", remote_url=str(remote))
    write(root / "publication_config.json", publication)
    result = exporter.export(root)
    assert result["status"] == "completed_pushed"
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == result["commit"]
    assert json.loads((root / "report_update.json").read_text())["git_commit"] == result["commit"]
    for name in exporter.backend.REPORT_FILES:
        assert (runtime / name).read_bytes() == (repo / exporter.BUNDLE / "report" / name).read_bytes()
        assert sha(root / "report_previous" / name) == publication["runtime_assets"][str(runtime / name)]
    assert (runtime.parent / "CT_REPRODUCIBILITY.md").read_bytes() == (
        repo / exporter.BUNDLE / "provenance/CT_REPRODUCIBILITY.source.md").read_bytes()
    assert (root / "report_previous/editorial_revision.json").read_bytes() == previous_editorial
    updated_editorial = json.loads(editorial.read_text())
    assert updated_editorial["pdf_sha256"] == sha(runtime / "sframes_ct_section.pdf")
    assert updated_editorial["tex_sha256"] == sha(runtime / "ct_experiments.tex")
    assert updated_editorial["publication_audit"]["git_commit"] == result["commit"]
    assert updated_editorial["numerical_results_unchanged"] is False
    assert "evaluation_audit" not in updated_editorial
    assert bundle.validate_bundle(repo / exporter.BUNDLE, require_archive=True)["status"] == "verified"
    assert exporter.export(root)["commit"] == result["commit"]
    assert git(repo, "rev-list", "--count", "HEAD") == "2"


@pytest.mark.parametrize("frozen", [True, False])
def test_concurrent_or_unfrozen_editorial_revision_blocks_publication(scientific_export, frozen):
    root, repo, publication, _, _ = scientific_export
    editorial = Path(publication["runtime_report"]) / "editorial_revision.json"
    write(editorial, dict(description="Previous editorial record"))
    if frozen:
        publication["runtime_assets"][str(editorial)] = sha(editorial)
    write(root / "publication_config.json", publication)
    write(editorial, dict(description="Concurrent editorial work"))
    with pytest.raises(ValueError, match="editorial revision was not frozen|Runtime publication changed"):
        exporter.export(root)
    assert git(repo, "rev-parse", "HEAD") == publication["expected_head"]
    assert not (root / "report_previous").exists()
    assert not (root / "report_update.json").exists()
    assert json.loads(editorial.read_text())["description"] == "Concurrent editorial work"


@pytest.mark.parametrize("mutation", ["incomplete", "pending", "evaluation_failure", "plan_count",
    "protocol", "report", "statistics", "supplement", "missing_asset"])
def test_incomplete_or_modified_evidence_cannot_be_staged(scientific_export, mutation):
    root, _, publication, audit, _ = scientific_export
    if mutation in ("pending", "evaluation_failure", "plan_count"):
        path = root / "report_summary.json"
        summary = json.loads(path.read_text())
        if mutation == "pending":
            summary["pending_runs"] = ["evaluation_fixture"]
        elif mutation == "evaluation_failure":
            summary["failed_runs"] = [dict(run_id="evaluation_fixture")]
        else:
            summary["planned_physical_runs"] = 1
        write(path, summary)
        audit["summary_sha256"] = sha(path)
    elif mutation == "incomplete":
        write(root / "completion.json", dict(status="pending"))
    elif mutation == "protocol":
        write(root / "protocol.json", dict(modified=True))
    elif mutation == "missing_asset":
        audit["published_sha256"].pop("sframes_ct_section.pdf")
    else:
        relative = {"report": "report/sframes_ct_section.pdf", "statistics": "report/statistics_360.json",
                    "supplement": "ROBUSTNESS_RESULTS.md"}[mutation]
        (Path(audit["stage"]) / relative).write_text("Changed after scientific validation.\n")
    with pytest.raises(ValueError):
        exporter.build_stage(root, publication, audit)
    assert not (root / "git_export_stage").exists()


@pytest.mark.parametrize("mutation", ["completed_count", "physical_count", "development_count", "evaluation_count",
    "failed_count", "duplicate_failure", "unknown_failure", "undeclared_failure", "missing_result",
    "missing_trace", "running_evaluation", "failed_evaluation", "wrong_metadata", "wrong_embedded_metadata",
    "false_failure", "incomplete_summary"])
def test_aggregate_counts_and_actual_outcomes_must_agree(scientific_export, mutation):
    root, _, publication, audit, _ = scientific_export
    summary = json.loads((root / "report_summary.json").read_text())
    folder = root / "runs/evaluation_fixture"
    if mutation == "completed_count":
        summary["completed_physical_runs"] = 0
    elif mutation in ("physical_count", "development_count", "evaluation_count"):
        audit[mutation.replace("_count", "_runs")] = 0
        if mutation == "evaluation_count":
            completion = json.loads((root / "completion.json").read_text())
            completion["evaluation_runs"] = 0
            write(root / "completion.json", completion)
    elif mutation == "failed_count":
        audit["failed_development_runs"] = 1
    elif mutation in ("duplicate_failure", "unknown_failure", "false_failure"):
        key = "missing_run" if mutation == "unknown_failure" else "development_fixture"
        summary["failed_runs"] = [dict(run_id=key, status="numerical_failure")]
        if mutation == "duplicate_failure":
            summary["failed_runs"] *= 2
        summary["completed_physical_runs"] = 2 - len(summary["failed_runs"])
        audit["physical_runs"] = summary["completed_physical_runs"]
        audit["failed_development_runs"] = len(summary["failed_runs"])
    elif mutation == "incomplete_summary":
        summary["stage"] = "development"
    elif mutation.startswith("missing_"):
        (folder / ("result.json" if mutation == "missing_result" else "trace.json")).unlink()
    elif mutation == "wrong_metadata":
        write(folder / "metadata.json", dict(config=dict(phase="evaluation", method="other")))
    else:
        if mutation == "undeclared_failure":
            folder = root / "runs/development_fixture"
        result = json.loads((folder / "result.json").read_text())
        if mutation == "wrong_embedded_metadata":
            result["metadata"] = dict(config={})
        else:
            result["status"] = "running" if mutation == "running_evaluation" else "numerical_failure"
        write(folder / "result.json", result)
    write(root / "report_summary.json", summary)
    audit["summary_sha256"] = sha(root / "report_summary.json")
    with pytest.raises(ValueError):
        exporter.build_stage(root, publication, audit)
    assert not (root / "git_export_stage").exists()


@pytest.mark.parametrize("failure_format", ["numerical_result", "candidate_failure"])
def test_declared_development_failures_remain_publishable_evidence(scientific_export, failure_format):
    root, _, publication, audit, _ = scientific_export
    folder = root / "runs/development_fixture"
    meta = json.loads((folder / "metadata.json").read_text())
    if failure_format == "candidate_failure":
        for name in ("metadata.json", "trace.json", "result.json"):
            (folder / name).unlink()
        write(folder / "candidate_failure.json", dict(config=meta["config"], error="Nonfinite candidate"))
    else:
        write(folder / "result.json", dict(status="numerical_failure", metadata=meta, error="Nonfinite candidate"))
    summary = json.loads((root / "report_summary.json").read_text())
    summary.update(completed_physical_runs=1,
                   failed_runs=[dict(run_id="development_fixture", status="numerical_failure")])
    write(root / "report_summary.json", summary)
    audit.update(physical_runs=1, failed_development_runs=1, summary_sha256=sha(root / "report_summary.json"))
    write(root / "staged_report.json", audit)
    plan = exporter.build_stage(root, publication, audit)
    assert plan["bundle_validation"]["status"] == "verified"
    archive = Path(plan["stage_files"]) / exporter.ARCHIVE
    manifest = json.loads((archive / "archive_manifest.json").read_text())
    archived = {item["path"] for part in manifest["compressed_archives"] for item in part["members"]}
    expected = "candidate_failure.json" if failure_format == "candidate_failure" else "result.json"
    assert "runs/development_fixture/" + expected in archived


@pytest.fixture
def local_publication(tmp_path):
    repo, remote, root = tmp_path / "repo", tmp_path / "remote.git", tmp_path / "runtime"
    initialize_repo(repo)
    root.mkdir()
    remote.mkdir()
    git(remote, "init", "--bare", "--initial-branch=stochastic-frames")
    (repo / "README.md").write_text("Unrelated documentation.\n")
    old_report = repo / exporter.BUNDLE / "README.md"
    old_report.parent.mkdir(parents=True)
    old_report.write_text("Original report.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Initial local fixture")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-u", "origin", "stochastic-frames")
    head = git(repo, "rev-parse", "HEAD")
    stage = root / "git_export_stage/files"
    files = {}
    for name, content in {exporter.ARCHIVE + "/archive_manifest.json": b'{"fixture": true}\n',
                          exporter.BUNDLE + "/README.md": b"Updated report.\n"}.items():
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[name] = exporter.backend.record(path)
    evidence = root / "scientific_evidence.json"
    write(evidence, dict(validated=True))
    plan = dict(stage_files=str(stage), files=files, input_files={str(evidence): sha(evidence)},
                expected_head=head, expected_branch="stochastic-frames")
    helper = dict(expected_head=head, expected_branch="stochastic-frames", remote="origin",
                  remote_branch="stochastic-frames", remote_url=str(remote))
    return repo, remote, root, plan, helper


def test_guarded_publication_scopes_commit_and_is_idempotent(local_publication):
    repo, remote, root, plan, helper = local_publication
    result = exporter.backend.publish_stage(root, repo, plan, helper)
    assert result["status"] == "completed_pushed"
    commit = git(repo, "rev-parse", "HEAD")
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == commit
    assert set(git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()) == set(plan["files"])
    assert git(repo, "log", "-1", "--format=%s") == "Publish the expanded SFRAMES mechanism and robustness evaluation"
    assert exporter.backend.publish_stage(root, repo, plan, helper)["status"] == "completed_pushed"
    assert git(repo, "rev-list", "--count", "HEAD") == "2"
    assert git(repo, "status", "--porcelain") == ""
    assert (repo / "README.md").read_text() == "Unrelated documentation.\n"


@pytest.mark.parametrize("mutation", ["tracked", "untracked", "evidence", "stage", "remote"])
def test_concurrent_changes_block_publication_without_installation(local_publication, mutation):
    repo, remote, root, plan, helper = local_publication
    if mutation in ("tracked", "untracked"):
        (repo / ("README.md" if mutation == "tracked" else "notes.txt")).write_text("Concurrent work.\n")
    elif mutation == "evidence":
        write(root / "scientific_evidence.json", dict(changed=True))
    elif mutation == "stage":
        (Path(plan["stage_files"]) / exporter.BUNDLE / "README.md").write_text("Unvalidated change.\n")
    else:
        # Advance the bare local branch without changing this checkout.
        commit = subprocess.check_output(["git", "-C", str(repo), "commit-tree", "HEAD^{tree}",
                                          "-p", "HEAD", "-m", "Concurrent remote commit"], text=True).strip()
        git(repo, "push", "origin", commit + ":refs/heads/stochastic-frames")
    result = exporter.backend.publish_stage(root, repo, plan, helper)
    assert result["status"] == "pending_git_review"
    assert git(repo, "rev-parse", "HEAD") == helper["expected_head"]
    assert not (repo / exporter.ARCHIVE).exists()
    assert (repo / exporter.BUNDLE / "README.md").read_text() == "Original report.\n"
    if mutation in ("tracked", "untracked"):
        assert (repo / ("README.md" if mutation == "tracked" else "notes.txt")).read_text() == "Concurrent work.\n"


def test_rejected_push_resumes_same_validated_commit(local_publication):
    repo, remote, root, plan, helper = local_publication
    hook = remote / "hooks/pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    pending = exporter.backend.publish_stage(root, repo, plan, helper)
    assert pending["status"] == "pending_push"
    commit = git(repo, "rev-parse", "HEAD")
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == helper["expected_head"]
    hook.unlink()
    result = exporter.backend.publish_stage(root, repo, plan, helper)
    assert result["status"] == "completed_pushed"
    assert result["commit"] == commit == git(repo, "rev-parse", "HEAD")
    assert git(repo, "rev-list", "--count", "HEAD") == "2"


def test_remote_advance_after_initial_guard_cannot_be_overwritten(local_publication, monkeypatch):
    repo, remote, root, plan, helper = local_publication
    original = exporter.backend.remote_head
    concurrent = []

    def remote_head_with_race(*args):
        head = original(*args)
        if not concurrent:
            commit = git(repo, "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "Remote race fixture")
            git(repo, "push", "origin", commit + ":refs/heads/stochastic-frames")
            concurrent.append(commit)
        return head

    monkeypatch.setattr(exporter.backend, "remote_head", remote_head_with_race)
    result = exporter.backend.publish_stage(root, repo, plan, helper)
    assert result["status"] == "pending_push"
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == concurrent[0]
    assert git(repo, "rev-parse", "HEAD") == result["commit"] != concurrent[0]
    # The retry also stops without a force push or automatic merge.
    again = exporter.backend.publish_stage(root, repo, plan, helper)
    assert again["status"] == "pending_push"
    assert "Remote branch advanced" in again["error"]
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == concurrent[0]


def test_exporter_whitelist_is_private_and_cannot_change_old_archives():
    assert predecessor.ARCHIVE == "results/stochastic_matrix_factorization/ct_nonnegative"
    assert "ROBUSTNESS_RESULTS.md" not in predecessor.BUNDLE_MUTATIONS
    for name in ("../escape.json", "/absolute.json", exporter.ARCHIVE + "/../old.json"):
        with pytest.raises(ValueError, match="inside the repository"):
            exporter.backend.allowed_path(name)
    for name in ("README.md",
                 "results/stochastic_matrix_factorization/ct/archive_manifest.json",
                 "results/stochastic_matrix_factorization/ct_nonnegative/report_update.json"):
        assert not exporter.backend.allowed_path(name)
    assert exporter.backend.allowed_path(exporter.ARCHIVE + "/runs-0000.tar.gz")
    assert exporter.backend.allowed_path(exporter.BUNDLE + "/ROBUSTNESS_RESULTS.md")
