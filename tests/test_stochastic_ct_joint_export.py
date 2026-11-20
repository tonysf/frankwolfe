"""Scoped Git publication and recovery using isolated local bare repositories."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from paper.experiments.stochastic_ct import joint_lbfgsb_git_export as exporter
from paper.experiments.stochastic_ct import nonnegative_git_export as predecessor_exporter
from paper.experiments.stochastic_ct import bundle


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def local_export(tmp_path):
    repo, remote, root = tmp_path / "repo", tmp_path / "remote.git", tmp_path / "runtime"
    repo.mkdir()
    remote.mkdir()
    root.mkdir()
    git(repo, "init", "--initial-branch=stochastic-frames")
    git(remote, "init", "--bare", "--initial-branch=stochastic-frames")
    git(repo, "config", "user.name", "CT archive test")
    git(repo, "config", "user.email", "archive-test@example.invalid")
    (repo / "README.md").write_text("Existing repository documentation.\n")
    report_readme = repo / "paper/experiments/stochastic_ct/README.md"
    report_readme.parent.mkdir(parents=True)
    report_readme.write_text("Previous captured report.\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Initial local fixture")
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-u", "origin", "stochastic-frames")
    initial = git(repo, "rev-parse", "HEAD")
    stage = root / "git_export_stage/files"
    payloads = {
        "results/stochastic_matrix_factorization/ct_joint_lbfgsb/archive_manifest.json": b'{"fixture": true}\n',
        "paper/experiments/stochastic_ct/README.md": b"Updated report, preserved prior scientific evidence.\n",
    }
    files = {}
    for name, content in payloads.items():
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    plan = dict(files=files, expected_head=initial, expected_branch="stochastic-frames", stage_files=str(stage))
    helper = dict(expected_head=initial, expected_branch="stochastic-frames", remote="origin",
                  remote_branch="stochastic-frames", remote_url=str(remote))
    return repo, remote, root, plan, helper


def test_scoped_commit_push_and_repeated_export_are_idempotent(local_export):
    repo, remote, root, plan, helper = local_export
    receipt = exporter.publish_stage(root, repo, plan, helper)
    assert receipt["status"] == "completed_pushed"
    commit = git(repo, "rev-parse", "HEAD")
    assert commit != helper["expected_head"]
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == commit
    assert set(git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()) == set(plan["files"])
    assert not git(repo, "status", "--porcelain")
    assert (repo / "README.md").read_text() == "Existing repository documentation.\n"
    again = exporter.publish_stage(root, repo, plan, helper)
    assert again["status"] == "completed_pushed"
    assert git(repo, "rev-parse", "HEAD") == commit
    assert git(repo, "rev-list", "--count", "HEAD") == "2"


@pytest.mark.parametrize("tracked", [True, False])
def test_concurrent_work_is_preserved_and_blocks_git_publication(local_export, tracked):
    repo, remote, root, plan, helper = local_export
    path = repo / ("README.md" if tracked else "unrelated_user_notes.txt")
    path.write_text("Concurrent user work must survive.\n")
    receipt = exporter.publish_stage(root, repo, plan, helper)
    assert receipt["status"] == "pending_git_review"
    assert git(repo, "rev-parse", "HEAD") == helper["expected_head"]
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == helper["expected_head"]
    assert path.read_text() == "Concurrent user work must survive.\n"
    assert not (repo / "results/stochastic_matrix_factorization/ct_joint_lbfgsb/archive_manifest.json").exists()


def test_failed_push_resumes_the_same_commit(local_export):
    repo, remote, root, plan, helper = local_export
    hook = remote / "hooks/pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    pending = exporter.publish_stage(root, repo, plan, helper)
    assert pending["status"] == "pending_push"
    commit = git(repo, "rev-parse", "HEAD")
    assert commit != helper["expected_head"]
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == helper["expected_head"]
    saved = json.loads((root / "git_export.json").read_text())
    assert saved["commit"] == commit
    hook.unlink()
    resumed = exporter.publish_stage(root, repo, plan, helper)
    assert resumed["status"] == "completed_pushed"
    assert git(repo, "rev-parse", "HEAD") == commit
    assert git(remote, "rev-parse", "refs/heads/stochastic-frames") == commit
    assert git(repo, "rev-list", "--count", "HEAD") == "2"


def test_scientific_stage_archives_both_cohorts_and_refreshes_bundle(tmp_path):
    repo = tmp_path / "repo"
    root = tmp_path / "ct/spectral_balls/expanded/joint_lbfgsb"
    root.mkdir(parents=True)
    repo.mkdir()
    git(repo, "init", "--initial-branch=stochastic-frames")
    git(repo, "config", "user.name", "CT archive test")
    git(repo, "config", "user.email", "archive-test@example.invalid")
    copied_bundle = repo / exporter.BUNDLE
    original = json.loads((bundle.BUNDLE / "manifest.json").read_text())
    for name in [*original["files"], "manifest.json"]:
        target = copied_bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle.BUNDLE / name, target)
    for document in ("README.md", "CT_REPRODUCIBILITY.md"):
        for _, relative in bundle.local_markdown_links(copied_bundle / document):
            target = (copied_bundle / relative).resolve()
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("Fixture placeholder for unchanged predecessor evidence.\n")
    git(repo, "add", "-f", ".")
    git(repo, "commit", "-m", "Captured predecessor bundle")
    head = git(repo, "rev-parse", "HEAD")

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for phase, count in (("development", 12), ("evaluation", 28)):
        configs, cases = [], []
        for i in range(count):
            run_id = f"{phase}_{i:02d}"
            directory = root / "runs" / run_id
            configs.append(dict(run_id=run_id, method="joint_lbfgsb"))
            case = dict(run_id=run_id, method="joint_lbfgsb", source_directory=str(directory))
            for name in ("metadata", "result", "trace"):
                path = directory / (name + ".json")
                write(path, dict(fixture=True, run_id=run_id, kind=name))
                case[name + "_sha256"] = sha(path)
            cases.append(case)
        write(root / (phase + "_plan.json"), dict(runs=configs))
        write(root / ("development_validation.json" if phase == "development" else "validation_summary.json"),
              dict(status="completed_validated", cases=cases))
    write(root / "selection.json", dict(fixture=True, selected_memory=10))
    frozen = root / "frozen_sources/dummy.py"
    frozen.parent.mkdir()
    frozen.write_text("# Synthetic fixture implementation.\n")
    stage = root / "report_stage_fixture"
    stage.mkdir()
    summary = stage / "joint_lbfgsb_comparison_summary.json"
    write(summary, dict(cases=[dict(observations=1) for _ in range(532)]))
    statistics = {}
    for size in (360, 1080):
        path = stage / f"statistics_{size}.json"
        write(path, dict(training_views=size, fixture=True))
        statistics[path.name] = sha(path)
    paper = root.parent / "reports/paper_section/revised"
    paper.mkdir(parents=True)
    assets = {}
    for name in exporter.REPORT_FILES:
        shutil.copyfile(copied_bundle / "report" / name, paper / name)
        assets[name] = sha(paper / name)
    appendix = paper.parent / "CT_REPRODUCIBILITY.md"
    appendix.write_text((copied_bundle / "provenance/CT_REPRODUCIBILITY.source.md").read_text()
                        + "\n[New selection](../../joint_lbfgsb/selection.json)\n")
    assets[appendix.name] = sha(appendix)
    write(root / "report_update.json", dict(status="completed_published", physical_runs=532,
        figures=2, tables=1, pages=5, published_utc="2026-11-20T12:00:00+00:00", stage=str(stage),
        comparison_summary=str(summary), comparison_summary_sha256=sha(summary), statistics=statistics,
        report_pdf=str(paper / "sframes_ct_section.pdf"), published_sha256=assets))
    write(root / "git_export_helper.json", dict(expected_head=head, expected_branch="stochastic-frames"))
    plan = exporter.build_stage(root, repo)
    exported = Path(plan["stage_files"])
    traces = list((exported / exporter.ARCHIVE / "runs").glob("*/trace.json"))
    assert len(traces) == 40
    for path in traces:
        assert path.read_bytes() == (root / path.relative_to(exported / exporter.ARCHIVE)).read_bytes()
    refreshed = json.loads((exported / exporter.BUNDLE / "manifest.json").read_text())
    assert refreshed["snapshot"]["physical_trajectories"] == 532
    assert refreshed["snapshot"]["comparison"] == "equal directional solver-view budget"
    assert not refreshed.get("document_revisions")
    assert "Stochastic gradient estimator" in (exported / exporter.BUNDLE / "report/ct_experiments.tex").read_text()
    assert (exported / exporter.BUNDLE / "report/trajectories_360.pdf").read_bytes() == (paper / "trajectories_360.pdf").read_bytes()
    assert (exported / exporter.BUNDLE / "provenance/CT_REPRODUCIBILITY.source.md").read_bytes() == appendix.read_bytes()
    assert "../../../results/stochastic_matrix_factorization/ct_joint_lbfgsb/selection.json" in (
        exported / exporter.BUNDLE / "CT_REPRODUCIBILITY.md").read_text()
    assert all(exporter.allowed_path(name) for name in plan["files"])
    assert not any("__pycache__" in name for name in plan["files"])
    assert predecessor_exporter.ARCHIVE == "results/stochastic_matrix_factorization/ct_nonnegative"
    assert not exporter.allowed_path("results/stochastic_matrix_factorization/ct_nonnegative/report_update.json")
    assert not exporter.allowed_path("results/stochastic_matrix_factorization/ct/archive_manifest.json")
    assert not git(repo, "status", "--porcelain")
    assert git(repo, "rev-parse", "HEAD") == head
