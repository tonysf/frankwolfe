import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_campaign_shard as campaign


@pytest.fixture(autouse=True)
def fixed_source_commit(monkeypatch):
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda argv, **kw: "fixture-commit\n" if "rev-parse" in argv else "")


def setup_case(tmp_path, count=6):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'instance.npz').write_bytes(b'frozen input')
    runs = [dict(run_id=f'method/run{i}', instance='instance.npz', budget=1,
                 instance_sha256=campaign.digest(data / 'instance.npz'),
                 argv=['run_benchmark.py', '--budget', '1']) for i in range(count)]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(runs=runs)))
    args = argparse.Namespace(manifest=manifest, data_root=data, out_root=tmp_path / 'outputs',
                              task_index=0, task_count=2, source_commit="fixture-commit")
    return args, runs


def fake_completed(command, **kwargs):
    out = Path(command[command.index('--out') + 1])
    out.mkdir()
    (out / 'metadata.json').write_text(json.dumps({'status': 'completed'}))
    return SimpleNamespace(returncode=0)


def test_disjoint_shards_and_idempotent_terminal_resume(tmp_path, monkeypatch):
    args, runs = setup_case(tmp_path)
    monkeypatch.setattr(campaign.subprocess, 'run', fake_completed)
    assert campaign.run_shard(args) == 0
    assert [p.name for p in sorted((args.out_root / 'method').iterdir())] == ['run0', 'run2', 'run4']
    def forbidden(*args, **kwargs):
        raise AssertionError('Terminal resume must not launch the solver again.')
    monkeypatch.setattr(campaign.subprocess, 'run', forbidden)
    assert campaign.run_shard(args) == 0
    monkeypatch.setattr(campaign.subprocess, 'run', fake_completed)
    args.task_index = 1
    assert campaign.run_shard(args) == 0
    assert len(list((args.out_root / 'method').iterdir())) == len(runs)


def test_scientific_failure_is_retained_and_does_not_abort_remaining_runs(tmp_path, monkeypatch):
    args, _ = setup_case(tmp_path, count=4)
    def fail(command, **kwargs):
        out = Path(command[command.index('--out') + 1])
        out.mkdir()
        (out / 'metadata.json').write_text(json.dumps({'status': 'failed', 'error': 'nonfinite output'}))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(campaign.subprocess, 'run', fail)
    assert campaign.run_shard(args) == 0
    ledger = json.loads((args.out_root / '_campaign/shard_00.json').read_text())
    assert ledger['status'] == 'completed_with_failed_runs'
    assert ledger['finished_count'] == 2


def test_three_bootstrap_failures_abort_with_evidence(tmp_path, monkeypatch):
    args, _ = setup_case(tmp_path, count=10)
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=127))
    assert campaign.run_shard(args) == 2
    ledger = json.loads((args.out_root / '_campaign/shard_00.json').read_text())
    assert ledger['status'] == 'aborted_infrastructure_failures'
    assert ledger['finished_count'] == 3
    assert all(r['status'] == 'infrastructure_failure' for r in ledger['records'])


def test_bad_hash_duplicate_id_and_budget_are_rejected(tmp_path):
    args, runs = setup_case(tmp_path)
    altered = json.loads(Path(args.manifest).read_text())
    altered['runs'][0]['instance_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        campaign.validate_manifest(altered, args.data_root)
    with pytest.raises(ValueError, match='Duplicate'):
        campaign.validate_manifest({'runs': [runs[0], runs[0]]}, args.data_root)
    altered['runs'][0]['instance_sha256'] = runs[0]['instance_sha256']
    altered['runs'][0]['budget'] = 2
    with pytest.raises(ValueError, match='budget disagrees'):
        campaign.validate_manifest(altered, args.data_root)


def test_source_drift_and_dirty_tree_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match='Expected source commit'):
        campaign.validate_source('different')
    monkeypatch.setattr(campaign.subprocess, 'check_output',
                        lambda argv, **kw: 'fixture-commit\n' if 'rev-parse' in argv else ' M gw_core.py\n')
    with pytest.raises(ValueError, match='dirty'):
        campaign.validate_source('fixture-commit')


def test_source_failure_after_ledger_creation_is_terminal(tmp_path, monkeypatch):
    args, _ = setup_case(tmp_path)
    calls = iter(['fixture-commit', ValueError('source changed')])
    def source_check(expected):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(campaign, 'validate_source', source_check)
    with pytest.raises(ValueError, match='source changed'):
        campaign.run_shard(args)
    ledger = json.loads((args.out_root / '_campaign/shard_00.json').read_text())
    assert ledger['status'] == 'aborted_exception'
    assert ledger['error'] == 'ValueError: source changed'
    assert not (args.out_root / 'method').exists()
