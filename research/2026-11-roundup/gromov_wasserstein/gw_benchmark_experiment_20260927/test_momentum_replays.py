"""Bounded control-plane tests; no optimizer replay or GPU use."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import prepare_momentum_replays as prep
import run_momentum_replay_shard as launcher


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path):
    original_root = tmp_path / 'originals'
    original = original_root / 'pilot/toy/smoothing-fw'
    original.mkdir(parents=True)
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'toy.npz').write_bytes(b'canonical opaque fixture input')
    snap = original / 'source_snapshot'
    snap.mkdir()
    for name in prep.SOURCES:
        (snap / name).write_text('{}' if name.endswith('.json') else '# archived opaque fixture\n')
    sources = prep.snapshot_hashes(original)
    config = dict(method='smoothing-fw', instance=str(data / 'toy.npz'), out=str(original),
                  budget=1., device='cpu', dtype='float64', seed=0, oracle='sketch')
    meta = dict(status='completed', source_dirty=False, source_commit='a' * 40,
                source_file_sha256={name: sources[name] for name in prep.SOURCES[:3]},
                config=config, instance_sha256=prep.digest(data / 'toy.npz'), counters={'completed_updates': 2})
    write_json(original / 'metadata.json', meta)
    (original / 'trace.csv').write_text('iteration,loss\n0,1\n2,.5\n')
    (original / 'final.npz').write_bytes(b'opaque final fixture')
    trial = dict(run_id='pilot/toy/smoothing-fw', original_phase='pilot', method='smoothing-fw',
        original_run_dir=str(original), original_manifest_sha256=None, original_run_spec=None,
        original_config=config, original_source_commit='a' * 40, source_snapshot_sha256=sources,
        original_artifact_sha256=prep.artifact_hashes(original), artifact_binding='frozen_at_preparation',
        instance='toy.npz', instance_sha256=meta['instance_sha256'])
    plan = dict(schema_version=1, purpose='missing_momentum_diagnostics_only', phase='pilot',
                original_roots=[str(original_root)], data_root=str(data),
                out_root=str(tmp_path / 'companion/pilot'), run_count=1, runs=[trial])
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, plan)
    args = argparse.Namespace(manifest=str(manifest), manifest_sha256=prep.digest(manifest),
                              source_commit='b' * 40, task_index=0, task_count=1)
    return SimpleNamespace(plan=plan, trial=trial, original=original, data=data, args=args, manifest=manifest)


def test_frozen_scientific_coverage_and_disjointness():
    root = Path(__file__).parent
    counts = {}
    originals = []
    for phase, path in [('tuning', 'configs/stage_c_manifests/tuning.json'),
                        ('final_product', 'configs/stage_c_final_manifests/final_product.json'),
                        ('final_randomized', 'configs/stage_c_final_manifests/final_randomized.json')]:
        runs = prep.read_json(root / path)['runs']
        chosen = [r for r in runs if r['method'] in prep.METHODS]
        counts[phase] = len(chosen)
        originals.extend(r['run_id'] for r in chosen)
        assert all(r['method'] != 'stochastic-kl-momentum' for r in chosen)
    assert counts == {'tuning': 504, 'final_product': 121, 'final_randomized': 165}
    assert len(set(originals)) == 790
    cfg = prep.read_json(root / 'configs/stage_c_pilot.json')
    assert len(cfg['instances']) * len({'smoothing-fw', 'smoothing-fw-exact'} & cfg['methods'].keys()) == 6


@pytest.mark.parametrize('root_kind', ['same', 'child', 'parent', 'data'])
def test_original_and_companion_roots_must_be_disjoint(fixture, root_kind):
    plan = copy.deepcopy(fixture.plan)
    root = Path(plan['original_roots'][0])
    plan['out_root'] = str({'same': root, 'child': root / 'new', 'parent': root.parent,
                            'data': Path(plan['data_root']) / 'new'}[root_kind])
    with pytest.raises(ValueError, match='disjoint'):
        prep.validate_plan(plan)


def test_duplicate_original_and_escaping_path_rejected(fixture):
    plan = copy.deepcopy(fixture.plan)
    plan['runs'].append({**plan['runs'][0], 'run_id': 'different/id'})
    plan['run_count'] = 2
    with pytest.raises(ValueError, match='Duplicate'):
        prep.validate_plan(plan)
    for value in ['../originals', '/absolute', 'pilot/../escape', './alias']:
        with pytest.raises(ValueError):
            prep.safe_relative(value)


def test_original_binding_rejects_changed_source_and_nonterminal(fixture):
    launcher.verify_original(fixture.trial, fixture.data)
    path = fixture.original / 'source_snapshot/gw_core.py'
    path.write_text('# changed')
    with pytest.raises(ValueError, match='archived source'):
        launcher.verify_original(fixture.trial, fixture.data)
    path.write_text('# archived opaque fixture\n')
    meta = prep.read_json(fixture.original / 'metadata.json')
    meta['status'] = 'running'
    write_json(fixture.original / 'metadata.json', meta)
    with pytest.raises(ValueError, match='nonterminal'):
        launcher.verify_original(fixture.trial, fixture.data)


def test_original_binding_rejects_changed_artifact_and_wrong_config(fixture):
    (fixture.original / 'trace.csv').write_text('changed')
    with pytest.raises(ValueError, match='artifacts changed'):
        launcher.verify_original(fixture.trial, fixture.data)
    launcher.assert_config({'argv': ['run_benchmark.py', '--method', 'smoothing-fw', '--rho0', '.3']},
                           {'method': 'smoothing-fw', 'rho0': 0.3})
    with pytest.raises(ValueError, match='--rho0'):
        launcher.assert_config({'argv': ['run_benchmark.py', '--rho0', '.3']}, {'rho0': 1})


def test_exact_oracle_alias_preserves_actual_pilot_config(fixture):
    fixture.trial['method'] = 'smoothing-fw-exact'
    # The archival label is distinct; implementation remains smoothing-fw.
    meta = prep.read_json(fixture.original / 'metadata.json')
    meta['config']['oracle'] = 'exact'
    fixture.trial['original_config'] = meta['config']
    write_json(fixture.original / 'metadata.json', meta)
    fixture.trial['original_artifact_sha256'] = prep.artifact_hashes(fixture.original)
    launcher.verify_original(fixture.trial, fixture.data)


@pytest.mark.parametrize('verified', [True, False])
def test_guarded_execution_and_resume_without_rerunning(fixture, monkeypatch, verified):
    calls = []
    monkeypatch.setattr(launcher, 'validate_source', lambda commit: commit)
    def fake_process(command, **kwargs):
        calls.append(command)
        out = Path(command[command.index('--out') + 1])
        out.mkdir()
        _, bindings, _ = launcher.verify_original(fixture.trial, fixture.data)
        proof = dict(schema_version=1, status='verified' if verified else 'failed',
                     verified=verified, bindings=bindings,
                     utility_sha256=prep.digest(launcher.ROOT / 'replay_momentum_diagnostics.py'))
        if verified:
            (out / 'diagnostics.csv').write_text('original_trace_row,output_iteration,momentum_error\n0,0,0.1\n')
            proof.update(diagnostics_sha256=prep.digest(out / 'diagnostics.csv'), checkpoint_count=1,
                checks={**{name: True for name in launcher.PROOF_CHECKS},
                        'checkpoint_scalars': [dict(original_trace_row=0, iteration=0,
                            compared_fields=['raw_loss', 'rounded_loss', 'true_smoothed_gap'], exact_match=True)]})
        write_json(out / 'verification.json', proof)
        return SimpleNamespace(returncode=0 if verified else 1)
    monkeypatch.setattr(launcher.subprocess, 'run', fake_process)
    assert launcher.execute_shard(fixture.args) == 0
    assert launcher.execute_shard(fixture.args) == 0
    assert len(calls) == 1
    ledger = prep.read_json(Path(fixture.plan['out_root']) / '_replay/shard_00.json')
    assert ledger['records'][0]['resumed_terminal_record']
    assert ledger['status'] == ('completed' if verified else 'completed_with_unverified_replays')
    assert calls[0][1] == '-B'
    before = prep.artifact_hashes(fixture.original)
    assert before == fixture.trial['original_artifact_sha256']
    fixture.args.task_count = 2
    with pytest.raises(ValueError, match='another manifest, sharding or source'):
        launcher.execute_shard(fixture.args)


def test_resume_refuses_tampered_verified_output(fixture, monkeypatch):
    monkeypatch.setattr(launcher, 'validate_source', lambda commit: commit)
    out = Path(fixture.plan['out_root']) / fixture.trial['run_id']
    out.mkdir(parents=True)
    write_json(out / 'replay_receipt.json', {'status': 'verified'})
    with pytest.raises(ValueError, match='differs or is incomplete'):
        launcher.execute_shard(fixture.args)
    ledger = prep.read_json(Path(fixture.plan['out_root']) / '_replay/shard_00.json')
    assert ledger['status'] == 'infrastructure_failure'


def test_manifest_hash_mismatch_writes_nothing(fixture, monkeypatch):
    monkeypatch.setattr(launcher, 'validate_source', lambda commit: commit)
    fixture.args.manifest_sha256 = '0' * 64
    with pytest.raises(ValueError, match='manifest hash'):
        launcher.execute_shard(fixture.args)
    assert not Path(fixture.plan['out_root']).exists()


def test_failed_original_missing_endpoint_is_bound_as_unavailable(fixture):
    meta = prep.read_json(fixture.original / 'metadata.json')
    meta['status'] = 'failed'
    write_json(fixture.original / 'metadata.json', meta)
    (fixture.original / 'final.npz').unlink()
    (fixture.original / 'trace.csv').unlink()
    fixture.trial['original_artifact_sha256'] = prep.artifact_hashes(fixture.original)
    _, bindings, _ = launcher.verify_original(fixture.trial, fixture.data)
    assert bindings['final_sha256'] is None and bindings['trace_sha256'] is None
    assert bindings['metadata_sha256']


def test_healthy_original_missing_endpoint_is_infrastructure_failure(fixture):
    (fixture.original / 'final.npz').unlink()
    with pytest.raises(FileNotFoundError):
        launcher.verify_original(fixture.trial, fixture.data)


def test_nested_replay_ids_rejected(fixture):
    plan = copy.deepcopy(fixture.plan)
    second = {**plan['runs'][0], 'run_id': plan['runs'][0]['run_id'] + '/nested',
              'original_run_dir': plan['original_roots'][0] + '/different'}
    plan['runs'].append(second)
    plan['run_count'] = 2
    with pytest.raises(ValueError, match='Nested'):
        prep.validate_plan(plan)


@pytest.mark.parametrize('tamper', ['utility', 'csv', *launcher.PROOF_CHECKS,
                                   'count', 'proof_length', 'exact_match', 'fields', 'row_index'])
def test_incomplete_or_tampered_fidelity_proof_rejected(fixture, tamper):
    out = Path(fixture.plan['out_root']) / fixture.trial['run_id']
    out.mkdir(parents=True)
    _, bindings, _ = launcher.verify_original(fixture.trial, fixture.data)
    csv_path = out / 'diagnostics.csv'
    csv_path.write_text('original_trace_row,output_iteration\n0,0\n')
    proof = dict(schema_version=1, status='verified', verified=True, bindings=bindings,
        utility_sha256=prep.digest(launcher.ROOT / 'replay_momentum_diagnostics.py'),
        diagnostics_sha256=prep.digest(csv_path), checkpoint_count=1,
        checks={**{name: True for name in launcher.PROOF_CHECKS},
                'checkpoint_scalars': [dict(original_trace_row=0, iteration=0,
                    compared_fields=['raw_loss', 'rounded_loss', 'true_smoothed_gap'], exact_match=True)]})
    write_json(out / 'verification.json', proof)
    assert launcher.verify_companion(out, bindings)[0] == 'verified'
    if tamper == 'utility':
        proof['utility_sha256'] = '0' * 64
    elif tamper == 'csv':
        csv_path.write_text('changed')
    elif tamper in launcher.PROOF_CHECKS:
        proof['checks'][tamper] = False
    elif tamper == 'count':
        proof['checkpoint_count'] = 0
    elif tamper == 'proof_length':
        proof['checks']['checkpoint_scalars'] = []
    elif tamper == 'exact_match':
        proof['checks']['checkpoint_scalars'][0]['exact_match'] = False
    elif tamper == 'fields':
        proof['checks']['checkpoint_scalars'][0]['compared_fields'] = []
    elif tamper == 'row_index':
        proof['checks']['checkpoint_scalars'][0]['original_trace_row'] = 1
    write_json(out / 'verification.json', proof)
    with pytest.raises(ValueError):
        launcher.verify_companion(out, bindings)
