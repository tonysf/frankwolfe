"""Minimal byte/metadata fixtures: no arrays, solvers, GPU work or remote access."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import pytest

import audit_replay_original_bindings as module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_coverage_is_exactly_796_and_roots_are_unambiguous():
    assert module.EXPECTED_COUNTS == {'tuning': 504, 'pilot': 6, 'finals': 286}
    assert sum(module.EXPECTED_ORIGINAL_COUNTS.values()) == 796
    with pytest.raises(ValueError, match='Duplicate'):
        module.parse_roots(['pilot=/one', 'pilot=/two'])


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Reduce only test populations, never expose a production CLI count override.
    monkeypatch.setattr(module, 'EXPECTED_COUNTS', {'tuning': 1, 'pilot': 1, 'finals': 2})
    monkeypatch.setattr(module, 'EXPECTED_ORIGINAL_COUNTS', dict.fromkeys(
        ('tuning', 'pilot', 'final_product', 'final_randomized'), 1))
    roots = {phase: tmp_path / 'originals' / phase for phase in module.EXPECTED_ORIGINAL_COUNTS}
    data = tmp_path / 'data'; data.mkdir()
    instance = data / 'toy.npz'; instance.write_bytes(b'opaque canonical NPZ bytes: never numerically loaded')
    data_hash = digest(instance)
    pairs, records, paths, manifests = [], [], {}, {}
    revision = 'a' * 40; theory = {'beta': .5}
    for replay_phase, original_phases in [('tuning', ['tuning']), ('pilot', ['pilot']),
                                         ('finals', ['final_product', 'final_randomized'])]:
        companion = tmp_path / 'companions' / replay_phase
        remote_roots = ['/remote/originals/' + phase for phase in original_phases]
        plan = dict(schema_version=1, purpose='missing_momentum_diagnostics_only', phase=replay_phase,
                    run_count=len(original_phases), data_root='/remote/data', original_roots=remote_roots,
                    out_root='/remote/companions/' + replay_phase, runs=[])
        pending = []
        for phase, remote_root in zip(original_phases, remote_roots):
            rid = ('pilot/toy/smoothing-fw' if phase == 'pilot' else phase + '/smoothing-fw/c00/toy/s0_i0')
            relative = rid.removeprefix('pilot/') if phase == 'pilot' else rid
            remote_original = remote_root + '/' + relative
            local = roots[phase] / relative; local.mkdir(parents=True)
            paths[phase] = local
            for name in module.SOURCES:
                p = local / 'source_snapshot' / name; p.parent.mkdir(exist_ok=True)
                p.write_text('# archived source bytes, not executed\n')
            write_json(local / 'source_snapshot/theory_config.json', theory)
            source_hashes = {name: digest(local / 'source_snapshot' / name) for name in module.SOURCES}
            source_hashes['theory_config.json'] = digest(local / 'source_snapshot/theory_config.json')
            status = 'failed' if phase == 'final_randomized' else 'completed'
            config = dict(method='smoothing-fw', oracle='sketch', out=remote_original, instance='/remote/data/toy.npz')
            meta = dict(status=status, config=config, counters={'completed_updates': 2}, timing={'solver_seconds': 1.},
                        source_commit=revision, source_dirty=False, source_file_sha256={k: source_hashes[k] for k in module.SOURCES},
                        instance_sha256=data_hash, theory_config=theory)
            write_json(local / 'metadata.json', meta)
            if status == 'completed':
                (local / 'trace.csv').write_text('iteration,raw_loss\n2,0.1\n')
                (local / 'final.npz').write_bytes(b'opaque plan bytes')
            bindings = dict(metadata_sha256=digest(local / 'metadata.json'),
                            trace_sha256=digest(local / 'trace.csv') if status == 'completed' else None,
                            final_sha256=digest(local / 'final.npz') if status == 'completed' else None,
                            instance_sha256=data_hash,
                            source_file_sha256={k: source_hashes[k] for k in module.SOURCES},
                            theory_config_sha256=source_hashes['theory_config.json'])
            spec = None if phase == 'pilot' else {'run_id': rid, 'seed': 0}
            trial = dict(run_id=rid, original_phase=phase, method='smoothing-fw', instance='toy.npz',
                         instance_sha256=data_hash, original_run_dir=remote_original, original_source_commit=revision,
                         source_snapshot_sha256=source_hashes, original_config=config if replay_phase != 'finals' else None,
                         original_run_spec=spec, original_manifest_sha256='b' * 64,
                         original_artifact_sha256=({'metadata.json': bindings['metadata_sha256'], 'trace.csv': bindings['trace_sha256'],
                                                    'final.npz': bindings['final_sha256']} if replay_phase != 'finals' else None))
            plan['runs'].append(trial)
            original_receipt_hash = None
            if spec is not None:
                write_json(local / 'campaign_receipt.json', dict(run_spec=spec, run_id=rid, manifest_sha256='b' * 64,
                                                               status=status, infrastructure_error=None))
                original_receipt_hash = digest(local / 'campaign_receipt.json')
            companion_status = 'failed' if status == 'failed' else 'verified'
            proof = dict(bindings=bindings, original_config=config, original_counters=meta['counters'], original_timing=meta['timing'],
                         original_status=status, original_source_commit=revision, run_dir=remote_original,
                         status=companion_status, verified=companion_status == 'verified')
            proof_path = companion / rid / 'verification.json'; write_json(proof_path, proof)
            record = dict(run_id=rid, original_phase=phase, method='smoothing-fw', instance='toy.npz',
                          original_status=status, original_source_commit=revision, controller_source_commit='c' * 40,
                          terminal=True, provenance_valid=True, issues=[], status=companion_status, verified=companion_status == 'verified',
                          original_bindings=bindings, original_counters=meta['counters'], original_timing=meta['timing'],
                          verification_sha256=digest(proof_path))
            records.append(record)
            pending.append((trial, record, original_receipt_hash, proof_path))
        manifest = tmp_path / 'manifests' / (replay_phase + '.json'); write_json(manifest, plan)
        manifests[replay_phase] = manifest
        pair = dict(phase=replay_phase, path=str(manifest), sha256=digest(manifest), errors=[], local_root=str(companion),
                    remote_root=plan['out_root'], controller_source_commit='c' * 40, prescribed_runs=len(original_phases))
        pairs.append(pair)
        for trial, record, old_receipt_hash, proof_path in pending:
            receipt_path = proof_path.parent / 'replay_receipt.json'
            write_json(receipt_path, dict(original_bindings=record['original_bindings'], run_spec=trial, manifest_sha256=pair['sha256'],
                                         original_campaign_receipt_sha256=old_receipt_hash, controller_source_commit='c' * 40,
                                         status=record['status'], output_sha256={'verification_sha256': record['verification_sha256']}))
            record['receipt_sha256'] = digest(receipt_path)
    report = dict(schema_version=1, status='complete_with_unverified_replays', execution_complete=True, provenance_valid=True,
                  all_diagnostics_verified=False, issues=[], prescribed_runs=4, terminal_runs=4, verified_runs=3,
                  unverified_terminal_runs=1, status_counts={'verified': 3, 'failed': 1}, runs=records, manifest_pairs=pairs,
                  script_sha256='d' * 64)
    report_path = tmp_path / 'report.json'; write_json(report_path, report)
    return dict(report_path=report_path, report=report, roots=roots, data=data, paths=paths, manifests=manifests)


def run(fixture):
    return module.audit(fixture['report_path'], fixture['roots'], fixture['data'])


def test_complete_cross_archive_mapping_includes_pilot_and_failed_missing_outputs(fixture):
    result = run(fixture)
    assert result['all_match'] and result['coverage_complete'], result
    assert result['matched_records'] == 4 and result['hash_accounting']['cache_hits'] > 0
    pilot = next(r for r in result['records'] if r['original_phase'] == 'pilot')
    assert pilot['original_relative_path'] == 'toy/smoothing-fw'
    failed = next(r for r in result['records'] if r['original_phase'] == 'final_randomized')
    assert failed['matched'] and failed['actual_bindings']['final_sha256'] is None
    assert result['inputs_unchanged_during_audit']


@pytest.mark.parametrize('tamper', ['plan_bit', 'source_file', 'unexpected_failed_output', 'manifest'])
def test_changed_original_bytes_or_binding_manifest_are_rejected(fixture, tamper):
    if tamper == 'plan_bit':
        path = fixture['paths']['final_product'] / 'final.npz'
        value = bytearray(path.read_bytes()); value[0] ^= 1; path.write_bytes(value)
    elif tamper == 'source_file':
        path = fixture['paths']['final_product'] / 'source_snapshot/gw_core.py'
        path.write_text(path.read_text() + '# modified\n')
    elif tamper == 'unexpected_failed_output':
        (fixture['paths']['final_randomized'] / 'trace.csv').write_text('unexpected restored output')
    else:
        path = fixture['manifests']['finals']; path.write_text(path.read_text() + '\n')
    result = run(fixture)
    assert not result['all_match']
    assert result['issues'] or result['mismatched_records']


@pytest.mark.parametrize('missing', ['record', 'root', 'original_directory'])
def test_missing_prescribed_records_roots_or_original_trial_fail_closed(fixture, missing):
    if missing == 'record':
        changed = copy.deepcopy(fixture['report']); changed['runs'].pop()
        write_json(fixture['report_path'], changed)
    elif missing == 'root':
        fixture['roots'].pop('pilot')
    else:
        fixture['paths']['final_product'].rename(fixture['paths']['final_product'].with_name('absent'))
    result = run(fixture)
    assert not result['all_match']
    if missing == 'original_directory':
        assert len(result['records']) == 4 and result['mismatched_records'] == 1
    else:
        assert result['issues'] and not result['coverage_complete']
