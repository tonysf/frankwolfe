"""Cross-bind a complete replay report to separately collected original archives.

This stdlib-only audit reads bytes and metadata; it never loads coupling arrays,
reruns an optimizer, modifies an original, or changes a replay/report. The public
CLI requires all 796 prescribed records and all four original archive roots.
Remote original_roots from the hash-bound replay manifests determine relative
paths. In particular the pilot's local root is the collected 1961459 directory,
whose children are dataset/method, not pilot/dataset/method.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time

EXPECTED_COUNTS = {'tuning': 504, 'pilot': 6, 'finals': 286}
EXPECTED_ORIGINAL_COUNTS = {'tuning': 504, 'pilot': 6, 'final_product': 121, 'final_randomized': 165}
SOURCES = ('run_benchmark.py', 'smoothing_fw.py', 'gw_core.py')
BINDING_KEYS = {'metadata_sha256', 'trace_sha256', 'final_sha256', 'instance_sha256',
                'source_file_sha256', 'theory_config_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def relative_path(value):
    require(isinstance(value, str), 'Relative path must be a string.')
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and '..' not in path.parts
            and str(path) == value, 'Unsafe relative path: ' + value)
    return path


def remote_path(value):
    require(isinstance(value, str), 'Remote path must be a string.')
    path = PurePosixPath(value)
    require(path.is_absolute() and '..' not in path.parts and str(path) == value,
            'Unsafe remote path: ' + value)
    return path


def contained(root, relative):
    path = (root / str(relative)).resolve()
    require(path != root and root in path.parents, 'Local archive path escapes its supplied root.')
    return path


class HashCache:
    """Hash each canonical path once, detecting replacement/change during audit."""
    def __init__(self):
        self.entries = {}
        self.bytes_hashed = 0
        self.cache_hits = 0

    @staticmethod
    def signature(path):
        if not path.exists():
            return None
        stat = path.stat()
        require(path.is_file(), 'Expected regular file: ' + str(path))
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def digest(self, path, optional=False):
        path = Path(path).resolve()
        signature = self.signature(path)
        if path in self.entries:
            previous, digest = self.entries[path]
            require(signature == previous, 'Input changed during audit: ' + str(path))
            self.cache_hits += 1
        else:
            digest = None
            if signature is not None:
                hasher = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(block)
                        self.bytes_hashed += len(block)
                digest = hasher.hexdigest()
                require(self.signature(path) == signature, 'Input changed while hashing: ' + str(path))
            self.entries[path] = signature, digest
        require(optional or digest is not None, 'Required file missing: ' + str(path))
        return digest

    def read_json(self, path):
        self.digest(path)
        value = json.loads(Path(path).read_text())
        self.digest(path)
        return value

    def assert_unchanged(self):
        for path, (signature, _) in self.entries.items():
            require(self.signature(path) == signature, 'Input changed during audit: ' + str(path))


def mapping(trial, plan, original_roots):
    original = remote_path(trial['original_run_dir'])
    matches = [remote_path(value) for value in plan['original_roots']
               if remote_path(value) in original.parents]
    require(len(matches) == 1, 'Original path must belong to exactly one manifest original_root.')
    phase = trial['original_phase']
    require(phase in original_roots, 'Missing supplied original root for phase: ' + phase)
    relative = original.relative_to(matches[0])
    local = contained(original_roots[phase], relative)
    return local, matches[0], relative


def inspect_original(trial, plan, pair, record, roots, data_root, cache):
    result = dict(run_id=trial['run_id'], original_phase=trial['original_phase'],
                  companion_status=record['status'], matched=False, issues=[],
                  expected_bindings=record.get('original_bindings'), actual_bindings=None)
    try:
        local, remote_root, relative = mapping(trial, plan, roots)
        require(local.is_dir(), 'Collected original directory missing: ' + str(local))
        result.update(original_local_dir=str(local), original_remote_dir=trial['original_run_dir'],
                      original_remote_root=str(remote_root), original_relative_path=str(relative))
        companion_root = Path(pair['local_root']).resolve()
        companion = contained(companion_root, relative_path(trial['run_id']))
        require(companion_root.is_dir(), 'Collected companion root missing.')
        expected = record['original_bindings']
        require(set(expected) == BINDING_KEYS and set(expected['source_file_sha256']) == set(SOURCES),
                'Incomplete original bindings schema.')
        require(all(valid_hash(expected[key]) for key in ('metadata_sha256', 'instance_sha256', 'theory_config_sha256'))
                and all(valid_hash(value) for value in expected['source_file_sha256'].values())
                and all(expected[key] is None or valid_hash(expected[key]) for key in ('trace_sha256', 'final_sha256')),
                'Malformed original binding digest.')
        proof_path = contained(companion, 'verification.json')
        receipt_path = contained(companion, 'replay_receipt.json')
        proof_hash, receipt_hash = cache.digest(proof_path), cache.digest(receipt_path)
        require(proof_hash == record['verification_sha256'], 'Companion verification differs from complete report.')
        require(receipt_hash == record['receipt_sha256'], 'Companion receipt differs from complete report.')
        proof, receipt = cache.read_json(proof_path), cache.read_json(receipt_path)
        require(proof['bindings'] == receipt['original_bindings'] == expected,
                'Report/proof/receipt original bindings disagree.')
        require(receipt['output_sha256']['verification_sha256'] == proof_hash,
                'Receipt does not bind current verification.')
        require(receipt['run_spec'] == trial and receipt['manifest_sha256'] == pair['sha256'],
                'Receipt does not bind prescribed replay trial/manifest.')
        require(receipt['controller_source_commit'] == record['controller_source_commit'] == pair['controller_source_commit'],
                'Controller source lineage differs.')
        require(proof['status'] == receipt['status'] == record['status']
                and proof['verified'] is record['verified'], 'Replay status differs across artifacts.')
        require(proof['run_dir'] == trial['original_run_dir'], 'Proof names another original directory.')
        require(record['original_source_commit'] == proof['original_source_commit'] == trial['original_source_commit'],
                'Original source revision differs.')
        canonical = contained(data_root, relative_path(trial['instance']))
        actual = dict(metadata_sha256=cache.digest(contained(local, 'metadata.json')),
                      trace_sha256=cache.digest(contained(local, 'trace.csv'), optional=True),
                      final_sha256=cache.digest(contained(local, 'final.npz'), optional=True),
                      instance_sha256=cache.digest(canonical),
                      source_file_sha256={name: cache.digest(contained(local, 'source_snapshot/' + name))
                                          for name in SOURCES},
                      theory_config_sha256=cache.digest(contained(local, 'source_snapshot/theory_config.json')))
        result['actual_bindings'] = actual
        for key in BINDING_KEYS:
            if actual[key] != expected[key]:
                result['issues'].append('Original archive binding mismatch: ' + key)
        require(actual['instance_sha256'] == trial['instance_sha256'], 'Canonical NPZ differs from replay manifest.')
        source_hashes = dict(actual['source_file_sha256'], **{'theory_config.json': actual['theory_config_sha256']})
        require(source_hashes == trial['source_snapshot_sha256'], 'Collected source snapshot differs from replay manifest.')
        meta = cache.read_json(contained(local, 'metadata.json'))
        require(meta['status'] == proof['original_status'] == record['original_status'],
                'Original status differs from proof/report.')
        require(meta['status'] in ('completed', 'failed'), 'Original is not a supported terminal trajectory.')
        for key in ('trace_sha256', 'final_sha256'):
            require(actual[key] is not None or (expected[key] is None and meta['status'] == 'failed'),
                    'Absent original output is allowed only for an explicitly bound failed original: ' + key)
        require(meta['config'] == proof['original_config'], 'Original metadata configuration differs from proof.')
        require(meta['counters'] == proof['original_counters'] == record['original_counters'],
                'Original metadata counters differ from proof/report.')
        require(meta['timing'] == proof['original_timing'] == record['original_timing'],
                'Original metadata timing differs from proof/report.')
        require(meta['source_commit'] == trial['original_source_commit'] and meta['source_dirty'] is False,
                'Original source revision/clean-state mismatch.')
        require(meta['source_file_sha256'] == actual['source_file_sha256'], 'Original metadata/source hashes disagree.')
        require(meta['instance_sha256'] == actual['instance_sha256'], 'Original metadata/input hash differs.')
        require(meta['config']['out'] == trial['original_run_dir'], 'Original recorded output path differs.')
        require(meta['config']['instance'] == str(remote_path(plan['data_root']) / relative_path(trial['instance'])),
                'Original recorded canonical input path differs.')
        require(cache.read_json(contained(local, 'source_snapshot/theory_config.json')) == meta['theory_config'],
                'Collected theory configuration differs from original metadata.')
        if trial.get('original_config') is not None:
            require(meta['config'] == trial['original_config'], 'Original configuration differs from frozen manifest.')
        if trial.get('original_artifact_sha256') is not None:
            require(trial['original_artifact_sha256'] == {
                'metadata.json': actual['metadata_sha256'], 'trace.csv': actual['trace_sha256'],
                'final.npz': actual['final_sha256']}, 'Collected original artifacts differ from preparation bindings.')
        if trial.get('original_run_spec') is not None:
            original_receipt_path = contained(local, 'campaign_receipt.json')
            original_receipt_hash = cache.digest(original_receipt_path)
            require(original_receipt_hash == receipt['original_campaign_receipt_sha256'],
                    'Collected original campaign receipt differs from replay-start binding.')
            original_receipt = cache.read_json(original_receipt_path)
            require(original_receipt['run_spec'] == trial['original_run_spec']
                    and original_receipt['run_id'] == trial['run_id']
                    and original_receipt['manifest_sha256'] == trial['original_manifest_sha256']
                    and original_receipt['status'] == meta['status']
                    and original_receipt['infrastructure_error'] is None,
                    'Original campaign receipt body differs from prescribed trial/status.')
            result['original_campaign_receipt_sha256'] = original_receipt_hash
        else:
            require(trial['original_phase'] == 'pilot' and receipt['original_campaign_receipt_sha256'] is None,
                    'Absent original campaign receipt is only supported for pilot.')
        result.update(original_status=meta['status'], verification_sha256=proof_hash, receipt_sha256=receipt_hash,
                      matched=not result['issues'])
    except (ValueError, KeyError, TypeError, OSError) as error:
        result['issues'].append(str(error))
    return result


def audit(report_path, original_roots, data_root):
    started = time.perf_counter()
    cache = HashCache()
    result = dict(schema_version=1, status='failed', all_match=False, coverage_complete=False,
                  audited_utc=datetime.now(timezone.utc).isoformat(), report_path=str(Path(report_path).resolve()),
                  expected_records=sum(EXPECTED_COUNTS.values()), records=[], issues=[],
                  expected_replay_phase_counts=EXPECTED_COUNTS,
                  expected_original_phase_counts=EXPECTED_ORIGINAL_COUNTS,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  policy='Original archives and companion artifacts are read-only. Only bytes/metadata are inspected; no arrays are loaded or numerical work performed.')
    try:
        result['report_sha256'] = cache.digest(report_path)
        report = cache.read_json(report_path)
        require(valid_hash(report['script_sha256']), 'Missing reporter source hash.')
        result['reporter_source_sha256'] = report['script_sha256']
        require(report['schema_version'] == 1 and report['status'] in ('complete_verified', 'complete_with_unverified_replays')
                and report['execution_complete'] is True and report['provenance_valid'] is True
                and not report['issues'], 'Require a complete provenance-valid replay report.')
        roots = {key: Path(value).resolve() for key, value in original_roots.items()}
        require(set(roots) == set(EXPECTED_ORIGINAL_COUNTS), 'Supply exactly tuning,pilot,final_product,final_randomized original roots.')
        require(all(path.is_dir() for path in roots.values()), 'A supplied original root is missing.')
        data_root = Path(data_root).resolve()
        require(data_root.is_dir(), 'Canonical data root is missing.')
        result.update(original_roots={key: str(path) for key, path in roots.items()}, data_root=str(data_root))
        records = report['runs']; count = sum(EXPECTED_COUNTS.values())
        require(len(records) == report['prescribed_runs'] == report['terminal_runs'] == count,
                'Missing/extra prescribed records; expected exactly ' + str(count) + '.')
        require(Counter(r['original_phase'] for r in records) == Counter(EXPECTED_ORIGINAL_COUNTS),
                'Original phase coverage differs from prescribed population.')
        ids = [r['run_id'] for r in records]
        require(len(ids) == len(set(ids)), 'Duplicate replay records.')
        require(all(r['terminal'] is True and r['provenance_valid'] is True and not r['issues']
                    and r['status'] in ('verified', 'failed') and r['verified'] is (r['status'] == 'verified')
                    for r in records), 'Report contains incomplete or invalid replay records.')
        counts = Counter(r['status'] for r in records)
        require(report['verified_runs'] == counts['verified'] and report['unverified_terminal_runs'] == counts['failed']
                and report['status_counts'] == dict(counts), 'Reported replay status counts differ.')
        require(report['all_diagnostics_verified'] is (counts['verified'] == count)
                and report['status'] == ('complete_verified' if counts['verified'] == count else 'complete_with_unverified_replays'),
                'Complete report status disagrees with replay statuses.')
        pairs = report['manifest_pairs']
        require(Counter(p['phase'] for p in pairs) == Counter(EXPECTED_COUNTS.keys()), 'Missing/duplicate replay manifest pair.')
        trial_map = {}; manifest_proofs = []; phase_remote_roots = {}
        for pair in pairs:
            require(not pair['errors'], 'Replay manifest pair has provenance errors.')
            path = Path(pair['path']).resolve()
            require(cache.digest(path) == pair['sha256'], 'Replay manifest differs from report binding.')
            plan = cache.read_json(path)
            require(plan['schema_version'] == 1 and plan['phase'] == pair['phase']
                    and plan['purpose'] == 'missing_momentum_diagnostics_only', 'Unexpected replay manifest.')
            require(plan['run_count'] == len(plan['runs']) == pair['prescribed_runs'] == EXPECTED_COUNTS[plan['phase']],
                    'Manifest prescribed coverage differs.')
            require(plan['out_root'] == pair['remote_root'], 'Companion remote root differs from manifest.')
            for trial in plan['runs']:
                rid = trial['run_id']; relative_path(rid)
                require(rid not in trial_map, 'Duplicate prescribed replay across manifests.')
                allowed = ('final_product', 'final_randomized') if plan['phase'] == 'finals' else (plan['phase'],)
                require(trial['original_phase'] in allowed, 'Original phase lies outside replay manifest phase.')
                local, remote, _ = mapping(trial, plan, roots)
                require(trial['original_phase'] not in phase_remote_roots or phase_remote_roots[trial['original_phase']] == str(remote),
                        'One original phase maps to multiple remote archive roots.')
                phase_remote_roots[trial['original_phase']] = str(remote)
                trial_map[rid] = (trial, plan, pair)
            manifest_proofs.append(dict(phase=plan['phase'], path=str(path), sha256=pair['sha256'],
                                        companion_local_root=pair['local_root'], original_remote_roots=plan['original_roots']))
        require(set(trial_map) == set(ids), 'Report IDs do not exactly cover the hash-bound replay manifests.')
        require(len({t[0]['original_run_dir'] for t in trial_map.values()}) == count,
                'Multiple replays bind the same original trajectory.')
        result.update(manifest_pairs=manifest_proofs, remote_root_mapping=phase_remote_roots,
                      coverage_complete=True, observed_records=len(records))
        for record in records:
            trial, plan, pair = trial_map[record['run_id']]
            require(record['original_phase'] == trial['original_phase'] and record['method'] == trial['method']
                    and record['instance'] == trial['instance'], 'Report scientific identity differs from manifest.')
            result['records'].append(inspect_original(trial, plan, pair, record, roots, data_root, cache))
        cache.assert_unchanged()
        result['inputs_unchanged_during_audit'] = True
        result['all_match'] = len(result['records']) == count and all(r['matched'] for r in result['records'])
        result['status'] = 'all_match' if result['all_match'] else 'mismatch'
    except (ValueError, KeyError, TypeError, OSError) as error:
        result['issues'].append(str(error))
    result['matched_records'] = sum(r['matched'] for r in result['records'])
    result['mismatched_records'] = sum(not r['matched'] for r in result['records'])
    result['phase_counts'] = {phase: dict(inspected=sum(r['original_phase'] == phase for r in result['records']),
                                         matched=sum(r['original_phase'] == phase and r['matched'] for r in result['records']))
                              for phase in EXPECTED_ORIGINAL_COUNTS}
    result['hash_accounting'] = dict(unique_canonical_paths=len(cache.entries), cache_hits=cache.cache_hits,
                                     bytes_hashed=cache.bytes_hashed)
    result['audit_elapsed_seconds'] = time.perf_counter() - started
    return result


def parse_roots(values):
    result = {}
    for value in values:
        require('=' in value, 'Use --original-root PHASE=PATH.')
        phase, path = value.split('=', 1)
        require(phase not in result and phase in EXPECTED_ORIGINAL_COUNTS and path,
                'Duplicate/unknown/empty original root: ' + phase)
        result[phase] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--original-root', action='append', required=True)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    roots = parse_roots(args.original_root)
    output = args.out.resolve()
    require(not output.exists(), 'Audit output already exists; use a fresh file.')
    protected = [Path(value).resolve() for value in roots.values()] + [args.data_root.resolve()]
    require(output != args.report.resolve() and all(output != root and root not in output.parents for root in protected),
            'Audit output must be separate from original archives and canonical data.')
    result = audit(args.report, roots, args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'all_match', 'coverage_complete', 'matched_records', 'mismatched_records')}))
    return 0 if result['all_match'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
