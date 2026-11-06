"""Release only 48 archived plan files, keeping remote metadata and restore map.

Run on Ruche only after the local archive receipt says every copy passed its
recorded SHA256. This performs metadata checks and bounded filesystem cleanup;
it does not run numerical computation or inspect the matrix contents.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

EXPECTED = Path('/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/shareseq_20260930/full_campaign_8f3b092_retry_20261001')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    receipt = json.loads(Path(sys.argv[1]).read_text())
    assert receipt['status'] == 'all_plans_verified'
    assert Path(receipt['source_root']) == EXPECTED
    assert len(receipt['files']) == 48 and len(receipt['metadata']) == 24
    assert digest(EXPECTED / 'completion.json') == receipt['completion_sha256']
    allowed = {f'final_{seed}_{method}/rank{rank}/{name}.npy'
               for seed in (500, 501, 502)
               for method in ('smoothing-fw', 'stochastic-kl-momentum')
               for rank in range(4) for name in ('P_raw', 'P_rounded')}
    assert {record['relative_path'] for record in receipt['files']} == allowed
    for record in receipt['metadata']:
        path = EXPECTED / record['relative_path']
        assert path.resolve().is_relative_to(EXPECTED)
        assert digest(path) == record['sha256']
    for record in receipt['files']:
        assert record['verified'] is True
        path = EXPECTED / record['relative_path']
        assert str(path) == record['remote'] and path.resolve().is_relative_to(EXPECTED)
        assert not path.is_symlink()
        if path.exists():
            assert path.is_file() and path.stat().st_size == record['bytes']
    destination = EXPECTED / 'ARCHIVED_PLANS_20261016.json'
    receipt.update(remote_sources_removed=False, status='archiving_verified_copies',
                   restore_instructions='Copy each files[].local from EliteDesk to files[].remote on Ruche, then verify SHA256 against files[].sha256. Metadata, traces, summaries and original datasets remain on Ruche.')
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    for record in receipt['files']:
        (EXPECTED / record['relative_path']).unlink(missing_ok=True)
    receipt.update(remote_sources_removed=True, status='archived_to_verified_local_copy',
                   finalized_utc=datetime.now(timezone.utc).isoformat())
    destination.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'files': 48,
                      'released_bytes': receipt['total_bytes'], 'receipt': str(destination)}))

if __name__ == '__main__':
    main()
