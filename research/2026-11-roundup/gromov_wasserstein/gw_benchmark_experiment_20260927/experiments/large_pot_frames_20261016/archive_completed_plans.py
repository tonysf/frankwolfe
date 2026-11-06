"""Copy and verify explicitly recorded completed plans; never delete remote data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
LOCAL = HERE / 'archive/shareseq_full_20261001'
REMOTE = Path('/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/shareseq_20260930/full_campaign_8f3b092_retry_20261001')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def copy_one(record):
    path = Path(record['local'])
    if not (path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256']):
        partial = path.with_suffix('.npy.partial')
        subprocess.run(['scp', '-C', '-o', 'BatchMode=yes', 'ruche:' + record['remote'], str(partial)], check=True)
        if partial.stat().st_size != record['bytes'] or sha(partial) != record['sha256']:
            raise RuntimeError('Archive verification failed: ' + str(partial))
        partial.replace(path)
    print('verified ' + record['relative_path'], flush=True)
    return dict(record, verified=True)

def main():
    completion = json.loads((LOCAL / 'completion.json').read_text())
    assert completion['status'] == 'completed'
    finals = [r for r in completion['results'] if r['phase'] == 'final']
    assert len(finals) == 6 and all(r['status'] == 'completed' for r in finals)
    records = []
    metadata = []
    for run in finals:
        relative = Path(run['path']).relative_to(REMOTE)
        for rank in range(4):
            meta = LOCAL / relative / ('rank' + str(rank)) / 'metadata.json'
            metadata.append({'relative_path': str(meta.relative_to(LOCAL)), 'sha256': sha(meta)})
            data = json.loads(meta.read_text())
            for name in ('P_raw', 'P_rounded'):
                plan = data['saved_plans'][name]
                rel = Path(plan['file']).relative_to(REMOTE)
                assert rel == relative / ('rank' + str(rank)) / (name + '.npy')
                records.append({'remote': plan['file'], 'local': str(LOCAL / rel),
                                'relative_path': str(rel), 'bytes': plan['bytes'],
                                'sha256': plan['file_sha256']})
    assert len(records) == 48 and len({r['remote'] for r in records}) == 48
    with ThreadPoolExecutor(max_workers=2) as pool:
        verified = list(pool.map(copy_one, records))
    receipt = {'status': 'all_plans_verified', 'verified_utc': datetime.now(timezone.utc).isoformat(),
               'source_root': str(REMOTE), 'archive_root': str(LOCAL), 'archive_host': 'EliteDesk',
               'completion_sha256': sha(LOCAL / 'completion.json'), 'metadata': metadata,
               'files': verified, 'total_bytes': sum(r['bytes'] for r in verified),
               'remote_sources_removed': False}
    temp = HERE / 'archive_receipt.json.tmp'
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    temp.replace(HERE / 'archive_receipt.json')
    print('ALL 48 PLANS VERIFIED', flush=True)

if __name__ == '__main__':
    main()
