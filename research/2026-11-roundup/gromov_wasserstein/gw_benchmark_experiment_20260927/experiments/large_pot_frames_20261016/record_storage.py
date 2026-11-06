"""Record exact Ruche user quota headroom before the isolated campaign launch."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path('/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/large_pot_frames_20261016_6a472d4')

output = subprocess.check_output(['mmlsquota', '-u', 'silvetian', 'gpfs:workdir'], text=True)
rows = [line.split() for line in output.splitlines() if line.split()[:3] == ['gpfs', 'workdir', 'USR']]
assert len(rows) == 1
used, quota, limit, in_doubt = map(int, rows[0][3:7])
assert 0 < used < quota <= limit
available = (min(quota, limit) - used - in_doubt) * 1024
assert available >= 60 * 2**30, 'Insufficient verified quota headroom'
assert (ROOT / 'checkpoints').is_dir()
receipt = dict(quota_verified=True, quota_available_bytes=available,
               checkpoint_root=str(ROOT / 'checkpoints'),
               verified_utc=datetime.now(timezone.utc).isoformat(),
               command=['mmlsquota', '-u', 'silvetian', 'gpfs:workdir'],
               raw_output=output, quota_used_kib=used, quota_soft_limit_kib=quota,
               quota_hard_limit_kib=limit, quota_in_doubt_kib=in_doubt,
               required_available_bytes=60 * 2**30,
               archive_receipt=str(ROOT / 'archive_source_receipt.json'))
(ROOT / 'storage_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
