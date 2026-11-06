"""Read the exact Ruche user quota and reserve headroom for both campaigns."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
assert root.parent == Path("/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927")
assert root.name.startswith("pot_entropy_init_20261018_")
assert (root / "source/SHA256SUMS").is_file()
(root / "checkpoints").mkdir(exist_ok=True)
command = ["mmlsquota", "-u", "silvetian", "gpfs:workdir"]
output = subprocess.check_output(command, text=True)
rows = [line.split() for line in output.splitlines() if line.split()[:3] == ["gpfs", "workdir", "USR"]]
assert len(rows) == 1
used, quota, limit, in_doubt = map(int, rows[0][3:7])
assert 0 < used < quota <= limit
available = (min(quota, limit)-used-in_doubt)*1024
assert available >= 68*2**30, "Need 60 GiB for queued SHARE plus 8 GiB for this campaign"
receipt = dict(quota_verified=True, quota_available_bytes=available,
    checkpoint_root=str(root / "checkpoints"), verified_utc=datetime.now(timezone.utc).isoformat(),
    command=command, raw_output=output, quota_used_kib=used, quota_soft_limit_kib=quota,
    quota_hard_limit_kib=limit, quota_in_doubt_kib=in_doubt,
    existing_share_reservation_bytes=60*2**30, new_campaign_reservation_bytes=8*2**30)
(root / "storage_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
