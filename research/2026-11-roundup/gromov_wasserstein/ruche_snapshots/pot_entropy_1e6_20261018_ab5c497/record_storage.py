"""Read Ruche quota before launching either of the two requested additions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--experiment", choices=("share_resumable_20261018", "pot_entropy_1e6_20261018"), required=True)
args = parser.parse_args()
root = args.root.resolve()
assert root.parent == Path("/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927")
assert root.name.startswith(args.experiment + "_")
config = json.loads((root / "source/extensions" / args.experiment / "campaign.json").read_text())
required = config["storage"]["required_quota_available_bytes"]
command = ["mmlsquota", "-u", "silvetian", "gpfs:workdir"]
output = subprocess.check_output(command, text=True)
rows = [line.split() for line in output.splitlines() if line.split()[:3] == ["gpfs", "workdir", "USR"]]
assert len(rows) == 1
used, quota, limit, in_doubt = map(int, rows[0][3:7])
assert 0 < used < quota <= limit
available = (min(quota, limit)-used-in_doubt)*1024
assert available >= required, ("Insufficient reserved quota", available, required)
(root / "checkpoints").mkdir(exist_ok=True)
receipt = dict(quota_verified=True, quota_available_bytes=available,
    checkpoint_root=str(root / "checkpoints"), verified_utc=datetime.now(timezone.utc).isoformat(),
    command=command, raw_output=output, quota_used_kib=used, quota_soft_limit_kib=quota,
    quota_hard_limit_kib=limit, quota_in_doubt_kib=in_doubt,
    required_available_bytes=required, share_retained_state_reservation_bytes=64*2**30,
    serialized_entropy_reservation_bytes=4*2**30)
(root / "storage_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(receipt))
