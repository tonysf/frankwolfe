#!/bin/bash
# Lightweight submission only. All numerical checks run under Slurm.
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?campaign root with verified storage_receipt.json}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" extensions/large_pot_frames_20261016/campaign.py submit --source "$GW_SOURCE" --root "$GW_ROOT"
