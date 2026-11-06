#!/bin/bash
# Run on Ruche after verifying the isolated source transfer.
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
test -d "$GW_ROOT/prepared/metric512_s1000"
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
# Preserve partial receipts and prevent accidental duplicate submission.
mkdir "$GW_ROOT/submission_lock"
mkdir -p "$GW_ROOT/scheduler"
GW_RECEIPT="$GW_ROOT/submission.tsv"
printf 'stage\tdataset\tjob_id\n' > "$GW_RECEIPT"
GW_TAG=$(cat SOURCE_COMMIT)
GW_TAG=${GW_TAG:0:7}
GW_GATE=$(sbatch --parsable --job-name="gw8192_${GW_TAG}_gate" \
    --time=00:45:00 --output="$GW_ROOT/scheduler/gate-%j.out" \
    extensions/probe8192_20261009/gpu.sbatch "$GW_SOURCE" "$GW_ROOT" preflight)
GW_GATE=${GW_GATE%%;*}
[[ "$GW_GATE" =~ ^[0-9]+$ ]]
printf 'preflight\tmetric512_s1000\t%s\n' "$GW_GATE" | tee -a "$GW_RECEIPT"
GW_RUNS=$(sbatch --parsable --job-name="gw8192_${GW_TAG}_synthetic" \
    --dependency="afterok:$GW_GATE" --kill-on-invalid-dep=yes --array=0-2%3 \
    --output="$GW_ROOT/scheduler/synthetic-%A_%a.out" \
    extensions/probe8192_20261009/gpu.sbatch "$GW_SOURCE" "$GW_ROOT" run metric512_s1000)
GW_RUNS=${GW_RUNS%%;*}
[[ "$GW_RUNS" =~ ^[0-9]+$ ]]
printf 'runs\tmetric512_s1000\t%s\n' "$GW_RUNS" | tee -a "$GW_RECEIPT"
GW_REPORT=$(sbatch --parsable --job-name="gw8192_${GW_TAG}_report" \
    --dependency="afterany:$GW_GATE:$GW_RUNS" \
    --output="$GW_ROOT/scheduler/report-%j.out" \
    extensions/probe8192_20261009/report.sbatch "$GW_SOURCE" "$GW_ROOT")
GW_REPORT=${GW_REPORT%%;*}
[[ "$GW_REPORT" =~ ^[0-9]+$ ]]
printf 'report\tmetric512_s1000\t%s\n' "$GW_REPORT" | tee -a "$GW_RECEIPT"
