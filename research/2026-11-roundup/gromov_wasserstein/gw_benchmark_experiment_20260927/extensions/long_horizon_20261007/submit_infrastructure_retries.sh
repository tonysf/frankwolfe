#!/bin/bash
# Only the three verified pre-update CUDA-unavailable failures are retried.
# Original source/config/results and successful or running jobs are preserved.
set -euo pipefail
GW_LAUNCH_SOURCE=${1:?new immutable scheduler source}
GW_ORIGINAL_SOURCE=${2:?original immutable numerical source}
GW_PREPARED=${3:?frozen prepared arrays}
GW_ROOT=${4:?new retry root}
GW_TAG=${5:?scheduler source tag}
GW_GATE=${6:?GPU integration job ID}
[[ "$GW_GATE" =~ ^[0-9]+$ ]]
cd "$GW_LAUNCH_SOURCE"
sha256sum --status -c SHA256SUMS
cd "$GW_ORIGINAL_SOURCE"
sha256sum --status -c SHA256SUMS
GW_EXTENSION="$GW_LAUNCH_SOURCE/extensions/long_horizon_20261007"
GW_CONFIG="$GW_ORIGINAL_SOURCE/extensions/long_horizon_20261007/campaign.json"
GW_OUTPUT="$GW_ROOT/results"
mkdir "$GW_ROOT/submission_started" "$GW_OUTPUT"
GW_RECEIPT="$GW_ROOT/submission.tsv"
printf 'stage\tdataset\tmethod\tjob_id\n' > "$GW_RECEIPT"
date --iso-8601=seconds > "$GW_ROOT/submitted_at.txt"
submit() {
    local stage=$1 dataset=$2 method=$3
    shift 3
    GW_JOB=$(sbatch --parsable --export=NONE --propagate=NONE "$@")
    GW_JOB=${GW_JOB%%;*}
    [[ "$GW_JOB" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s\n' "$stage" "$dataset" "$method" "$GW_JOB" >> "$GW_RECEIPT"
    printf '%s\t%s\t%s\t%s\n' "$stage" "$dataset" "$method" "$GW_JOB"
}
GW_ALL_JOBS=""
while read -r GW_INSTANCE GW_METHOD GW_WALLTIME; do
    submit sentinel "$GW_INSTANCE" "$GW_METHOD" \
        --job-name="gwretry_${GW_TAG}_${GW_INSTANCE}_${GW_METHOD}_s" \
        --dependency="afterok:$GW_GATE" --kill-on-invalid-dep=yes \
        --exclude=ruche-gpu13,ruche-gpu16,ruche-gpu19 \
        --gres=gpu:1 --cpus-per-task=2 --mem=32G --time="$GW_WALLTIME" \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_ORIGINAL_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" \
        "$GW_INSTANCE" "$GW_METHOD" 0 "$GW_CONFIG"
    GW_SENTINEL=$GW_JOB
    GW_ALL_JOBS="${GW_ALL_JOBS:+$GW_ALL_JOBS:}$GW_JOB"
    submit replicas "$GW_INSTANCE" "$GW_METHOD" \
        --job-name="gwretry_${GW_TAG}_${GW_INSTANCE}_${GW_METHOD}_r" \
        --dependency="afterok:$GW_SENTINEL" --kill-on-invalid-dep=yes --array=1-5%2 \
        --exclude=ruche-gpu13,ruche-gpu16,ruche-gpu19 \
        --gres=gpu:1 --cpus-per-task=2 --mem=32G --time="$GW_WALLTIME" \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_ORIGINAL_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" \
        "$GW_INSTANCE" "$GW_METHOD" 0 "$GW_CONFIG"
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
done <<'RETRY_PAIRS'
snare_full smoothing-fw 01:10:00
snare_full published-smd 04:15:00
bmmc6224 published-msbpg 06:15:00
RETRY_PAIRS
submit report all all --job-name="gwretry_${GW_TAG}_report" --dependency="afterany:$GW_ALL_JOBS" \
    "$GW_EXTENSION/report.sbatch" "$GW_ORIGINAL_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" "$GW_CONFIG"
date --iso-8601=seconds > "$GW_ROOT/submission_complete.txt"
