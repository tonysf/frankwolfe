#!/bin/bash
# Called once on Ruche after immutable source transfer and checksum verification.
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_PREPARED=${2:?frozen prepared inputs}
GW_ROOT=${3:?new SHARE supplement root}
GW_TAG=${4:?source short commit}
GW_GATE=${5:?long-horizon GPU integration job ID}
[[ "$GW_GATE" =~ ^[0-9]+$ ]]
GW_EXTENSION="$GW_SOURCE/extensions/long_horizon_20261007"
GW_CONFIG="$GW_SOURCE/extensions/long_horizon_share_20261007/campaign.json"
GW_OUTPUT="$GW_ROOT/results"
GW_INSTANCE=shareseq_full
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
test -f "$GW_CONFIG"
test -f "$GW_PREPARED/$GW_INSTANCE/manifest.json"
mkdir "$GW_ROOT/submission_started" "$GW_OUTPUT"
GW_RECEIPT="$GW_ROOT/submission.tsv"
printf 'stage\tdataset\tmethod\tjob_id\n' > "$GW_RECEIPT"
date --iso-8601=seconds > "$GW_ROOT/submitted_at.txt"
printf '%s\n' "$GW_GATE" > "$GW_ROOT/existing_validation_job.txt"
submit() {
    local stage=$1 dataset=$2 method=$3
    shift 3
    GW_JOB=$(sbatch --parsable --export=NONE --propagate=NONE "$@")
    GW_JOB=${GW_JOB%%;*}
    [[ "$GW_JOB" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\t%s\n' "$stage" "$dataset" "$method" "$GW_JOB" >> "$GW_RECEIPT"
    printf '%s\t%s\t%s\t%s\n' "$stage" "$dataset" "$method" "$GW_JOB"
}
GW_ALL_JOBS=$GW_GATE
for GW_METHOD in smoothing-fw published-smd published-msbpg; do
    case "$GW_METHOD" in
        smoothing-fw) GW_WALLTIME=02:15:00 ;;
        published-smd) GW_WALLTIME=12:15:00 ;;
        published-msbpg) GW_WALLTIME=18:15:00 ;;
    esac
    submit sentinel "$GW_INSTANCE" "$GW_METHOD" \
        --job-name="gwlong_${GW_TAG}_share_${GW_METHOD}_s" \
        --dependency="afterok:$GW_GATE" --kill-on-invalid-dep=yes \
        --gres=gpu:4 --cpus-per-task=8 --mem=128G --time="$GW_WALLTIME" \
        --exclude=ruche-gpu13,ruche-gpu16,ruche-gpu19 \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" \
        "$GW_INSTANCE" "$GW_METHOD" 0 "$GW_CONFIG"
    GW_SENTINEL=$GW_JOB
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
    submit replicas "$GW_INSTANCE" "$GW_METHOD" \
        --job-name="gwlong_${GW_TAG}_share_${GW_METHOD}_r" \
        --dependency="afterok:$GW_SENTINEL" --kill-on-invalid-dep=yes \
        --array=1-5%1 --gres=gpu:4 --cpus-per-task=8 --mem=128G --time="$GW_WALLTIME" \
        --exclude=ruche-gpu13,ruche-gpu16,ruche-gpu19 \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" \
        "$GW_INSTANCE" "$GW_METHOD" 0 "$GW_CONFIG"
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
done
submit report all all --job-name="gwlong_${GW_TAG}_share_report" \
    --dependency="afterany:$GW_ALL_JOBS" \
    "$GW_EXTENSION/report.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" "$GW_CONFIG"
date --iso-8601=seconds > "$GW_ROOT/submission_complete.txt"
