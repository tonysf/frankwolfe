#!/bin/bash
# Run on Ruche after transferring and verifying the immutable source package.
# Source, prepared data, and output paths must be verified absolute paths.
set -euo pipefail
GW_SOURCE=${1:?immutable source root}
GW_PREPARED=${2:?existing verified prepared data root}
GW_ROOT=${3:?new campaign root}
GW_TAG=${4:?source short commit}
GW_OUTPUT="$GW_ROOT/results"
GW_EXTENSION="$GW_SOURCE/extensions/msbpg_20261005"
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
for GW_INSTANCE in bmmc6224 snare_full shareseq_full; do
    test -f "$GW_PREPARED/$GW_INSTANCE/manifest.json"
done
mkdir "$GW_ROOT/submission_started" "$GW_OUTPUT"
GW_RECEIPT="$GW_ROOT/submission.tsv"
printf 'stage\tinstance\tjob_id\n' > "$GW_RECEIPT"
date --iso-8601=seconds > "$GW_ROOT/submitted_at.txt"
submit() {
    local stage=$1 instance=$2
    shift 2
    GW_JOB=$(sbatch --parsable --export=NONE --propagate=NONE "$@")
    GW_JOB=${GW_JOB%%;*}
    [[ "$GW_JOB" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\n' "$stage" "$instance" "$GW_JOB" >> "$GW_RECEIPT"
    printf '%s\t%s\t%s\n' "$stage" "$instance" "$GW_JOB"
}
submit validation all --job-name="gwms_${GW_TAG}_validate" \
    "$GW_EXTENSION/validate.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_ROOT/validation"
GW_GATE=$GW_JOB
GW_ALL_JOBS=$GW_GATE
for GW_INSTANCE in bmmc6224 snare_full shareseq_full; do
    GW_GPUS=1 GW_CPUS=2 GW_MEM=32G GW_THROTTLE=2
    if [[ "$GW_INSTANCE" == shareseq_full ]]; then
        GW_GPUS=4 GW_CPUS=8 GW_MEM=128G GW_THROTTLE=1
    fi
    submit tuning "$GW_INSTANCE" --job-name="gwms_${GW_TAG}_${GW_INSTANCE}_tune" \
        --dependency="afterok:$GW_GATE" --kill-on-invalid-dep=yes \
        --array="0-11%$GW_THROTTLE" --gres="gpu:$GW_GPUS" --cpus-per-task="$GW_CPUS" --mem="$GW_MEM" \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" tuning "$GW_INSTANCE"
    GW_TUNING_JOB=$GW_JOB
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
    submit select "$GW_INSTANCE" --job-name="gwms_${GW_TAG}_${GW_INSTANCE}_select" \
        --dependency="afterok:$GW_TUNING_JOB" --kill-on-invalid-dep=yes \
        "$GW_EXTENSION/campaign_cpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" select "$GW_INSTANCE"
    GW_SELECT_JOB=$GW_JOB
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
    submit final "$GW_INSTANCE" --job-name="gwms_${GW_TAG}_${GW_INSTANCE}_final" \
        --dependency="afterok:$GW_SELECT_JOB" --kill-on-invalid-dep=yes \
        --array="0-5%$GW_THROTTLE" --gres="gpu:$GW_GPUS" --cpus-per-task="$GW_CPUS" --mem="$GW_MEM" --time=02:15:00 \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" final "$GW_INSTANCE"
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
done
submit report all --job-name="gwms_${GW_TAG}_report" --dependency="afterany:$GW_ALL_JOBS" \
    "$GW_EXTENSION/campaign_cpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" report
date --iso-8601=seconds > "$GW_ROOT/submission_complete.txt"
