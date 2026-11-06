#!/bin/bash
# Called once on Ruche after source transfer and checksum verification.
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_PREPARED=${2:?frozen prepared inputs}
GW_ROOT=${3:?new campaign root}
GW_TAG=${4:?source short commit}
GW_EXTENSION="$GW_SOURCE/extensions/long_horizon_20261007"
GW_OUTPUT="$GW_ROOT/results"
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
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
submit validation all all --job-name="gwlong_${GW_TAG}_validate" \
    "$GW_EXTENSION/validate.sbatch" "$GW_SOURCE" "$GW_ROOT/validation"
GW_GATE=$GW_JOB
GW_ALL_JOBS=$GW_GATE
while IFS=$'\t' read -r GW_INSTANCE GW_METHOD GW_GPUS GW_CPUS GW_MEM GW_WALLTIME; do
    [[ "$GW_INSTANCE" == dataset ]] && continue
    test -f "$GW_PREPARED/$GW_INSTANCE/manifest.json"
    GW_THROTTLE=2
    [[ "$GW_GPUS" == 4 ]] && GW_THROTTLE=1
    submit sentinel "$GW_INSTANCE" "$GW_METHOD" \
        --job-name="gwlong_${GW_TAG}_${GW_INSTANCE}_${GW_METHOD}_s" \
        --dependency="afterok:$GW_GATE" --kill-on-invalid-dep=yes \
        --gres="gpu:$GW_GPUS" --cpus-per-task="$GW_CPUS" --mem="$GW_MEM" --time="$GW_WALLTIME" \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" "$GW_INSTANCE" "$GW_METHOD" 0
    GW_SENTINEL=$GW_JOB
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
    submit replicas "$GW_INSTANCE" "$GW_METHOD" \
        --job-name="gwlong_${GW_TAG}_${GW_INSTANCE}_${GW_METHOD}_r" \
        --dependency="afterok:$GW_SENTINEL" --kill-on-invalid-dep=yes \
        --array="1-5%$GW_THROTTLE" --gres="gpu:$GW_GPUS" --cpus-per-task="$GW_CPUS" --mem="$GW_MEM" --time="$GW_WALLTIME" \
        "$GW_EXTENSION/campaign_gpu.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT" "$GW_INSTANCE" "$GW_METHOD"
    GW_ALL_JOBS="$GW_ALL_JOBS:$GW_JOB"
done < "$GW_EXTENSION/submission_plan.tsv"
submit report all all --job-name="gwlong_${GW_TAG}_report" --dependency="afterany:$GW_ALL_JOBS" \
    "$GW_EXTENSION/report.sbatch" "$GW_SOURCE" "$GW_PREPARED" "$GW_OUTPUT"
date --iso-8601=seconds > "$GW_ROOT/submission_complete.txt"
