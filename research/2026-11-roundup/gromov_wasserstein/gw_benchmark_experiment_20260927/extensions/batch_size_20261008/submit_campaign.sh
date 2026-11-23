#!/bin/bash
# Run only on the Ruche login host after immutable transfer verification.
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
test -d "$GW_ROOT/prepared/bmmc6224"
test -d "$GW_ROOT/prepared/metric512_s1000"
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
# Exclusive marker prevents an uncertain/partial submission being repeated.
mkdir "$GW_ROOT/submission_lock"
mkdir -p "$GW_ROOT/scheduler"
GW_RECEIPT="$GW_ROOT/submission.tsv"
printf 'stage\tdataset\tjob_id\n' > "$GW_RECEIPT"
GW_TAG=$(cat SOURCE_COMMIT)
GW_TAG=${GW_TAG:0:7}
GW_GATE=$(sbatch --parsable --job-name="gwbatch_${GW_TAG}_gate" \
    --time=00:45:00 --output="$GW_ROOT/scheduler/gate-%j.out" \
    extensions/batch_size_20261008/gpu.sbatch "$GW_SOURCE" "$GW_ROOT" preflight)
GW_GATE=${GW_GATE%%;*}
[[ "$GW_GATE" =~ ^[0-9]+$ ]]
printf 'preflight\tall\t%s\n' "$GW_GATE" | tee -a "$GW_RECEIPT"
GW_ARRAYS=()
for GW_INSTANCE in bmmc6224 metric512_s1000; do
    GW_JOB=$(sbatch --parsable --job-name="gwbatch_${GW_TAG}_${GW_INSTANCE}" \
        --dependency="afterok:$GW_GATE" --kill-on-invalid-dep=yes --array=0-8%2 \
        --output="$GW_ROOT/scheduler/${GW_INSTANCE}-%A_%a.out" \
        extensions/batch_size_20261008/gpu.sbatch "$GW_SOURCE" "$GW_ROOT" run "$GW_INSTANCE")
    GW_JOB=${GW_JOB%%;*}
    [[ "$GW_JOB" =~ ^[0-9]+$ ]]
    GW_ARRAYS+=("$GW_JOB")
    printf 'runs\t%s\t%s\n' "$GW_INSTANCE" "$GW_JOB" | tee -a "$GW_RECEIPT"
done
GW_REPORT=$(sbatch --parsable --job-name="gwbatch_${GW_TAG}_report" \
    --dependency="afterany:$GW_GATE:${GW_ARRAYS[0]}:${GW_ARRAYS[1]}" \
    --output="$GW_ROOT/scheduler/report-%j.out" \
    extensions/batch_size_20261008/report.sbatch "$GW_SOURCE" "$GW_ROOT")
GW_REPORT=${GW_REPORT%%;*}
[[ "$GW_REPORT" =~ ^[0-9]+$ ]]
printf 'report\tall\t%s\n' "$GW_REPORT" | tee -a "$GW_RECEIPT"
