#!/bin/bash
# Submit once from the verified Ruche source directory. No monitoring loop.
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated experiment root}
GW_CPU_CHECKS=${3:?validated numerical CPU receipt}
GW_GPU_CHECKS=${4:?validated numerical A100 receipt}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
test -f "$GW_CPU_CHECKS"
test -f "$GW_GPU_CHECKS"
(cd "$GW_ROOT/inputs" && sha256sum --status -c SHA256SUMS)
(cd "$GW_ROOT/wheelhouse" && sha256sum --status -c SHA256SUMS)
mkdir "$GW_ROOT/submission_lock"
mkdir -p "$GW_ROOT/scheduler"
GW_EXT=extensions/scot_reproduction_20261014
GW_RECEIPT="$GW_ROOT/submission.tsv"
printf 'stage\tjob_id\n' > "$GW_RECEIPT"
record_job() {
    local stage=$1 result=$2
    result=${result%%;*}
    [[ "$result" =~ ^[0-9]+$ ]]
    printf '%s\t%s\n' "$stage" "$result" >> "$GW_RECEIPT"
    printf '%s\n' "$result"
}
GW_REF=$(record_job reference "$(sbatch --parsable --output="$GW_ROOT/scheduler/reference-%j.out" \
    "$GW_EXT/reference.sbatch" "$GW_SOURCE" "$GW_ROOT")")
GW_PREFLIGHT=$(record_job preflight "$(sbatch --parsable --job-name=gw_scot_preflight --array=0-26%4 --time=01:00:00 \
    --dependency="afterok:$GW_REF" --kill-on-invalid-dep=yes --output="$GW_ROOT/scheduler/preflight-%A_%a.out" \
    "$GW_EXT/gpu.sbatch" "$GW_SOURCE" "$GW_ROOT" preflight "$GW_CPU_CHECKS" "$GW_GPU_CHECKS")")
GW_TUNE=$(record_job tuning "$(sbatch --parsable --job-name=gw_scot_tuning --array=0-26%4 \
    --dependency="afterok:$GW_PREFLIGHT" --kill-on-invalid-dep=yes --output="$GW_ROOT/scheduler/tuning-%A_%a.out" \
    "$GW_EXT/gpu.sbatch" "$GW_SOURCE" "$GW_ROOT" tune "$GW_CPU_CHECKS" "$GW_GPU_CHECKS")")
GW_SELECT=$(record_job selection "$(sbatch --parsable --job-name=gw_scot_select --dependency="afterok:$GW_TUNE" --kill-on-invalid-dep=yes \
    --output="$GW_ROOT/scheduler/selection-%j.out" "$GW_EXT/summary.sbatch" "$GW_SOURCE" "$GW_ROOT" select)")
GW_CONFIRM=$(record_job confirmation "$(sbatch --parsable --job-name=gw_scot_confirm --array=0-8%4 \
    --dependency="afterok:$GW_SELECT" --kill-on-invalid-dep=yes --output="$GW_ROOT/scheduler/confirmation-%A_%a.out" \
    "$GW_EXT/gpu.sbatch" "$GW_SOURCE" "$GW_ROOT" confirm "$GW_CPU_CHECKS" "$GW_GPU_CHECKS")")
GW_REPORT=$(record_job report "$(sbatch --parsable --job-name=gw_scot_report --dependency="afterany:$GW_REF:$GW_PREFLIGHT:$GW_TUNE:$GW_SELECT:$GW_CONFIRM" \
    --output="$GW_ROOT/scheduler/report-%j.out" "$GW_EXT/summary.sbatch" "$GW_SOURCE" "$GW_ROOT" report)")
cat "$GW_RECEIPT"
