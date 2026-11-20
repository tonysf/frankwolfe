#!/bin/bash
# Run on Ruche after source and input references have been verified.
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
for GW_INSTANCE in metric512_s1000 bmmc6224 snare_full shareseq_full; do
    test -f "$GW_ROOT/prepared/$GW_INSTANCE/manifest.json"
done
mkdir "$GW_ROOT/submission_lock"
mkdir -p "$GW_ROOT/scheduler"
GW_RECEIPT="$GW_ROOT/submission.tsv"
printf 'stage\tdataset\tmethod\tjob_id\n' > "$GW_RECEIPT"
GW_EXT=extensions/pair_sampling_20261011
GW_TAG=$(cat SOURCE_COMMIT); GW_TAG=${GW_TAG:0:7}
GW_CPU=$(sbatch --parsable --job-name="gwpair_${GW_TAG}_cpu" --output="$GW_ROOT/scheduler/cpu-%j.out" "$GW_EXT/cpu_checks.sbatch" "$GW_SOURCE" "$GW_ROOT")
GW_CPU=${GW_CPU%%;*}; [[ "$GW_CPU" =~ ^[0-9]+$ ]]
printf 'cpu_checks\tall\tall\t%s\n' "$GW_CPU" | tee -a "$GW_RECEIPT"
GW_ALL="$GW_CPU"
for GW_WORLD in 1 4; do
    GW_MEM=16G; [[ "$GW_WORLD" == 4 ]] && GW_MEM=128G
    GW_GPU=$(sbatch --parsable --job-name="gwpair_${GW_TAG}_check$GW_WORLD" --dependency="afterok:$GW_CPU" --kill-on-invalid-dep=yes \
        --gres="gpu:$GW_WORLD" --cpus-per-task="$((2*GW_WORLD))" --mem="$GW_MEM" --output="$GW_ROOT/scheduler/check$GW_WORLD-%j.out" \
        "$GW_EXT/gpu_checks.sbatch" "$GW_SOURCE" "$GW_ROOT" "$GW_WORLD")
    GW_GPU=${GW_GPU%%;*}; [[ "$GW_GPU" =~ ^[0-9]+$ ]]
    printf 'gpu_checks\tworld%s\tall\t%s\n' "$GW_WORLD" "$GW_GPU" | tee -a "$GW_RECEIPT"
    GW_ALL="$GW_ALL:$GW_GPU"
    if [[ "$GW_WORLD" == 1 ]]; then GW_GPU1=$GW_GPU; else GW_GPU4=$GW_GPU; fi
done
for GW_INSTANCE in metric512_s1000 bmmc6224 snare_full shareseq_full; do
    GW_WORLD=1; GW_MEM=16G; GW_GPU=$GW_GPU1
    if [[ "$GW_INSTANCE" == shareseq_full ]]; then GW_WORLD=4; GW_MEM=128G; GW_GPU=$GW_GPU4; fi
    for GW_METHOD in smoothing-fw published-smd published-msbpg; do
        GW_GATE=$(sbatch --parsable --job-name="gwpair_${GW_TAG}_gate" --dependency="afterok:$GW_GPU" --kill-on-invalid-dep=yes \
            --gres="gpu:$GW_WORLD" --cpus-per-task="$((2*GW_WORLD))" --mem="$GW_MEM" --time=02:15:00 \
            --output="$GW_ROOT/scheduler/gate-$GW_INSTANCE-$GW_METHOD-%j.out" "$GW_EXT/gpu.sbatch" "$GW_SOURCE" "$GW_ROOT" preflight "$GW_INSTANCE" "$GW_METHOD" "$GW_WORLD")
        GW_GATE=${GW_GATE%%;*}; [[ "$GW_GATE" =~ ^[0-9]+$ ]]
        printf 'preflight\t%s\t%s\t%s\n' "$GW_INSTANCE" "$GW_METHOD" "$GW_GATE" | tee -a "$GW_RECEIPT"
        GW_RUNS=$(sbatch --parsable --job-name="gwpair_${GW_TAG}_run" --dependency="afterok:$GW_GATE" --kill-on-invalid-dep=yes --array=0-2%3 \
            --gres="gpu:$GW_WORLD" --cpus-per-task="$((2*GW_WORLD))" --mem="$GW_MEM" \
            --output="$GW_ROOT/scheduler/run-$GW_INSTANCE-$GW_METHOD-%A_%a.out" "$GW_EXT/gpu.sbatch" "$GW_SOURCE" "$GW_ROOT" run "$GW_INSTANCE" "$GW_METHOD" "$GW_WORLD")
        GW_RUNS=${GW_RUNS%%;*}; [[ "$GW_RUNS" =~ ^[0-9]+$ ]]
        printf 'production\t%s\t%s\t%s\n' "$GW_INSTANCE" "$GW_METHOD" "$GW_RUNS" | tee -a "$GW_RECEIPT"
        GW_ALL="$GW_ALL:$GW_GATE:$GW_RUNS"
    done
done
GW_REPORT=$(sbatch --parsable --job-name="gwpair_${GW_TAG}_report" --dependency="afterany:$GW_ALL" \
    --output="$GW_ROOT/scheduler/report-%j.out" "$GW_EXT/report.sbatch" "$GW_SOURCE" "$GW_ROOT")
GW_REPORT=${GW_REPORT%%;*}; [[ "$GW_REPORT" =~ ^[0-9]+$ ]]
printf 'report\tall\tall\t%s\n' "$GW_REPORT" | tee -a "$GW_RECEIPT"
