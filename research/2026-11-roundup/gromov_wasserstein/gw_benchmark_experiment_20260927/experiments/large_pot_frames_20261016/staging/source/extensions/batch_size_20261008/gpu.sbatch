#!/bin/bash
#SBATCH --job-name=gw_batch_size
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:15:00
#SBATCH --exclude=ruche-gpu13,ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
GW_MODE=${3:?preflight or run}
GW_INSTANCE=${4:-}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_TASK_LOG="$GW_ROOT/scheduler/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-gate}"
mkdir -p "$GW_TASK_LOG"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_TASK_LOG/exit_code.txt"' EXIT
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_TASK_LOG/gpus.csv"
GW_ARGUMENTS=(--config "$GW_SOURCE/extensions/batch_size_20261008/campaign.json"
              --source "$GW_SOURCE" --prepared "$GW_ROOT/prepared")
if [[ "$GW_MODE" == preflight ]]; then
    "$GW_PYTHON" -u extensions/batch_size_20261008/campaign.py preflight \
        "${GW_ARGUMENTS[@]}" --out "$GW_ROOT/preflight"
elif [[ "$GW_MODE" == run ]]; then
    "$GW_PYTHON" -u extensions/batch_size_20261008/campaign.py run \
        "${GW_ARGUMENTS[@]}" --out "$GW_ROOT/results" \
        --gate "$GW_ROOT/preflight/gate.json" --instance "$GW_INSTANCE" \
        --task-index "${SLURM_ARRAY_TASK_ID:?array task required}"
else
    printf 'Unknown mode: %s\n' "$GW_MODE" >&2
    exit 2
fi
