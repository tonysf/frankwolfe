#!/bin/bash
#SBATCH --job-name=gw_long_run
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:15:00
#SBATCH --exclude=ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/long-run-%A_%a.out
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_PREPARED=${2:?frozen prepared inputs}
GW_OUTPUT=${3:?new results}
GW_INSTANCE=${4:?dataset}
GW_METHOD=${5:?method}
GW_TASK=${SLURM_ARRAY_TASK_ID:-${6:-0}}
GW_CONFIG=${7:-$GW_SOURCE/extensions/long_horizon_20261007/campaign.json}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_TASK_LOG="$GW_OUTPUT/scheduler/${SLURM_JOB_ID}_${GW_TASK}"
mkdir -p "$GW_TASK_LOG"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_TASK_LOG/exit_code.txt"' EXIT
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_TASK_LOG/gpus.csv"
"$GW_PYTHON" -u extensions/long_horizon_20261007/campaign.py \
    --config "$GW_CONFIG" \
    --source "$GW_SOURCE" --prepared "$GW_PREPARED" --out "$GW_OUTPUT" \
    --instance "$GW_INSTANCE" --method "$GW_METHOD" --task-index "$GW_TASK"
