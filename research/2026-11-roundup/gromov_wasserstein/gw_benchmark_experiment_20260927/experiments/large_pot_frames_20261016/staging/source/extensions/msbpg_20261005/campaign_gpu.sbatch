#!/bin/bash
#SBATCH --job-name=gw_msbpg_campaign
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/msbpg-campaign-%A_%a.out
set -euo pipefail
GW_SOURCE=${1:?immutable source root}
GW_PREPARED=${2:?prepared root}
GW_OUTPUT=${3:?campaign output root}
GW_STAGE=${4:?tuning or final}
GW_INSTANCE=${5:?instance ID}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_EXTENSION="$GW_SOURCE/extensions/msbpg_20261005"
GW_TASK_LOG="$GW_OUTPUT/scheduler/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$GW_TASK_LOG"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_TASK_LOG/exit_code.txt"' EXIT
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_TASK_LOG/gpus.csv"
"$GW_PYTHON" -u "$GW_EXTENSION/campaign.py" --config "$GW_EXTENSION/campaign.json" \
    --source "$GW_SOURCE" --prepared "$GW_PREPARED" --out "$GW_OUTPUT" \
    --stage "$GW_STAGE" --instance "$GW_INSTANCE" --task-index "$SLURM_ARRAY_TASK_ID"
