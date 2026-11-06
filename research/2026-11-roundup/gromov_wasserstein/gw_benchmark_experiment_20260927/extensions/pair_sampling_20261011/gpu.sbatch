#!/bin/bash
#SBATCH --job-name=gw_pair
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --exclude=ruche-gpu13,ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
GW_MODE=${3:?preflight or run}
GW_INSTANCE=${4:?dataset}
GW_METHOD=${5:?method}
GW_WORLD=${6:?GPU topology}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_LOG="$GW_ROOT/scheduler/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-gate}"
mkdir -p "$GW_LOG"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_LOG/exit_code.txt"' EXIT
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_LOG/gpus.csv"
GW_ARGS=(--config "$GW_SOURCE/extensions/pair_sampling_20261011/campaign.json" --source "$GW_SOURCE" --prepared "$GW_ROOT/prepared" --instance "$GW_INSTANCE" --method "$GW_METHOD")
if [[ "$GW_MODE" == preflight ]]; then
    "$GW_PYTHON" -u extensions/pair_sampling_20261011/campaign.py preflight "${GW_ARGS[@]}" \
        --out "$GW_ROOT/preflight/$GW_INSTANCE/$GW_METHOD" --cpu-checks "$GW_ROOT/checks/cpu/checks.json" --gpu-checks "$GW_ROOT/checks/gpu$GW_WORLD/checks.json"
elif [[ "$GW_MODE" == run ]]; then
    "$GW_PYTHON" -u extensions/pair_sampling_20261011/campaign.py run "${GW_ARGS[@]}" \
        --out "$GW_ROOT/results" --gate "$GW_ROOT/preflight/$GW_INSTANCE/$GW_METHOD/gate.json" --task-index "${SLURM_ARRAY_TASK_ID:?array task}"
else
    exit 2
fi
