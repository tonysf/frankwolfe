#!/bin/bash
#SBATCH --job-name=gw_scot_compare
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
GW_ROOT=${2:?isolated experiment root}
GW_MODE=${3:?preflight tune or confirm}
GW_CPU_CHECKS=${4:?validated numerical CPU receipt}
GW_GPU_CHECKS=${5:?validated numerical A100 receipt}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_LOG="$GW_ROOT/scheduler/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:?array task}"
mkdir "$GW_LOG"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_LOG/exit_code.txt"' EXIT
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_LOG/gpus.csv"
GW_ARGS=(--config "$GW_SOURCE/extensions/scot_reproduction_20261014/config.json"
    --source "$GW_SOURCE" --reference "$GW_ROOT/reference/reference.json" --out "$GW_ROOT/comparison")
case "$GW_MODE" in
    preflight)
        GW_ARGS+=(--candidate-index "$SLURM_ARRAY_TASK_ID" --cpu-checks "$GW_CPU_CHECKS" --gpu-checks "$GW_GPU_CHECKS");;
    tune)
        GW_ARGS+=(--candidate-index "$SLURM_ARRAY_TASK_ID");;
    confirm)
        GW_ARGS+=(--task-index "$SLURM_ARRAY_TASK_ID");;
    *) exit 2;;
esac
"$GW_PYTHON" -u extensions/scot_reproduction_20261014/comparison.py "$GW_MODE" "${GW_ARGS[@]}"
