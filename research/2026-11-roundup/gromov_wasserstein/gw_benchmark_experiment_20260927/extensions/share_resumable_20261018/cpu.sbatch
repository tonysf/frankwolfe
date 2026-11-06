#!/bin/bash
#SBATCH --job-name=gw_share_resume_cpu
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
GW_PHASE=${3:?cpu-gate dispatch or report}
GW_DATASET=${4:-}
GW_PARENT_JOB=${5:-}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL CUBLAS_WORKSPACE_CONFIG=:4096:8
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_ARGS=("$GW_PHASE" --source "$GW_SOURCE" --root "$GW_ROOT")
if [[ -n "$GW_DATASET" ]]; then GW_ARGS+=(--dataset "$GW_DATASET"); fi
if [[ -n "$GW_PARENT_JOB" ]]; then GW_ARGS+=(--parent-job "$GW_PARENT_JOB"); fi
"$GW_PYTHON" -u extensions/share_resumable_20261018/campaign.py "${GW_ARGS[@]}"
