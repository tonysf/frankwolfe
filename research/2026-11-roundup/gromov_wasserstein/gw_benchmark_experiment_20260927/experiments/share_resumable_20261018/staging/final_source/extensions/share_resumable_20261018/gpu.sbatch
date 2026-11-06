#!/bin/bash
#SBATCH --job-name=gw_share_resume
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Continuations are submitted by the afterany CPU dispatcher, not automatic requeue.
#SBATCH --no-requeue
# Historical infrastructure workaround: prior CUDA busy/unavailable failures,
# not a site policy. No subsequent repaired-node validation is recorded.
#SBATCH --exclude=ruche-gpu13,ruche-gpu15,ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
GW_PHASE=${3:?tiny-gate dataset-gate or production}
GW_DATASET=${4:-}
GW_WORLD=${5:?GPU count}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL CUBLAS_WORKSPACE_CONFIG=:4096:8
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_ARGS=("$GW_PHASE" --source "$GW_SOURCE" --root "$GW_ROOT" --world "$GW_WORLD")
if [[ -n "$GW_DATASET" ]]; then GW_ARGS+=(--dataset "$GW_DATASET"); fi
"$GW_PYTHON" -u extensions/share_resumable_20261018/campaign.py "${GW_ARGS[@]}"
