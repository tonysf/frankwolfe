#!/bin/bash
#SBATCH --job-name=gwtime_gpu
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# Historical CUDA busy/unavailable failures; excludes are an experiment workaround.
#SBATCH --exclude=ruche-gpu13,ruche-gpu15,ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
GW_PHASE=${3:?gpu-gate or production}
GW_UNUSED_DATASET=${4:-}
GW_WORLD=${5:?GPU count}
if [[ -n "$GW_UNUSED_DATASET" || "$GW_WORLD" != 1 ]]; then
  echo 'Replay requires one serial stream on one A100' >&2
  exit 2
fi
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL CUBLAS_WORKSPACE_CONFIG=:4096:8
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" -u extensions/wallclock_replay_20261018/campaign.py "$GW_PHASE" --source "$GW_SOURCE" --root "$GW_ROOT" --world 1
