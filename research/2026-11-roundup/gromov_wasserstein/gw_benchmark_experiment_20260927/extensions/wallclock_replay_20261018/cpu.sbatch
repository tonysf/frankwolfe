#!/bin/bash
#SBATCH --job-name=gwtime_cpu
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
GW_PHASE=${3:?cpu-gate dispatch or report}
GW_UNUSED_DATASET=${4:-}
GW_PARENT_JOB=${5:-}
if [[ -n "$GW_UNUSED_DATASET" ]]; then
  echo 'Replay has one serial stream; per-dataset submission is invalid' >&2
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
GW_ARGS=("$GW_PHASE" --source "$GW_SOURCE" --root "$GW_ROOT")
if [[ -n "$GW_PARENT_JOB" ]]; then GW_ARGS+=(--parent-job "$GW_PARENT_JOB"); fi
"$GW_PYTHON" -u extensions/wallclock_replay_20261018/campaign.py "${GW_ARGS[@]}"
