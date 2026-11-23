#!/bin/bash
#SBATCH --job-name=gw_bmmc_campaign
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/bmmc-campaign-%A_%a.out
set -euo pipefail
GW_EXTENSION=${1:?immutable extension directory}
GW_PHASE=${2:?tuning or final}
GW_OUTPUT=${3:?unique campaign output root}
case "$GW_PHASE" in tuning|final) ;; *) exit 2;; esac
cd "$GW_EXTENSION"
sha256sum -c SHA256SUMS
GW_ENVIRONMENT="$GW_OUTPUT/environments/${GW_PHASE}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}"
mkdir -p "$GW_ENVIRONMENT"
date --iso-8601=seconds > "$GW_ENVIRONMENT/setup_started.txt"
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$GW_ENVIRONMENT/matplotlib"
GW_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_ROOT/.venv/bin/python"
date --iso-8601=seconds > "$GW_ENVIRONMENT/capture_started.txt"
set +e
timeout --signal=TERM --kill-after=30 600 "$GW_PYTHON" -u "$GW_EXTENSION/startup_probe.py" \
  "$GW_ROOT/releases/b585198/capture_environment.py" --device cuda --out "$GW_ENVIRONMENT/environment.json" \
  > "$GW_ENVIRONMENT/startup.log" 2>&1
GW_STARTUP_EXIT=$?
set -e
printf '{"returncode":%s,"timeout_seconds":600}\n' "$GW_STARTUP_EXIT" > "$GW_ENVIRONMENT/startup_status.json"
if (( GW_STARTUP_EXIT != 0 )); then exit "$GW_STARTUP_EXIT"; fi
"$GW_PYTHON" "$GW_EXTENSION/campaign.py" "$GW_PHASE" --out "$GW_OUTPUT" --task "$SLURM_ARRAY_TASK_ID"
