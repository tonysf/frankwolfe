#!/bin/bash
#SBATCH --job-name=gw_replay_check
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/momentum-replay-check-%j.out
set -euo pipefail
GW_REPLAY_SOURCE=${1:?absolute immutable replay utility checkout required}
GW_REPLAY_COMMIT=${2:?full replay utility source commit required}
GW_REPLAY_PILOT_MANIFEST=${3:?absolute frozen pilot replay manifest required}
GW_REPLAY_PILOT_MANIFEST_SHA256=${4:?pilot replay manifest SHA256 required}
cd "$GW_REPLAY_SOURCE"
test "$(git rev-parse HEAD)" = "$GW_REPLAY_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_REPLAY_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
GW_REPLAY_OUTPUT="$GW_EXPERIMENT_ROOT/a100_campaign/momentum_replay_preflight_$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$GW_REPLAY_OUTPUT/matplotlib"
"$GW_REPLAY_PYTHON" validate_momentum_replay_gpu.py \
  --experiment-root "$GW_EXPERIMENT_ROOT" --source-commit "$GW_REPLAY_COMMIT" \
  --pilot-manifest "$GW_REPLAY_PILOT_MANIFEST" \
  --pilot-manifest-sha256 "$GW_REPLAY_PILOT_MANIFEST_SHA256" --out "$GW_REPLAY_OUTPUT"
