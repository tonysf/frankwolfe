#!/bin/bash
#SBATCH --job-name=gw_momentum_replay
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/momentum-replay-%A_%a.out
set -euo pipefail
# Diagnostic reconstruction only: original scientific outputs and times are read-only.
# The sbatch caller supplies --array and, for finals, the original-array dependency.
GW_REPLAY_MANIFEST=${1:?absolute frozen replay manifest required}
GW_REPLAY_MANIFEST_SHA256=${2:?replay manifest SHA256 required}
GW_REPLAY_SOURCE=${3:?absolute immutable replay utility checkout required}
GW_REPLAY_COMMIT=${4:?full replay utility source commit required}
GW_REPLAY_SHARDS=${5:?explicit shard count required}
GW_REPLAY_TASK=${6:-${SLURM_ARRAY_TASK_ID:-0}}
cd "$GW_REPLAY_SOURCE"
test "$(git rev-parse HEAD)" = "$GW_REPLAY_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_REPLAY_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
# The launcher records environment/artifact bindings for each child; it must
# reject overlapping original/companion paths and changed manifest/source files.
"$GW_REPLAY_PYTHON" run_momentum_replay_shard.py \
  --manifest "$GW_REPLAY_MANIFEST" --manifest-sha256 "$GW_REPLAY_MANIFEST_SHA256" \
  --source-commit "$GW_REPLAY_COMMIT" --task-index "$GW_REPLAY_TASK" \
  --task-count "$GW_REPLAY_SHARDS"
