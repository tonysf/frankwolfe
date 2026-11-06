#!/bin/bash
#SBATCH --job-name=gw_stage_c
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/campaign-%A_%a.out
set -euo pipefail
# Submit from an immutable source checkout and pass the frozen manifest and
# unique campaign output root as positional arguments. An optional fifth argument
# sets the shard count; pass the matching --array range to sbatch. No jobs mutate source.
GW_CAMPAIGN_MANIFEST=${1:?frozen manifest path required}
GW_CAMPAIGN_OUTPUT=${2:?unique output root required}
GW_SOURCE_DIRECTORY=${3:?absolute immutable source directory required}
GW_SOURCE_COMMIT=${4:?expected full source commit required}
GW_SHARD_COUNT=${5:-4}
GW_TASK_INDEX=${6:-$SLURM_ARRAY_TASK_ID}
cd "$GW_SOURCE_DIRECTORY"
test "$(git rev-parse HEAD)" = "$GW_SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
GW_SHARD_ENV="$GW_CAMPAIGN_OUTPUT/_campaign/environment_${GW_TASK_INDEX}_${SLURM_JOB_ID}"
mkdir -p "$GW_CAMPAIGN_OUTPUT/_campaign"
mkdir "$GW_SHARD_ENV"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$GW_SHARD_ENV/matplotlib"
"$GW_PYTHON" capture_environment.py --device cuda --out "$GW_SHARD_ENV/environment.json"
"$GW_PYTHON" -m pip freeze > "$GW_SHARD_ENV/environment.lock.txt"
"$GW_PYTHON" run_campaign_shard.py --manifest "$GW_CAMPAIGN_MANIFEST" --source-commit "$GW_SOURCE_COMMIT" \
  --data-root "$GW_EXPERIMENT_ROOT/data" --out-root "$GW_CAMPAIGN_OUTPUT" \
  --task-index "$GW_TASK_INDEX" --task-count "$GW_SHARD_COUNT"
