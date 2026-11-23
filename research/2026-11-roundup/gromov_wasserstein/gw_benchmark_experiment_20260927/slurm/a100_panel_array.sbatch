#!/bin/bash
#SBATCH --job-name=gw_final_panels
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --array=0-55%4
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/final-panels-%A_%a.out
set -euo pipefail
GW_PLAN=${1:?absolute frozen scheduling plan required}
GW_PLAN_SHA256=${2:?scheduling plan SHA256 required}
GW_SOURCE_DIRECTORY=${3:?absolute immutable source directory required}
GW_SOURCE_COMMIT=${4:?expected full source commit required}
cd "$GW_SOURCE_DIRECTORY"
# The dispatcher uses only Python's standard library. Environment activation and
# all numerical work happen in the existing campaign wrapper inside this allocation.
/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python run_panel_array.py \
  --plan "$GW_PLAN" --plan-sha256 "$GW_PLAN_SHA256" --task-index "$SLURM_ARRAY_TASK_ID" \
  --source-directory "$GW_SOURCE_DIRECTORY" --source-commit "$GW_SOURCE_COMMIT"
