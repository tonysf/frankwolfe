#!/bin/bash
#SBATCH --job-name=gw_published_control
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/published-control-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source root}
GW_PREPARED=${2:?prepared root}
GW_OUTPUT=${3:?campaign output root}
GW_STAGE=${4:?prepare, select or report}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SLURM_EXPORT_ENV=ALL MPLCONFIGDIR="$GW_OUTPUT/matplotlib"
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_EXTENSION="$GW_SOURCE/extensions/published_stochastic_20261002"
GW_TASK_LOG="$GW_OUTPUT/scheduler/$SLURM_JOB_ID"
mkdir -p "$GW_TASK_LOG"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_TASK_LOG/exit_code.txt"' EXIT
case "$GW_STAGE" in
  prepare)
    "$GW_PYTHON" -m pip freeze > "$GW_TASK_LOG/environment.lock.txt"
    "$GW_PYTHON" -m pytest -q "$GW_EXTENSION" > "$GW_TASK_LOG/cpu_tests.txt" 2>&1
    "$GW_PYTHON" -u "$GW_EXTENSION/prepare_instances.py" --config "$GW_EXTENSION/campaign.json" --out "$GW_PREPARED"
    ;;
  select)
    "$GW_PYTHON" -u "$GW_EXTENSION/campaign.py" --config "$GW_EXTENSION/campaign.json" \
      --source "$GW_SOURCE" --prepared "$GW_PREPARED" --out "$GW_OUTPUT" \
      --stage select --instance "${5:?instance ID}"
    ;;
  report)
    "$GW_PYTHON" -u "$GW_EXTENSION/report_campaign.py" --config "$GW_EXTENSION/campaign.json" --out "$GW_OUTPUT"
    ;;
  *) exit 2 ;;
esac
