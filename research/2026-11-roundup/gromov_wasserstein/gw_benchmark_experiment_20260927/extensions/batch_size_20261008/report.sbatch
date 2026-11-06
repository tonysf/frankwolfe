#!/bin/bash
#SBATCH --job-name=gw_batch_report
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL MPLCONFIGDIR="$GW_ROOT/report/matplotlib"
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" -u extensions/batch_size_20261008/report_campaign.py \
    --config "$GW_SOURCE/extensions/batch_size_20261008/campaign.json" \
    --source "$GW_SOURCE" --prepared "$GW_ROOT/prepared" \
    --out "$GW_ROOT/results" --report-dir "$GW_ROOT/report"
