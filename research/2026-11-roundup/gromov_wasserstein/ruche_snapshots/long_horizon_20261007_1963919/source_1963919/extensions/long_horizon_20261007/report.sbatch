#!/bin/bash
#SBATCH --job-name=gw_long_report
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/long-report-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_PREPARED=${2:?frozen prepared inputs}
GW_OUTPUT=${3:?new results}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SLURM_EXPORT_ENV=ALL MPLCONFIGDIR="$GW_OUTPUT/matplotlib"
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" -u extensions/long_horizon_20261007/report_campaign.py \
    --config "$GW_SOURCE/extensions/long_horizon_20261007/campaign.json" \
    --source "$GW_SOURCE" --prepared "$GW_PREPARED" --out "$GW_OUTPUT"
