#!/bin/bash
#SBATCH --job-name=gw_long_all_report
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/long-all-report-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable reporting source}
GW_MANIFEST=${2:?frozen campaign routes}
GW_OUTPUT=${3:?combined report output}
GW_MANIFEST_SHA=${4:?route manifest checksum}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
printf '%s  %s\n' "$GW_MANIFEST_SHA" "$GW_MANIFEST" | sha256sum --status -c -
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SLURM_EXPORT_ENV=ALL MPLCONFIGDIR="$GW_OUTPUT/matplotlib"
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" -u extensions/long_horizon_20261007/combine_campaigns.py --manifest "$GW_MANIFEST" --out "$GW_OUTPUT"
