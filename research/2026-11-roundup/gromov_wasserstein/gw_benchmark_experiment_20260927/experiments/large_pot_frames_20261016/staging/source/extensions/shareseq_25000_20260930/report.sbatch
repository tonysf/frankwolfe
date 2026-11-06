#!/bin/bash
#SBATCH --job-name=gw_shareseq_full_report
#SBATCH --account=fwllm
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/shareseq-report-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable report source}
GW_CAMPAIGN=${2:?completed campaign directory}
cd "$GW_SOURCE"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$GW_CAMPAIGN/matplotlib"
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" "$GW_SOURCE/report.py" --campaign "$GW_CAMPAIGN"
