#!/bin/bash
#SBATCH --job-name=gw_bmmc_control
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/bmmc-control-%j.out
set -euo pipefail
GW_EXTENSION=${1:?immutable extension directory}
GW_PHASE=${2:?gate, select or report}
GW_OUTPUT=${3:?unique campaign output root}
cd "$GW_EXTENSION"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$GW_OUTPUT/report_matplotlib"
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
case "$GW_PHASE" in
  gate) "$GW_PYTHON" "$GW_EXTENSION/campaign.py" gate --out "$GW_OUTPUT" ;;
  select) "$GW_PYTHON" "$GW_EXTENSION/campaign.py" select --out "$GW_OUTPUT" ;;
  report) "$GW_PYTHON" "$GW_EXTENSION/report.py" --out "$GW_OUTPUT" ;;
  *) exit 2 ;;
esac
