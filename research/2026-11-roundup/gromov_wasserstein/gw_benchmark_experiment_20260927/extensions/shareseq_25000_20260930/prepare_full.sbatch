#!/bin/bash
#SBATCH --job-name=gw_shareseq_full_prepare
#SBATCH --account=fwllm
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/shareseq-full-prepare-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source directory}
GW_PREVIOUS=${2:?previous completed preparation data directory}
GW_OUTPUT=${3:?new full-population preparation output}
cd "$GW_SOURCE"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
GW_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_ROOT/.venv/bin/python"
export PYTHONPATH="$GW_ROOT/shareseq_20260930/preparation_8c98135/preparation_dependencies:$GW_ROOT/cross_dataset_20260929/preparation_430ae68/preparation_dependencies"
"$GW_PYTHON" "$GW_SOURCE/prepare_full.py" --previous-preparation "$GW_PREVIOUS" --out-dir "$GW_OUTPUT/data" --threads 8
