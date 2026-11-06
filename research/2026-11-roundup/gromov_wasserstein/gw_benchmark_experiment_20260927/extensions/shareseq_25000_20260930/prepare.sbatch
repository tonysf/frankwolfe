#!/bin/bash
#SBATCH --job-name=gw_shareseq_prepare
#SBATCH --account=fwllm
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/shareseq-prepare-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source directory}
GW_RAW=${2:?frozen public raw data directory}
GW_OUTPUT=${3:?new preparation output}
cd "$GW_SOURCE"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
GW_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_ROOT/.venv/bin/python"
mkdir -p "$GW_OUTPUT"
"$GW_PYTHON" -m pip install --no-index --no-deps --target "$GW_OUTPUT/preparation_dependencies" "$GW_RAW"/wheels/*.whl
export PYTHONPATH="$GW_OUTPUT/preparation_dependencies:$GW_ROOT/cross_dataset_20260929/preparation_430ae68/preparation_dependencies"
"$GW_PYTHON" "$GW_SOURCE/prepare_shareseq.py" --raw-dir "$GW_RAW" --out-dir "$GW_OUTPUT/data"
