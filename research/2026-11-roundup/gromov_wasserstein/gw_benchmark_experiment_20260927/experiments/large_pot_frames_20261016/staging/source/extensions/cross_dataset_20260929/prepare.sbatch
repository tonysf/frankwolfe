#!/bin/bash
#SBATCH --job-name=gw_snare_prepare
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/snare-prepare-%j.out
set -euo pipefail
GW_PREP_SOURCE=${1:?immutable preparation payload}
GW_OUTPUT=${2:?new preparation output}
cd "$GW_PREP_SOURCE"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
mkdir -p "$GW_OUTPUT"
"$GW_PYTHON" -m pip install --no-index --no-deps --target "$GW_OUTPUT/preparation_dependencies" \
  "$GW_PREP_SOURCE"/wheels/*.whl
export PYTHONPATH="$GW_OUTPUT/preparation_dependencies"
"$GW_PYTHON" "$GW_PREP_SOURCE/prepare_snare.py" --raw-dir "$GW_PREP_SOURCE/raw" --out-dir "$GW_OUTPUT/data"
cp /gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/data/bmmc1000.npz "$GW_OUTPUT/data/bmmc1000.npz"
cp /gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/data/bmmc1000.json "$GW_OUTPUT/data/bmmc1000.json"
sha256sum "$GW_OUTPUT/data/bmmc1000.npz" > "$GW_OUTPUT/canonical_bmmc_sha256.txt"
