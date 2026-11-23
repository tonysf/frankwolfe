#!/bin/bash
#SBATCH --job-name=gw_pair_gpucheck
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=ruche-gpu13,ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated campaign root}
GW_WORLD=${3:?one or four GPUs}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
export CUBLAS_WORKSPACE_CONFIG=:4096:8
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" -u extensions/pair_sampling_20261011/verify_checks.py --mode gpu --world-size "$GW_WORLD" --source "$GW_SOURCE" --out "$GW_ROOT/checks/gpu$GW_WORLD"
