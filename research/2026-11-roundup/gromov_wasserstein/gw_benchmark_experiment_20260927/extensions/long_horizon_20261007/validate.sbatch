#!/bin/bash
#SBATCH --job-name=gw_long_validate
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --exclude=ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/long-validate-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_VALIDATION=${2:?new validation output}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
"$GW_PYTHON" -u extensions/long_horizon_20261007/validate_smoke.py --source "$GW_SOURCE" --out "$GW_VALIDATION"
