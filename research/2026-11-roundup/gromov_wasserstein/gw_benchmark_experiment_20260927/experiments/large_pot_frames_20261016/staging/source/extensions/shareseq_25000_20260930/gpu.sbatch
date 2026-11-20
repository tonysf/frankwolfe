#!/bin/bash
#SBATCH --job-name=gw_shareseq_full_4a100
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/shareseq-4a100-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source directory}
GW_PREPARED=${2:?preparation data directory}
GW_OUTPUT=${3:?new GPU output directory}
cd "$GW_SOURCE"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
mkdir -p "$GW_OUTPUT"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_OUTPUT/gpus.csv"
nvidia-smi topo -m > "$GW_OUTPUT/topology.txt"
"$GW_PYTHON" -u "$GW_SOURCE/campaign.py" --source "$GW_SOURCE" --prepared "$GW_PREPARED" --out "$GW_OUTPUT"
