#!/bin/bash
#SBATCH --job-name=gw_published_smd_check
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --exclude=ruche-gpu16
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/published-smd-validation-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source root}
GW_PREPARED=${2:?prepared SHARE-seq data directory}
GW_OUTPUT=${3:?new validation output directory}
cd "$GW_SOURCE"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_EXTENSION="$GW_SOURCE/extensions/published_stochastic_20261002"
mkdir -p "$GW_OUTPUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_OUTPUT/exit_code.txt"' EXIT
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_OUTPUT/gpus.csv"
"$GW_PYTHON" -m pip freeze > "$GW_OUTPUT/environment.lock.txt"
"$GW_PYTHON" -m pytest -q "$GW_EXTENSION" > "$GW_OUTPUT/cpu_tests.txt" 2>&1
"$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$GW_EXTENSION/validate_distributed.py" --device cuda --out "$GW_OUTPUT/distributed_cuda.json"
# Actual-size runner smoke follows only after numerical correctness passes.
# Three updates validate integration/memory/tolerances, not scientific ranking.
"$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$GW_EXTENSION/run_published.py" --device cuda --data-dir "$GW_PREPARED" \
    --method published-smd --iterations 3 --batch 128 --seed 610 --output-seed 910 \
    --dtype float32 --out "$GW_OUTPUT/shareseq_full_smd_smoke"
"$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$GW_EXTENSION/run_published.py" --device cuda --data-dir "$GW_PREPARED" \
    --method smoothing-fw --iterations 3 --batch 128 --seed 610 --output-seed 910 \
    --gamma0 .1 --rho0 .3 --mu-scale .1 --dtype float32 --out "$GW_OUTPUT/shareseq_full_fw_smoke"
