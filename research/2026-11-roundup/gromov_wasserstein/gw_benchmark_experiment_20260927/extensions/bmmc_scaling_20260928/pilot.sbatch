#!/bin/bash
#SBATCH --job-name=gw_bmmc_scale
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/bmmc-scale-%j.out
set -euo pipefail
GW_EXTENSION=${1:?absolute immutable extension directory required}
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_RUNTIME="$GW_EXPERIMENT_ROOT/releases/b585198"
GW_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
GW_PILOT_ROOT="$GW_EXPERIMENT_ROOT/bmmc_scaling_20260928"
GW_OUTPUT="$GW_PILOT_ROOT/job_$SLURM_JOB_ID"
GW_DEPENDENCIES="$GW_PILOT_ROOT/dependencies_$SLURM_JOB_ID"
cd "$GW_EXTENSION"
sha256sum -c SHA256SUMS
# Read the scheduler deadline before dependency setup consumes allocation time.
GW_JOB_END_TIME=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')
export GW_PILOT_DEADLINE_EPOCH
GW_PILOT_DEADLINE_EPOCH=$(date -d "$GW_JOB_END_TIME" +%s)
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$GW_PILOT_ROOT/matplotlib_$SLURM_JOB_ID"
# An isolated data-reader dependency; the validated experiment venv stays intact.
mkdir -p "$GW_PILOT_ROOT"
"$GW_PYTHON" -m pip install --no-index --no-deps --target "$GW_DEPENDENCIES" "$GW_EXTENSION"/wheelhouse/h5py-*.whl
export PYTHONPATH="$GW_DEPENDENCIES"
export PYTHONDONTWRITEBYTECODE=1
"$GW_PYTHON" "$GW_EXTENSION/run_pilot.py" --root "$GW_EXPERIMENT_ROOT" --runtime "$GW_RUNTIME" --out "$GW_OUTPUT"
