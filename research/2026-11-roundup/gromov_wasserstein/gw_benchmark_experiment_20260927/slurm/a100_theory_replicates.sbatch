#!/bin/bash
#SBATCH --job-name=gw_theory_repeats
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/theory-%j.out
set -euo pipefail
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
GW_OUTPUT="$GW_EXPERIMENT_ROOT/a100_theory/$SLURM_JOB_ID"
mkdir -p "$GW_OUTPUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$GW_PYTHON" capture_environment.py --device cuda --out "$GW_OUTPUT/environment.json"
git rev-parse HEAD > "$GW_OUTPUT/source_commit.txt"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  "$GW_PYTHON" run_smoothing.py --instance "$GW_EXPERIMENT_ROOT/data/metric128_s1000.npz" --oracle sketch --batch 128 --steps 1000 --device cuda --dtype float32 --seed "$seed" --mu-scaled 1 --out "$GW_OUTPUT/theory_replicates/seed$seed"
done
printf 'completed\n' > "$GW_OUTPUT/COMPLETED"
