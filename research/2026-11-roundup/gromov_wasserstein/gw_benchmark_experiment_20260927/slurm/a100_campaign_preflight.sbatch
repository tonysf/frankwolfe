#!/bin/bash
#SBATCH --job-name=gw_campaign_check
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/campaign-check-%j.out
set -euo pipefail
GW_SOURCE_DIRECTORY=${1:?absolute immutable source directory required}
GW_SOURCE_COMMIT=${2:?expected full source commit required}
cd "$GW_SOURCE_DIRECTORY"
test "$(git rev-parse HEAD)" = "$GW_SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
GW_OUTPUT="$GW_EXPERIMENT_ROOT/a100_campaign/preflight_$SLURM_JOB_ID"
mkdir -p "$GW_OUTPUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$GW_OUTPUT/matplotlib"
"$GW_PYTHON" capture_environment.py --device cuda --out "$GW_OUTPUT/environment.json"
"$GW_PYTHON" -m pytest -q --junitxml="$GW_OUTPUT/pytest.xml"
"$GW_PYTHON" validate_projection_gpu.py --out "$GW_OUTPUT/projection_gpu_validation"
for initialization in product randomized; do
  for method in smoothing-fw smoothing-row-projection full-kl stochastic-kl stochastic-kl-momentum fw pg; do
    "$GW_PYTHON" run_benchmark.py --instance "$GW_EXPERIMENT_ROOT/data/toy16_s1000.npz" \
      --method "$method" --device cuda --dtype float32 --threads 1 --batch 128 \
      --initialization "$initialization" --eta 0.00390625 --eta-power 0.25 \
      --inner-tol 1e-6 --inner-max 1000 --budget 0.1 --checkpoints 0.03,0.1 \
      --seed 100 --init-seed 900100 --rho0 0.3 --out "$GW_OUTPUT/$initialization/$method"
  done
done
"$GW_PYTHON" run_native_sampled.py --instance "$GW_EXPERIMENT_ROOT/data/toy16_s1000.npz" \
  --device cuda --dtype float32 --threads 1 --batch 128 --epsilon 0.1 \
  --budget 0.1 --checkpoints 0.03,0.1 --seed 100 --init-seed 900100 --out "$GW_OUTPUT/native-product"
# Validate the new momentum-KL path at both campaign dimensions, using development data only.
for instance in metric512_noisy_s1000 bmmc1000; do
  for initialization in product randomized; do
    "$GW_PYTHON" run_benchmark.py --instance "$GW_EXPERIMENT_ROOT/data/$instance.npz" \
      --method stochastic-kl-momentum --device cuda --dtype float32 --threads 1 --batch 128 \
      --initialization "$initialization" --rho0 0.3 --eta 0.1 --eta-power 0 \
      --inner-tol 1e-6 --inner-max 1000 --budget 0.2 --checkpoints 0.05,0.2 \
      --seed 100 --init-seed 900100 --out "$GW_OUTPUT/momentum-campaign-size/$instance/$initialization"
  done
done
"$GW_PYTHON" validate_panel_dispatch_gpu.py --experiment-root "$GW_EXPERIMENT_ROOT" \
  --out "$GW_OUTPUT/dispatcher_validation" --source-commit "$GW_SOURCE_COMMIT"
printf 'completed\n' > "$GW_OUTPUT/COMPLETED"
