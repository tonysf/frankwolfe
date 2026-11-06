#!/bin/bash
#SBATCH --job-name=gw_a100_pilot
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/practical-%j.out
set -euo pipefail
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
GW_OUTPUT="$GW_EXPERIMENT_ROOT/a100_practical/$SLURM_JOB_ID"
mkdir -p "$GW_OUTPUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$GW_OUTPUT/matplotlib"
pwd > "$GW_OUTPUT/source_directory.txt"
git rev-parse HEAD > "$GW_OUTPUT/source_commit.txt"
"$GW_PYTHON" capture_environment.py --device cuda --out "$GW_OUTPUT/environment.json"
"$GW_PYTHON" -m pytest -q --junitxml="$GW_OUTPUT/pytest.xml"
"$GW_PYTHON" validate_native_gpu.py --instance "$GW_EXPERIMENT_ROOT/data/toy16_s1000.npz" --out "$GW_OUTPUT/native_gpu_validation"
"$GW_PYTHON" -m pip freeze > "$GW_OUTPUT/environment.lock.txt"
for method in smoothing-fw full-kl stochastic-kl fw; do
  "$GW_PYTHON" run_benchmark.py --instance "$GW_EXPERIMENT_ROOT/data/toy16_s1000.npz" --method "$method" --device cuda --dtype float32 --threads 1 --batch 128 --inner-tol 1e-6 --inner-max 1000 --budget 0.1 --checkpoints 0.03,0.1 --seed 0 --init-seed 2026 --out "$GW_OUTPUT/runner_gpu_smoke/$method"
done
"$GW_PYTHON" run_native_sampled.py --instance "$GW_EXPERIMENT_ROOT/data/toy16_s1000.npz" --device cuda --dtype float32 --batch 128 --epsilon 0.1 --budget 0.1 --checkpoints 0.03,0.1 --seed 0 --init-seed 2026 --out "$GW_OUTPUT/runner_gpu_smoke/native"

# All six methods get the same 30-second target budget on each frozen development instance.
for instance in metric512_s1001 metric512_noisy_s1001 bmmc1000; do
  for method in smoothing-fw smoothing-fw-exact full-kl stochastic-kl fw native-sampled-gw; do
    if [[ "$method" == native-sampled-gw ]]; then
      "$GW_PYTHON" run_native_sampled.py --instance "$GW_EXPERIMENT_ROOT/data/$instance.npz" --device cuda --dtype float32 --batch 128 --epsilon 0.1 --budget 30 --checkpoints 1,3,10,30 --seed 0 --init-seed 2026 --out "$GW_OUTPUT/$instance/$method"
    else
      implementation="$method"
      oracle=sketch
      eta=1
      if [[ "$method" == smoothing-fw-exact ]]; then implementation=smoothing-fw; oracle=exact; fi
      if [[ "$method" == full-kl ]]; then eta=10; fi
      "$GW_PYTHON" run_benchmark.py --instance "$GW_EXPERIMENT_ROOT/data/$instance.npz" --method "$implementation" --device cuda --dtype float32 --threads 1 --oracle "$oracle" --oracle-selection-status initial_development_pilot_q128_frozen --batch 128 --mu-scale 1 --eta "$eta" --inner-tol 1e-6 --inner-max 1000 --budget 30 --checkpoints 1,3,10,30 --seed 0 --init-seed 2026 --out "$GW_OUTPUT/$instance/$method"
    fi
  done
done
printf 'completed\n' > "$GW_OUTPUT/COMPLETED"
