#!/bin/bash
#SBATCH --job-name=gw_a100_validate
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
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/a100-%j.out
set -euo pipefail
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
GW_EXPERIMENT_ROOT=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
GW_PYTHON="$GW_EXPERIMENT_ROOT/.venv/bin/python"
GW_OUTPUT="$GW_EXPERIMENT_ROOT/a100_diagnostics/$SLURM_JOB_ID"
mkdir -p "$GW_OUTPUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$GW_OUTPUT/matplotlib"
# sbatch --chdir selects an immutable source release, recorded below.
pwd > "$GW_OUTPUT/source_directory.txt"
git rev-parse HEAD > "$GW_OUTPUT/source_commit.txt"
"$GW_PYTHON" -m pytest -q --junitxml="$GW_OUTPUT/pytest.xml"
"$GW_PYTHON" capture_environment.py --device cuda --out "$GW_OUTPUT/environment.json"
for n in 16 128; do
  if [[ "$n" == 16 ]]; then instance=toy16_s1000; else instance=metric128_s1000; fi
  "$GW_PYTHON" validate_gpu.py --instance "$GW_EXPERIMENT_ROOT/data/$instance.npz" --out "$GW_OUTPUT/precision_n$n"
done
"$GW_PYTHON" -m pip freeze > "$GW_OUTPUT/environment.lock.txt"
for n in 128 512; do
  "$GW_PYTHON" oracle_diagnostics.py --instance "$GW_EXPERIMENT_ROOT/data/metric${n}_s1000.npz" --device cuda --dtype float32 --repetitions 100 --out "$GW_OUTPUT" --label "oracle_a100_n${n}_s1000"
  "$GW_PYTHON" run_smoothing.py --instance "$GW_EXPERIMENT_ROOT/data/metric${n}_s1000.npz" --oracle exact --steps 200 --device cuda --dtype float32 --seed 0 --mu-scaled 1 --out "$GW_OUTPUT/smoothing/n${n}_exact"
  for batch in 8 32 128; do
    "$GW_PYTHON" run_smoothing.py --instance "$GW_EXPERIMENT_ROOT/data/metric${n}_s1000.npz" --oracle sketch --batch "$batch" --steps 200 --device cuda --dtype float32 --seed 0 --mu-scaled 1 --out "$GW_OUTPUT/smoothing/n${n}_sketch_q${batch}"
  done
done
printf 'completed\n' > "$GW_OUTPUT/COMPLETED"
