#!/usr/bin/env bash
#SBATCH --job-name=qpt_msgd_tune
#SBATCH --partition=gpua100
#SBATCH --account=fwllm
#SBATCH --nodelist=ruche-gpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=tuning_job/slurm-%j.out
#SBATCH --error=tuning_job/slurm-%j.err

set -euo pipefail

campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_momentum_sgd_20260924_01
source_root=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast
source_record=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/preflight_retry1/selection.json
data=/gpfs/workdir/silvetian/frankwolfe/results/qpt/on_demand_profile_20260912_04/n8.npz

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export JAX_ENABLE_X64=1
export PYTHONPATH="$source_root"

cd "$campaign"
sha256sum --check campaign_files.sha256
python campaign_tools.py validate-inputs \
  --source "$source_root" --source-record "$source_record" --data "$data" \
  --baseline "$campaign/momentum_sgd_qpt.py" \
  --save "$campaign/tuning/input_validation.json"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader \
  > "$campaign/tuning/gpu.txt"
python -m py_compile momentum_sgd_qpt.py campaign_tools.py

mkdir -p tuning/smoke
python momentum_sgd_qpt.py \
  --data "$data" --save tuning/smoke/final.npz \
  --steps 2 --metrics-every 1 --eta-scale 1.0 \
  > tuning/smoke/run.log 2>&1

for eta in 0.1 0.3 1.0 3.0; do
  label=${eta/./p}
  run_dir="$campaign/tuning/eta_$label"
  mkdir -p "$run_dir"
  python momentum_sgd_qpt.py \
    --data "$data" --save "$run_dir/final.npz" \
    --steps 20000 --metrics-every 1000 --eta-scale "$eta" \
    > "$run_dir/run.log" 2>&1
done

python campaign_tools.py select \
  --tuning-dir "$campaign/tuning" --steps 20000 \
  --save "$campaign/selection.json"

