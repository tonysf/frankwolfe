#!/usr/bin/env bash
#SBATCH --job-name=qpt_quiroga_cmp
#SBATCH --partition=gpua100
#SBATCH --account=fwllm
#SBATCH --nodelist=ruche-gpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=comparison_job/slurm-%A_%a.out
#SBATCH --error=comparison_job/slurm-%A_%a.err
set -euo pipefail
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_quiroga_adafgd_20260927_01
source_root=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast
record=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/preflight_retry1/selection.json
data=/gpfs/workdir/silvetian/frankwolfe/results/qpt/on_demand_profile_20260912_04/n8.npz
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export JAX_ENABLE_X64=1
export PYTHONPATH="$source_root"
cd "$campaign"
sha256sum --check campaign_files.sha256
python validate_inputs.py --source "$source_root" --record "$record" --data "$data"
seed=$((SLURM_ARRAY_TASK_ID + 1))
scale=$(python -c 'import json; print(json.load(open("selection.json"))["selected_eta_scale"])')
weight=$(python -c 'import json; print(json.load(open("selection.json"))["selected_tp_weight"])')
run_dir="$campaign/results/quiroga_seed${seed}"
mkdir -p "$run_dir"
python adafgd_qpt.py --data "$data" --save "$run_dir/final.npz" --steps 200000 --metrics-every 1000 --sampling-seed "$seed" --eta-scale "$scale" --tp-weight "$weight" --quiet > "$run_dir/run.log" 2>&1
python verify_factor.py --data "$data" --result "$run_dir/final.npz" --save "$run_dir/verification.json" > "$run_dir/verification.log" 2>&1
sha256sum "$run_dir/final.npz" "$run_dir/verification.json" > "$run_dir/checksums.sha256"
