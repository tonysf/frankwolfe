#!/usr/bin/env bash
#SBATCH --job-name=qpt_quiroga_smoke
#SBATCH --partition=gpua100
#SBATCH --account=fwllm
#SBATCH --nodelist=ruche-gpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=smoke_retry1/slurm-%j.out
#SBATCH --error=smoke_retry1/slurm-%j.err
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
python adafgd_qpt.py --data "$data" --save smoke_retry1/final.npz --steps 2 --metrics-every 1 --eta-scale 1e2 --tp-weight 1e-3 --quiet > smoke_retry1/run.log 2>&1
python verify_factor.py --data "$data" --result smoke_retry1/final.npz --save smoke_retry1/verification.json
