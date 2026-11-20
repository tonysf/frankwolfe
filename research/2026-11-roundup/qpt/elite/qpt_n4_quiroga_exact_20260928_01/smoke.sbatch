#!/usr/bin/env bash
#SBATCH --job-name=qpt_n4_exact_smoke
#SBATCH --partition=gpua100
#SBATCH --account=fwllm
#SBATCH --nodelist=ruche-gpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=smoke/slurm-%j.out
#SBATCH --error=smoke/slurm-%j.err
set -euo pipefail
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n4_quiroga_exact_20260928_01
source_root=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast
record=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/preflight_retry1/selection.json
data=/gpfs/workdir/silvetian/frankwolfe/results/qpt/scaling_ybuXlx/n4.npz
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export JAX_ENABLE_X64=1
export PYTHONPATH="$source_root:$campaign"
cd "$campaign"
sha256sum --check campaign_files.sha256
python validate_inputs.py --source "$source_root" --record "$record" --data "$data"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > smoke/gpu.txt
python exact_adafgd.py --data "$data" --save smoke/quiroga.npz --steps 2 --metrics-every 1 --eta-scale 1 --tp-weight 0.1 > smoke/quiroga.log 2>&1
python verify_factor.py --data "$data" --result smoke/quiroga.npz --save smoke/quiroga_verification.json
python frames_subset.py --data "$data" --save smoke/frames.npz --steps 2 --beta0 100 --quiet > smoke/frames.log 2>&1
python verify_factor.py --data "$data" --result smoke/frames.npz --save smoke/frames_verification.json
sha256sum smoke/quiroga.npz smoke/frames.npz smoke/*verification.json > smoke/checksums.sha256
