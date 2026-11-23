#!/usr/bin/env bash
#SBATCH --job-name=qpt_q_sum_cmp
#SBATCH --partition=gpua100
#SBATCH --account=fwllm
#SBATCH --nodelist=ruche-gpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=paper_sum_retry2/comparison_job/slurm-%A_%a.out
#SBATCH --error=paper_sum_retry2/comparison_job/slurm-%A_%a.err
set -euo pipefail
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n4_quiroga_exact_20260928_01
root="$campaign/paper_sum_retry2"
source_root=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast
record=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/preflight_retry1/selection.json
data=/gpfs/workdir/silvetian/frankwolfe/results/qpt/scaling_ybuXlx/n4.npz
module purge; module load anaconda3/2023.09-0/none-none; source activate qpt
export JAX_ENABLE_X64=1 PYTHONPATH="$source_root:$campaign"
cd "$campaign"; sha256sum --check campaign_files.sha256
python validate_inputs.py --source "$source_root" --record "$record" --data "$data"
task=${SLURM_ARRAY_TASK_ID:?}
if ((task < 3)); then method=quiroga; seed=$((task+1)); else method=frames; seed=$((task-2)); fi
dir="$root/results/${method}_seed${seed}"; mkdir -p "$dir"
if [[ "$method" == quiroga ]]; then
  scale=$(python -c 'import json; print(json.load(open("paper_sum_retry2/selection.json"))["selected_eta_scale"])')
  weight=$(python -c 'import json; print(json.load(open("paper_sum_retry2/selection.json"))["selected_tp_weight"])')
  python exact_adafgd.py --data "$data" --save "$dir/final.npz" --steps 10000 --metrics-every 100 --subset-seed "$seed" --eta-scale "$scale" --tp-weight "$weight" --quiet > "$dir/run.log" 2>&1
else
  python frames_subset.py --data "$data" --save "$dir/final.npz" --steps 10000 --subset-seed "$seed" --beta0 100 --quiet > "$dir/run.log" 2>&1
fi
python verify_factor.py --data "$data" --result "$dir/final.npz" --save "$dir/verification.json"
sha256sum "$dir/final.npz" "$dir/verification.json" > "$dir/checksums.sha256"
