#!/usr/bin/env bash
#SBATCH --job-name=qpt_frames_msgd
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

campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_momentum_sgd_20260924_01
source_root=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast
source_record=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/preflight_retry1/selection.json
data=/gpfs/workdir/silvetian/frankwolfe/results/qpt/on_demand_profile_20260912_04/n8.npz
verifier=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/verify_saved_factors.py

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export JAX_ENABLE_X64=1
export PYTHONPATH="$source_root"

cd "$campaign"
sha256sum --check campaign_files.sha256

task=${SLURM_ARRAY_TASK_ID:?}
if ((task < 3)); then
  method=frames
  seed=$((task + 1))
else
  method=momentum_sgd
  seed=$((task - 2))
fi
run_dir="$campaign/results/${method}_seed${seed}"
mkdir -p "$run_dir"

python campaign_tools.py validate-inputs \
  --source "$source_root" --source-record "$source_record" --data "$data" \
  --baseline "$campaign/momentum_sgd_qpt.py" \
  --save "$run_dir/input_validation.json"

common=(
  --data "$data" --steps 200000 --tau 20 --batch-size 65536
  --chunk-steps 1 --metrics-every 1000 --metric-samples 512
  --metric-batch-size 32 --metric-seed 12345
  --initialization-seed 0 --sampling-seed "$seed"
  --rho-scale 0.6 --rho-offset 4 --rho-exponent 0.5
  --smoothing-scale 10000000 --smoothing-offset 1 --smoothing-exponent 0.25
)

if [[ "$method" == frames ]]; then
  python -m paper.experiments.quantum_process_tomography_structured_jax \
    "${common[@]}" \
    --step-scale 0.5 --step-offset 10 --step-exponent 0.75 \
    --measurement-backend product-state --device gpu --precision 64 --prefetch \
    --save "$run_dir/final.npz" > "$run_dir/run.log" 2>&1
else
  eta=$(python -c 'import json; print(json.load(open("selection.json"))["selected_eta_scale"])')
  python momentum_sgd_qpt.py \
    "${common[@]}" --eta-scale "$eta" --eta-offset 10 --eta-exponent 0.75 \
    --save "$run_dir/final.npz" > "$run_dir/run.log" 2>&1
fi

python "$verifier" --data "$data" --result "$run_dir/final.npz" \
  --save "$run_dir/verification.json" > "$run_dir/verification.log" 2>&1
sha256sum "$run_dir/final.npz" "$run_dir/verification.json" \
  > "$run_dir/checksums.sha256"

