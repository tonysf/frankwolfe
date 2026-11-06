#!/usr/bin/env bash
#SBATCH --job-name=qpt_n4_exact_tune
#SBATCH --partition=gpua100
#SBATCH --account=fwllm
#SBATCH --nodelist=ruche-gpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=tuning_job/slurm-%j.out
#SBATCH --error=tuning_job/slurm-%j.err
set -euo pipefail
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n4_quiroga_exact_20260928_01
source_root=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast
record=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/preflight_retry1/selection.json
data=/gpfs/workdir/silvetian/frankwolfe/results/qpt/scaling_ybuXlx/n4.npz
module purge; module load anaconda3/2023.09-0/none-none; source activate qpt
export JAX_ENABLE_X64=1 PYTHONPATH="$source_root:$campaign"
cd "$campaign"; sha256sum --check campaign_files.sha256
python validate_inputs.py --source "$source_root" --record "$record" --data "$data"
for ci in 0 1 2; do
  case "$ci" in 0) scale=0.1;; 1) scale=1;; 2) scale=10;; esac
  for wi in 0 1 2; do
    case "$wi" in 0) weight=0.01;; 1) weight=0.1;; 2) weight=1;; esac
    dir="tuning/quiroga/c${ci}_w${wi}"; mkdir -p "$dir"
    if ! python exact_adafgd.py --data "$data" --save "$dir/final.npz" --steps 100 --metrics-every 10 --eta-scale "$scale" --tp-weight "$weight" --quiet > "$dir/run.log" 2>&1; then printf 'failed\n' > "$dir/FAILED"; fi
  done
done
for beta in 10 100 1000 10000; do
  dir="tuning/frames/beta${beta}"; mkdir -p "$dir"
  python frames_subset.py --data "$data" --save "$dir/final.npz" --steps 10000 --beta0 "$beta" --quiet > "$dir/run.log" 2>&1
done
python campaign_tools.py select --root tuning --save selection.json
scale=$(python -c 'import json; print(json.load(open("selection.json"))["selected_eta_scale"])')
weight=$(python -c 'import json; print(json.load(open("selection.json"))["selected_tp_weight"])')
mkdir -p confirmation_seed0
python exact_adafgd.py --data "$data" --save confirmation_seed0/final.npz --steps 2000 --metrics-every 100 --eta-scale "$scale" --tp-weight "$weight" --quiet > confirmation_seed0/run.log 2>&1
python verify_factor.py --data "$data" --result confirmation_seed0/final.npz --save confirmation_seed0/verification.json
python - <<'PY'
import numpy as np
with np.load('confirmation_seed0/final.npz',allow_pickle=False) as z:
 f=float(z['process_fidelity_proxy'][-1]); tp=float(z['tp_violation'][-1])
assert np.isfinite(f) and np.isfinite(tp) and tp <= 1.0 and f >= 0.1, (f,tp)
print('confirmation gate passed',f,tp)
PY
