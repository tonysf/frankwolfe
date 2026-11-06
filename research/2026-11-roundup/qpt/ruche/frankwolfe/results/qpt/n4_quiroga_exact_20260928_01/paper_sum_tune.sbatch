#!/usr/bin/env bash
#SBATCH --job-name=qpt_q_sum_tune
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
#SBATCH --output=paper_sum_retry2/tuning_job/slurm-%j.out
#SBATCH --error=paper_sum_retry2/tuning_job/slurm-%j.err
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
python audit_formulation.py --data "$data" --save "$root/formulation_audit.json"
for si in 0 1 2 3 4; do
  case "$si" in 0) scale=1e-8;; 1) scale=1e-7;; 2) scale=1e-6;; 3) scale=1e-5;; 4) scale=1e-4;; esac
  for wi in 0 1 2 3; do
    case "$wi" in 0) weight=0.1;; 1) weight=1;; 2) weight=10;; 3) weight=100;; esac
    dir="$root/tuning/s${si}_w${wi}"; mkdir -p "$dir"
    if ! python exact_adafgd.py --data "$data" --save "$dir/final.npz" --steps 500 --metrics-every 50 --eta-scale "$scale" --tp-weight "$weight" --quiet > "$dir/run.log" 2>&1; then printf 'failed\n' > "$dir/FAILED"; fi
  done
done
python paper_sum_tools.py select --root "$root/tuning" --save "$root/selection.json"
scale=$(python -c 'import json; print(json.load(open("paper_sum_retry2/selection.json"))["selected_eta_scale"])')
weight=$(python -c 'import json; print(json.load(open("paper_sum_retry2/selection.json"))["selected_tp_weight"])')
mkdir -p "$root/confirmation_seed0"
python exact_adafgd.py --data "$data" --save "$root/confirmation_seed0/final.npz" --steps 5000 --metrics-every 100 --eta-scale "$scale" --tp-weight "$weight" --quiet > "$root/confirmation_seed0/run.log" 2>&1
python verify_factor.py --data "$data" --result "$root/confirmation_seed0/final.npz" --save "$root/confirmation_seed0/verification.json"
python - <<'PY'
import numpy as np
with np.load('paper_sum_retry2/confirmation_seed0/final.npz',allow_pickle=False) as z:
 f=float(z['process_fidelity_proxy'][-1]); tp=float(z['tp_violation'][-1])
assert np.isfinite(f) and np.isfinite(tp) and tp <= 1.0 and f >= 0.1, (f,tp)
print('paper-sum confirmation gate passed',f,tp)
PY
