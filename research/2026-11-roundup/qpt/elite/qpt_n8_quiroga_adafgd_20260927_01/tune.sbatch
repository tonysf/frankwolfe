#!/usr/bin/env bash
#SBATCH --job-name=qpt_quiroga_tune
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
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > tuning/gpu.txt
for wi in 0 1 2 3; do
  case "$wi" in 0) weight=1e-5;; 1) weight=1e-3;; 2) weight=1e-1;; 3) weight=1e1;; esac
  for si in 0 1 2 3; do
    case "$si" in 0) scale=1e2;; 1) scale=1e4;; 2) scale=1e6;; 3) scale=1e8;; esac
    run_dir="$campaign/tuning/w${wi}_s${si}"
    mkdir -p "$run_dir"
    if ! python adafgd_qpt.py --data "$data" --save "$run_dir/final.npz" --steps 2000 --metrics-every 200 --eta-scale "$scale" --tp-weight "$weight" --quiet > "$run_dir/run.log" 2>&1; then
      printf 'failed\n' > "$run_dir/FAILED"
    fi
  done
done
python campaign_tools.py select --root tuning --save selection.json
scale=$(python -c 'import json; print(json.load(open("selection.json"))["selected_eta_scale"])')
weight=$(python -c 'import json; print(json.load(open("selection.json"))["selected_tp_weight"])')
mkdir -p confirmation_seed0
python adafgd_qpt.py --data "$data" --save confirmation_seed0/final.npz --steps 20000 --metrics-every 1000 --eta-scale "$scale" --tp-weight "$weight" --quiet > confirmation_seed0/run.log 2>&1
python verify_factor.py --data "$data" --result confirmation_seed0/final.npz --save confirmation_seed0/verification.json > confirmation_seed0/verification.log 2>&1
python - <<'PY'
import json, numpy as np
with np.load("confirmation_seed0/final.npz", allow_pickle=False) as z:
    m=json.loads(str(z["metadata_json"].item()))
    f=float(z["process_fidelity_proxy"][-1]); tp=float(z["tp_violation"][-1])
assert m["optimizer"] == "quiroga_adafgd_fixed_tp" and np.isfinite(f) and np.isfinite(tp)
assert tp <= 1.0 and f >= 1e-5
print("seed-0 confirmation validated", f, tp)
PY
sha256sum confirmation_seed0/final.npz confirmation_seed0/verification.json > confirmation_seed0/checksums.sha256
