#!/usr/bin/env bash
#SBATCH --job-name=qpt_mifgd_p2
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
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_mifgd_penalty_tuning_20260927_01
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
for spec in w0p1_e0p01:0.1:0.01 w0p01_e0p1:0.01:0.1 w0p001_e1:0.001:1.0; do
  label=${spec%%:*}; rest=${spec#*:}; weight=${rest%%:*}; eta=${rest##*:}
  run_dir="$campaign/tuning/$label"; mkdir -p "$run_dir"
  if ! python mifgd_qpt.py --data "$data" --save "$run_dir/final.npz" --steps 20000 --metrics-every 1000 --eta "$eta" --mu 0.75 --tp-weight "$weight" > "$run_dir/run.log" 2>&1; then
    printf 'failed\n' > "$run_dir/FAILED"
  fi
done
python campaign_tools.py select --root tuning --save selection.json
eta=$(python -c 'import json; print(json.load(open("selection.json"))["selected_eta"])')
weight=$(python -c 'import json; print(json.load(open("selection.json"))["selected_tp_weight"])')
mkdir -p fgd_validation
python mifgd_qpt.py --data "$data" --save fgd_validation/final.npz --steps 20000 --metrics-every 1000 --eta "$eta" --mu 0 --tp-weight "$weight" > fgd_validation/run.log 2>&1
python - <<'PY'
import json, numpy as np
with np.load("fgd_validation/final.npz",allow_pickle=False) as z:
    m=json.loads(str(z["metadata_json"].item())); f=float(z["process_fidelity_proxy"][-1]); tp=float(z["tp_violation"][-1])
assert m["optimizer"] == "fgd_fixed_tp" and np.isfinite(f) and np.isfinite(tp) and tp <= 1.0
print("FGD control validated",f,tp)
PY
