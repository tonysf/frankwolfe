#!/usr/bin/env bash
#SBATCH --job-name=qpt_mifgd_tune
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
#SBATCH --output=job/slurm-%j.out
#SBATCH --error=job/slurm-%j.err

set -euo pipefail

campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_mifgd_20260925_01
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

python - <<'PY'
import hashlib, json
from pathlib import Path
source = Path("/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast")
record = json.load(open("/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/preflight_retry1/selection.json"))
for relative, expected in record["source_sha256"].items():
    actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"source mismatch: {relative}")
data = Path("/gpfs/workdir/silvetian/frankwolfe/results/qpt/on_demand_profile_20260912_04/n8.npz")
if hashlib.sha256(data.read_bytes()).hexdigest() != "c05b02bdbc1b8955a89ae0cff2da77a986c76f96d2971860f3d2f7d0b925e439":
    raise RuntimeError("data mismatch")
print("frozen source and data validated")
PY

mkdir -p tuning/smoke
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader > tuning/gpu.txt
python mifgd_qpt.py --data "$data" --save tuning/smoke/final.npz \
  --steps 2 --metrics-every 1 --eta 0.001 --mu 0.75 --tp-weight 1.0 \
  > tuning/smoke/run.log 2>&1

for pair in 1e-6:1em06 1e-5:1em05 1e-4:1em04 1e-3:1em03; do
  eta=${pair%%:*}
  label=${pair##*:}
  run_dir="$campaign/tuning/eta_$label"
  mkdir -p "$run_dir"
  if ! python mifgd_qpt.py --data "$data" --save "$run_dir/final.npz" \
    --steps 20000 --metrics-every 1000 --eta "$eta" --mu 0.75 --tp-weight 1.0 \
    > "$run_dir/run.log" 2>&1; then
    printf 'candidate failed: eta=%s\n' "$eta" > "$run_dir/FAILED"
  fi
done

python select_tuning.py --root tuning --save selection.json

