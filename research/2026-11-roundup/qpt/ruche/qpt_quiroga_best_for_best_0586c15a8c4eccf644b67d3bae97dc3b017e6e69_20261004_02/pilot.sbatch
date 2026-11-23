#!/bin/bash
#SBATCH --job-name=qpt_q_bfb_pilot
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_0586c15a8c4eccf644b67d3bae97dc3b017e6e69_20261004_02/pilot-%j.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_0586c15a8c4eccf644b67d3bae97dc3b017e6e69_20261004_02/pilot-%j.err

set -euo pipefail
umask 077

commit=0586c15a8c4eccf644b67d3bae97dc3b017e6e69
repo="/gpfs/workdir/silvetian/frankwolfe-quiroga-${commit}"
output_root="/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_${commit}_20261004_02"
out="${output_root}/calibration-pilot-${SLURM_JOB_ID}"

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

test -d "${output_root}"
test -e "${repo}/.git"
mkdir "${out}"
cd "${repo}"
test "$(git rev-parse HEAD)" = "${commit}"
test -z "$(git status --porcelain --untracked-files=all)"
export PYTHONPATH="${repo}"

echo "status=started"
echo "scientific_role=representative_anchor_pilot_not_selected_comparison"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_account=${SLURM_JOB_ACCOUNT:-fwllm}"
echo "commit=${commit}"
echo "output_directory=${out}"
echo "python=$(command -v python)"
python -c "import numpy, scipy, sys; print('python_version='+sys.version.replace(chr(10),' ')); print('numpy_version='+numpy.__version__); print('scipy_version='+scipy.__version__)"

# Preserve the runner's nonzero status without exiting before archiving evidence.
# Exit 2 is expected when a method fails after writing its JSON and NPZ files.
if python -m paper.experiments.qpt_quiroga_direct_comparison \
  --n-qubits 2 \
  --row-epochs 2000 \
  --checkpoint-epochs 100 \
  --printed-tp-weight-per-row 0.05 \
  --frames-initial-tp-weight-per-row 0.005 \
  --frames-penalty-exponent 0.25 \
  --eta-scale 0.0001 \
  --rank 1 \
  --tau 2.0 \
  --subset-size 0 \
  --subset-seed 400000 \
  --channel-seed 0 \
  --initialization-seed 200000 \
  --sampling-seed 300000 \
  --observation-mode noiseless \
  --noise-seed 0 \
  --shot-seed 0 \
  --batch-size 32 \
  --spectral-norm-method dense \
  --rho-scale 2.0 \
  --rho-offset 4.0 \
  --rho-exponent 0.6 \
  --step-scale 2.0 \
  --step-offset 2.0 \
  --step-exponent 1.0 \
  --gate-name noiseless_recovery_v1 \
  --gate-fidelity-min 0.99 \
  --gate-rmse-max 0.000001 \
  --gate-normalized-tp-max 0.001 \
  --cohort-id n2-full-calibration-pilot-q005-e00001-f0005-0586c15 \
  --trial-id "n2-full-calibration-pilot-c0-i0-s0-20261005-${SLURM_JOB_ID}" \
  --split calibration \
  --expected-commit "${commit}" \
  --require-clean \
  --save "${out}/pair.npz" \
  --quiet \
  --fail-on-method-failure; then
  method_exit_code=0
else
  method_exit_code=$?
fi

# This is an execution record, not a scientific validator. The runner's method
# statuses and physical recovery gates must also be checked before proceeding.
if python - "${out}" "${method_exit_code}" "${commit}" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

out = Path(sys.argv[1])
runner_exit_code = int(sys.argv[2])
artifacts = {}
for name in ("pair.json", "pair.npz"):
    path = out / name
    if path.is_file() and path.stat().st_size > 0:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[name] = {"present_nonempty": True, "sha256": digest,
                           "bytes": path.stat().st_size}
        print(f"{digest}  {path}", flush=True)
    else:
        artifacts[name] = {"present_nonempty": False, "sha256": None, "bytes": 0}
complete_archive = all(item["present_nonempty"] for item in artifacts.values())
if runner_exit_code == 0 and complete_archive:
    status = "runner_completed_archive_present_gate_not_asserted"
elif runner_exit_code == 2 and complete_archive:
    status = "method_failure_archive_preserved"
else:
    status = "runner_failure_or_incomplete_archive"
final_exit_code = runner_exit_code if runner_exit_code != 0 else (0 if complete_archive else 3)
record = {
    "schema_version": 1,
    "scientific_role": "representative_anchor_pilot_not_selected_comparison",
    "status": status,
    "runner_exit_code": runner_exit_code,
    "intended_scheduler_exit_code": final_exit_code,
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_commit": sys.argv[3],
    "hostname": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_account": os.environ.get("SLURM_JOB_ACCOUNT"),
    "output_directory": str(out),
    "artifacts": artifacts,
}
with (out / "execution_status.json").open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
raise SystemExit(0 if complete_archive else 3)
PY
then
  archive_exit_code=0
else
  archive_exit_code=$?
fi

final_exit_code="${method_exit_code}"
if [ "${final_exit_code}" -eq 0 ] && [ "${archive_exit_code}" -ne 0 ]; then
  final_exit_code="${archive_exit_code}"
fi
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "runner_exit_code=${method_exit_code}"
echo "archive_exit_code=${archive_exit_code}"
echo "scheduler_exit_code=${final_exit_code}"
exit "${final_exit_code}"
