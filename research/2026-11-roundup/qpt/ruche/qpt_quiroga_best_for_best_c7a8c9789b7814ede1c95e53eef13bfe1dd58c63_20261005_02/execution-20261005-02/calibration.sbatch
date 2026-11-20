#!/bin/bash
#SBATCH --job-name=qpt_q_bfb_cal
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%4
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_c7a8c9789b7814ede1c95e53eef13bfe1dd58c63_20261005_02/calibration-%A_%a.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_c7a8c9789b7814ede1c95e53eef13bfe1dd58c63_20261005_02/calibration-%A_%a.err

set -euo pipefail
umask 077
launcher_started_epoch=$(date +%s)
commit=c7a8c9789b7814ede1c95e53eef13bfe1dd58c63
repo="/gpfs/workdir/silvetian/frankwolfe-quiroga-${commit}"
output_root="/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_${commit}_20261005_02"
staged="${output_root}/execution-20261005-02"
campaign_out="${output_root}/calibration-${SLURM_ARRAY_JOB_ID}"
out="${campaign_out}/channel-${SLURM_ARRAY_TASK_ID}"
deadline_epoch="${SLURM_JOB_END_TIME:-$((launcher_started_epoch + 1800))}"

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${repo}"

test -d "${output_root}"
test -f "${staged}/calibration_plan.json"
test -f "${staged}/run_calibration.py"
cd "${repo}"
test "$(git rev-parse HEAD)" = "${commit}"
test -z "$(git status --porcelain --untracked-files=all)"
mkdir -p "${campaign_out}"
mkdir "${out}"

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "scientific_role=independent_calibration_not_selected_comparison"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "deadline_epoch=${deadline_epoch}"
echo "commit=${commit}"
echo "output_directory=${out}"
sha256sum "${staged}/calibration_plan.json" "${staged}/run_calibration.py" "${staged}/calibration.sbatch"
python -c "import numpy, scipy, sys; print('python_version='+sys.version.replace(chr(10),' ')); print('numpy_version='+numpy.__version__); print('scipy_version='+scipy.__version__)"

if python "${staged}/run_calibration.py" \
  --plan "${staged}/calibration_plan.json" \
  --channel "${SLURM_ARRAY_TASK_ID}" \
  --out "${out}" \
  --deadline-epoch "${deadline_epoch}"; then
  calibration_exit_code=0
else
  calibration_exit_code=$?
fi
test ! -f "${out}/inventory.json" || sha256sum "${out}/inventory.json"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "scheduler_exit_code=${calibration_exit_code}"
exit "${calibration_exit_code}"
