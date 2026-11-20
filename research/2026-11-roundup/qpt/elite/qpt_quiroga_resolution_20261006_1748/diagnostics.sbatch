#!/bin/bash
#SBATCH --job-name=qpt_resolution
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --array=0-1%1
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_resolution_c7a8c9789b78_20261006_1748/diagnostics-%A_%a.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_resolution_c7a8c9789b78_20261006_1748/diagnostics-%A_%a.err

set -euo pipefail
umask 077
began_epoch=$(date +%s)
commit=c7a8c9789b7814ede1c95e53eef13bfe1dd58c63
repo="/gpfs/workdir/silvetian/frankwolfe-quiroga-${commit}"
output_root=/gpfs/workdir/silvetian/qpt_quiroga_resolution_c7a8c9789b78_20261006_1748
staged="${output_root}/stage"
campaign_out="${output_root}/diagnostics-${SLURM_ARRAY_JOB_ID}"
out="${campaign_out}/channel-${SLURM_ARRAY_TASK_ID}"
deadline_epoch="${SLURM_JOB_END_TIME:-$((began_epoch + 1200))}"

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${repo}"

cd "${repo}"
test "$(git rev-parse HEAD)" = "${commit}"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "${staged}/run_diagnostics.py"
test -f "${staged}/diagnostic_plan.json"
mkdir -p "${campaign_out}"
test ! -e "${out}"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "scientific_role=exploratory_calibration_diagnostic_no_validation_reranking"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID}"
echo "source_commit=${commit}"
echo "output_directory=${out}"
sha256sum "${staged}/diagnostic_plan.json" "${staged}/run_diagnostics.py" "${staged}/diagnostics.sbatch"
python -c "import sys,numpy,scipy; print('python='+sys.version.replace(chr(10),' ')); print('numpy='+numpy.__version__); print('scipy='+scipy.__version__)"
if python "${staged}/run_diagnostics.py" \
    --plan "${staged}/diagnostic_plan.json" \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    --out "${out}" \
    --deadline-epoch "${deadline_epoch}"; then
  diagnostic_exit=0
else
  diagnostic_exit=$?
fi
test ! -f "${out}/inventory.json" || sha256sum "${out}/inventory.json"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "scheduler_exit_code=${diagnostic_exit}"
exit "${diagnostic_exit}"
