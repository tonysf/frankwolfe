#!/bin/bash
#SBATCH --job-name=qpt_quiroga_tune
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
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_tune-%A_%a.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_tune-%A_%a.err
set -euo pipefail
umask 077
began_epoch=$(date +%s)
commit=${1:?Full pinned commit required}
output_root=${2:?Fresh staged output root required}
plan_sha256=${3:?Frozen plan SHA256 required}
[[ ${commit} =~ ^[0-9a-f]{40}$ && ${plan_sha256} =~ ^[0-9a-f]{64}$ ]]
[[ ${output_root} == /gpfs/workdir/silvetian/qpt_quiroga_tuning_* ]]
repo="/gpfs/workdir/silvetian/frankwolfe-quiroga-${commit}"
plan="${output_root}/stage/tuning_plan.json"
campaign_out="${output_root}/tuning-${SLURM_ARRAY_JOB_ID}"
out="${campaign_out}/task-${SLURM_ARRAY_TASK_ID}"
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
mkdir -p "${campaign_out}"
test ! -e "${out}"
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) hostname=$(hostname) source_commit=${commit} output=${out}"
python -c "import sys,numpy,scipy; print('python='+sys.version.replace(chr(10),' ')); print('numpy='+numpy.__version__); print('scipy='+scipy.__version__)"
selection_args=()
if test -f "${output_root}/stage/calibration_audit.json"; then
  selection_args=(--calibration-audit "${output_root}/stage/calibration_audit.json")
fi
if python "${repo}/scripts/qpt_quiroga_tuning/run_tuning.py" --plan "${plan}" --plan-sha256 "${plan_sha256}" \
    --array-index "${SLURM_ARRAY_TASK_ID}" --out "${out}" --deadline-epoch "${deadline_epoch}" "${selection_args[@]}"; then
  tuning_exit=0
else
  tuning_exit=$?
fi
test ! -f "${out}/inventory.json" || sha256sum "${out}/inventory.json"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) scheduler_exit_code=${tuning_exit}"
exit "${tuning_exit}"
