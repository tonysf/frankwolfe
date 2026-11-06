#!/bin/bash
#SBATCH --job-name=qpt_q_bfb_smoke
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_0586c15a8c4eccf644b67d3bae97dc3b017e6e69_20261004_02/slurm-%j.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_0586c15a8c4eccf644b67d3bae97dc3b017e6e69_20261004_02/slurm-%j.err

set -euo pipefail
umask 077

commit=0586c15a8c4eccf644b67d3bae97dc3b017e6e69

repo="/gpfs/workdir/silvetian/frankwolfe-quiroga-${commit}"
output_root="/gpfs/workdir/silvetian/qpt_quiroga_best_for_best_${commit}_20261004_02"
out="${output_root}/smoke-${SLURM_JOB_ID}"

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
echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "commit=${commit}"
echo "python=$(command -v python)"
python -c "import numpy, scipy, sys; print('python_version='+sys.version.replace(chr(10),' ')); print('numpy_version='+numpy.__version__); print('scipy_version='+scipy.__version__)"

python -m paper.experiments.qpt_quiroga_direct_comparison \
  --n-qubits 2 \
  --row-epochs 2 \
  --checkpoint-epochs 1 \
  --printed-tp-weight-per-row 0.05 \
  --frames-initial-tp-weight-per-row 0.005 \
  --frames-penalty-exponent 0.25 \
  --eta-scale 0.0001 \
  --rank 1 \
  --tau 2.0 \
  --subset-size 0 \
  --subset-seed 0 \
  --channel-seed 0 \
  --initialization-seed 0 \
  --sampling-seed 0 \
  --observation-mode noiseless \
  --noise-seed 0 \
  --shot-seed 0 \
  --batch-size 32 \
  --spectral-norm-method dense \
  --gate-name noiseless_recovery_v1 \
  --gate-fidelity-min 0.99 \
  --gate-rmse-max 0.000001 \
  --gate-normalized-tp-max 0.001 \
  --cohort-id n2-full-best-for-best-smoke-0586c15 \
  --trial-id n2-full-c0-i0-s0-smoke-0586c15 \
  --split calibration \
  --expected-commit "${commit}" \
  --require-clean \
  --save "${out}/pair.npz" \
  --quiet \
  --fail-on-method-failure

test -s "${out}/pair.npz"
test -s "${out}/pair.json"
sha256sum "${out}/pair.json" "${out}/pair.npz"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "status=complete"
