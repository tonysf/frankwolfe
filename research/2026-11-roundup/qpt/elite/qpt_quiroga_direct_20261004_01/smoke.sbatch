#!/bin/bash
#SBATCH --job-name=qpt_q_direct_smoke
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_direct_3bf36c4_20261004_01/smoke/slurm-%j.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_direct_3bf36c4_20261004_01/smoke/slurm-%j.err

set -euo pipefail
umask 077

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

repo=/gpfs/workdir/silvetian/frankwolfe-quiroga-3bf36c4
out=/gpfs/workdir/silvetian/qpt_quiroga_direct_3bf36c4_20261004_01/smoke
commit=3bf36c4f47ac2c637df9fe3e1d9db0d8173484e2

test -e "${repo}/.git"
mkdir -p "${out}"
cd "${repo}"
test "$(git rev-parse HEAD)" = "${commit}"
test -z "$(git status --porcelain --untracked-files=all)"
export PYTHONPATH="${repo}"

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
  --tp-weight-per-row 0.005 \
  --eta-scale 0.003 \
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
  --cohort-id n2-full-smoke-v1 \
  --trial-id n2-full-c0-i0-s0-smoke-3bf36c4 \
  --split calibration \
  --expected-commit "${commit}" \
  --require-clean \
  --save "${out}/pair.npz" \
  --quiet \
  --fail-on-method-failure

sha256sum "${out}/pair.npz" "${out}/pair.json"
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
