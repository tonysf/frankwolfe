#!/bin/bash
#SBATCH --job-name=qpt_q_n4_long_fgd
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-1%2
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/long_armijo/slurm-%A_%a.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/long_armijo/slurm-%A_%a.err

set -euo pipefail
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export PYTHONPATH=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

repo=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
out=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/long_armijo
cd "${repo}"
test "$(git rev-parse HEAD)" = d62fcf192f27d8fb207ce61a1c466fc5059a7e1f

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    beta=1000
    lambda=4.096
    ;;
  1)
    beta=10000
    lambda=0.4096
    ;;
  *)
    echo "unexpected array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 64
    ;;
esac

echo "exploratory_long_run=1 beta=${beta} lambda=${lambda} commit=$(git rev-parse HEAD)"
python -m paper.experiments.qpt_quiroga_campaign \
  --n-qubits 4 --methods armijo-fgd --steps 20000 --metrics-every 1000 \
  --channel-seed 0 --initialization-seed 0 --eta-scale 0.001 \
  --tp-weight "${lambda}" --save "${out}/armijo_beta${beta}_extended20000.npz" \
  --quiet --fail-on-method-failure
echo "completed beta=${beta} artifact=${out}/armijo_beta${beta}_extended20000.npz"
