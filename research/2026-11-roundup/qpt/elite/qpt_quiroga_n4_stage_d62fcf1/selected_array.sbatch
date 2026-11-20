#!/bin/bash
#SBATCH --job-name=qpt_q_n4_selected
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/selected/slurm-%A_%a.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/selected/slurm-%A_%a.err

set -euo pipefail
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export PYTHONPATH=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

repo=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
out=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/selected
cd "${repo}"
test "$(git rev-parse HEAD)" = d62fcf192f27d8fb207ce61a1c466fc5059a7e1f

task=${SLURM_ARRAY_TASK_ID}
strict=()
extra=()
case "${task}" in
  0)
    method=printed-adafgd
    steps=2000
    metrics_every=100
    eta=0.003
    name=ada_beta100_eta3e-3_extended2000
    extra=(--spectral-norm-method dense)
    ;;
  1)
    method=armijo-fgd
    steps=2000
    metrics_every=100
    eta=0.001
    name=armijo_beta100_extended2000
    strict=(--fail-on-method-failure)
    ;;
  2)
    method=frames
    steps=400
    metrics_every=20
    eta=0.001
    name=frames_beta100_extended2000epochs
    extra=(--frames-steps 512000 --frames-metrics-every 25600 --batch-size 32)
    strict=(--fail-on-method-failure)
    ;;
  *)
    echo "unexpected array task ${task}" >&2
    exit 64
    ;;
esac

echo "task=${task} method=${method} lambda=40.96 eta=${eta} commit=$(git rev-parse HEAD)"
python -m paper.experiments.qpt_quiroga_campaign \
  --n-qubits 4 --methods "${method}" --steps "${steps}" \
  --metrics-every "${metrics_every}" --channel-seed 0 \
  --initialization-seed 0 --sampling-seed 0 --eta-scale "${eta}" \
  --tp-weight 40.96 --save "${out}/${name}.npz" --quiet \
  "${extra[@]}" "${strict[@]}"
echo "completed task=${task} artifact=${out}/${name}.npz"
