#!/bin/bash
#SBATCH --job-name=qpt_q_n4_hold_cmp
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --array=0-5%6
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/heldout_compare/slurm-%A_%a.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/heldout_compare/slurm-%A_%a.err

set -euo pipefail
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export PYTHONPATH=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

repo=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
out=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/heldout_compare
cd "${repo}"
test "$(git rev-parse HEAD)" = d62fcf192f27d8fb207ce61a1c466fc5059a7e1f

task=${SLURM_ARRAY_TASK_ID}
strict=()
extra=()
if (( task < 3 )); then
  seed=$((task + 1))
  method=printed-adafgd
  steps=2000
  metrics_every=100
  eta=0.003
  lambda=40.96
  name=seed${seed}_ada_beta100_eta3e-3_extended2000
  extra=(--spectral-norm-method dense)
else
  seed=$((task - 2))
  method=frames
  steps=400
  metrics_every=20
  eta=0.001
  lambda=4.096
  name=seed${seed}_frames_beta1000_extended2000epochs
  extra=(--frames-steps 512000 --frames-metrics-every 25600 --batch-size 32)
  strict=(--fail-on-method-failure)
fi

echo "heldout=1 task=${task} seed=${seed} method=${method} lambda=${lambda} eta=${eta} commit=$(git rev-parse HEAD)"
python -m paper.experiments.qpt_quiroga_campaign \
  --n-qubits 4 --methods "${method}" --steps "${steps}" \
  --metrics-every "${metrics_every}" --channel-seed "${seed}" \
  --initialization-seed "${seed}" --sampling-seed "${seed}" \
  --eta-scale "${eta}" --tp-weight "${lambda}" \
  --save "${out}/${name}.npz" --quiet "${extra[@]}" "${strict[@]}"
echo "completed heldout task=${task} seed=${seed} method=${method}"
