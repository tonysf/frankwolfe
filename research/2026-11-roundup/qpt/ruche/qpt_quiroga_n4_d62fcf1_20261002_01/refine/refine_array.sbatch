#!/bin/bash
#SBATCH --job-name=qpt_q_n4_refine
#SBATCH --partition=cpu_med
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-15%8
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/refine/slurm-%A_%a.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/refine/slurm-%A_%a.err

set -euo pipefail
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export PYTHONPATH=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

repo=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
out=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/refine
cd "${repo}"
test "$(git rev-parse HEAD)" = d62fcf192f27d8fb207ce61a1c466fc5059a7e1f

etas=(0.0001 0.0003 0.001 0.003 0.01 0.03)
eta_tags=(1e-4 3e-4 1e-3 3e-3 1e-2 3e-2)
betas=(1000 10000)
lambdas=(4.096 0.4096)
task=${SLURM_ARRAY_TASK_ID}
strict=()
extra=()

if (( task < 12 )); then
  beta_index=$((task / 6))
  eta_index=$((task % 6))
  method=printed-adafgd
  steps=400
  metrics_every=20
  eta=${etas[eta_index]}
  name=ada_beta${betas[beta_index]}_eta${eta_tags[eta_index]}
  extra=(--spectral-norm-method dense)
elif (( task < 14 )); then
  beta_index=$((task - 12))
  method=armijo-fgd
  steps=400
  metrics_every=20
  eta=0.001
  name=armijo_beta${betas[beta_index]}
  strict=(--fail-on-method-failure)
else
  beta_index=$((task - 14))
  method=frames
  steps=400
  metrics_every=20
  eta=0.001
  name=frames_beta${betas[beta_index]}
  extra=(--frames-steps 102400 --frames-metrics-every 5120 --batch-size 32)
  strict=(--fail-on-method-failure)
fi

lambda=${lambdas[beta_index]}
echo "exploratory_refinement=1 task=${task} method=${method} beta=${betas[beta_index]} lambda=${lambda} eta=${eta} commit=$(git rev-parse HEAD)"
python -m paper.experiments.qpt_quiroga_campaign \
  --n-qubits 4 --methods "${method}" --steps "${steps}" \
  --metrics-every "${metrics_every}" --channel-seed 0 \
  --initialization-seed 0 --sampling-seed 0 --eta-scale "${eta}" \
  --tp-weight "${lambda}" --save "${out}/${name}.npz" --quiet \
  "${extra[@]}" "${strict[@]}"
echo "completed task=${task} artifact=${out}/${name}.npz"
