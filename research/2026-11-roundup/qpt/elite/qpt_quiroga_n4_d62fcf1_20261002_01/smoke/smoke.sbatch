#!/bin/bash
#SBATCH --job-name=qpt_q_n4_smoke
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/smoke/slurm-%j.out
#SBATCH --error=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/smoke/slurm-%j.err

set -euo pipefail
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export PYTHONPATH=/gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /gpfs/workdir/silvetian/frankwolfe-quiroga-d62fcf1
out=/gpfs/workdir/silvetian/qpt_quiroga_n4_d62fcf1_20261002_01/smoke

python -m paper.experiments.qpt_quiroga_campaign \
  --n-qubits 4 --methods armijo-fgd --steps 5 --metrics-every 1 \
  --channel-seed 0 --initialization-seed 0 --eta-scale 0.001 \
  --tp-weight 40.96 --save "${out}/armijo_beta100.npz" --quiet \
  --fail-on-method-failure

python -m paper.experiments.qpt_quiroga_campaign \
  --n-qubits 4 --methods printed-adafgd --steps 2 --metrics-every 1 \
  --channel-seed 0 --initialization-seed 0 --eta-scale 0.001 \
  --tp-weight 40.96 --spectral-norm-method dense \
  --save "${out}/ada_beta100_eta1e-3.npz" --quiet

python -m paper.experiments.qpt_quiroga_campaign \
  --n-qubits 4 --methods frames --steps 5 --frames-steps 100 \
  --metrics-every 1 --frames-metrics-every 20 --channel-seed 0 \
  --initialization-seed 0 --sampling-seed 0 --eta-scale 0.001 \
  --tp-weight 40.96 --batch-size 32 \
  --save "${out}/frames_beta100.npz" --quiet --fail-on-method-failure
