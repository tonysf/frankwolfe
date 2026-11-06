#!/bin/bash
#SBATCH --job-name=qpt_structured
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

set -eo pipefail
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt

set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the frankwolfe directory}"

export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

qpt_h5="${SLURM_SUBMIT_DIR}/../QPT_BFW/jax_arr_gt_xi_0.05.h5"
qpt_run_dir="${SLURM_SUBMIT_DIR}/results/qpt/structured_${SLURM_JOB_ID}"

[[ -f "$qpt_h5" ]] || { echo "Missing input: $qpt_h5" >&2; exit 1; }
mkdir -p results/qpt
mkdir "$qpt_run_dir"

# Convert inside the allocation; preserve the original noisy observations.
python -m paper.experiments.qpt_structured_data \
    --h5 "$qpt_h5" \
    --save "$qpt_run_dir/data.npz" \
    --verification full --chunk-size 32

python -m paper.experiments.quantum_process_tomography_structured_jax \
    --data "$qpt_run_dir/data.npz" \
    --device gpu --precision 64 \
    --steps 10000 --rank 1 --tau 10 --batch-size 32 \
    --chunk-steps 100 --metrics-every 100 \
    --metric-mode sampled --metric-samples 512 --metric-batch-size 32 \
    --initialization-seed 0 --sampling-seed 0 --metric-seed 12345 \
    --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 \
    --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 \
    --step-scale 2 --step-offset 2 --step-exponent 1 \
    --save "$qpt_run_dir/result.npz" \
    --plot "$qpt_run_dir/result.png"
