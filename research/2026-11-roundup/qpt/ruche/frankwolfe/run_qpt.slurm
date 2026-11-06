#!/bin/bash
#SBATCH --job-name=qpt_jax
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
cd ${SLURM_SUBMIT_DIR}
QPT_H5=${SLURM_SUBMIT_DIR}/../QPT_BFW/jax_arr_gt_xi_0.05.h5

bash scripts/run_qpt_jax_gpu.sh "$QPT_H5" --steps 10000 --rank 1 --tau 10 --batch-size 32 --initialization-seed 0 --sampling-seed 0 --rho-scale 2 --rho-offset 4 --rho-exponent 0.6 --smoothing-scale 10 --smoothing-offset 1 --smoothing-exponent 0.25 --step-scale 2 --step-offset 2 --step-exponent 1 --metrics-every 100 --save results/qpt/qpt_stochastic_frames_jax.npz --plot results/qpt/qpt_stochastic_frames_jax.png --quiet
