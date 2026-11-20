#!/bin/bash
#SBATCH --job-name=qpt_gen
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --partition=cpu_med
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE

module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
cd ${SLURM_SUBMIT_DIR}
export JAX_ENABLE_X64=1
python3 gen_data.py
