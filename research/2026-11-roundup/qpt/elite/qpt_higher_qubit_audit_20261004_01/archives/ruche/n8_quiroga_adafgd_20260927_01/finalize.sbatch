#!/usr/bin/env bash
#SBATCH --job-name=qpt_quiroga_sum
#SBATCH --partition=cpu_short
#SBATCH --account=fwllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=finalize_job/slurm-%j.out
#SBATCH --error=finalize_job/slurm-%j.err
set -euo pipefail
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_quiroga_adafgd_20260927_01
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
cd "$campaign"
sha256sum --check campaign_files.sha256
python campaign_tools.py summarize --selection selection.json --results results --save summary.json
sha256sum selection.json summary.json confirmation_seed0/final.npz results/*/final.npz results/*/verification.json > final_checksums.sha256
