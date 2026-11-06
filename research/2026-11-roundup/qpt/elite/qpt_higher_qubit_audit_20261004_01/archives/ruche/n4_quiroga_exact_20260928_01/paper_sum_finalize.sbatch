#!/usr/bin/env bash
#SBATCH --job-name=qpt_q_sum_final
#SBATCH --partition=cpu_short
#SBATCH --account=fwllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=paper_sum_retry2/finalize_job/slurm-%j.out
#SBATCH --error=paper_sum_retry2/finalize_job/slurm-%j.err
set -euo pipefail
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n4_quiroga_exact_20260928_01
root="$campaign/paper_sum_retry2"
module purge; module load anaconda3/2023.09-0/none-none; source activate qpt
cd "$campaign"; sha256sum --check campaign_files.sha256
python paper_sum_tools.py summarize --selection "$root/selection.json" --results "$root/results" --save "$root/summary.json"
sha256sum "$root/selection.json" "$root/summary.json" "$root"/results/*/final.npz "$root"/results/*/verification.json > "$root/final_checksums.sha256"
