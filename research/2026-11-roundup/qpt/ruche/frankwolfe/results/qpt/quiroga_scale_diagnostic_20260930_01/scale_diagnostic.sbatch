#!/usr/bin/env bash
#SBATCH --job-name=qpt_q_scale_diag
#SBATCH --partition=gpua100
#SBATCH --account=fwllm
#SBATCH --nodelist=ruche-gpu15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
campaign=/gpfs/workdir/silvetian/frankwolfe/results/qpt/quiroga_scale_diagnostic_20260930_01
source_root=/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/source_fast
module purge
module load anaconda3/2023.09-0/none-none
source activate qpt
export JAX_ENABLE_X64=1 PYTHONPATH="$source_root:$campaign"
cd "$campaign"
sha256sum --check campaign_files.sha256
mkdir -p data results
for n in 2 3; do
  for noise_label in noise0 noise05; do
    if [[ "$noise_label" == noise0 ]]; then noise=0; else noise=0.05; fi
    data_file="data/n${n}_${noise_label}.npz"
    if [[ ! -f "$data_file" ]]; then
      python -m paper.experiments.qpt_generate_data --n-qubits "$n" --noise-std "$noise" --channel-seed 0 --noise-seed 0 --save "$data_file"
    fi
  done
done
scales=(1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1)
weights=(0.1 1 10 100)
for n in 2 3; do
  if [[ "$n" == 2 ]]; then subset=128; else subset=1024; fi
  for noise_label in noise0 noise05; do
    for initialization in trace_normalized raw_uniform; do
      case_name="n${n}_${noise_label}_${initialization}"
      for si in "${!scales[@]}"; do
        for wi in "${!weights[@]}"; do
          dir="results/${case_name}/s${si}_w${wi}"
          mkdir -p "$dir"
          if ! python exact_adafgd.py --data "data/n${n}_${noise_label}.npz" --save "$dir/final.npz" --steps 1000 --metrics-every 100 --subset-size "$subset" --subset-seed 0 --initialization "$initialization" --initialization-seed 0 --eta-scale "${scales[$si]}" --tp-weight "${weights[$wi]}" --quiet >"$dir/run.log" 2>&1; then
            printf 'failed\n' >"$dir/FAILED"
          fi
        done
      done
    done
  done
done
python summarize.py
sha256sum summary.json results/*/*/final.npz > result_checksums.sha256
