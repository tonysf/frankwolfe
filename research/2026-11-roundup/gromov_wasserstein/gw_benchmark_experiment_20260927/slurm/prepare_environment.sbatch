#!/bin/bash
#SBATCH --job-name=gw_env_prep
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/prepare-%j.out
set -euo pipefail
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
cd /gpfs/workdir/silvetian/gw_benchmark_experiment_20260927
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -e .venv/pyvenv.cfg ]]; then
  python -m venv --system-site-packages .venv
fi
# Resume only this experiment's environment after a recorded dependency failure.
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements-ruche.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch, numpy, scipy, ot; print(torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__, ot.__version__)'
.venv/bin/python -m pip freeze > environment.ruche.lock.txt
