#!/bin/bash
#SBATCH --job-name=gw_scot_reference
#SBATCH --account=fwllm
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --propagate=NONE
set -euo pipefail
GW_SOURCE=${1:?immutable source}
GW_ROOT=${2:?isolated experiment root}
cd "$GW_SOURCE"
sha256sum --status -c SHA256SUMS
(cd "$GW_ROOT/inputs" && sha256sum --status -c SHA256SUMS)
(cd "$GW_ROOT/wheelhouse" && sha256sum --status -c SHA256SUMS)
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
mkdir "$GW_ROOT/python_deps"
"$GW_PYTHON" -m pip install --no-index --no-deps --no-compile --disable-pip-version-check \
    --find-links "$GW_ROOT/wheelhouse" --target "$GW_ROOT/python_deps" \
    scikit-learn==1.5.2 joblib==1.4.2 threadpoolctl==3.5.0
export PYTHONPATH="$GW_ROOT/python_deps${PYTHONPATH:+:$PYTHONPATH}"
export SCOT_AUTHOR_SOURCE="$GW_ROOT/inputs/scotv1.py"
mkdir -p "$GW_ROOT/checks"
"$GW_PYTHON" -m pytest -q extensions/scot_reproduction_20261014 \
    --junitxml "$GW_ROOT/checks/scot_tests.xml"
"$GW_PYTHON" -u extensions/scot_reproduction_20261014/reference.py \
    --input-dir "$GW_ROOT/inputs" --out "$GW_ROOT/reference" --threads 1
