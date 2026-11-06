#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_qpt_jax_gpu.sh HDF5_PATH [QPT_OPTIONS...]

Run stochastic FRAMES with JAX on a GPU. HDF5_PATH must name an existing
QPT_BFW data file. Additional options are forwarded to the experiment, so
schedule flags such as --rho-scale, --smoothing-scale, and --step-scale work
normally.

Environment:
  PYTHON_BIN       Python executable to use (default: python3)
  JAX_ENABLE_X64   JAX 64-bit setting (default: 1)

The launcher always enforces --device gpu, --precision 64, and
--execution-mode scan.
EOF
}

if (($# < 1)); then
    usage
    exit 64
fi

data_path=$1
shift

if [[ ! -f "$data_path" ]]; then
    printf 'error: QPT HDF5 file does not exist: %s\n' "$data_path" >&2
    exit 66
fi

python_bin=${PYTHON_BIN:-python3}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'error: Python executable not found: %s\n' "$python_bin" >&2
    exit 127
fi

export JAX_ENABLE_X64=${JAX_ENABLE_X64:-1}

exec "$python_bin" \
    -m paper.experiments.quantum_process_tomography_jax \
    --h5 "$data_path" \
    "$@" \
    --device gpu \
    --precision 64 \
    --execution-mode scan
