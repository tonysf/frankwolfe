#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PYTHON_COMMAND="${PYTHON_BIN:-python3}"
readonly REQUESTED_VARIANT="${JAX_CUDA_VARIANT:-auto}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

case "${REQUESTED_VARIANT}" in
    auto|cuda12|cuda13)
        ;;
    *)
        fail "JAX_CUDA_VARIANT must be auto, cuda12, or cuda13 (got '${REQUESTED_VARIANT}')."
        ;;
esac

command -v "${PYTHON_COMMAND}" >/dev/null 2>&1 || \
    fail "Python executable '${PYTHON_COMMAND}' was not found. Set PYTHON_BIN to a valid Python executable."

python_version="$("${PYTHON_COMMAND}" -c 'import platform; print(platform.python_version())')" || \
    fail "Could not query the Python version from '${PYTHON_COMMAND}'."
if ! "${PYTHON_COMMAND}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
    fail "The current JAX GPU wheels require Python 3.12 or newer; '${PYTHON_COMMAND}' is Python ${python_version}. Create a Python 3.12+ environment or set PYTHON_BIN."
fi

cuda_variant="${REQUESTED_VARIANT}"
driver_version="unknown"

if [[ "${REQUESTED_VARIANT}" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || \
        fail "JAX_CUDA_VARIANT=auto requires nvidia-smi. Install the NVIDIA driver, or explicitly set JAX_CUDA_VARIANT=cuda12/cuda13."

    driver_output="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1)" || \
        fail "nvidia-smi could not query the NVIDIA driver: ${driver_output}"
    driver_version="${driver_output%%$'\n'*}"
    driver_version="${driver_version//[[:space:]]/}"
    driver_major="${driver_version%%.*}"

    [[ "${driver_major}" =~ ^[0-9]+$ ]] || \
        fail "Could not parse NVIDIA driver version '${driver_version}' from nvidia-smi."

    if (( driver_major >= 580 )); then
        cuda_variant="cuda13"
    elif (( driver_major >= 525 )); then
        cuda_variant="cuda12"
    else
        fail "NVIDIA driver ${driver_version} is too old for the supported JAX CUDA wheels. Upgrade to driver 525+ (CUDA 12) or 580+ (CUDA 13)."
    fi
elif command -v nvidia-smi >/dev/null 2>&1; then
    driver_output="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${driver_output}" ]]; then
        driver_version="${driver_output%%$'\n'*}"
        driver_version="${driver_version//[[:space:]]/}"
        driver_major="${driver_version%%.*}"
        if [[ "${driver_major}" =~ ^[0-9]+$ ]]; then
            if [[ "${cuda_variant}" == "cuda13" ]] && (( driver_major < 580 )); then
                fail "JAX_CUDA_VARIANT=cuda13 requires NVIDIA driver 580+; detected ${driver_version}."
            fi
            if [[ "${cuda_variant}" == "cuda12" ]] && (( driver_major < 525 )); then
                fail "JAX_CUDA_VARIANT=cuda12 requires NVIDIA driver 525+; detected ${driver_version}."
            fi
        fi
    fi
fi

printf 'Repository: %s\n' "${REPOSITORY_ROOT}"
printf 'Python:     %s (%s)\n' "${PYTHON_COMMAND}" "${python_version}"
printf 'Driver:     %s\n' "${driver_version}"
printf 'JAX wheel:  jax[%s]\n' "${cuda_variant}"

# The JAX accelerator extra selects the matching official CUDA plugin and
# jaxlib dependency.  Do not pin jaxlib separately: JAX owns that compatibility
# constraint.
"${PYTHON_COMMAND}" -m pip install --upgrade \
    -e "${REPOSITORY_ROOT}[qpt-jax]" \
    "jax[${cuda_variant}]"

printf '\nRunning the GPU preflight...\n'
"${PYTHON_COMMAND}" "${SCRIPT_DIR}/check_qpt_jax_environment.py" "$@" --device gpu

printf '\nQPT JAX GPU environment is ready.\n'
