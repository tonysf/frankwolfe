#!/usr/bin/env python3
"""Preflight a JAX device and, optionally, a QPT_BFW HDF5 data file."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    # Make the checkout importable even when this script is invoked by its
    # absolute path before/without an editable installation.
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _format_bytes(byte_count: int) -> str:
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    raise AssertionError("unreachable")


def _device_description(device) -> str:
    platform_name = getattr(device, "platform", "unknown")
    device_kind = getattr(device, "device_kind", type(device).__name__)
    device_id = getattr(device, "id", "?")
    return f"{platform_name}:{device_id} ({device_kind})"


def _load_jax():
    try:
        import jax
        import jaxlib
        import jax.numpy as jnp
    except ImportError as error:
        raise RuntimeError(
            "JAX is not installed in this Python environment. Run "
            "scripts/setup_qpt_jax_gpu.sh, or install the project's "
            "qpt-jax extra and the official JAX CUDA extra."
        ) from error
    return jax, jaxlib, jnp


def _select_device(jax, requested_platform: str, device_index: int):
    from paper.experiments.quantum_process_tomography_jax import (
        select_jax_device,
    )

    try:
        return select_jax_device(requested_platform, device_index)
    except (RuntimeError, ValueError) as error:
        if requested_platform == "gpu":
            raise RuntimeError(
                f"GPU preflight failed: {error} Verify that nvidia-smi works "
                "and that the JAX CUDA extra was installed into this exact "
                f"Python interpreter ({sys.executable})."
            ) from error
        raise


def _run_complex128_kernel(jax, jnp, selected_device):
    try:
        jax.config.update("jax_enable_x64", True)
    except RuntimeError as error:
        raise RuntimeError(
            "Could not enable JAX 64-bit mode. Run this preflight in a fresh "
            "Python process before creating JAX arrays."
        ) from error

    with jax.default_device(selected_device):
        matrix = jnp.asarray(
            [[1.0 + 2.0j, 3.0 - 1.0j], [0.5j, -2.0 + 4.0j]],
            dtype=jnp.complex128,
        )

        @jax.jit
        def kernel(value):
            gram = value.conj().T @ value
            return jnp.real(jnp.trace(gram))

        result = kernel(matrix)
        result.block_until_ready()

    if matrix.dtype != jnp.dtype(jnp.complex128):
        raise RuntimeError(
            "JAX truncated complex128 to a lower precision. Ensure 64-bit "
            "mode is enabled and rerun in a fresh process."
        )
    if not bool(jnp.isfinite(result)):
        raise RuntimeError("The complex128 JIT smoke kernel returned a non-finite value.")
    return float(result)


def _run_qpt_smoke(jax, selected_device):
    """Compile and execute the actual stochastic-FRAMES scan kernel."""

    import numpy as np

    from paper.experiments.quantum_process_tomography import QPTData
    from paper.experiments.quantum_process_tomography_jax import (
        run_qpt_stochastic_frames_jax,
    )

    data = QPTData(
        f_vector=np.asarray([0.25, 0.5]),
        D_tensors=np.asarray([[[1.0]], [[0.5]]], dtype=np.complex128),
        A_basis=np.asarray([[[1.0]]], dtype=np.complex128),
        chi_star=np.asarray([[1.0]], dtype=np.complex128),
    )
    platform_devices = list(jax.devices(selected_device.platform))
    try:
        platform_index = platform_devices.index(selected_device)
    except ValueError as error:
        raise RuntimeError(
            "The selected device disappeared before the QPT smoke test."
        ) from error

    result = run_qpt_stochastic_frames_jax(
        data,
        n_steps=2,
        rank=1,
        tau=2.0,
        batch_size=1,
        x0=np.asarray([0.5, 0.25]),
        rho_schedule=[0.4, 0.6].__getitem__,
        smoothing_schedule=[1.0, 0.8].__getitem__,
        step_size_schedule=[0.2, 0.1].__getitem__,
        metrics_frequency=0,
        device=selected_device.platform,
        device_index=platform_index,
        precision="64",
        execution_mode="scan",
        warmup=False,
        show_progress=False,
    )
    if result.device_platform != selected_device.platform:
        raise RuntimeError(
            "The QPT smoke test executed on an unexpected JAX platform: "
            f"{result.device_platform}."
        )
    if not np.all(np.isfinite(result.final_x)):
        raise RuntimeError("The QPT smoke test produced a non-finite iterate.")
    return result


def _validate_hdf5(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"QPT HDF5 file does not exist or is not a file: {path}")

    try:
        from paper.experiments.quantum_process_tomography import QPTData

        data = QPTData.from_hdf5(path)
    except (ImportError, KeyError, OSError, ValueError) as error:
        raise RuntimeError(f"Could not load QPT HDF5 data from {path}: {error}") from error

    tensors = (
        ("f_vector", data.f_vector),
        ("D_tensors", data.D_tensors),
        ("A_basis", data.A_basis),
        ("B_tensors", data.B_tensors),
        ("chi_star", data.chi_star),
    )
    total_bytes = 0
    print(f"QPT data:    {path.resolve()}")
    print(
        "QPT shape:   "
        f"m={data.m}, d={data.d}, process_dimension={data.process_dimension}"
    )
    for name, tensor in tensors:
        if tensor is None:
            print(f"  {name:<11} absent")
            continue
        tensor_bytes = int(tensor.nbytes)
        total_bytes += tensor_bytes
        print(
            f"  {name:<11} shape={str(tensor.shape):<20} "
            f"dtype={str(tensor.dtype):<12} host={_format_bytes(tensor_bytes)}"
        )
    print(f"Tensor host bytes: {total_bytes} ({_format_bytes(total_bytes)})")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verify JAX device availability with a synchronized complex128 "
            "JIT kernel and optionally validate a QPT_BFW HDF5 file."
        )
    )
    parser.add_argument(
        "--device",
        choices=("gpu", "cpu", "auto"),
        default="gpu",
        help="required JAX platform (default: gpu; no silent CPU fallback)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="zero-based device index within the selected platform",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        help="optional QPT_BFW HDF5 file to load and validate",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.device_index < 0:
        raise RuntimeError("--device-index must be nonnegative.")

    jax, jaxlib, jnp = _load_jax()
    print(f"Repository:  {REPOSITORY_ROOT}")
    print(f"Python:      {platform.python_version()} ({sys.executable})")
    print(f"JAX:         {jax.__version__}")
    print(f"jaxlib:      {jaxlib.__version__}")

    try:
        default_devices = list(jax.devices())
    except RuntimeError as error:
        raise RuntimeError(f"JAX could not initialize a backend: {error}") from error
    print(
        "Devices:     "
        + (", ".join(_device_description(device) for device in default_devices) or "none")
    )

    selected_device = _select_device(jax, args.device, args.device_index)
    print(f"Selected:    {_device_description(selected_device)}")
    kernel_result = _run_complex128_kernel(jax, jnp, selected_device)
    print(f"JAX x64:     {bool(jax.config.jax_enable_x64)}")
    print(f"JIT result:  {kernel_result:.6g} (complex128, synchronized)")

    qpt_result = _run_qpt_smoke(jax, selected_device)
    print(
        "QPT smoke:   PASS "
        f"({qpt_result.mode}, {qpt_result.total_seconds:.6g}s synchronized)"
    )

    if args.h5 is not None:
        _validate_hdf5(args.h5)

    print("Preflight:   PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
