"""Memory-bounded structured QPT on one JAX device.

The measurement bank and Pauli transforms replace dense D/B tensors. Only
smooth measurement gradients enter momentum; the TP Moreau term stays exact.
Scans return scalar gaps, not an iterate history. Metrics use a fixed sample
(or an explicitly requested full streaming pass) outside optimizer timing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .quantum_process_tomography import (
    PowerSchedule, checkpoint_iterations, make_factor_initial_point, unpack_factor,
)
from .quantum_process_tomography_jax import (
    _require_jax, _configure_jax_precision, _materialize_schedules, select_jax_device,
)
from .qpt_structured_data import StructuredQPTData
from .qpt_structured_operators import (
    measurement_loss_and_gradient, measurement_values,
    trace_preserving_loss_and_gradient, rank_one_measurement_vectors,
    rank_one_measurement_values, rank_one_measurement_loss_and_gradient,
)


@dataclass
class StructuredQPTResult:
    metadata: dict
    final_factor: np.ndarray
    gradient_estimate: np.ndarray
    checkpoint_steps: np.ndarray
    checkpoint_factors: np.ndarray
    checkpoint_smoothing_parameters: np.ndarray
    optimizer_seconds: np.ndarray
    measurement_loss: np.ndarray
    tp_violation: np.ndarray
    smoothed_objective: np.ndarray
    smoothed_gap: np.ndarray
    process_fidelity_proxy: np.ndarray
    estimated_gaps: np.ndarray
    momentum_weights: np.ndarray
    smoothing_parameters: np.ndarray
    step_sizes: np.ndarray
    metric_symbols: np.ndarray

    @property
    def final_x(self):
        """Packed-real compatibility view, materialized only on request."""
        return np.concatenate((self.final_factor.real.ravel(), self.final_factor.imag.ravel()))

    @property
    def final_gradient_estimate(self):
        return np.concatenate((self.gradient_estimate.real.ravel(), self.gradient_estimate.imag.ravel()))

    @property
    def total_seconds(self):
        return self.metadata["optimizer_seconds"]


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _lmo(jnp, gradient, tau):
    norm = jnp.linalg.norm(gradient)
    if gradient.shape[1] == 1:
        atom = -tau * gradient / jnp.maximum(norm, jnp.finfo(gradient.real.dtype).tiny)
    else:
        left, _, right = jnp.linalg.svd(gradient, full_matrices=False)
        atom = -tau * (left @ right)
    return jnp.where(norm > 0, atom, jnp.zeros_like(gradient))


def _executable_memory_estimate(executable):
    """Optional backend accounting; unavailable is not zero memory usage."""
    try:
        analysis = executable.memory_analysis()
    except (AttributeError, NotImplementedError):
        return None
    if analysis is None:
        return None
    return int(sum(getattr(analysis, name, 0) for name in (
        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
    )) - getattr(analysis, "alias_size_in_bytes", 0))


def _build_scan(jax, jnp, tau, noiseless, rank_one=False):
    values_fn = rank_one_measurement_values if rank_one else measurement_values
    gradient_fn = rank_one_measurement_loss_and_gradient if rank_one else measurement_loss_and_gradient

    def scan(state, start, symbols, observations, rho, beta, gamma, bank, basis, truth):
        def step(carry, inputs):
            factor, previous = carry
            iteration, batch, targets, weight, smoothing, step_size = inputs
            if noiseless:
                targets = values_fn(truth, batch, bank, xp=jnp)
            _, sampled = gradient_fn(factor, batch, targets, bank, xp=jnp)
            estimate = jax.lax.cond(
                iteration == 0, lambda: sampled,
                lambda: (1.0 - weight) * previous + weight * sampled,
            )
            _, tp_gradient = trace_preserving_loss_and_gradient(factor, basis, xp=jnp)
            moreau = tp_gradient / smoothing
            combined = estimate + moreau
            atom = _lmo(jnp, combined, tau)
            gap = jnp.vdot(combined, factor - atom).real
            next_factor = (1.0 - step_size) * factor + step_size * atom
            # No factor history in scan outputs: memory is independent of T*N*r.
            return (next_factor, estimate), gap

        iterations = start + jnp.arange(symbols.shape[0], dtype=jnp.int32)
        return jax.lax.scan(step, state, (iterations, symbols, observations, rho, beta, gamma))
    return jax.jit(scan)


def run_qpt_structured_jax(
    data, *, n_steps=1000, rank=1, tau=10.0, batch_size=32,
    initialization_seed=0, sampling_seed=0, x0=None, beta0=10.0,
    rho_schedule=None, smoothing_schedule=None, step_size_schedule=None,
    chunk_steps=100, metrics_frequency=100, metric_mode="sampled",
    metric_samples=512, metric_batch_size=32, metric_seed=12345,
    store_checkpoints=False, device="auto", device_index=0, precision="64",
    warmup=False, show_progress=True, measurement_backend="auto",
):
    """Run structured FRAMES with host-resident observations and bounded scans.

    ``metric_mode='sampled'`` reuses one independent fixed sample at every
    checkpoint. Its gap is an empirical diagnostic, NOT an exact full-loss FW
    gap. ``full`` streams all stored measurements through bounded metric batches.
    TP residuals and gradients remain exact in either mode. A terminal metric
    uses beta[T-1], matching the dense runner. Compilation is always separated
    from synchronized optimizer time; optional warmup runs only one chunk.
    """
    if not isinstance(data, StructuredQPTData):
        raise TypeError("data must be StructuredQPTData.")
    n_steps = _positive_integer(n_steps, "n_steps")
    rank = _positive_integer(rank, "rank")
    batch_size = _positive_integer(batch_size, "batch_size")
    chunk_steps = _positive_integer(chunk_steps, "chunk_steps")
    metric_batch_size = _positive_integer(metric_batch_size, "metric_batch_size")
    metric_samples = _positive_integer(metric_samples, "metric_samples")
    if n_steps > np.iinfo(np.int32).max:
        raise ValueError("n_steps exceeds the JAX scan iteration range.")
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(beta0) or beta0 <= 0:
        raise ValueError("tau and beta0 must be finite and positive.")
    if metric_mode not in {"sampled", "full"}:
        raise ValueError("metric_mode must be sampled or full.")
    if measurement_backend not in {"auto", "tensor", "rank-one"}:
        raise ValueError("measurement_backend must be auto, tensor or rank-one.")
    vectors = rank_one_measurement_vectors(data.local_measurements)
    if measurement_backend == "rank-one" and vectors is None:
        raise ValueError("rank-one backend requires a verified rank-one PSD bank.")
    use_rank_one = measurement_backend != "tensor" and vectors is not None
    if metric_mode == "full" and data.m > np.iinfo(np.int64).max:
        raise ValueError("Full enumeration exceeds int64; use sampled metrics.")
    for flag, name in ((store_checkpoints, "store_checkpoints"), (warmup, "warmup"), (show_progress, "show_progress")):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")
    checkpoints = checkpoint_iterations(n_steps, metrics_frequency)
    rho, beta, gamma = _materialize_schedules(
        n_steps, beta0, rho_schedule, smoothing_schedule, step_size_schedule,
    )
    rng = np.random.default_rng(sampling_seed)
    metric_rng = np.random.default_rng(metric_seed)
    metric_symbols = (
        data.sample_symbols(metric_rng, metric_samples)
        if metric_mode == "sampled" else np.empty((0, data.n_qubits), dtype=np.int32)
    )
    x0_provided = x0 is not None
    if x0 is None:
        x0 = make_factor_initial_point(data, rank, initialization_seed)
    factor0 = unpack_factor(np.asarray(x0), data.process_dimension, rank)
    if not np.all(np.isfinite(factor0)) or np.linalg.norm(factor0, ord=2) > tau + 1e-10:
        raise ValueError("x0 must be finite and feasible for the operator-norm ball.")

    jax = _require_jax()
    precision = _configure_jax_precision(jax, precision)
    import jax.numpy as jnp
    import jaxlib
    selected = select_jax_device(device, device_index)
    complex_dtype = np.complex128 if precision == "64" else np.complex64
    real_dtype = np.float64 if precision == "64" else np.float32
    noiseless = data.observation_mode == "noiseless"
    start_wall = perf_counter()
    compile_seconds = optimizer_total = metric_seconds = transfer_seconds = 0.0
    plan_digest = hashlib.sha256()
    scalar_gaps = np.empty(n_steps, dtype=real_dtype)
    stored_factors = []
    rows = []
    times = []
    compiled = {}
    executable_memory_bytes = None
    did_warmup = False

    with jax.default_device(selected):
        def put(value, dtype):
            return jax.device_put(np.asarray(value, dtype=dtype), selected)

        bank = put(vectors if use_rank_one else data.local_measurements, complex_dtype)
        basis = put(data.local_basis, complex_dtype)
        truth = put(
            data.truth_factor if data.truth_factor is not None
            else np.empty((data.process_dimension, 0)), complex_dtype,
        )
        factor = put(factor0, complex_dtype)
        state = (factor, jnp.zeros_like(factor))
        jax.block_until_ready((bank, basis, truth, state))
        scan = _build_scan(jax, jnp, float(tau), noiseless, use_rank_one)
        values_fn = rank_one_measurement_values if use_rank_one else measurement_values
        gradient_fn = rank_one_measurement_loss_and_gradient if use_rank_one else measurement_loss_and_gradient

        @jax.jit
        def metric_batch(factor, symbols, observations, bank, truth):
            if noiseless:
                observations = values_fn(truth, symbols, bank, xp=jnp)
            return gradient_fn(factor, symbols, observations, bank, xp=jnp)

        @jax.jit
        def finish_metrics(factor, gradient, loss, smoothing, basis, truth):
            tp_loss, tp_gradient = trace_preserving_loss_and_gradient(factor, basis, xp=jnp)
            violation = jnp.sqrt(2.0 * tp_loss)
            moreau = tp_gradient / smoothing
            combined = gradient + moreau
            gap = jnp.vdot(combined, factor - _lmo(jnp, combined, float(tau))).real
            if truth.shape[1]:
                overlap = jnp.sum(jnp.abs(jnp.conj(truth.T) @ factor) ** 2)
                trace = jnp.sum(jnp.abs(factor) ** 2)
                fidelity = jnp.where(trace > 0, overlap / (data.d * trace), jnp.nan)
            else:
                fidelity = jnp.asarray(jnp.nan)
            return jnp.stack((loss, violation, loss + tp_loss / smoothing, gap, fidelity))

        def record(step):
            nonlocal metric_seconds
            started = perf_counter()
            factor = state[0]
            population = metric_samples if metric_mode == "sampled" else data.m
            gradient = jnp.zeros_like(factor)
            loss = jnp.asarray(0.0, dtype=real_dtype)
            for offset in range(0, population, metric_batch_size):
                count = min(metric_batch_size, population - offset)
                if metric_mode == "sampled":
                    symbols = metric_symbols[offset:offset + count]
                else:
                    symbols = data.indices_to_symbols(np.arange(offset, offset + count, dtype=np.int64))
                observations = (
                    np.zeros(count, dtype=real_dtype) if noiseless
                    else data.observations_for_symbols(symbols)
                )
                batch_loss, batch_gradient = metric_batch(
                    factor, put(symbols, np.int32), put(observations, real_dtype), bank, truth,
                )
                loss = loss + (count / population) * batch_loss
                gradient = gradient + (count / population) * batch_gradient
                # Prevent asynchronous metric dispatch accumulating batches.
                jax.block_until_ready((loss, gradient))
            values = np.asarray(jax.device_get(finish_metrics(
                factor, gradient, loss, beta[min(step, n_steps - 1)], basis, truth,
            )))
            if not np.all(np.isfinite(values[:4])):
                raise FloatingPointError(f"Nonfinite checkpoint metrics at step {step}.")
            rows.append(values)
            times.append(optimizer_total)
            if store_checkpoints:
                stored_factors.append(np.asarray(jax.device_get(factor)))
            metric_seconds += perf_counter() - started

        record(0)
        completed = 0
        for checkpoint in checkpoints[1:]:
            while completed < checkpoint:
                length = min(chunk_steps, int(checkpoint) - completed)
                transfer_started = perf_counter()
                symbols = data.sample_symbols(rng, length * batch_size).reshape(length, batch_size, data.n_qubits)
                plan_digest.update(np.asarray(symbols, dtype=np.int32).tobytes())
                observations = (
                    np.zeros((length, batch_size), dtype=real_dtype) if noiseless
                    else data.observations_for_symbols(symbols.reshape(-1, data.n_qubits)).reshape(length, batch_size)
                )
                args = (
                    state, put(completed, np.int32), put(symbols, np.int32),
                    put(observations, real_dtype), put(rho[completed:completed+length], real_dtype),
                    put(beta[completed:completed+length], real_dtype),
                    put(gamma[completed:completed+length], real_dtype), bank, basis, truth,
                )
                jax.block_until_ready(args)
                transfer_seconds += perf_counter() - transfer_started
                if length not in compiled:
                    compile_started = perf_counter()
                    executable = scan.lower(*args).compile()
                    compile_seconds += perf_counter() - compile_started
                    compiled[length] = executable
                    # Backend estimate, not a substitute for measured GPU RSS.
                    estimate = _executable_memory_estimate(executable)
                    if estimate is not None:
                        executable_memory_bytes = max(executable_memory_bytes or 0, estimate)
                executable = compiled[length]
                if warmup and not did_warmup:
                    warm = executable(*args)
                    jax.block_until_ready(warm)
                    del warm
                    did_warmup = True
                started = perf_counter()
                state, gaps = executable(*args)
                jax.block_until_ready((state, gaps))
                optimizer_total += perf_counter() - started
                scalar_gaps[completed:completed+length] = np.asarray(jax.device_get(gaps))
                completed += length
                if not np.all(np.isfinite(scalar_gaps[completed-length:completed])):
                    raise FloatingPointError("Nonfinite estimated gaps during structured QPT.")
            record(int(checkpoint))
            if show_progress:
                print(f"step {checkpoint}/{n_steps}: loss={rows[-1][0]:.6g}, TP={rows[-1][1]:.6g}, optimizer={optimizer_total:.3f}s")

        final_factor, estimate = (np.asarray(jax.device_get(value)) for value in state)
    if not np.all(np.isfinite(final_factor)) or not np.all(np.isfinite(estimate)):
        raise FloatingPointError("Nonfinite final factor or gradient estimate.")
    metrics = np.asarray(rows)
    metadata = {
        "schema_version": 1, "format": "qpt_structured_result",
        "formulation": "nonlinear_tp_moreau_exact", "penalty_coefficient": "1/(2*beta)",
        "backend": "jax", "device": str(selected), "device_platform": selected.platform,
        "device_kind": selected.device_kind, "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__, "numpy_version": np.__version__,
        "precision": precision, "rank": rank, "tau": float(tau), "n_qubits": data.n_qubits,
        "n_steps": n_steps, "batch_size": batch_size, "chunk_steps": chunk_steps,
        "initialization_seed": None if x0_provided else int(initialization_seed),
        "x0_provided": x0_provided,
        "initial_factor_sha256": hashlib.sha256(np.ascontiguousarray(factor0).tobytes()).hexdigest(),
        "initial_factor_dtype": str(factor0.dtype),
        "sampling_seed": int(sampling_seed),
        "batch_symbols_sha256": plan_digest.hexdigest(), "metric_seed": int(metric_seed),
        "metric_mode": metric_mode, "metric_samples": metric_samples if metric_mode == "sampled" else data.m,
        "metric_batch_size": metric_batch_size, "metrics_frequency": int(metrics_frequency),
        "gap_is_exact": metric_mode == "full", "observation_mode": data.observation_mode,
        "measurement_backend": "rank-one" if use_rank_one else "tensor",
        "observation_count": data.m, "sampled_measurements": n_steps * batch_size,
        "optimizer_seconds": float(optimizer_total), "compile_seconds": float(compile_seconds),
        "sampling_transfer_seconds": float(transfer_seconds), "metric_seconds": float(metric_seconds),
        "wall_seconds": float(perf_counter() - start_wall), "warmup": bool(warmup),
        "store_checkpoints": bool(store_checkpoints),
        "compiled_memory_estimate_bytes": executable_memory_bytes,
        "operator_bank_bytes": int(data.local_measurements.nbytes + data.local_basis.nbytes),
        "device_operator_bytes": int(bank.nbytes + basis.nbytes),
        "host_observation_bytes": int(data.observations.nbytes) if data.observations is not None else 0,
        "data_metadata": data.metadata,
    }
    return StructuredQPTResult(
        metadata, final_factor, estimate, checkpoints,
        np.stack(stored_factors) if store_checkpoints else np.empty((0,) + final_factor.shape, dtype=complex_dtype),
        beta[np.minimum(checkpoints, n_steps - 1)], np.asarray(times),
        metrics[:, 0], metrics[:, 1], metrics[:, 2], metrics[:, 3], metrics[:, 4],
        scalar_gaps, rho, beta, gamma, metric_symbols,
    )


def save_structured_result(path, result):
    """Save factors and scalar traces; never materialize the dense chi matrix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field.name: getattr(result, field.name) for field in fields(result) if field.name != "metadata"}
    payload["metadata_json"] = np.asarray(json.dumps(result.metadata, sort_keys=True, allow_nan=False))
    if any(np.asarray(value).dtype.hasobject for value in payload.values()):
        raise TypeError("Object arrays are forbidden in structured QPT archives.")
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return path


def load_structured_result(path):
    with np.load(path, allow_pickle=False) as archive:
        expected = {field.name for field in fields(StructuredQPTResult)} - {"metadata"}
        if set(archive.files) != expected | {"metadata_json"}:
            raise ValueError("Unexpected structured result archive fields.")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema_version") != 1 or metadata.get("format") != "qpt_structured_result":
            raise ValueError("Unsupported structured result format/version.")
        return StructuredQPTResult(metadata=metadata, **{key: archive[key].copy() for key in expected})


def plot_structured_result(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    qualifier = "Fixed-sample" if result.metadata["metric_mode"] == "sampled" else "Full"
    for axis, values, title in zip(axes, (result.measurement_loss, result.tp_violation, result.smoothed_gap),
                                    (f"{qualifier} measurement loss", "Exact TP violation", f"{qualifier} smoothed gap")):
        axis.plot(result.optimizer_seconds, values)
        axis.set(xlabel="Synchronized optimizer seconds", title=title)
        axis.grid(alpha=0.25)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Verified compact data produced by qpt_structured_data.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-steps", type=int, default=100)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--metric-mode", choices=("sampled", "full"), default="sampled")
    parser.add_argument("--metric-samples", type=int, default=512)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--metric-seed", type=int, default=12345)
    parser.add_argument("--measurement-backend", choices=("auto", "tensor", "rank-one"), default="auto")
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "tpu"), default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--precision", choices=("64", "32"), default="64")
    parser.add_argument("--store-checkpoints", action="store_true")
    parser.add_argument("--warmup", action="store_true", help="Execute one extra chunk outside optimizer timing.")
    for name, scale, offset, exponent in (("rho", 4.0, 8.0, 2.0/3.0), ("smoothing", 10.0, 1.0, 0.25), ("step", 1.0, 1.0, 0.5)):
        parser.add_argument(f"--{name}-scale", type=float, default=scale)
        parser.add_argument(f"--{name}-offset", type=float, default=offset)
        parser.add_argument(f"--{name}-exponent", type=float, default=exponent)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    data = StructuredQPTData.load_npz(args.data)
    schedules = {
        name: PowerSchedule(getattr(args, name+"_scale"), getattr(args, name+"_offset"),
                            getattr(args, name+"_exponent"), cap=None if name == "smoothing" else 1.0)
        for name in ("rho", "smoothing", "step")
    }
    result = run_qpt_structured_jax(
        data, n_steps=args.steps, rank=args.rank, tau=args.tau, batch_size=args.batch_size,
        initialization_seed=args.initialization_seed, sampling_seed=args.sampling_seed,
        rho_schedule=schedules["rho"], smoothing_schedule=schedules["smoothing"], step_size_schedule=schedules["step"],
        chunk_steps=args.chunk_steps, metrics_frequency=args.metrics_every, metric_mode=args.metric_mode,
        metric_samples=args.metric_samples, metric_batch_size=args.metric_batch_size, metric_seed=args.metric_seed,
        device=args.device, device_index=args.device_index, precision=args.precision,
        store_checkpoints=args.store_checkpoints, warmup=args.warmup, show_progress=not args.quiet,
        measurement_backend=args.measurement_backend,
    )
    save_structured_result(args.save, result)
    if args.plot:
        plot_structured_result(result, args.plot)
    print(f"Saved {args.save}; optimizer={result.total_seconds:.3f}s; compilation={result.metadata['compile_seconds']:.3f}s; metrics={result.metadata['metric_seconds']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
