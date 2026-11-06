"""JAX/GPU implementation of the nonconvex stochastic-FRAMES QPT run.

This module mirrors :mod:`quantum_process_tomography` while keeping the
optimization loop and all dense tensor contractions on a selected JAX
device.  JAX is an optional dependency and is imported only when a JAX run or
device query is requested, so importing this module and displaying its CLI
help do not require JAX.

The decision variable is the packed-real representation of QPT_BFW's complex
Burer--Monteiro factor ``U``.  Only a sampled measurement-loss gradient enters
the momentum estimator.  The trace-preserving Moreau term is evaluated
exactly at the current point, using the nonlinear map's Jacobian adjoint as if
the map were linear, and is divided by the scheduled smoothing parameter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np

from paper.experiments.quantum_process_tomography import (
    PowerSchedule,
    QPTData,
    QPTMeasurementObjective,
    Schedule,
    _checkpoint_smoothing_parameters,
    _hermitian_part,
    checkpoint_iterations,
    create_operator_norm_factor_lmo,
    make_factor_initial_point,
    plot_qpt_result,
    process_fidelity_proxy,
    unpack_factor,
)


def _require_jax():
    """Import JAX lazily and provide an actionable installation error."""

    try:
        import jax
    except ImportError as error:
        raise ImportError(
            "The JAX QPT runner requires the 'qpt-jax' optional "
            "dependencies. Install them with "
            "`python -m pip install -e '.[qpt-jax]'`. For a CUDA GPU, "
            "install the JAX wheel matching the target CUDA runtime as "
            "documented by JAX."
        ) from error
    return jax


def _configure_jax_precision(jax, precision):
    """Configure JAX precision before this module creates device arrays."""

    precision = str(precision)
    if precision not in {"32", "64"}:
        raise ValueError("precision must be either '32' or '64'.")
    try:
        jax.config.update("jax_enable_x64", precision == "64")
    except RuntimeError as error:
        raise RuntimeError(
            "JAX precision could not be configured. Call the QPT runner "
            "before creating JAX arrays, or start a fresh process and set "
            f"precision={precision!r} there."
        ) from error
    return precision


def _devices_for_platform(jax, platform):
    """Return devices for a platform without leaking backend-init errors."""

    try:
        return list(jax.devices(platform))
    except RuntimeError:
        return []


def select_jax_device(device="auto", index=0):
    """Select a JAX device, failing loudly when a request is unavailable.

    ``auto`` prefers GPU, then TPU, then CPU.  An explicit platform never
    falls back to another platform.  ``index`` is interpreted within the
    selected platform.
    """

    jax = _require_jax()
    device = str(device).lower()
    if device not in {"auto", "cpu", "gpu", "tpu"}:
        raise ValueError("device must be one of: auto, cpu, gpu, tpu.")
    if not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError("device index must be a nonnegative integer.")
    index = int(index)

    if device == "auto":
        selected_platform = None
        available = []
        for platform in ("gpu", "tpu", "cpu"):
            candidates = _devices_for_platform(jax, platform)
            if candidates:
                selected_platform = platform
                available = candidates
                break
        if selected_platform is None:
            raise RuntimeError("JAX did not report any usable devices.")
    else:
        selected_platform = device
        available = _devices_for_platform(jax, selected_platform)
        if not available:
            detected = []
            for platform in ("cpu", "gpu", "tpu"):
                if _devices_for_platform(jax, platform):
                    detected.append(platform)
            detected_text = ", ".join(detected) if detected else "none"
            raise RuntimeError(
                f"Requested JAX {selected_platform!r} device, but that "
                f"backend is unavailable. Detected backends: {detected_text}. "
                "Install a JAX build for the requested accelerator or choose "
                "--device cpu/auto."
            )

    if index >= len(available):
        raise RuntimeError(
            f"Requested {selected_platform} device index {index}, but JAX "
            f"reported {len(available)} device(s) for that backend."
        )
    return available[index]


def _resolve_host_schedule(schedule, default, name, n_steps, predicate, rule):
    """Evaluate every requested schedule value exactly once on the host."""

    if schedule is None:
        schedule_fn = default
    elif callable(schedule):
        schedule_fn = schedule
    elif np.isscalar(schedule):
        scalar = float(schedule)

        def schedule_fn(_iteration, value=scalar):
            return value

    else:
        raise TypeError(f"{name} must be a scalar, callable, or None.")

    values = np.empty(n_steps, dtype=np.float64)
    for iteration in range(n_steps):
        value = schedule_fn(iteration)
        if not np.isscalar(value):
            raise TypeError(
                f"{name} must return a scalar; got {type(value).__name__} "
                f"at iteration {iteration}."
            )
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                f"{name} must return a real scalar at iteration "
                f"{iteration}; got {value!r}."
            ) from error
        if not np.isfinite(value) or not predicate(value):
            raise ValueError(
                f"{name} must return {rule}; got {value!r} at iteration "
                f"{iteration}."
            )
        values[iteration] = value
    return values


def _materialize_schedules(
    n_steps,
    beta0,
    rho_schedule,
    smoothing_schedule,
    step_size_schedule,
):
    """Materialize FRAMES schedules once, before JIT tracing begins."""

    rho = _resolve_host_schedule(
        rho_schedule,
        lambda iteration: min(
            1.0, 4.0 / (iteration + 8) ** (2.0 / 3.0)
        ),
        "rho_schedule",
        n_steps,
        lambda value: 0.0 < value <= 1.0,
        "a finite value in (0, 1]",
    )
    smoothing = _resolve_host_schedule(
        smoothing_schedule,
        lambda iteration: beta0 / (iteration + 1) ** 0.25,
        "smoothing_schedule",
        n_steps,
        lambda value: value > 0.0,
        "a positive finite value",
    )
    step_size = _resolve_host_schedule(
        step_size_schedule,
        lambda iteration: 1.0 / (iteration + 1) ** 0.5,
        "step_size_schedule",
        n_steps,
        lambda value: 0.0 <= value <= 1.0,
        "a finite value in [0, 1]",
    )
    return rho, smoothing, step_size


@dataclass
class QPTJAXExperimentResult:
    """Optimization trace and backend metadata for a JAX QPT run."""

    rank: int
    tau: float
    final_x: np.ndarray
    final_factor: np.ndarray
    final_chi: np.ndarray
    checkpoint_steps: np.ndarray
    optimizer_seconds: np.ndarray
    measurement_loss: np.ndarray
    tp_violation: np.ndarray
    smoothed_objective: np.ndarray
    process_fidelity_proxy: np.ndarray
    exact_smoothed_gap: np.ndarray
    minimum_eigenvalue: np.ndarray
    checkpoint_smoothing_parameters: np.ndarray
    estimated_gaps: np.ndarray
    momentum_weights: np.ndarray
    smoothing_parameters: np.ndarray
    step_sizes: np.ndarray
    cumulative_stochastic_oracles: np.ndarray
    cumulative_sampled_measurements: np.ndarray
    backend: str
    device: str
    device_platform: str
    device_kind: str
    jax_version: str
    jaxlib_version: str
    precision: str
    mode: str
    compile_seconds: float
    total_seconds: float
    step_times: np.ndarray
    batch_indices: np.ndarray
    final_gradient_estimate: np.ndarray
    warmup: bool
    algorithm: object = None

    @property
    def execution_mode(self):
        """Alias matching the runner argument name."""

        return self.mode


def _build_jax_step(jax, jnp, process_dimension, rank, tau):
    """Build one pure packed-real stochastic-FRAMES update."""

    factor_size = process_dimension * rank

    def unpack(vector):
        return vector[:factor_size].reshape(process_dimension, rank) + 1j * (
            vector[factor_size:].reshape(process_dimension, rank)
        )

    def pack(matrix):
        return jnp.concatenate(
            (jnp.real(matrix).reshape(-1), jnp.imag(matrix).reshape(-1))
        )

    def hermitian_part(matrix):
        return 0.5 * (matrix + jnp.conj(matrix.T))

    def measurement_gradient(vector, indices, f_vector, D_tensors):
        factor = unpack(vector)
        chi = factor @ jnp.conj(factor.T)
        tensors = D_tensors[indices]
        sensing = jnp.real(
            jnp.einsum("mab,ab->m", jnp.conj(tensors), chi)
        )
        residual = sensing - f_vector[indices]
        chi_gradient = jnp.einsum(
            "m,mab->ab", residual, tensors
        ) / indices.shape[0]
        chi_gradient = hermitian_part(chi_gradient)
        return 2.0 * pack(chi_gradient @ factor), chi, factor

    def moreau_gradient(vector, factor, chi, beta, B_tensors, identity):
        del vector
        trace_map = jnp.einsum("nm,nmij->ij", chi, B_tensors)
        residual = trace_map - identity
        chi_gradient = jnp.einsum(
            "nmij,ij->nm", jnp.conj(B_tensors), residual
        )
        chi_gradient = hermitian_part(chi_gradient)
        return 2.0 * pack(chi_gradient @ factor) / beta

    def lmo(packed_gradient):
        gradient = unpack(packed_gradient)
        norm = jnp.linalg.norm(gradient)
        if rank == 1:
            safe_norm = jnp.maximum(norm, jnp.finfo(gradient.real.dtype).tiny)
            atom = -tau * gradient / safe_norm
        else:
            left, _, right_h = jnp.linalg.svd(
                gradient, full_matrices=False
            )
            atom = -tau * (left @ right_h)
        # SVD(0) may return an arbitrary polar factor.  The exact LMO used by
        # the NumPy implementation deliberately returns the zero atom when
        # the objective is constant, so guard every rank here as well.
        atom = jnp.where(norm > 0.0, atom, jnp.zeros_like(atom))
        return pack(atom)

    def step(
        state,
        iteration,
        indices,
        rho,
        beta,
        step_size,
        f_vector,
        D_tensors,
        B_tensors,
        identity,
    ):
        vector, previous_estimate = state
        sampled_gradient, chi, factor = measurement_gradient(
            vector, indices, f_vector, D_tensors
        )
        gradient_estimate = jax.lax.cond(
            iteration == 0,
            lambda _: sampled_gradient,
            lambda _: (
                (1.0 - rho) * previous_estimate
                + rho * sampled_gradient
            ),
            operand=None,
        )
        combined_gradient = gradient_estimate + moreau_gradient(
            vector, factor, chi, beta, B_tensors, identity
        )
        atom = lmo(combined_gradient)
        estimated_gap = jnp.vdot(
            combined_gradient, vector - atom
        ).real
        next_vector = (1.0 - step_size) * vector + step_size * atom
        return (next_vector, gradient_estimate), estimated_gap

    return step


def _run_scan_mode(
    jax,
    jnp,
    step,
    state0,
    batch_indices,
    rho,
    smoothing,
    step_sizes,
    f_vector,
    D_tensors,
    B_tensors,
    identity,
    warmup,
):
    """Run the complete optimizer as one compiled ``lax.scan``."""

    def scan_function(
        initial_state,
        batches,
        rho_values,
        beta_values,
        gamma_values,
        f_values,
        D_values,
        B_values,
        identity_value,
    ):
        iterations = jnp.arange(batches.shape[0], dtype=jnp.int32)

        def body(state, inputs):
            iteration, indices, rho_value, beta, gamma = inputs
            next_state, gap = step(
                state,
                iteration,
                indices,
                rho_value,
                beta,
                gamma,
                f_values,
                D_values,
                B_values,
                identity_value,
            )
            return next_state, (next_state[0], gap)

        return jax.lax.scan(
            body,
            initial_state,
            (iterations, batches, rho_values, beta_values, gamma_values),
        )

    jitted = jax.jit(scan_function)
    arguments = (
        state0,
        batch_indices,
        rho,
        smoothing,
        step_sizes,
        f_vector,
        D_tensors,
        B_tensors,
        identity,
    )
    compile_seconds = 0.0
    if warmup:
        compile_started = perf_counter()
        executable = jitted.lower(*arguments).compile()
        compile_seconds = perf_counter() - compile_started
        run_function = executable
        warmup_output = run_function(*arguments)
        jax.block_until_ready(warmup_output)
    else:
        run_function = jitted

    started = perf_counter()
    final_state, (iterate_history, gaps) = run_function(*arguments)
    jax.block_until_ready((final_state, iterate_history, gaps))
    total_seconds = perf_counter() - started
    return (
        final_state,
        iterate_history,
        gaps,
        float(compile_seconds),
        float(total_seconds),
        np.asarray([total_seconds], dtype=float),
    )


def _run_step_mode(
    jax,
    step,
    state0,
    batch_indices,
    rho,
    smoothing,
    step_sizes,
    f_vector,
    D_tensors,
    B_tensors,
    identity,
    checkpoint_steps,
    warmup,
    show_progress,
):
    """Run a compiled update per Python iteration and synchronize each one."""

    jitted = jax.jit(step)
    warmup_arguments = (
        state0,
        np.int32(0),
        batch_indices[0],
        rho[0],
        smoothing[0],
        step_sizes[0],
        f_vector,
        D_tensors,
        B_tensors,
        identity,
    )
    compile_seconds = 0.0
    if warmup:
        compile_started = perf_counter()
        executable = jitted.lower(*warmup_arguments).compile()
        compile_seconds = perf_counter() - compile_started
        run_function = executable
        warmup_output = run_function(*warmup_arguments)
        jax.block_until_ready(warmup_output)
    else:
        run_function = jitted

    iterations = range(batch_indices.shape[0])
    if show_progress:
        from tqdm.auto import tqdm

        iterations = tqdm(iterations, desc="JAX stochastic FRAMES")

    requested = set(np.asarray(checkpoint_steps, dtype=int).tolist())
    checkpoint_states = {0: state0[0]}
    gaps = []
    elapsed = np.empty(batch_indices.shape[0], dtype=float)
    state = state0
    for iteration in iterations:
        started = perf_counter()
        state, gap = run_function(
            state,
            np.int32(iteration),
            batch_indices[iteration],
            rho[iteration],
            smoothing[iteration],
            step_sizes[iteration],
            f_vector,
            D_tensors,
            B_tensors,
            identity,
        )
        jax.block_until_ready((state, gap))
        elapsed[iteration] = perf_counter() - started
        gaps.append(gap)
        completed_steps = iteration + 1
        if completed_steps in requested:
            checkpoint_states[completed_steps] = state[0]

    checkpoint_iterates = [checkpoint_states[int(k)] for k in checkpoint_steps]
    total_seconds = float(np.sum(elapsed))
    return (
        state,
        checkpoint_iterates,
        gaps,
        float(compile_seconds),
        total_seconds,
        elapsed,
    )


def _posthoc_metrics(data, rank, tau, vectors, steps, smoothing_parameters):
    """Evaluate common NumPy objective diagnostics after device timing."""

    objective = QPTMeasurementObjective(data, rank=rank, batch_size=1, seed=0)
    lmo = create_operator_norm_factor_lmo(
        data.process_dimension, rank, tau
    )
    checkpoint_beta = _checkpoint_smoothing_parameters(
        steps, smoothing_parameters
    )
    identity = np.eye(data.d, dtype=np.complex128)

    measurement_loss = []
    tp_violation = []
    smoothed_objective = []
    fidelity_proxy = []
    exact_smoothed_gap = []
    minimum_eigenvalue = []
    for vector, beta in zip(vectors, checkpoint_beta):
        loss, smooth_gradient = objective.loss_and_gradient(vector)
        residual = objective.linear_operator(vector) - identity
        violation = np.linalg.norm(residual, ord="fro")
        moreau_gradient = (
            objective.linear_operator_adjoint_at(vector, residual) / beta
        )
        combined_gradient = smooth_gradient + moreau_gradient
        atom = lmo(combined_gradient)
        chi = objective.process_matrix(vector)

        measurement_loss.append(loss)
        tp_violation.append(violation)
        smoothed_objective.append(loss + 0.5 * violation**2 / beta)
        fidelity_proxy.append(
            process_fidelity_proxy(chi, data.chi_star, data.d)
        )
        exact_smoothed_gap.append(
            np.dot(combined_gradient, vector - atom)
        )
        minimum_eigenvalue.append(
            np.linalg.eigvalsh(_hermitian_part(chi))[0]
        )

    return {
        "checkpoint_smoothing_parameters": checkpoint_beta.copy(),
        "measurement_loss": np.asarray(measurement_loss),
        "tp_violation": np.asarray(tp_violation),
        "smoothed_objective": np.asarray(smoothed_objective),
        "process_fidelity_proxy": np.asarray(fidelity_proxy),
        "exact_smoothed_gap": np.asarray(exact_smoothed_gap),
        "minimum_eigenvalue": np.asarray(minimum_eigenvalue),
    }


def run_qpt_stochastic_frames_jax(
    data,
    *,
    n_steps=1000,
    rank=1,
    tau=10.0,
    batch_size=1,
    initialization_seed=0,
    sampling_seed=0,
    x0=None,
    beta0=10.0,
    rho_schedule: Schedule = None,
    smoothing_schedule: Schedule = None,
    step_size_schedule: Schedule = None,
    metrics_frequency=100,
    device="auto",
    device_index=0,
    precision="64",
    execution_mode="scan",
    warmup=True,
    show_progress=True,
):
    """Run nonconvex stochastic FRAMES on a selected JAX device.

    Schedules are evaluated exactly once per iteration on the host before any
    JIT tracing.  Measurement batches are also generated on the host with
    ``numpy.random.default_rng(sampling_seed)`` and sampled uniformly with
    replacement, making them inspectable and reproducible across JAX devices.

    ``execution_mode='scan'`` compiles the entire loop as one ``lax.scan`` for
    throughput.  Its ``step_times`` contains the single synchronized scan
    duration, and only endpoint optimizer times are intrinsically observable.
    ``execution_mode='step'`` invokes a jitted update and synchronizes after
    every iteration; its timings match QPT_BFW's per-iteration wall-clock
    convention but incur Python dispatch overhead.

    With ``warmup=True``, compilation is performed before the timed run and
    recorded separately in ``compile_seconds``; the compiled executable is
    then run once and synchronized outside the timer (a complete scan in scan
    mode, or one update in step mode).  With ``warmup=False``, the first timed
    invocation includes JIT compilation and ``compile_seconds`` is zero
    because compilation is not separately observable.
    """

    jax = _require_jax()
    precision = _configure_jax_precision(jax, precision)
    import jax.numpy as jnp
    import jaxlib

    selected_device = select_jax_device(device, device_index)
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if not isinstance(rank, (int, np.integer)) or rank <= 0:
        raise ValueError("rank must be a positive integer.")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite number.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not np.isfinite(beta0) or beta0 <= 0:
        raise ValueError("beta0 must be a positive finite number.")
    if execution_mode not in {"scan", "step"}:
        raise ValueError("execution_mode must be either 'scan' or 'step'.")
    if not isinstance(warmup, (bool, np.bool_)):
        raise TypeError("warmup must be a boolean.")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress must be a boolean.")

    n_steps = int(n_steps)
    rank = int(rank)
    batch_size = int(batch_size)
    checkpoint_steps = checkpoint_iterations(n_steps, metrics_frequency)
    rho_values, beta_values, gamma_values = _materialize_schedules(
        n_steps,
        beta0,
        rho_schedule,
        smoothing_schedule,
        step_size_schedule,
    )
    rng = np.random.default_rng(sampling_seed)
    sampled_indices = rng.integers(
        0, data.m, size=(n_steps, batch_size), dtype=np.int64
    )

    if x0 is None:
        x0 = make_factor_initial_point(data, rank, initialization_seed)
    x0 = np.asarray(x0, dtype=float)
    expected_size = 2 * data.process_dimension * rank
    if x0.ndim != 1 or x0.size != expected_size:
        raise ValueError(
            f"x0 must have shape ({expected_size},); got {x0.shape}."
        )
    if not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain finite values.")
    if np.linalg.norm(
        unpack_factor(x0, data.process_dimension, rank), ord=2
    ) > tau + 1e-10:
        raise ValueError("x0 must satisfy the operator-norm constraint.")

    real_dtype = np.float64 if precision == "64" else np.float32
    complex_dtype = np.complex128 if precision == "64" else np.complex64
    with jax.default_device(selected_device):
        x_device = jax.device_put(
            np.asarray(x0, dtype=real_dtype), selected_device
        )
        zero_estimate = jnp.zeros_like(x_device)
        state0 = (x_device, zero_estimate)
        batches_device = jax.device_put(
            sampled_indices.astype(np.int32), selected_device
        )
        rho_device = jax.device_put(
            rho_values.astype(real_dtype), selected_device
        )
        beta_device = jax.device_put(
            beta_values.astype(real_dtype), selected_device
        )
        gamma_device = jax.device_put(
            gamma_values.astype(real_dtype), selected_device
        )
        f_device = jax.device_put(
            data.f_vector.astype(real_dtype), selected_device
        )
        D_device = jax.device_put(
            data.D_tensors.astype(complex_dtype, copy=False), selected_device
        )
        B_device = jax.device_put(
            data.B_tensors.astype(complex_dtype, copy=False), selected_device
        )
        identity_device = jax.device_put(
            np.eye(data.d, dtype=complex_dtype), selected_device
        )
        # Exclude one-time host-to-device transfer from optimizer timings.
        jax.block_until_ready(
            (
                state0,
                batches_device,
                rho_device,
                beta_device,
                gamma_device,
                f_device,
                D_device,
                B_device,
                identity_device,
            )
        )

        step = _build_jax_step(
            jax, jnp, data.process_dimension, rank, float(tau)
        )
        if execution_mode == "scan":
            (
                final_state,
                iterate_history,
                gaps_device,
                compile_seconds,
                total_seconds,
                step_times,
            ) = _run_scan_mode(
                jax,
                jnp,
                step,
                state0,
                batches_device,
                rho_device,
                beta_device,
                gamma_device,
                f_device,
                D_device,
                B_device,
                identity_device,
                bool(warmup),
            )
            iterate_history = np.asarray(jax.device_get(iterate_history))
            initial_vector = np.asarray(jax.device_get(x_device))
            vectors = np.concatenate(
                (initial_vector[None, :], iterate_history), axis=0
            )[checkpoint_steps]
            optimizer_seconds = np.full(
                checkpoint_steps.shape, np.nan, dtype=float
            )
            optimizer_seconds[checkpoint_steps == 0] = 0.0
            optimizer_seconds[checkpoint_steps == n_steps] = total_seconds
        else:
            (
                final_state,
                checkpoint_iterates,
                gaps_device,
                compile_seconds,
                total_seconds,
                step_times,
            ) = _run_step_mode(
                jax,
                step,
                state0,
                batches_device,
                rho_device,
                beta_device,
                gamma_device,
                f_device,
                D_device,
                B_device,
                identity_device,
                checkpoint_steps,
                bool(warmup),
                bool(show_progress),
            )
            vectors = np.stack(
                [
                    np.asarray(jax.device_get(vector))
                    for vector in checkpoint_iterates
                ]
            )
            optimizer_seconds = np.concatenate(
                ([0.0], np.cumsum(step_times))
            )[checkpoint_steps]

    final_x = np.asarray(jax.device_get(final_state[0]))
    final_gradient_estimate = np.asarray(jax.device_get(final_state[1]))
    if execution_mode == "scan":
        estimated_gaps = np.asarray(jax.device_get(gaps_device))
    else:
        estimated_gaps = np.asarray(
            jax.device_get(gaps_device), dtype=real_dtype
        )
    estimated_gaps = estimated_gaps.reshape(n_steps)
    if not np.all(np.isfinite(final_x)):
        raise FloatingPointError("The final JAX iterate contains nonfinite values.")
    if not np.all(np.isfinite(final_gradient_estimate)):
        raise FloatingPointError(
            "The final JAX momentum estimate contains nonfinite values."
        )
    if not np.all(np.isfinite(estimated_gaps)):
        raise FloatingPointError("The JAX estimated gaps contain nonfinite values.")

    metrics = _posthoc_metrics(
        data,
        rank,
        float(tau),
        vectors,
        checkpoint_steps,
        beta_values,
    )
    final_factor = unpack_factor(final_x, data.process_dimension, rank)
    cumulative_oracles = np.arange(1, n_steps + 1, dtype=int)
    return QPTJAXExperimentResult(
        rank=rank,
        tau=float(tau),
        final_x=final_x.copy(),
        final_factor=final_factor,
        final_chi=final_factor @ final_factor.conj().T,
        checkpoint_steps=checkpoint_steps,
        optimizer_seconds=optimizer_seconds,
        estimated_gaps=estimated_gaps,
        momentum_weights=rho_values.copy(),
        smoothing_parameters=beta_values.copy(),
        step_sizes=gamma_values.copy(),
        cumulative_stochastic_oracles=cumulative_oracles,
        cumulative_sampled_measurements=cumulative_oracles * batch_size,
        backend="jax",
        device=str(selected_device),
        device_platform=str(selected_device.platform),
        device_kind=str(selected_device.device_kind),
        jax_version=str(jax.__version__),
        jaxlib_version=str(jaxlib.__version__),
        precision=precision,
        mode=execution_mode,
        compile_seconds=compile_seconds,
        total_seconds=total_seconds,
        step_times=np.asarray(step_times),
        batch_indices=sampled_indices,
        final_gradient_estimate=final_gradient_estimate.copy(),
        warmup=bool(warmup),
        algorithm=None,
        **metrics,
    )


def save_qpt_jax_result(path, result, metadata=None):
    """Save a JAX result and backend metadata without pickled objects."""

    payload = {
        "meta_formulation": np.asarray(
            "nonconvex_factor_nonlinear_tp_smoothing"
        ),
        "meta_nonlinear_composite": np.asarray(
            "jacobian_adjoint_as_linear"
        ),
        "meta_backend": np.asarray(result.backend),
        "meta_device": np.asarray(result.device),
        "meta_device_platform": np.asarray(result.device_platform),
        "meta_device_kind": np.asarray(result.device_kind),
        "meta_jax_version": np.asarray(result.jax_version),
        "meta_jaxlib_version": np.asarray(result.jaxlib_version),
        "meta_precision": np.asarray(result.precision),
        "meta_execution_mode": np.asarray(result.mode),
        "meta_warmup": np.asarray(result.warmup),
        "meta_rank": np.asarray(result.rank),
        "meta_tau": np.asarray(result.tau),
        "compile_seconds": np.asarray(result.compile_seconds),
        "total_seconds": np.asarray(result.total_seconds),
        "step_times": result.step_times,
        "batch_indices": result.batch_indices,
        "final_gradient_estimate": result.final_gradient_estimate,
        "final_x": result.final_x,
        "final_factor": result.final_factor,
        "final_chi": result.final_chi,
        "checkpoint_steps": result.checkpoint_steps,
        "optimizer_seconds": result.optimizer_seconds,
        "measurement_loss": result.measurement_loss,
        "tp_violation": result.tp_violation,
        "smoothed_objective": result.smoothed_objective,
        "process_fidelity_proxy": result.process_fidelity_proxy,
        "exact_smoothed_gap": result.exact_smoothed_gap,
        "minimum_eigenvalue": result.minimum_eigenvalue,
        "checkpoint_smoothing_parameters": (
            result.checkpoint_smoothing_parameters
        ),
        "estimated_gaps": result.estimated_gaps,
        "momentum_weights": result.momentum_weights,
        "smoothing_parameters": result.smoothing_parameters,
        "step_sizes": result.step_sizes,
        "cumulative_stochastic_oracles": (
            result.cumulative_stochastic_oracles
        ),
        "cumulative_sampled_measurements": (
            result.cumulative_sampled_measurements
        ),
    }
    if metadata:
        payload.update(
            {f"meta_{key}": np.asarray(value) for key, value in metadata.items()}
        )
    np.savez_compressed(path, **payload)


def plot_qpt_jax_result(result, path=None):
    """Plot JAX checkpoint metrics using the common QPT plot layout."""

    return plot_qpt_result(result, path)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run nonconvex stochastic FRAMES on QPT_BFW's factorized QPT "
            "problem using JAX on an explicitly selected device."
        )
    )
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="HDF5 file generated by QPT_BFW/qutomo_gt_gen.ipynb.",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--initialization-seed", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)

    schedules = parser.add_argument_group(
        "power schedules",
        "Each schedule is scale / (iteration + offset)^exponent. Rho "
        "and step size are capped at one. rho_0 is recorded, while d_0 "
        "is initialized from the complete first sampled gradient.",
    )
    schedules.add_argument("--rho-scale", type=float, default=4.0)
    schedules.add_argument("--rho-offset", type=float, default=8.0)
    schedules.add_argument("--rho-exponent", type=float, default=2.0 / 3.0)
    schedules.add_argument("--smoothing-scale", type=float, default=10.0)
    schedules.add_argument("--smoothing-offset", type=float, default=1.0)
    schedules.add_argument(
        "--smoothing-exponent", type=float, default=0.25
    )
    schedules.add_argument("--step-scale", type=float, default=1.0)
    schedules.add_argument("--step-offset", type=float, default=1.0)
    schedules.add_argument("--step-exponent", type=float, default=0.5)

    backend = parser.add_argument_group("JAX backend")
    backend.add_argument(
        "--device", choices=("auto", "cpu", "gpu", "tpu"), default="auto"
    )
    backend.add_argument("--device-index", type=int, default=0)
    backend.add_argument("--precision", choices=("64", "32"), default="64")
    backend.add_argument(
        "--execution-mode",
        "--mode",
        dest="execution_mode",
        choices=("scan", "step"),
        default="scan",
    )
    backend.add_argument(
        "--no-warmup",
        action="store_true",
        help="Include first-call JIT compilation in optimizer timing.",
    )

    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    data = QPTData.from_hdf5(args.h5)
    rho_schedule = PowerSchedule(
        args.rho_scale,
        args.rho_offset,
        args.rho_exponent,
        cap=1.0,
    )
    smoothing_schedule = PowerSchedule(
        args.smoothing_scale,
        args.smoothing_offset,
        args.smoothing_exponent,
    )
    step_size_schedule = PowerSchedule(
        args.step_scale,
        args.step_offset,
        args.step_exponent,
        cap=1.0,
    )

    if not args.quiet:
        print(
            f"Loaded QPT data: d={data.d}, process dimension="
            f"{data.process_dimension}, measurements={data.m}"
        )
    result = run_qpt_stochastic_frames_jax(
        data,
        n_steps=args.steps,
        rank=args.rank,
        tau=args.tau,
        batch_size=args.batch_size,
        initialization_seed=args.initialization_seed,
        sampling_seed=args.sampling_seed,
        rho_schedule=rho_schedule,
        smoothing_schedule=smoothing_schedule,
        step_size_schedule=step_size_schedule,
        metrics_frequency=args.metrics_every,
        device=args.device,
        device_index=args.device_index,
        precision=args.precision,
        execution_mode=args.execution_mode,
        warmup=not args.no_warmup,
        show_progress=not args.quiet,
    )
    print(
        f"JAX device={result.device}, mode={result.mode}, "
        f"precision={result.precision}; compile={result.compile_seconds:.3f}s, "
        f"optimizer={result.total_seconds:.3f}s"
    )
    print(
        f"Final measurement loss={result.measurement_loss[-1]:.6e}, "
        f"TP violation={result.tp_violation[-1]:.6e}, "
        f"exact smoothed gap={result.exact_smoothed_gap[-1]:.6e}"
    )
    if np.isfinite(result.process_fidelity_proxy[-1]):
        print(
            "Final process-fidelity proxy="
            f"{result.process_fidelity_proxy[-1]:.6f}"
        )
    print(
        "Sampled measurements: "
        f"{int(result.cumulative_sampled_measurements[-1])}"
    )

    if args.save is not None:
        save_qpt_jax_result(
            args.save,
            result,
            metadata={
                "source_h5": str(args.h5.resolve()),
                "d": data.d,
                "process_dimension": data.process_dimension,
                "measurement_count": data.m,
                "steps": args.steps,
                "rank": args.rank,
                "tau": args.tau,
                "batch_size": args.batch_size,
                "metrics_every": args.metrics_every,
                "initialization_seed": args.initialization_seed,
                "sampling_seed": args.sampling_seed,
                "rho_scale": args.rho_scale,
                "rho_offset": args.rho_offset,
                "rho_exponent": args.rho_exponent,
                "smoothing_scale": args.smoothing_scale,
                "smoothing_offset": args.smoothing_offset,
                "smoothing_exponent": args.smoothing_exponent,
                "step_scale": args.step_scale,
                "step_offset": args.step_offset,
                "step_exponent": args.step_exponent,
            },
        )
        print(f"Saved results to {args.save}")
    if args.plot is not None:
        plot_qpt_jax_result(result, args.plot)
        print(f"Saved plot to {args.plot}")
    return result


if __name__ == "__main__":
    main()
