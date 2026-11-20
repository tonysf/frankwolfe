"""Bounded, fixed-input diagnostics for the unmodified structured QPT operators.

The CPU reference explicitly builds only the selected minibatch's Kronecker
matrices, plus the global Pauli/TP tensors.  It uses the existing dense NumPy
objective, not the structured operator formulas.  The byte guard is a
conservative allocation estimate, not a measured process-memory limit.
"""

from __future__ import annotations

import hashlib
import itertools
from time import perf_counter

import numpy as np

from .qpt_structured_operators import (
    measurement_loss_and_gradient,
    rank_one_measurement_loss_and_gradient,
    rank_one_measurement_vectors,
    trace_preserving_loss_and_gradient,
)
from .quantum_process_tomography import (
    QPTData, QPTMeasurementObjective, create_operator_norm_factor_lmo,
    pack_factor, unpack_factor,
)
from .quantum_process_tomography_structured_jax import _lmo


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _array_summary(value):
    value = np.asarray(value)
    result = {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": _digest(value), "finite": bool(np.all(np.isfinite(value))),
    }
    if value.ndim == 0 and not np.iscomplexobj(value):
        result["value"] = _finite_float(value)
    return result


def _compare(actual, expected, *, rtol, atol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    finite = bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = actual - expected
        absolute = np.max(np.abs(delta))
        expected_norm = np.linalg.norm(expected.reshape(-1))
        delta_norm = np.linalg.norm(delta.reshape(-1))
        relative = delta_norm / expected_norm if expected_norm else (0.0 if not delta_norm else np.nan)
        matches = bool(finite and np.allclose(actual, expected, rtol=rtol, atol=atol))
    return {
        "finite": finite,
        "exact": bool(finite and np.array_equal(actual, expected)),
        "allclose": matches,
        "max_abs_difference": _finite_float(absolute),
        "relative_l2_difference": _finite_float(relative),
        "reference_l2_norm": _finite_float(expected_norm),
        "reference_is_zero": bool(expected_norm == 0),
    }


def _compare_outputs(actual, expected, *, rtol, atol):
    return {name: _compare(value, expected[name], rtol=rtol, atol=atol)
            for name, value in actual.items()}


def _reference_allocation_estimate(dimension, rank, batch):
    # complex128: selected D, global basis, global B, process matrices,
    # factor/gradient/output working storage.  Four copies cover constructors,
    # conjugation, dense-objective gather/einsum temporaries and retained output.
    core = 16 * (batch * dimension**2 + dimension**3 + 8 * dimension**2
                 + 32 * dimension * rank + 8 * batch)
    return int(4 * core)


def _kron_product(factors):
    result = np.ones((1, 1), dtype=np.complex128)
    for factor in factors:
        result = np.kron(result, factor)
    return result


def _dense_reference(factor, previous, symbols, observations, local_measurements,
                     local_basis, rho, beta, gamma, tau, iteration):
    n_qubits = symbols.shape[1]
    dimension, rank = factor.shape
    factor = np.asarray(factor, dtype=np.complex128)
    previous = np.asarray(previous, dtype=np.complex128)
    dense = np.empty((len(symbols), dimension, dimension), dtype=np.complex128)
    for index, row in enumerate(symbols):
        dense[index] = _kron_product(local_measurements[row])
    basis = np.empty((dimension, 2**n_qubits, 2**n_qubits), dtype=np.complex128)
    for index, row in enumerate(itertools.product(range(4), repeat=n_qubits)):
        basis[index] = _kron_product(local_basis[list(row)])
    objective = QPTMeasurementObjective(
        QPTData(observations, dense, basis), rank=rank, batch_size=len(symbols),
    )
    vector = pack_factor(factor)
    loss, packed_gradient = objective.loss_and_gradient(vector)
    gradient = unpack_factor(packed_gradient, dimension, rank)
    residual = objective.trace_preserving_residual(vector)
    tp_loss = 0.5 * np.vdot(residual, residual).real
    tp_gradient = unpack_factor(
        objective.linear_operator_adjoint_at(vector, residual), dimension, rank,
    )
    momentum = gradient if iteration == 0 else (1.0 - rho) * previous + rho * gradient
    moreau = tp_gradient / beta
    combined = momentum + moreau
    atom = unpack_factor(
        create_operator_norm_factor_lmo(dimension, rank, tau)(pack_factor(combined)),
        dimension, rank,
    )
    outputs = {
        "measurement_loss": np.asarray(loss), "measurement_gradient": gradient,
        "tp_loss": np.asarray(tp_loss), "tp_gradient": tp_gradient,
        "moreau_gradient": moreau, "momentum_gradient": momentum,
        "combined_gradient": combined, "lmo_atom": atom,
        "next_factor": (1.0 - gamma) * factor + gamma * atom,
        "estimated_gap": np.asarray(np.vdot(combined, factor - atom).real),
    }
    return outputs


def _validated_inputs(factor, previous, symbols, observations, local_measurements,
                      local_basis, rho, beta, gamma, tau, iteration, repeats,
                      rtol, atol, max_reference_bytes):
    factor, previous = np.asarray(factor), np.asarray(previous)
    if factor.ndim != 2 or min(factor.shape) < 1:
        raise ValueError("factor must have nonempty shape (4**n, rank).")
    dimension, n_qubits = factor.shape[0], 0
    while dimension > 1 and dimension % 4 == 0:
        dimension //= 4
        n_qubits += 1
    if dimension != 1 or n_qubits == 0 or previous.shape != factor.shape:
        raise ValueError("factor must have shape (4**n, rank); previous must match it.")
    symbols = np.asarray(symbols)
    if symbols.ndim != 2 or symbols.shape[0] < 1 or symbols.shape[1] != n_qubits:
        raise ValueError("symbols must have nonempty shape (batch, n_qubits).")
    if not np.issubdtype(symbols.dtype, np.integer):
        raise TypeError("symbols must contain integers.")
    if np.any(symbols < 0) or np.any(symbols >= 24):
        raise ValueError("symbols must lie in [0, 24).")
    if observations is None:
        raise ValueError("stored observations are required; probes do not regenerate targets.")
    observations = np.asarray(observations)
    if observations.shape != (symbols.shape[0],) or np.iscomplexobj(observations):
        raise ValueError("observations must be a real array of shape (batch,).")
    if np.shape(local_measurements) != (24, 4, 4) or np.shape(local_basis) != (4, 2, 2):
        raise ValueError("local_measurements/basis must have shapes (24,4,4)/(4,2,2).")
    for value, name in ((iteration, "iteration"), (repeats, "repeats"),
                        (max_reference_bytes, "max_reference_bytes")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
    if iteration < 0 or repeats < 2 or max_reference_bytes < 0:
        raise ValueError("iteration/cap must be nonnegative; repeats must be at least two.")
    scalars = (rho, beta, gamma, tau, rtol, atol)
    if not all(np.ndim(value) == 0 and np.isfinite(value) for value in scalars):
        raise ValueError("schedules, tau and tolerances must be finite scalars.")
    if not 0 <= rho <= 1 or beta <= 0 or not 0 <= gamma <= 1 or tau <= 0 or rtol < 0 or atol < 0:
        raise ValueError("invalid schedules, tau or tolerances.")
    for value in (factor, previous, observations, local_measurements, local_basis):
        if not np.all(np.isfinite(value)):
            raise ValueError("all probe inputs must be finite.")
    if factor.dtype not in (np.dtype("complex64"), np.dtype("complex128")):
        raise TypeError("factor must have dtype complex64 or complex128.")
    dtype = factor.dtype
    real_dtype = factor.real.dtype
    local_measurements = np.asarray(local_measurements, dtype=dtype)
    if not np.allclose(local_measurements, local_measurements.conj().swapaxes(1, 2),
                       rtol=1e-6 if dtype == np.complex64 else 1e-10, atol=1e-12):
        raise ValueError("local_measurements must be Hermitian.")
    converted = (factor, np.asarray(previous, dtype=dtype), symbols.astype(np.int32),
                 observations.astype(real_dtype), local_measurements,
                 np.asarray(local_basis, dtype=dtype))
    if any(not np.all(np.isfinite(value)) for value in converted):
        raise ValueError("probe inputs must remain finite in the execution precision.")
    return converted


def run_operator_probes(jax, jnp, device, *, factor, previous, symbols,
                       observations, local_measurements, local_basis, rho, beta,
                       gamma, tau, measurement_backend, repeats=3, rtol=1e-10,
                       atol=1e-12, max_reference_bytes=128 * 1024**2, iteration=1,
                       selected_bank=None):
    """Compare one fixed-input JIT operator probe with an explicit CPU reference.

    ``iteration=0`` selects the optimizer's special first momentum update;
    otherwise the supplied ``previous`` is the prior measurement estimator.
    Input dtypes determine execution precision; callers configure JAX x64
    before invoking this helper.  The byte cap covers only reference creation,
    not JAX execution.  A skipped reference is never reported as validated.
    No arrays are phase-aligned and no fresh observations or RNG batches are
    generated.  ``selected_bank`` optionally reuses the exact host bank from
    the caller's scan instead of repeating its rank-one eigendecomposition.
    Reports contain only finite JSON numbers, booleans and nulls.
    """
    original_measurements = np.asarray(local_measurements)
    factor, previous, symbols, observations, local_measurements, local_basis = _validated_inputs(
        factor, previous, symbols, observations, local_measurements, local_basis,
        rho, beta, gamma, tau, iteration, repeats, rtol, atol, max_reference_bytes,
    )
    if measurement_backend not in {"auto", "tensor", "rank-one"}:
        raise ValueError("measurement_backend must be auto, tensor or rank-one.")
    # Keep rank-one verification identical to the runner's host setup.  Hash
    # both the recovered vectors and the precision-cast bank actually uploaded.
    vectors = None
    if selected_bank is None:
        vectors = rank_one_measurement_vectors(original_measurements)
        if measurement_backend == "rank-one" and vectors is None:
            raise ValueError("rank-one backend requires a verified rank-one PSD bank.")
        rank_one = vectors is not None and measurement_backend != "tensor"
        bank = np.asarray(vectors if rank_one else local_measurements, dtype=factor.dtype)
    else:
        bank = np.asarray(selected_bank, dtype=factor.dtype)
        rank_one = bank.shape == (24, 4) if measurement_backend == "auto" else measurement_backend == "rank-one"
        if bank.shape != ((24, 4) if rank_one else (24, 4, 4)) or not np.all(np.isfinite(bank)):
            raise ValueError("selected_bank must be finite and match the selected backend's shape.")
    real_dtype = factor.real.dtype
    schedules = np.asarray([rho, beta, gamma], dtype=real_dtype)
    if not np.all(np.isfinite(schedules)) or schedules[1] <= 0:
        raise ValueError("schedules must remain finite with positive beta in the execution precision.")
    rho, beta, gamma = (float(value) for value in schedules)
    selected_gradient = rank_one_measurement_loss_and_gradient if rank_one else measurement_loss_and_gradient

    @jax.jit
    def probe(u, old, batch, targets, sensing, basis, weights):
        weight, smoothing, step_size = weights
        loss, gradient = selected_gradient(u, batch, targets, sensing, xp=jnp)
        tp_loss, tp_gradient = trace_preserving_loss_and_gradient(u, basis, xp=jnp)
        momentum = gradient if iteration == 0 else (1.0 - weight) * old + weight * gradient
        moreau = tp_gradient / smoothing
        combined = momentum + moreau
        atom = _lmo(jnp, combined, float(tau))
        return {
            "measurement_loss": loss, "measurement_gradient": gradient,
            "tp_loss": tp_loss, "tp_gradient": tp_gradient,
            "moreau_gradient": moreau, "momentum_gradient": momentum,
            "combined_gradient": combined, "lmo_atom": atom,
            "next_factor": (1.0 - step_size) * u + step_size * atom,
            "estimated_gap": jnp.vdot(combined, u - atom).real,
        }

    host_inputs = (factor, previous, symbols, observations, bank, local_basis, schedules)
    with jax.default_device(device):
        device_inputs = tuple(jax.device_put(value, device) for value in host_inputs)
        jax.block_until_ready(device_inputs)
        if np.dtype(device_inputs[0].dtype) != factor.dtype:
            raise ValueError("JAX changed the requested factor precision; configure x64 before probes.")
        started = perf_counter()
        executable = probe.lower(*device_inputs).compile()
        compile_seconds = perf_counter() - started

        allocation = _reference_allocation_estimate(factor.shape[0], factor.shape[1], len(symbols))
        reference = None
        reference_report = {
            "status": "skipped", "validated": False,
            "estimated_allocation_bytes": allocation,
            "max_reference_bytes": int(max_reference_bytes),
            "reason": "conservative dense-reference allocation estimate exceeds byte cap",
            "source": "Explicit minibatch Kronecker matrices and global Pauli/B tensors; NumPy QPTMeasurementObjective",
            "scope": "Selected minibatch only; not an independent revalidation of HDF5 row decoding/conversion",
        }
        if allocation <= max_reference_bytes:
            started = perf_counter()
            reference = _dense_reference(
                factor, previous, symbols, observations, local_measurements,
                local_basis, rho, beta, gamma, float(tau), iteration,
            )
            reference_report.update({
                "status": "pending", "reason": None,
                "seconds": perf_counter() - started,
                "outputs": {name: _array_summary(value) for name, value in reference.items()},
            })

        first = None
        rows = []
        for repeat in range(repeats):
            started = perf_counter()
            output = executable(*device_inputs)
            jax.block_until_ready(output)
            seconds = perf_counter() - started
            actual = {name: np.asarray(value) for name, value in jax.device_get(output).items()}
            if first is None:
                first = actual
            rows.append({
                "repeat": repeat, "synchronized_execution_seconds": seconds,
                "outputs": {name: _array_summary(value) for name, value in actual.items()},
                "versus_first": _compare_outputs(actual, first, rtol=rtol, atol=atol),
                "versus_cpu": None if reference is None else _compare_outputs(actual, reference, rtol=rtol, atol=atol),
            })

    if reference is not None:
        matches = all(item["allclose"] for row in rows for item in row["versus_cpu"].values())
        reference_report.update({"status": "validated" if matches else "mismatch", "validated": matches})
    return {
        "measurement_backend": "rank-one" if rank_one else "tensor",
        "iteration": int(iteration), "rho": rho, "beta": beta, "gamma": gamma,
        "tau": float(tau), "rtol": float(rtol), "atol": float(atol),
        "compile_seconds": compile_seconds, "compiled_executables": 1,
        "inputs": {name: _array_summary(value) for name, value in zip(
            ("factor", "previous", "symbols", "observations", "selected_bank", "local_basis", "schedules"), host_inputs)},
        "local_measurements_sha256": _digest(local_measurements),
        "derived_rank_one_bank_sha256": None if vectors is None else _digest(vectors),
        "selected_bank_sha256": _digest(bank),
        "selected_bank_source": "supplied by caller" if selected_bank is not None else "derived by probe",
        "cpu_reference": reference_report, "runs": rows,
        "replay_exact": all(item["exact"] for row in rows for item in row["versus_first"].values()),
        "replay_allclose": all(item["allclose"] for row in rows for item in row["versus_first"].values()),
        "all_outputs_finite": all(item["finite"] for row in rows for item in row["outputs"].values()),
        "interpretation": "Fixed-factor probe uses the production operator helpers but is a separately compiled graph, not the optimizer scan. Agreement does not establish long-trajectory equality; raw factors/atoms are never phase-aligned.",
    }
