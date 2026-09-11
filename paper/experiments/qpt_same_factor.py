"""CPU-only full-data QPT evaluation at one fixed complex128 factor.

The dense reference constructs only one minibatch of sensing matrices at a
time.  The trace-preserving reference uses an explicit global Pauli/B tensor;
the structured path uses its production NumPy operators.  No optimization,
sampling, target regeneration, JAX import, or GPU initialization occurs.
"""

from __future__ import annotations

import itertools

import numpy as np

from .qpt_structured_data import StructuredQPTData
from .qpt_structured_operators import (
    measurement_loss_and_gradient, rank_one_measurement_loss_and_gradient,
    rank_one_measurement_vectors, trace_preserving_loss_and_gradient,
    trace_preserving_residual,
)
from .quantum_process_tomography import (
    QPTData, QPTMeasurementObjective, create_operator_norm_factor_lmo,
    pack_factor, unpack_factor,
)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or not np.isrealobj(value):
        raise ValueError(f"{name} must be a positive finite real scalar.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar.")
    return value


def _reference_bytes(dimension, rank, batch_size, n_qubits):
    # Four times the major complex128 arrays conservatively covers reference
    # constructors, conjugation, selected-row gathers, einsum temporaries,
    # structured batch intermediates, and retained gradients.  This is not an
    # OS RSS cap and excludes the caller-owned stored observation vector.
    complex_elements = (dimension**3 + 2 * batch_size * dimension**2
                        + 8 * dimension**2 + 4 * batch_size * dimension * rank
                        + 32 * dimension * rank + 8 * batch_size)
    return int(4 * (16 * complex_elements + 8 * batch_size * n_qubits))


def reference_allocation_estimate(dimension, rank, batch_size):
    """Conservative evaluator working-allocation estimate, excluding source data.

    ``dimension`` is the process dimension ``4**n``.  Callers may use this
    before loading factor-derived geometry or starting full-data evaluation.
    The estimate is not an OS-enforced peak-memory limit.
    """
    dimension = _positive_integer(dimension, "dimension")
    rank = _positive_integer(rank, "rank")
    batch_size = _positive_integer(batch_size, "batch_size")
    rest, n_qubits = dimension, 0
    while rest > 1 and rest % 4 == 0:
        rest //= 4
        n_qubits += 1
    if rest != 1 or n_qubits < 1:
        raise ValueError("dimension must be a positive power of four, at least four.")
    return _reference_bytes(dimension, rank, batch_size, n_qubits)


def _dense_row_symbols(start, stop, n_qubits):
    """Independent scalar mixed-radix decoder for the legacy measurement rows.

    Input and axis digits advance least-significant first; outcome bits
    advance most-significant first.  Do not delegate to the structured data
    decoder: disagreement in that adapter should be visible in this reference.
    """
    result = np.empty((stop - start, n_qubits), dtype=np.int64)
    for offset, index in enumerate(range(start, stop)):
        settings, outcomes = divmod(index, 2**n_qubits)
        inputs, axes = divmod(settings, 3**n_qubits)
        for qubit in range(n_qubits):
            inputs, input_digit = divmod(inputs, 4)
            axes, axis_digit = divmod(axes, 3)
            outcome_digit = (outcomes >> (n_qubits - qubit - 1)) & 1
            result[offset, qubit] = 6 * input_digit + 2 * axis_digit + outcome_digit
    return result


def _kron_product(matrices):
    result = np.ones((1, 1), dtype=np.complex128)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def _global_reference_basis(local_basis, n_qubits):
    dimension, d = 4**n_qubits, 2**n_qubits
    basis = np.empty((dimension, d, d), dtype=np.complex128)
    for index, symbols in enumerate(itertools.product(range(4), repeat=n_qubits)):
        basis[index] = _kron_product(local_basis[list(symbols)])
    b_tensors = np.einsum("mki,nkj->nmij", basis.conj(), basis, optimize=True)
    return basis, b_tensors


def _structured_lmo(gradient, tau):
    norm = np.linalg.norm(gradient)
    if norm == 0:
        return np.zeros_like(gradient)
    if gradient.shape[1] == 1:
        return -tau * gradient / max(norm, np.finfo(gradient.real.dtype).tiny)
    left, _, right = np.linalg.svd(gradient, full_matrices=False)
    return -tau * (left @ right)


def _finish(factor, measurement_loss, measurement_gradient, residual, tp_loss,
            tp_gradient, beta, tau, *, dense):
    moreau_gradient = tp_gradient / beta
    full_gradient = measurement_gradient + moreau_gradient
    if dense:
        atom = unpack_factor(
            create_operator_norm_factor_lmo(*factor.shape, tau)(pack_factor(full_gradient)),
            *factor.shape,
        )
    else:
        atom = _structured_lmo(full_gradient, tau)
    operator_norm = float(np.linalg.norm(factor, ord=2))
    result = {
        "measurement_loss": float(measurement_loss),
        "measurement_gradient": measurement_gradient,
        "tp_residual": residual,
        "tp_violation": float(np.linalg.norm(residual, ord="fro")),
        "tp_loss": float(tp_loss),
        "tp_gradient": tp_gradient,
        "moreau_gradient": moreau_gradient,
        "full_gradient": full_gradient,
        "smoothed_objective": float(measurement_loss + tp_loss / beta),
        "lmo_atom": atom,
        "smoothed_gap": float(np.vdot(full_gradient, factor - atom).real),
        "factor_operator_norm": operator_norm,
        "feasibility_violation": max(0.0, operator_norm - tau),
    }
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise FloatingPointError("Nonfinite full-data same-factor evaluation.")
    return result


def evaluate_same_factor(data, factor, beta, tau, *, batch_size=32,
                         max_reference_bytes=128 * 2**20,
                         measurement_backend="tensor"):
    """Return independent dense/structured full-data evaluations at ``factor``.

    Every stored observation is evaluated once in legacy row order.  Chunk
    losses and gradients are weighted by their actual number of observations,
    including a short final chunk.  The returned ``full_gradient`` contains no
    stochastic momentum: it is the gradient of ``measurement_loss +
    tp_loss/beta``, where ``tp_loss = ||T(UU^H)-I||_F**2/2``.

    The allocation guard is checked before global reference construction.
    Passing an explicitly larger guard permits correspondingly larger CPU
    work; ``dense_measurement_entries`` reports its unavoidable ``m*N*N``
    streamed-entry count.  No global ``m*N*N`` array is ever allocated.
    """
    if not isinstance(data, StructuredQPTData):
        raise TypeError("data must be StructuredQPTData.")
    if data.observation_mode != "stored" or data.observations is None:
        raise ValueError("Same-factor evaluation requires original stored observations.")
    factor = np.asarray(factor)
    if factor.dtype != np.dtype(np.complex128):
        raise TypeError("factor must have dtype complex128; this evaluator is explicitly CPU precision64.")
    if factor.ndim != 2 or factor.shape[0] != data.process_dimension or factor.shape[1] < 1:
        raise ValueError("factor must have nonempty shape (4**n_qubits, rank).")
    if not np.all(np.isfinite(factor)):
        raise ValueError("factor must contain finite values.")
    beta, tau = _positive_scalar(beta, "beta"), _positive_scalar(tau, "tau")
    batch_size = _positive_integer(batch_size, "batch_size")
    max_reference_bytes = _positive_integer(max_reference_bytes, "max_reference_bytes")
    if measurement_backend not in {"tensor", "rank-one"}:
        raise ValueError("measurement_backend must be tensor or rank-one.")
    if data.m > np.iinfo(np.int64).max:
        raise ValueError("Full legacy-row enumeration exceeds int64 capacity.")
    # Validate borrowed mutable data as well as the immutable shape contract.
    if np.shape(data.observations) != (data.m,) or np.iscomplexobj(data.observations):
        raise ValueError("stored observations must be a real vector in full legacy row order.")
    if np.shape(data.local_measurements) != (24, 4, 4) or np.shape(data.local_basis) != (4, 2, 2):
        raise ValueError("local measurement bank and basis have invalid shapes.")
    if any(not np.all(np.isfinite(value)) for value in (data.local_measurements, data.local_basis)):
        raise ValueError("local measurement bank and basis must be finite.")
    if not np.allclose(data.local_measurements, data.local_measurements.conj().swapaxes(1, 2),
                       rtol=1e-10, atol=1e-12):
        raise ValueError("local measurements must be Hermitian.")
    batch_size = min(batch_size, data.m)
    estimate = reference_allocation_estimate(data.process_dimension, factor.shape[1], batch_size)
    if estimate > max_reference_bytes:
        raise ValueError(f"Dense reference estimate {estimate} bytes exceeds max_reference_bytes={max_reference_bytes}; no reference arrays constructed.")
    bank = data.local_measurements
    structured_gradient = measurement_loss_and_gradient
    if measurement_backend == "rank-one":
        bank = rank_one_measurement_vectors(data.local_measurements)
        if bank is None:
            raise ValueError("rank-one backend requires a verified rank-one PSD measurement bank.")
        structured_gradient = rank_one_measurement_loss_and_gradient

    basis, b_tensors = _global_reference_basis(data.local_basis, data.n_qubits)
    packed_factor = pack_factor(factor)
    dense_loss = structured_loss = 0.0
    dense_gradient = np.zeros_like(factor)
    tensor_gradient = np.zeros_like(factor)
    batches = 0
    # Floating-point overflow invalidates an evaluation; it must never turn
    # into an apparently successful comparison or non-standard JSON output.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for start in range(0, data.m, batch_size):
            stop = min(start + batch_size, data.m)
            count = stop - start
            symbols = data.indices_to_symbols(np.arange(start, stop, dtype=np.int64))
            dense_symbols = _dense_row_symbols(start, stop, data.n_qubits)
            observations = data.observations[start:stop]
            if not np.all(np.isfinite(observations)):
                raise ValueError("stored observations must be finite.")
            dense_rows = np.empty((count, data.process_dimension, data.process_dimension), dtype=np.complex128)
            for index, row in enumerate(dense_symbols):
                dense_rows[index] = _kron_product(data.local_measurements[row])
            objective = QPTMeasurementObjective(QPTData(
                observations, dense_rows, basis, B_tensors=b_tensors,
            ), rank=factor.shape[1], batch_size=count)
            loss, gradient = objective.loss_and_gradient(packed_factor)
            dense_loss += count * loss
            dense_gradient += count * unpack_factor(gradient, *factor.shape)
            loss, gradient = structured_gradient(factor, symbols, observations, bank, xp=np)
            structured_loss += count * loss
            tensor_gradient += count * gradient
            batches += 1
        dense_loss /= data.m
        structured_loss /= data.m
        dense_gradient /= data.m
        tensor_gradient /= data.m
        # The final minibatch objective shares the single immutable global B
        # reference.  Its TP methods are independent of minibatch observations.
        dense_residual = objective.trace_preserving_residual(packed_factor)
        dense_tp_loss = 0.5 * np.vdot(dense_residual, dense_residual).real
        dense_tp_gradient = unpack_factor(
            objective.linear_operator_adjoint_at(packed_factor, dense_residual), *factor.shape,
        )
        structured_residual = trace_preserving_residual(factor, data.local_basis, xp=np)
        structured_tp_loss, structured_tp_gradient = trace_preserving_loss_and_gradient(factor, data.local_basis, xp=np)
        dense_result = _finish(factor, dense_loss, dense_gradient, dense_residual,
                               dense_tp_loss, dense_tp_gradient, beta, tau, dense=True)
        structured_result = _finish(factor, structured_loss, tensor_gradient, structured_residual,
                                    structured_tp_loss, structured_tp_gradient, beta, tau, dense=False)
    return {
        "dense": dense_result, "structured": structured_result,
        "m": data.m, "batches": batches,
        "reference_allocation_estimate_bytes": estimate,
        "dense_measurement_entries": int(data.m * data.process_dimension**2),
        "measurement_backend": measurement_backend,
        "evaluated_measurement_backend": measurement_backend,
    }
