"""Matrix-free QPT operators shared by NumPy and JAX implementations.

``factor`` always has shape ``(4**n, rank)``.  Its first index uses the
lexicographic tensor-product ordering of the single-qubit basis.  Measurement
``symbols`` have shape ``(batch, n)`` and select one of 24 local ``4 x 4``
matrices at each tensor slot.  Decoding a dataset's flat row indices into these
symbols belongs to the dataset adapter: the QPT_BFW row ordering is *not* the
ordinary base-24 expansion of the row index.

Passing ``xp=jax.numpy`` makes every numerical operation JIT-compatible.  The
operators materialize neither process matrices nor global sensing/Pauli/TP
tensors.  The largest measurement intermediate has shape ``(batch, 4**n,
rank)``; Pauli transforms use ``rank * 4**n`` elements.
"""

import numpy as np


def _power_exponent(size, base, name):
    """Check a static tensor dimension without floating-point logarithms."""

    exponent = 0
    remaining = size
    while remaining > 1 and remaining % base == 0:
        remaining //= base
        exponent += 1
    if remaining != 1 or exponent == 0:
        raise ValueError(f"{name} must be a positive power of {base}.")
    return exponent


def _factor_qubits(factor):
    if factor.ndim != 2 or factor.shape[1] < 1:
        raise ValueError("factor must have shape (4**n, rank), with rank >= 1.")
    return _power_exponent(factor.shape[0], 4, "factor.shape[0]")


def _local_pauli_array(local_paulis, xp):
    local_paulis = xp.asarray(local_paulis)
    if local_paulis.shape != (4, 2, 2):
        raise ValueError("local_paulis must have shape (4, 2, 2).")
    return local_paulis


def pack_factor(matrix, *, xp=np):
    """Pack a complex factor/gradient as real entries then imaginary entries."""

    matrix = xp.asarray(matrix)
    return xp.concatenate((xp.real(matrix).reshape(-1), xp.imag(matrix).reshape(-1)))


def apply_measurements(factor, symbols, local_measurements, *, xp=np):
    """Return ``A_s @ factor`` for each row, using only local tensor actions.

    ``local_measurements`` has shape ``(24, 4, 4)``.  The application works for
    arbitrary local matrices; the loss-gradient helper below assumes Hermitian
    measurements.  Dataset adapters must validate that symbols are integer row
    selectors in ``[0, 24)`` before passing them to compiled JAX code.
    """

    factor = xp.asarray(factor)
    symbols = xp.asarray(symbols)
    local_measurements = xp.asarray(local_measurements)
    n_qubits = _factor_qubits(factor)
    if local_measurements.shape != (24, 4, 4):
        raise ValueError("local_measurements must have shape (24, 4, 4).")
    if symbols.ndim != 2 or symbols.shape[1] != n_qubits or symbols.shape[0] < 1:
        raise ValueError("symbols must have nonempty shape (batch, n_qubits).")
    if not np.issubdtype(symbols.dtype, np.integer):
        raise TypeError("symbols must contain integers.")
    if xp is np and (np.any(symbols < 0) or np.any(symbols >= 24)):
        raise IndexError("measurement symbols must be in [0, 24).")

    batch = symbols.shape[0]
    tensor = xp.broadcast_to(factor, (batch,) + factor.shape).reshape(
        (batch,) + (4,) * n_qubits + (factor.shape[1],)
    )
    for slot in range(n_qubits):
        # Move only the acted-on tensor index.  The ellipsis retains every
        # other index, including factor rank, without constructing a Kronecker
        # matrix or coupling different measurement rows.
        tensor = xp.moveaxis(tensor, slot + 1, -1)
        tensor = xp.einsum(
            "bij,b...j->b...i", local_measurements[symbols[:, slot]], tensor
        )
        tensor = xp.moveaxis(tensor, -1, slot + 1)
    return tensor.reshape((batch,) + factor.shape)


def measurement_values(factor, symbols, local_measurements, *, xp=np):
    """Return real ``<A_s, U U^H>`` for Hermitian sensing matrices."""

    factor = xp.asarray(factor)
    applied = apply_measurements(factor, symbols, local_measurements, xp=xp)
    return xp.real(xp.einsum("nr,bnr->b", xp.conj(factor), applied))


def measurement_loss_and_gradient(
    factor, symbols, observations, local_measurements, *, xp=np
):
    """Mean half-squared loss and its complex, real-Frobenius gradient.

    The returned gradient has the same shape as ``factor`` and satisfies
    ``d loss[direction] = real(vdot(gradient, direction))``.  ``pack_factor``
    converts it to the real coordinates used by the existing QPT runner.
    The 24 local matrices must be Hermitian, as they are for QPT measurements.
    """

    factor = xp.asarray(factor)
    observations = xp.asarray(observations)
    applied = apply_measurements(factor, symbols, local_measurements, xp=xp)
    if observations.shape != (applied.shape[0],):
        raise ValueError("observations must have shape (batch,).")
    predicted = xp.real(xp.einsum("nr,bnr->b", xp.conj(factor), applied))
    residual = predicted - observations
    loss = 0.5 * xp.mean(residual**2)
    gradient = (2.0 / applied.shape[0]) * xp.einsum("b,bnr->nr", residual, applied)
    return loss, gradient


def rank_one_measurement_vectors(local_measurements, *, rtol=1e-10, atol=1e-12):
    """Recover a verified host ``(24, 4)`` bank of rank-one sensing vectors.

    Return ``None`` when any matrix cannot be reconstructed as ``h h^H``
    within the specified tolerances.  Zero matrices are supported as zero
    vectors.  This check is performed once during dataset setup, outside JIT;
    eigendecompositions and their phase choices do not enter optimization.
    """

    bank = np.asarray(local_measurements, dtype=np.complex128)
    if bank.shape != (24, 4, 4):
        raise ValueError("local_measurements must have shape (24, 4, 4).")
    if not np.all(np.isfinite(bank)):
        raise ValueError("local_measurements must contain finite values.")
    if not np.allclose(bank, bank.conj().swapaxes(1, 2), rtol=rtol, atol=atol):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(bank)
    vectors = eigenvectors[:, :, -1] * np.sqrt(np.maximum(eigenvalues[:, -1], 0))[:, None]
    reconstructed = np.einsum("bi,bj->bij", vectors, vectors.conj())
    if not np.allclose(reconstructed, bank, rtol=rtol, atol=atol):
        return None
    return vectors


def product_measurement_vectors(symbols, local_vectors, *, xp=np):
    """Build only the selected tensor-product sensing vectors, shape ``(b,N)``.

    This costs ``O(b*N)`` arithmetic and storage, and does not construct any
    ``N x N`` matrices.  Each row is the product of its single-qubit vectors.
    """

    symbols = xp.asarray(symbols)
    local_vectors = xp.asarray(local_vectors)
    if local_vectors.shape != (24, 4):
        raise ValueError("local_vectors must have shape (24, 4).")
    if symbols.ndim != 2 or min(symbols.shape) < 1:
        raise ValueError("symbols must have nonempty shape (batch, n_qubits).")
    if not np.issubdtype(symbols.dtype, np.integer):
        raise TypeError("symbols must contain integers.")
    if xp is np and (np.any(symbols < 0) or np.any(symbols >= 24)):
        raise IndexError("measurement symbols must be in [0, 24).")
    product = local_vectors[symbols[:, 0]]
    for slot in range(1, symbols.shape[1]):
        product = (
            product[:, :, None] * local_vectors[symbols[:, slot]][:, None, :]
        ).reshape((symbols.shape[0], -1))
    return product


def _rank_one_amplitudes(factor, symbols, local_vectors, xp):
    factor = xp.asarray(factor)
    _factor_qubits(factor)
    vectors = product_measurement_vectors(symbols, local_vectors, xp=xp)
    if factor.shape[0] != vectors.shape[1]:
        raise ValueError("factor and measurement symbols have different qubit counts.")
    return vectors, xp.matmul(xp.conj(vectors), factor)


def rank_one_measurement_values(factor, symbols, local_vectors, *, xp=np):
    """Return ``sum_a |h_s^H U[:,a]|**2`` for a verified rank-one bank."""

    _, amplitudes = _rank_one_amplitudes(factor, symbols, local_vectors, xp)
    return xp.sum(xp.real(xp.conj(amplitudes) * amplitudes), axis=1)


def rank_one_measurement_loss_and_gradient(
    factor, symbols, observations, local_vectors, *, xp=np
):
    """Rank-one measurement loss and complex real-Frobenius gradient.

    With ``H[s,:] = h_s``, the gradient is
    ``(2/b) H.T @ (residual[:,None] * (H.conj() @ U))``.  The selected sensing
    vectors occupy ``b*N`` elements, independent of the factor rank.  Applying
    them and evaluating the gradient costs ``O(b*N*rank)`` arithmetic.
    """

    observations = xp.asarray(observations)
    vectors, amplitudes = _rank_one_amplitudes(factor, symbols, local_vectors, xp)
    if observations.shape != (vectors.shape[0],):
        raise ValueError("observations must have shape (batch,).")
    predicted = xp.sum(xp.real(xp.conj(amplitudes) * amplitudes), axis=1)
    residual = predicted - observations
    loss = 0.5 * xp.mean(residual**2)
    gradient = (2.0 / vectors.shape[0]) * xp.matmul(
        vectors.T, residual[:, None] * amplitudes
    )
    return loss, gradient


def _transform_axis(tensor, transform, axis, xp):
    tensor = xp.moveaxis(tensor, axis, -1)
    tensor = xp.matmul(tensor, xp.swapaxes(transform, -1, -2))
    return xp.moveaxis(tensor, -1, axis)


def pauli_coefficients_to_matrices(factor, local_paulis, *, xp=np):
    """Return Kraus matrices ``K[a] = sum_k factor[k,a] P_k``.

    The local basis need not be Hermitian: QPT_BFW uses a real, antisymmetric
    Y basis element.  This transform is linear in ``factor``, with no conjugate
    on its coefficients.  Only the local ``(4, 2, 2)`` basis is stored.
    """

    factor = xp.asarray(factor)
    n_qubits = _factor_qubits(factor)
    local_paulis = _local_pauli_array(local_paulis, xp)
    rank = factor.shape[1]
    tensor = factor.reshape((4,) * n_qubits + (rank,))
    transform = local_paulis.reshape(4, 4).T
    for slot in range(n_qubits):
        tensor = _transform_axis(tensor, transform, slot, xp)
    tensor = tensor.reshape((2, 2) * n_qubits + (rank,))
    # The local output indices start as (row_0, col_0, row_1, col_1, ...).
    # Global matrices gather all row bits first, followed by all column bits.
    permutation = (2 * n_qubits,) + tuple(range(0, 2 * n_qubits, 2)) + tuple(
        range(1, 2 * n_qubits, 2)
    )
    return xp.transpose(tensor, permutation).reshape((rank, 2**n_qubits, 2**n_qubits))


def pauli_matrices_to_coefficients(matrices, local_paulis, *, xp=np):
    """Hermitian adjoint of :func:`pauli_coefficients_to_matrices`.

    This is an inverse only when the local basis is orthonormal.  General
    complex bases are supported so that differentiation never assumes the
    basis elements themselves are Hermitian.
    """

    matrices = xp.asarray(matrices)
    local_paulis = _local_pauli_array(local_paulis, xp)
    if matrices.ndim != 3 or matrices.shape[0] < 1 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("matrices must have shape (rank, 2**n, 2**n).")
    rank, d, _ = matrices.shape
    n_qubits = _power_exponent(d, 2, "matrices.shape[1]")
    tensor = matrices.reshape((rank,) + (2,) * (2 * n_qubits))
    permutation = tuple(
        index
        for slot in range(n_qubits)
        for index in (1 + slot, 1 + n_qubits + slot)
    ) + (0,)
    tensor = xp.transpose(tensor, permutation).reshape((4,) * n_qubits + (rank,))
    transform = xp.conj(local_paulis.reshape(4, 4))
    for slot in range(n_qubits):
        tensor = _transform_axis(tensor, transform, slot, xp)
    return tensor.reshape((4**n_qubits, rank))


def trace_preserving_map(factor, local_paulis, *, xp=np):
    """Return ``T(U U^H) = sum_a K[a]^H K[a]`` exactly.

    This follows the existing code's convention ``B[n,m] = P_m^H P_n``.
    It requires ``O(rank * d**3)`` arithmetic and ``O(rank * d**2)`` storage,
    with no global ``B`` tensor and no ``(d**2, d**2)`` process matrix.
    """

    kraus = pauli_coefficients_to_matrices(factor, local_paulis, xp=xp)
    return xp.einsum("rai,raj->ij", xp.conj(kraus), kraus)


def trace_preserving_residual(factor, local_paulis, *, xp=np):
    """Return the exact trace-preserving map minus the identity."""

    mapped = trace_preserving_map(factor, local_paulis, xp=xp)
    return mapped - xp.eye(mapped.shape[0], dtype=mapped.dtype)


def trace_preserving_jacobian_adjoint(factor, matrix, local_paulis, *, xp=np):
    """Return the real-Frobenius Jacobian adjoint at ``factor``.

    For an arbitrary complex ``matrix``, this is the Pauli transform adjoint
    of ``K[a] @ (matrix + matrix^H)``.  Passing the TP residual and dividing by
    ``beta`` gives the gradient of ``||T(UU^H)-I||_F**2 / (2*beta)``, matching
    the existing FRAMES Moreau envelope (including its factor of one half).
    """

    matrix = xp.asarray(matrix)
    kraus = pauli_coefficients_to_matrices(factor, local_paulis, xp=xp)
    if matrix.shape != kraus.shape[1:]:
        raise ValueError("matrix must have shape (2**n, 2**n).")
    kraus_gradient = xp.matmul(kraus, matrix + xp.conj(matrix.T))
    return pauli_matrices_to_coefficients(kraus_gradient, local_paulis, xp=xp)


def trace_preserving_loss_and_gradient(factor, local_paulis, *, xp=np):
    """Return ``0.5 * ||T(UU^H)-I||_F**2`` and its complex gradient.

    This combined path reuses the Kraus matrices for the map and Jacobian
    adjoint, requiring only one forward and one adjoint Pauli transform.
    Divide the returned loss and gradient by ``beta`` for the Moreau penalty.
    """

    kraus = pauli_coefficients_to_matrices(factor, local_paulis, xp=xp)
    mapped = xp.einsum("rai,raj->ij", xp.conj(kraus), kraus)
    residual = mapped - xp.eye(mapped.shape[0], dtype=mapped.dtype)
    loss = 0.5 * xp.real(xp.vdot(residual, residual))
    kraus_gradient = xp.matmul(kraus, residual + xp.conj(residual.T))
    gradient = pauli_matrices_to_coefficients(kraus_gradient, local_paulis, xp=xp)
    return loss, gradient
