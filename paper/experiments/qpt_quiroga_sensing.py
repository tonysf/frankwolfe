"""Quiroga-style QPT sensing: d**2 global probes and a 2d-outcome POVM.

This backend is separate from, and leaves unchanged, the QPT_BFW local-Pauli
benchmark (``qpt_structured_*``: 4**n product inputs, 3**n Pauli settings and
2**n outcomes, 24**n rows).  It implements the reduced design of Quiroga and
Kyrillidis (arXiv:2312.01311, Sec. III): d**2 generic global probe states and
one 2d-outcome POVM that is informationally complete for pure states, giving
2*d**3 = 2*8**n rows.  That paper leaves the POVM constants unreported, so the
conventions below are explicit choices; docs/qpt_quiroga_sensing.md explains
them and compares the two benchmarks.

Choi coordinates.  A rank-r factor ``U`` has shape ``(d**2, r)``; column ``a``
is the Choi vector |K_a>> = sum_i |i> (x) K_a|i>, so ``U[i*d + o, a] =
K_a[o, i]`` with the input index ``i`` major.  Then J = U U^H =
sum_ij |i><j| (x) E(|i><j|) is the Choi matrix of Baldwin, Kalev and Deutsch
[PRA 90, 012110 (2014), Eq. (7)], with Tr J = d for a trace-preserving map.
Row ``s = 2*d*p + j`` senses the Born probability

    f_s(U) = Tr(D_s^H J) = Tr(E_j E(rho_p)),   D_s^H = D_s = rho_p^T (x) E_j.

Probes ``p`` (``probe_table``): ``|k>`` for k < d, then ``(|k> + |l>)/sqrt(2)``
and then ``(|k> + i|l>)/sqrt(2)``, each over k < l in lexicographic order.

Outcomes ``j`` (Baldwin et al. Eq. (18)): j = 0 is ``a|0><0|``; j = m in
[1, d) is ``b(I + |0><m| + |m><0|)``; j = d - 1 + m is
``b(I + i|0><m| - i|m><0|)``; j = 2d - 1 is the throw-away element
``I - sum(others)``.  By default ``a = b = 2 / (4d - 1 + sqrt(8d - 7))``, the
largest common value for which the throw-away element satisfies ``E >= b I``.

Every element is ``alpha I + beta|0><0| + |0><u| + |u><0|``, so all 2d
outcomes of one probe cost O(r d).  Predictions and factor gradients cost
O(r d) per sampled row and O(r d**3) for all rows; no D_s, J or other
d**2-by-d**2 matrix is formed.  Pass ``xp=jax.numpy`` for JIT-compatible
evaluation; NumPy inputs are validated, JAX inputs are not.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import json
import math
import platform
from time import perf_counter
from typing import Optional

import numpy as np

from .qpt_observation_noise import NOISE_GENERATOR, fixed_row_noise


CONVENTION = "quiroga_global_probes_2d_povm_choi_v1"
POVM_CONSTANT_RULE = "a=b=2/(4d-1+sqrt(8d-7)); lambda_min(E_2d)=b"
SHOT_GENERATOR = "pcg64_seedsequence_per_probe_multinomial_v1"
OBSERVATION_MODES = ("noiseless", "gaussian", "shots", "stored")
# Separates shot streams from any other SeedSequence use of the same seed.
_SHOT_DOMAIN = 0x51504F56
_UINT64_MAX = int(np.iinfo(np.uint64).max)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _seed(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= value <= _UINT64_MAX):
        raise ValueError(f"{name} must be an integer in [0, 2**64).")
    return int(value)


def _qubit_dimension(d):
    d = _positive_integer(d, "d")
    if d < 2 or d & (d - 1):
        raise ValueError("d must be a power of two, at least two.")
    return d


def _process_dimension(size):
    d = math.isqrt(size)
    if d * d != size:
        raise ValueError("factor must have shape (d**2, rank).")
    return _qubit_dimension(d)


def _choi_tensor(factor, xp):
    """Return ``factor`` as ``[input, output, rank]`` and ``d``."""
    factor = xp.asarray(factor)
    if factor.ndim != 2 or factor.shape[1] < 1:
        raise ValueError("factor must have shape (d**2, rank), with rank >= 1.")
    d = _process_dimension(factor.shape[0])
    return factor.reshape(d, d, factor.shape[1]), d


# ---------------------------------------------------------------------------
# POVM constants and elements


def default_povm_constants(d):
    """Return the documented ``(a, b)`` for dimension ``d``.

    With ``a = b`` the throw-away element has smallest eigenvalue
    ``1 - 2b(d-1) - b(1 + sqrt(8d-7))/2``.  Setting it equal to ``b`` gives
    ``b = 2/(4d - 1 + sqrt(8d - 7))``, which is ``1 - b`` times the largest
    common value that keeps the element positive semidefinite.
    """
    d = _qubit_dimension(d)
    value = 2.0 / (4 * d - 1 + math.sqrt(8 * d - 7))
    return value, value


def throwaway_min_eigenvalue(d, a, b):
    """Closed-form smallest eigenvalue of the throw-away element ``E_{2d}``.

    ``E_{2d} = (1 - 2b(d-1)) I - a|0><0| - b(|0><w| + |w><0|)`` with
    ``|w> = (1 - i) sum_{m>0} |m>``.  On span{|0>, |w>} the subtracted part
    has eigenvalues ``(a +- sqrt(a**2 + 8 b**2 (d-1)))/2`` and it vanishes on
    the orthogonal complement.  The other elements have spectra ``{a, 0}``
    and ``b {0, 2, 1}``.
    """
    d = _qubit_dimension(d)
    return 1.0 - 2.0 * b * (d - 1) - 0.5 * (a + math.sqrt(a * a + 8.0 * b * b * (d - 1)))


def validate_povm_constants(d, a, b):
    """Return float ``(a, b)`` if all 2d POVM elements are PSD, else raise."""
    d = _qubit_dimension(d)
    for value, name in ((a, "a"), (b, "b")):
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or not np.isrealobj(value):
            raise ValueError(f"POVM constant {name} must be a real scalar.")
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError("POVM constants a and b must be finite and positive.")
    floor = throwaway_min_eigenvalue(d, a, b)
    if floor < 0:
        raise ValueError(
            f"POVM constants a={a!r}, b={b!r} make the throw-away element "
            f"indefinite for d={d} (smallest eigenvalue {floor:.3e})."
        )
    return a, b


def povm_elements(d, a=None, b=None, *, max_dimension=64):
    """Dense ``(2d, d, d)`` elements in outcome order, for validation only."""
    d = _qubit_dimension(d)
    if d > max_dimension:
        raise ValueError(f"Dense POVM elements for d={d} exceed max_dimension={max_dimension}.")
    if a is None and b is None:
        a, b = default_povm_constants(d)
    a, b = validate_povm_constants(d, a, b)
    identity = np.eye(d, dtype=np.complex128)
    elements = np.zeros((2 * d, d, d), dtype=np.complex128)
    elements[0, 0, 0] = a
    for m in range(1, d):
        elements[m] = b * identity
        elements[m, 0, m] += b
        elements[m, m, 0] += b
        elements[d - 1 + m] = b * identity
        elements[d - 1 + m, 0, m] += 1j * b
        elements[d - 1 + m, m, 0] -= 1j * b
    elements[-1] = identity - elements[:-1].sum(axis=0)
    return elements


def _arrow_parameters(weights, a, b, xp):
    """Write ``sum_j w_j E_j = alpha I + beta|0><0| + |0><u| + |u><0|``.

    ``weights`` has shape ``(batch, 2d)`` and must be real.  Returns
    ``alpha``, ``beta`` and ``u[1:]`` (``u[0] = 0``) for every row.
    """
    d = weights.shape[1] // 2
    first, real, imag, throw = (
        weights[:, 0], weights[:, 1:d], weights[:, d:2 * d - 1], weights[:, 2 * d - 1]
    )
    alpha = b * (xp.sum(real, axis=1) + xp.sum(imag, axis=1)) + (1.0 - 2.0 * b * (d - 1)) * throw
    beta = a * (first - throw)
    arrow = b * real - 1j * b * imag - b * (1.0 - 1j) * throw[:, None]
    return alpha, beta, arrow


def _povm_values(phi, a, b, xp):
    """All 2d outcome values ``sum_r phi_r^H E_j phi_r`` for outputs ``phi``.

    ``phi`` has shape ``(batch, d, rank)``: output vectors ``K_r psi``.
    """
    d = phi.shape[1]
    norm = xp.sum(xp.real(phi * xp.conj(phi)), axis=(1, 2))
    cross = xp.sum(xp.conj(phi[:, :1, :]) * phi, axis=2)
    first = xp.real(cross[:, 0])
    rest = cross[:, 1:]
    total = xp.sum(rest, axis=1)
    throw = ((1.0 - 2.0 * b * (d - 1)) * norm - a * first
             - 2.0 * b * (xp.real(total) - xp.imag(total)))
    return xp.concatenate((
        (a * first)[:, None],
        b * (norm[:, None] + 2.0 * xp.real(rest)),
        b * (norm[:, None] - 2.0 * xp.imag(rest)),
        throw[:, None],
    ), axis=1)


def _povm_apply(phi, weights, a, b, xp):
    """Return ``(sum_j weights[:, j] E_j) phi`` for real ``weights``."""
    alpha, beta, arrow = _arrow_parameters(weights, a, b, xp)
    head = phi[:, 0, :]
    top = (alpha + beta)[:, None] * head + xp.einsum("bm,bma->ba", xp.conj(arrow), phi[:, 1:, :])
    tail = alpha[:, None, None] * phi[:, 1:, :] + arrow[:, :, None] * head[:, None, :]
    return xp.concatenate((top[:, None, :], tail), axis=1)


# ---------------------------------------------------------------------------
# Probes and row indexing


@lru_cache(maxsize=None)
def _probe_table(d):
    upper_k, upper_l = np.triu_indices(d, 1)
    basis = np.arange(d)
    count = upper_k.size
    root = 1.0 / math.sqrt(2.0)
    k = np.concatenate((basis, upper_k, upper_k)).astype(np.int64)
    l = np.concatenate((basis, upper_l, upper_l)).astype(np.int64)
    ck = np.concatenate((np.ones(d), np.full(2 * count, root))).astype(np.complex128)
    cl = np.concatenate((np.zeros(d), np.full(count, root), np.full(count, 1j * root)))
    for array in (k, l, ck, cl):
        array.setflags(write=False)
    return k, l, ck, cl


def probe_table(d):
    """Return read-only ``(k, l, ck, cl)`` with ``psi_p = ck|k> + cl|l>``.

    Computational probes use ``l = k`` and ``cl = 0``.  This is the order of
    Quiroga's list of generic states (and Baldwin et al. Eq. (9)).
    """
    return _probe_table(_qubit_dimension(d))


def probe_states(d, *, max_dimension=64):
    """Dense ``(d**2, d)`` probe vectors, for validation only."""
    d = _qubit_dimension(d)
    if d > max_dimension:
        raise ValueError(f"Dense probe states for d={d} exceed max_dimension={max_dimension}.")
    k, l, ck, cl = probe_table(d)
    states = np.zeros((d * d, d), dtype=np.complex128)
    probes = np.arange(d * d)
    states[probes, k] += ck
    states[probes, l] += cl
    return states


def choi_standard_basis(d):
    """Operator basis ``A[i*d + o] = |o><i|`` matching the Choi factor index.

    With it, ``sum_n U[n, a] A[n] = K_a``, so legacy code using
    ``T(chi) = sum_nm chi_nm A_m^H A_n`` evaluates ``sum_a K_a^H K_a``.
    """
    d = _qubit_dimension(d)
    basis = np.zeros((d * d, d, d), dtype=np.complex128)
    inputs, outputs = np.divmod(np.arange(d * d), d)
    basis[np.arange(d * d), outputs, inputs] = 1.0
    return basis


def unitary_choi_factor(unitary):
    """Return the ``(d**2, 1)`` Choi vector ``|V>>`` of ``rho -> V rho V^H``."""
    unitary = np.asarray(unitary, dtype=np.complex128)
    if unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1]:
        raise ValueError("unitary must be a square matrix.")
    _qubit_dimension(unitary.shape[0])
    return np.ascontiguousarray(unitary.T).reshape(-1, 1)


def kraus_operators(factor):
    """Return ``K[a]`` with ``K[a][o, i] = factor[i*d + o, a]``."""
    tensor, _ = _choi_tensor(np.asarray(factor), np)
    return np.transpose(tensor, (2, 1, 0))


# ---------------------------------------------------------------------------
# Trace preservation in Choi coordinates


def choi_output_partial_trace(factor, *, xp=np):
    """Return ``Tr_out(U U^H) = (sum_a K_a^H K_a)^T`` in O(r d**3).

    The TP helpers multiply the ``(d, d*r)`` input-row view of ``U`` with
    matmul, so NumPy uses BLAS instead of an unoptimized einsum loop.
    """
    tensor, d = _choi_tensor(factor, xp)
    rows = tensor.reshape(d, -1)
    return rows @ xp.conj(rows).T


def trace_preserving_residual(factor, *, xp=np):
    """Return ``Tr_out(U U^H) - I``; its Frobenius norm is ``||sum K^H K - I||``."""
    mapped = choi_output_partial_trace(factor, xp=xp)
    return mapped - xp.eye(mapped.shape[0], dtype=mapped.dtype)


def trace_preserving_jacobian_adjoint(factor, matrix, *, xp=np):
    """Real-Frobenius Jacobian adjoint of ``U -> Tr_out(U U^H)`` at ``factor``.

    For any complex ``M`` this is ``((M + M^H) (x) I) U``, applied on the
    input index.  With ``M`` the TP residual it is the gradient of
    ``||Tr_out(UU^H) - I||_F**2 / 2``.
    """
    tensor, d = _choi_tensor(factor, xp)
    matrix = xp.asarray(matrix)
    if matrix.shape != (d, d):
        raise ValueError("matrix must have shape (d, d).")
    applied = (matrix + xp.conj(matrix.T)) @ tensor.reshape(d, -1)
    return applied.reshape(d * d, tensor.shape[2])


def trace_preserving_loss_and_gradient(factor, *, xp=np):
    """Return ``0.5 ||Tr_out(UU^H) - I||_F**2`` and its exact factor gradient.

    The gradient ``2 (R (x) I) U`` also equals Quiroga's published factor term
    ``grad_chi H(UU^H) U`` for the unhalved penalty ``H = ||R||_F**2``.
    """
    tensor, d = _choi_tensor(factor, xp)
    rows = tensor.reshape(d, -1)
    residual = rows @ xp.conj(rows).T
    residual = residual - xp.eye(d, dtype=residual.dtype)
    loss = 0.5 * xp.real(xp.vdot(residual, residual))
    gradient = (residual + xp.conj(residual.T)) @ rows
    return loss, gradient.reshape(d * d, tensor.shape[2])


def pure_target_process_fidelity(factor, truth, *, xp=np):
    """Process fidelity of trace-normalized ``UU^H`` with a pure Choi target.

    For a rank-one target ``t t^H`` this is ``sum_a |t^H U_a|**2 /
    (||t||**2 ||U||_F**2)``, i.e. the fidelity of the normalized Choi states
    used by Qiskit's ``process_fidelity``.  It matches the overlap formula of
    the existing QPT runners.
    """
    factor, truth = xp.asarray(factor), xp.asarray(truth)
    if truth.ndim == 1:
        truth = truth[:, None]
    if truth.ndim != 2 or truth.shape[1] != 1 or factor.ndim != 2 or truth.shape[0] != factor.shape[0]:
        raise ValueError("truth must be one Choi vector matching the factor's first dimension.")
    overlap = xp.sum(xp.abs(xp.conj(truth.T) @ factor) ** 2)
    return overlap / (xp.sum(xp.abs(truth) ** 2) * xp.sum(xp.abs(factor) ** 2))


def _index_add(shape, index, values, xp):
    if xp is np:
        result = np.zeros(shape, dtype=values.dtype)
        np.add.at(result, index, values)
        return result
    return xp.zeros(shape, dtype=values.dtype).at[index].add(values)


# ---------------------------------------------------------------------------
# The sensing operator


@dataclass(frozen=True)
class QuirogaSensingDesign:
    """Matrix-free sensing operator ``U -> (Tr(D_s^H U U^H))_s``.

    ``povm_a``/``povm_b`` default to :func:`default_povm_constants`; custom
    values must be given together and pass :func:`validate_povm_constants`.
    ``reduction='mean'`` matches the existing FRAMES runners' mean
    half-squared loss; ``'sum'`` is Quiroga's published ``F``.
    """

    n_qubits: int
    povm_a: Optional[float] = None
    povm_b: Optional[float] = None

    def __post_init__(self):
        n_qubits = _positive_integer(self.n_qubits, "n_qubits")
        if n_qubits > 20:
            raise ValueError("Row indices 2*8**n_qubits exceed int64 above 20 qubits.")
        d = 2**n_qubits
        if (self.povm_a is None) != (self.povm_b is None):
            raise ValueError("Specify both POVM constants or neither.")
        if self.povm_a is None:
            a, b = default_povm_constants(d)
        else:
            a, b = validate_povm_constants(d, self.povm_a, self.povm_b)
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "povm_a", a)
        object.__setattr__(self, "povm_b", b)

    @property
    def d(self):
        return 2**self.n_qubits

    @property
    def process_dimension(self):
        return 4**self.n_qubits

    @property
    def probe_count(self):
        return 4**self.n_qubits

    @property
    def outcome_count(self):
        return 2 * self.d

    @property
    def row_count(self):
        return 2 * self.d**3

    def describe(self):
        """JSON-compatible record of every convention needed to rebuild rows."""
        return {
            "convention": CONVENTION, "n_qubits": self.n_qubits, "d": self.d,
            "probes": self.probe_count, "outcomes": self.outcome_count, "rows": self.row_count,
            "row_index": "s = 2*d*probe + outcome",
            "probe_order": "|k>; (|k>+|l>)/sqrt2 for k<l; (|k>+i|l>)/sqrt2 for k<l (lexicographic)",
            "outcome_order": "a|0><0|; b(I+|0><m|+|m><0|); b(I+i|0><m|-i|m><0|) for m=1..d-1; I-sum",
            "povm_a": self.povm_a, "povm_b": self.povm_b,
            "povm_constant_rule": POVM_CONSTANT_RULE,
            "throwaway_min_eigenvalue": throwaway_min_eigenvalue(self.d, self.povm_a, self.povm_b),
            "choi_convention": "U[i*d+o,a]=K_a[o,i]; J=sum_ij |i><j| (x) E(|i><j|); D_s^H=rho^T (x) E",
        }

    def validate_rows(self, rows):
        """Return host rows as nonempty one-dimensional int64 indices."""
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.size == 0 or rows.dtype.kind not in "iu":
            raise ValueError("rows must be a nonempty one-dimensional integer array.")
        if np.any(rows < 0) or np.any(rows >= self.row_count):
            raise IndexError("sensing row is out of range.")
        return rows.astype(np.int64, copy=False)

    def decode_rows(self, rows, *, xp=np):
        """Return ``(probe, outcome)`` for rows ``s = 2*d*probe + outcome``."""
        if xp is np:
            rows = self.validate_rows(rows)
        rows = xp.asarray(rows)
        return rows // self.outcome_count, rows % self.outcome_count

    def encode_rows(self, probes, outcomes):
        probes, outcomes = np.asarray(probes), np.asarray(outcomes)
        if probes.dtype.kind not in "iu" or outcomes.dtype.kind not in "iu" or probes.shape != outcomes.shape:
            raise ValueError("probes and outcomes must be integer arrays of equal shape.")
        if (np.any(probes < 0) or np.any(probes >= self.probe_count)
                or np.any(outcomes < 0) or np.any(outcomes >= self.outcome_count)):
            raise IndexError("probe or outcome is out of range.")
        return probes.astype(np.int64) * self.outcome_count + outcomes.astype(np.int64)

    def _tensor(self, factor, xp):
        tensor, d = _choi_tensor(factor, xp)
        if d != self.d:
            raise ValueError(f"factor must have {self.process_dimension} rows for {self.n_qubits} qubits.")
        return tensor

    def _outputs(self, tensor, probes, xp):
        k, l, ck, cl = (xp.asarray(item)[probes] for item in probe_table(self.d))
        phi = ck[:, None, None] * tensor[k] + cl[:, None, None] * tensor[l]
        return phi, (k, l, ck, cl)

    def _scatter(self, applied, structure, xp):
        k, l, ck, cl = structure
        values = xp.concatenate((xp.conj(ck)[:, None, None] * applied,
                                 xp.conj(cl)[:, None, None] * applied))
        return _index_add((self.d,) + applied.shape[1:], xp.concatenate((k, l)), values, xp)

    def _chunks(self, probe_chunk, rank):
        if probe_chunk is None:
            probe_chunk = max(1, (1 << 22) // (self.d * rank))
        probe_chunk = _positive_integer(probe_chunk, "probe_chunk")
        for start in range(0, self.probe_count, probe_chunk):
            yield start, min(start + probe_chunk, self.probe_count)

    def _weights(self, weights, shape, xp):
        if xp is np:
            weights = np.asarray(weights)
            if weights.shape != shape or not np.isrealobj(weights) or not np.all(np.isfinite(weights)):
                raise ValueError(f"weights/targets must be finite real values of shape {shape}.")
        return xp.asarray(weights)

    # Sampled rows: O(r d) each.

    def probe_values(self, factor, probes, *, xp=np):
        """All ``2d`` outcome values of the given probes, shape ``(P, 2d)``."""
        if xp is np:
            probes = np.asarray(probes)
            if probes.ndim != 1 or probes.dtype.kind not in "iu" or np.any(probes < 0) or np.any(probes >= self.probe_count):
                raise ValueError("probes must be one-dimensional integers in range.")
        phi, _ = self._outputs(self._tensor(factor, xp), xp.asarray(probes), xp)
        return _povm_values(phi, self.povm_a, self.povm_b, xp)

    def values(self, factor, rows, *, xp=np):
        """Return ``Tr(D_s^H U U^H)`` for the requested rows."""
        probes, outcomes = self.decode_rows(rows, xp=xp)
        phi, _ = self._outputs(self._tensor(factor, xp), probes, xp)
        values = _povm_values(phi, self.povm_a, self.povm_b, xp)
        return xp.take_along_axis(values, outcomes[:, None], axis=1)[:, 0]

    def adjoint(self, matrix, rows, weights, *, xp=np):
        """Return ``sum_s weights[s] D_s @ matrix`` for real row weights."""
        probes, outcomes = self.decode_rows(rows, xp=xp)
        tensor = self._tensor(matrix, xp)
        weights = self._weights(weights, (probes.shape[0],), xp)
        phi, structure = self._outputs(tensor, probes, xp)
        selector = outcomes[:, None] == xp.arange(self.outcome_count)[None, :]
        applied = _povm_apply(phi, xp.where(selector, weights[:, None], 0.0), self.povm_a, self.povm_b, xp)
        return self._scatter(applied, structure, xp).reshape(self.process_dimension, tensor.shape[2])

    def loss_and_gradient(self, factor, rows, targets, *, reduction="mean", xp=np):
        """Half-squared loss over rows and its real-Frobenius factor gradient.

        ``d loss[V] = real(vdot(gradient, V))``; the gradient is
        ``2 w sum_s r_s D_s U`` with ``w = 1/len(rows)`` (mean) or 1 (sum).
        """
        probes, outcomes = self.decode_rows(rows, xp=xp)
        scale = self._scale(reduction, 1.0 / probes.shape[0])
        tensor = self._tensor(factor, xp)
        targets = self._weights(targets, (probes.shape[0],), xp)
        phi, structure = self._outputs(tensor, probes, xp)
        values = _povm_values(phi, self.povm_a, self.povm_b, xp)
        residual = xp.take_along_axis(values, outcomes[:, None], axis=1)[:, 0] - targets
        selector = outcomes[:, None] == xp.arange(self.outcome_count)[None, :]
        weights = xp.where(selector, (2.0 * scale * residual)[:, None], 0.0)
        applied = _povm_apply(phi, weights, self.povm_a, self.povm_b, xp)
        gradient = self._scatter(applied, structure, xp).reshape(self.process_dimension, tensor.shape[2])
        return 0.5 * scale * xp.sum(residual**2), gradient

    # All rows, grouped by probe: O(r d**3) with bounded probe chunks.

    def full_values(self, factor, *, xp=np, probe_chunk=None):
        """All ``2 d**3`` values in row order."""
        tensor = self._tensor(factor, xp)
        blocks = []
        for start, stop in self._chunks(probe_chunk, tensor.shape[2]):
            phi, _ = self._outputs(tensor, xp.arange(start, stop), xp)
            blocks.append(_povm_values(phi, self.povm_a, self.povm_b, xp).reshape(-1))
        return xp.concatenate(blocks)

    def full_adjoint(self, matrix, weights, *, xp=np, probe_chunk=None):
        """Return ``sum_s weights[s] D_s @ matrix`` over all rows."""
        tensor = self._tensor(matrix, xp)
        weights = self._weights(weights, (self.row_count,), xp).reshape(self.probe_count, self.outcome_count)
        result = xp.zeros(tensor.shape, dtype=tensor.dtype)
        for start, stop in self._chunks(probe_chunk, tensor.shape[2]):
            phi, structure = self._outputs(tensor, xp.arange(start, stop), xp)
            applied = _povm_apply(phi, weights[start:stop], self.povm_a, self.povm_b, xp)
            result = result + self._scatter(applied, structure, xp)
        return result.reshape(self.process_dimension, tensor.shape[2])

    def full_loss_and_gradient(self, factor, targets, *, reduction="mean", xp=np, probe_chunk=None):
        """Loss and gradient over every row, without a per-row pass."""
        scale = self._scale(reduction, 1.0 / self.row_count)
        tensor = self._tensor(factor, xp)
        targets = self._weights(targets, (self.row_count,), xp).reshape(self.probe_count, self.outcome_count)
        loss = 0.0
        gradient = xp.zeros(tensor.shape, dtype=tensor.dtype)
        for start, stop in self._chunks(probe_chunk, tensor.shape[2]):
            phi, structure = self._outputs(tensor, xp.arange(start, stop), xp)
            residual = _povm_values(phi, self.povm_a, self.povm_b, xp) - targets[start:stop]
            loss = loss + xp.sum(residual**2)
            applied = _povm_apply(phi, 2.0 * scale * residual, self.povm_a, self.povm_b, xp)
            gradient = gradient + self._scatter(applied, structure, xp)
        return 0.5 * scale * loss, gradient.reshape(self.process_dimension, tensor.shape[2])

    @staticmethod
    def _scale(reduction, mean_scale):
        if reduction == "mean":
            return mean_scale
        if reduction == "sum":
            return 1.0
        raise ValueError("reduction must be 'mean' or 'sum'.")

    # Dense references for small systems only.

    def dense_sensing_matrices(self, rows=None, *, max_entries=2**26):
        """Return ``D_s = rho_p^T (x) E_j`` with shape ``(rows, d**2, d**2)``."""
        rows = np.arange(self.row_count, dtype=np.int64) if rows is None else self.validate_rows(rows)
        if rows.size * self.process_dimension**2 > max_entries:
            raise ValueError("Dense sensing matrices exceed max_entries; use the matrix-free operators.")
        probes, outcomes = self.decode_rows(rows)
        states = probe_states(self.d)
        transposed = np.einsum("pi,pj->pij", states.conj(), states)
        elements = povm_elements(self.d, self.povm_a, self.povm_b)
        dense = np.einsum("bij,bxy->bixjy", transposed[probes], elements[outcomes])
        return dense.reshape(rows.size, self.process_dimension, self.process_dimension)

    def dense_adjoint(self, weights, *, max_process_dimension=1024):
        """Dense Hermitian ``A^H(w) = sum_s w_s D_s`` of shape ``(d**2, d**2)``.

        This is the matrix whose spectral norm Quiroga's adaptive step needs.
        It is built from ``sum_p rho_p^T (x) W_p`` in O(d**4) work, but still
        occupies ``16 d**4`` bytes, so it is refused above
        ``max_process_dimension`` (default ``n <= 5``).
        """
        if self.process_dimension > max_process_dimension:
            raise ValueError(
                f"A dense {self.process_dimension}x{self.process_dimension} adjoint needs "
                f"{16 * self.process_dimension**2} bytes; raise max_process_dimension explicitly."
            )
        d = self.d
        weights = self._weights(weights, (self.row_count,), np).reshape(self.probe_count, self.outcome_count)
        alpha, beta, arrow = _arrow_parameters(weights, self.povm_a, self.povm_b, np)
        blocks = alpha[:, None, None] * np.eye(d, dtype=np.complex128)
        blocks[:, 0, 0] += beta
        blocks[:, 0, 1:] += np.conj(arrow)
        blocks[:, 1:, 0] += arrow
        k, l, ck, cl = probe_table(d)
        result = np.zeros((d, d, d, d), dtype=np.complex128)
        for left, right, coefficient in ((k, k, np.abs(ck) ** 2), (k, l, np.conj(ck) * cl),
                                         (l, k, np.conj(cl) * ck), (l, l, np.abs(cl) ** 2)):
            np.add.at(result, (left, right), coefficient[:, None, None] * blocks)
        return result.transpose(0, 2, 1, 3).reshape(self.process_dimension, self.process_dimension)


# ---------------------------------------------------------------------------
# Synthetic truth and observation models


def _array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


@dataclass
class QuirogaSensingData:
    """Truth plus an observation model for :class:`QuirogaSensingDesign`.

    ``noiseless`` returns exact probabilities of ``truth_factor``.
    ``gaussian`` adds ``fixed_row_noise(s, noise_seed, noise_std)`` to row
    ``s``: a random-access draw that is invariant to batching, order and
    repetition.  ``shots`` draws, for each probe independently, ``shots``
    outcomes of the full 2d-outcome POVM from a multinomial seeded by
    ``SeedSequence((shot_seed, domain), spawn_key=(probe,))`` and reports
    frequencies.  ``stored`` looks up a complete observation vector.
    """

    design: QuirogaSensingDesign
    truth_factor: Optional[np.ndarray] = None
    observation_mode: str = "noiseless"
    noise_std: float = 0.0
    noise_seed: int = 0
    shots: Optional[int] = None
    shot_seed: int = 0
    observations: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.design, QuirogaSensingDesign):
            raise TypeError("design must be a QuirogaSensingDesign.")
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}.")
        if self.truth_factor is not None:
            truth = np.asarray(self.truth_factor)
            if truth.ndim == 1:
                truth = truth[:, None]
            if (truth.ndim != 2 or truth.shape[0] != self.process_dimension or truth.shape[1] < 1
                    or truth.dtype.kind not in "iufc" or not np.all(np.isfinite(truth))):
                raise ValueError("truth_factor must be finite with shape (d**2, rank).")
            self.truth_factor = np.asarray(truth, dtype=np.complex128)
        elif self.observation_mode != "stored":
            raise ValueError(f"{self.observation_mode} observations require truth_factor.")
        if self.observation_mode == "stored":
            observations = np.asarray(self.observations) if self.observations is not None else None
            if (observations is None or observations.shape != (self.m,) or observations.dtype.kind not in "iuf"
                    or not np.all(np.isfinite(observations))):
                raise ValueError(f"stored mode requires {self.m} finite real observations.")
            self.observations = np.asarray(observations, dtype=np.float64)
        elif self.observations is not None:
            raise ValueError("Only stored mode accepts an observations vector.")
        self.noise_seed = _seed(self.noise_seed, "noise_seed")
        self.shot_seed = _seed(self.shot_seed, "shot_seed")
        if isinstance(self.noise_std, (bool, np.bool_)) or np.ndim(self.noise_std) != 0 or not np.isrealobj(self.noise_std):
            raise ValueError("noise_std must be a finite nonnegative real scalar.")
        self.noise_std = float(self.noise_std)
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be a finite nonnegative real scalar.")
        if self.noise_std and self.observation_mode != "gaussian":
            raise ValueError("noise_std is only used in gaussian mode.")
        if self.observation_mode == "shots":
            self.shots = _positive_integer(self.shots, "shots")
        elif self.shots is not None:
            raise ValueError("shots is only used in shots mode.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON-compatible dictionary.")
        self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))

    @property
    def n_qubits(self):
        return self.design.n_qubits

    @property
    def d(self):
        return self.design.d

    @property
    def process_dimension(self):
        return self.design.process_dimension

    @property
    def m(self):
        return self.design.row_count

    def noiseless_values(self, rows=None):
        """Exact truth predictions for ``rows`` (all rows when omitted)."""
        if self.truth_factor is None:
            raise ValueError("No truth factor is available.")
        if rows is None:
            return self.design.full_values(self.truth_factor)
        return self.design.values(self.truth_factor, rows)

    def observations_for_rows(self, rows):
        """Deterministic observations for any rows, including repeated rows."""
        rows = self.design.validate_rows(rows)
        if self.observation_mode == "stored":
            return self.observations[rows]
        if self.observation_mode == "shots":
            probes, outcomes = self.design.decode_rows(rows)
            unique, inverse = np.unique(probes, return_inverse=True)
            return self.shot_frequencies(unique)[inverse, outcomes]
        values = self.design.values(self.truth_factor, rows)
        if self.observation_mode == "gaussian":
            values = values + fixed_row_noise(rows, self.noise_seed, self.noise_std)
        return values

    def all_observations(self, *, probe_chunk=4096):
        """Complete observation vector; equal to ``observations_for_rows``."""
        if self.observation_mode == "stored":
            return self.observations.copy()
        if self.observation_mode == "shots":
            probe_chunk = _positive_integer(probe_chunk, "probe_chunk")
            return np.concatenate([
                self.shot_frequencies(np.arange(start, min(start + probe_chunk, self.design.probe_count))).reshape(-1)
                for start in range(0, self.design.probe_count, probe_chunk)
            ])
        values = self.design.full_values(self.truth_factor)
        if self.observation_mode == "gaussian":
            step = probe_chunk * self.design.outcome_count
            for start in range(0, self.m, step):
                rows = np.arange(start, min(start + step, self.m), dtype=np.int64)
                values[rows] += fixed_row_noise(rows, self.noise_seed, self.noise_std)
        return values

    def shot_frequencies(self, probes):
        """Multinomial outcome frequencies, shape ``(P, 2d)``, per probe."""
        if self.observation_mode != "shots":
            raise ValueError("shot_frequencies requires shots mode.")
        probes = np.asarray(probes)
        probabilities = self.design.probe_values(self.truth_factor, probes)
        totals = probabilities.sum(axis=1)
        if np.min(probabilities) < -1e-12 or np.max(np.abs(totals - 1.0)) > 1e-10:
            raise ValueError("Shot sampling requires a CPTP truth with valid outcome probabilities.")
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        frequencies = np.empty_like(probabilities)
        for index, probe in enumerate(probes.tolist()):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                (self.shot_seed, _SHOT_DOMAIN), spawn_key=(probe,))))
            frequencies[index] = rng.multinomial(self.shots, probabilities[index]) / self.shots
        return frequencies

    def sample_rows(self, rng, batch_size, population=None):
        """Uniform rows with replacement from all rows or ``population``."""
        batch_size = _positive_integer(batch_size, "batch_size")
        if population is None:
            return rng.integers(0, self.m, size=batch_size, dtype=np.int64)
        population = self.design.validate_rows(population)
        return population[rng.integers(0, population.size, size=batch_size)]

    def fixed_row_subset(self, size, seed):
        """Sorted rows drawn without replacement by ``default_rng(seed)``."""
        size = _positive_integer(size, "size")
        if size > self.m:
            raise ValueError("size exceeds the number of sensing rows.")
        rng = np.random.default_rng(_seed(seed, "seed"))
        return np.sort(rng.choice(self.m, size=size, replace=False)).astype(np.int64)

    def materialize(self, *, probe_chunk=4096):
        """Return a stored-mode copy holding the complete observation vector."""
        metadata = dict(self.metadata, materialized_from=self.observation_mode,
                        noise_std=self.noise_std, noise_seed=self.noise_seed,
                        shots=self.shots, shot_seed=self.shot_seed)
        return replace(self, observation_mode="stored", observations=self.all_observations(probe_chunk=probe_chunk),
                       noise_std=0.0, shots=None, metadata=metadata)

    def fidelity(self, factor):
        """Pure-target process fidelity; requires a rank-one truth."""
        if self.truth_factor is None or self.truth_factor.shape[1] != 1:
            raise ValueError("Fidelity requires a rank-one truth factor.")
        return float(pure_target_process_fidelity(np.asarray(factor), self.truth_factor))

    def to_dense_qpt_data(self, rows=None, observations=None, *, max_entries=2**26):
        """Dense ``QPTData`` for the existing small-system runners.

        The operator basis is :func:`choi_standard_basis`, so the legacy TP
        map returns ``(Tr_out(UU^H))^T`` (the same residual norm and factor
        gradient) and its fidelity proxy equals :meth:`fidelity`.
        """
        from .quantum_process_tomography import QPTData
        rows = np.arange(self.m, dtype=np.int64) if rows is None else self.design.validate_rows(rows)
        targets = self.observations_for_rows(rows) if observations is None else np.asarray(observations)
        if targets.shape != rows.shape:
            raise ValueError("observations must match rows.")
        chi_star = None
        if self.truth_factor is not None:
            chi_star = self.truth_factor @ self.truth_factor.conj().T
        return QPTData(
            f_vector=targets,
            D_tensors=self.design.dense_sensing_matrices(rows, max_entries=max_entries),
            A_basis=choi_standard_basis(self.d),
            chi_star=chi_star,
        )


def haar_unitary_choi_truth(n_qubits, channel_seed=0):
    """Choi vector of the Haar unitary that ``qpt_generate_data`` draws.

    The same ``channel_seed`` yields the same unitary ``V`` as the local-Pauli
    generator; that generator's legacy convention senses ``conj(V)``.
    """
    from .qpt_generate_data import _haar_unitary
    d = 2 ** _positive_integer(n_qubits, "n_qubits")
    return unitary_choi_factor(_haar_unitary(d, _seed(channel_seed, "channel_seed")))


def generate_quiroga_sensing_data(n_qubits, *, channel_seed=0, observation_mode="noiseless",
                                  noise_std=0.0, noise_seed=0, shots=None, shot_seed=0,
                                  povm_a=None, povm_b=None):
    """Haar-unitary synthetic experiment on the Quiroga-style design."""
    if observation_mode == "stored":
        raise ValueError("Generate a synthetic mode, then call materialize().")
    design = QuirogaSensingDesign(n_qubits, povm_a, povm_b)
    truth = haar_unitary_choi_truth(design.n_qubits, channel_seed)
    violation = float(np.linalg.norm(trace_preserving_residual(truth)))
    if violation > 1e-10 * math.sqrt(design.d):
        raise FloatingPointError("Generated unitary truth failed its trace-preservation check.")
    metadata = {
        "generator": "qpt_quiroga_sensing_v1", "design": design.describe(),
        "channel_model": "haar_unitary", "channel_seed": int(channel_seed),
        "channel_rng": "Generator(PCG64(SeedSequence(channel_seed, spawn_key=(0,))))",
        "truth_rank": 1, "truth_trace": float(np.vdot(truth, truth).real),
        "truth_tp_violation": violation, "truth_factor_sha256": _array_sha256(truth),
        "noise_generator": NOISE_GENERATOR if observation_mode == "gaussian" else None,
        "shot_generator": SHOT_GENERATOR if observation_mode == "shots" else None,
        "numpy_version": np.__version__, "python_version": platform.python_version(),
    }
    return QuirogaSensingData(design, truth, observation_mode, noise_std, noise_seed,
                              shots, shot_seed, metadata=metadata)


# ---------------------------------------------------------------------------
# Scaling


def scaling_summary(n_qubits, rank=1):
    """Exact counts and byte sizes for one qubit count (complex128/float64)."""
    n_qubits, rank = _positive_integer(n_qubits, "n_qubits"), _positive_integer(rank, "rank")
    d = 2**n_qubits
    return {
        "n_qubits": n_qubits, "d": d, "probes": d * d, "outcomes": 2 * d, "rows": 2 * d**3,
        "local_pauli_rows": 24**n_qubits, "factor_bytes": 16 * d * d * rank,
        "observation_bytes": 8 * 2 * d**3, "full_pass_order_r_d3": rank * d**3,
        "dense_sensing_bytes": 16 * 2 * d**3 * d**4, "dense_choi_bytes": 16 * d**4,
        "adafgd_dense_matrix_bytes": 16 * d**4, "adafgd_eigensolver_order_d6": d**6,
    }


def _median_seconds(function, repeats):
    function()  # Warm up allocations, caches and BLAS threads.
    times = []
    for _ in range(repeats):
        began = perf_counter()
        function()
        times.append(perf_counter() - began)
    return float(np.median(times))


def time_full_pass(n_qubits, *, rank=1, repeats=1, seed=0):
    """Median NumPy seconds for full values+gradient and for exact TP.

    Each kernel is timed in its own loop after one untimed warm-up call.
    """
    design = QuirogaSensingDesign(n_qubits)
    repeats = _positive_integer(repeats, "repeats")
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(design.process_dimension, rank)) + 1j * rng.normal(size=(design.process_dimension, rank))
    targets = np.zeros(design.row_count)
    return {
        "full_sensing_loss_gradient_seconds": _median_seconds(
            lambda: design.full_loss_and_gradient(factor, targets), repeats),
        "tp_loss_gradient_seconds": _median_seconds(lambda: trace_preserving_loss_and_gradient(factor), repeats),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print exact counts (and optional NumPy timings) for this backend.")
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--time", action="store_true",
                        help="Also time full sensing and TP passes per n (median of --repeats after a warm-up).")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.min_qubits <= args.max_qubits:
        parser.error("Require 1 <= --min-qubits <= --max-qubits.")
    if args.rank < 1 or args.repeats < 1:
        parser.error("--rank and --repeats must be positive.")
    rows = []
    for n_qubits in range(args.min_qubits, args.max_qubits + 1):
        row = scaling_summary(n_qubits, args.rank)
        if args.time:
            row.update(time_full_pass(n_qubits, rank=args.rank, repeats=args.repeats))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    main()
