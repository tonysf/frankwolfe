"""Explicit initializer-coordinate conventions for Quiroga diagnostic campaigns.

The optional Gell-Mann mode changes only the distribution of the physical
initial factor.  All optimization still runs in the existing input-major Choi
coordinates.  This is a documented diagnostic, not a claim to reproduce the
paper's unspecified initializer normalization or uniform sampling law.
"""

from __future__ import annotations

import math

import numpy as np


INITIALIZATION_BASES = ("choi", "gell-mann")
GELL_MANN_CONVENTION = "hs_unit_gell_mann_identity_sym_antisym_diag_v1"


def initialization_descriptor(initialization_basis):
    """Describe the common coefficient draw and its physical-coordinate map."""
    if (not isinstance(initialization_basis, str)
            or initialization_basis not in INITIALIZATION_BASES):
        raise ValueError(f"initialization_basis must be one of {INITIALIZATION_BASES}.")
    descriptor = {
        "convention": "quiroga_campaign_initializer_coordinates_v1",
        "basis": initialization_basis,
        "coefficient_draw": {
            "generator": "numpy.random.default_rng (PCG64)",
            "shape": "(d**2, rank)",
            "distribution": "independent real and imaginary uniform[0,1)",
            "draw_order": "all real coefficients, then all imaginary coefficients; C order",
            "seed_configuration_key": "initialization_seed",
        },
        "normalization": {
            "stage": "coefficient factor before mapping; no second normalization",
            "rule": "C *= sqrt(d)/||C||_F; ||U0||_F**2 = d up to roundoff",
        },
        "operator_basis_hilbert_schmidt_norm": 1.0,
        "physical_coordinates": "U0[i*d+o,a] = K_a[o,i] (input-major Choi)",
    }
    if initialization_basis == "choi":
        descriptor["operator_basis"] = {
            "convention": "choi_matrix_units_input_major_v1",
            "ordering": "E[i*d+o] = |o><i|",
            "mapping": "U0 = C (identity)",
        }
    else:
        descriptor["operator_basis"] = {
            "convention": GELL_MANN_CONVENTION,
            "ordering": (
                "identity; symmetric pairs j<k in lexicographic order; "
                "antisymmetric pairs j<k in lexicographic order; diagonals l=1,...,d-1"
            ),
            "identity": "I/sqrt(d)",
            "symmetric": "(|j><k|+|k><j|)/sqrt(2)",
            "antisymmetric": "(-i|j><k|+i|k><j|)/sqrt(2)",
            "diagonal": "diag(1 repeated l times,-l,0,...,0)/sqrt(l*(l+1))",
            "mapping": "K_a=sum_b C[b,a]*E_b; U0[i*d+o,a]=K_a[o,i]",
            "implementation": "O(d**2*rank) synthesis; no dense basis or basis-change matrix",
        }
    return descriptor


def gell_mann_coefficients_to_choi(coefficients):
    """Synthesize normalized Hermitian Gell-Mann coefficients in Choi order.

    ``coefficients`` has shape ``(d**2, rank)`` with power-of-two ``d >= 2``.
    Basis ordering and signs are specified by :func:`initialization_descriptor`.
    The complex-linear map is unitary, so it preserves inner products and
    Frobenius/operator norms.  It neither normalizes nor projects the input.
    Runtime and storage are O(d**2 * rank), including for d=256 (eight qubits).
    """
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim != 2 or coefficients.shape[1] < 1:
        raise ValueError("coefficients must have shape (d**2, rank), rank >= 1.")
    d = math.isqrt(coefficients.shape[0])
    if (d * d != coefficients.shape[0] or d < 2 or d & (d - 1)):
        raise ValueError("coefficients must have d**2 rows for power-of-two d >= 2.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be finite.")

    rank = coefficients.shape[1]
    tensor = np.empty((d, d, rank), dtype=np.complex128)
    upper_rows, upper_columns = np.triu_indices(d, k=1)
    pairs = upper_rows.size
    symmetric = coefficients[1:1 + pairs]
    antisymmetric = coefficients[1 + pairs:1 + 2 * pairs]
    # Tensor axes are [input, output, rank], so K[j,k] occupies tensor[k,j].
    tensor[upper_columns, upper_rows] = (symmetric - 1j * antisymmetric) / math.sqrt(2)
    tensor[upper_rows, upper_columns] = (symmetric + 1j * antisymmetric) / math.sqrt(2)

    levels = np.arange(1, d, dtype=np.float64)
    weights = coefficients[1 + 2 * pairs:] / np.sqrt(levels * (levels + 1))[:, None]
    suffix_sums = np.cumsum(weights[::-1], axis=0)[::-1]
    diagonal = np.empty((d, rank), dtype=np.complex128)
    diagonal[0] = suffix_sums[0]
    diagonal[1:] = suffix_sums - (levels + 1)[:, None] * weights
    diagonal += coefficients[0] / math.sqrt(d)
    indices = np.arange(d)
    tensor[indices, indices] = diagonal
    return tensor.reshape(d * d, rank)
