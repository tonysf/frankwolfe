"""Numerically audit the published sum-loss gradient mapping."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper.experiments import quantum_process_tomography_structured_jax as qpt
from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.qpt_structured_operators import (
    product_measurement_vectors,
    rank_one_measurement_loss_and_gradient,
    rank_one_measurement_vectors,
    trace_preserving_loss_and_gradient,
)
from subset import fixed_subset

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--save", type=Path, required=True)
a = p.parse_args()
data = StructuredQPTData.load_npz(a.data)
_, symbols, observations, hashes = fixed_subset(data, 128, 991)
bank = rank_one_measurement_vectors(data.local_measurements)
vectors = product_measurement_vectors(symbols, bank)
factor = qpt.unpack_factor(qpt.make_factor_initial_point(data, 1, 0), data.process_dimension, 1)
amplitudes = vectors.conj() @ factor
predicted = np.sum(np.abs(amplitudes) ** 2, axis=1)
residual = predicted - observations
matrix_gradient = vectors.T @ (residual[:, None] * vectors.conj())
matrix_gradient = 0.5 * (matrix_gradient + matrix_gradient.conj().T)
published_factor_term = matrix_gradient @ factor
mean_loss, mean_factor_gradient = rank_one_measurement_loss_and_gradient(
    factor, symbols, observations, bank
)
assert np.allclose(published_factor_term, 0.5 * len(symbols) * mean_factor_gradient,
                   rtol=1e-11, atol=1e-12)
sum_loss = 0.5 * np.sum(residual**2)
assert np.isclose(sum_loss, len(symbols) * mean_loss, rtol=1e-12, atol=1e-12)
tp_half_loss, tp_factor_gradient = trace_preserving_loss_and_gradient(
    factor, data.local_basis
)
rng = np.random.default_rng(77)
direction = rng.normal(size=factor.shape) + 1j * rng.normal(size=factor.shape)
direction /= np.linalg.norm(direction)
epsilon = 1e-6
plus = trace_preserving_loss_and_gradient(factor + epsilon * direction, data.local_basis)[0]
minus = trace_preserving_loss_and_gradient(factor - epsilon * direction, data.local_basis)[0]
finite_difference = (plus - minus) / (2 * epsilon)
analytic = np.vdot(tp_factor_gradient, direction).real
assert np.isclose(finite_difference, analytic, rtol=1e-6, atol=1e-7)
report = dict(
    status="verified", subset_count=len(symbols), sum_loss=float(sum_loss),
    mean_loss=float(mean_loss), matrix_gradient_spectral_norm=float(np.linalg.norm(matrix_gradient, 2)),
    factor_gradient_mapping="G_sum@U == 0.5*m*real_Frobenius_gradient_of_mean_loss",
    tp_mapping="gradient_of_0.5*H_wrt_factor == published_grad_chi_H@U",
    tp_half_loss=float(tp_half_loss), tp_directional_finite_difference=float(finite_difference),
    tp_directional_analytic=float(analytic), **hashes,
)
a.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(report, indent=2))
