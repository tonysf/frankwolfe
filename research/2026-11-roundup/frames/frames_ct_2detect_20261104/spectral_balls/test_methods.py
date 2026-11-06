"""Regression checks that distinguish spectral geometry from Frobenius geometry."""

import numpy as np
import pytest

from methods import op_norm, project_factor, spectral_lmo


def test_projection_keeps_two_singular_values_at_the_cap():
    projected, certificate = project_factor(np.diag([2.0, 1.5, -0.3]), 1.0,
                                            return_certificate=True)
    np.testing.assert_allclose(projected, np.diag([1.0, 1.0, 0.0]), atol=1e-12)
    assert op_norm(projected) <= 1.0 + 1e-12
    assert np.linalg.norm(projected) == pytest.approx(np.sqrt(2))
    assert certificate.primal_feasibility <= 1e-12


def test_lmo_attains_nuclear_norm_support_and_leaves_the_null_space_zero():
    gradient = np.diag([3.0, 2.0, 0.0])
    radius = 1.5
    atom = spectral_lmo(gradient, radius)
    np.testing.assert_allclose(atom, np.diag([-radius, -radius, 0.0]))
    assert np.sum(gradient * atom) == pytest.approx(-radius * 5.0)


def test_coupled_projection_produces_a_feasible_point_and_certificate():
    z = np.array([[2.0, -1.0, 0.7], [-1.0, 3.0, 2.0], [1.0, -2.0, 4.0]])
    projected, certificate = project_factor(z, 1.0, iteration=3, return_certificate=True)
    assert op_norm(projected) <= 1.0 + 1e-12
    assert np.min(projected) >= 0.0
    assert certificate.method == "dykstra"
    assert certificate.iterations > 1
    assert certificate.status in {"requested_accuracy", "numerical_floor"}
    assert certificate.primal_feasibility <= 1e-12
