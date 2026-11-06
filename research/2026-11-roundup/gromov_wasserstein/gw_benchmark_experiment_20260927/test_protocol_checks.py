"""Additional Stage A checks; no proposed optimizer or theorem is assumed.

These tests independently check scalar derivatives and small closed-form optima,
including protocol requirements not explicitly covered by the starter's tests.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gw_core import (
    GWProblem, assignment_lmo, fw_row_update, kl_prox,
    project_transport_dykstra, round_transport, row_lmo,
)

DTYPE = torch.float64


def _problem(n=4, m=3):
    rng = torch.Generator().manual_seed(91027)
    A = torch.rand((n, n), dtype=DTYPE, generator=rng)
    B = torch.rand((m, m), dtype=DTYPE, generator=rng)
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    A.fill_diagonal_(0)
    B.fill_diagonal_(0)
    a = torch.arange(1, n + 1, dtype=DTYPE)
    b = torch.arange(1, m + 1, dtype=DTYPE)
    a, b = a / a.sum(), b / b.sum()
    P = torch.rand((n, m), dtype=DTYPE, generator=rng)
    P = a[:, None] * P / P.sum(1, keepdim=True)
    return GWProblem(A, B, a, b), P


def test_two_point_negative_curvature():
    A = torch.tensor([[0., 1.], [1., 0.]], dtype=DTYPE)
    a = torch.full((2,), .5, dtype=DTYPE)
    problem = GWProblem(A, A, a, a)
    for value in (0., .125, .25, .375, .5):
        t = torch.tensor(value, dtype=DTYPE, requires_grad=True)
        P = torch.stack((torch.stack((t, .5 - t)),
                         torch.stack((.5 - t, t))))
        loss = problem.loss(P)
        first, = torch.autograd.grad(loss, t, create_graph=True)
        second, = torch.autograd.grad(first, t)
        torch.testing.assert_close(loss, 4 * t - 8 * t.square(), rtol=0, atol=1e-14)
        torch.testing.assert_close(second, t.new_tensor(-16.), rtol=0, atol=1e-14)


def test_each_anchor_is_gradient_of_its_scalar_loss():
    problem, P = _problem()
    n, m = P.shape
    sampled_losses = []
    for k in range(n):
        for ell in range(m):
            X = P.clone().requires_grad_(True)
            K = (problem.A[:, k, None] - problem.B[:, ell][None, :]).square()
            phi = n * m * X[k, ell] * (K * X).sum()
            gradient, = torch.autograd.grad(phi, X)
            actual = problem.anchor_gradient(P, torch.tensor([[k, ell]]))
            torch.testing.assert_close(actual, gradient, rtol=1e-13, atol=1e-13)
            sampled_losses.append(phi.detach())
    torch.testing.assert_close(torch.stack(sampled_losses).mean(), problem.loss(P),
                               rtol=1e-13, atol=1e-13)


def test_repeated_row_fw_updates_preserve_nonuniform_masses_and_ties():
    problem, P = _problem()
    generator = torch.Generator().manual_seed(9127)
    initial = P.clone()
    for step in range(100):
        H = torch.randn(P.shape, dtype=DTYPE, generator=generator)
        if step % 9 == 0:
            H.zero_()  # A tied row must choose the lowest column index.
            atom = row_lmo(H, problem.a)
            torch.testing.assert_close(atom[:, 0], problem.a, rtol=0, atol=0)
            assert torch.count_nonzero(atom[:, 1:]).item() == 0
        # These are test coefficients, not a proposed algorithm schedule.
        gamma = (0., .05, .4, 1.)[step % 4]
        old = P.clone()
        P = fw_row_update(P, H, problem.a, gamma)
        torch.testing.assert_close(P, (1 - gamma) * old + gamma * row_lmo(H, problem.a),
                                   rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(P.sum(1), problem.a, rtol=1e-13, atol=1e-14)
        assert P.min().item() >= 0
    assert initial.shape == P.shape


def test_rounding_feasible_fixed_point_and_zero_column_bound():
    problem, P = _problem()
    feasible = torch.outer(problem.a, problem.b)
    torch.testing.assert_close(round_transport(feasible, problem.a, problem.b), feasible,
                               rtol=1e-14, atol=1e-14)
    # A concentrated row-feasible plan exercises c_j=0 and the repair bound.
    concentrated = torch.zeros_like(P)
    concentrated[:, 0] = problem.a
    before = concentrated.clone()
    rounded = round_transport(concentrated, problem.a, problem.b)
    torch.testing.assert_close(concentrated, before, rtol=0, atol=0)
    torch.testing.assert_close(rounded.sum(0), problem.b, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(rounded.sum(1), problem.a, rtol=1e-13, atol=1e-14)
    assert rounded.min().item() >= 0
    violation = (concentrated.sum(0) - problem.b).abs().sum()
    assert (rounded - concentrated).abs().sum() <= violation + 1e-13
    kappa = (problem.A[:, :, None, None] - problem.B[None, None, :, :]).square().max()
    assert abs(problem.loss(rounded) - problem.loss(concentrated)) <= 2 * kappa * violation + 1e-13


@pytest.mark.parametrize('invalid', [-.01, float('nan'), float('inf')])
def test_rounding_rejects_invalid_output(invalid):
    problem, P = _problem()
    P[0, 0] = invalid
    with pytest.raises(ValueError, match='finite, nonnegative'):
        round_transport(P, problem.a, problem.b)


@pytest.mark.parametrize('entries', [
    [[.3, -.1], [.2, .4]],
    [[3., -4.], [-2., 2.]],
    [[-4., 3.], [2., -2.]],
])
def test_dykstra_matches_independent_two_by_two_projection(entries):
    Z = torch.tensor(entries, dtype=DTYPE)
    a = torch.full((2,), .5, dtype=DTYPE)
    # Every feasible plan is [[t, .5-t], [.5-t, t]], 0 <= t <= .5.
    t = np.clip((1 + entries[0][0] + entries[1][1]
                 - entries[0][1] - entries[1][0]) / 4, 0, .5)
    expected = torch.tensor([[t, .5 - t], [.5 - t, t]], dtype=DTYPE)
    actual, info = project_transport_dykstra(Z, a, a, gap_tol=1e-12)
    assert info['converged'], info
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)
    assert info['projection_primal_dual_gap'] >= -1e-12


def test_kl_prox_matches_independent_two_by_two_optimum():
    P = torch.tensor([[.3, .2], [.2, .3]], dtype=DTYPE)
    G = torch.tensor([[.7, -.4], [.3, 1.2]], dtype=DTYPE)
    a = torch.full((2,), .5, dtype=DTYPE)
    eta = .8
    # Scalar first-order optimality gives t/(.5-t) from the kernel odds ratio.
    log_odds = .5 * (P[0, 0].log() + P[1, 1].log()
                     - P[0, 1].log() - P[1, 0].log()
                     - eta * (G[0, 0] + G[1, 1] - G[0, 1] - G[1, 0]))
    t = .5 * torch.sigmoid(log_odds)
    expected = torch.stack((torch.stack((t, .5 - t)), torch.stack((.5 - t, t))))
    actual, log_actual, info = kl_prox(P.log(), G, a, a, eta=eta, tol=1e-13)
    assert info['converged'], info
    assert info['underflow_entries'] == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(log_actual.exp(), actual, rtol=0, atol=0)


def test_assignment_lmo_matches_enumerated_permutations():
    import itertools
    G = torch.tensor([[.2, .5, -.1], [.8, -.4, .2], [.3, .7, .9]], dtype=DTYPE)
    a = torch.full((3,), 1 / 3, dtype=DTYPE)
    atom = assignment_lmo(G, a, a)
    costs = [sum(G[i, j].item() for i, j in enumerate(perm)) / 3
             for perm in itertools.permutations(range(3))]
    assert abs((G * atom).sum().item() - min(costs)) < 1e-14
    torch.testing.assert_close(atom.sum(0), a, rtol=0, atol=1e-14)
    torch.testing.assert_close(atom.sum(1), a, rtol=0, atol=1e-14)
