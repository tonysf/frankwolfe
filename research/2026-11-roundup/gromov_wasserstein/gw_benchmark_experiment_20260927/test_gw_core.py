import itertools
import numpy as np
import torch
from gw_core import (GWProblem, row_lmo, fw_row_update, round_transport,
                     simplex_rows, assignment_lmo, original_fw_gap,
                     smoothed_gap, kl_prox, project_transport_dykstra)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def setup_problem(n=4, m=3):
    gen = torch.Generator().manual_seed(2026)
    A = torch.rand(n, n, generator=gen); A = (A+A.T)/2; A.fill_diagonal_(0)
    B = torch.rand(m, m, generator=gen); B = (B+B.T)/2; B.fill_diagonal_(0)
    a, b = torch.full((n,), 1/n), torch.full((m,), 1/m)
    prob = GWProblem(A, B, a, b)
    P = torch.rand(n, m, generator=gen)
    P = P/P.sum(1, keepdim=True)*a[:, None]
    return prob, P


def test_full_quadratic_and_gradient_off_feasibility():
    pr, P = setup_problem()
    X = P.clone().requires_grad_(True)
    K = (pr.A[:, :, None, None]-pr.B[None, None, :, :]).square()
    brute = torch.einsum('ikjl,ij,kl->', K, X, X)
    grad, = torch.autograd.grad(brute, X)
    assert torch.allclose(pr.loss(P), brute, atol=1e-12)
    assert torch.allclose(pr.gradient(P), grad, atol=1e-12)
    assert (P.sum(0)-pr.b).norm() > 1e-3  # actually tests off-feasible gradient


def test_anchor_unbiased_exact_enumeration():
    pr, P = setup_problem()
    pairs = torch.tensor(list(itertools.product(range(4), range(3))))
    assert torch.allclose(pr.anchor_gradient(P, pairs), pr.gradient(P), atol=1e-12)


def test_sketch_unbiased_exact_enumeration():
    pr, P = setup_problem()
    Z = torch.tensor(list(itertools.product([-1., 1.], repeat=3))).T
    assert torch.allclose(pr.sketch_loss(P, Z), pr.loss(P), atol=1e-12)
    assert torch.allclose(pr.sketch_gradient(P, Z), pr.gradient(P), atol=1e-12)


def test_sketch_is_gradient_of_scalar_sample():
    pr, P = setup_problem()
    Z = torch.tensor([[1., -1.], [-1., -1.], [1., 1.]])
    X = P.clone().requires_grad_(True)
    grad, = torch.autograd.grad(pr.sketch_loss(X, Z), X)
    assert torch.allclose(grad, pr.sketch_gradient(P, Z), atol=1e-12)


def test_lmo_and_fw_invariant():
    pr, P = setup_problem()
    G = pr.gradient(P)
    S = row_lmo(G, pr.a)
    assert torch.allclose(S.sum(1), pr.a)
    assert torch.allclose((G*S).sum(), (pr.a*G.min(1).values).sum())
    Q = fw_row_update(P, G, pr.a, .2)
    assert torch.allclose(Q, .8*P+.2*S)
    assert torch.allclose(Q.sum(1), pr.a)
    assert Q.min() >= 0


def test_rounding_row_feasible_and_objective_bound():
    pr, P = setup_problem()
    Q = round_transport(P, pr.a, pr.b)
    assert Q.min() >= 0
    assert torch.allclose(Q.sum(1), pr.a, atol=1e-12)
    assert torch.allclose(Q.sum(0), pr.b, atol=1e-12)
    v = (P.sum(0)-pr.b).abs().sum()
    assert (Q-P).abs().sum() <= v+1e-12
    K = (pr.A[:, :, None, None]-pr.B[None, None, :, :]).square()
    assert abs(pr.loss(Q)-pr.loss(P)) <= 2*K.max()*v+1e-12


def test_general_rounding_and_zero_plan():
    pr, P = setup_problem()
    for X in (7*P, P*.01, torch.zeros_like(P)):
        Q = round_transport(X, pr.a, pr.b)
        assert torch.allclose(Q.sum(1), pr.a, atol=1e-12)
        assert torch.allclose(Q.sum(0), pr.b, atol=1e-12)
        assert Q.min() >= 0


def test_planted_zero_and_assignment_gap():
    pr, _ = setup_problem(5, 5)
    perm = torch.tensor([2, 4, 1, 0, 3])
    B = pr.A[perm][:, perm]
    pr = GWProblem(pr.A, B, pr.a, pr.b)
    Pstar = torch.zeros(5, 5)
    Pstar[torch.arange(5), torch.argsort(perm)] = .2
    assert abs(pr.loss(Pstar)) < 1e-12
    assert abs(original_fw_gap(pr, Pstar)) < 1e-12


def test_gap_transfer_bound():
    pr, P = setup_problem(4, 4)
    Q = round_transport(P, pr.a, pr.b)
    v = (P.sum(0)-pr.b).abs().sum().item()
    K = (pr.A[:, :, None, None]-pr.B[None, None, :, :]).square()
    for mu in (.01, .5, 100):
        assert original_fw_gap(pr, Q) <= smoothed_gap(pr, P, mu)+6*K.max().item()*v+1e-11


def test_row_simplex_projection():
    V = torch.tensor([[.5, -.3, .8], [-2., -2., -2.]])
    Q = simplex_rows(V, torch.tensor([1., .3]))
    assert torch.allclose(Q, torch.tensor([[.35, 0., .65], [.1, .1, .1]]))


def test_kl_prox_feasibility_and_constant_cost_invariance():
    pr, P = setup_problem()
    P = torch.outer(pr.a, pr.b)
    G = pr.gradient(P)
    Q, _, info = kl_prox(P.log(), G, pr.a, pr.b, eta=.7, tol=1e-12)
    assert info['converged']
    assert torch.allclose(Q.sum(1), pr.a, atol=1e-12)
    assert torch.allclose(Q.sum(0), pr.b, atol=1e-12)
    row_shift = torch.tensor([.2, -.3, 1., .9])[:, None]
    col_shift = torch.tensor([-.1, .6, .2])[None, :]
    R, _, _ = kl_prox(P.log(), G+row_shift+col_shift, pr.a, pr.b, eta=.7, tol=1e-12)
    assert torch.allclose(Q, R, atol=1e-12)


def test_euclidean_projection_certificate():
    pr, P = setup_problem(3, 3)
    Z = P - .1*pr.gradient(P)
    R, info = project_transport_dykstra(Z, pr.a, pr.b, gap_tol=1e-11)
    assert info['converged'], info
    assert torch.allclose(R.sum(1), pr.a, atol=1e-11)
    assert torch.allclose(R.sum(0), pr.b, atol=1e-11)
    # A feasible point projects to itself.
    R2, info2 = project_transport_dykstra(R, pr.a, pr.b, gap_tol=1e-11)
    assert torch.allclose(R, R2, atol=1e-10)
