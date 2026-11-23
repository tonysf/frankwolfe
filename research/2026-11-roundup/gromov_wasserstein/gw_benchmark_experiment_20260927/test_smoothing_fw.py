"""Correctness tests with TEST-ONLY constants; no theorem schedule is asserted."""
from dataclasses import replace

import pytest
import torch

from gw_core import GWProblem, rademacher
from smoothing_fw import MomentumConfig, initialize_state, step

DTYPE = torch.float64


def _problem():
    A = torch.tensor([[0., .6, .9], [.6, 0., .7], [.9, .7, 0.]], dtype=DTYPE)
    B = torch.tensor([[0., .8], [.8, 0.]], dtype=DTYPE)
    a = torch.tensor([.2, .3, .5], dtype=DTYPE)
    b = torch.tensor([.4, .6], dtype=DTYPE)
    return GWProblem(A, B, a, b)


def _test_config(oracle='exact', **changes):
    # Arbitrary toy choices to expose indexing and algebra errors, not run defaults.
    config = MomentumConfig(alpha=.7, beta=.4, delta=.3, gamma0=.35, rho0=.6,
                            mu0=.8, oracle=oracle, batch=5)
    return replace(config, **changes)


def _direct_gradient(problem, P):
    K = (problem.A[:, :, None, None] - problem.B[None, None, :, :]).square()
    return 2 * torch.einsum('ikjl,kl->ij', K, P)


def _manual_row_fw(P, d, problem, gamma, mu):
    H = d + (P.sum(0) - problem.b)[None, :] / mu
    Q = (1 - gamma) * P.clone()
    for i in range(P.shape[0]):
        winner = min(range(P.shape[1]), key=lambda j: (H[i, j].item(), j))
        Q[i, winner] += gamma * problem.a[i]
    return Q


def test_schedule_explicit_constants_and_zero_based_index():
    config = _test_config()
    assert config.schedule(0) == {'gamma': .35, 'rho': .6, 'mu': .8, 'eta': None}
    values = config.schedule(3)
    assert values['gamma'] == pytest.approx(.35 / 4 ** .7)
    assert values['rho'] == pytest.approx(.6 / 4 ** .4)
    assert values['mu'] == pytest.approx(.8 / 4 ** .3)
    with pytest.raises(TypeError):
        MomentumConfig(oracle='exact', batch=1)
    with pytest.raises(ValueError):
        config.schedule(-1)
    assert config.metadata()['theory_metadata_complete'] is False
    described = replace(config, theory_id='test-only-placeholder', output_statistic='test-only')
    assert described.metadata()['theory_verification'] == 'not_verified_by_this_implementation'


def test_initialization_and_two_steps_match_independent_manual_result():
    problem, config = _problem(), _test_config()
    state = initialize_state(problem, config, seed=204)
    P0 = torch.outer(problem.a, problem.b)
    d0 = _direct_gradient(problem, P0)
    torch.testing.assert_close(state.P, P0, rtol=0, atol=0)
    torch.testing.assert_close(state.d, d0, rtol=1e-13, atol=1e-13)
    assert state.counts.oracle_evaluations == state.counts.initialization_oracle_calls == 1
    initial_reference = state.P
    first = step(problem, state)
    P1 = _manual_row_fw(P0, d0, problem, gamma=.35, mu=.8)
    torch.testing.assert_close(state.P, P1, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(first.d_used, d0, rtol=1e-13, atol=1e-13)
    assert first.P_before is initial_reference
    assert first.update_index == 0 and state.iteration == 1
    assert first.effective_momentum_weight == 1
    assert first.oracle_evaluations_this_step == 0
    assert first.counts.oracle_evaluations == 1
    # The first step has made the other marginal infeasible, exposing penalty bugs.
    assert (P1.sum(0) - problem.b).abs().sum().item() > .05
    g1 = _direct_gradient(problem, P1)
    rho1, mu1, gamma1 = .6 / 2 ** .4, .8 / 2 ** .3, .35 / 2 ** .7
    d1 = (1 - rho1) * d0 + rho1 * g1
    P2 = _manual_row_fw(P1, d1, problem, gamma=gamma1, mu=mu1)
    second = step(problem, state, config)
    torch.testing.assert_close(second.P_before, P1, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(second.d_used, d1, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(state.P, P2, rtol=1e-13, atol=1e-13)
    # Adding the penalty inside EMA would give this incorrect, distinguishable d.
    wrong_d = d1 + rho1 * (P1.sum(0) - problem.b)[None, :] / mu1
    assert torch.linalg.vector_norm(second.d_used - wrong_d).item() > .01
    assert second.update_index == 1 and second.completed_updates == 2
    assert second.oracle_evaluations_this_step == 1
    assert state.counts.exact_gradient_calls == state.counts.oracle_evaluations == 2
    assert state.counts.row_lmo_calls == 2
    assert state.counts.stochastic_gradient_calls == 0
    # Earlier diagnostic references must remain unchanged after the next step.
    torch.testing.assert_close(initial_reference, P0, rtol=0, atol=0)
    torch.testing.assert_close(first.d_used, d0, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('oracle', ['sketch', 'anchor'])
def test_fresh_samples_initialization_reuse_and_exact_call_counts(oracle):
    problem, config = _problem(), _test_config(oracle)
    state = initialize_state(problem, config, seed=211)
    manual_generator = torch.Generator().manual_seed(211)

    def next_expected(P):
        if oracle == 'sketch':
            Z = rademacher(P.shape[1], config.batch, like=P, generator=manual_generator)
            # Differentiate the scalar sample independently of sketch_gradient.
            X = P.clone().requires_grad_(True)
            g, = torch.autograd.grad(problem.sketch_loss(X, Z), X)
            return g
        n, m = P.shape
        flat = torch.randint(n * m, (config.batch,), generator=manual_generator)
        X = P.clone().requires_grad_(True)
        terms = []
        for anchor in flat.tolist():
            k, ell = divmod(anchor, m)
            K = (problem.A[:, k, None] - problem.B[:, ell][None, :]).square()
            terms.append(n * m * X[k, ell] * (K * X).sum())
        g, = torch.autograd.grad(torch.stack(terms).mean(), X)
        return g

    first_expected = next_expected(state.P)
    torch.testing.assert_close(state.d, first_expected, rtol=1e-13, atol=1e-13)
    after_initial_sample_rng = state.generator.get_state().clone()
    first = step(problem, state)
    assert torch.equal(state.generator.get_state(), after_initial_sample_rng)
    torch.testing.assert_close(first.gradient_estimate, first_expected, rtol=1e-13, atol=1e-13)
    second_expected = next_expected(state.P)
    second = step(problem, state)
    assert not torch.equal(state.generator.get_state(), after_initial_sample_rng)
    assert torch.equal(state.generator.get_state(), manual_generator.get_state())
    torch.testing.assert_close(second.gradient_estimate, second_expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(second.d_used,
                               (1 - second.rho) * first_expected + second.rho * second_expected,
                               rtol=1e-13, atol=1e-13)
    assert state.counts.oracle_evaluations == state.counts.stochastic_gradient_calls == 2
    assert state.counts.exact_gradient_calls == 0
    assert state.counts.initialization_oracle_calls == 1
    assert state.counts.sketch_probes == (2 * config.batch if oracle == 'sketch' else 0)
    assert state.counts.anchor_samples == (2 * config.batch if oracle == 'anchor' else 0)


@pytest.mark.parametrize('oracle', ['exact', 'sketch', 'anchor'])
@pytest.mark.parametrize('update', ['fw', 'row_projection'])
def test_repeated_updates_preserve_rows_without_repair_and_are_reproducible(oracle, update):
    problem = _problem()
    projection = {'eta0': .04, 'eta_exponent': .2} if update == 'row_projection' else {}
    config = _test_config(oracle, update=update, **projection)
    first = initialize_state(problem, config, seed=222)
    second = initialize_state(problem, config, seed=222)
    for k in range(30):
        info = step(problem, first)
        step(problem, second)
        torch.testing.assert_close(first.P.sum(1), problem.a, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(first.P, second.P, rtol=0, atol=0)
        torch.testing.assert_close(first.d, second.d, rtol=0, atol=0)
        assert first.P.min().item() >= 0
        assert first.counts.oracle_evaluations == k + 1
        assert info.P_before is not first.P
    assert first.counts.row_lmo_calls == (30 if update == 'fw' else 0)
    assert first.counts.row_projection_calls == (30 if update == 'row_projection' else 0)


def test_row_projection_uses_separate_eta_and_independent_simplex_solution():
    problem = _problem()
    config = _test_config(update='row_projection', eta0=.17, eta_exponent=.8)
    state = initialize_state(problem, config, seed=233)
    for k in range(2):
        P = state.P
        old_d = state.d
        rho = .6 / (k + 1) ** .4
        g = _direct_gradient(problem, P)
        d = g if k == 0 else (1 - rho) * old_d + rho * g
        mu, eta = .8 / (k + 1) ** .3, .17 / (k + 1) ** .8
        V = P - eta * (d + (P.sum(0) - problem.b)[None, :] / mu)
        # Closed form for a two-coordinate simplex of mass a_i.
        first_column = ((problem.a + V[:, 0] - V[:, 1]) / 2).clamp_min(0)
        first_column = torch.minimum(first_column, problem.a)
        expected = torch.stack((first_column, problem.a - first_column), dim=1)
        info = step(problem, state)
        assert info.eta == pytest.approx(eta)
        torch.testing.assert_close(state.P, expected, rtol=1e-12, atol=1e-13)


def test_frozen_run_and_explicit_projection_configuration():
    problem, config = _problem(), _test_config()
    state = initialize_state(problem, config, seed=244)
    with pytest.raises(ValueError, match='cannot change'):
        step(problem, state, replace(config, mu0=.7))
    with pytest.raises(ValueError, match='same frozen'):
        step(_problem(), state)
    with pytest.raises(ValueError, match='own positive eta0'):
        replace(config, update='row_projection')
    with pytest.raises(ValueError, match='explicit nonnegative eta_exponent'):
        replace(config, update='row_projection', eta0=.1)


def test_manuscript_schedule_and_affine_constraint_lift():
    problem = _problem()
    # The supplied Proposition 2.11 fixes these exponents and unit prefactors.
    # mu0 below remains an arbitrary positive test constant, not a tuned value.
    config = MomentumConfig(alpha=.75, beta=.5, delta=.25, gamma0=1., rho0=1.,
                            mu0=.9, oracle='sketch', batch=5)
    values = config.schedule(15)
    assert values['gamma'] == pytest.approx(1 / 8)
    assert values['rho'] == pytest.approx(1 / 4)
    assert values['mu'] == pytest.approx(.9 / 2)
    assert config.alpha + config.delta >= 1  # Lemma 2.7 assumption.
    state = initialize_state(problem, config, seed=255)
    step(problem, state)
    X = state.P.clone().requires_grad_(True)
    n = X.shape[0]
    residual = X.sum(0) - problem.b
    projection = X - residual[None, :] / n
    torch.testing.assert_close(projection.sum(0), problem.b, rtol=1e-13, atol=1e-13)
    distance_squared = (X - projection).square().sum()
    torch.testing.assert_close(distance_squared, residual.square().sum() / n,
                               rtol=1e-13, atol=1e-13)
    mu = values['mu']
    beta_theorem = mu / n
    lifted_penalty = distance_squared / (2 * beta_theorem)
    torch.testing.assert_close(lifted_penalty, residual.square().sum() / (2 * mu),
                               rtol=1e-13, atol=1e-13)
    gradient, = torch.autograd.grad(lifted_penalty, X)
    expected = (residual / mu)[None, :].expand_as(X)
    torch.testing.assert_close(gradient, expected, rtol=1e-13, atol=1e-13)
