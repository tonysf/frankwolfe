"""Protocol section 12 matched starts and independent random streams."""
import csv
import hashlib

import numpy as np
import pytest
import torch

from gw_core import GWProblem, rademacher
from make_instances import random_metric, write_instance
from run_benchmark import METHODS, make_initial_plan, parser, run
from smoothing_fw import MomentumConfig, initialize_state, step


def problem(n=8, dtype=torch.float64):
    rng = np.random.default_rng(51)
    A = torch.as_tensor(random_metric(n, rng), dtype=dtype)
    B = torch.as_tensor(random_metric(n, rng), dtype=dtype)
    mass = torch.full((n,), 1/n, dtype=dtype)
    return GWProblem(A, B, mass, mass.clone())


def config(oracle='sketch'):
    return MomentumConfig(alpha=.75, beta=.5, delta=.25, gamma0=.5, rho0=1.,
                          mu0=.1, oracle=oracle, batch=4)


def plan_hash(P):
    return hashlib.sha256(P.detach().cpu().numpy().tobytes()).hexdigest()


def numpy_rng_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_randomized_plan_exact_formula_feasibility_and_private_rng(dtype):
    # Include a non-power-of-two size to test ordinary float32 marginal rounding.
    prob = problem(10, dtype)
    numpy_before, torch_before = np.random.get_state(), torch.random.get_rng_state().clone()
    P0, details = make_initial_plan(prob, 'randomized', 900100)
    assert numpy_rng_equal(numpy_before, np.random.get_state())
    assert torch.equal(torch_before, torch.random.get_rng_state())
    permutation = np.random.default_rng(900100).permutation(10)
    assert details['independent_permutation'] == permutation.tolist()
    atom = torch.zeros_like(P0)
    atom[torch.arange(10), torch.from_numpy(permutation)] = 1/10
    expected = .9 * torch.outer(prob.a, prob.b) + .1 * atom
    torch.testing.assert_close(P0, expected, rtol=2*torch.finfo(dtype).eps, atol=0)
    assert bool((P0 > 0).all())
    torch.testing.assert_close(P0.sum(0), prob.b, rtol=8*torch.finfo(dtype).eps, atol=0)
    torch.testing.assert_close(P0.sum(1), prob.a, rtol=8*torch.finfo(dtype).eps, atol=0)
    assert plan_hash(make_initial_plan(prob, 'randomized', 900100)[0]) == plan_hash(P0)
    assert plan_hash(make_initial_plan(prob, 'randomized', 900101)[0]) != plan_hash(P0)
    # Initial plans are independent of the costs/planted correspondence.
    other = GWProblem(prob.B, prob.A, prob.a, prob.b)
    assert plan_hash(make_initial_plan(other, 'randomized', 900100)[0]) == plan_hash(P0)


def test_product_default_is_unchanged_and_does_not_use_initialization_randomness():
    prob, cfg = problem(), config()
    product, details = make_initial_plan(prob, 'product', 900100)
    other, _ = make_initial_plan(prob, 'product', 900101)
    torch.testing.assert_close(product, torch.outer(prob.a, prob.b), rtol=0, atol=0)
    torch.testing.assert_close(product, other, rtol=0, atol=0)
    assert details['rng'] is None and details['independent_permutation'] is None
    old_default = initialize_state(prob, cfg, seed=103)
    explicit = initialize_state(prob, cfg, seed=103, P0=product)
    torch.testing.assert_close(old_default.P, explicit.P, rtol=0, atol=0)
    torch.testing.assert_close(old_default.d, explicit.d, rtol=0, atol=0)
    assert torch.equal(old_default.generator.get_state(), explicit.generator.get_state())
    for _ in range(3):
        step(prob, old_default)
        step(prob, explicit)
        torch.testing.assert_close(old_default.P, explicit.P, rtol=0, atol=0)
    assert cfg.metadata()['initialization'] == 'P0=outer(a,b)'
    assert 'supplied' in cfg.metadata(initialization='provided')['momentum_initialization']


@pytest.mark.parametrize('oracle', ['sketch', 'anchor'])
def test_initialization_seed_and_oracle_seed_are_independent_streams(oracle):
    prob, cfg = problem(), config(oracle)
    P0, _ = make_initial_plan(prob, 'randomized', 900100)
    alternate, _ = make_initial_plan(prob, 'randomized', 900101)
    first = initialize_state(prob, cfg, seed=100, P0=P0)
    different_start = initialize_state(prob, cfg, seed=100, P0=alternate)
    different_oracle = initialize_state(prob, cfg, seed=101, P0=P0)
    # Changing P0 does not change how many or which oracle random numbers are drawn.
    assert torch.equal(first.generator.get_state(), different_start.generator.get_state())
    # Changing the oracle seed cannot change the initial coupling.
    torch.testing.assert_close(first.P, different_oracle.P, rtol=0, atol=0)
    assert not torch.equal(first.d, different_oracle.d)
    manual = torch.Generator().manual_seed(100)
    if oracle == 'sketch':
        samples = rademacher(P0.shape[1], cfg.batch, like=P0, generator=manual)
        expected = prob.sketch_gradient(P0, samples)
    else:
        flat = torch.randint(P0.numel(), (cfg.batch,), generator=manual)
        expected = prob.anchor_gradient(P0, torch.stack((flat//P0.shape[1], flat%P0.shape[1]), dim=1))
    torch.testing.assert_close(first.d, expected, rtol=0, atol=0)
    assert torch.equal(first.generator.get_state(), manual.get_state())
    original = P0.clone()
    step(prob, first)
    torch.testing.assert_close(P0, original, rtol=0, atol=0)
    assert first.P.data_ptr() != P0.data_ptr()


@pytest.mark.parametrize('defect', ['row', 'column', 'zero', 'negative', 'nan', 'shape', 'dtype'])
def test_supplied_start_rejects_invalid_inputs_without_repair(defect):
    prob = problem()
    P0, _ = make_initial_plan(prob, 'randomized', 900100)
    if defect == 'row':
        P0[0,0] += .001
        P0[1,0] -= .001
    elif defect == 'column':
        P0[0,0] += .001
        P0[0,1] -= .001
    elif defect == 'zero':
        P0 = torch.eye(8, dtype=P0.dtype)/8
    elif defect == 'negative':
        P0[0,0] = -.001
    elif defect == 'nan':
        P0[0,0] = torch.nan
    elif defect == 'shape':
        P0 = P0[:7]
    else:
        P0 = P0.float()
    before = P0.clone()
    with pytest.raises(ValueError):
        initialize_state(prob, config(), seed=100, P0=P0)
    torch.testing.assert_close(P0, before, rtol=0, atol=0, equal_nan=True)


def test_randomized_constructor_rejects_nonuniform_marginals():
    prob = problem()
    a = prob.a.clone()
    a[0] += .01
    a[1] -= .01
    other = GWProblem(prob.A, prob.B, a, prob.b)
    with pytest.raises(ValueError, match='square uniform'):
        make_initial_plan(other, 'randomized', 900100)
    product, _ = make_initial_plan(other, 'product', 900100)
    torch.testing.assert_close(product, torch.outer(a, prob.b), rtol=0, atol=0)


def test_all_practical_methods_receive_the_same_randomized_plan(tmp_path, monkeypatch):
    rng = np.random.default_rng(1000)
    A, B = random_metric(8, rng), random_metric(8, rng)
    instance = tmp_path/'toy8.npz'
    # This deliberately irrelevant extra is never loaded by initialization code.
    write_instance(instance, A, B, dict(family='initialization_test',seed=1000,noise=.1),
                   true_target=np.full(8,-12345))
    first_stochastic_kl_gradients = []
    import run_benchmark
    original_kl = run_benchmark.kl_prox
    def capture_kl(logP, G, *args, **kwargs):
        first_stochastic_kl_gradients.append((logP.exp().clone(), G.clone()))
        return original_kl(logP,G,*args,**kwargs)
    matched_hashes, matched_permutations = [], []
    for method in METHODS:
        args = parser().parse_args([
            '--instance',str(instance),'--method',method,'--out',str(tmp_path/method),
            '--budget','1e-9','--seed','100','--init-seed','900100',
            '--initialization','randomized','--batch','4','--eta','.05',
            '--inner-tol','1e-8','--inner-max','1000'])
        if method == 'stochastic-kl-momentum':
            args.batch, args.rho0 = 128, .3
        if method == 'stochastic-kl':
            monkeypatch.setattr(run_benchmark,'kl_prox',capture_kl)
        else:
            monkeypatch.setattr(run_benchmark,'kl_prox',original_kl)
        result = run(args)
        assert result['status'] in ('completed','completed_with_inner_failures'),result['error']
        assert result['config']['initialization'] == 'randomized'
        assert result['initialization_seed'] == 900100 and result['stochastic_seed'] == 100
        assert result['initialization_residuals']['both_l1'] < 1e-12
        assert result['initialization_residuals']['min_entry'] > 0
        assert result['native_sampled_gw'].startswith('omitted from randomized-start panel')
        matched_hashes.append(result['initialization_plan_sha256'])
        matched_permutations.append(result['initialization_details']['independent_permutation'])
        if method.startswith('smoothing-'):
            assert result['smoothing_config']['initialization'] == 'supplied strictly positive feasible P0'
        with (tmp_path/method/'trace.csv').open() as stream:
            initial = next(csv.DictReader(stream))
        assert initial['kind'] == 'initial' and int(initial['iteration']) == 0
    assert len(set(matched_hashes)) == 1
    assert all(p == matched_permutations[0] for p in matched_permutations)
    prob = GWProblem(torch.from_numpy(A).double(),torch.from_numpy(B).double(),
                     torch.full((8,),1/8,dtype=torch.float64),torch.full((8,),1/8,dtype=torch.float64))
    P0,_ = make_initial_plan(prob,'randomized',900100)
    state = initialize_state(prob,config(),seed=100,P0=P0)
    assert first_stochastic_kl_gradients
    kl_P0, kl_G0 = first_stochastic_kl_gradients[0]
    torch.testing.assert_close(kl_P0, P0, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(kl_G0, state.d, rtol=0, atol=0)
    assert plan_hash(P0) == matched_hashes[0]
