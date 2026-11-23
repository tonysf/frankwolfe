"""Independent checks of the theorem specialization, schedule, and KL updates."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'shareseq_25000_20260930'))
from sharded_gw import RowShardedGW
from msbpg import (constants_from_problem, theorem_constants, schedule,
                   normalized_displacement, msbpg_step, output_probabilities,
                   select_output_index, ProximalSolveError)
from stable_prox import StableCertifiedKLProx


def tiny_problem():
    A = torch.tensor([[0., .7], [.7, 0.]], dtype=torch.float64)
    B = torch.tensor([[0., 1.2], [1.2, 0.]], dtype=torch.float64)
    a = torch.tensor([.35, .65], dtype=torch.float64)
    b = torch.tensor([.6, .4], dtype=torch.float64)
    return RowShardedGW(A, B, a, b)


def scalar_optimum(reference, gradient, a, b, eta):
    """Bisection over the sole free coordinate of the 2x2 transport set."""
    lo, hi = max(0., a[0].item()+b[0].item()-1), min(a[0].item(), b[0].item())
    direction = torch.tensor([[1.,-1.],[-1.,1.]],dtype=torch.float64)
    for _ in range(100):
        t = (lo+hi)/2
        candidate = torch.tensor([[t,a[0]-t],[b[0]-t,1-a[0]-b[0]+t]],dtype=torch.float64)
        derivative = ((gradient+(candidate.log()-reference.log())/eta)*direction).sum()
        if derivative > 0:
            hi=t
        else:
            lo=t
    return candidate


def test_ambient_hessian_bound_against_explicit_rectangular_cost_matrix():
    generator=torch.Generator().manual_seed(176)
    for n,m in [(2,3),(5,2),(7,7)]:
        x,y=torch.randn(n,3,generator=generator,dtype=torch.float64),torch.randn(m,2,generator=generator,dtype=torch.float64)
        A,B=torch.cdist(x,x),torch.cdist(y,y)
        a,b=torch.full((n,),1/n,dtype=torch.float64),torch.full((m,),1/m,dtype=torch.float64)
        p=RowShardedGW(A,B,a,b)
        constants=constants_from_problem(p,block_rows=2)
        W=(A[:,None,:,None]-B[None,:,None,:]).square().reshape(n*m,n*m)
        L=constants['hessian_bound']
        assert L >= torch.linalg.matrix_norm(2*W,ord=2).item()
        assert constants['raw_hessian_bound'] == pytest.approx(2*W.sum(1).max().item(),rel=1e-14)
        assert constants['hessian_fp64_absolute_guard'] > 0
        assert constants['maximum_feasible_entry'] == min(1/n,1/m)
        assert constants['mu']==1
        assert constants['kappa']==2*L*L


def test_zero_objective_has_explicit_positive_constant_convention():
    a=torch.full((2,),.5,dtype=torch.float64)
    p=RowShardedGW(torch.zeros((2,2),dtype=torch.float64),torch.zeros((2,2),dtype=torch.float64),a,a)
    c=constants_from_problem(p)
    assert c['hessian_bound']==1
    assert c['hessian_bound_zero_objective_convention']
    assert schedule(0,c)['alpha']>0


def test_exact_index_shift_coupling_and_admissible_tuned_prefactors():
    c=theorem_constants(hessian_bound=7.,maximum_entry=.2)
    for fraction in [1.,.3,.01]:
        for theta in [1.,.7,.03]:
            for k in [0,1,2,99]:
                s=schedule(k,c,coupling_fraction=fraction,momentum_prefactor=theta)
                assert s['beta']==(None if k==0 else theta/math.sqrt(k))
                assert s['beta_next']==theta/math.sqrt(k+1)
                assert s['alpha']==s['coupling_c']*s['beta_next']
                assert s['eta']==s['alpha']/.2
                assert s['coupling_c']<=1/(2*math.sqrt(c['kappa']))
                assert s['alpha']<1/c['relative_smoothness']
    assert schedule(0,c,momentum_prefactor=.7)['beta_next']==schedule(1,c,momentum_prefactor=.7)['beta']


@pytest.mark.parametrize('field',['coupling_fraction','momentum_prefactor'])
@pytest.mark.parametrize('bad',[0.,-1.,1.001,float('nan'),float('inf')])
def test_prefactors_cannot_escape_the_theorem(field,bad):
    with pytest.raises(ValueError):
        schedule(0,theorem_constants(hessian_bound=1.,maximum_entry=.2),**{field:bad})


def test_three_momentum_updates_match_independent_scalar_constrained_optima():
    p=tiny_problem()
    constants=constants_from_problem(p)
    prox=StableCertifiedKLProx(p)
    reference=torch.outer(p.a,p.b)
    logP=reference.log()
    gradients=[torch.tensor(g,dtype=torch.float64) for g in
               [[[3.,-2.],[-4.,1.]],[[-1.,4.],[2.,-3.]],[[2.,-1.],[-2.,3.]]]]
    momentum=None
    expected_momentum=None
    for k,G in enumerate(gradients):
        expected_momentum=G if k==0 else (1-.6/math.sqrt(k))*expected_momentum+.6/math.sqrt(k)*G
        alpha=.8/(2*math.sqrt(2)*constants['hessian_bound'])*.6/math.sqrt(k+1)
        expected=scalar_optimum(reference,expected_momentum,p.a,p.b,alpha/constants['kernel_scale'])
        P,logP,momentum,info=msbpg_step(logP,G,momentum,iteration=k,constants=constants,
             prox=prox,marginal_residuals=prox.marginal_residuals,inner_tol=1e-13,inner_max=10000,
             coupling_fraction=.8,momentum_prefactor=.6,check_every=1)
        torch.testing.assert_close(momentum,expected_momentum,rtol=0,atol=0)
        torch.testing.assert_close(P,expected,rtol=0,atol=1e-13)
        assert info['alpha']==pytest.approx(alpha,rel=1e-15)
        assert info['marginal_l1']<=1e-13
        assert info['normalized_gap_upper_bound']<=1e-7
        reference=expected


def test_selected_statistic_uses_theoretical_alpha_and_stable_log_difference():
    logP=torch.tensor([[-20.,-800.]],dtype=torch.float64)
    logQ=logP+torch.tensor([[1e-12,700.]],dtype=torch.float64)
    alpha=1e-10
    expected=((np.exp(np.longdouble(logQ.numpy()))-np.exp(np.longdouble(logP.numpy())))/np.longdouble(alpha))
    assert normalized_displacement(logP,logQ,alpha,block_rows=1)==pytest.approx(float(np.square(expected).sum()),rel=3e-7)
    assert normalized_displacement(logP,logP,alpha)==0
    assert normalized_displacement(logP[:0],logQ[:0],alpha)==0


def test_output_law_is_exactly_proportional_to_theoretical_steps():
    c=theorem_constants(hessian_bound=4.,maximum_entry=.125)
    alpha=torch.tensor([schedule(k,c,coupling_fraction=.7,momentum_prefactor=.2)['alpha'] for k in range(31)],dtype=torch.float64)
    torch.testing.assert_close(output_probabilities(31),alpha/alpha.sum(),atol=1e-16,rtol=1e-15)
    assert 0<=select_output_index(31,generator=torch.Generator().manual_seed(21))<31


def test_approximately_symmetric_inputs_are_rejected_by_the_theorem_bound():
    a=torch.full((2,),.5,dtype=torch.float64)
    A=torch.tensor([[0.,1.],[1.+1e-8,0.]],dtype=torch.float64)
    B=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
    p=RowShardedGW(A,B,a,a)
    with pytest.raises(ValueError,match='exactly symmetric'):
        constants_from_problem(p)
