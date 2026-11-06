"""Independent high-precision checks of the unchanged proximal certificate."""
from decimal import Decimal, localcontext
import math
from pathlib import Path
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent/'shareseq_25000_20260930'))
from sharded_gw import RowShardedGW
from stable_prox import StableCertifiedKLProx
from certified_prox import CertifiedKLProx


def problem(dtype=torch.float64):
    distances = torch.tensor([[0.,1.],[1.,0.]], dtype=dtype)
    marginal = torch.tensor([.5,.5], dtype=dtype)
    return RowShardedGW(distances, distances, marginal, marginal)


def dec(x):
    return Decimal.from_float(float(x))


def high_precision_objectives(Q, logP, G, eta, u, v):
    """120-digit evaluation of the defining formulas, without stable rewrites."""
    with localcontext() as context:
        context.prec = 120
        step = dec(eta)
        primal = Decimal(0)
        dual_numerator = sum(dec(x)*Decimal('.5') for x in u)
        dual_numerator += sum(dec(x)*Decimal('.5') for x in v)
        for i in range(2):
            for j in range(2):
                q, lp, g = dec(Q[i,j]), dec(logP[i,j]), dec(G[i,j])
                reference = lp.exp()
                dual_plan = (lp-step*g+dec(u[i])+dec(v[j])).exp()
                kl = (q*(q.ln()-lp) if q else Decimal(0)) - q + reference
                primal += g*q + kl/step
                dual_numerator += reference-dual_plan
        dual = dual_numerator/step
        return float(primal), float(dual), float(primal-dual)


def analytic_symmetric_solution(logP, eta):
    """Solve the symmetric 2x2 constrained problem in 120-digit arithmetic."""
    with localcontext() as context:
        context.prec = 120
        step = dec(eta)
        diag = (dec(logP[0,0])-step).exp()
        off = (dec(logP[0,1])+step).exp()
        u = (Decimal('.5')/(diag+off)).ln()
        diagonal = float(diag*u.exp())
        # Exact feasible floating-point symmetric pair for these test values.
        Q = torch.tensor([[diagonal,.5-diagonal],[.5-diagonal,diagonal]], dtype=torch.float64)
        return Q, torch.full((2,),float(u),dtype=torch.float64), torch.zeros(2,dtype=torch.float64)


@pytest.mark.parametrize('eta',[3.,.1,1e-6,1e-10])
@pytest.mark.parametrize('small_entry',[.13,1e-9])
def test_stable_objectives_match_independent_decimal_definitions(eta,small_entry):
    p = problem()
    solver = StableCertifiedKLProx(p)
    reference = torch.tensor([[small_entry,.5-small_entry],[.5-small_entry,small_entry]],dtype=torch.float64)
    logP = reference.log()
    G = torch.tensor([[1.,-1.],[-1.,1.]],dtype=torch.float64)
    Q,u,v = analytic_symmetric_solution(logP,eta)
    assert torch.equal(Q.sum(0),p.b) and torch.equal(Q.sum(1),p.a)
    info = solver.certificate(Q,logP,G,eta,row_potential=u,col_potential=v)
    exact_primal,exact_dual,exact_gap = high_precision_objectives(Q,logP,G,eta,u,v)
    allowance = info['gap_roundoff_tolerance']
    assert abs(info['primal_objective']-exact_primal) <= allowance
    assert abs(info['dual_objective']-exact_dual) <= allowance
    assert abs(info['primal_dual_gap']-exact_gap) <= allowance
    assert abs(info['stable_primal_dual_gap']-exact_gap) <= allowance
    assert info['certificate_valid'] and info['proximal_optimality_certified']
    assert allowance/info['gap_normalizer'] < 1e-7
    assert info['certificate_primal']=='FP64 rounded feasible witness of returned plan'


def test_tiny_step_removes_old_cancellation_floor_without_changing_tolerance():
    p = problem()
    reference = torch.full((2,2),.25,dtype=torch.float64)
    G = torch.tensor([[1.,-1.],[-1.,1.]],dtype=torch.float64)
    eta=1e-10
    Q,u,v=analytic_symmetric_solution(reference.log(),eta)
    old=CertifiedKLProx(p).certificate(Q,reference.log(),G,eta,row_potential=u,col_potential=v)
    new=StableCertifiedKLProx(p).certificate(Q,reference.log(),G,eta,row_potential=u,col_potential=v)
    assert not old['proximal_optimality_certified']
    assert old['gap_roundoff_tolerance']/old['gap_normalizer']>1e-7
    assert new['proximal_optimality_certified']
    assert new['gap_roundoff_tolerance']/new['gap_normalizer']<1e-7


@pytest.mark.parametrize('eta',[1.,1e-10])
def test_bad_dual_potentials_are_rejected_against_decimal_gap(eta):
    p=problem()
    reference=torch.tensor([[1e-9,.5-1e-9],[.5-1e-9,1e-9]],dtype=torch.float64)
    G=torch.tensor([[1.,-1.],[-1.,1.]],dtype=torch.float64)
    Q,u,v=analytic_symmetric_solution(reference.log(),eta)
    u=u+.1
    info=StableCertifiedKLProx(p).certificate(Q,reference.log(),G,eta,row_potential=u,col_potential=v)
    primal,dual,gap=high_precision_objectives(Q,reference.log(),G,eta,u,v)
    assert gap/max(1.,abs(primal),abs(dual))>1e-7
    assert not info['proximal_optimality_certified']
    assert abs(info['stable_primal_dual_gap']-gap)<=info['gap_roundoff_tolerance']


def test_fp32_witness_error_is_still_rejected_when_real_gap_exceeds_tolerance():
    p=problem(torch.float32)
    reference=torch.tensor([[.13123456789,.36876543211],[.36876543211,.13123456789]],dtype=torch.float64)
    G=torch.zeros((2,2),dtype=torch.float32)
    Q=reference.float()
    zero=torch.zeros(2,dtype=torch.float64)
    info=StableCertifiedKLProx(p).certificate(Q,reference.log(),G,1e-12,row_potential=zero,col_potential=zero)
    assert info['stable_primal_dual_gap']/info['gap_normalizer']>1e-7
    assert not info['proximal_optimality_certified']
    assert info['plan_dtype']=='torch.float32'


def test_inherited_sinkhorn_with_small_step_uses_same_stopping_contract():
    p=problem(torch.float32)
    solver=StableCertifiedKLProx(p)
    reference=torch.full((2,2),.25,dtype=torch.float64)
    G=torch.tensor([[1.,-1.],[-1.,1.]],dtype=torch.float32)
    Q,logQ,info=solver(reference.log(),G,1e-10,tol=1e-7,max_iter=100,check_every=1)
    assert info['converged'] and info['proximal_optimality_certified']
    assert info['marginal_l1']<=1e-7
    assert info['normalized_gap_tolerance']==1e-7
    assert (max(info['primal_dual_gap'],info['stable_primal_dual_gap'])+info['gap_roundoff_tolerance'])/info['gap_normalizer']<=1e-7
    assert Q.dtype==torch.float32 and logQ.dtype==torch.float64


@pytest.mark.parametrize('diagonal',[0.,.17])
@pytest.mark.parametrize('eta',[.3,1e-10])
def test_general_and_boundary_witness_formulas_match_decimal(diagonal,eta):
    p=problem()
    reference=torch.tensor([[.11,.23],[.31,.35]],dtype=torch.float64)
    Q=torch.tensor([[diagonal,.5-diagonal],[.5-diagonal,diagonal]],dtype=torch.float64)
    G=torch.tensor([[.3,-.5],[1.2,2.7]],dtype=torch.float64)
    u=torch.tensor([.2,-.1],dtype=torch.float64)
    v=torch.tensor([.03,-.07],dtype=torch.float64)
    info=StableCertifiedKLProx(p).certificate(Q,reference.log(),G,eta,row_potential=u,col_potential=v)
    primal,dual,gap=high_precision_objectives(Q,reference.log(),G,eta,u,v)
    allowance=info['gap_roundoff_tolerance']
    assert abs(info['primal_objective']-primal)<=allowance
    assert abs(info['dual_objective']-dual)<=allowance
    assert abs(info['stable_primal_dual_gap']-gap)<=allowance
    assert not info['proximal_optimality_certified']
