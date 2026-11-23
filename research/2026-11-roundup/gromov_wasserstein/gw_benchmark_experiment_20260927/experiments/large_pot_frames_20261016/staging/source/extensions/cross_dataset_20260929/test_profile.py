import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).parent/'runtime'))
from profile_components import replay
from gw_core import GWProblem

@pytest.mark.parametrize('method,params',[('smoothing-fw',dict(gamma0=.3,mu_scale=10)),('stochastic-kl-momentum',dict(eta=.1,inner_tol=1e-4))])
def test_profile_hooks_preserve_update_sequence(method,params):
    X=torch.arange(5,dtype=torch.float64)[:,None]
    A=torch.cdist(X,X);B=A.roll(1,0).roll(1,1);a=torch.full((5,),.2,dtype=torch.float64)
    prob=GWProblem(A,B,a,a)
    before,_=replay(prob,method,params,seed=17,warmup=2,measured=3,instrument=False)
    after,stats=replay(prob,method,params,seed=17,warmup=2,measured=3,instrument=True)
    assert torch.equal(before,after)
    assert stats['components']['gradient_kernel']['calls']==3
    assert stats['components']['probe_generation']['calls']==3
    assert stats['other_update_seconds']>=0
