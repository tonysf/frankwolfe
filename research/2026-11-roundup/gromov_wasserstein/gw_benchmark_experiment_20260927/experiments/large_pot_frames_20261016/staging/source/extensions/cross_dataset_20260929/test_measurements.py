import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
spec=importlib.util.spec_from_file_location('cross_measurements', HERE/'runtime/measurements.py')
measure=importlib.util.module_from_spec(spec);spec.loader.exec_module(measure)


def test_permutation_and_uniform_ties_do_not_favor_vertex_order():
    n=5; forward=torch.tensor([3,0,4,2,1]); inverse=torch.argsort(forward)
    exact=torch.zeros(n,n);exact[torch.arange(n),forward]=1/n
    uniform=torch.ones(n,n)/(n*n)
    for chunk in (1,3,100):
        found=measure.correspondence_metrics(exact,(forward,inverse),chunk)
        assert found['correspondence_top1_accuracy']==1
        assert found['correspondence_rank_fraction']==0
        assert found['correspondence_true_pair_mass']==pytest.approx(1)
        null=measure.correspondence_metrics(uniform,(forward,inverse),chunk)
        assert null['correspondence_top1_accuracy']==pytest.approx(1/n)
        assert null['correspondence_rank_fraction']==.5


def test_wrong_matches_and_no_mutation():
    P=torch.eye(4)/4; before=P.clone();state=torch.random.get_rng_state().clone()
    forward=torch.tensor([1,2,3,0]); inverse=torch.argsort(forward)
    found=measure.correspondence_metrics(P,(forward,inverse))
    assert found['correspondence_top1_accuracy']==0
    assert found['correspondence_true_pair_mass']==0
    assert found['correspondence_rank_fraction']==pytest.approx(2/3)
    assert torch.equal(P,before) and torch.equal(torch.random.get_rng_state(),state)


def test_binding_and_inverse_validation(tmp_path):
    p=tmp_path/'pairs.npz';np.savez(p,forward=np.array([1,0]),inverse=np.array([1,0]))
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert measure.load_correspondence(p,sha,2,2,'cpu')[0].tolist()==[1,0]
    with pytest.raises(ValueError,match='SHA256'):measure.load_correspondence(p,'bad',2,2,'cpu')
    np.savez(p,forward=np.array([1,0]),inverse=np.array([0,1]));sha=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='inverse'):measure.load_correspondence(p,sha,2,2,'cpu')
    assert measure.load_correspondence(None,None,2,2,'cpu') is None
