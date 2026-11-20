import json
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from run import run, SFrames
from core import FactorizationProblem
from setup_problem import make_problem_config


def test_sframes_momentum_contains_only_smooth_gradient():
    rng = np.random.default_rng(19)
    A = rng.uniform(size=(8, 5))
    p = FactorizationProblem(A, 2, 2., 2., 1.)
    U = -rng.uniform(0, .1, size=(8, 2))
    V = rng.uniform(0, .1, size=(5, 2))
    c = dict(gamma0=.3, rho0=.1, mu0=.5, batch=5, seed=3)
    solver = SFrames(p, U, V, c)
    exact = p.full_grad(U, V)
    solver.step()
    np.testing.assert_allclose(solver.mU, exact[0], atol=1e-15)
    np.testing.assert_allclose(solver.mV, exact[1], atol=1e-15)
    beforeU, beforeV = solver.mU.copy(), solver.mV.copy()
    second = p.full_grad(solver.U, solver.V)
    solver.step()
    rho = .1 / np.sqrt(2)
    np.testing.assert_allclose(solver.mU, (1-rho)*beforeU + rho*second[0], atol=1e-15)
    np.testing.assert_allclose(solver.mV, (1-rho)*beforeV + rho*second[1], atol=1e-15)


def test_runner_resumes_both_methods_and_accounts_oracles(tmp_path):
    rng = np.random.default_rng(10)
    A = rng.uniform(size=(8, 5))
    A /= np.linalg.norm(A, axis=0)
    data = tmp_path / "data.npz"
    np.savez(data, A=A)
    base = dict(make_problem_config(A, 2), data_path=str(data), batch=2, seed=12,
                iterations=8, probe_every=2, checkpoint_every=2,
                gamma0=.3, rho0=.1, mu0=2.)
    for method in ("sframes", "spring_sarah"):
        config = dict(base, method=method)
        full, resumed = tmp_path / (method+"_full"), tmp_path / (method+"_resumed")
        run(config, full)
        partial = run(config, resumed, stop_at=3)
        assert partial["status"] == "checkpointed"
        result = run(config, resumed)
        with (full / "checkpoint.pkl").open("rb") as stream:
            a = pickle.load(stream)
        with (resumed / "checkpoint.pkl").open("rb") as stream:
            b = pickle.load(stream)
        np.testing.assert_array_equal(a["solver"]["U"], b["solver"]["U"])
        np.testing.assert_array_equal(a["solver"]["V"], b["solver"]["V"])
        assert a["counters"]["component_gradient_calls"] == b["counters"]["component_gradient_calls"] > 0
        assert result["iteration"] == 8
        if method == "sframes":
            assert result["component_gradient_calls"] == 16
        else:
            assert result["cumulative_prox_iterations"] >= 16
