"""Check the block Hessians, paired draws and restart on the production inputs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

from run import atomic_json, load_problem, make_solver, now
from setup_problem import initialize
from baseline import local_block_step
from core import FactorizationProblem
import numpy as np


HERE = Path(__file__).resolve().parent


def load_frozen_solver():
    spec = importlib.util.spec_from_file_location("frozen_baseline", HERE.parent / "baseline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SpringSARAH


def check_curvature():
    rng = np.random.default_rng(52)
    m, n, rank = 8, 9, 3
    problem = FactorizationProblem(rng.normal(size=(m, n)), rank, 10, 10, 50)
    U = rng.normal(size=(m, rank))
    V = rng.normal(size=(n, rank))
    dU = rng.normal(size=U.shape)
    dV = rng.normal(size=V.shape)
    epsilon = 1e-5
    for cols in (np.arange(n), np.array([0, 4, 7])):
        b = len(cols)
        grad = (problem.full_grad if b == n else
                lambda u, v: problem.sampled_grad_columns(u, v, cols))
        gu, gv = grad(U, V)
        actual_u = (grad(U + epsilon * dU, V)[0] - gu) / epsilon
        actual_v = (grad(U, V + epsilon * dV)[1] - gv) / epsilon
        expected_u = dU @ (V[cols].T @ V[cols]) / (m * b)
        expected_v = np.zeros_like(dV)
        expected_v[cols] = dV[cols] @ (U.T @ U) / (m * b)
        np.testing.assert_allclose(actual_u, expected_u, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(actual_v, expected_v, atol=1e-10, rtol=1e-8)
        rank_one = np.outer(U[:, 0], U[0])
        np.testing.assert_allclose(local_block_step(rank_one, m, b, rng, 1.0),
                                   0.5 * m * b / np.linalg.norm(rank_one, 2)**2)
    assert local_block_step(np.zeros_like(U), m, n, rng, 1.0) == 1.0


def check_pair(practical_config, strict_config, frozen_class):
    problem = load_problem(practical_config)
    U, V = initialize(problem, practical_config["seed"])
    practical = make_solver(problem, U, V, practical_config)
    strict = make_solver(problem, U, V, strict_config)
    frozen = frozen_class(
        U, V, full_gradient=problem.full_grad, batch_gradient=problem.sampled_grad_columns,
        n_components=problem.n, radius_u=problem.RU, radius_v=problem.RV,
        l1_radius=problem.l1_radius, component_lipschitz=problem.certificates()["image_L_bound"],
        block_lipschitz_u=problem.RV**2 / problem.m, block_lipschitz_v=problem.RU**2 / problem.m,
        batch_size=strict_config["batch"], seed=strict_config["seed"] + 100000,
    )
    draws = [[], [], []]
    for solver, history in zip((practical, strict, frozen), draws):
        def record(rng, b, history=history):
            indices = rng.choice(problem.n, b, replace=False)
            history.append(indices.tolist())
            return indices
        solver.sample_indices = record

    result = {"initial_loss": problem.objective(U, V)}
    for iteration in range(1, 121):
        local_step = practical.step()
        strict_step = strict.step()
        if iteration <= 5:
            assert frozen.step() == strict_step
            np.testing.assert_array_equal(frozen.U, strict.U)
            np.testing.assert_array_equal(frozen.V, strict.V)
            assert draws[2] == draws[1]
        for key in ("refresh_u", "refresh_v", "step_oracle_calls"):
            assert local_step[key] == strict_step[key]
        assert practical.rng.bit_generator.state == strict.rng.bit_generator.state
        if iteration == 1:
            result["first_steps"] = [local_step["step_u"], local_step["step_v"]]
            result["first_loss"] = problem.objective(practical.U, practical.V)
        if iteration == 47:
            restart_state = practical.state_dict()
    assert draws[0] == draws[1]
    assert result["first_loss"] < result["initial_loss"]
    resumed = make_solver(problem, U, V, practical_config)
    resumed.load_state_dict(deepcopy(restart_state))
    while resumed.iteration < practical.iteration:
        resumed.step()
    for key, value in practical.state_dict().items():
        if isinstance(value, (np.ndarray, tuple)):
            np.testing.assert_equal(resumed.state_dict()[key], value)
        else:
            assert resumed.state_dict()[key] == value
    result.update(iterations=practical.iteration, final_raw_loss=problem.objective(practical.U, practical.V),
                  feasibility=problem.feasibility(practical.U, practical.V),
                  component_gradient_calls=practical.oracle_calls,
                  full_block_refreshes=practical.full_gradient_calls)
    return result


if __name__ == "__main__":
    check_curvature()
    frozen = load_frozen_solver()
    manifest = json.loads((HERE / "manifest.json").read_text())
    runs = {(r["dataset"], r["step_rule"]): r["config"] for r in manifest["runs"]}
    results = {dataset: check_pair(runs[dataset, "local_curvature"], runs[dataset, "theorem"], frozen)
               for dataset in ("orl64", "yaleb32")}
    atomic_json(HERE / "validation.json", dict(
        status="passed", checked_utc=now(), results=results,
        checks=["full and minibatch finite-difference block Hessians",
                "rank-one power estimate and degenerate fallback",
                "bitwise isolated/frozen strict updates on both real datasets",
                "paired minibatch arrays, block refreshes and counts at each smoke step",
                "bitwise practical checkpoint restart across both RNG streams"]))
    print(json.dumps(results, indent=2))
